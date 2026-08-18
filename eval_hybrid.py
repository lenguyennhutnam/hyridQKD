#!/usr/bin/env python3
"""Compare mode-selection policies on identical jamming trajectories.

Usage (inside WSL):
    python3 eval_hybrid.py [--seeds 10] [--horizon 3600] [--episodes 200]
    python3 eval_hybrid.py --lam 2.5 --b-max 1e6 --show-sweep
    python3 eval_hybrid.py --no-dqn            # baselines only, seconds

This module produces the study's answer to "does switching pay", so it is
written to be hard to flatter. Four properties carry that weight:

1. **Every policy sees the SAME seeds and the same envs.** A difference in
   eta is then a difference in the policy, not in the weather.
2. **Every baseline gets its best operating point.** GreedyThreshold is swept
   over `thresholds`; BufferAwareThreshold is swept over the FULL
   threshold x reserve grid, not a cheaper coordinate descent that can miss
   the grid optimum. A baseline denied its best point manufactures a win for
   whatever it is compared against.
3. **The DQN is scored on seeds it never trained on.** `dqn.train` consumes
   exactly `training_seed_range(config)`; compare() refuses to run if any
   evaluation seed falls inside it, for any of the configs it was given.
4. **No row is ever overwritten in silence.** `DQNPolicy.name` is a class
   constant and GreedyThreshold renders its threshold at three decimals, so
   two comparands really can collide on a label. `label_rows` raises instead.

The default spec parameters are DEGENERATE -- AlwaysPM serves every message
(eta = 1.0000) because B_max holds ~1200 slots of demand while the longest
jamming episode is ~40 slots -- so `compare()` takes the HybridParams and
EveParams to run at, and the CLI exposes the two knobs (--lam, --b-max) that
remove the degeneracy.

Spec: docs/superpowers/specs/2026-08-17-hybrid-policy-design.md
"""

import argparse
import dataclasses
import statistics

from dqn import DQNConfig, train, training_seed_range
from evemarkov import EveParams
from hybridenv import HybridEnv, HybridParams
from policies import (AlwaysENT, AlwaysPM, BufferAwareThreshold,
                      GreedyThreshold, run_episode)

__all__ = ["THRESHOLDS", "RESERVES", "EVAL_SEED_BASE", "evaluation_seeds",
           "assert_seeds_disjoint", "label_rows", "sweep_threshold",
           "sweep_buffer_aware", "compare", "format_rows", "main"]

# Candidate switch points. PM's death QBER at the design channel sits near
# 4.8% and ENT's near 11.6%, so the interesting range is the approach to PM's
# death: switching after it is already too late, switching far before it
# throws away PM's much higher rate.
THRESHOLDS = (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05)
# reserve = 0 makes BufferAwareThreshold exactly GreedyThreshold, so the grid
# always contains the plain policy: the buffer-aware row can never score
# below the greedy row because of the sweep.
RESERVES = (0.0, 0.1, 0.25, 0.5)

# Evaluation seeds live in a block no plausible training run reaches:
# dqn.SEED_STRIDE is 100_000, so this is config seed 990's block, and
# `episodes` is capped at the stride. See dqn's module docstring.
EVAL_SEED_BASE = 99_000_000


def evaluation_seeds(n, dqn_config=None, base=EVAL_SEED_BASE):
    """`n` seeds that no training run of `dqn_config` can have consumed.

    The naive choice -- 1..n -- lands squarely inside the default
    DQNConfig()'s training range of 0..199, which would score the agent on
    trajectories it memorised.

    `base` is only a starting point. Nothing stops a caller from training
    with config.seed = 990, whose block IS the reserved one, so when the
    default block overlaps we move past the end of the training range
    instead. That keeps the promise in the first line true for every config
    rather than for the ones we happened to think of.
    """
    base = int(base)
    if dqn_config is not None:
        # training_seed_range is contiguous, so one step past its end clears
        # it whether `base` sat inside the range or just short of it.
        base = max(base, training_seed_range(dqn_config).stop)
    seeds = tuple(base + i for i in range(int(n)))
    if dqn_config is not None:
        assert_seeds_disjoint(seeds, dqn_config)
    return seeds


def assert_seeds_disjoint(seeds, dqn_config, label="dqn"):
    """Refuse evaluation seeds that `dqn_config` would train on.

    This is a hard error rather than a warning: an overlap does not break
    anything visibly, it just inflates the learned policy's score, which is
    the exact failure this whole module exists to avoid.
    """
    trained = training_seed_range(dqn_config)
    clash = sorted({int(s) for s in seeds if int(s) in trained})
    if clash:
        raise ValueError(
            "evaluation seeds %s are inside the training range of %r "
            "(%d..%d): the agent would be scored on trajectories it trained "
            "on. Use evaluation_seeds()." % (clash, label, trained.start,
                                             trained.stop - 1))


def label_rows(rows, what="policy label"):
    """Index rows by their "policy" field; refuse a duplicate.

    Building the table as a plain dict comprehension would let a second row
    with the same label replace the first with no symptom whatsoever.
    """
    out = {}
    for row in rows:
        key = row["policy"]
        if key in out:
            raise ValueError("duplicate %s %r: a row would be lost"
                             % (what, key))
        out[key] = row
    return out


def _envs_for(env_factory, seeds):
    """One HybridEnv per seed, built once and shared by every policy.

    Construction re-runs channel.derive()/derive_ent(), which dominates the
    cost of everything here, while run_episode(env, policy, seed) reseeds
    every stream -- so one env per seed serves any number of policies, and
    serves all of them the identical jamming trajectory.
    """
    return {s: env_factory(s) for s in seeds}


def _score(envs, policy, seeds):
    """Run `policy` once per seed. The seed is ALWAYS explicit: run_episode
    with seed=None replays a single trajectory, which would collapse an
    n-seed average into one episode repeated n times."""
    return [run_episode(envs[s], policy, seed=s) for s in seeds]


def _keyed(pairs):
    """A dict from (key, policy) pairs that refuses a duplicate key."""
    out = {}
    for key, policy in pairs:
        if key in out:
            raise ValueError("duplicate sweep point %r" % (key,))
        out[key] = policy
    return out


def _sweep(envs, seeds, candidates):
    """(best key, {key: mean eta}) over {key: Policy}.

    Keys are the numeric sweep coordinates, never the policy names: those
    render at fixed precision, so 0.0405 and 0.0410 would land on the same
    string and one point of the sweep would vanish.
    """
    table = {key: statistics.mean(r["eta"] for r in _score(envs, pol, seeds))
             for key, pol in candidates.items()}
    return max(table, key=table.get), table


def sweep_threshold(env_factory, thresholds=THRESHOLDS, seeds=(1, 2, 3)):
    """Return (best threshold, {threshold: mean eta}) for GreedyThreshold.

    `env_factory(seed) -> HybridEnv` is called once per seed; every threshold
    is then scored on that same set of envs and seeds.
    """
    seeds = tuple(seeds)
    envs = _envs_for(env_factory, seeds)
    return _sweep(envs, seeds,
                  _keyed((float(t), GreedyThreshold(t)) for t in thresholds))


def sweep_buffer_aware(env_factory, thresholds=THRESHOLDS, reserves=RESERVES,
                       seeds=(1, 2, 3)):
    """Return (best (threshold, reserve), {pair: mean eta}).

    The FULL grid, not "best threshold from the greedy sweep, then best
    reserve": the greedy optimum need not be the first coordinate of the
    buffer-aware optimum, and a coordinate descent that misses it would hand
    the learned policy a weakened baseline to beat.
    """
    seeds = tuple(seeds)
    envs = _envs_for(env_factory, seeds)
    return _sweep(envs, seeds,
                  _keyed(((float(t), float(r)), BufferAwareThreshold(t, r))
                         for t in thresholds for r in reserves))


def _mean_row(label, results):
    """One summary row. Each field averages ITS OWN key across episodes."""
    def mean(key):
        return statistics.mean(r[key] for r in results)
    return {"policy": label, "episodes": len(results), "eta": mean("eta"),
            "switches": mean("switches"), "expired": mean("expired"),
            "secret_bits": mean("secret_bits"),
            "ent_fraction": mean("ent_fraction"),
            "starved": mean("starved_fraction")}


def _dqn_configs(dqn_config):
    """Normalise the dqn_config argument to {label: DQNConfig}.

    None means "one agent at the defaults"; an empty mapping means "no agent
    at all", which is what makes a baselines-only run cheap.
    """
    if dqn_config is None:
        return {"dqn": DQNConfig()}
    if isinstance(dqn_config, dict):
        return dict(dqn_config)
    return {"dqn": dqn_config}


def compare(seeds, params=None, eve_params=None, dqn_config=None,
            thresholds=THRESHOLDS, reserves=RESERVES, sweeps=None):
    """Run every policy over `seeds`; return one summary row each.

    `seeds` must lie outside the training range of every config in
    `dqn_config` -- see assert_seeds_disjoint. Pass `sweeps={}` to receive
    the two sweep tables back for reporting.
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("compare() needs at least one seed")
    configs = _dqn_configs(dqn_config)
    # Checked BEFORE any training happens, so a bad seed set costs nothing.
    for label, cfg in configs.items():
        assert_seeds_disjoint(seeds, cfg, label)

    params = params or HybridParams()
    eve = eve_params or EveParams()

    def factory(seed):
        return HybridEnv(dataclasses.replace(params, seed=seed),
                         dataclasses.replace(eve, seed=seed))

    envs = _envs_for(factory, seeds)

    best_th, th_table = _sweep(
        envs, seeds, _keyed((float(t), GreedyThreshold(t))
                            for t in thresholds))
    best_pair, ba_table = _sweep(
        envs, seeds, _keyed(((float(t), float(r)), BufferAwareThreshold(t, r))
                            for t in thresholds for r in reserves))
    if sweeps is not None:
        sweeps["threshold"] = th_table
        sweeps["buffer_aware"] = ba_table

    candidates = [AlwaysPM(), AlwaysENT(), GreedyThreshold(best_th),
                  BufferAwareThreshold(*best_pair)]
    rows = [_mean_row(p.name, _score(envs, p, seeds)) for p in candidates]

    for label, cfg in configs.items():
        # Training gets its own env, constructed from a seed in the config's
        # OWN training block. train() resets it per episode so the
        # construction seed never reaches a trajectory, but taking it from
        # the training block rather than from `seeds` keeps the separation
        # visible: nothing on the training side is built from an evaluation
        # seed, and the evaluation envs are never advanced by training.
        trained = train(factory(training_seed_range(cfg).start), cfg)
        rows.append(_mean_row(label, _score(envs, trained, seeds)))

    label_rows(rows)        # refuse a silent collision; the dict is a guard
    return rows


def format_rows(rows):
    """The comparison table as text."""
    head = ("%-26s %8s %9s %9s %12s %9s %9s"
            % ("policy", "eta", "switches", "expired", "secret bits",
               "ENT frac", "starved"))
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append("%-26s %8.4f %9.1f %9.1f %12.3e %9.3f %9.3f"
                     % (r["policy"], r["eta"], r["switches"], r["expired"],
                        r["secret_bits"], r["ent_fraction"], r["starved"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of evaluation episodes per policy")
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--episodes", type=int, default=200,
                    help="DQN training episodes")
    ap.add_argument("--lam", type=float, default=None,
                    help="mean arrivals per slot; the spec default of 1.0 is "
                         "degenerate (AlwaysPM scores eta = 1)")
    ap.add_argument("--b-max", type=float, default=None,
                    help="key buffer capacity in bits, e.g. 1e6")
    ap.add_argument("--t0", type=int, default=None, help="deadline in slots")
    ap.add_argument("--no-dqn", action="store_true",
                    help="baselines only; skips training entirely")
    ap.add_argument("--show-sweep", action="store_true",
                    help="print the baseline sweep tables too")
    args = ap.parse_args(argv)

    overrides = {"horizon": args.horizon}
    if args.lam is not None:
        overrides["lam"] = args.lam
    if args.b_max is not None:
        overrides["b_max"] = int(args.b_max)
    if args.t0 is not None:
        overrides["t0_deadline"] = args.t0
    params = HybridParams(**overrides)

    cfg = DQNConfig(episodes=args.episodes)
    dqn_config = {} if args.no_dqn else cfg
    seeds = evaluation_seeds(args.seeds, None if args.no_dqn else cfg)

    sweeps = {}
    rows = compare(seeds=seeds, params=params, dqn_config=dqn_config,
                   sweeps=sweeps)

    print("params: lam=%.3f b_max=%.3g horizon=%d T0=%d n_x=%d d_km=%.1f"
          % (params.lam, params.b_max, params.horizon, params.t0_deadline,
             params.n_x, params.d_km))
    print("evaluation seeds: %d..%d (training range %d..%d)"
          % (seeds[0], seeds[-1], training_seed_range(cfg).start,
             training_seed_range(cfg).stop - 1))
    if args.show_sweep:
        print("\ngreedy threshold sweep:")
        for th in sorted(sweeps["threshold"]):
            print("  %.4f  eta=%.4f" % (th, sweeps["threshold"][th]))
        print("buffer-aware grid (threshold, reserve):")
        for key in sorted(sweeps["buffer_aware"]):
            print("  %.4f %.2f  eta=%.4f"
                  % (key[0], key[1], sweeps["buffer_aware"][key]))
    print()
    print(format_rows(rows))


if __name__ == "__main__":
    main()
