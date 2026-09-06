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
5. **The table carries its own upper bound.** `oracle` and
   `always-pm-relaxed` are not policies: they run in the relaxed environment
   of `HybridEnv.step_relaxed` (clairvoyant, free switching, continuous key
   delivery). Their difference is the ENTIRE value switching could ever have
   had -- see `switching_value()`. Without it "no policy we tried beat
   AlwaysPM" is a statement about the policies we happened to try.

The default spec parameters are DEGENERATE -- AlwaysPM serves essentially
every message (eta = 0.9957 over the eight evaluation seeds: exactly 1.0000
on seven of them) because B_max holds ~1200 slots of demand and PM refills it
between jamming episodes. It is NOT the length of a single episode that breaks
this: HEAVY runs measured over those seeds average ~40 slots and reach 101-314,
and the one seed that drops below 1.0 has the SHORTEST maximum run of the
eight. It takes a dense enough cluster of them to outrun the buffer, which is
rare at these transition probabilities. So `compare()` takes the HybridParams
and EveParams to run at, and the CLI exposes the two knobs (--lam, --b-max)
that remove the degeneracy.

Spec: docs/superpowers/specs/2026-08-17-hybrid-policy-design.md
"""

import argparse
import dataclasses
import statistics

from dqn import DQNConfig, train, training_seed_range
from evemarkov import EveParams
from hybridenv import HybridEnv, HybridParams
from mobility import MobilityParams
from policies import (AlwaysENT, AlwaysPM, BlockAwareGreedy,
                      BufferAwareThreshold, GreedyThreshold,
                      HysteresisThreshold, Oracle, RangeOrQber,
                      RangeThreshold, RelaxedAlwaysPM, run_episode,
                      run_relaxed_episode)

__all__ = ["THRESHOLDS", "RESERVES", "RANGE_POINTS", "BLOCK_HORIZONS",
           "HYSTERESIS_SHAPES", "EVAL_SEED_BASE",
           "BOUNDS", "evaluation_seeds", "assert_seeds_disjoint", "label_rows",
           "range_switches", "sweep_threshold", "sweep_buffer_aware",
           "sweep_range", "sweep_range_qber", "sweep_block_aware",
           "sweep_hysteresis", "compare", "switching_value",
           "format_rows", "main"]

# Candidate switch points, in measured QBER. At the configured operating point
# (n_X = 1e5, 55 km) PM dies at QBER 3.3535% and ENT at 8.6194%
# (channel.death_qber / death_qber_ent; spec section 4 quotes the same 3.35% /
# 8.62%), so the interesting range is the approach to PM's death: switching
# after it is already too late, switching far before it throws away PM's much
# higher rate.
#
# The grid must also REACH PAST the point where switching stops, or its argmax
# lands on its own edge and "best threshold" only means "largest tried". Under
# the default Eve (f_max = 0.08) the highest QBER PM can display is
# E_PM(0.08) = 4.94%, so a policy that starts on PM never fires above that and
# is bit-for-bit AlwaysPM; the grid runs on to 0.08, past ENT's own ceiling
# E_ENT(0.08) = 6.93%, so that plateau is a property of the jamming rather than
# of the starting mode. The honest consequence is that the swept greedy
# optimum COINCIDES with AlwaysPM -- see the report; it is the finding, not a
# defect in the sweep.
THRESHOLDS = (0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055,
              0.06, 0.07, 0.08)
# reserve = 0 makes BufferAwareThreshold exactly GreedyThreshold, so the grid
# always contains the plain policy: the buffer-aware row can never score
# below the greedy row because of the sweep.
RESERVES = (0.0, 0.1, 0.25, 0.5)

# How many switch DISTANCES the range policies are swept over. The grid itself
# cannot be a module constant the way THRESHOLDS is: a switch distance only
# means something relative to where the vessel actually goes, and that is a
# parameter. range_switches() builds the grid from the patrol band.
RANGE_POINTS = 9

# Planning horizons H for BlockAwareGreedy, in slots. H decides how far ahead
# the rule counts finished blocks, and the grid has to straddle the block fill
# time of the SLOWER mode or the answer is fixed by arithmetic rather than by
# the data: at chapman_lowmu ENT needs 28.7 slots to fill n_X = 1e5, so
# H = 5 can never choose it and H = 320 always can. The default T0 = 20 sits
# between them deliberately -- that is the whole question the sweep asks.
BLOCK_HORIZONS = (5, 20, 80, 320)

# (band, min_dwell) shapes for HysteresisThreshold, swept against every
# threshold: the band is subtracted from the threshold to get the lower edge.
# (0.0, 0) makes it exactly GreedyThreshold, so -- as with RESERVES -- the
# grid always contains the policy it generalises and the hysteresis row can
# never score below the greedy row because of the search.
HYSTERESIS_SHAPES = ((0.0, 0), (0.010, 0), (0.020, 0), (0.010, 20))

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


def range_switches(params, count=RANGE_POINTS):
    """Candidate switch distances [km] for RangeThreshold / RangeOrQber.

    The grid spans the patrol band and OVERHANGS IT AT BOTH ENDS, for exactly
    the reason THRESHOLDS runs past the highest QBER Eve can produce: an
    argmax that lands on the edge of its own grid means "the largest value we
    tried", not "the best value there is". A point below d_min makes the
    policy always-ENT and a point above d_max makes it always-PM, so the
    sweep's optimum is interior by construction.

    With mobility off the distance is a constant, so the band is a point and
    the grid degenerates to a few values either side of it -- the range
    policies then reduce to AlwaysPM or AlwaysENT, which is the honest
    answer: with nothing moving, a range threshold has nothing to switch on.
    """
    count = max(int(count), 2)
    mobility = getattr(params, "mobility", None)
    if mobility is not None and mobility.enabled:
        lo, hi = float(mobility.d_min), float(mobility.d_max)
    else:
        lo = hi = float(params.d_km)
    # A one-point band still needs a non-zero span, or every grid point
    # coincides and _keyed() rejects the duplicates.
    margin = 0.05 * max(hi - lo, 1.0)
    lo, hi = lo - margin, hi + margin
    return tuple(lo + (hi - lo) * i / (count - 1) for i in range(count))


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


def _resolve_envs(env_factory, envs, seeds):
    """The envs a sweep runs on: built here, or handed in by a caller.

    compare() already holds one env per seed and shares it across both sweeps
    and every reported row. Letting a sweep rebuild them would pay the
    channel.derive()/derive_ent() cost a second and third time -- and, worse,
    would mean the sweep that picks the operating point and the row that
    reports it ran on different objects. Exactly one of the two arguments
    must be given, so neither can silently win over the other.
    """
    if (env_factory is None) == (envs is None):
        raise ValueError("pass exactly one of env_factory= or envs=")
    if envs is None:
        return _envs_for(env_factory, seeds)
    missing = [s for s in seeds if s not in envs]
    if missing:
        raise ValueError("no env supplied for seeds %s" % (missing,))
    return envs


# The two reference bounds, in the order they are reported: the control
# first, then the Oracle, because the pair is only meaningful read as a
# difference. RelaxedAlwaysPM is NOT optional -- dropping it leaves
# Oracle - AlwaysPM, which measures the relaxation and the switching
# together and reads as several times more headroom than switching has.
BOUNDS = (RelaxedAlwaysPM, Oracle)


def _score(envs, policy, seeds):
    """Run `policy` once per seed. The seed is ALWAYS explicit: run_episode
    with seed=None replays a single trajectory, which would collapse an
    n-seed average into one episode repeated n times."""
    return [run_episode(envs[s], policy, seed=s) for s in seeds]


def _score_bound(envs, bound, seeds):
    """As _score, but through the relaxed dynamics. Same envs, same seeds:
    the bound faces the identical jamming and arrivals as every policy row,
    which is what makes the subtraction below mean anything."""
    return [run_relaxed_episode(envs[s], bound, seed=s) for s in seeds]


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


def sweep_threshold(env_factory=None, thresholds=THRESHOLDS, seeds=(1, 2, 3),
                    envs=None):
    """Return (best threshold, {threshold: mean eta}) for GreedyThreshold.

    `env_factory(seed) -> HybridEnv` is called once per seed; every threshold
    is then scored on that same set of envs and seeds. `envs={seed: env}` is
    the alternative for a caller that already built them -- compare() uses it
    so the study runs THIS function rather than a copy of it.
    """
    seeds = tuple(seeds)
    envs = _resolve_envs(env_factory, envs, seeds)
    return _sweep(envs, seeds,
                  _keyed((float(t), GreedyThreshold(t)) for t in thresholds))


def sweep_buffer_aware(env_factory=None, thresholds=THRESHOLDS,
                       reserves=RESERVES, seeds=(1, 2, 3), envs=None):
    """Return (best (threshold, reserve), {pair: mean eta}).

    The FULL grid, not "best threshold from the greedy sweep, then best
    reserve": the greedy optimum need not be the first coordinate of the
    buffer-aware optimum, and a coordinate descent that misses it would hand
    the learned policy a weakened baseline to beat.

    `envs` as in sweep_threshold.
    """
    seeds = tuple(seeds)
    envs = _resolve_envs(env_factory, envs, seeds)
    return _sweep(envs, seeds,
                  _keyed(((float(t), float(r)), BufferAwareThreshold(t, r))
                         for t in thresholds for r in reserves))


def sweep_range(env_factory=None, switches=(), seeds=(1, 2, 3), envs=None):
    """Return (best switch distance, {distance: mean eta}) for RangeThreshold.

    `switches` comes from range_switches(params); it is deliberately not
    defaulted here, because a distance grid that ignores the patrol band is
    meaningless.
    """
    seeds = tuple(seeds)
    envs = _resolve_envs(env_factory, envs, seeds)
    return _sweep(envs, seeds,
                  _keyed((float(d), RangeThreshold(d)) for d in switches))


def sweep_range_qber(env_factory=None, switches=(), thresholds=THRESHOLDS,
                     seeds=(1, 2, 3), envs=None):
    """Return (best (distance, threshold), {pair: mean eta}) for RangeOrQber.

    The FULL product grid, like sweep_buffer_aware and for the same reason: a
    coordinate descent seeded from the two single-signal optima can miss the
    joint optimum, and a baseline denied its best operating point manufactures
    a win for whatever it is compared against.
    """
    seeds = tuple(seeds)
    envs = _resolve_envs(env_factory, envs, seeds)
    return _sweep(envs, seeds,
                  _keyed(((float(d), float(t)), RangeOrQber(d, t))
                         for d in switches for t in thresholds))


def sweep_block_aware(env_factory=None, horizons=BLOCK_HORIZONS,
                      seeds=(1, 2, 3), envs=None):
    """Return (best horizon, {horizon: mean eta}) for BlockAwareGreedy.

    A FRESH policy per grid point, like every other sweep here: this one
    carries a bound model and a settling clock, so a shared object would
    score the later horizons with the earlier one's state.
    """
    envs = _resolve_envs(env_factory, envs, seeds)
    return _sweep(envs, seeds,
                  _keyed((int(h), BlockAwareGreedy(horizon=h))
                         for h in horizons))


def sweep_hysteresis(env_factory=None, thresholds=THRESHOLDS,
                     shapes=HYSTERESIS_SHAPES, seeds=(1, 2, 3), envs=None):
    """Return (best (threshold, band, dwell), {key: mean eta}).

    The FULL product grid, not a coordinate descent from the greedy optimum:
    the best band can easily sit at a threshold that is not the best single
    threshold, and a baseline denied its joint optimum manufactures a win for
    whatever is being compared against it.

    The lower edge is clamped at 0, where the policy never returns to PM at
    all. That is a legal sweep point rather than an error -- it is
    "switch once and stay" -- and it keeps a wide band from being an
    exception the caller has to know about.
    """
    envs = _resolve_envs(env_factory, envs, seeds)
    return _sweep(envs, seeds,
                  _keyed(((float(t), float(band), int(dwell)),
                          HysteresisThreshold(t, max(0.0, t - band), dwell))
                         for t in thresholds for band, dwell in shapes))


def _mean_row(label, results):
    """One summary row. Each field averages ITS OWN key across episodes."""
    def mean(key):
        return statistics.mean(r[key] for r in results)
    return {"policy": label, "episodes": len(results), "eta": mean("eta"),
            "switches": mean("switches"), "expired": mean("expired"),
            "secret_bits": mean("secret_bits"),
            "ent_fraction": mean("ent_fraction"),
            "starved": mean("starved_fraction"),
            "mean_d_km": mean("mean_d_km")}


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
            thresholds=THRESHOLDS, reserves=RESERVES, sweeps=None,
            switches=None, horizons=BLOCK_HORIZONS,
            shapes=HYSTERESIS_SHAPES):
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

    # The public sweeps, not private copies of them: a second implementation
    # here is what would let the study run a cheaper search than the one the
    # tests pin (a coordinate descent, say) with every test still green.
    best_th, th_table = sweep_threshold(thresholds=thresholds, seeds=seeds,
                                        envs=envs)
    best_pair, ba_table = sweep_buffer_aware(thresholds=thresholds,
                                             reserves=reserves, seeds=seeds,
                                             envs=envs)
    # The distance-aware baselines get the same treatment as the others: their
    # own full grid, on the same envs and the same seeds. Their grid is
    # derived from the patrol band rather than fixed, so a mobility setting
    # the caller chose cannot leave them sweeping distances the vessel never
    # reaches.
    switches = (range_switches(params) if switches is None
                else tuple(float(d) for d in switches))
    best_d, rg_table = sweep_range(switches=switches, seeds=seeds, envs=envs)
    best_rq, rq_table = sweep_range_qber(switches=switches,
                                         thresholds=thresholds, seeds=seeds,
                                         envs=envs)
    best_h, bh_table = sweep_block_aware(horizons=horizons, seeds=seeds,
                                         envs=envs)
    best_hy, hy_table = sweep_hysteresis(thresholds=thresholds, shapes=shapes,
                                         seeds=seeds, envs=envs)
    if sweeps is not None:
        sweeps["threshold"] = th_table
        sweeps["buffer_aware"] = ba_table
        sweeps["range"] = rg_table
        sweeps["range_qber"] = rq_table
        sweeps["block_aware"] = bh_table
        sweeps["hysteresis"] = hy_table

    hy_th, hy_band, hy_dwell = best_hy
    candidates = [AlwaysPM(), AlwaysENT(), GreedyThreshold(best_th),
                  BufferAwareThreshold(*best_pair), RangeThreshold(best_d),
                  RangeOrQber(*best_rq),
                  HysteresisThreshold(hy_th, max(0.0, hy_th - hy_band),
                                      hy_dwell),
                  BlockAwareGreedy(horizon=best_h)]
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

    # The bounds go last: they are not policies, and reading them as one
    # would misreport the study. Each is a FRESH instance, so no two rows can
    # be the same object -- `rows[-1] is rows[-2]` would make the two numbers
    # trivially equal and the decomposition trivially zero.
    rows.extend(_mean_row(b.name, _score_bound(envs, b(), seeds))
                for b in BOUNDS)

    label_rows(rows)        # refuse a silent collision; the dict is a guard
    return rows


def switching_value(rows):
    """Split the Oracle's headroom over AlwaysPM into its two causes.

    `relaxation` is what the relaxed environment alone is worth (clairvoyance
    is idle for a policy that never switches, so this is free switching plus
    continuous delivery); `switching` is what CHOOSING between the modes is
    worth on top, which is the quantity the whole study is about. Reporting
    only `total` overstates switching's value several-fold.

    Returns None when the rows carry no bounds -- a caller that ran only
    baselines gets a missing number rather than a made-up one.
    """
    by = label_rows(rows)
    needed = (AlwaysPM.name, RelaxedAlwaysPM.name, Oracle.name)
    if any(k not in by for k in needed):
        return None
    pm, relaxed_pm, oracle = (by[k]["eta"] for k in needed)
    return {"always_pm": pm, "always_pm_relaxed": relaxed_pm,
            "oracle": oracle, "relaxation": relaxed_pm - pm,
            "switching": oracle - relaxed_pm, "total": oracle - pm}


def format_rows(rows):
    """The comparison table as text.

    "distilled" is GROSS distilled key, not delivered key: hybridenv counts a
    finished block before KeyBuffer clamps the pool at B_max, so bits thrown
    away by a full buffer are still in this column. Measured for AlwaysPM over
    one 3600-slot episode: at the spec defaults 21.0% of the column never
    reaches the buffer (4.988e7 distilled, 3.939e7 banked or spent), while at
    --lam 2.5 --b-max 1e6 none of it is lost. Two parameter settings printing
    the same figure therefore do NOT mean the same key economy.
    """
    head = ("%-26s %8s %9s %9s %12s %9s %9s %8s"
            % ("policy", "eta", "switches", "expired", "distilled",
               "ENT frac", "starved", "mean d"))
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append("%-26s %8.4f %9.1f %9.1f %12.3e %9.3f %9.3f %8.2f"
                     % (r["policy"], r["eta"], r["switches"], r["expired"],
                        r["secret_bits"], r["ent_fraction"], r["starved"],
                        r.get("mean_d_km", 0.0)))
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
                         "degenerate (AlwaysPM scores eta = 0.9957, and "
                         "exactly 1.0 on 7 of the 8 evaluation seeds)")
    ap.add_argument("--b-max", type=float, default=None,
                    help="key buffer capacity in bits, e.g. 1e6")
    ap.add_argument("--t0", type=int, default=None, help="deadline in slots")
    ap.add_argument("--ent-preset", default=None,
                    help="ENT operating point, e.g. ent_advanced. The hardware "
                         "assumption is the single biggest lever on whether "
                         "switching has any value at all, so it belongs on the "
                         "command line rather than buried in a default.")
    ap.add_argument("--pm-preset", default=None,
                    help="PM operating point (default aurea)")
    ap.add_argument("--mobility", action="store_true",
                    help="give the SSV a track: the link length varies over "
                         "the episode instead of sitting at --d-km. Off by "
                         "default, and with it off every number this script "
                         "prints is what it printed before mobility existed.")
    ap.add_argument("--d-km", type=float, default=None,
                    help="fixed link length, and the vessel's starting point "
                         "when --mobility is on (clamped into the band)")
    ap.add_argument("--d-min", type=float, default=None,
                    help="closest approach [km] (with --mobility)")
    ap.add_argument("--d-max", type=float, default=None,
                    help="furthest excursion [km] (with --mobility). At the "
                         "default speed limit a 3600-slot episode covers at "
                         "most 46 km, so a band much wider than that is "
                         "crossed at most once.")
    ap.add_argument("--v-max", type=float, default=None,
                    help="speed limit in km per slot; the default 0.0129 is "
                         "25 knots at a 1 s slot")
    ap.add_argument("--loiter-min", type=int, default=None,
                    help="fewest slots held on station at a waypoint")
    ap.add_argument("--loiter-max", type=int, default=None,
                    help="most slots held on station at a waypoint")
    ap.add_argument("--d-grid", type=float, default=None,
                    help="distance quantisation [km] for the physics cache "
                         "(default 0.5; <=0 recomputes every distance, which "
                         "is exact and much slower)")
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
    if args.ent_preset is not None:
        overrides["ent_preset"] = args.ent_preset
    if args.pm_preset is not None:
        overrides["pm_preset"] = args.pm_preset
    if args.d_km is not None:
        overrides["d_km"] = args.d_km
    if args.d_grid is not None:
        overrides["d_grid_km"] = args.d_grid
    # The band flags are read whether or not --mobility is given, but only
    # take effect through a MobilityParams(enabled=True): passing --d-max
    # alone must not quietly start the vessel moving.
    if args.mobility:
        track = {}
        for flag, field in (("d_min", "d_min"), ("d_max", "d_max"),
                            ("v_max", "v_max"), ("loiter_min", "loiter_min"),
                            ("loiter_max", "loiter_max")):
            value = getattr(args, flag)
            if value is not None:
                track[field] = value
        overrides["mobility"] = MobilityParams(enabled=True, **track)
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
    print("presets: pm=%s ent=%s" % (params.pm_preset, params.ent_preset))
    if params.mobility is not None and params.mobility.enabled:
        m = params.mobility
        print("mobility: d in [%.1f, %.1f] km, v_max=%.4f km/slot "
              "(<=%.1f km per %d-slot episode), loiter %d..%d slots, "
              "grid %.2f km"
              % (m.d_min, m.d_max, m.v_max, m.v_max * params.horizon,
                 params.horizon, m.loiter_min, m.loiter_max,
                 params.d_grid_km))
    else:
        print("mobility: off (fixed link at %.1f km)" % params.d_km)
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
        print("range switch sweep [km]:")
        for d in sorted(sweeps["range"]):
            print("  %8.2f  eta=%.4f" % (d, sweeps["range"][d]))
        print("range-or-qber grid (distance, threshold):")
        for key in sorted(sweeps["range_qber"]):
            print("  %8.2f %.4f  eta=%.4f"
                  % (key[0], key[1], sweeps["range_qber"][key]))
        print("block-aware planning horizon sweep [slots]:")
        for h in sorted(sweeps["block_aware"]):
            print("  %5d  eta=%.4f" % (h, sweeps["block_aware"][h]))
        print("hysteresis grid (threshold, band, dwell):")
        for key in sorted(sweeps["hysteresis"]):
            print("  %.4f %.4f %3d  eta=%.4f"
                  % (key[0], key[1], key[2], sweeps["hysteresis"][key]))
    print()
    print(format_rows(rows))

    split = switching_value(rows)
    if split is not None:
        print()
        print("upper bound (sees f_t, switches free, key delivered "
              "continuously):")
        print("  relaxation alone  %+.6f eta   (always-pm-relaxed - "
              "always-pm)" % split["relaxation"])
        print("  switching itself  %+.6f eta   (oracle - "
              "always-pm-relaxed)" % split["switching"])
        print("  total headroom    %+.6f eta   (oracle - always-pm)"
              % split["total"])


if __name__ == "__main__":
    main()
