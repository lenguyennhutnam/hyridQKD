#!/usr/bin/env python3
"""Baseline mode-selection policies, and the episode runner they share.

These exist so the DQN has something honest to beat. GreedyThreshold with a
swept threshold is a STRONG baseline, not a straw man: if the learned policy
cannot beat it, the correct conclusion is that the problem does not need
reinforcement learning.

Alongside them are the two REFERENCE BOUNDS of spec section 7, `Oracle` and
`RelaxedAlwaysPM`. They are not policies and cannot be run by run_episode:
they live in a relaxed environment (HybridEnv.step_relaxed) that no real
controller could have. A negative result -- "switching does not pay" -- says
almost nothing on its own; with the Oracle it becomes "no policy of any kind
could have gained more than X", which is the claim worth publishing.

A myopic clairvoyant in the UNRELAXED environment is indeed no upper bound:
eq (5) destroys the partial block on every switch, so acting on the
instantaneous best mode thrashes and can score below AlwaysPM. That refutes
one construction, not the concept; the bound here relaxes the switching cost
away instead of paying it.
"""

from hybridenv import (ENT, I_AGE, I_BLOCK, I_BUFFER, I_MODE, I_QBER, I_QUEUE,
                       PM)

# The observation layout is defined once, in hybridenv, next to the _observe()
# that fixes the order. It is re-exported here only so a policy never has to
# index the state vector with a bare integer -- keeping a second definition
# would let the two drift apart silently.
__all__ = ["ENT", "PM", "I_QBER", "I_MODE", "I_BLOCK", "I_BUFFER", "I_QUEUE",
           "I_AGE", "Policy", "AlwaysPM", "AlwaysENT", "GreedyThreshold",
           "BufferAwareThreshold", "RelaxationBound", "Oracle",
           "RelaxedAlwaysPM", "run_episode", "run_relaxed_episode"]


class Policy:
    """Interface every policy shares."""

    name = "policy"

    def reset(self):
        """Called once per episode, before the first act()."""

    def act(self, obs):
        """Return PM or ENT for the slot described by `obs`."""
        raise NotImplementedError


class AlwaysPM(Policy):
    """Control condition: never switch. At short range this is hard to beat."""

    name = "always-pm"

    def act(self, obs):
        return PM


class AlwaysENT(Policy):
    """Control condition: never switch, the other way."""

    name = "always-ent"

    def act(self, obs):
        return ENT


class GreedyThreshold(Policy):
    """Switch to ENT once the measured QBER reaches `threshold`.

    The comparison is `>=`, not `>`: the natural setting for `threshold` is
    PM's death QBER, and *at* that point PM already distils nothing.
    """

    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.name = "greedy-%.3f" % self.threshold

    def act(self, obs):
        return ENT if obs[I_QBER] >= self.threshold else PM


class BufferAwareThreshold(Policy):
    """Move earlier when the key buffer is thin.

    ENT delivers nothing until a whole block completes, so switching at the
    moment PM dies leaves a hole the buffer must cover. `reserve` is the
    fraction of B_max below which the policy starts switching at a lower
    QBER -- the effective threshold shrinks as the buffer drains, down to
    half the nominal threshold at an empty buffer.

    Above `reserve` it is exactly GreedyThreshold: with stock in hand there
    is no gap worth pre-empting. reserve = 0 therefore degenerates to the
    plain threshold, which is a legal (and useful) sweep point.
    """

    def __init__(self, threshold, reserve):
        self.threshold = float(threshold)
        self.reserve = float(reserve)
        self.name = "buffer-aware-%.3f-%.2f" % (self.threshold, self.reserve)

    def effective_threshold(self, fullness):
        """The QBER at which this policy switches, given B_t / B_max.

        Non-DECREASING in `fullness`, which is the entire claim of the
        policy: a thinner buffer must never switch later than a fatter one.
        (It rises from half the nominal threshold at an empty buffer to the
        full threshold at `reserve`; reading it the other way round invites
        exactly the wrong edit.)
        """
        if self.reserve <= 0.0 or fullness >= self.reserve:
            return self.threshold
        # Linear taper: full threshold at the reserve, half of it when empty.
        return self.threshold * (0.5 + 0.5 * fullness / self.reserve)

    def act(self, obs):
        if obs[I_QBER] >= self.effective_threshold(obs[I_BUFFER]):
            return ENT
        return PM


def _rollout(env, label, seed, advance):
    """Run one episode and account for it. `advance(obs) -> env step tuple`.

    The accounting lives here ONCE so that a relaxed run and a real run
    cannot report their eta, switches or expiries by different rules -- the
    whole point of the bound is that it is measured on the same ruler as the
    thing it bounds.
    """
    obs = env.reset() if seed is None else env.reset(seed)
    done = False
    switches = expired = deposited = 0
    slots_empty = 0
    last_mode = None
    ent_slots = 0
    while not done:
        obs, _, done, info = advance(obs)
        if last_mode is not None and info["mode"] != last_mode:
            switches += 1
        last_mode = info["mode"]
        expired += info["expired"]
        deposited += info["deposited"]
        ent_slots += 1 if info["mode"] == ENT else 0
        slots_empty += 1 if info["buffer"] < env.params.l0 else 0
    return {"policy": label, "eta": env.eta, "switches": switches,
            "expired": expired, "secret_bits": deposited,
            "ent_fraction": ent_slots / env.params.horizon,
            "starved_fraction": slots_empty / env.params.horizon}


class RelaxationBound:
    """A reference upper bound, run under HybridEnv.step_relaxed.

    Deliberately NOT a Policy subclass, and it has no act(). run_episode()
    would happily accept one and score it under the true dynamics, producing
    a number that is neither the bound nor a policy -- staying outside the
    hierarchy turns that mistake into an AttributeError instead of a
    plausible-looking extra row.
    """

    name = "bound"

    def reset(self):
        """Called once per episode, before the first choose()."""

    def choose(self, f, rates, obs):
        """Return PM or ENT, given the TRUE jamming fraction `f` this slot,
        `rates` = (kappa_PM(f), kappa_ENT(f)), and the observation a
        realisable policy would have had instead."""
        raise NotImplementedError


class Oracle(RelaxationBound):
    """Spec section 7's mandatory bound: clairvoyant, free switching,
    continuous delivery (the three relaxations are HybridEnv.step_relaxed's).

    Per slot it takes the argmax of kappa_m(f_t). Under the relaxation that
    is optimal, not merely sensible: deposits are the only coupling from the
    quantum side into eq (2), the buffer level is monotone in the deposits
    and service is monotone in the buffer level, so maximising every slot's
    deposit maximises the buffer at every instant and therefore weakly
    maximises the messages served. Once switching is free there is nothing
    left to trade off across slots, which is precisely why the myopic
    construction fails only in the unrelaxed problem.

    Ties go to PM: under heavy jamming both rates can be 0, and a bound that
    switched on a tie would report switches that buy nothing.
    """

    name = "oracle"

    def choose(self, f, rates, obs):
        return ENT if rates[ENT] > rates[PM] else PM


class RelaxedAlwaysPM(RelaxationBound):
    """AlwaysPM under the SAME relaxation. Not decoration -- the whole point.

    Oracle minus AlwaysPM conflates two different things: the value of the
    relaxation (free switching, continuous delivery, a clairvoyant view) and
    the value of SWITCHING. Only Oracle minus this row isolates the second,
    and at the measured operating points most of the apparent headroom turns
    out to be the first.
    """

    name = "always-pm-relaxed"

    def choose(self, f, rates, obs):
        return PM


def run_relaxed_episode(env, bound, seed=None):
    """Run `bound` for one episode of the RELAXED environment.

    Same return shape as run_episode, deliberately, so a bound drops straight
    into the comparison table. `switches` and `ent_fraction` still mean what
    they mean elsewhere -- they are what shows an Oracle actually switching
    rather than settling on one mode.
    """
    bound.reset()
    return _rollout(env, bound.name, seed,
                    lambda obs: env.step_relaxed(bound.choose))


def run_episode(env, policy, seed=None):
    """Run one full episode; return eta and the diagnostics that explain it.

    `eta` is read from the environment (eq 24: served / arrived), not
    recomputed here from per-slot counters, so the runner cannot disagree
    with the quantity the study reports.

    A "switch" is a change of mode between consecutive slots. The first slot
    has no previous mode and therefore can never be one -- an episode of N
    slots has at most N-1 switches.

    `seed` is handed straight to env.reset(). seed=None keeps reset()'s
    no-argument semantics, which replay ONE trajectory byte for byte -- that
    is what lets Task 7 score several policies against identical jamming on a
    single env. An integer draws a different episode deterministically.

    Task 6 MUST pass a fresh seed per training episode, or the agent trains
    on one trajectory forever; Task 7 MUST evaluate on a seed set disjoint
    from training's, or the agent is scored on what it memorised. The seed
    goes here rather than through a new HybridEnv per episode because
    constructing one re-runs channel.derive()/derive_ent(), which is by far
    the most expensive thing in the loop and does not depend on the seed.
    """
    policy.reset()
    return _rollout(env, policy.name, seed,
                    lambda obs: env.step(policy.act(obs)))
