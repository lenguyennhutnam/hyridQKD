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

import math

from hybridenv import (DIST_REF_KM, ENT, I_AGE, I_BLOCK, I_BUFFER, I_DECOY,
                       I_DIST, I_MODE, I_QBER, I_QUEUE, PM)

# The observation layout is defined once, in hybridenv, next to the _observe()
# that fixes the order. It is re-exported here only so a policy never has to
# index the state vector with a bare integer -- keeping a second definition
# would let the two drift apart silently.
__all__ = ["ENT", "PM", "I_QBER", "I_MODE", "I_BLOCK", "I_BUFFER", "I_QUEUE",
           "I_AGE", "I_DIST", "I_DECOY", "DIST_REF_KM", "Policy", "AlwaysPM",
           "AlwaysENT", "GreedyThreshold", "BufferAwareThreshold",
           "HysteresisThreshold", "BlockAwareGreedy",
           "RangeThreshold", "RangeOrQber", "RelaxationBound", "Oracle",
           "RelaxedAlwaysPM", "run_episode", "run_relaxed_episode",
           "observed_distance_km"]


def observed_distance_km(obs):
    """The link length in km that `obs` reports (hybridenv.I_DIST is scaled).

    One place undoes the normalisation, so a change to DIST_REF_KM cannot
    leave one range policy reading kilometres and another reading a fraction.
    """
    return float(obs[I_DIST]) * DIST_REF_KM


class Policy:
    """Interface every policy shares."""

    name = "policy"

    def bind(self, model):
        """Called by run_episode with the environment this episode runs in.

        Only MODEL-BASED policies need it (BlockAwareGreedy); for everything
        else it is a no-op, which is why it lives here rather than in a second
        base class. The reason it is a per-episode hook and not a constructor
        argument: eval_hybrid._score runs ONE policy object over every seed's
        environment, so a model latched at construction would answer for the
        first seed's hardware while running inside the last seed's episode.
        """

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


class RangeThreshold(Policy):
    """PM while the SSV is closer than `d_switch`; ENT at or beyond it.

    The distance is the half of the weather the vessel actually KNOWS (Eve's
    jamming is not), so this is the cheapest policy that can exploit the one
    thing mobility adds. It ignores the QBER entirely -- deliberately: paired
    with RangeOrQber it separates "how much is knowing the range worth" from
    "how much does adding the inferred signal add on top".

    `>=` matches GreedyThreshold's convention, and for the same reason: the
    natural setting is PM's death DISTANCE, at which PM already distils
    nothing.
    """

    def __init__(self, d_switch):
        self.d_switch = float(d_switch)
        self.name = "range-%.1f" % self.d_switch

    def act(self, obs):
        return ENT if observed_distance_km(obs) >= self.d_switch else PM


class RangeOrQber(Policy):
    """ENT when the SSV is far OR the link looks noisy; PM otherwise.

    The two signals fail in different ways and neither subsumes the other:
    the range is exact but says nothing about jamming, while the windowed
    QBER sees jamming but is lagged by up to w_window slots and noisy in
    proportion to the block. `or` rather than `and` because either condition
    alone can already have killed PM, and waiting for both is waiting for PM
    to be dead twice over.

    Setting `d_switch` beyond the patrol band makes this exactly
    GreedyThreshold(threshold), so the sweep grid always contains the
    QBER-only policy and this row can never score below it because of the
    search.
    """

    def __init__(self, d_switch, threshold):
        self.d_switch = float(d_switch)
        self.threshold = float(threshold)
        self.name = "range-qber-%.1f-%.3f" % (self.d_switch, self.threshold)

    def act(self, obs):
        if observed_distance_km(obs) >= self.d_switch:
            return ENT
        return ENT if obs[I_QBER] >= self.threshold else PM


class _ModeDwell:
    """Counts the slots the OBSERVED mode has been running.

    Both anti-thrash policies need it, for different reasons -- one to refuse
    a reversal, one to wait out the QBER window -- so the bookkeeping lives
    once, here.

    It is driven by obs[I_MODE], the mode the ENVIRONMENT actually ran, not by
    the policy's own last decision. They cannot disagree today, but a rule
    counting its own intentions would keep counting through a slot the
    environment refused, which is the sort of drift that shows up only as a
    number that will not reproduce.

    The first slot of an episode enters at `entry` rather than 0: the
    environment starts in PM by construction, and treating that as a switch
    the policy just made would hold a link that is already noisy on a dead
    mode for no reason.
    """

    def _dwell_reset(self):
        self._dwell = 0
        self._dwell_mode = None

    def _dwell_tick(self, mode, entry):
        if self._dwell_mode is None:
            self._dwell_mode = mode
            self._dwell = int(entry)
        elif mode != self._dwell_mode:
            self._dwell_mode = mode
            self._dwell = 0
        else:
            self._dwell += 1
        return self._dwell


class HysteresisThreshold(_ModeDwell, Policy):
    """Switch up at `up`, and only back down at `down` -- with `down` < `up`.

    GreedyThreshold has ONE threshold, so a QBER dithering around it switches
    on every crossing, and each crossing pays eq (5): the partial block is
    destroyed. Measured, that is not a hypothetical -- greedy-0.045 at
    chapman_lowmu switches 30 times and lands BELOW always-pm.

    This is the classical control answer to that, and it deliberately needs
    NO model of the hardware: it is the honest comparand for BlockAwareGreedy,
    which derives its band instead of being given one. `min_dwell` is the
    second classical knob -- a floor on how long a mode is held, in slots,
    whatever the QBER does -- and it caps the switch count at
    horizon / min_dwell by construction rather than by tuning.

    down == up is a legal (and useful) sweep point: it is exactly
    GreedyThreshold(up), so a grid over the band always contains the
    single-threshold baseline this generalises.

    The current mode is read from the OBSERVATION (m_{t-1}), not from a
    private copy: the environment is what decides which mode actually ran,
    and a policy tracking its own idea of that would drift the moment the two
    disagreed.
    """

    def __init__(self, up, down, min_dwell=0):
        up, down = float(up), float(down)
        if down > up:
            raise ValueError("hysteresis needs down <= up, got %g > %g"
                             % (down, up))
        self.up = up
        self.down = down
        self.min_dwell = int(min_dwell)
        self.name = "hysteresis-%.3f-%.3f-%d" % (up, down, self.min_dwell)
        self.reset()

    def reset(self):
        # Cleared per episode, so seed 2's first slots cannot depend on
        # seed 1's last ones.
        self._dwell_reset()

    def act(self, obs):
        mode = ENT if int(obs[I_MODE]) == ENT else PM
        dwell = self._dwell_tick(mode, self.min_dwell)
        want = mode
        if mode == PM and obs[I_QBER] >= self.up:
            want = ENT
        elif mode == ENT and obs[I_QBER] <= self.down:
            want = PM
        if want != mode and dwell < self.min_dwell:
            return mode
        return want


class BlockAwareGreedy(_ModeDwell, Policy):
    """Switch only when the gain outruns the block the switch destroys.

    GreedyThreshold's defect is not its threshold, it is that it compares the
    two modes and ignores the COST of acting on the comparison: eq (5) wipes
    the partial block on every change of mode. This policy prices that.

    The rule counts the WHOLE BLOCKS each choice retires inside a planning
    horizon H, with m the mode running, m' the other one, and b the sifted
    bits already banked toward the current block:

        stay = l_m(f_hat)  * floor((b + ns_m  * H) / n_X)
        go   = l_m'(f_hat) * floor(     ns_m' * H  / n_X)
        switch  <=>  go > stay

    where l_m(f) = kappa_m(f) * n_X / ns_m is one finished block's key.

    WHOLE blocks, not a continuous rate, and that is the load-bearing choice.
    Continuous delivery is relaxation 3 of the Oracle bound; a policy pricing
    its own decisions that way would be quietly awarding itself the
    relaxation it is meant to sit below, and it shows: the continuous form
    switches 48 times at chapman_lowmu to a mode that never finishes a block
    inside the horizon, and lands BELOW always-pm.

    Three properties fall out of the arithmetic rather than being bolted on:

    * HYSTERESIS. `b` can only ever tip the count by one block, and it does
      so only when the current mode is close to finishing -- so a block just
      started is free to abandon and one nearly complete is not. No second
      threshold and no tuned dead band; the width comes from the physics.
    * A BOUND ON THRASHING. Switching is only ever free near a block
      boundary, so the mode cannot change much more often than once a block.
    * NO MOVE TO A MODE THAT CANNOT DELIVER IN TIME. If m' needs longer than
      H to fill one block, floor() makes `go` exactly zero however rich its
      rate looks. This is what the continuous form gets wrong.

    And the case where a fixed dead band would be actively harmful is handled
    for free: when the current mode is dead, l_m = 0 makes `stay` zero and the
    policy leaves at once, however full the block.

    WHAT IT IS ALLOWED TO KNOW. `model` supplies kappa_m(.), the error floors
    e_m and the sifted rates ns_m -- all of them design-time properties of the
    declared hardware, the same quantities gui.py forecasts before a run. It
    is NOT clairvoyant: the true f_t never reaches it, and the jamming is
    estimated by inverting E_eff(f) = e_m (1 - f) + f/2 on the windowed QBER
    the observation carries, lag, binomial noise and all. That is the whole
    difference between this and `Oracle`.

    `horizon` is the planning horizon H in slots; None means the message
    deadline T0, on the grounds that key arriving later than the deadline
    cannot rescue the messages currently at risk.

    `settle` is the SETTLING TIME of that estimator, in slots, and it is not
    optional. The observed QBER averages the last w_window slots, so for
    w_window slots after a change of mode it is a BLEND of the two modes and
    inverting it against either floor is meaningless -- and the error has a
    sign that reverses with the mode, which is the worst possible kind. At
    ent_advanced (floors 0.0102 and 0.0458) the blend reads "no jamming"
    while on ENT and "f = 0.073" while on PM, so the rule flips every slot:
    measured, 2,140 switches in 3,600 slots and eta = 0.12 against
    always-pm's 0.61. Holding the new mode until the window has flushed
    removes it. None means w_window, which is exactly the contamination
    length; 0 disables it and reproduces the oscillation.

    `f_step` quantises the estimate. HybridEnv.kappa memoises on (mode, f,
    distance) and a miss costs a finite-key evaluation, so an unquantised
    estimate would miss on all 3,600 slots of an episode. DECLARED
    APPROXIMATION: the rule is evaluated at a jamming level at most
    f_step/2 from the estimate. f_step <= 0 disables it (exact, slow).
    """

    def __init__(self, horizon=None, f_step=0.005, settle=None):
        self.horizon = None if horizon is None else int(horizon)
        self.f_step = float(f_step)
        self.settle = None if settle is None else int(settle)
        self.name = ("block-aware" if self.horizon is None
                     else "block-aware-%d" % self.horizon)
        self.model = None
        self.reset()

    def bind(self, model):
        self.model = model

    def reset(self):
        self._dwell_reset()

    def _model(self):
        if self.model is None:
            raise RuntimeError(
                "%s must be bound to an environment before it can act; "
                "run_episode() does that, a bare act() does not" % self.name)
        return self.model

    def planning_horizon(self):
        """H, in slots. The message deadline unless one was given."""
        if self.horizon is not None:
            return self.horizon
        return int(self._model().params.t0_deadline)

    def settling_slots(self):
        """Slots to hold a new mode while the QBER window flushes."""
        if self.settle is not None:
            return self.settle
        return int(self._model().params.w_window)

    def estimated_jamming(self, obs):
        """f_hat: E_eff(f) = e_m (1 - f) + f/2 inverted on the windowed QBER.

        Clipped to [0, 1] because a lucky binomial draw can put the measured
        QBER below the floor, and a negative fraction would ask the rate model
        for a point outside its domain.
        """
        model = self._model()
        mode = ENT if int(obs[I_MODE]) == ENT else PM
        floor = model.error_floor[mode]
        span = 0.5 - floor
        if span <= 0.0:
            return 0.0
        f = (float(obs[I_QBER]) - floor) / span
        f = min(1.0, max(0.0, f))
        if self.f_step > 0.0:
            f = min(1.0, max(0.0, round(f / self.f_step) * self.f_step))
        return f

    def _key_over(self, mode, f, banked, horizon):
        """Secret bits `mode` retires in `horizon` slots, starting `banked`
        sifted bits into a block. Zero when it cannot finish even one."""
        model = self.model
        n_x = model.params.n_x
        sifted = model.sifted_per_slot[mode]
        if not sifted or not n_x:
            return 0.0
        blocks = math.floor((banked + sifted * horizon) / n_x)
        per_block = model.kappa(mode, f) * n_x / sifted
        return per_block * blocks

    def act(self, obs):
        model = self._model()
        mode = ENT if int(obs[I_MODE]) == ENT else PM
        other = PM if mode == ENT else ENT
        settle = self.settling_slots()
        if self._dwell_tick(mode, settle) < settle:
            # The estimator has not flushed yet; anything decided from this
            # observation would be decided from the previous mode's QBER.
            return mode
        f = self.estimated_jamming(obs)
        horizon = self.planning_horizon()
        banked = float(obs[I_BLOCK]) * model.params.n_x
        stay = self._key_over(mode, f, banked, horizon)
        # eq (5): switching wipes the partial block, so the other mode starts
        # from zero. That subtraction IS the switching cost -- there is no
        # separate penalty term anywhere in this policy.
        go = self._key_over(other, f, 0.0, horizon)
        return other if go > stay else mode


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
    # Counted rather than taken from params.horizon: mean_d_km is the one
    # figure that shows whether the vessel moved at all, and dividing it by a
    # length the loop did not actually run would quietly bias it.
    slots = 0
    d_sum = 0.0
    while not done:
        obs, _, done, info = advance(obs)
        if last_mode is not None and info["mode"] != last_mode:
            switches += 1
        last_mode = info["mode"]
        expired += info["expired"]
        deposited += info["deposited"]
        ent_slots += 1 if info["mode"] == ENT else 0
        slots_empty += 1 if info["buffer"] < env.params.l0 else 0
        slots += 1
        d_sum += info["d_km"]
    return {"policy": label, "eta": env.eta, "switches": switches,
            "expired": expired, "secret_bits": deposited,
            "ent_fraction": ent_slots / env.params.horizon,
            "starved_fraction": slots_empty / env.params.horizon,
            "mean_d_km": d_sum / slots if slots else 0.0}


class RelaxationBound:
    """A reference upper bound, run under HybridEnv.step_relaxed.

    Deliberately NOT a Policy subclass, and it has no act(). run_episode()
    would happily accept one and score it under the true dynamics, producing
    a number that is neither the bound nor a policy -- staying outside the
    hierarchy turns that mistake into an AttributeError instead of a
    plausible-looking extra row.
    """

    name = "bound"

    def bind(self, model):
        """As Policy.bind. Present so a runner can bind either kind without
        asking which it is holding; no bound currently needs it."""

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
    bound.bind(env)
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
    policy.bind(env)
    policy.reset()
    return _rollout(env, policy.name, seed,
                    lambda obs: env.step(policy.act(obs)))
