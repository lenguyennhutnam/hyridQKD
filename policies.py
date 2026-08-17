#!/usr/bin/env python3
"""Baseline mode-selection policies, and the episode runner they share.

These exist so the DQN has something honest to beat. GreedyThreshold with a
swept threshold is a STRONG baseline, not a straw man: if the learned policy
cannot beat it, the correct conclusion is that the problem does not need
reinforcement learning.

There is deliberately no clairvoyant "oracle" upper bound here. A policy that
sees f_t and picks the better mode for the current slot is not an upper
bound: eq (5) destroys the partial block on every switch, so acting on the
instantaneous best mode thrashes and can score below AlwaysPM.
"""

from hybridenv import (ENT, I_AGE, I_BLOCK, I_BUFFER, I_MODE, I_QBER, I_QUEUE,
                       PM)

# The observation layout is defined once, in hybridenv, next to the _observe()
# that fixes the order. It is re-exported here only so a policy never has to
# index the state vector with a bare integer -- keeping a second definition
# would let the two drift apart silently.
__all__ = ["ENT", "PM", "I_QBER", "I_MODE", "I_BLOCK", "I_BUFFER", "I_QUEUE",
           "I_AGE", "Policy", "AlwaysPM", "AlwaysENT", "GreedyThreshold",
           "BufferAwareThreshold", "run_episode"]


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

        Non-increasing in `fullness`, which is the entire claim of the
        policy: a thinner buffer must never switch later than a fatter one.
        """
        if self.reserve <= 0.0 or fullness >= self.reserve:
            return self.threshold
        # Linear taper: full threshold at the reserve, half of it when empty.
        return self.threshold * (0.5 + 0.5 * fullness / self.reserve)

    def act(self, obs):
        if obs[I_QBER] >= self.effective_threshold(obs[I_BUFFER]):
            return ENT
        return PM


def run_episode(env, policy):
    """Run one full episode; return eta and the diagnostics that explain it.

    `eta` is read from the environment (eq 24: served / arrived), not
    recomputed here from per-slot counters, so the runner cannot disagree
    with the quantity the study reports.

    A "switch" is a change of mode between consecutive slots. The first slot
    has no previous mode and therefore can never be one -- an episode of N
    slots has at most N-1 switches.
    """
    policy.reset()
    obs = env.reset()
    done = False
    switches = expired = deposited = 0
    slots_empty = 0
    last_mode = None
    ent_slots = 0
    while not done:
        action = policy.act(obs)
        obs, _, done, info = env.step(action)
        if last_mode is not None and info["mode"] != last_mode:
            switches += 1
        last_mode = info["mode"]
        expired += info["expired"]
        deposited += info["deposited"]
        ent_slots += 1 if info["mode"] == ENT else 0
        slots_empty += 1 if info["buffer"] < env.params.l0 else 0
    return {"policy": policy.name, "eta": env.eta, "switches": switches,
            "expired": expired, "secret_bits": deposited,
            "ent_fraction": ent_slots / env.params.horizon,
            "starved_fraction": slots_empty / env.params.horizon}
