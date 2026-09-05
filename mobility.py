#!/usr/bin/env python3
"""The SSV's own motion: a waypoint process with a hard speed limit.

The other exogenous process in this study is Eve (`evemarkov`), and this one
is deliberately built to the same shape -- a params dataclass, a seeded object
with `reset(seed)` and `step()`, no I/O, no environment -- so the two can be
read side by side and tested the same way.

They differ in exactly one respect, and it is the point of the model:

    source of variation      does the policy observe it?
    ----------------------   ---------------------------
    distance (the SSV moves)              YES
    jamming  (Eve)                        no

A vessel knows its own position, so the distance is a legitimate input to the
mode decision (`hybridenv.I_DIST`). The jamming fraction stays hidden and can
only be inferred from the lagged, noisy windowed QBER. Making one signal known
and the other unknown is what gives the hybrid controller something to do
beyond reacting to noise.

Motion model
------------
Pick a target uniformly in [d_min, d_max]; each slot close at most `v_max` km
of the gap; on arrival loiter for a uniformly drawn number of slots, then pick
a new target. The vessel therefore travels, it does not teleport:

    |d_t - d_{t-1}| <= v_max     for every t, every seed

is the defining property of this module and is pinned by a test rather than
left to inspection.

Speed anchor
------------
`v_max` is quoted per SLOT because that is the unit the environment steps in.
A 25-knot surface vessel makes 25 * 1.852 = 46.3 km/h; with the study's
1-second slot that is 46.3 / 3600 = 0.01286 km per slot. That is the anchor
for the default; a slot length other than 1 s needs it rescaled.

PLACEHOLDER DEFAULTS
--------------------
`d_min` / `d_max` / `loiter_*` are placeholders, not findings. The band is
centred on `HybridParams.d_km = 55` (+/- 15 km) purely so that enabling
mobility with no arguments patrols around the link the rest of the study
already uses, and so that no operating range is smuggled in as a default. The
real range comes from the deployment literature and is meant to be passed in
(`eval_hybrid.py --d-min/--d-max`, or `MobilityParams(...)` directly).

Note what a bounded speed costs: at `v_max = 0.0129` km/slot a 3600-slot
episode covers at most 46 km of travel. A band much wider than that is
crossed at most once per episode, and a band wider than ~92 km cannot be
traversed and returned from at all. Ranges must be chosen against the horizon.
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["MobilityParams", "VesselTrack"]


@dataclass
class MobilityParams:
    """Where the vessel may go and how fast it gets there.

    Every limit is a parameter; see the module docstring for which of them
    are anchored to something real (`v_max`) and which are placeholders
    (`d_min`, `d_max`, `loiter_min`, `loiter_max`).
    """

    # OFF by default. With enabled=False nothing in this module runs and
    # HybridEnv keeps the fixed-distance behaviour it has always had.
    enabled: bool = False
    d_min: float = 40.0             # closest approach to HQ [km]
    d_max: float = 70.0             # furthest excursion [km]
    v_max: float = 0.0129           # speed limit [km per slot]; 25 kn @ 1 s
    loiter_min: int = 0             # slots held on station on arrival
    loiter_max: int = 600           # ...drawn uniformly from [min, max]
    seed: int = 0


class VesselTrack:
    """Draws d_t. Deterministic given the seed, so two policies can be
    compared on the identical track as well as identical jamming."""

    def __init__(self, params=None, d_start=None):
        """`d_start` is where the episode begins, CLAMPED into [d_min, d_max].

        Clamping rather than raising is deliberate: the caller is
        `HybridEnv`, whose `d_km` is the fixed-link distance and has no reason
        to sit inside a patrol band chosen later. Starting outside the band
        would otherwise burn most of an episode steaming toward it at
        `v_max` -- 100 km of approach is 7,752 slots, more than the 3,600-slot
        horizon, so the vessel would never reach the band the study is about.
        """
        self.params = params or MobilityParams()
        p = self.params
        if p.d_min > p.d_max:
            raise ValueError("d_min %g exceeds d_max %g" % (p.d_min, p.d_max))
        if p.d_min < 0.0:
            raise ValueError("d_min %g is negative" % p.d_min)
        if p.v_max <= 0.0:
            raise ValueError("v_max must be positive, got %g" % p.v_max)
        if p.loiter_min < 0 or p.loiter_max < p.loiter_min:
            raise ValueError("loiter range [%r, %r] is empty or negative"
                             % (p.loiter_min, p.loiter_max))
        self._d_start = None if d_start is None else float(d_start)
        self._rng = np.random.RandomState(p.seed)
        self._d = 0.0
        self._target = 0.0
        self._loiter = 0
        self.reset()

    def reset(self, seed=None):
        """Rewind to the start of a track; return the starting distance.

        seed=None replays the one track fixed by `params.seed`, exactly as
        `EveChain.reset()` replays one jamming trajectory -- that is what lets
        several policies be scored on identical weather.
        """
        p = self.params
        self._rng = np.random.RandomState(p.seed if seed is None
                                          else int(seed))
        start = p.d_min if self._d_start is None else self._d_start
        self._d = float(min(max(start, p.d_min), p.d_max))
        self._loiter = 0
        self._target = self._draw_target()
        return self._d

    def _draw_target(self):
        return float(self._rng.uniform(self.params.d_min, self.params.d_max))

    def _draw_loiter(self):
        # randint's high bound is exclusive, so +1 makes loiter_max reachable
        # and loiter_min == loiter_max a legal (fixed-dwell) setting.
        return int(self._rng.randint(self.params.loiter_min,
                                     self.params.loiter_max + 1))

    def step(self):
        """Advance one slot; return the new distance d_t.

        The step is capped at `v_max` in both branches: on the final approach
        the remaining gap is by definition no larger than `v_max`, so snapping
        to the target cannot break the speed bound.
        """
        p = self.params
        if self._loiter > 0:
            self._loiter -= 1
            return self._d
        gap = self._target - self._d
        if abs(gap) <= p.v_max:
            self._d = self._target          # |move| = |gap| <= v_max
            self._loiter = self._draw_loiter()
            self._target = self._draw_target()
        else:
            self._d += math.copysign(p.v_max, gap)
        return self._d

    @property
    def distance(self):
        """Current d_t [km]."""
        return self._d

    @property
    def target(self):
        """The waypoint currently being steamed toward [km]."""
        return self._target

    @property
    def loitering(self):
        """Slots still to be spent on station before the next leg."""
        return self._loiter
