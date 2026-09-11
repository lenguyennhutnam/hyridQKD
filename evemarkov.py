#!/usr/bin/env python3
"""Eve's attack process for the hybrid mode-switching problem.

Her posture follows a 3-state Markov chain that the policy CANNOT observe: it
only sees noisy functions of the state. That partial observability is the
point -- with the state visible the problem collapses to a lookup table.

TWO LEVERS, ONE CHAIN
---------------------
`f_levels` is the JAMMING fraction: it raises the QBER of whichever mode is
running, so it hurts BOTH and is plainly visible in the windowed QBER.

`s_levels` is the strength of a PHOTON-NUMBER-SPLITTING attack (see
keyrate.KeyParams.pns). It certifies PM's key away while introducing NO
errors, and BBM92 has no weak-coherent-state photon-number channel for it to
touch -- so it is mode-selective without Eve having to know which mode is
running, and it is INVISIBLE to the QBER. Measured at 55 km: s = 0.5 takes
PM's certified key to zero while the QBER does not move and the click rate
falls 5.4%.

They ride ONE chain deliberately. Eve commits her hardware to a posture per
slot; a second chain would also double the unobservable state the policy has
to infer, for no modelled reason. A state may carry either lever, both, or
neither.

`s_levels` defaults to all zeros, so every result measured before the PNS
attack existed replays bit for bit -- the chain consumes the same random
draws either way.

State f values are chosen from the measured kappa table (spec section 5):
QUIET and LIGHT are regimes where PM wins, HEAVY is the regime where PM
produces zero key and ENT still produces some.
"""

from dataclasses import dataclass

import numpy as np

QUIET, LIGHT, HEAVY = 0, 1, 2


@dataclass
class EveParams:
    """Jamming levels and how the adversary moves between them."""

    # f per state. At the design channel PM dies at f = 4.77%, ENT at 11.62%.
    f_levels: tuple = (0.0, 0.03, 0.08)
    # Row i = P(next | current = i). Mean dwell 40 / 25 / 40 slots: long enough
    # that switching mode can pay back the lost block, short enough that a
    # 3600-slot episode sees many transitions.
    transition: tuple = ((0.975, 0.020, 0.005),
                         (0.030, 0.960, 0.010),
                         (0.010, 0.015, 0.975))
    # PNS strength per state -- the fraction of single-photon pulses the
    # adversary blocks. All zero means "no PNS anywhere", which is what every
    # published result was measured at.
    s_levels: tuple = (0.0, 0.0, 0.0)
    seed: int = 0


class EveChain:
    """Draws f_t. Deterministic given the seed, so policies can be compared
    on identical jamming trajectories."""

    def __init__(self, params=None):
        self.params = params or EveParams()
        self._matrix = np.asarray(self.params.transition, dtype=float)
        # Row-wise cumulative distribution: one uniform draw per step beats
        # np.random.choice, which re-validates the probability vector on every
        # call. The dwell-time test alone advances the chain 200k times.
        self._cumulative = self._matrix.cumsum(axis=1)
        self._levels = np.asarray(self.params.f_levels, dtype=float)
        self._pns_levels = np.asarray(self.params.s_levels, dtype=float)
        if self._pns_levels.shape != self._levels.shape:
            raise ValueError("s_levels has %d entries and f_levels %d; every "
                             "state needs both"
                             % (self._pns_levels.size, self._levels.size))
        if self._pns_levels.min() < 0.0 or self._pns_levels.max() > 1.0:
            raise ValueError("a PNS level is a fraction of blocked "
                             "single-photon pulses, got %s"
                             % (self.params.s_levels,))
        self._state = QUIET
        self._rng = np.random.RandomState(self.params.seed)

    def reset(self, seed=None):
        """Rewind to the quiet state; return that state index."""
        self._rng = np.random.RandomState(
            self.params.seed if seed is None else seed)
        self._state = QUIET
        return self._state

    def step(self):
        """Advance one slot; return the new f_t."""
        draw = self._rng.rand()
        self._state = int(np.searchsorted(self._cumulative[self._state], draw))
        # Guard against a cumulative row ending a hair below 1.0 in floating
        # point, which searchsorted would map past the last state.
        if self._state >= len(self._levels):
            self._state = len(self._levels) - 1
        return self.f

    @property
    def state(self):
        return self._state

    @property
    def f(self):
        return float(self._levels[self._state])

    @property
    def pns(self):
        """PNS strength in the current state.

        A separate property rather than a second return value from step(), so
        every existing caller and every existing test keeps working unchanged.
        """
        return float(self._pns_levels[self._state])
