#!/usr/bin/env python3
"""Eve's jamming process for the hybrid mode-switching problem.

The jamming fraction f_t follows a 3-state Markov chain that the policy
CANNOT observe: it only sees the sliding-window QBER estimate, which is a
noisy function of f_t. That partial observability is the point -- with f_t
visible the problem collapses to a lookup table.

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
