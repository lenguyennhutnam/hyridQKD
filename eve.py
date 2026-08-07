#!/usr/bin/env python3
"""Eve: eavesdropper QBER model. Pure formulas, no I/O, protocol-agnostic.

Eve jams a fraction f of the time slots completely (intercept-resend of a random
state), which hw_sim reproduces by forcing QBER to 50% while jammed. The block-
averaged QBER is therefore a linear blend of the clean channel and 50% noise:

    E_eff(f) = e_base*(1 - f) + 0.5*f

f=0.5 gives ~25%, the textbook full intercept-resend QBER for BB84 AND BBM92, so
this model carries over to the ENT protocol unchanged (only e_base differs).
"""

from dataclasses import dataclass


@dataclass
class EveParams:
    """Eavesdropper control. f = fraction of slots jammed (0..1)."""

    f: float = 0.0
    period_s: float = 0.1   # duty-cycle period for the signal controller


def effective_qber(e_base, f):
    """Block-averaged QBER when Eve jams a fraction f of slots."""
    if not 0.0 <= f <= 1.0:
        raise ValueError("jamming fraction f must be in [0, 1], got %r" % f)
    return e_base * (1.0 - f) + 0.5 * f


def death_fraction(e_base, e_death):
    """Smallest jamming fraction f whose E_eff reaches the key-death threshold.

    Inverts effective_qber: f = (e_death - e_base) / (0.5 - e_base), clamped to
    [0, 1]. Returns 0.0 if the clean channel is already at/above e_death, and
    1.0 if e_death is at/above the 50% jamming ceiling (unreachable -> full jam).
    """
    if e_base >= e_death:
        return 0.0
    if e_death >= 0.5:
        return 1.0
    return max(0.0, min(1.0, (e_death - e_base) / (0.5 - e_base)))
