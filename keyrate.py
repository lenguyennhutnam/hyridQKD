#!/usr/bin/env python3
"""Secret-key length for one finite-key block (1-decoy BB84 / PM mode).

Bridges the qline collector to DecoyState's Rusca-2018 finite-key bound:

    measured (hw_sim)          modelled (channel formulas)
    -----------------          ---------------------------
    QBER, sifted count    ->   per-intensity split of the block
                          ->   compute_final_length()  ->  l (secret bits)

WHY THE SPLIT IS MODELLED, NOT COUNTED
    hw_sim tags each pulse mu1/mu2 with probability p1 but does NOT simulate
    photon-number statistics: a mu2 pulse is exactly as likely to produce a
    click as a mu1 pulse. Feeding those raw labels to the decoy bounds makes
    them infer a ~67% vacuum yield and return l > nZ, which is impossible.
    So the intensity split is taken from the standard gain model
    (R_i = 1 - exp(-mu_i*eta) + pdc) while QBER stays the live measured value.
    That keeps l responsive to the real channel while the decoy algebra sees a
    self-consistent photon-number picture.

Standalone check (sweeps QBER, prints where the key dies):
    python3 keyrate.py
"""

import contextlib
import io
import os
import sys

import numpy as np

# DecoyState lives next to qkd_app/ in the workspace.
_DECOY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "DecoyState")
if _DECOY_DIR not in sys.path:
    sys.path.insert(0, _DECOY_DIR)

from compute_final_length import compute_final_length, hbin  # noqa: E402


class KeyParams:
    """Protocol + channel parameters for the finite-key bound.

    Defaults mirror hw_sim's decoy config (mu1/mu2/p1/eta) and the security
    parameters used in DecoyState/params_aurea.json.
    """

    def __init__(self, mu1=0.5, mu2=0.1, p1=0.7, eta=0.1, pdc=1.33e-6,
                 esec=1e-10, ecor=1e-10, fEC=1.16, K=19, x_over_z=1.0):
        self.mu1, self.mu2, self.p1 = mu1, mu2, p1
        self.eta, self.pdc = eta, pdc
        self.esec, self.ecor, self.fEC, self.K = esec, ecor, fEC, K
        # hw_sim's 4 angles give a symmetric protocol: Alice states {0,3} form
        # the key basis, {1,2} the check basis, so nX == nZ.
        self.x_over_z = x_over_z

    def intensity_fraction(self):
        """Fraction of detections that came from the signal intensity mu1."""
        p2 = 1 - self.p1
        r1 = 1 - np.exp(-self.mu1 * self.eta) + self.pdc
        r2 = 1 - np.exp(-self.mu2 * self.eta) + self.pdc
        return self.p1 * r1 / (self.p1 * r1 + p2 * r2)


def secret_key_length(n_z, qber, params=None):
    """Secret bits distillable from a block of n_z sifted key-basis bits.

    n_z   : sifted bits in the key (Z) basis for this block
    qber  : measured error rate on the sifted bits (0..1)
    Returns 0 when the block is too noisy or too short to certify a key.
    """
    p = params or KeyParams()
    if n_z <= 0:
        return 0

    f1 = p.intensity_fraction()
    n_z1, n_z2 = n_z * f1, n_z * (1 - f1)
    n_x = n_z * p.x_over_z
    n_x1, n_x2 = n_x * f1, n_x * (1 - f1)

    # hw_sim applies one QBER to every pulse, so both intensities share it.
    m_z1, m_z2 = qber * n_z1, qber * n_z2
    m_x1, m_x2 = qber * n_x1, qber * n_x2

    l_ec = p.fEC * hbin(qber) * n_z

    # compute_final_length prints its intermediate bounds; keep callers' output clean.
    with contextlib.redirect_stdout(io.StringIO()):
        ell = compute_final_length(n_z, n_z1, n_z2, m_z1, m_z2,
                                   n_x, n_x1, n_x2, m_x1, m_x2,
                                   p.mu1, p.mu2, p.p1, l_ec,
                                   p.esec, p.ecor, p.K)

    if ell is None or ell <= 0:
        return 0
    if ell > n_z:
        # A key longer than the raw block is unphysical; treat the bound as
        # having broken down rather than reporting it.
        raise ValueError(f"unphysical key length {ell} > n_z {n_z} "
                         f"(qber={qber:.4f}) — check the parameter mapping")
    return int(ell)


if __name__ == "__main__":
    n_z = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000
    p = KeyParams()
    print(f"block n_z = {n_z:,} sifted key-basis bits")
    print(f"mu1={p.mu1} mu2={p.mu2} p1={p.p1} eta={p.eta} "
          f"-> mu1 share of detections = {p.intensity_fraction():.3f}")
    print(f"{'QBER':>7} {'secret bits':>14} {'bits/sifted':>12}")
    for q in (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
              0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.15, 0.30, 0.50, 0.90):
        ell = secret_key_length(n_z, q, p)
        print(f"{100*q:6.1f}% {ell:>14,} {ell/n_z:>12.4f}")
