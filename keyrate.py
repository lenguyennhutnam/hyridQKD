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
import math
import os
import sys

import numpy as np

# Rusca 2018 bound (1-decoy finite key) comes from Veriqloud's DecoyState repo
# and is vendored verbatim under third_party/ so this app runs from a clone of
# this repo alone -- nothing else in the workspace is needed to import it. Read
# third_party/NOTICE.md for its provenance and how it was checked against the
# source. Do NOT import the sibling ../DecoyState checkout: it is a second copy
# that could drift, and a bound silently differing between two machines is
# exactly the kind of disagreement no test here would catch.
_DECOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
if _DECOY_DIR not in sys.path:
    sys.path.insert(0, _DECOY_DIR)

from compute_final_length import compute_final_length, hbin  # noqa: E402


# Photon numbers summed over when a PNS attacker is present. The Poisson
# weight at n = 12 is below 1e-19 for every mu this study uses, so the
# truncation is far below the precision of anything downstream. The honest
# channel does not use the sum at all -- it has a closed form -- so pns = 0 is
# bit-for-bit what this module computed before the attack existed.
PNS_NMAX = 12


def _poisson(n, mu):
    return math.exp(-mu) * mu ** n / math.factorial(n)


class KeyParams:
    """Protocol + channel parameters for the finite-key bound.

    Defaults mirror hw_sim's decoy config (mu1/mu2/p1/eta) and the security
    parameters used in DecoyState/params_aurea.json.

    `pns` is the strength of a PHOTON-NUMBER-SPLITTING attack: the fraction of
    single-photon pulses the adversary blocks. See the module docstring for
    what she does with the rest, and tests/test_pns.py for what it costs.
    0 is the honest channel and is the default everywhere.
    """

    def __init__(self, mu1=0.5, mu2=0.1, p1=0.7, eta=0.1, pdc=1.33e-6,
                 esec=1e-10, ecor=1e-10, fEC=1.16, K=19, x_over_z=1.0,
                 pns=0.0):
        self.mu1, self.mu2, self.p1 = mu1, mu2, p1
        self.eta, self.pdc = eta, pdc
        self.esec, self.ecor, self.fEC, self.K = esec, ecor, fEC, K
        # hw_sim's 4 angles give a symmetric protocol: Alice states {0,3} form
        # the key basis, {1,2} the check basis, so nX == nZ.
        self.x_over_z = x_over_z
        if not 0.0 <= float(pns) <= 1.0:
            raise ValueError("pns is a fraction of blocked single-photon "
                             "pulses, got %r" % (pns,))
        self.pns = float(pns)

    # ---------------- the channel Bob observes ----------------

    def _honest_gain(self, mu):
        """Detection probability per pulse of intensity `mu`, no adversary.

        Closed form, and deliberately so: it is what this module always used,
        so the pns = 0 path cannot drift by a floating-point hair from every
        number the study has already published.
        """
        return 1 - np.exp(-mu * self.eta) + self.pdc

    def _honest_yield(self, n):
        """Probability Bob detects an n-photon pulse through the real link."""
        if n == 0:
            return self.pdc
        return 1 - (1 - self.eta) ** n + self.pdc

    def _pns_yields(self):
        """Per-photon-number yields the PNS attacker presents to Bob.

        Her construction, and the only thing she is allowed to condition on is
        the PHOTON NUMBER -- never the intensity label, which is exactly what
        decoy states buy:

          * a fraction `pns` of the SINGLE-photon pulses is blocked. Those are
            the only pulses a decoy bound certifies key from.
          * the multi-photon pulses are forwarded through her own lossless
            channel, at whatever rate holds Q_mu1 at its honest value, so the
            intensity Bob sees most of keeps its count rate. She splits one
            photon off each and reads it once the basis is announced, which
            costs no errors at all.

        Because the same yields apply to both intensities, holding Q_mu1 fixed
        necessarily drags Q_mu2 down -- and that is the entire leak the decoy
        analysis picks up.
        """
        lost = self.pns * _poisson(1, self.mu1) * self._honest_yield(1)
        supply = sum(_poisson(n, self.mu1) * (1 - self._honest_yield(n))
                     for n in range(2, PNS_NMAX))
        # She cannot forward more than every multi-photon pulse she has. If
        # the supply runs out she takes what she can and Q_mu1 does fall --
        # at every operating point in this study it does not run out.
        boost = min(1.0, lost / supply) if supply > 0 else 0.0

        def yields(n):
            if n == 0:
                return self.pdc
            if n == 1:
                return (1 - self.pns) * self._honest_yield(1)
            honest = self._honest_yield(n)
            return min(1.0, honest + boost * (1 - honest))
        return yields

    def gains(self):
        """(Q_mu1, Q_mu2): what Bob's detection probability actually is.

        These are OBSERVABLE -- Bob counts them -- which is why the attack
        shows up in their ratio and nowhere else.
        """
        if self.pns <= 0.0:
            return self._honest_gain(self.mu1), self._honest_gain(self.mu2)
        yields = self._pns_yields()
        return tuple(sum(_poisson(n, mu) * yields(n)
                         for n in range(PNS_NMAX))
                     for mu in (self.mu1, self.mu2))

    def click_factor(self):
        """Observed click rate as a fraction of the unattacked one.

        The number an operator watching only the count rate would have to
        notice: measured 0.946 at the attack strength that already certifies
        zero key.
        """
        if self.pns <= 0.0:
            return 1.0
        p2 = 1 - self.p1
        q1, q2 = self.gains()
        h1, h2 = self._honest_gain(self.mu1), self._honest_gain(self.mu2)
        return (self.p1 * q1 + p2 * q2) / (self.p1 * h1 + p2 * h2)

    def gain_ratio(self):
        """Q_mu1 / Q_mu2 -- the one observable a PNS attack moves."""
        q1, q2 = self.gains()
        return q1 / q2 if q2 else float("inf")

    def intensity_fraction(self):
        """Fraction of detections that came from the signal intensity mu1."""
        p2 = 1 - self.p1
        r1, r2 = self.gains()
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
