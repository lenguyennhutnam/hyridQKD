#!/usr/bin/env python3
"""Secret-key length for one finite-key block (BBM92 / ENT mode).

Analytic model from J. C. Chapman, C. C. W. Lim, P. G. Kwiat,
"Hyperentangled Time-bin and Polarization Quantum Key Distribution",
Phys. Rev. Applied 18, 044027 (2022), d=2 branch. Equation numbers below
refer to its Appendix A; the local extraction (checked line-by-line against
the PDF) is docs/chapman2022-bbm92-formula-sheet.md.

Model assumptions (App. A1): pair source in Alice's lab (one-way loss —
eta_a is local, only eta_b carries the channel), both bases generate key
with p=1/2, multi-pair detections get a random bit (never discarded).
No decoy states: multi-pair effects live in the error terms instead.

Two call paths, mirroring keyrate.py:
  - runtime: measured QBER from the collector goes straight into
    secret_key_length_ent(m_raw, qber).
  - forecast: q2_est() models the expected QBER from channel parameters.

Security parameters default to eps_sec = eps_cor = 1e-10, matching the PM
branch (project decision; Chapman Table I uses 1e-9/1e-12).

Standalone check (sweeps QBER, prints where the key dies):
    python3 keyrate_ent.py
"""

import math
import sys

# Truncate the infinite photon-pair sums at n = 10 pairs/pulse
# (Chapman Sec. V C: P_10 ~ 1e-10 at gamma = 0.1, negligible).
N_MAX = 10

_E0 = 0.5  # error probability of a random bit assignment, e0 = (d-1)/d, d=2


def pair_prob(n, gamma):
    """(A3) P_n: probability of n photon pairs in one pump pulse.

    gamma = sinh^2(chi); mean pairs per pulse mu = 2*gamma.
    """
    return (n + 1) * gamma**n / (1 + gamma)**(n + 2)


def yield_n(n, eta_a, eta_b, xi_a, xi_b):
    """(A5) Y_n: P(coincidence | n pairs emitted), incl. background xi."""
    return ((1.0 - (1.0 - xi_a) * (1.0 - eta_a)**n)
            * (1.0 - (1.0 - xi_b) * (1.0 - eta_b)**n))


def coincidence_prob(gamma, eta_a, eta_b, xi_a, xi_b):
    """(A6) R: coincidence probability per pump pulse, closed form."""
    return (1.0
            - (1.0 - xi_a) / (1.0 + eta_a * gamma)**2
            - (1.0 - xi_b) / (1.0 + eta_b * gamma)**2
            + (1.0 - xi_a) * (1.0 - xi_b)
            / (1.0 + eta_a * gamma + eta_b * gamma
               - eta_a * eta_b * gamma)**2)


def _background_errors(gamma, eta_a, eta_b, xi_a, xi_b):
    """(A7) E_b: background-related errors.

    The paper's inner binomial sums collapse: sum_{k>=1} C(n,k) eta^k
    (1-eta)^(n-k) = 1 - (1-eta)^n.
    """
    total = 0.0
    for n in range(N_MAX + 1):
        pn = pair_prob(n, gamma)
        t1 = (1.0 - (1.0 - eta_a)**n) * xi_b * (1.0 - eta_b)**n
        t2 = (1.0 - (1.0 - eta_b)**n) * xi_a * (1.0 - eta_a)**n
        t3 = xi_a * xi_b * (1.0 - eta_a)**n * (1.0 - eta_b)**n
        total += pn * (t1 + t2 + t3)
    return _E0 * total


def _phi_errors(gamma, eta_a, eta_b, e2):
    """(A11) E_|Phi_n>: errors when ALL produced photons are detected.

    d=2 reduction: d*(n!)^2*N_n^2 = 2/(n+1) with N_n^2 = 1/(n!*(n+1)!),
    and e_d/(d-1) = e2.
    """
    total = 0.0
    for n in range(1, N_MAX + 1):
        pn = pair_prob(n, gamma)
        detect_all = (eta_a * eta_b)**n
        same_mode = (e2**n) * 2.0 / (n + 1)
        total += pn * detect_all * same_mode
        if n >= 2:
            total += pn * detect_all * (1.0 - same_mode) * _E0
    return total


def _multi_pair_errors(gamma, eta_a, eta_b):
    """(A12) E_MPE: >=2 pairs created, >=1 photon detected on each side,
    minus the all-detected case (already in E_|Phi_n>).

    NOTE: the published (A12) prints eta_A in BOTH sums — an inconsistency
    with its own subtracted term eta_A^n * eta_B^n. The second sum must be
    over Bob's side (eta_B); we implement the corrected form (see formula
    sheet, verified against the PDF).
    """
    total = 0.0
    for n in range(2, N_MAX + 1):
        pn = pair_prob(n, gamma)
        pa = 1.0 - (1.0 - eta_a)**n
        pb = 1.0 - (1.0 - eta_b)**n
        total += pn * (pa * pb - (eta_a**n) * (eta_b**n))
    return _E0 * total


def q2_est(gamma, eta_a, eta_b, xi_a, xi_b, e2):
    """(A13) Q_2,est: modeled QBER upper bound = (E_b+E_Phi+E_MPE)/R.

    Capped at 0.5 (random-guess ceiling) — the bound over-counts slightly
    by construction and can exceed it for pathological parameters.
    """
    r = coincidence_prob(gamma, eta_a, eta_b, xi_a, xi_b)
    if r <= 0.0:
        return 0.5
    q = (_background_errors(gamma, eta_a, eta_b, xi_a, xi_b)
         + _phi_errors(gamma, eta_a, eta_b, e2)
         + _multi_pair_errors(gamma, eta_a, eta_b)) / r
    return min(q, 0.5)


def h2(x):
    """(A18) binary entropy; 0 outside (0,1) so callers can pass edge values."""
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


# Scan grids for the (A19) maximisation. r: key-generation share of the raw
# block; beta in (0, eps_sec/4) as fractions of the open upper end.
_R_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99)
_BETA_FRACS = (0.999, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-6)


def secret_key_length_ent(m_raw, qber, eps_sec=1e-10, eps_cor=1e-10):
    """Secret bits distillable from a block of m_raw sifted bits (both bases).

    m_raw : sifted bits in this block (BBM92 keys BOTH bases, so this is the
            full sifted count, not a single-basis subset).
    qber  : measured (or modeled) error rate on the sifted bits (0..1).
    Optimises r and beta by grid scan, per (A16)-(A20). Returns 0 when the
    block is too noisy or too short to certify a key.
    """
    if m_raw <= 0:
        return 0
    log_cor = math.log2(2.0 / eps_cor)
    leak_h = 1.12 * h2(qber)          # (A16) Leak_EC coefficient per key bit
    best = 0
    for r in _R_GRID:
        n = int(m_raw * r)
        k = m_raw - n
        if n <= 0 or k <= 1:
            continue
        for frac in _BETA_FRACS:
            beta = (eps_sec / 4.0) * frac
            # (A17) statistical fluctuation from finite parameter estimation
            delta = math.sqrt((n + k) / (n * k) * (k + 1) / k
                              * math.log(1.0 / beta))
            q_pe = qber + delta
            if q_pe >= 0.5:
                continue  # h2 would bend back down and fake a key
            # (A20) extractable length, (A19) secret length
            n_ext = n * (1.0 - h2(q_pe)) - leak_h * n - log_cor
            ell = math.floor(n_ext + 4.0 * math.log2(beta) - 2.0)
            if ell > best:
                best = ell
    if best > m_raw:
        raise ValueError("unphysical key length %d > m_raw %d (qber=%.4f)"
                         % (best, m_raw, qber))
    return best


if __name__ == "__main__":
    m = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10_000_000
    print("block m_raw = {:,} sifted bits (both bases)".format(m))
    print("{:>7} {:>14} {:>12}".format("QBER", "secret bits", "bits/sifted"))
    for q in (0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.09,
              0.10, 0.105, 0.11, 0.12, 0.15):
        ell = secret_key_length_ent(m, q)
        print("{:6.1f}% {:>14,} {:>12.4f}".format(100 * q, ell, ell / m))
