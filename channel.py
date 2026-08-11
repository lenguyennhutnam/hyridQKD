#!/usr/bin/env python3
"""Channel model: environment parameters -> hw_sim knobs (eta, qberr) + forecast.

Pure formulas, no I/O. Every equation is lifted from the DecoyState repo (the
project's finite-key reference) — sources cited inline. See the design spec:
docs/superpowers/specs/2026-08-07-qkd-control-panel-design.md

HONESTY CAVEAT (spec §8): hw_sim does NOT simulate photon-number statistics —
it is a correlation model that flips bits with probability qberr, nothing
more. So `eta` only rate-limits how fast the stream produces clicks, and
`qberr` is injected verbatim rather than measured from optics. The gain/error
formulas below (e_det, pdc, dead time, ...) are only there to translate
"real-world" environment knobs into that single eta/qberr pair we hand to
hw_sim, and to give the decoy bound a self-consistent photon-number picture
(see keyrate.py). They do not mean hw_sim is modelling those effects itself.
"""

from dataclasses import dataclass

import numpy as np

import keyrate_ent
from keyrate import KeyParams, secret_key_length


@dataclass
class EnvParams:
    """Environment + protocol parameters (defaults = DecoyState params_aurea.json)."""

    d_km: float = 0.0          # Alice-Bob fiber length
    alpha: float = 0.23        # fiber attenuation [dB/km]
    odr_losses: float = 13.0   # fixed optics/receiver losses [dB]
    eta_bob: float = 0.2       # detector efficiency
    pdc: float = 1.33e-6       # dark-count probability per gate
    e_det: float = 0.01        # intrinsic optical error rate
    dead_us: float = 10.0      # detector dead time [us]
    coeff: float = 0.78        # dead-time correction factor (v13_compact)
    f_rep: float = 80e6        # laser repetition rate [Hz]
    mu1: float = 0.29          # signal intensity
    mu2: float = 0.103         # decoy intensity
    p1: float = 0.62           # P(signal)
    qber_sigma: float = 0.001  # Gaussian jitter std_dev on qberr (0 = fixed)


@dataclass
class ChannelDerived:
    """Derived quantities to write into hw_sim configs and show in the panel."""

    eta: float           # channel x detector transmittance -> hw_sim backend_config.eta
    d1: float            # gain of mu1: 1 - exp(-mu1*eta) + pdc
    d2: float            # gain of mu2
    e1: float            # modeled QBER of mu1 pulses
    e2: float            # modeled QBER of mu2 pulses
    d_avg: float         # p1*d1 + (1-p1)*d2
    e_avg: float         # gain-weighted QBER -> hw_sim backend_config.qberr
    cdt: float           # dead-time correction factor (hardware reference only,
                         # NOT applied to clicks_per_s -- see derive())
    clicks_per_s: float  # expected detection rate (display only)
    pulse_distance: float  # 1/f_rep -> hw_sim backend_config.pulse_distance
    l_predict: int       # forecast secret bits per block (0 = no key)
    n_z_block: int       # block size the forecast is for


def derive(env, n_z_block=10_000_000):
    """Map environment parameters to hw_sim knobs and a key-length forecast.

    Raises ValueError (from keyrate) if the decoy bound returns an unphysical
    length — callers should surface that instead of hiding it.
    """
    # eta = 10^(-(alpha*d + odr)/10) * eta_bob   [compute_final_length.py:212]
    eta = 10 ** (-(env.alpha * env.d_km + env.odr_losses) / 10) * env.eta_bob

    # Gains D_i = 1 - exp(-mu_i*eta) + pdc       [v13_compact.py:165-166]
    d1 = 1.0 - np.exp(-env.mu1 * eta) + env.pdc
    d2 = 1.0 - np.exp(-env.mu2 * eta) + env.pdc

    # E_i = ((1-exp(-mu_i*eta))*e_det + pdc/2) / D_i   [v13_compact.py:193-196,
    # with p_ap = 0 and coeff = 1: v1 has no afterpulse model]
    e1 = ((1.0 - np.exp(-env.mu1 * eta)) * env.e_det + env.pdc / 2.0) / d1
    e2 = ((1.0 - np.exp(-env.mu2 * eta)) * env.e_det + env.pdc / 2.0) / d2

    # Gain-weighted average -> the single qberr hw_sim applies to every pulse.
    g1, g2 = env.p1 * d1, (1.0 - env.p1) * d2
    d_avg = g1 + g2
    e_avg = (g1 * e1 + g2 * e2) / d_avg

    # Dead-time correction cdt = coeff/(1 + f_rep*D*tau)   [v13_compact.py:173]
    # This is a real hardware quantity (reported for comparison), but hw_sim
    # itself has NO detector dead time, so it must NOT be applied below.
    cdt = env.coeff / (1.0 + env.f_rep * d_avg * env.dead_us * 1e-6)

    # hw_sim's actual stream rate: effective_eta * f_rep, where effective_eta
    # = p1*(1-exp(-mu1*eta)) + (1-p1)*(1-exp(-mu2*eta)) = d_avg - pdc
    # [hw_sim/sim_lib/src/simulation/mod.rs:47-50,236-263]. No cdt factor.
    clicks_per_s = (d_avg - env.pdc) * env.f_rep
    pulse_distance = 1.0 / env.f_rep

    params = KeyParams(mu1=env.mu1, mu2=env.mu2, p1=env.p1, eta=eta, pdc=env.pdc)
    l_predict = secret_key_length(n_z_block, float(e_avg), params)

    return ChannelDerived(eta=float(eta), d1=float(d1), d2=float(d2),
                          e1=float(e1), e2=float(e2), d_avg=float(d_avg),
                          e_avg=float(e_avg), cdt=float(cdt),
                          clicks_per_s=float(clicks_per_s),
                          pulse_distance=float(pulse_distance),
                          l_predict=int(l_predict), n_z_block=int(n_z_block))


def death_qber(env, n_z_block=10_000_000, lo=0.0, hi=0.15, iters=25):
    """Bisect for the highest QBER that still yields a nonzero secret key.

    Holds eta and the decoy params (mu1, mu2, p1) fixed at env's values and
    sweeps only qber -- this is the "how noisy can the channel get before the
    key dies" threshold drawn on the QBER plot (spec §6), independent of the
    e_det/pdc mapping used to pick a single operating point in derive().

    Costs ~`iters` calls to compute_final_length (each is not free) -- call
    this once per Apply, never from the per-keystroke forecast.
    """
    eta = 10 ** (-(env.alpha * env.d_km + env.odr_losses) / 10) * env.eta_bob
    params = KeyParams(mu1=env.mu1, mu2=env.mu2, p1=env.p1, eta=eta, pdc=env.pdc)

    def alive(q):
        try:
            return secret_key_length(n_z_block, q, params) > 0
        except ValueError:
            return False

    # compute_final_length is degenerate for QBER extremely close to zero
    # (its parameter-estimation step effectively divides by an error count
    # that rounds to 0) and can spuriously report "dead" there, even though
    # that is not the real noise ceiling we are bisecting for (see CLAUDE.md
    # §11's exact-zero-QBER gotcha). Walk lo up geometrically until it lands
    # in the genuinely-alive region before bisecting down from hi.
    if not alive(lo):
        probe = max(lo, 1e-6)
        while probe < hi and not alive(probe):
            probe *= 10
        lo = probe
    if not alive(lo):
        return 0.0
    if alive(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if alive(mid):
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class EntEnvParams:
    """Environment + protocol parameters for ENT (BBM92) mode.

    Source-at-Alice model (Chapman 2022 App. A1): eta_a is Alice's LOCAL
    efficiency (never sees the channel), only eta_b carries the fiber loss.
    Defaults: channel params match the PM defaults; device params follow
    Chapman Table I (current system) with Alice's total efficiency 0.3
    (Fig. 8) and e_intr = mean of the two measured basis QBERs
    (0.88%, 1.85%).
    """

    d_km: float = 0.0          # Alice-Bob fiber length
    alpha: float = 0.23        # fiber attenuation [dB/km]
    odr_losses: float = 13.0   # fixed optics/receiver losses [dB]
    eta_a: float = 0.3         # Alice total efficiency (local arm)
    eta_bob: float = 0.4       # Bob detector efficiency (x channel = eta_b)
    xi_a: float = 1e-5         # background prob / pump pulse, Alice
    xi_b: float = 1e-5         # background prob / pump pulse, Bob
    e_intr: float = 0.0136     # intrinsic basis error e2
    mu: float = 0.045          # mean pairs per pump pulse (= 2*gamma)
    f_rep: float = 80e6        # pump repetition rate [Hz]
    qber_sigma: float = 0.001  # Gaussian jitter std_dev on qberr (0 = fixed)


@dataclass
class EntDerived:
    """Derived ENT quantities for hw_sim plain configs and the panel."""

    eta_b: float           # channel x detector transmittance, Bob's arm
    r_coinc: float         # coincidence prob per pulse -> hw_sim eta
    q2: float              # modeled QBER Q_2,est -> hw_sim qberr
    clicks_per_s: float    # expected coincidence rate (display only)
    pulse_distance: float  # 1/f_rep -> hw_sim pulse_distance
    l_predict: int         # forecast secret bits per block (0 = no key)
    n_z_block: int         # block size the forecast is for


def derive_ent(env, n_z_block=10_000_000):
    """Map ENT environment parameters to hw_sim plain-mode knobs + forecast.

    hw_sim knows nothing about photon pairs; we make its two knobs carry the
    BBM92 meaning: eta := R so the stream rate equals the coincidence rate,
    qberr := Q_2,est so the measured QBER fluctuates around the model value.
    Writing eta := R verbatim is correct because plain (non-decoy) mode uses
    the configured eta *directly* as the per-pulse click probability -- no
    transform to invert (mod.rs:236 `decoy_effective_eta().unwrap_or(self.eta)`,
    mod.rs:237-238 `BATCH_SIZE / effective_eta` pulses per batch).
    hw_sim's uniform angle choice sifts 50% -- exactly Chapman's basis-match
    probability (2 MUB chosen with p=1/2 each), so the collector's sifted
    count IS the raw key m with no correction factor.
    """
    eta_b = 10 ** (-(env.alpha * env.d_km + env.odr_losses) / 10) * env.eta_bob
    gamma = env.mu / 2.0
    r = keyrate_ent.coincidence_prob(gamma, env.eta_a, eta_b,
                                     env.xi_a, env.xi_b)
    q2 = keyrate_ent.q2_est(gamma, env.eta_a, eta_b,
                            env.xi_a, env.xi_b, env.e_intr)
    l_predict = keyrate_ent.secret_key_length_ent(n_z_block, q2)
    return EntDerived(eta_b=float(eta_b), r_coinc=float(r), q2=float(q2),
                      clicks_per_s=float(r * env.f_rep),
                      pulse_distance=1.0 / env.f_rep,
                      l_predict=int(l_predict), n_z_block=int(n_z_block))


def death_qber_ent(env, n_z_block=10_000_000, lo=0.0, hi=0.25, iters=25):
    """Bisect for the highest QBER with a nonzero ENT key.

    Unlike the PM bound, secret_key_length_ent is well-behaved at QBER=0
    (h2(0)=0), so no low-end probing is needed. env is accepted for
    signature symmetry with death_qber() but the ENT bound depends only on
    the block size.
    """
    def alive(q):
        return keyrate_ent.secret_key_length_ent(n_z_block, q) > 0

    if not alive(lo):
        return 0.0
    if alive(hi):
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if alive(mid):
            lo = mid
        else:
            hi = mid
    return lo
