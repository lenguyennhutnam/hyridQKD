#!/usr/bin/env python3
"""Channel model: environment parameters -> hw_sim knobs (eta, qberr) + forecast.

Pure formulas, no I/O. Every equation is lifted from the DecoyState repo (the
project's finite-key reference) — sources cited inline. See the design spec:
docs/superpowers/specs/2026-08-07-qkd-control-panel-design.md
"""

from dataclasses import dataclass

import numpy as np

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
    cdt: float           # dead-time correction factor
    clicks_per_s: float  # expected detection rate (display only)
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
    cdt = env.coeff / (1.0 + env.f_rep * d_avg * env.dead_us * 1e-6)
    clicks_per_s = env.f_rep * d_avg * cdt

    params = KeyParams(mu1=env.mu1, mu2=env.mu2, p1=env.p1, eta=eta, pdc=env.pdc)
    l_predict = secret_key_length(n_z_block, float(e_avg), params)

    return ChannelDerived(eta=float(eta), d1=float(d1), d2=float(d2),
                          e1=float(e1), e2=float(e2), d_avg=float(d_avg),
                          e_avg=float(e_avg), cdt=float(cdt),
                          clicks_per_s=float(clicks_per_s),
                          l_predict=int(l_predict), n_z_block=int(n_z_block))
