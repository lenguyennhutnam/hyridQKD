"""ENT branch of the channel model: env params -> hw_sim knobs (plain mode)."""
import unittest

import keyrate_ent as ke
from channel import EntEnvParams, derive_ent, death_qber_ent


class TestDeriveEnt(unittest.TestCase):
    def test_eta_config_is_coincidence_prob(self):
        env = EntEnvParams()
        d = derive_ent(env)
        eta_b = 10 ** (-(env.alpha * env.d_km + env.odr_losses) / 10) * env.eta_bob
        self.assertAlmostEqual(d.eta_b, eta_b)
        self.assertAlmostEqual(
            d.r_coinc,
            ke.coincidence_prob(env.mu / 2, env.eta_a, eta_b, env.xi_a, env.xi_b))

    def test_qberr_config_is_q2_est(self):
        env = EntEnvParams()
        d = derive_ent(env)
        eta_b = d.eta_b
        self.assertAlmostEqual(
            d.q2, ke.q2_est(env.mu / 2, env.eta_a, eta_b, env.xi_a, env.xi_b,
                            env.e_intr))
        self.assertGreater(d.q2, 0.0)
        self.assertLess(d.q2, 0.5)

    def test_rate_and_pulse_distance(self):
        env = EntEnvParams(f_rep=400e6)
        d = derive_ent(env)
        self.assertAlmostEqual(d.clicks_per_s, d.r_coinc * 400e6)
        self.assertAlmostEqual(d.pulse_distance, 1 / 400e6)

    def test_short_link_produces_key(self):
        # Chapman-like short link must forecast a nonzero key
        env = EntEnvParams(d_km=0.0, odr_losses=0.0, eta_bob=0.24, alpha=0.2)
        d = derive_ent(env, n_z_block=10_000_000)
        self.assertGreater(d.l_predict, 0)

    def test_distance_monotonic(self):
        # longer fiber -> lower coincidence rate, higher modeled QBER
        near = derive_ent(EntEnvParams(d_km=0.0))
        far = derive_ent(EntEnvParams(d_km=100.0))
        self.assertLess(far.r_coinc, near.r_coinc)
        self.assertGreaterEqual(far.q2, near.q2)


class TestDeathQberEnt(unittest.TestCase):
    def test_threshold_in_expected_band(self):
        # no decoy penalty: death near the 2.12*h2(Q)=1 point (~11%),
        # a bit lower from finite statistics
        t = death_qber_ent(EntEnvParams(), n_z_block=10_000_000)
        self.assertGreater(t, 0.08)
        self.assertLess(t, 0.13)


if __name__ == "__main__":
    unittest.main()
