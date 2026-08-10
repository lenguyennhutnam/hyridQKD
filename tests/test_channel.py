"""Tests for channel.py against DecoyState-sourced formulas."""
import dataclasses
import unittest

from channel import EntEnvParams, EnvParams, death_qber, derive


class TestDerive(unittest.TestCase):
    def test_eta_matches_decoystate_formula_at_d0(self):
        # eta = 10^(-(alpha*d + odr)/10) * eta_bob  (compute_final_length.py:212)
        d = derive(EnvParams())  # aurea defaults, d_km=0
        self.assertAlmostEqual(d.eta, 10 ** (-13.0 / 10) * 0.2, places=12)

    def test_eta_decreases_with_distance(self):
        etas = [derive(EnvParams(d_km=x)).eta for x in (0, 10, 30, 60)]
        self.assertEqual(etas, sorted(etas, reverse=True))

    def test_qber_grows_with_distance(self):
        # Dark counts dominate as the channel darkens -> E_avg rises toward 0.5.
        e0 = derive(EnvParams(d_km=0)).e_avg
        e60 = derive(EnvParams(d_km=60)).e_avg
        self.assertLess(e0, e60)
        self.assertLess(e60, 0.5 + 1e-9)

    def test_e_avg_close_to_edet_when_pdc_negligible(self):
        d = derive(EnvParams(d_km=0))
        self.assertAlmostEqual(d.e_avg, 0.01, delta=0.002)

    def test_predict_sane_at_aurea_d0(self):
        # Oracle neighborhood: DecoyState params_aurea gives l/nZ ~ 0.47 at d=0
        # (modeled per-intensity E); our uniform-E_avg mapping must stay same
        # order of magnitude and physical.
        d = derive(EnvParams(), n_z_block=10_000_000)
        self.assertGreater(d.l_predict, 1_000_000)
        self.assertLess(d.l_predict, d.n_z_block)

    def test_predict_zero_when_channel_dead(self):
        d = derive(EnvParams(d_km=200), n_z_block=10_000_000)
        self.assertEqual(d.l_predict, 0)

    def test_rate_estimate_positive_and_below_frep(self):
        d = derive(EnvParams())
        self.assertGreater(d.clicks_per_s, 0)
        self.assertLess(d.clicks_per_s, 80e6)

    def test_clicks_per_s_matches_hw_sim_rate_model(self):
        # hw_sim's actual stream rate: effective_eta * f_rep, where
        # effective_eta = p1*(1-exp(-mu1*eta)) + (1-p1)*(1-exp(-mu2*eta))
        # = d_avg - pdc. No dead-time (cdt) factor -- hw_sim has none.
        env = EnvParams()
        d = derive(env)
        self.assertAlmostEqual(d.clicks_per_s, (d.d_avg - env.pdc) * env.f_rep,
                               places=6)

    def test_pulse_distance_is_inverse_of_f_rep(self):
        env = EnvParams(f_rep=80e6)
        d = derive(env)
        self.assertAlmostEqual(d.pulse_distance, 1.0 / 80e6, places=20)

    def test_env_has_qber_sigma_default(self):
        self.assertEqual(EnvParams().qber_sigma, 0.001)


class TestDeathQber(unittest.TestCase):
    def test_death_qber_in_range_and_brackets_key_survival(self):
        env = EnvParams()
        q = death_qber(env)
        self.assertGreater(q, 0.0)
        self.assertLess(q, 0.15)

        from keyrate import KeyParams, secret_key_length
        eta = 10 ** (-(env.alpha * env.d_km + env.odr_losses) / 10) * env.eta_bob
        params = KeyParams(mu1=env.mu1, mu2=env.mu2, p1=env.p1, eta=eta,
                           pdc=env.pdc)
        self.assertGreater(secret_key_length(10_000_000, q - 0.005, params), 0)
        self.assertEqual(secret_key_length(10_000_000, q + 0.005, params), 0)


class TestGuiFieldsCoverage(unittest.TestCase):
    """The GUI form must expose exactly the fields of each dataclass.

    The form is split into a shared region plus one region per protocol;
    ENT keys carry an "ent_" prefix (eta_bob exists in both dataclasses).
    """

    def _gui(self):
        try:
            import gui
        except ImportError:
            self.skipTest("matplotlib/tkinter not available")
        return gui

    def test_gui_fields_match_env_params(self):
        gui = self._gui()
        field_names = {f for f, _ in gui.FIELDS_SHARED + gui.FIELDS_PM}
        env_field_names = {f.name for f in dataclasses.fields(EnvParams)}
        self.assertEqual(field_names, env_field_names)

    def test_gui_fields_match_ent_env_params(self):
        gui = self._gui()
        field_names = {f for f, _ in gui.FIELDS_SHARED}
        field_names |= {f[4:] for f, _ in gui.FIELDS_ENT}
        env_field_names = {f.name for f in dataclasses.fields(EntEnvParams)}
        self.assertEqual(field_names, env_field_names)

    def test_ent_entry_keys_are_prefixed(self):
        gui = self._gui()
        self.assertTrue(all(f.startswith("ent_") for f, _ in gui.FIELDS_ENT))
        shared_pm = {f for f, _ in gui.FIELDS_SHARED + gui.FIELDS_PM}
        # No entry key may be claimed twice (eta_bob collision guard).
        self.assertFalse(shared_pm & {f for f, _ in gui.FIELDS_ENT})


if __name__ == "__main__":
    unittest.main()
