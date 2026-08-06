"""Tests for channel.py against DecoyState-sourced formulas."""
import math
import unittest

from channel import EnvParams, derive


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


if __name__ == "__main__":
    unittest.main()
