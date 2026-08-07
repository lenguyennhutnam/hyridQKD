"""Tests for eve.py — the eavesdropper QBER model (protocol-agnostic)."""
import unittest

from eve import EveParams, effective_qber, death_fraction


class TestEffectiveQber(unittest.TestCase):
    def test_no_jamming_returns_base(self):
        self.assertAlmostEqual(effective_qber(0.0103, 0.0), 0.0103)

    def test_full_jamming_is_half(self):
        self.assertAlmostEqual(effective_qber(0.0103, 1.0), 0.5)

    def test_half_jamming_is_intercept_resend_25pct(self):
        # f=0.5 -> ~25%, the textbook intercept-resend QBER.
        self.assertAlmostEqual(effective_qber(0.0, 0.5), 0.25)

    def test_monotonic_in_f(self):
        es = [effective_qber(0.01, f) for f in (0.0, 0.1, 0.3, 0.6, 1.0)]
        self.assertEqual(es, sorted(es))

    def test_f_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            effective_qber(0.01, -0.1)
        with self.assertRaises(ValueError):
            effective_qber(0.01, 1.5)


class TestDeathFraction(unittest.TestCase):
    def test_roundtrip_with_effective_qber(self):
        # The f death_fraction returns must map back to e_death via effective_qber.
        f = death_fraction(0.0103, 0.052)
        self.assertAlmostEqual(effective_qber(0.0103, f), 0.052, places=6)

    def test_aurea_death_around_8pct(self):
        # e_base=1.03%, death threshold ~5.2% -> f ~ 0.085 (report figure).
        f = death_fraction(0.0103, 0.052)
        self.assertGreater(f, 0.07)
        self.assertLess(f, 0.10)

    def test_already_dead_returns_zero(self):
        self.assertEqual(death_fraction(0.10, 0.05), 0.0)

    def test_threshold_at_or_above_half_returns_one(self):
        self.assertEqual(death_fraction(0.01, 0.5), 1.0)
        self.assertEqual(death_fraction(0.01, 0.6), 1.0)


if __name__ == "__main__":
    unittest.main()
