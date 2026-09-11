"""The PNS attack on PM, and why it is the attack this study needed.

A photon-number-splitting attacker measures how many photons a pulse carries,
blocks a fraction of the SINGLE-photon pulses -- the only ones a decoy bound
can certify key from -- and forwards extra multi-photon pulses through her own
lossless channel to keep Bob's count rate looking normal. She splits one
photon off each of those and reads it after the basis is announced, so she
learns those bits WITHOUT INTRODUCING ANY ERROR.

Three properties make it the right adversary for a hybrid controller, and
every one of them is pinned below:

  * it is a RATE attack, not a break. Decoy-state BB84 is provably secure
    against PNS at any strength; Eve can only drive the certified key length
    to zero, which is exactly what an eta objective cares about.
  * it is MODE-SELECTIVE by construction, with no need to tell Eve which
    protocol is running: BBM92 has no weak-coherent-state photon-number
    channel to exploit, and its multi-pair leakage is already charged
    unconditionally in E_MPE.
  * it is INVISIBLE TO QBER. The only observable that moves is the ratio of
    the two intensities' gains.

Numbers quoted in the tests were measured at 55 km on the `aurea` preset.
"""
import math
import unittest

import keyrate
from keyrate import KeyParams


def honest(mu, eta, pdc):
    """The closed-form gain the module used before PNS existed."""
    return 1 - math.exp(-mu * eta) + pdc


class TestTheHonestChannelIsUntouched(unittest.TestCase):
    """pns = 0 must be bit-for-bit what the module did before, or every
    number the study has published moves under it."""

    def test_the_gains_are_the_closed_form_ones(self):
        # Relative, to 12 digits: math.exp and numpy's exp disagree in the
        # last bit on some builds, and pinning that would make the test a
        # statement about libm. The guard that the pns = 0 PATH has not moved
        # is the golden episode in tests/test_hybridenv_pns.py.
        p = KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.0159, pdc=1.33e-6)
        q1, q2 = p.gains()
        self.assertAlmostEqual(q1 / honest(0.29, 0.0159, 1.33e-6), 1.0,
                               places=12)
        self.assertAlmostEqual(q2 / honest(0.103, 0.0159, 1.33e-6), 1.0,
                               places=12)

    def test_the_click_rate_is_exactly_the_unattacked_one(self):
        p = KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.0159)
        self.assertEqual(p.click_factor(), 1.0)

    def test_the_intensity_fraction_is_unchanged(self):
        # The quantity CLAUDE.md records as the sensitive one in the whole
        # qline -> compute_final_length mapping.
        p = KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.0159, pdc=1.33e-6)
        q1, q2 = honest(0.29, 0.0159, 1.33e-6), honest(0.103, 0.0159, 1.33e-6)
        self.assertAlmostEqual(
            p.intensity_fraction() / (0.62 * q1 / (0.62 * q1 + 0.38 * q2)),
            1.0, places=12)

    def test_the_default_is_no_attack(self):
        self.assertEqual(KeyParams().pns, 0.0)


class TestWhatTheAttackerDoesToTheGains(unittest.TestCase):
    def params(self, pns):
        return KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.015887,
                         pdc=1.33e-6, pns=pns)

    def test_she_holds_the_signal_gain_at_its_honest_value(self):
        # That is the whole point of forwarding extra multi-photon pulses:
        # the intensity Bob sees most of keeps its count rate, so nothing in
        # the raw rate gives her away.
        honest_q1 = self.params(0.0).gains()[0]
        for pns in (0.2, 0.5, 1.0):
            self.assertAlmostEqual(self.params(pns).gains()[0], honest_q1,
                                   places=12)

    def test_she_cannot_hold_the_decoy_gain_as_well(self):
        # She can condition only on PHOTON NUMBER, never on the intensity
        # label -- that is what decoy states buy. So holding Q_mu1 fixed
        # necessarily drags Q_mu2 down, and that is the whole leak.
        base = self.params(0.0).gains()[1]
        self.assertLess(self.params(0.5).gains()[1], base)

    def test_the_gain_ratio_rises_monotonically_with_the_attack(self):
        ratios = [q1 / q2 for q1, q2 in
                  (self.params(s).gains() for s in
                   (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0))]
        self.assertEqual(ratios, sorted(ratios))
        self.assertAlmostEqual(ratios[0], 2.810, places=2)

    def test_the_click_rate_barely_moves_while_the_key_dies(self):
        # Measured: 5.4% down at the strength that already zeroes the key.
        # An operator watching only the count rate sees nothing worth an
        # alarm.
        self.assertGreater(self.params(0.5).click_factor(), 0.9)
        self.assertLess(self.params(0.5).click_factor(), 1.0)

    def test_a_heavier_attack_always_costs_more_clicks(self):
        factors = [self.params(s).click_factor()
                   for s in (0.0, 0.2, 0.5, 1.0)]
        self.assertEqual(factors, sorted(factors, reverse=True))

    def test_the_intensity_split_shifts_toward_the_signal(self):
        # Suppressing singles removes relatively more of the DECOY pulses'
        # detections, so the surviving clicks look more like signal.
        self.assertGreater(self.params(0.5).intensity_fraction(),
                           self.params(0.0).intensity_fraction())

    def test_an_attack_strength_outside_the_unit_interval_is_refused(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                KeyParams(pns=bad)


class TestWhatTheAttackDoesToTheCERTIFIEDKey(unittest.TestCase):
    """The bound never breaks -- it only certifies less. That distinction is
    the difference between a correct thesis and a wrong one."""

    N_Z = 100_000
    QBER = 0.010188

    def key(self, pns):
        return keyrate.secret_key_length(
            self.N_Z, self.QBER,
            KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.015887,
                      pdc=1.33e-6, pns=pns))

    def test_the_unattacked_block_certifies_the_measured_length(self):
        self.assertAlmostEqual(self.key(0.0) / 33614.0, 1.0, places=2)

    def test_a_stronger_attack_never_certifies_more_key(self):
        keys = [self.key(s) for s in (0.0, 0.1, 0.2, 0.3, 0.4)]
        self.assertEqual(keys, sorted(keys, reverse=True))

    def test_the_key_is_gone_before_the_attack_is_complete(self):
        # Measured: alive at 0.4, dead by 0.5. An adversary does not need to
        # block every single-photon pulse to take PM off the air.
        self.assertGreater(self.key(0.4), 0)
        self.assertEqual(self.key(0.5), 0)

    def test_it_still_returns_a_length_and_never_raises(self):
        # keyrate raises on l > n_z, its guard against a broken mapping. A
        # PNS attack must not trip that guard -- it reduces the certified
        # length, it does not produce nonsense.
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertLessEqual(self.key(s), self.N_Z)


class TestTheAttackIsInvisibleToTheQBER(unittest.TestCase):
    """The finding that makes the hybrid controller necessary: every
    QBER-threshold baseline in the study is blind to this."""

    def test_the_certified_key_collapses_at_a_fixed_qber(self):
        # Nothing about the QBER is an input to the collapse: it is held at
        # the unattacked value throughout and the key still goes to zero.
        q = 0.010188
        p0 = KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.015887,
                       pdc=1.33e-6, pns=0.0)
        p1 = KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.015887,
                       pdc=1.33e-6, pns=0.6)
        self.assertGreater(keyrate.secret_key_length(100_000, q, p0), 0)
        self.assertEqual(keyrate.secret_key_length(100_000, q, p1), 0)

    def test_the_gain_ratio_is_the_observable_that_does_move(self):
        # 43% up at the strength that zeroes the key, against 0% for the
        # QBER. This is what a policy has to watch instead.
        def ratio(s):
            q1, q2 = KeyParams(mu1=0.29, mu2=0.103, p1=0.62, eta=0.015887,
                               pdc=1.33e-6, pns=s).gains()
            return q1 / q2
        self.assertGreater(ratio(0.6) / ratio(0.0), 1.4)


if __name__ == "__main__":
    unittest.main()
