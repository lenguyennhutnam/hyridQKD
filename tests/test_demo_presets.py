"""Tests for the demo_* presets: same channel as thesis, distinct hardware.

The demo_* presets exist so that a defense run (or a saved policyviz panel)
can pin one name and not silently drift the channel between the PM and ENT
operating points -- `aurea` ships odr_losses = 13 dB while `ent_advanced`
ships 0 dB, and comparing them unedited accepts PM ~20x transmittance
(CLAUDE.md section "So sánh PM với ENT — hai bẫy"). These three carry the
same 55 km / 0 dB / alpha = 0.2 channel as policyviz's default VizConfig.
"""
import dataclasses
import unittest

import presets
from channel import EnvParams, EntEnvParams


class TestDemoPresets(unittest.TestCase):
    CHANNEL = dict(d_km=55.0, alpha=0.2, odr_losses=0.0)

    def setUp(self):
        self.pm = presets.load_presets()
        self.ent = presets.load_ent_presets()

    def test_all_three_exist(self):
        self.assertIn("demo_pm", self.pm)
        self.assertIn("demo_ent_advanced", self.ent)
        self.assertIn("demo_ent_current", self.ent)

    def test_demo_pm_is_aurea_with_channel_replaced(self):
        base = dataclasses.replace(self.pm["aurea"], d_km=55.0,
                                      alpha=0.2, odr_losses=0.0)
        # aurea is authored at d_km=0 / alpha=0.23 / odr=13; the demo preset
        # is the same hardware lifted onto the shared 55 km channel.
        self.assertEqual(self.pm["demo_pm"], base)

    def test_demo_ent_advanced_matches_ent_advanced(self):
        base = self.ent["ent_advanced"]
        demo = self.ent["demo_ent_advanced"]
        self.assertEqual(dataclasses.replace(base, **self.CHANNEL), demo)

    def test_demo_ent_current_matches_chapman_current(self):
        base = self.ent["chapman_current"]
        demo = self.ent["demo_ent_current"]
        self.assertEqual(dataclasses.replace(base, **self.CHANNEL), demo)

    def test_all_share_one_channel(self):
        for env in (self.pm["demo_pm"], self.ent["demo_ent_advanced"],
                    self.ent["demo_ent_current"]):
            for key, want in self.CHANNEL.items():
                self.assertEqual(getattr(env, key), want, key)

    def test_demo_pm_is_pm_and_demoo_ent_is_ent(self):
        self.assertIsInstance(self.pm["demo_pm"], EnvParams)
        self.assertIsInstance(self.ent["demo_ent_advanced"], EntEnvParams)


if __name__ == "__main__":
    unittest.main()
