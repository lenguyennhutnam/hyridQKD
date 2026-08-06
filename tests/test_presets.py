"""Tests for preset loading/saving."""
import os
import tempfile
import unittest

from channel import EnvParams
from presets import load_presets, save_preset


class TestPresets(unittest.TestCase):
    def test_builtin_presets_load(self):
        p = load_presets()
        self.assertIn("aurea", p)
        self.assertIn("hwsim_default", p)
        self.assertIsInstance(p["aurea"], EnvParams)
        self.assertEqual(p["aurea"].mu1, 0.29)
        self.assertEqual(p["hwsim_default"].mu1, 0.5)

    def test_save_then_load_roundtrip(self):
        env = EnvParams(d_km=25.0, e_det=0.02)
        path = save_preset("_tmp_test", env)
        try:
            p = load_presets()
            self.assertEqual(p["_tmp_test"].d_km, 25.0)
            self.assertEqual(p["_tmp_test"].e_det, 0.02)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
