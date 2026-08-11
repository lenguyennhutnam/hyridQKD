"""Tests for preset loading/saving."""
import os
import tempfile
import unittest

import presets
from channel import EntEnvParams, EnvParams
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


class TestEntPresets(unittest.TestCase):
    def test_pm_loader_skips_ent_files(self):
        names = set(presets.load_presets())
        self.assertNotIn("chapman_current", names)
        self.assertNotIn("chapman_future", names)
        self.assertIn("aurea", names)  # PM presets untouched

    def test_ent_loader_returns_ent_params(self):
        ent = presets.load_ent_presets()
        self.assertIn("chapman_current", ent)
        self.assertIn("chapman_future", ent)
        cur = ent["chapman_current"]
        self.assertIsInstance(cur, EntEnvParams)
        self.assertAlmostEqual(cur.eta_a, 0.3)
        self.assertAlmostEqual(cur.eta_bob, 0.24)
        fut = ent["chapman_future"]
        self.assertAlmostEqual(fut.f_rep, 400e6)
        self.assertAlmostEqual(fut.xi_b, 1e-6)

    def test_loaders_are_disjoint(self):
        self.assertFalse(set(presets.load_presets())
                         & set(presets.load_ent_presets()))


class TestReservedName(unittest.TestCase):
    """"default" belongs to the saved panel state, not to a preset file."""

    def test_default_json_is_rejected(self):
        path = os.path.join(presets.PRESET_DIR, "default.json")
        self.assertFalse(os.path.exists(path), "real default.json must not exist")
        with open(path, "w") as f:
            f.write('{"d_km": 1.0}\n')
        try:
            with self.assertRaises(ValueError):
                presets.load_presets()
            with self.assertRaises(ValueError):
                presets.load_ent_presets()
        finally:
            os.remove(path)

    def test_loaders_work_once_it_is_gone(self):
        self.assertIn("aurea", presets.load_presets())


if __name__ == "__main__":
    unittest.main()
