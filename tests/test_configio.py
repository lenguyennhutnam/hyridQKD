"""Tests for hw_sim config read/write. Uses temp copies, never the real files.

Skipped as a whole when the sibling hw_sim checkout is absent. Every test here
copies the REAL config files into a temp dir first, so there is nothing to test
without them, and they cover only the hardware-stack writer -- none of the
numeric-model code (`policyviz`, `eval_hybrid`, `optmap`) goes through
`configio`. Skipping is what lets a clone of this repo alone report a green
suite instead of nine errors that say nothing about the study.
"""
import json
import os
import shutil
import tempfile
import unittest

import configio


_HAVE_HW_SIM = all(os.path.exists(p) for p in (
    configio.DEFAULT_ALICE, configio.DEFAULT_BOB,
    configio.DEFAULT_ALICE_PLAIN, configio.DEFAULT_BOB_PLAIN))

_NEEDS_HW_SIM = unittest.skipUnless(
    _HAVE_HW_SIM,
    "hw_sim config files not found beside qkd_app; these cover the "
    "hardware-stack config writer only")


@_NEEDS_HW_SIM
class TestConfigIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.alice = os.path.join(self.tmp, "alice.json")
        self.bob = os.path.join(self.tmp, "bob.json")
        shutil.copy(configio.DEFAULT_ALICE, self.alice)
        shutil.copy(configio.DEFAULT_BOB, self.bob)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_update_writes_both_sides_and_preserves_rest(self):
        before = configio.read_config(self.alice)
        configio.update_hw_sim_configs(0.0123, 0.0456, 0.29, 0.103, 0.62,
                                       1.25e-8,
                                       alice_path=self.alice, bob_path=self.bob)
        for p in (self.alice, self.bob):
            cfg = configio.read_config(p)
            self.assertEqual(cfg["backend_config"]["eta"], 0.0123)
            self.assertEqual(cfg["backend_config"]["qberr"], 0.0456)
            self.assertEqual(cfg["backend_config"]["decoy_states"],
                             {"mu1": 0.29, "mu2": 0.103, "p1": 0.62})
            self.assertEqual(cfg["backend_config"]["pulse_distance"], 1.25e-8)
        after = configio.read_config(self.alice)
        self.assertEqual(before["backend_config"]["seed"],
                         after["backend_config"]["seed"])
        self.assertEqual(before["backend_config"]["angles"],
                         after["backend_config"]["angles"])
        self.assertEqual(before["ipc_config"], after["ipc_config"])

    def test_mismatched_seed_detected(self):
        cfg = configio.read_config(self.bob)
        cfg["backend_config"]["seed"] = 999
        with open(self.bob, "w") as f:
            json.dump(cfg, f)
        # Capture state before attempted update
        alice_before = configio.read_config(self.alice)
        bob_before = configio.read_config(self.bob)
        # Attempt update should fail and raise ValueError
        with self.assertRaises(ValueError):
            configio.update_hw_sim_configs(0.1, 0.05, 0.5, 0.1, 0.7, 1e-8,
                                           alice_path=self.alice, bob_path=self.bob)
        # Verify both files are unchanged on disk after the error
        alice_after = configio.read_config(self.alice)
        bob_after = configio.read_config(self.bob)
        self.assertEqual(alice_before, alice_after)
        self.assertEqual(bob_before, bob_after)

    def test_default_paths_exist(self):
        self.assertTrue(os.path.exists(configio.DEFAULT_ALICE))
        self.assertTrue(os.path.exists(configio.DEFAULT_BOB))

    def test_gaussian_qberr_written_when_sigma_positive(self):
        configio.update_hw_sim_configs(0.0123, 0.0456, 0.29, 0.103, 0.62,
                                       1.25e-8, qber_sigma=0.002,
                                       alice_path=self.alice, bob_path=self.bob)
        for p in (self.alice, self.bob):
            q = configio.read_config(p)["backend_config"]["qberr"]
            self.assertEqual(q, {"type": "Gaussian", "mean": 0.0456,
                                 "std_dev": 0.002})

    def test_fixed_qberr_when_sigma_zero(self):
        configio.update_hw_sim_configs(0.0123, 0.0456, 0.29, 0.103, 0.62,
                                       1.25e-8, qber_sigma=0.0,
                                       alice_path=self.alice, bob_path=self.bob)
        for p in (self.alice, self.bob):
            q = configio.read_config(p)["backend_config"]["qberr"]
            self.assertEqual(q, 0.0456)


@_NEEDS_HW_SIM
class TestPlainConfigIO(unittest.TestCase):
    """ENT mode writes the PLAIN (non-decoy) config pair."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.alice = os.path.join(self.tmp, "alice.json")
        self.bob = os.path.join(self.tmp, "bob.json")
        shutil.copy(configio.DEFAULT_ALICE_PLAIN, self.alice)
        shutil.copy(configio.DEFAULT_BOB_PLAIN, self.bob)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_plain_paths_exist(self):
        self.assertTrue(os.path.exists(configio.DEFAULT_ALICE_PLAIN))
        self.assertTrue(os.path.exists(configio.DEFAULT_BOB_PLAIN))

    def test_update_writes_both_sides_no_decoy(self):
        configio.update_hw_sim_plain_configs(0.0032, 0.0151, 1.25e-8,
                                             alice_path=self.alice,
                                             bob_path=self.bob)
        for p in (self.alice, self.bob):
            cfg = configio.read_config(p)["backend_config"]
            self.assertEqual(cfg["eta"], 0.0032)
            self.assertEqual(cfg["qberr"], 0.0151)
            self.assertEqual(cfg["pulse_distance"], 1.25e-8)
            # plain mode MUST have no decoy_states (mismatch with the other
            # side's mode desyncs the preshared-seed RNG -> QBER 50%)
            self.assertNotIn("decoy_states", cfg)

    def test_gaussian_qberr_when_sigma_positive(self):
        configio.update_hw_sim_plain_configs(0.0032, 0.0151, 1.25e-8,
                                             qber_sigma=0.002,
                                             alice_path=self.alice,
                                             bob_path=self.bob)
        for p in (self.alice, self.bob):
            q = configio.read_config(p)["backend_config"]["qberr"]
            self.assertEqual(q, {"type": "Gaussian", "mean": 0.0151,
                                 "std_dev": 0.002})

    def test_mismatched_seed_detected(self):
        cfg = configio.read_config(self.bob)
        cfg["backend_config"]["seed"] = 999
        with open(self.bob, "w") as f:
            json.dump(cfg, f)
        with self.assertRaises(ValueError):
            configio.update_hw_sim_plain_configs(0.1, 0.05, 1e-8,
                                                 alice_path=self.alice,
                                                 bob_path=self.bob)


if __name__ == "__main__":
    unittest.main()
