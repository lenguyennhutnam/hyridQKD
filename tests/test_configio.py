"""Tests for hw_sim config read/write. Uses temp copies, never the real files."""
import json
import os
import shutil
import tempfile
import unittest

import configio


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
                                       alice_path=self.alice, bob_path=self.bob)
        for p in (self.alice, self.bob):
            cfg = configio.read_config(p)
            self.assertEqual(cfg["backend_config"]["eta"], 0.0123)
            self.assertEqual(cfg["backend_config"]["qberr"], 0.0456)
            self.assertEqual(cfg["backend_config"]["decoy_states"],
                             {"mu1": 0.29, "mu2": 0.103, "p1": 0.62})
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
        with self.assertRaises(ValueError):
            configio.update_hw_sim_configs(0.1, 0.05, 0.5, 0.1, 0.7,
                                           alice_path=self.alice, bob_path=self.bob)

    def test_default_paths_exist(self):
        self.assertTrue(os.path.exists(configio.DEFAULT_ALICE))
        self.assertTrue(os.path.exists(configio.DEFAULT_BOB))


if __name__ == "__main__":
    unittest.main()
