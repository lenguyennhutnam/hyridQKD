"""Integration tests for stack/collector lifecycle. Skipped if bin/ is missing."""
import os
import subprocess
import time
import unittest

import stackctl

HAVE_BIN = os.path.exists(os.path.join(os.path.dirname(stackctl.__file__),
                                       "bin", "simulator"))


def _pgrep(name):
    return subprocess.run(["pgrep", "-x", name],
                          stdout=subprocess.DEVNULL).returncode == 0


def _pgrep_f(pattern):
    return subprocess.run(["pgrep", "-f", pattern],
                          stdout=subprocess.DEVNULL).returncode == 0


@unittest.skipUnless(HAVE_BIN, "stack binaries not built (bash build.sh)")
class TestStackLifecycle(unittest.TestCase):
    def tearDown(self):
        stackctl.stop_all()

    def test_start_and_stop_stack(self):
        stackctl.start_stack()
        time.sleep(1)
        self.assertTrue(stackctl.stack_running())
        stackctl.stop_all()
        time.sleep(1)
        for name in ("simulator", "gc_alice", "gc_bob", "qber_bob"):
            self.assertFalse(_pgrep(name), name + " survived stop_all")

    def test_collector_lifecycle(self):
        stackctl.start_stack()
        time.sleep(1)
        csv = os.path.join(stackctl.LOG_DIR, "test_live.csv")
        if os.path.exists(csv):
            os.remove(csv)
        pid = stackctl.start_collector(csv, ["--interval", "1"])
        self.assertGreater(pid, 0)
        time.sleep(4)
        self.assertTrue(stackctl.collector_running())
        self.assertTrue(os.path.exists(csv), "collector produced no CSV")
        stackctl.stop_all()
        time.sleep(1)
        self.assertFalse(stackctl.collector_running())
        self.assertFalse(_pgrep_f("collector.py --csv"),
                         "collector process survived stop_all")

    def test_stop_all_idempotent_when_nothing_runs(self):
        stackctl.stop_all()
        stackctl.stop_all()  # must not raise


if __name__ == "__main__":
    unittest.main()
