"""Tests for evectl.py — the duty-cycle signal controller.

A fake signal_sender records (timestamp, signal) so we can assert the duty
cycle without any real process. SIGUSR1 = start jam, SIGUSR2 = stop jam.
"""
import os
import signal
import tempfile
import threading
import time
import unittest

from evectl import EveController, Episode


class Recorder:
    def __init__(self):
        self.calls = []          # list of (t, sig)
        self.lock = threading.Lock()

    def __call__(self, sig):
        with self.lock:
            self.calls.append((time.monotonic(), sig))

    def sigs(self):
        with self.lock:
            return [s for _, s in self.calls]


class TestEveController(unittest.TestCase):
    def setUp(self):
        self.rec = Recorder()
        fd, self.csv = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        os.remove(self.csv)  # controller creates it
        self.c = EveController(signal_sender=self.rec, period_s=0.05,
                               csv_path=self.csv)

    def tearDown(self):
        self.c.stop()
        if os.path.exists(self.csv):
            os.remove(self.csv)

    def test_full_jam_sends_only_usr1(self):
        self.c.start(1.0)
        time.sleep(0.25)
        self.c.stop()
        sigs = self.rec.sigs()
        # While jamming at f=1 the on-phase covers the whole period, so no
        # USR2 is sent until stop(); the final signal is always USR2.
        self.assertIn(signal.SIGUSR1, sigs)
        self.assertEqual(sigs[-1], signal.SIGUSR2)
        # No USR2 in the middle (all USR1 until the final stop).
        self.assertEqual(sigs[:-1].count(signal.SIGUSR2), 0)

    def test_half_jam_mixes_usr1_and_usr2(self):
        self.c.start(0.5)
        time.sleep(0.4)
        self.c.stop()
        sigs = self.rec.sigs()
        self.assertGreater(sigs.count(signal.SIGUSR1), 1)
        self.assertGreater(sigs.count(signal.SIGUSR2), 1)

    def test_stop_always_ends_unjammed(self):
        self.c.start(0.7)
        time.sleep(0.1)
        self.c.stop()
        self.assertEqual(self.rec.sigs()[-1], signal.SIGUSR2)
        self.assertFalse(self.c.active)

    def test_stop_is_idempotent(self):
        self.c.stop()
        self.c.stop()  # must not raise

    def test_episode_logged_on_close(self):
        self.c.start(0.3)
        time.sleep(0.1)
        self.c.stop()
        eps = self.c.episodes()
        self.assertEqual(len(eps), 1)
        self.assertGreater(eps[0].t_end, eps[0].t_start)
        self.assertAlmostEqual(eps[0].f, 0.3)
        with open(self.csv) as f:
            lines = f.read().strip().splitlines()
        self.assertEqual(lines[0], "t_start,t_end,f")
        self.assertEqual(len(lines), 2)

    def test_open_episode_has_zero_tend(self):
        self.c.start(0.4)
        time.sleep(0.05)
        eps = self.c.episodes()
        self.assertEqual(len(eps), 1)
        self.assertEqual(eps[0].t_end, 0.0)
        self.c.stop()

    def test_bad_fraction_raises(self):
        with self.assertRaises(ValueError):
            self.c.start(1.5)


if __name__ == "__main__":
    unittest.main()
