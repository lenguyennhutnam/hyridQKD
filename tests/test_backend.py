"""Backend tests: CSV parsing + status logic. Stack calls are monkeypatched."""
import os
import tempfile
import time
import unittest

import backend as backend_mod
from backend import Backend, Snapshot

HEADER = ("unix_time,window_s,clicks,sifted,errors,qber,n_sig,m_sig,"
          "n_dec,m_dec,total_sifted,blocks,block_progress,secret_bits\n")


class TestRefresh(unittest.TestCase):
    def setUp(self):
        fd, self.csv = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        self.b = Backend(csv_path=self.csv)
        # collector "not running" by default
        self._orig = backend_mod.stackctl.collector_running
        backend_mod.stackctl.collector_running = lambda: False
        # no-op stop_all so stop() doesn't try to kill real processes
        self._orig_stop_all = backend_mod.stackctl.stop_all
        backend_mod.stackctl.stop_all = lambda: None

    def tearDown(self):
        backend_mod.stackctl.collector_running = self._orig
        backend_mod.stackctl.stop_all = self._orig_stop_all
        os.remove(self.csv)

    def _write_row(self, t, qber="0.0100"):
        with open(self.csv, "a") as f:
            if f.tell() == 0:
                f.write(HEADER)
            f.write("%f,2.0,5000000,2500000,25000,%s,1700000,17000,800000,8000,"
                    "2500000,1,2500000,0\n" % (t, qber))

    def test_down_when_nothing(self):
        self.assertEqual(self.b.refresh().status, "down")

    def test_starting_when_running_but_no_rows(self):
        backend_mod.stackctl.collector_running = lambda: True
        self.assertEqual(self.b.refresh().status, "starting")

    def test_streaming_parses_last_row(self):
        backend_mod.stackctl.collector_running = lambda: True
        self._write_row(time.time() - 100, qber="0.0100")
        self._write_row(time.time(), qber="0.0200")
        snap = self.b.refresh()
        self.assertEqual(snap.status, "streaming")
        self.assertAlmostEqual(snap.qber, 0.02)
        self.assertEqual(snap.total_sifted, 2500000)
        self.assertEqual(snap.blocks, 1)

    def test_error_when_rows_stale_but_running(self):
        backend_mod.stackctl.collector_running = lambda: True
        self._write_row(time.time() - 60)
        self.assertEqual(self.b.refresh().status, "error")

    def test_error_when_collector_died_after_rows(self):
        # Scenario: we started the collector and it died unexpectedly
        # (not a deliberate Stop) -> should still be "error".
        self.b._started = True
        self._write_row(time.time())
        self.assertEqual(self.b.refresh().status, "error")

    def test_down_after_deliberate_stop(self):
        # Scenario: user presses Stop. Rows from the run are still on disk
        # (only apply() clears the CSV), but this was a clean, deliberate
        # shutdown -> should read as "down", not "error".
        self._write_row(time.time())
        self.b._started = True
        self.b.stop()
        self.assertEqual(self.b.refresh().status, "down")


if __name__ == "__main__":
    unittest.main()
