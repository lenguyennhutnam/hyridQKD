"""Backend tests: CSV parsing + status logic. Stack calls are monkeypatched."""
import os
import shutil
import tempfile
import time
import unittest

import backend as backend_mod
from backend import Backend, Snapshot
import channel as channel_mod

HEADER = ("unix_time,window_s,clicks,sifted,errors,qber,n_sig,m_sig,"
          "n_dec,m_dec,total_sifted,blocks,block_progress,secret_bits\n")


class _FakeEve:
    def __init__(self, order):
        self._order = order
        self._active = False
        self._f = 0.0

    @property
    def active(self):
        return self._active

    @property
    def current_f(self):
        return self._f

    def stop(self):
        self._order.append("eve_stop")
        self._active = False

    def start(self, f):
        self._active = True
        self._f = f


class TestRefresh(unittest.TestCase):
    def setUp(self):
        fd, self.csv = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        self.b = Backend(csv_path=self.csv)
        # Capture the REAL originals before any monkeypatching below, so
        # tearDown always restores genuine functions -- not a lambda left
        # over from an earlier line in this same setUp. Otherwise a later
        # test module run via `discover` would inherit a stale stub.
        self._orig = backend_mod.stackctl.collector_running
        self._orig_apply = (backend_mod.stackctl.stop_all,
                            backend_mod.stackctl.start_stack,
                            backend_mod.stackctl.start_collector,
                            backend_mod.configio.update_hw_sim_configs,
                            backend_mod.configio.update_hw_sim_plain_configs)
        # collector "not running" by default
        backend_mod.stackctl.collector_running = lambda: False
        # no-op stop_all so stop() doesn't try to kill real processes
        backend_mod.stackctl.stop_all = lambda: None

    def tearDown(self):
        backend_mod.stackctl.collector_running = self._orig
        (backend_mod.stackctl.stop_all, backend_mod.stackctl.start_stack,
         backend_mod.stackctl.start_collector,
         backend_mod.configio.update_hw_sim_configs,
         backend_mod.configio.update_hw_sim_plain_configs) = self._orig_apply
        if os.path.exists(self.csv):
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

    def test_apply_stops_eve_first(self):
        # apply() must stop Eve before restarting the stack (PIDs change).
        order = []
        self.b.eve = _FakeEve(order)
        backend_mod.stackctl.stop_all = lambda: order.append("stop_all")
        backend_mod.stackctl.start_stack = lambda *a, **k: order.append("start_stack")
        backend_mod.stackctl.start_collector = lambda *a, **k: order.append("start_collector")
        backend_mod.configio.update_hw_sim_configs = lambda *a, **k: order.append("cfg")
        self.b.apply(channel_mod.EnvParams(), 1000)
        self.assertLess(order.index("eve_stop"), order.index("start_stack"))

    def test_apply_archives_the_old_run_before_wiping_it(self):
        # Order is load-bearing: wiping first destroys the run being archived.
        order = []
        self.b.eve = _FakeEve(order)
        seen = {}

        def fake_archive(csv_path, meta=None, **kw):
            order.append("archive")
            seen["existed"] = os.path.exists(csv_path)
            seen["meta"] = meta
            return "/tmp/fake_run.csv"

        orig_archive = backend_mod.runlog.archive_run
        backend_mod.runlog.archive_run = fake_archive
        backend_mod.stackctl.stop_all = lambda: None
        backend_mod.stackctl.start_stack = lambda *a, **k: None
        backend_mod.stackctl.start_collector = lambda *a, **k: None
        backend_mod.configio.update_hw_sim_configs = lambda *a, **k: None
        try:
            self._write_row(time.time())
            self.b.apply(channel_mod.EnvParams(), 1000)
        finally:
            backend_mod.runlog.archive_run = orig_archive
        self.assertIn("archive", order)
        self.assertTrue(seen["existed"], "archive ran after the CSV was wiped")
        self.assertFalse(os.path.exists(self.csv), "CSV still wiped afterwards")

    def test_run_is_labelled_with_its_own_parameters(self):
        # apply() archives the PREVIOUS run while holding the NEW parameters,
        # so the description must come from the sidecar written when that run
        # started -- otherwise every archived run gets mislabelled with the
        # settings that replaced it.
        self.b.eve = _FakeEve([])
        backend_mod.stackctl.stop_all = lambda: None
        backend_mod.stackctl.start_stack = lambda *a, **k: None
        backend_mod.stackctl.start_collector = lambda *a, **k: None
        backend_mod.configio.update_hw_sim_configs = lambda *a, **k: None
        runs_dir = tempfile.mkdtemp()
        orig_archive = backend_mod.runlog.archive_run
        backend_mod.runlog.archive_run = (
            lambda path, meta=None, **kw: orig_archive(path, meta,
                                                       runs_dir=runs_dir))
        try:
            self.b.apply(channel_mod.EnvParams(mu1=0.29), 1234)  # run A starts
            self._write_row(time.time())                          # run A streams
            self.b.apply(channel_mod.EnvParams(mu1=0.5), 999)     # archives A
            archived = backend_mod.runlog.list_runs(runs_dir)
            self.assertEqual(len(archived), 1)
            meta = backend_mod.runlog.load_run(archived[0]).meta
            self.assertEqual(meta["protocol"], "pm")
            self.assertEqual(meta["n_z_block"], 1234)     # run A's block size
            self.assertAlmostEqual(meta["env"]["mu1"], 0.29)   # run A's mu1
            self.assertIn("l_predict", meta["forecast"])
        finally:
            backend_mod.runlog.archive_run = orig_archive
            shutil.rmtree(runs_dir, ignore_errors=True)
            for stale in (self.csv[:-4] + ".json",):
                if os.path.exists(stale):
                    os.remove(stale)

    def test_refresh_reports_eve_state(self):
        self.b.eve = _FakeEve([])
        self.b.eve._active = True
        self.b.eve._f = 0.3
        backend_mod.stackctl.collector_running = lambda: False
        snap = self.b.refresh()
        self.assertTrue(snap.eve_active)
        self.assertAlmostEqual(snap.eve_f, 0.3)

    def test_apply_ent_uses_plain_path(self):
        order = []
        calls = {}
        self.b.eve = _FakeEve(order)
        backend_mod.stackctl.stop_all = lambda: order.append("stop_all")
        backend_mod.stackctl.start_stack = (
            lambda mode="decoy", **k: calls.setdefault("mode", mode))
        backend_mod.stackctl.start_collector = (
            lambda csv, args: calls.setdefault("collector_args", list(args)))
        backend_mod.configio.update_hw_sim_configs = (
            lambda *a, **k: order.append("decoy_cfg"))
        backend_mod.configio.update_hw_sim_plain_configs = (
            lambda *a, **k: order.append("plain_cfg"))
        d = self.b.apply(channel_mod.EntEnvParams(), 1000, protocol="ent")
        self.assertIn("plain_cfg", order)
        self.assertNotIn("decoy_cfg", order)
        self.assertEqual(calls["mode"], "plain")
        self.assertIn("--protocol", calls["collector_args"])
        self.assertIn("ent", calls["collector_args"])
        self.assertGreaterEqual(d.r_coinc, 0.0)  # EntDerived came back

    def test_apply_pm_default_unchanged(self):
        order = []
        calls = {}
        self.b.eve = _FakeEve(order)
        backend_mod.stackctl.stop_all = lambda: None
        backend_mod.stackctl.start_stack = (
            lambda mode="decoy", **k: calls.setdefault("mode", mode))
        backend_mod.stackctl.start_collector = lambda csv, args: None
        backend_mod.configio.update_hw_sim_configs = (
            lambda *a, **k: order.append("decoy_cfg"))
        backend_mod.configio.update_hw_sim_plain_configs = (
            lambda *a, **k: order.append("plain_cfg"))
        self.b.apply(channel_mod.EnvParams(), 1000)
        self.assertIn("decoy_cfg", order)
        self.assertNotIn("plain_cfg", order)
        self.assertEqual(calls["mode"], "decoy")

    def test_snapshot_reports_protocol(self):
        self.assertEqual(self.b.refresh().protocol, "pm")
        self.b._protocol = "ent"
        self.assertEqual(self.b.refresh().protocol, "ent")


if __name__ == "__main__":
    unittest.main()
