"""The Export session button: writes a run file and renders it."""
import os
import shutil
import tempfile
import unittest


def _load():
    try:
        import gui
        import panelstate
        import runlog
    except ImportError:
        return None, None, None
    return gui, panelstate, runlog


class _StubBackend:
    class _Eve:
        active = False
        current_f = 0.0

        def episodes(self):
            return []

    def __init__(self):
        self.eve = self._Eve()

    def predict(self, env, n_z_block=10_000_000, protocol="pm"):
        import channel
        if protocol == "ent":
            return channel.derive_ent(env, n_z_block)
        return channel.derive(env, n_z_block)

    def refresh(self):
        import backend
        return backend.Snapshot()

    def stop(self):
        pass


class TestExportSession(unittest.TestCase):
    def setUp(self):
        self.gui, self.panelstate, self.runlog = _load()
        if self.gui is None:
            self.skipTest("matplotlib/tkinter not available")
        self.tmp = tempfile.mkdtemp()
        fd, self.default_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.default_path)
        self._orig_default = self.panelstate.DEFAULT_PATH
        self.panelstate.DEFAULT_PATH = self.default_path
        try:
            self.app = self.gui.App(_StubBackend())
        except Exception as exc:
            self.panelstate.DEFAULT_PATH = self._orig_default
            self.skipTest("cannot create Tk window: %s" % exc)
        self.app.withdraw()
        # Redirect the export destinations at the App, not at the modules, so
        # nothing can write into the developer's real logs/ directory.
        self.app._export_runs_dir = os.path.join(self.tmp, "runs")
        self.app._export_figs_dir = os.path.join(self.tmp, "figures")

    def tearDown(self):
        self.panelstate.DEFAULT_PATH = self._orig_default
        if getattr(self, "app", None) is not None:
            self.app.update_idletasks()
            self.app.destroy()
        shutil.rmtree(self.tmp, ignore_errors=True)
        if os.path.exists(self.default_path):
            os.remove(self.default_path)

    def _session(self):
        tl = self.app.timeline
        tl.begin_downtime(100.0, "pm", 0.0522)
        for i in range(3):
            tl.add_sample(110.0 + 2 * i, 5.0, 1000, 400 * i, clicks=2000,
                          window_s=2.0, errors=50, total_sifted=1000 * (i + 1))
        tl.begin_downtime(120.0, "ent", 0.0994)
        for i in range(3):
            tl.add_sample(135.0 + 2 * i, 9.0, 900, 300 * i, clicks=1800,
                          window_s=2.0, errors=81, total_sifted=5000)

    def test_export_writes_csv_json_and_png(self):
        self._session()
        paths = self.app._export_session(name="demo", confirm_overwrite=False)
        self.assertIsNotNone(paths)
        for path in paths:
            self.assertTrue(os.path.exists(path), path)
        self.assertTrue(any(p.endswith(".png") for p in paths))

    def test_exported_file_reads_back_with_its_segments(self):
        self._session()
        csv_path = self.app._export_session(name="demo",
                                            confirm_overwrite=False)[0]
        run = self.runlog.load_run(csv_path)
        self.assertEqual(len(run.t), 6)
        self.assertEqual([s["protocol"] for s in run.segments], ["pm", "ent"])
        self.assertTrue(run.gaps[-1]["is_switch"])

    def test_panel_state_travels_in_the_sidecar(self):
        self._session()
        csv_path = self.app._export_session(name="demo",
                                            confirm_overwrite=False)[0]
        meta = self.runlog.load_run(csv_path).meta
        self.assertIn("panel", meta)
        self.assertIn("fields", meta["panel"])
        self.assertGreater(meta["b_max"], 0)

    def test_empty_session_writes_nothing(self):
        result = self.app._export_session(name="demo", confirm_overwrite=False)
        self.assertIsNone(result)
        self.assertFalse(os.path.exists(self.app._export_runs_dir))
        self.assertIn("nothing to export", self.app.error_var.get().lower())

    def test_unsafe_name_cannot_escape_the_runs_directory(self):
        self._session()
        paths = self.app._export_session(name="../../pwned",
                                         confirm_overwrite=False)
        for path in paths:
            self.assertTrue(
                os.path.realpath(path).startswith(
                    os.path.realpath(self.tmp)), path)

    def test_blank_name_is_reported_not_crashed(self):
        self._session()
        self.assertIsNone(self.app._export_session(name="   ",
                                                   confirm_overwrite=False))
        self.assertIn("name", self.app.error_var.get().lower())

    def test_reserved_sidecar_keys_survive_the_panel_metadata(self):
        """extra_meta is merged AFTER sessionio's own reserved keys, so the
        panel payload must never carry a name sessionio owns."""
        self._session()
        csv_path = self.app._export_session(name="demo",
                                            confirm_overwrite=False)[0]
        meta = self.runlog.load_run(csv_path).meta
        self.assertEqual(meta["kind"], "session")
        self.assertIs(meta["secret_bits_are_cumulative"], True)
        self.assertIs(meta["per_intensity_columns_are_zero"], True)


if __name__ == "__main__":
    unittest.main()
