"""Save-as-default behaviour, driven through a real App with a stub backend.

Needs a display (WSLg provides one); skips cleanly without it. The saved file
is redirected to a temp path so a developer's real panel_default.json is never
touched.
"""
import os
import tempfile
import unittest


def _load_gui():
    try:
        import gui
        import panelstate
    except ImportError:  # matplotlib/tkinter missing
        return None, None
    return gui, panelstate


class TestSaveAsDefault(unittest.TestCase):
    def setUp(self):
        self.gui, self.panelstate = _load_gui()
        if self.gui is None:
            self.skipTest("matplotlib/tkinter not available")
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)                     # start with "no default"
        self._orig_path = self.panelstate.DEFAULT_PATH
        self.panelstate.DEFAULT_PATH = self.path
        try:
            self.app = self.gui.App(_StubBackend())
        except Exception as exc:                 # no display, etc.
            self.panelstate.DEFAULT_PATH = self._orig_path
            self.skipTest("cannot create Tk window: %s" % exc)
        self.app.withdraw()

    def tearDown(self):
        self.panelstate.DEFAULT_PATH = self._orig_path
        if getattr(self, "app", None) is not None:
            self.app.destroy()
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_snapshot_covers_every_form_field(self):
        # A field missing here would be silently lost on save.
        state = self.app._collect_panel_state()
        expected = {f for f, _ in (self.gui.FIELDS_SHARED + self.gui.FIELDS_PM
                                   + self.gui.FIELDS_ENT)}
        self.assertEqual(set(state["fields"]), expected)
        self.assertIn(state["protocol"], ("pm", "ent"))
        self.assertGreater(state["block"], 0)
        self.assertGreater(state["b_max"], 0)

    def test_save_then_restore_round_trip(self):
        # Edit one field per region + block + B_max + mode, save, scramble
        # the form, then reload the default and check everything came back.
        self.app.proto_var.set("ent")
        self.app._set_entry(self.app.entries["d_km"], "42.5")
        self.app._set_entry(self.app.entries["mu1"], "0.31")
        self.app._set_entry(self.app.entries["ent_mu"], "0.02")
        self.app._set_entry(self.app.block_entry, "2000000")
        self.app._set_entry(self.app.bmax_entry, "3000000")
        self.app._save_default()
        self.assertTrue(os.path.exists(self.path))

        self.app.proto_var.set("pm")
        self.app._set_entry(self.app.entries["d_km"], "0.0")
        self.app._set_entry(self.app.entries["mu1"], "0.5")
        self.app._set_entry(self.app.entries["ent_mu"], "0.045")
        self.app._set_entry(self.app.block_entry, "10000000")
        self.app._set_entry(self.app.bmax_entry, "10000000")

        self.app._load_preset(self.panelstate.RESERVED_NAME)
        self.assertEqual(float(self.app.entries["d_km"].get()), 42.5)
        self.assertEqual(float(self.app.entries["mu1"].get()), 0.31)
        self.assertEqual(float(self.app.entries["ent_mu"].get()), 0.02)
        self.assertEqual(int(float(self.app.block_entry.get())), 2_000_000)
        self.assertEqual(int(float(self.app.bmax_entry.get())), 3_000_000)
        self.assertEqual(self.app.proto_var.get(), "ent")

    def test_dropdown_gains_default_only_after_saving(self):
        self.assertNotIn(self.panelstate.RESERVED_NAME,
                         self.app.preset_box.cget("values"))
        self.app._save_default()
        self.assertIn(self.panelstate.RESERVED_NAME,
                      self.app.preset_box.cget("values"))

    def test_bad_entry_does_not_write_a_file(self):
        self.app._set_entry(self.app.entries["alpha"], "not a number")
        self.app._save_default()
        self.assertFalse(os.path.exists(self.path))
        self.assertIn("failed", self.app.error_var.get())

    def test_loading_default_after_file_vanished_is_graceful(self):
        self.app._save_default()
        os.remove(self.path)
        self.app._load_preset(self.panelstate.RESERVED_NAME)   # must not raise
        self.assertIn("no saved default", self.app.error_var.get())
        self.assertNotIn(self.panelstate.RESERVED_NAME,
                         self.app.preset_box.cget("values"))

    def test_restore_ignores_fields_the_form_no_longer_has(self):
        # Simulates loading a default written by an older/newer build.
        self.app._apply_panel_state({"fields": {"d_km": 7.0,
                                                "field_from_the_future": 1.0}})
        self.assertEqual(float(self.app.entries["d_km"].get()), 7.0)


class TestStartupLoadsDefault(unittest.TestCase):
    """The point of the feature: reopening the app restores the last save."""

    def setUp(self):
        self.gui, self.panelstate = _load_gui()
        if self.gui is None:
            self.skipTest("matplotlib/tkinter not available")
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self._orig_path = self.panelstate.DEFAULT_PATH
        self.panelstate.DEFAULT_PATH = self.path
        self.app = None

    def tearDown(self):
        self.panelstate.DEFAULT_PATH = self._orig_path
        if self.app is not None:
            self.app.destroy()
        if os.path.exists(self.path):
            os.remove(self.path)

    def _open(self):
        try:
            self.app = self.gui.App(_StubBackend())
        except Exception as exc:
            self.skipTest("cannot create Tk window: %s" % exc)
        self.app.withdraw()
        return self.app

    def test_saved_values_survive_a_restart(self):
        self.panelstate.save_default(
            {"protocol": "ent", "block": 2_000_000, "b_max": 3_000_000,
             "fields": {"d_km": 42.5, "ent_mu": 0.02}})
        app = self._open()
        self.assertEqual(float(app.entries["d_km"].get()), 42.5)
        self.assertEqual(float(app.entries["ent_mu"].get()), 0.02)
        self.assertEqual(int(float(app.block_entry.get())), 2_000_000)
        self.assertEqual(int(float(app.bmax_entry.get())), 3_000_000)
        self.assertEqual(app.proto_var.get(), "ent")
        self.assertEqual(app.preset_box.get(), self.panelstate.RESERVED_NAME)

    def test_fields_absent_from_the_default_keep_preset_values(self):
        # Only d_km was saved; every other entry must still hold a usable
        # number from the startup presets, never a blank.
        self.panelstate.save_default({"fields": {"d_km": 7.0}})
        app = self._open()
        self.assertEqual(float(app.entries["d_km"].get()), 7.0)
        for field in app.entries:
            float(app.entries[field].get())     # raises if blank/garbage

    def test_corrupt_default_still_opens_the_app(self):
        with open(self.path, "w") as f:
            f.write("{ this is not json")
        app = self._open()
        self.assertEqual(app.proto_var.get(), "pm")       # fell back to aurea
        self.assertEqual(float(app.entries["mu1"].get()), 0.29)


class _StubBackend:
    """Enough surface for App.__init__ and one poll; touches no processes."""

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


if __name__ == "__main__":
    unittest.main()
