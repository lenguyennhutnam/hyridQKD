"""Panel-state persistence: the "Save as default" file.

Unlike presets/*.json (hand-written, typos must fail loudly), this file is
machine-written, so loading it must NEVER raise -- a corrupt or outdated
default must degrade to "no default" rather than break app startup.
"""
import json
import os
import tempfile
import unittest

import panelstate


def _state():
    return {"protocol": "ent", "block": 1_000_000, "b_max": 5_000_000,
            "fields": {"d_km": 12.5, "alpha": 0.2, "ent_mu": 0.045}}


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)  # start from "no file"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_save_then_load_returns_same_state(self):
        panelstate.save_default(_state(), self.path)
        self.assertEqual(panelstate.load_default(self.path), _state())

    def test_save_returns_the_path(self):
        self.assertEqual(panelstate.save_default(_state(), self.path), self.path)

    def test_save_overwrites(self):
        panelstate.save_default(_state(), self.path)
        second = dict(_state(), protocol="pm")
        panelstate.save_default(second, self.path)
        self.assertEqual(panelstate.load_default(self.path)["protocol"], "pm")


class TestLoadNeverRaises(unittest.TestCase):
    """Every malformed input degrades to None instead of killing startup."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _write(self, text):
        with open(self.path, "w") as f:
            f.write(text)

    def test_missing_file(self):
        os.remove(self.path)
        self.assertIsNone(panelstate.load_default(self.path))

    def test_corrupt_json(self):
        self._write("{not json at all")
        self.assertIsNone(panelstate.load_default(self.path))

    def test_json_that_is_not_a_dict(self):
        self._write("[1, 2, 3]")
        self.assertIsNone(panelstate.load_default(self.path))

    def test_fields_not_a_dict(self):
        self._write(json.dumps({"fields": "nope"}))
        self.assertEqual(panelstate.load_default(self.path)["fields"], {})

    def test_non_numeric_field_dropped(self):
        self._write(json.dumps({"fields": {"d_km": "abc", "alpha": 0.2}}))
        self.assertEqual(panelstate.load_default(self.path)["fields"],
                         {"alpha": 0.2})


class TestForwardCompat(unittest.TestCase):
    """An old default file must still load after the form gains/loses fields."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _write(self, obj):
        with open(self.path, "w") as f:
            json.dump(obj, f)

    def test_unknown_top_level_keys_ignored(self):
        self._write(dict(_state(), some_future_key=42))
        loaded = panelstate.load_default(self.path)
        self.assertNotIn("some_future_key", loaded)
        self.assertEqual(loaded["fields"], _state()["fields"])

    def test_missing_optional_keys_omitted_not_defaulted(self):
        # Caller keeps whatever the form already shows for anything absent.
        self._write({"fields": {"alpha": 0.2}})
        loaded = panelstate.load_default(self.path)
        self.assertEqual(loaded["fields"], {"alpha": 0.2})
        self.assertNotIn("protocol", loaded)
        self.assertNotIn("block", loaded)
        self.assertNotIn("b_max", loaded)

    def test_bad_protocol_dropped(self):
        self._write(dict(_state(), protocol="quantum-telepathy"))
        self.assertNotIn("protocol", panelstate.load_default(self.path))

    def test_bad_block_dropped(self):
        self._write(dict(_state(), block="lots"))
        loaded = panelstate.load_default(self.path)
        self.assertNotIn("block", loaded)
        self.assertEqual(loaded["b_max"], 5_000_000)

    def test_nonpositive_block_dropped(self):
        self._write(dict(_state(), block=0, b_max=-5))
        loaded = panelstate.load_default(self.path)
        self.assertNotIn("block", loaded)
        self.assertNotIn("b_max", loaded)


class TestDefaultPath(unittest.TestCase):
    def test_default_path_is_in_app_dir(self):
        app_dir = os.path.dirname(os.path.abspath(panelstate.__file__))
        self.assertEqual(os.path.dirname(panelstate.DEFAULT_PATH), app_dir)
        self.assertTrue(panelstate.DEFAULT_PATH.endswith("panel_default.json"))

    def test_module_constant_is_honoured_at_call_time(self):
        # Guards the redirect the GUI tests rely on: binding DEFAULT_PATH as a
        # default argument would make patching it silently ineffective and
        # write to the developer's real file.
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        original = panelstate.DEFAULT_PATH
        panelstate.DEFAULT_PATH = path
        try:
            panelstate.save_default(_state())
            self.assertTrue(os.path.exists(path))
            self.assertTrue(panelstate.default_exists())
            self.assertEqual(panelstate.load_default(), _state())
        finally:
            panelstate.DEFAULT_PATH = original
            if os.path.exists(path):
                os.remove(path)

    def test_exists_reports_presence(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            self.assertTrue(panelstate.default_exists(path))
            os.remove(path)
            self.assertFalse(panelstate.default_exists(path))
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
