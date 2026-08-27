"""The policy visualiser's saved configuration.

Written against the three ways this file could quietly ruin a session rather
than against the happy path:

* an EDITED operating point could be written back as the bare preset name,
  which loads without complaint and silently discards the edit -- the exact
  failure the Save button exists to fix;
* an UNEDITED one could be written back as a frozen copy of the preset, which
  also loads without complaint and silently stops tracking presets/*.json;
* a corrupt file could raise, which would stop the window opening at all.

Like panelstate (gui.py's equivalent) and unlike presets.py, loading must
NEVER raise: this file is machine-written and read during startup.
"""
import json
import os
import tempfile
import unittest

import channel
import panelstate
import vizstate


def _state(**kw):
    """A complete saved state, with one mode edited and one not."""
    base = {"fields": {"lam": 2.5, "horizon": 3600, "d_km": 55.0},
            "policies": ["always-pm", "greedy", "oracle"],
            "theta": 0.045, "ba_theta": 0.05, "ba_reserve": 0.25,
            "pm_preset": "aurea", "pm_env": None,
            "ent_preset": "ent_advanced",
            "ent_env": channel.EntEnvParams(mu=0.031),
            "hidden_traces": ["always-ent"], "speed": 20}
    base.update(kw)
    return base


class _TempFile(unittest.TestCase):
    """Every test writes to a temp path, never the developer's real file."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)          # start from "nothing saved"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _write(self, text):
        with open(self.path, "w") as f:
            f.write(text if isinstance(text, str) else json.dumps(text))


class TestRoundTrip(_TempFile):
    def test_save_then_load_returns_the_same_state(self):
        vizstate.save_state(_state(), self.path)
        self.assertEqual(vizstate.load_state(self.path), _state())

    def test_save_returns_the_path(self):
        self.assertEqual(vizstate.save_state(_state(), self.path), self.path)

    def test_save_overwrites(self):
        vizstate.save_state(_state(), self.path)
        vizstate.save_state(_state(theta=0.02), self.path)
        self.assertEqual(vizstate.load_state(self.path)["theta"], 0.02)

    def test_an_edited_operating_point_keeps_its_numbers(self):
        """THE user-facing point of the whole file.

        Saving the preset NAME instead of the dataclass would round-trip
        cleanly and lose the edit -- which is precisely the complaint ("tweak
        one number and there is nowhere to keep it").
        """
        vizstate.save_state(_state(ent_env=channel.EntEnvParams(mu=0.031)),
                            self.path)
        loaded = vizstate.load_state(self.path)["ent_env"]
        self.assertIsInstance(loaded, channel.EntEnvParams)
        self.assertAlmostEqual(loaded.mu, 0.031)

    def test_an_unedited_operating_point_stays_none(self):
        """None is not "missing": it means "run presets/aurea.json as it
        ships". Freezing a copy of the preset here would keep working and
        quietly ignore every later edit to that preset file."""
        vizstate.save_state(_state(pm_env=None), self.path)
        loaded = vizstate.load_state(self.path)
        self.assertIn("pm_env", loaded)
        self.assertIsNone(loaded["pm_env"])

    def test_a_pm_point_round_trips_as_a_pm_dataclass(self):
        vizstate.save_state(_state(pm_env=channel.EnvParams(mu1=0.31)),
                            self.path)
        loaded = vizstate.load_state(self.path)["pm_env"]
        self.assertIsInstance(loaded, channel.EnvParams)
        self.assertAlmostEqual(loaded.mu1, 0.31)

    def test_the_two_modes_do_not_swap(self):
        vizstate.save_state(_state(pm_env=channel.EnvParams(mu1=0.31),
                                   ent_env=channel.EntEnvParams(mu=0.031)),
                            self.path)
        loaded = vizstate.load_state(self.path)
        self.assertIsInstance(loaded["pm_env"], channel.EnvParams)
        self.assertIsInstance(loaded["ent_env"], channel.EntEnvParams)


class TestTheEnvTriState(_TempFile):
    """absent / null / dict are three different instructions, not two."""

    def test_absent_is_not_none(self):
        # Absent means "the file says nothing about this mode; leave the form
        # alone". Null means "this mode is on its preset, unedited". Collapse
        # them and an old file silently resets an operating point.
        self._write({"fields": {}, "pm_env": None})
        loaded = vizstate.load_state(self.path)
        self.assertIn("pm_env", loaded)
        self.assertNotIn("ent_env", loaded)

    def test_a_dict_becomes_a_dataclass(self):
        self._write({"fields": {}, "ent_env": {"mu": 0.02, "protocol": "ent"}})
        env = vizstate.load_state(self.path)["ent_env"]
        self.assertAlmostEqual(env.mu, 0.02)

    def test_the_protocol_tag_is_written(self):
        self.assertEqual(
            vizstate.env_to_dict(channel.EntEnvParams())["protocol"], "ent")
        self.assertEqual(
            vizstate.env_to_dict(channel.EnvParams())["protocol"], "pm")

    def test_a_point_tagged_as_the_other_mode_is_refused(self):
        # Filed under pm_env, tagged ent: one of the two is wrong and there is
        # no way to tell which, so neither is trusted.
        self.assertIsNone(vizstate.env_from_dict("pm", {"protocol": "ent"}))

    def test_an_unreadable_field_discards_the_whole_point(self):
        # NOT just that field: substituting the dataclass default for a number
        # the user saved would run different physics under the label they
        # chose. Falling back to the named preset is at least honest -- the
        # mode button then says "aurea", not "custom".
        self.assertIsNone(
            vizstate.env_from_dict("ent", {"mu": "lots", "protocol": "ent"}))

    def test_a_field_this_version_no_longer_has_is_dropped(self):
        env = vizstate.env_from_dict(
            "ent", {"mu": 0.02, "protocol": "ent", "removed_in_v9": 1.0})
        self.assertAlmostEqual(env.mu, 0.02)

    def test_an_unusable_point_leaves_the_key_out(self):
        self._write({"fields": {}, "ent_env": {"mu": "lots"}})
        self.assertNotIn("ent_env", vizstate.load_state(self.path))


class TestLoadNeverRaises(_TempFile):
    """Every malformed input degrades to "nothing saved" instead of killing
    startup."""

    def test_missing_file(self):
        self.assertIsNone(vizstate.load_state(self.path))

    def test_corrupt_json(self):
        self._write("{not json at all")
        self.assertIsNone(vizstate.load_state(self.path))

    def test_truncated_json(self):
        self._write('{"fields": {"lam": 2.5}')
        self.assertIsNone(vizstate.load_state(self.path))

    def test_json_that_is_not_a_dict(self):
        self._write("[1, 2, 3]")
        self.assertIsNone(vizstate.load_state(self.path))

    def test_an_unreadable_path(self):
        # A directory where the file should be: open() raises IsADirectoryError
        # (an OSError), which must be caught like any other unreadable file.
        d = tempfile.mkdtemp()
        try:
            self.assertIsNone(vizstate.load_state(d))
        finally:
            os.rmdir(d)

    def test_fields_not_a_dict(self):
        self._write({"fields": "nope"})
        self.assertEqual(vizstate.load_state(self.path)["fields"], {})

    def test_non_numeric_field_dropped(self):
        self._write({"fields": {"lam": "abc", "horizon": 300}})
        self.assertEqual(vizstate.load_state(self.path)["fields"],
                         {"horizon": 300})

    def test_a_boolean_is_not_a_number(self):
        # bool is an int subclass; a stray true would restore lam = 1.
        self._write({"fields": {"lam": True}})
        self.assertEqual(vizstate.load_state(self.path)["fields"], {})

    def test_infinities_and_nans_are_dropped(self):
        self._write('{"fields": {"lam": Infinity, "alpha": NaN}, '
                    '"theta": Infinity}')
        loaded = vizstate.load_state(self.path)
        self.assertEqual(loaded["fields"], {})
        self.assertNotIn("theta", loaded)


class TestForwardCompat(_TempFile):
    """A file written by an older (or newer) version still loads."""

    def test_unknown_top_level_keys_ignored(self):
        self._write(dict(_state(pm_env=None, ent_env=None),
                         some_future_control=42))
        self.assertNotIn("some_future_control",
                         vizstate.load_state(self.path))

    def test_missing_keys_are_omitted_not_defaulted(self):
        # Absent means "leave that control as it is", which is what lets a
        # file written before a control existed still load.
        self._write({"fields": {"lam": 2.5}})
        loaded = vizstate.load_state(self.path)
        self.assertEqual(loaded["fields"], {"lam": 2.5})
        for key in ("policies", "theta", "pm_preset", "hidden_traces",
                    "speed"):
            self.assertNotIn(key, loaded)

    def test_an_unknown_policy_spec_is_kept_and_simply_never_matches(self):
        # Checking names against this version's checkbox list here would make
        # the file version-locked; a stem nothing matches is inert.
        self._write({"fields": {}, "policies": ["always-pm", "warp-drive"]})
        self.assertEqual(vizstate.load_state(self.path)["policies"],
                         ["always-pm", "warp-drive"])

    def test_an_empty_policy_selection_is_dropped(self):
        # It could not be applied, so restoring it would produce a form whose
        # Apply button only ever errors.
        self._write({"fields": {}, "policies": []})
        self.assertNotIn("policies", vizstate.load_state(self.path))

    def test_non_string_policies_are_filtered_out(self):
        self._write({"fields": {}, "policies": ["always-pm", 7, "  "]})
        self.assertEqual(vizstate.load_state(self.path)["policies"],
                         ["always-pm"])

    def test_an_empty_preset_name_is_dropped(self):
        self._write({"fields": {}, "pm_preset": "   ", "ent_preset": 7})
        loaded = vizstate.load_state(self.path)
        self.assertNotIn("pm_preset", loaded)
        self.assertNotIn("ent_preset", loaded)

    def test_hidden_traces_are_sorted_and_deduplicated(self):
        self._write({"fields": {}, "hidden_traces": ["b", "a", "b", 3]})
        self.assertEqual(vizstate.load_state(self.path)["hidden_traces"],
                         ["a", "b"])

    def test_a_nonpositive_speed_is_dropped(self):
        self._write({"fields": {}, "speed": 0})
        self.assertNotIn("speed", vizstate.load_state(self.path))

    def test_integers_stay_integers(self):
        # The horizon box should read "3600", not "3600.0".
        self._write({"fields": {"horizon": 3600}})
        self.assertIsInstance(
            vizstate.load_state(self.path)["fields"]["horizon"], int)


class TestTheDefaultPath(unittest.TestCase):
    def test_it_is_in_the_app_directory(self):
        app_dir = os.path.dirname(os.path.abspath(vizstate.__file__))
        self.assertEqual(os.path.dirname(vizstate.DEFAULT_PATH), app_dir)
        self.assertTrue(
            vizstate.DEFAULT_PATH.endswith("policyviz_default.json"))

    def test_it_is_not_the_other_panels_file(self):
        # panel_default.json belongs to gui.py (panelstate.py) and holds a
        # different form entirely; sharing it would have one panel's Save
        # rewrite the other panel's startup state.
        self.assertNotEqual(vizstate.DEFAULT_PATH, panelstate.DEFAULT_PATH)
        self.assertNotIn("panel_default.json", vizstate.DEFAULT_PATH)

    def test_the_module_constant_is_honoured_at_call_time(self):
        # Guards the redirect every window test relies on: binding
        # DEFAULT_PATH as a default argument value would make patching it
        # silently ineffective and write to the developer's real file.
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)
        original = vizstate.DEFAULT_PATH
        vizstate.DEFAULT_PATH = path
        try:
            self.assertFalse(vizstate.state_exists())
            vizstate.save_state(_state())
            self.assertTrue(os.path.exists(path))
            self.assertTrue(vizstate.state_exists())
            self.assertEqual(vizstate.load_state(), _state())
        finally:
            vizstate.DEFAULT_PATH = original
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
