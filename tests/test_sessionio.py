"""Writing a GUI session out as a collector-shaped run file."""
import csv
import json
import os
import shutil
import tempfile
import unittest

import runlog
import sessionio
from timeline import Timeline


def _session():
    """PM stretch, a switch, then an ENT stretch."""
    tl = Timeline()
    tl.begin_downtime(100.0, "pm", 0.0522)
    for i in range(3):
        tl.add_sample(110.0 + 2 * i, 5.0, 1000, 400 * i, clicks=2000,
                      window_s=2.0, errors=50, total_sifted=1000 * (i + 1),
                      blocks=0, block_progress=100 * i)
    tl.begin_downtime(120.0, "ent", 0.0994)
    for i in range(2):
        tl.add_sample(135.0 + 2 * i, 9.0, 900, 300 * i, clicks=1800,
                      window_s=2.0, errors=81, total_sifted=5000 + 900 * i,
                      blocks=1, block_progress=50 * i)
    return tl


class TestSanitizeName(unittest.TestCase):
    def test_keeps_safe_characters(self):
        self.assertEqual(sessionio.sanitize_name("run_01.a-B"), "run_01.a-B")

    def test_replaces_unsafe_characters(self):
        self.assertEqual(sessionio.sanitize_name("my run #2"), "my_run__2")

    def test_strips_path_separators(self):
        # A hand-typed name must never escape the runs directory.
        for hostile in ("../../etc/passwd", "a/b", "a\\b", ".."):
            cleaned = sessionio.sanitize_name(hostile)
            self.assertNotIn("/", cleaned)
            self.assertNotIn("\\", cleaned)
            self.assertNotEqual(cleaned, "..")

    def test_empty_name_is_rejected(self):
        for empty in ("", "   ", "///"):
            with self.assertRaises(ValueError):
                sessionio.sanitize_name(empty)

    def test_strips_a_csv_suffix_so_it_is_not_doubled(self):
        self.assertEqual(sessionio.sanitize_name("demo.csv"), "demo")


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _export(self, tl=None, name="demo", extra=None):
        # NOTE: deliberately `is None`, not `tl or _session()` -- Timeline
        # defines __len__, so a fresh (empty) Timeline() is falsy and `or`
        # would silently swap in _session(), defeating
        # test_empty_timeline_is_refused below.
        if tl is None:
            tl = _session()
        return sessionio.export_session(tl, name,
                                        extra_meta=extra, runs_dir=self.tmp)

    def test_writes_both_files(self):
        csv_path, json_path = self._export()
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(csv_path.endswith("demo.csv"))
        self.assertTrue(json_path.endswith("demo.json"))

    def test_csv_has_the_collector_columns_plus_protocol(self):
        csv_path, _ = self._export()
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(rows[0]), list(sessionio.CSV_COLUMNS))
        self.assertEqual(rows[0]["protocol"], "pm")
        self.assertEqual(rows[-1]["protocol"], "ent")

    def test_secret_bits_column_is_the_session_cumulative_value(self):
        # Unlike live.csv (per collector run), this file describes a whole
        # session, so the column must never step backwards at a switch.
        csv_path, _ = self._export()
        with open(csv_path, newline="") as f:
            secrets = [int(r["secret_bits"]) for r in csv.DictReader(f)]
        self.assertEqual(secrets, sorted(secrets))
        self.assertEqual(secrets[-1], _session().total_secret)

    def test_sidecar_records_segments_and_gaps(self):
        _, json_path = self._export()
        with open(json_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["kind"], "session")
        self.assertEqual([s["protocol"] for s in meta["segments"]],
                         ["pm", "ent"])
        self.assertAlmostEqual(meta["segments"][0]["death_qber"], 0.0522)
        self.assertEqual(meta["segments"][1]["i0"], 3)
        self.assertTrue(meta["gaps"][-1]["is_switch"])
        self.assertEqual(meta["secret_by_protocol"],
                         _session().secret_by_protocol)

    def test_sidecar_declares_the_two_honesty_flags(self):
        # Readers must not mistake these for collector-measured values.
        _, json_path = self._export()
        with open(json_path) as f:
            meta = json.load(f)
        self.assertTrue(meta["secret_bits_are_cumulative"])
        self.assertTrue(meta["per_intensity_columns_are_zero"])

    def test_extra_meta_is_merged(self):
        _, json_path = self._export(extra={"b_max": 12345})
        with open(json_path) as f:
            self.assertEqual(json.load(f)["b_max"], 12345)

    def test_export_is_readable_by_runlog(self):
        # The whole point: the exported file is a normal run file.
        csv_path, _ = self._export()
        run = runlog.load_run(csv_path)
        self.assertEqual(len(run.t), 5)
        self.assertEqual(int(run.secret_bits[-1]), _session().total_secret)
        self.assertEqual(run.meta["kind"], "session")

    def test_empty_timeline_is_refused(self):
        with self.assertRaises(ValueError):
            self._export(tl=Timeline())

    def test_creates_the_runs_directory(self):
        nested = os.path.join(self.tmp, "deep", "runs")
        csv_path, _ = sessionio.export_session(_session(), "demo",
                                               runs_dir=nested)
        self.assertTrue(os.path.exists(csv_path))

    def test_session_paths_matches_what_export_writes(self):
        expected = sessionio.session_paths("demo", runs_dir=self.tmp)
        self.assertEqual(self._export(), expected)


if __name__ == "__main__":
    unittest.main()
