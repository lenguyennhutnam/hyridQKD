"""Reading and archiving collector run logs.

Log files are machine-written and read by a plotting command, so a damaged
line must be skipped rather than abort the whole figure (same policy as
panelstate.py, the opposite of presets.py).
"""
import json
import os
import shutil
import tempfile
import unittest

import runlog

HEADER = ("unix_time,window_s,clicks,sifted,errors,qber,n_sig,m_sig,"
          "n_dec,m_dec,total_sifted,blocks,block_progress,secret_bits\n")


def _row(t, qber=0.01, sifted=1000, clicks=2000, total=1000, blocks=0,
         progress=1000, secret=0):
    return ("%.3f,2.000,%d,%d,%d,%.6f,%d,%d,0,0,%d,%d,%d,%d\n"
            % (t, clicks, sifted, int(sifted * qber), qber, sifted,
               int(sifted * qber), total, blocks, progress, secret))


class _TmpDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv = os.path.join(self.tmp, "live.csv")
        self.runs = os.path.join(self.tmp, "runs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rows, path=None):
        with open(path or self.csv, "w") as f:
            f.write(HEADER)
            for r in rows:
                f.write(r)


class TestArchive(_TmpDirCase):
    def test_moves_csv_and_writes_metadata(self):
        self._write([_row(1000.0), _row(1002.0)])
        meta = {"protocol": "ent", "n_z_block": 1_000_000}
        dest = runlog.archive_run(self.csv, meta, runs_dir=self.runs)
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.exists(dest))
        self.assertFalse(os.path.exists(self.csv), "source must be moved")
        with open(dest[:-4] + ".json") as f:
            saved = json.load(f)
        self.assertEqual(saved["protocol"], "ent")
        self.assertEqual(saved["n_z_block"], 1_000_000)
        self.assertIn("archived_at", saved)

    def test_missing_file_archives_nothing(self):
        self.assertIsNone(runlog.archive_run(self.csv, {}, runs_dir=self.runs))
        self.assertFalse(os.path.isdir(self.runs), "must not create clutter")

    def test_header_only_archives_nothing(self):
        self._write([])
        self.assertIsNone(runlog.archive_run(self.csv, {}, runs_dir=self.runs))
        self.assertFalse(os.path.exists(self.csv), "empty log is discarded")

    def test_two_archives_do_not_collide(self):
        self._write([_row(1000.0)])
        first = runlog.archive_run(self.csv, {}, runs_dir=self.runs)
        self._write([_row(2000.0)])
        second = runlog.archive_run(self.csv, {}, runs_dir=self.runs)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first))
        self.assertTrue(os.path.exists(second))

    def test_metadata_optional(self):
        self._write([_row(1000.0)])
        dest = runlog.archive_run(self.csv, None, runs_dir=self.runs)
        with open(dest[:-4] + ".json") as f:
            self.assertIn("archived_at", json.load(f))

    def test_live_sidecar_labels_the_run_it_belongs_to(self):
        # The run carries the parameters recorded when IT started, not the
        # ones being applied at archive time.
        self._write([_row(1000.0)])
        runlog.write_live_meta(self.csv, {"protocol": "pm", "mu1": 0.29})
        dest = runlog.archive_run(self.csv, runs_dir=self.runs)
        with open(dest[:-4] + ".json") as f:
            saved = json.load(f)
        self.assertEqual(saved["mu1"], 0.29)
        self.assertIn("started_at", saved)

    def test_explicit_meta_overrides_the_live_sidecar(self):
        self._write([_row(1000.0)])
        runlog.write_live_meta(self.csv, {"protocol": "pm"})
        dest = runlog.archive_run(self.csv, {"protocol": "ent"},
                                  runs_dir=self.runs)
        with open(dest[:-4] + ".json") as f:
            self.assertEqual(json.load(f)["protocol"], "ent")

    def test_live_sidecar_is_consumed(self):
        # Left behind, it would mislabel the NEXT run archived from live.csv.
        self._write([_row(1000.0)])
        runlog.write_live_meta(self.csv, {"protocol": "ent"})
        runlog.archive_run(self.csv, runs_dir=self.runs)
        self.assertFalse(os.path.exists(self.csv[:-4] + ".json"))

    def test_sidecar_dropped_when_run_had_no_samples(self):
        self._write([])
        runlog.write_live_meta(self.csv, {"protocol": "ent"})
        self.assertIsNone(runlog.archive_run(self.csv, runs_dir=self.runs))
        self.assertFalse(os.path.exists(self.csv[:-4] + ".json"))


class TestLoadRun(_TmpDirCase):
    def test_parses_columns_and_relative_time(self):
        self._write([_row(1000.0, qber=0.01, sifted=500),
                     _row(1002.0, qber=0.02, sifted=700),
                     _row(1004.0, qber=0.03, sifted=900)])
        run = runlog.load_run(self.csv)
        self.assertEqual(len(run.t), 3)
        self.assertAlmostEqual(run.t[0], 0.0)
        self.assertAlmostEqual(run.t[-1], 4.0)
        self.assertAlmostEqual(run.qber[1], 0.02)
        self.assertEqual(int(run.sifted[2]), 900)
        self.assertAlmostEqual(run.unix_t[0], 1000.0)

    def test_empty_log_is_not_an_error(self):
        self._write([])
        run = runlog.load_run(self.csv)
        self.assertEqual(len(run.t), 0)
        self.assertEqual(run.duration_s, 0.0)

    def test_missing_file_is_not_an_error(self):
        run = runlog.load_run(os.path.join(self.tmp, "nope.csv"))
        self.assertEqual(len(run.t), 0)

    def test_damaged_rows_are_skipped(self):
        with open(self.csv, "w") as f:
            f.write(HEADER)
            f.write(_row(1000.0))
            f.write("garbage,not,numbers\n")
            f.write("1002.0,2.0\n")          # truncated row
            f.write(_row(1004.0))
        run = runlog.load_run(self.csv)
        self.assertEqual(len(run.t), 2)
        self.assertAlmostEqual(run.t[-1], 4.0)

    def test_sidecar_metadata_is_attached(self):
        self._write([_row(1000.0)])
        dest = runlog.archive_run(self.csv, {"protocol": "pm"}, runs_dir=self.runs)
        run = runlog.load_run(dest)
        self.assertEqual(run.meta["protocol"], "pm")

    def test_meta_is_empty_dict_without_sidecar(self):
        self._write([_row(1000.0)])
        self.assertEqual(runlog.load_run(self.csv).meta, {})

    def test_duration_and_bounds(self):
        self._write([_row(1000.0), _row(1010.0)])
        run = runlog.load_run(self.csv)
        self.assertAlmostEqual(run.duration_s, 10.0)
        self.assertAlmostEqual(run.t_start, 1000.0)
        self.assertAlmostEqual(run.t_end, 1010.0)


class TestListRuns(_TmpDirCase):
    def test_newest_first(self):
        os.makedirs(self.runs)
        for name in ("run_20260101-000000.csv", "run_20260103-000000.csv",
                     "run_20260102-000000.csv"):
            open(os.path.join(self.runs, name), "w").close()
        names = [os.path.basename(p) for p in runlog.list_runs(self.runs)]
        self.assertEqual(names, ["run_20260103-000000.csv",
                                 "run_20260102-000000.csv",
                                 "run_20260101-000000.csv"])

    def test_missing_dir_is_empty(self):
        self.assertEqual(runlog.list_runs(self.runs), [])

    def test_sidecars_are_not_listed_as_runs(self):
        os.makedirs(self.runs)
        open(os.path.join(self.runs, "run_20260101-000000.csv"), "w").close()
        open(os.path.join(self.runs, "run_20260101-000000.json"), "w").close()
        self.assertEqual(len(runlog.list_runs(self.runs)), 1)


class TestEveEpisodes(_TmpDirCase):
    def _write_eve(self, rows):
        path = os.path.join(self.tmp, "eve.csv")
        with open(path, "w") as f:
            f.write("t_start,t_end,f\n")
            for r in rows:
                f.write(r)
        return path

    def test_only_episodes_overlapping_the_window(self):
        path = self._write_eve([
            "100.0,200.0,1.0000\n",     # before window
            "950.0,1050.0,0.5000\n",    # overlaps start
            "1100.0,1200.0,0.2000\n",   # inside
            "5000.0,5100.0,1.0000\n",   # after window
        ])
        eps = runlog.load_eve_episodes(1000.0, 2000.0, path)
        self.assertEqual(len(eps), 2)
        self.assertAlmostEqual(eps[0].f, 0.5)
        self.assertAlmostEqual(eps[1].t_start, 1100.0)

    def test_open_episode_extends_to_window_end(self):
        path = self._write_eve(["1500.0,0.000,0.3000\n"])
        eps = runlog.load_eve_episodes(1000.0, 2000.0, path)
        self.assertEqual(len(eps), 1)
        self.assertAlmostEqual(eps[0].t_end, 2000.0)

    def test_missing_file_is_empty(self):
        self.assertEqual(
            runlog.load_eve_episodes(0.0, 1.0, os.path.join(self.tmp, "no.csv")),
            [])

    def test_damaged_rows_skipped(self):
        path = self._write_eve(["oops\n", "1100.0,1200.0,0.2000\n"])
        self.assertEqual(len(runlog.load_eve_episodes(1000.0, 2000.0, path)), 1)


if __name__ == "__main__":
    unittest.main()
