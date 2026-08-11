"""Figure building. Rendering only -- data handling is tested in test_runlog."""
import os
import shutil
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")            # no display in tests

import numpy as np  # noqa: E402

import plots  # noqa: E402
import runlog  # noqa: E402


def _run(n=30, blocks_at=(12, 24)):
    """A synthetic run: rising QBER, two completed blocks."""
    t = np.arange(n, dtype=float) * 2.0
    run = runlog.RunData(path="run_20260101-000000.csv",
                         meta={"protocol": "pm", "n_z_block": 1000,
                               "env": {"d_km": 25.0, "mu1": 0.29},
                               "forecast": {"l_predict": 500}})
    run.unix_t = 1_000_000.0 + t
    run.t = t
    run.window_s = np.full(n, 2.0)
    run.qber = np.linspace(0.01, 0.05, n)
    run.clicks = np.full(n, 200_000, dtype=np.int64)
    run.sifted = np.full(n, 100_000, dtype=np.int64)
    run.errors = (run.sifted * run.qber).astype(np.int64)
    run.total_sifted = np.cumsum(run.sifted)
    blocks = np.zeros(n, dtype=np.int64)
    secret = np.zeros(n, dtype=np.int64)
    for i, idx in enumerate(blocks_at, start=1):
        blocks[idx:] = i
        secret[idx:] = i * 500
    run.blocks = blocks
    run.secret_bits = secret
    progress = np.arange(n, dtype=np.int64) * 100
    for idx in blocks_at:
        progress[idx:] -= progress[idx]
    run.block_progress = progress
    return run


def _session_run(n=20, split=10):
    """A run whose sidecar describes a PM stretch, a gap, then an ENT stretch."""
    run = _run(n=n, blocks_at=(6, 15))
    run.segments = [
        {"protocol": "pm", "t_start": run.unix_t[0],
         "t_end": run.unix_t[split - 1], "death_qber": 0.0522,
         "i0": 0, "i1": split - 1},
        {"protocol": "ent", "t_start": run.unix_t[split],
         "t_end": run.unix_t[-1], "death_qber": 0.0994,
         "i0": split, "i1": n - 1},
    ]
    run.gaps = [{"t_start": run.unix_t[split - 1] + 0.5,
                 "t_end": run.unix_t[split] - 0.5, "is_switch": True}]
    run.meta = dict(run.meta, kind="session",
                    secret_by_protocol={"pm": 500, "ent": 500})
    return run


class TestFigureRun(unittest.TestCase):
    def test_has_four_panels(self):
        fig = plots.figure_run(_run())
        self.assertEqual(len(fig.axes), 4)

    def test_optional_overlays_do_not_change_panel_count(self):
        fig = plots.figure_run(_run(), death_qber=0.0522, b_max=2000,
                               eve=[runlog.EveEpisode(1_000_010.0,
                                                      1_000_020.0, 0.5)])
        self.assertEqual(len(fig.axes), 4)

    def test_empty_run_still_builds_a_figure(self):
        fig = plots.figure_run(runlog.RunData())
        self.assertEqual(len(fig.axes), 4)

    def test_single_sample_run(self):
        run = _run(n=1, blocks_at=())
        self.assertEqual(len(plots.figure_run(run).axes), 4)

    def test_title_mentions_the_run_name(self):
        fig = plots.figure_run(_run())
        text = " ".join(t.get_text() for t in fig.texts)
        self.assertIn("run_20260101-000000", text)

    def test_subtitle_lists_parameters_from_metadata(self):
        fig = plots.figure_run(_run())
        text = " ".join(t.get_text() for t in fig.texts)
        self.assertIn("PM", text.upper())
        self.assertIn("25", text)          # d_km from env metadata

    def test_plain_subtitle_has_no_math_markup(self):
        # The CLI listing prints this to a terminal, where "$\mu=..." is noise.
        plain = plots._subtitle(_run(), latex=False)
        self.assertNotIn("$", plain)
        self.assertNotIn("\\", plain)
        self.assertIn("25", plain)
        self.assertIn("PM", plain)


class TestPanels(unittest.TestCase):
    def _ax(self):
        from matplotlib.figure import Figure
        return Figure().add_subplot(1, 1, 1)

    def test_qber_panel_draws_death_line(self):
        ax = self._ax()
        plots.panel_qber(ax, _run(), death_qber=0.0522)
        self.assertTrue(any(ln.get_linestyle() == "--" for ln in ax.lines))

    def test_qber_panel_shades_eve_episodes(self):
        ax = self._ax()
        before = len(ax.patches)
        plots.panel_qber(ax, _run(),
                         eve=[runlog.EveEpisode(1_000_010.0, 1_000_020.0, 1.0)])
        self.assertGreater(len(ax.patches), before)

    def test_secret_panel_draws_b_max_line(self):
        ax = self._ax()
        plots.panel_secret(ax, _run(), b_max=2000)
        self.assertTrue(ax.lines)

    def test_panels_accept_an_empty_run(self):
        for panel in (plots.panel_qber, plots.panel_rate,
                      plots.panel_secret, plots.panel_blocks):
            panel(self._ax(), runlog.RunData())


class TestSaveFigures(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_combined_writes_one_file(self):
        paths = plots.save_run_figures(_run(), self.out)
        self.assertEqual(len(paths), 1)
        self.assertTrue(os.path.exists(paths[0]))
        self.assertGreater(os.path.getsize(paths[0]), 1000)

    def test_split_writes_one_file_per_panel(self):
        paths = plots.save_run_figures(_run(), self.out, split=True)
        self.assertEqual(len(paths), 4)
        for p in paths:
            self.assertTrue(os.path.exists(p))

    def test_creates_missing_output_dir(self):
        nested = os.path.join(self.out, "deep", "figs")
        paths = plots.save_run_figures(_run(), nested)
        self.assertTrue(os.path.exists(paths[0]))


class TestSessionFigure(unittest.TestCase):
    def test_mode_bands_and_downtime_band_are_drawn(self):
        import matplotlib.colors as mcolors
        fig = plots.figure_run(_session_run())
        ax = fig.axes[0]
        faces = {tuple(p.get_facecolor()[:3]) for p in ax.patches}
        self.assertIn(mcolors.to_rgb(plots.MODE_BAND["pm"]), faces)
        self.assertIn(mcolors.to_rgb(plots.MODE_BAND["ent"]), faces)
        self.assertIn(mcolors.to_rgb(plots.DOWNTIME_BAND), faces)

    def test_one_line_per_segment_so_the_gap_is_not_bridged(self):
        # A single trace across both segments would draw a fake line over the
        # downtime, which is the whole point of splitting by segment.
        run = _session_run()
        fig = plots.figure_run(run)
        ax = fig.axes[0]
        t0 = run.t_start
        gap_lo = run.gaps[0]["t_start"] - t0
        gap_hi = run.gaps[0]["t_end"] - t0
        traces = [ln for ln in ax.lines if len(ln.get_xdata()) > 1]
        self.assertGreaterEqual(len(traces), 2)
        for line in traces:
            xs = list(line.get_xdata())
            spans_gap = min(xs) < gap_lo and max(xs) > gap_hi
            self.assertFalse(spans_gap,
                             "a trace bridges the downtime: %r" % (xs,))

    def test_threshold_steps_between_segments(self):
        fig = plots.figure_run(_session_run())
        ax = fig.axes[0]
        ys = sorted({round(float(c.get_segments()[0][0][1]), 2)
                     for c in ax.collections if c.get_segments()})
        self.assertEqual(ys, [5.22, 9.94])

    def test_buffer_panel_splits_by_protocol(self):
        fig = plots.figure_run(_session_run())
        # panel 3 is the cumulative secret bits
        self.assertGreaterEqual(len(fig.axes[2].lines), 1)

    def test_subtitle_mentions_the_switch_cost(self):
        fig = plots.figure_run(_session_run())
        text = " ".join(t.get_text() for t in fig.texts)
        self.assertIn("switch", text.lower())


class TestPlainRunUnchanged(unittest.TestCase):
    """A run without segments must render exactly as it did before."""

    def test_no_bands_are_added(self):
        fig = plots.figure_run(_run())
        self.assertEqual(len(fig.axes[0].patches), 0)

    def test_single_trace(self):
        fig = plots.figure_run(_run())
        traces = [l for l in fig.axes[0].lines if len(l.get_xdata()) > 1]
        self.assertEqual(len(traces), 1)

    def test_death_line_still_drawn_from_the_argument(self):
        fig = plots.figure_run(_run(), death_qber=0.0522)
        self.assertTrue(any(ln.get_linestyle() == "--"
                            for ln in fig.axes[0].lines))


if __name__ == "__main__":
    unittest.main()
