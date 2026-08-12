"""GUI history survives Apply/Stop and mode switches.

Drives the real App with a stub backend (same approach as
tests/test_gui_default.py); skips cleanly where there is no display.
"""
import os
import tempfile
import unittest

import matplotlib.colors as mcolors


def _load_gui():
    try:
        import gui
        import panelstate
    except ImportError:
        return None, None
    return gui, panelstate


class _StubSnapshot:
    """Mirrors backend.Snapshot -- every field _poll_once() reads.

    window_s/errors are not shown anywhere in the panel, but the GUI forwards
    them into the Timeline for the session export, so a stub without them
    turns every poll into a caught AttributeError.
    """

    def __init__(self, status="streaming", protocol="pm", unix_time=0.0,
                 qber=0.01, sifted=1000, secret_bits=0, clicks=None,
                 window_s=2.0, errors=0, total_sifted=None, blocks=0,
                 block_progress=0):
        self.status = status
        self.protocol = protocol
        self.unix_time = unix_time
        self.window_s = window_s
        self.qber = qber
        self.clicks = 2 * sifted if clicks is None else clicks
        self.sifted = sifted
        self.errors = errors
        self.total_sifted = sifted if total_sifted is None else total_sifted
        self.blocks = blocks
        self.block_progress = block_progress
        self.secret_bits = secret_bits
        self.log_lines = []
        self.eve_active = False
        self.eve_f = 0.0


class _StubBackend:
    class _Eve:
        active = False
        current_f = 0.0

        def episodes(self):
            return []

    def __init__(self):
        self.eve = self._Eve()
        self.snapshot = _StubSnapshot(status="down")
        # Set to an exception to simulate a restart/stop that fails. The real
        # backend.apply() validates params BEFORE stopping the stack, so an
        # early failure leaves the old stack streaming.
        self.apply_error = None
        self.stop_error = None

    def predict(self, env, n_z_block=10_000_000, protocol="pm"):
        import channel
        if protocol == "ent":
            return channel.derive_ent(env, n_z_block)
        return channel.derive(env, n_z_block)

    def apply(self, env, n_z_block=10_000_000, protocol="pm"):
        if self.apply_error is not None:
            raise self.apply_error
        return self.predict(env, n_z_block, protocol)

    def refresh(self):
        return self.snapshot

    def stop(self):
        if self.stop_error is not None:
            raise self.stop_error


class _AppCase(unittest.TestCase):
    """Boots a real App on the stub backend; skips without a display.

    Redirects panelstate.DEFAULT_PATH to a throwaway file for the duration of
    the test. App.__init__ loads any saved default over the seeded preset, so
    without this a developer's real panel_default.json leaks into these
    fixtures: a saved b_max < 5000 clips the ENT buffer bar to width 0 (one
    fewer patch than the tests expect), and a saved non-positive block makes
    _read_n_z_block() raise before begin_downtime() is ever reached.
    """

    def setUp(self):
        self.gui, self.panelstate = _load_gui()
        if self.gui is None:
            self.skipTest("matplotlib/tkinter not available")
        fd, self.default_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.default_path)              # start with "no default"
        self._orig_default_path = self.panelstate.DEFAULT_PATH
        self.panelstate.DEFAULT_PATH = self.default_path
        self.backend = _StubBackend()
        try:
            self.app = self.gui.App(self.backend)
        except Exception as exc:
            self.panelstate.DEFAULT_PATH = self._orig_default_path
            self.skipTest("cannot create Tk window: %s" % exc)
        self.app.withdraw()

    def tearDown(self):
        self.panelstate.DEFAULT_PATH = self._orig_default_path
        if getattr(self, "app", None) is not None:
            self.app.update_idletasks()
            self.app.destroy()
        if os.path.exists(self.default_path):
            os.remove(self.default_path)

    def _stream(self, protocol, times, secret_seq):
        for unix_t, secret in zip(times, secret_seq):
            self.backend.snapshot = _StubSnapshot(
                status="streaming", protocol=protocol, unix_time=unix_t,
                qber=0.01, sifted=1000, secret_bits=secret)
            self.app._poll_once()


class TestSnapshotForwarding(_AppCase):
    """_poll_once() must hand the Timeline the WHOLE snapshot.

    The Timeline carries ten collector columns but defaults six of them to 0,
    so a GUI that forwards only the four plotted values still passes every
    Timeline and sessionio test -- and exports a file whose clicks, block fill
    and block count are flat zeros. This seam is the one place that catches it.
    """

    def test_poll_forwards_every_collector_column(self):
        self.backend.snapshot = _StubSnapshot(
            status="streaming", protocol="pm", unix_time=1000.0, qber=0.0531,
            sifted=123_456, secret_bits=777, clicks=246_912, window_s=2.5,
            errors=6_555, total_sifted=987_654, blocks=3,
            block_progress=444_555)
        self.app._poll_once()
        self.assertEqual(self.app.error_var.get(), "",
                         "the poll itself failed, so this proves nothing")
        tl = self.app.timeline
        self.assertEqual(tl.t[-1], 1000.0)
        self.assertAlmostEqual(tl.qber[-1], 5.31)
        self.assertEqual(tl.sifted[-1], 123_456)
        self.assertEqual(tl.secret_total[-1], 777)
        self.assertEqual(tl.clicks[-1], 246_912)
        self.assertEqual(tl.window_s[-1], 2.5)
        self.assertEqual(tl.errors[-1], 6_555)
        self.assertEqual(tl.total_sifted[-1], 987_654)
        self.assertEqual(tl.blocks[-1], 3)
        self.assertEqual(tl.block_progress[-1], 444_555)


class TestTimelineAcrossSwitch(_AppCase):
    def test_history_survives_a_mode_switch(self):
        self.app.timeline.begin_downtime(100.0, "pm", 0.0522)
        self._stream("pm", [110.0, 112.0], [0, 3000])
        self.app.timeline.begin_downtime(120.0, "ent", 0.0994)
        self._stream("ent", [135.0, 137.0], [0, 2000])

        tl = self.app.timeline
        self.assertEqual(len(tl), 4, "samples from both modes must be kept")
        self.assertEqual([s.protocol for s in tl.segments], ["pm", "ent"])
        self.assertTrue(tl.gaps[-1].is_switch)

    def test_buffer_does_not_drop_at_the_switch(self):
        self.app.timeline.begin_downtime(100.0, "pm", 0.0522)
        self._stream("pm", [110.0, 112.0], [0, 3000])
        self.app.timeline.begin_downtime(120.0, "ent", 0.0994)
        self._stream("ent", [135.0], [0])
        self.assertEqual(self.app.timeline.total_secret, 3000)

    def test_redraw_survives_multiple_segments(self):
        self.app.timeline.begin_downtime(100.0, "pm", 0.0522)
        self._stream("pm", [110.0, 112.0], [0, 3000])
        self.app.timeline.begin_downtime(120.0, "ent", 0.0994)
        self._stream("ent", [135.0, 137.0], [0, 2000])
        self.app._redraw()                       # must not raise
        self.assertEqual(len(self.app.canvas.figure.axes), 3)

    def test_stop_opens_a_gap_without_losing_history(self):
        self.app.timeline.begin_downtime(100.0, "pm", 0.0522)
        self._stream("pm", [110.0, 112.0], [0, 3000])
        self.app._stop()
        self.assertEqual(len(self.app.timeline), 2)
        self.assertEqual(self.app.timeline.gaps[-1].t_end, 0.0)


class TestFailedRestartLeavesNoPhantomGap(_AppCase):
    """backend.apply() validates params before stopping the stack, so an early
    failure leaves the old stack streaming. The optimistic downtime recorded by
    _apply() must then be withdrawn, or it fakes a switch episode and makes the
    next segment double-count the still-running collector's key total."""

    def test_apply_failing_early_records_no_gap_and_no_buffer_jump(self):
        self.app.timeline.begin_downtime(100.0, "pm", 0.0522)
        self._stream("pm", [110.0, 112.0], [800, 1000])
        self.app.timeline._gaps.clear()          # that first gap is closed anyway
        self.backend.apply_error = ValueError("bad params")
        self.app.proto_var.set("ent")            # a flip that must NOT be logged
        self.app._apply()
        self.assertEqual(self.app.timeline.gaps, [])
        self.assertIn("Apply failed", self.app.error_var.get())
        # The same collector keeps counting; the total must follow it exactly.
        self._stream("pm", [114.0], [1200])
        self.assertEqual(self.app.timeline.total_secret, 1200)
        self.assertEqual(len(self.app.timeline.segments), 1)

    def test_stop_failing_while_the_stack_streams_records_no_gap(self):
        self.app.timeline.begin_downtime(100.0, "pm", 0.0522)
        self._stream("pm", [110.0], [1000])
        self.app.timeline._gaps.clear()
        self.backend.stop_error = RuntimeError("stop failed")
        self.app._stop()
        self.assertEqual(self.app.timeline.gaps, [])
        self.assertIn("Stop failed", self.app.error_var.get())

    def test_stop_failing_after_the_stack_died_keeps_the_gap(self):
        # Late failure: the stack is already down, so refresh() no longer
        # reports streaming and the gap is real.
        self.app.timeline.begin_downtime(100.0, "pm", 0.0522)
        self._stream("pm", [110.0], [1000])
        self.app.timeline._gaps.clear()
        self.backend.stop_error = RuntimeError("stop failed")
        self.backend.snapshot = _StubSnapshot(status="down", unix_time=112.0)
        self.app._stop()
        self.assertEqual(len(self.app.timeline.gaps), 1)
        self.assertEqual(self.app.timeline.gaps[0].t_end, 0.0)


class TestSegmentDrawing(_AppCase):
    def setUp(self):
        super().setUp()
        tl = self.app.timeline
        tl.begin_downtime(100.0, "pm", 0.0522)
        for t, s in ((110.0, 0), (112.0, 3000)):
            tl.add_sample(t, 5.0, 1000, s)
        tl.begin_downtime(120.0, "ent", 0.0994)
        for t, s in ((135.0, 0), (137.0, 2000)):
            tl.add_sample(t, 9.0, 900, s)
        self.app._redraw()

    def test_one_line_per_segment_so_the_gap_is_not_bridged(self):
        # A single Line2D across both segments would draw a fake trace over
        # the downtime, which here runs from x=10 to x=25 (unix 120..135,
        # shifted by t0=110). Counting lines alone is not enough: the
        # _mark_current axhline already makes two. So also require that no
        # single line carries points on both sides of the gap.
        self.assertGreaterEqual(len(self.app.ax_qber.lines), 2)
        for line in self.app.ax_qber.lines:
            xs = list(line.get_xdata())
            before = [x for x in xs if x <= 10.0]
            after = [x for x in xs if x >= 25.0]
            self.assertFalse(before and after,
                             "a single line bridges the downtime gap")

    def test_mode_band_reaches_the_apply_press_not_the_last_sample(self):
        # PM's last sample is at unix 112 but Apply was pressed at 120; the
        # tint must reach 120 (x=10) or an untinted sliver appears before the
        # grey band, reading as "stopped" while the link was still streaming.
        import matplotlib.colors as mcolors
        want = mcolors.to_rgb(self.gui.MODE_BAND["pm"])
        pm = [p for p in self.app.ax_qber.patches
              if tuple(p.get_facecolor()[:3]) == want]
        self.assertEqual(len(pm), 1)
        xs = [pt[0] for pt in pm[0].get_xy()]
        self.assertAlmostEqual(max(xs), 10.0, places=6)   # unix 120 - t0 110

    def test_the_running_segment_still_ends_at_its_last_sample(self):
        import matplotlib.colors as mcolors
        want = mcolors.to_rgb(self.gui.MODE_BAND["ent"])
        ent = [p for p in self.app.ax_qber.patches
               if tuple(p.get_facecolor()[:3]) == want]
        self.assertEqual(len(ent), 1)
        xs = [pt[0] for pt in ent[0].get_xy()]
        self.assertAlmostEqual(max(xs), 27.0, places=6)   # unix 137 - t0 110

    def test_mode_bands_and_a_grey_downtime_band_are_drawn(self):
        facecolors = {tuple(p.get_facecolor()[:3])
                     for p in self.app.ax_qber.patches}
        self.assertGreaterEqual(len(self.app.ax_qber.patches), 3,
                                "two mode bands plus one downtime band")
        self.assertIn(mcolors.to_rgb(self.gui.DOWNTIME_BAND), facecolors)
        self.assertIn(mcolors.to_rgb(self.gui.MODE_BAND["pm"]), facecolors)
        self.assertIn(mcolors.to_rgb(self.gui.MODE_BAND["ent"]), facecolors)

    def test_threshold_is_drawn_per_segment_not_once(self):
        # Both thresholds are above DEATH_LINE_SHOW_FRAC of their segment's
        # QBER here (5% vs 5.22%, 9% vs 9.94%), so both must appear as their
        # OWN hlines() call — a single session-wide threshold line would
        # collapse this to one collection and fail the length check below.
        segs = self.app.ax_qber.collections
        self.assertEqual(len(segs), 2)
        found = []
        for col in segs:
            (x0, y0), (x1, y1) = col.get_segments()[0]
            self.assertAlmostEqual(y0, y1)
            found.append((round(y0, 2), (min(x0, x1), max(x0, x1))))
        found.sort()
        self.assertEqual(found, [(5.22, (0.0, 2.0)), (9.94, (25.0, 27.0))])
        # x-extents must be disjoint: the step is what makes the PM vs ENT
        # threshold change visible at the switch.
        (_, (pm_x0, pm_x1)), (_, (ent_x0, ent_x1)) = found
        self.assertLess(pm_x1, ent_x0)

    def test_buffer_is_split_by_protocol(self):
        self.app._draw_buffer()
        # track + PM portion + ENT portion
        self.assertGreaterEqual(len(self.app.ax_buffer.patches), 3)

    def test_buffer_total_matches_the_timeline(self):
        self.assertEqual(self.app.timeline.total_secret, 5000)
        self.assertEqual(self.app.timeline.secret_by_protocol,
                         {"pm": 3000, "ent": 2000})

    def test_no_band_reaches_left_of_the_first_sample(self):
        # The session's FIRST gap opens on Apply, before any sample exists, so
        # its raw span starts at a negative x (t0 is the first SAMPLE). Drawn
        # unclamped it would drag the x-axis into negative time.
        for patch in self.app.ax_qber.patches:
            xs = [point[0] for point in patch.get_xy()]
            self.assertGreaterEqual(min(xs), 0.0,
                                    "gap/mode bands must be clamped to x >= 0")


if __name__ == "__main__":
    unittest.main()
