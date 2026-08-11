"""GUI history survives Apply/Stop and mode switches.

Drives the real App with a stub backend (same approach as
tests/test_gui_default.py); skips cleanly where there is no display.
"""
import unittest


def _load_gui():
    try:
        import gui
    except ImportError:
        return None
    return gui


class _StubSnapshot:
    def __init__(self, status="streaming", protocol="pm", unix_time=0.0,
                 qber=0.01, sifted=1000, secret_bits=0):
        self.status = status
        self.protocol = protocol
        self.unix_time = unix_time
        self.qber = qber
        self.clicks = 2 * sifted
        self.sifted = sifted
        self.total_sifted = sifted
        self.blocks = 0
        self.block_progress = 0
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
    """Boots a real App on the stub backend; skips without a display."""

    def setUp(self):
        self.gui = _load_gui()
        if self.gui is None:
            self.skipTest("matplotlib/tkinter not available")
        self.backend = _StubBackend()
        try:
            self.app = self.gui.App(self.backend)
        except Exception as exc:
            self.skipTest("cannot create Tk window: %s" % exc)
        self.app.withdraw()

    def tearDown(self):
        if getattr(self, "app", None) is not None:
            self.app.destroy()

    def _stream(self, protocol, times, secret_seq):
        for unix_t, secret in zip(times, secret_seq):
            self.backend.snapshot = _StubSnapshot(
                status="streaming", protocol=protocol, unix_time=unix_t,
                qber=0.01, sifted=1000, secret_bits=secret)
            self.app._poll_once()


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


if __name__ == "__main__":
    unittest.main()
