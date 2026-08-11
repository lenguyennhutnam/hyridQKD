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

    def predict(self, env, n_z_block=10_000_000, protocol="pm"):
        import channel
        if protocol == "ent":
            return channel.derive_ent(env, n_z_block)
        return channel.derive(env, n_z_block)

    def apply(self, env, n_z_block=10_000_000, protocol="pm"):
        return self.predict(env, n_z_block, protocol)

    def refresh(self):
        return self.snapshot

    def stop(self):
        pass


class TestTimelineAcrossSwitch(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
