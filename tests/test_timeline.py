"""Continuous GUI history across Apply/Stop and PM<->ENT switches.

The collector's secret_bits counter restarts at 0 on every Apply, so the
per-segment offset arithmetic here is what keeps the key buffer from dropping
back to zero at a mode switch. That is the whole point of the module, so most
of these tests pin it.
"""
import unittest

from timeline import Gap, Segment, Timeline


class TestEmpty(unittest.TestCase):
    def test_new_timeline_is_empty(self):
        tl = Timeline()
        self.assertEqual(len(tl), 0)
        self.assertEqual(tl.segments, [])
        self.assertEqual(tl.gaps, [])
        self.assertEqual(tl.total_secret, 0)
        self.assertEqual(tl.secret_by_protocol, {"pm": 0, "ent": 0})
        self.assertIsNone(tl.latest_protocol)

    def test_t0_of_empty_timeline_is_zero(self):
        self.assertEqual(Timeline().t0, 0.0)


class TestSingleSegment(unittest.TestCase):
    def setUp(self):
        self.tl = Timeline()
        self.tl.begin_downtime(100.0, "pm", 0.0522)

    def test_first_sample_opens_a_segment_and_closes_the_gap(self):
        self.tl.add_sample(110.0, 1.0, 500, 0)
        self.assertEqual(len(self.tl.segments), 1)
        seg = self.tl.segments[0]
        self.assertEqual(seg.protocol, "pm")
        self.assertAlmostEqual(seg.t_start, 110.0)
        self.assertAlmostEqual(seg.death_qber, 0.0522)
        self.assertEqual(len(self.tl.gaps), 1)
        self.assertAlmostEqual(self.tl.gaps[0].t_start, 100.0)
        self.assertAlmostEqual(self.tl.gaps[0].t_end, 110.0)

    def test_open_gap_before_the_first_sample(self):
        self.assertEqual(len(self.tl.gaps), 1)
        self.assertEqual(self.tl.gaps[0].t_end, 0.0)   # still down

    def test_samples_extend_the_segment(self):
        for i in range(3):
            self.tl.add_sample(110.0 + 2 * i, 1.0, 500, 100 * i)
        seg = self.tl.segments[0]
        self.assertAlmostEqual(seg.t_end, 114.0)
        self.assertEqual(seg.i0, 0)
        self.assertEqual(seg.i1, 2)
        self.assertEqual(len(self.tl), 3)

    def test_t0_is_the_first_sample(self):
        self.tl.add_sample(110.0, 1.0, 500, 0)
        self.tl.add_sample(112.0, 1.0, 500, 0)
        self.assertAlmostEqual(self.tl.t0, 110.0)

    def test_qber_and_sifted_are_stored_verbatim(self):
        self.tl.add_sample(110.0, 5.94, 1234, 7)
        self.assertAlmostEqual(self.tl.qber[0], 5.94)
        self.assertEqual(self.tl.sifted[0], 1234)


class TestSwitch(unittest.TestCase):
    def setUp(self):
        self.tl = Timeline()
        self.tl.begin_downtime(100.0, "pm", 0.0522)
        self.tl.add_sample(110.0, 1.0, 500, 0)
        self.tl.add_sample(112.0, 1.0, 500, 3000)     # PM distilled 3000 bits

    def test_secret_continues_across_a_switch(self):
        # The collector restarts its own counter at 0; the buffer must not.
        self.tl.begin_downtime(120.0, "ent", 0.0994)
        self.tl.add_sample(135.0, 6.0, 900, 0)        # fresh collector run
        self.assertEqual(self.tl.total_secret, 3000)
        self.tl.add_sample(137.0, 6.0, 900, 2000)     # ENT distilled 2000
        self.assertEqual(self.tl.total_secret, 5000)
        self.assertEqual(self.tl.secret_total[-1], 5000)

    def test_secret_never_decreases_at_the_boundary(self):
        self.tl.begin_downtime(120.0, "ent", 0.0994)
        self.tl.add_sample(135.0, 6.0, 900, 0)
        self.assertEqual(self.tl.secret_total,
                         sorted(self.tl.secret_total))

    def test_split_by_protocol(self):
        self.tl.begin_downtime(120.0, "ent", 0.0994)
        self.tl.add_sample(135.0, 6.0, 900, 0)
        self.tl.add_sample(137.0, 6.0, 900, 2000)
        self.assertEqual(self.tl.secret_by_protocol, {"pm": 3000, "ent": 2000})

    def test_two_segments_with_their_own_thresholds(self):
        self.tl.begin_downtime(120.0, "ent", 0.0994)
        self.tl.add_sample(135.0, 6.0, 900, 0)
        segs = self.tl.segments
        self.assertEqual([s.protocol for s in segs], ["pm", "ent"])
        self.assertAlmostEqual(segs[0].death_qber, 0.0522)
        self.assertAlmostEqual(segs[1].death_qber, 0.0994)
        self.assertEqual(segs[1].i0, 2)

    def test_gap_between_the_segments_is_a_switch(self):
        self.tl.begin_downtime(120.0, "ent", 0.0994)
        self.tl.add_sample(135.0, 6.0, 900, 0)
        gap = self.tl.gaps[-1]
        self.assertAlmostEqual(gap.t_start, 120.0)
        self.assertAlmostEqual(gap.t_end, 135.0)
        self.assertTrue(gap.is_switch)

    def test_restart_in_the_same_mode_is_not_a_switch(self):
        self.tl.begin_downtime(120.0, "pm", 0.0522)
        self.tl.add_sample(130.0, 1.0, 500, 0)
        self.assertFalse(self.tl.gaps[-1].is_switch)

    def test_stop_then_apply_same_mode_is_not_a_switch(self):
        # Stop does not know what comes next; the verdict is settled when the
        # next segment opens.
        self.tl.begin_downtime(120.0)                 # Stop
        self.tl.add_sample(130.0, 1.0, 500, 0)        # same mode came back
        self.assertFalse(self.tl.gaps[-1].is_switch)

    def test_stop_then_apply_other_mode_is_a_switch(self):
        self.tl.begin_downtime(120.0)                 # Stop
        self.tl.begin_downtime(125.0, "ent", 0.0994)  # Apply ENT
        self.tl.add_sample(135.0, 6.0, 900, 0)
        self.assertTrue(self.tl.gaps[-1].is_switch)
        # Stop and the Apply that follows it must coalesce into ONE gap --
        # its width is the measured switch cost t_switch. Only the setUp
        # gap (already closed) plus this single Stop->Apply gap may exist,
        # and the gap must start at the Stop time (120.0), not the later
        # Apply time (125.0). A version that always appends a new Gap on
        # begin_downtime would instead strand a phantom permanently-open
        # gap here and misreport the switch cost as 10s instead of 15s.
        self.assertEqual(len(self.tl.gaps), 2)
        self.assertAlmostEqual(self.tl.gaps[-1].t_start, 120.0)

    def test_latest_protocol_follows_the_running_segment(self):
        self.assertEqual(self.tl.latest_protocol, "pm")
        self.tl.begin_downtime(120.0, "ent", 0.0994)
        self.tl.add_sample(135.0, 6.0, 900, 0)
        self.assertEqual(self.tl.latest_protocol, "ent")


class TestCollectorRestartWithinASegment(unittest.TestCase):
    def test_counter_going_backwards_is_treated_as_a_new_run(self):
        # The collector can die and be respawned without the GUI pressing
        # Apply; its secret_bits then restarts at 0 mid-segment. Without this
        # guard the buffer would visibly drop.
        tl = Timeline()
        tl.begin_downtime(100.0, "pm", 0.0522)
        tl.add_sample(110.0, 1.0, 500, 1000)
        tl.add_sample(112.0, 1.0, 500, 4000)
        tl.add_sample(114.0, 1.0, 500, 500)      # restarted, counter reset
        self.assertEqual(tl.total_secret, 4500)
        self.assertEqual(tl.secret_total, sorted(tl.secret_total))

    def test_secret_by_protocol_sums_to_total_after_respawn_and_switch(self):
        # The 4000 bits produced before the respawn must not vanish from the
        # per-protocol split just because they got banked into
        # secret_offset instead of staying in secret_final.
        tl = Timeline()
        tl.begin_downtime(100.0, "pm", 0.0522)
        tl.add_sample(110.0, 1.0, 500, 1000)
        tl.add_sample(112.0, 1.0, 500, 4000)
        tl.add_sample(114.0, 1.0, 500, 500)      # restarted, counter reset
        self.assertEqual(tl.total_secret, 4500)
        self.assertEqual(sum(tl.secret_by_protocol.values()), tl.total_secret)
        self.assertEqual(tl.secret_by_protocol, {"pm": 4500, "ent": 0})

        tl.begin_downtime(120.0, "ent", 0.0994)
        tl.add_sample(135.0, 6.0, 900, 2000)     # ENT distilled 2000
        self.assertEqual(tl.total_secret, 6500)
        self.assertEqual(sum(tl.secret_by_protocol.values()), tl.total_secret)
        self.assertEqual(tl.secret_by_protocol, {"pm": 4500, "ent": 2000})


class TestOrphanSamples(unittest.TestCase):
    def test_sample_without_a_prior_downtime_still_works(self):
        # e.g. the app attaches to a stack that was already streaming.
        tl = Timeline()
        tl.add_sample(110.0, 1.0, 500, 42)
        self.assertEqual(len(tl.segments), 1)
        self.assertEqual(tl.segments[0].protocol, "pm")   # documented default
        self.assertEqual(tl.total_secret, 42)
        self.assertEqual(tl.gaps, [])


class TestReset(unittest.TestCase):
    def test_reset_clears_everything(self):
        tl = Timeline()
        tl.begin_downtime(100.0, "pm", 0.0522)
        tl.add_sample(110.0, 1.0, 500, 1000)
        tl.reset()
        self.assertEqual(len(tl), 0)
        self.assertEqual(tl.segments, [])
        self.assertEqual(tl.gaps, [])
        self.assertEqual(tl.total_secret, 0)
        self.assertIsNone(tl.latest_protocol)


class TestPendingDeathQber(unittest.TestCase):
    def test_threshold_set_after_downtime_reaches_the_segment(self):
        tl = Timeline()
        tl.begin_downtime(100.0, "ent", None)
        tl.set_pending_death_qber(0.0994)
        tl.add_sample(110.0, 6.0, 900, 0)
        self.assertAlmostEqual(tl.segments[0].death_qber, 0.0994)


class TestCancelDowntime(unittest.TestCase):
    """A restart that fails before the stack was stopped never went down."""

    def test_cancelled_downtime_leaves_no_gap_and_no_offset_jump(self):
        tl = Timeline()
        tl.begin_downtime(100.0, "pm", 0.0522)
        tl.add_sample(110.0, 1.0, 500, 1000)
        tl.begin_downtime(120.0, "ent", 0.0994)   # Apply pressed...
        tl.cancel_downtime()                      # ...but it raised early
        # Only the original (closed) startup gap survives: no new one is left
        # open, so the record shows uninterrupted streaming.
        self.assertEqual(tl.gaps, [Gap(t_start=100.0, t_end=110.0,
                                       is_switch=False)])
        # The still-running collector keeps counting: same segment, no offset.
        tl.add_sample(130.0, 1.0, 500, 1200)
        self.assertEqual(len(tl.segments), 1)
        self.assertEqual(tl.total_secret, 1200)
        self.assertEqual(tl.segments[0].protocol, "pm")   # no phantom switch

    def test_cancel_without_an_open_downtime_is_a_no_op(self):
        tl = Timeline()
        tl.cancel_downtime()                      # empty timeline
        self.assertEqual(tl.gaps, [])
        tl.begin_downtime(100.0, "pm", 0.0522)
        tl.add_sample(110.0, 1.0, 500, 1000)      # closes the gap
        tl.cancel_downtime()                      # must not eat a real gap
        self.assertEqual(len(tl.gaps), 1)
        self.assertEqual(tl.gaps[0].t_end, 110.0)


class TestUnknownProtocol(unittest.TestCase):
    def test_unknown_protocol_falls_back_to_pm_bucket(self):
        # secret_by_protocol must not raise on an unexpected string.
        tl = Timeline()
        tl.begin_downtime(100.0, "weird", None)
        tl.add_sample(110.0, 1.0, 500, 700)
        self.assertEqual(tl.secret_by_protocol["pm"], 700)


if __name__ == "__main__":
    unittest.main()
