"""The SSV's own motion: bounded speed, bounded range, reproducible."""
import unittest

import numpy as np

from mobility import MobilityParams, VesselTrack

# A band and a speed that make a 4,000-slot run cover the whole band several
# times over, so the properties below are exercised rather than merely stated.
BAND = MobilityParams(enabled=True, d_min=150.0, d_max=190.0, v_max=0.5,
                      loiter_min=0, loiter_max=20)


def track_of(track, slots):
    """`slots` successive distances from `track`, starting after one step."""
    return [track.step() for _ in range(slots)]


class TestTheSpeedBound(unittest.TestCase):
    """|d_t - d_{t-1}| <= v_max is THE defining property of this module."""

    def test_the_vessel_never_moves_further_than_v_max_in_one_slot(self):
        # MUTATION: jump straight to the target instead of closing v_max of
        # the gap (the "teleport" the user ruled out). Over 12 seeds x 4,000
        # slots there is no seed on which that stays hidden.
        for seed in range(12):
            t = VesselTrack(BAND)
            previous = t.reset(seed)
            for slot in range(4000):
                now = t.step()
                self.assertLessEqual(
                    abs(now - previous), BAND.v_max + 1e-12,
                    "seed %d slot %d: moved %.6f km, limit %.6f"
                    % (seed, slot, abs(now - previous), BAND.v_max))
                previous = now

    def test_the_bound_is_not_satisfied_by_standing_still(self):
        # Without this, a track that never moves passes the test above
        # perfectly. The run must really travel, and travel BOTH ways.
        t = VesselTrack(BAND)
        t.reset(3)
        d = track_of(t, 4000)
        steps = np.diff([BAND.d_min] + d)
        self.assertGreater(max(d) - min(d), 0.8 * (BAND.d_max - BAND.d_min))
        self.assertGreater(sum(1 for s in steps if s > 0), 200)
        self.assertGreater(sum(1 for s in steps if s < 0), 200)

    def test_a_full_speed_leg_moves_exactly_v_max(self):
        # The bound must bind, not merely hold: on a long leg the step is the
        # speed limit itself, so an implementation that crawls at v_max/2
        # (and would pass every inequality above) is caught here.
        t = VesselTrack(MobilityParams(enabled=True, d_min=0.0, d_max=100.0,
                                       v_max=0.25, loiter_min=0,
                                       loiter_max=0))
        start = t.reset(1)
        d = [start] + track_of(t, 500)
        moves = [abs(b - a) for a, b in zip(d, d[1:])]
        self.assertAlmostEqual(max(moves), 0.25, places=12)

    def test_the_speed_limit_is_a_parameter_and_is_obeyed(self):
        # A hardcoded default hiding behind the parameter would pass every
        # test that uses the default. Two different limits, two behaviours.
        for v_max in (0.01, 0.0129, 2.5):
            p = MobilityParams(enabled=True, d_min=0.0, d_max=200.0,
                               v_max=v_max, loiter_min=0, loiter_max=0)
            t = VesselTrack(p)
            previous = t.reset(5)
            steps = []
            for _ in range(1500):
                now = t.step()
                steps.append(abs(now - previous))
                previous = now
            self.assertLessEqual(max(steps), v_max + 1e-12)
            self.assertAlmostEqual(max(steps), v_max, places=12)


class TestTheBand(unittest.TestCase):
    def test_the_vessel_stays_inside_the_band(self):
        for seed in range(8):
            t = VesselTrack(BAND)
            d = [t.reset(seed)] + track_of(t, 3000)
            self.assertGreaterEqual(min(d), BAND.d_min - 1e-12)
            self.assertLessEqual(max(d), BAND.d_max + 1e-12)

    def test_it_visits_both_ends_of_the_band(self):
        # Targets are drawn across the WHOLE band, not around its middle.
        t = VesselTrack(BAND)
        t.reset(2)
        d = track_of(t, 20000)
        span = BAND.d_max - BAND.d_min
        self.assertLess(min(d), BAND.d_min + 0.05 * span)
        self.assertGreater(max(d), BAND.d_max - 0.05 * span)

    def test_it_never_overshoots_the_waypoint_it_is_steaming_to(self):
        # Closing the gap by v_max unconditionally would sail past a target
        # less than v_max away and then oscillate around it forever.
        t = VesselTrack(MobilityParams(enabled=True, d_min=10.0, d_max=60.0,
                                       v_max=0.7, loiter_min=0,
                                       loiter_max=0))
        previous = t.reset(4)
        target = t.target
        for _ in range(3000):
            now = t.step()
            if target is not None and (now - target) * (previous - target) < 0:
                self.fail("overshot %.6f: %.6f -> %.6f"
                          % (target, previous, now))
            previous, target = now, t.target

    def test_a_start_outside_the_band_is_clamped_into_it(self):
        # HybridParams.d_km is the fixed-link distance and has no reason to
        # sit inside a patrol band chosen later; starting 100 km outside would
        # otherwise burn 7,752 slots steaming toward it.
        self.assertEqual(VesselTrack(BAND, d_start=55.0).distance,
                         BAND.d_min)
        self.assertEqual(VesselTrack(BAND, d_start=400.0).distance,
                         BAND.d_max)
        self.assertEqual(VesselTrack(BAND, d_start=170.0).distance, 170.0)

    def test_the_start_survives_a_reset(self):
        t = VesselTrack(BAND, d_start=170.0)
        track_of(t, 500)
        self.assertEqual(t.reset(9), 170.0)


class TestLoitering(unittest.TestCase):
    def test_on_arrival_the_vessel_holds_station(self):
        # A fixed dwell makes the count exact: every arrival is followed by
        # exactly `loiter` motionless slots.
        hold = 40
        t = VesselTrack(MobilityParams(enabled=True, d_min=0.0, d_max=5.0,
                                       v_max=1.0, loiter_min=hold,
                                       loiter_max=hold))
        t.reset(6)
        d = track_of(t, 2000)
        runs, current = [], 0
        for a, b in zip(d, d[1:]):
            if a == b:
                current += 1
            else:
                if current:
                    runs.append(current)
                current = 0
        self.assertGreater(len(runs), 5)
        # hold slots of zero movement produce hold-1 equal ADJACENT pairs
        # inside the run plus the pair that enters it.
        self.assertEqual(set(runs), {hold})

    def test_zero_loiter_means_the_vessel_never_stops(self):
        t = VesselTrack(MobilityParams(enabled=True, d_min=0.0, d_max=50.0,
                                       v_max=0.3, loiter_min=0,
                                       loiter_max=0))
        t.reset(7)
        d = track_of(t, 1500)
        self.assertEqual(sum(1 for a, b in zip(d, d[1:]) if a == b), 0)

    def test_the_loiter_draw_spans_its_configured_range(self):
        t = VesselTrack(MobilityParams(enabled=True, d_min=0.0, d_max=2.0,
                                       v_max=1.0, loiter_min=2,
                                       loiter_max=6))
        t.reset(8)
        seen = set()
        for _ in range(4000):
            t.step()
            if t.loitering:
                seen.add(t.loitering)
        # loiter_max must be REACHABLE: randint's exclusive upper bound is the
        # classic off-by-one here.
        self.assertIn(6, seen)
        self.assertGreaterEqual(min(seen), 1)


class TestDeterminism(unittest.TestCase):
    """Same shape as EveChain: two policies must face the identical track."""

    def test_the_same_seed_replays_the_same_track(self):
        a, b = VesselTrack(BAND), VesselTrack(BAND)
        a.reset(11)
        b.reset(11)
        self.assertEqual(track_of(a, 800), track_of(b, 800))

    def test_different_seeds_give_different_tracks(self):
        a, b = VesselTrack(BAND), VesselTrack(BAND)
        a.reset(11)
        b.reset(12)
        self.assertNotEqual(track_of(a, 800), track_of(b, 800))

    def test_a_bare_reset_replays_the_track_fixed_by_params_seed(self):
        t = VesselTrack(MobilityParams(enabled=True, d_min=150.0, d_max=190.0,
                                       v_max=0.5, seed=21))
        first = track_of(t, 400)
        t.reset()
        self.assertEqual(track_of(t, 400), first)
        # ...and params.seed is really the thing that fixes it.
        other = VesselTrack(MobilityParams(enabled=True, d_min=150.0,
                                           d_max=190.0, v_max=0.5, seed=22))
        self.assertNotEqual(track_of(other, 400), first)

    def test_reset_rewinds_mid_track(self):
        t = VesselTrack(BAND, d_start=160.0)
        t.reset(13)
        first = track_of(t, 300)
        track_of(t, 500)
        t.reset(13)
        self.assertEqual(track_of(t, 300), first)


class TestConfigurationIsChecked(unittest.TestCase):
    """A silently absurd band produces a plausible-looking run, which is the
    worst outcome; these are hard errors."""

    def test_an_inverted_band_is_refused(self):
        with self.assertRaises(ValueError):
            VesselTrack(MobilityParams(enabled=True, d_min=90.0, d_max=10.0))

    def test_a_negative_distance_is_refused(self):
        with self.assertRaises(ValueError):
            VesselTrack(MobilityParams(enabled=True, d_min=-5.0, d_max=10.0))

    def test_a_non_positive_speed_is_refused(self):
        # v_max = 0 is not "a stationary vessel", it is a vessel that can
        # never reach its target and loiters nowhere: an infinite leg.
        for v_max in (0.0, -1.0):
            with self.assertRaises(ValueError):
                VesselTrack(MobilityParams(enabled=True, v_max=v_max))

    def test_an_empty_loiter_range_is_refused(self):
        with self.assertRaises(ValueError):
            VesselTrack(MobilityParams(enabled=True, loiter_min=10,
                                       loiter_max=3))
        with self.assertRaises(ValueError):
            VesselTrack(MobilityParams(enabled=True, loiter_min=-1))

    def test_a_single_point_band_is_legal_and_stands_still(self):
        t = VesselTrack(MobilityParams(enabled=True, d_min=77.0, d_max=77.0))
        self.assertEqual(set(track_of(t, 200)), {77.0})


class TestTheDefaults(unittest.TestCase):
    def test_mobility_is_off_by_default(self):
        # Every result in the study predating the vessel was measured on a
        # fixed link; the default must not silently change what they mean.
        self.assertFalse(MobilityParams().enabled)

    def test_the_default_speed_is_the_quoted_25_knot_anchor(self):
        # 25 kn = 25 * 1.852 km/h = 46.3 km/h; at a 1 s slot that is
        # 46.3/3600 km/slot. The docstring quotes 0.0129 and the number must
        # match the arithmetic, not the other way round.
        self.assertAlmostEqual(MobilityParams().v_max,
                               25 * 1.852 / 3600.0, places=4)

    def test_the_default_band_brackets_the_studys_standing_link(self):
        # 55 km is HybridParams.d_km. The default band is a PLACEHOLDER and
        # is centred there so that enabling mobility with no arguments cannot
        # smuggle in an operating range the study has not justified.
        p = MobilityParams()
        self.assertLess(p.d_min, 55.0)
        self.assertGreater(p.d_max, 55.0)


if __name__ == "__main__":
    unittest.main()
