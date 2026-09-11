"""Baseline policies for the hybrid mode-switching problem.

These baselines are what the learned policy of Task 6 must beat, so the tests
here are written to kill the plausible *weakened* implementations, not merely
to exercise the happy path: an inverted comparison, a BufferAwareThreshold
that quietly ignores the buffer, a switch counter that fires on the first
slot, and an episode runner that reports something other than eq (24).
"""
import unittest

import numpy as np

import hybridenv
from evemarkov import EveParams
from hybridenv import (DIST_REF_KM, ENT, I_BUFFER, I_DIST, I_QBER, PM,
                       HybridEnv, HybridParams)
from evemarkov import EveParams
from policies import (AlwaysENT, AlwaysPM, BlockAwareGreedy,
                      BufferAwareThreshold, DecoyWatch, GreedyThreshold,
                      HysteresisThreshold, Oracle, Policy, RangeOrQber,
                      RangeThreshold, RelaxationBound, RelaxedAlwaysPM,
                      observed_distance_km, run_episode, run_relaxed_episode)


def obs_with(qber=0.0, mode=PM, block=0.0, buffer=0.0, queue=0.0, age=0.0,
             d_km=0.0, decoy=0.0):
    """Build an observation by NAME, using the layout hybridenv owns.

    Positional literals would keep passing if the index constants ever drifted
    apart from hybridenv._observe(); routing through the real constants means
    a drift shows up here.

    `d_km` is given in KILOMETRES and scaled here, so a test reads as the
    physical situation it means rather than as a normalised fraction.
    """
    v = np.zeros(hybridenv.OBS_DIM)
    v[hybridenv.I_QBER] = qber
    v[hybridenv.I_MODE] = float(mode)
    v[hybridenv.I_BLOCK] = block
    v[hybridenv.I_BUFFER] = buffer
    v[hybridenv.I_QUEUE] = queue
    v[hybridenv.I_AGE] = age
    v[hybridenv.I_DIST] = d_km / DIST_REF_KM
    v[hybridenv.I_DECOY] = decoy
    return v


def obs_at_distance(d_km):
    """An otherwise-blank observation of a vessel `d_km` from HQ."""
    return obs_with(d_km=d_km)


class TestPolicyBase(unittest.TestCase):
    def test_the_base_class_refuses_to_act(self):
        with self.assertRaises(NotImplementedError):
            Policy().act(obs_with())

    def test_every_baseline_carries_a_distinct_label(self):
        # Task 7 keys its result tables on these, so two policies sharing a
        # name would silently overwrite each other's row.
        names = [AlwaysPM().name, AlwaysENT().name,
                 GreedyThreshold(0.03).name, GreedyThreshold(0.05).name,
                 BufferAwareThreshold(0.03, 0.5).name,
                 BufferAwareThreshold(0.03, 0.8).name,
                 RangeThreshold(150.0).name, RangeThreshold(190.0).name,
                 RangeOrQber(150.0, 0.03).name,
                 RangeOrQber(150.0, 0.05).name]
        self.assertEqual(len(set(names)), len(names))


class TestObservationConstantsAreShared(unittest.TestCase):
    def test_policies_reexports_the_layout_hybridenv_defines(self):
        # hybridenv._observe() fixes the order; policies must not keep a
        # second copy that can drift out of step with it.
        import policies
        for name in ("I_QBER", "I_MODE", "I_BLOCK", "I_BUFFER", "I_QUEUE",
                     "I_AGE", "I_DIST", "DIST_REF_KM"):
            self.assertEqual(getattr(policies, name),
                             getattr(hybridenv, name), name)


class TestFixedPolicies(unittest.TestCase):
    def test_always_pm_only_ever_picks_pm(self):
        p = AlwaysPM()
        for _ in range(10):
            self.assertEqual(p.act(np.zeros(hybridenv.OBS_DIM)), PM)

    def test_always_ent_only_ever_picks_ent(self):
        p = AlwaysENT()
        for _ in range(10):
            self.assertEqual(p.act(np.zeros(hybridenv.OBS_DIM)), ENT)

    def test_the_fixed_policies_ignore_a_screaming_observation(self):
        # A "fixed" policy that peeked at anything would stop being the
        # control condition the study compares against.
        loud = obs_with(qber=0.49, mode=ENT, block=0.99, buffer=0.0,
                        queue=2.0, age=1.0)
        self.assertEqual(AlwaysPM().act(loud), PM)
        self.assertEqual(AlwaysENT().act(loud), ENT)


class TestGreedyThreshold(unittest.TestCase):
    def test_it_stays_on_pm_while_qber_is_low(self):
        p = GreedyThreshold(threshold=0.04)
        self.assertEqual(p.act(obs_with(qber=0.01, buffer=0.5)), PM)

    def test_it_moves_to_ent_once_qber_crosses(self):
        p = GreedyThreshold(threshold=0.04)
        self.assertEqual(p.act(obs_with(qber=0.05, buffer=0.5)), ENT)

    def test_the_threshold_itself_already_counts_as_crossed(self):
        # Pins >= rather than >. PM's death QBER is the threshold's natural
        # setting, and at exactly the death point PM yields nothing.
        p = GreedyThreshold(threshold=0.04)
        self.assertEqual(p.act(obs_with(qber=0.04, buffer=0.5)), ENT)

    def test_it_ignores_every_feature_except_qber(self):
        # This is what makes it the *plain* threshold, and what
        # BufferAwareThreshold is defined against.
        p = GreedyThreshold(threshold=0.04)
        for buffer in (0.0, 0.25, 0.5, 1.0):
            for queue in (0.0, 2.0):
                self.assertEqual(
                    p.act(obs_with(qber=0.035, buffer=buffer, queue=queue,
                                   age=queue / 2.0)), PM)
                self.assertEqual(
                    p.act(obs_with(qber=0.045, buffer=buffer, queue=queue,
                                   age=queue / 2.0)), ENT)

    def test_it_reads_the_qber_slot_a_real_environment_writes(self):
        # End to end against hybridenv rather than a hand-built vector: if the
        # index constants and _observe() ever disagree, the hand-built vectors
        # above would all still pass while the policy read the wrong feature.
        env = HybridEnv(HybridParams(horizon=5, seed=1),
                        EveParams(f_levels=(0.5, 0.5, 0.5), seed=1))
        env.reset()
        obs, _, _, _ = env.step(PM)
        self.assertGreater(obs[I_QBER], 0.2)     # fully jammed slot
        self.assertEqual(GreedyThreshold(threshold=0.04).act(obs), ENT)
        self.assertEqual(GreedyThreshold(threshold=0.90).act(obs), PM)


class TestBufferAwareThreshold(unittest.TestCase):
    """Switching only when QBER is already high is too late: ENT needs a whole
    block before it delivers anything (spec 2.1), so the buffer has to cover
    that gap. This policy switches early when the buffer is thin."""

    def test_a_thin_buffer_switches_before_the_plain_threshold_would(self):
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        obs = obs_with(qber=0.035, buffer=0.05)      # buffer nearly empty
        self.assertEqual(p.act(obs), ENT)
        # ... and the plain threshold, on the very same observation, does not.
        self.assertEqual(GreedyThreshold(threshold=0.04).act(obs), PM)

    def test_a_full_buffer_rides_out_the_same_qber_on_pm(self):
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        self.assertEqual(p.act(obs_with(qber=0.035, buffer=0.90)), PM)

    def test_above_the_reserve_it_is_exactly_the_plain_threshold(self):
        # `reserve` is a floor, not a slope that keeps acting above it: with
        # stock in hand there is no gap to pre-empt.
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        plain = GreedyThreshold(threshold=0.04)
        for buffer in (0.5, 0.7, 1.0):
            for qber in (0.0, 0.02, 0.039, 0.04, 0.06):
                obs = obs_with(qber=qber, buffer=buffer)
                self.assertEqual(p.act(obs), plain.act(obs),
                                 "buffer=%s qber=%s" % (buffer, qber))

    def test_a_thinner_buffer_never_switches_later_than_a_fatter_one(self):
        # The whole claim of this policy is monotone in the buffer. An
        # implementation that tightened the wrong way, or one that ignored the
        # buffer entirely, breaks the strict part of this.
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        levels = np.linspace(0.0, 1.0, 21)
        actions = [p.act(obs_with(qber=0.035, buffer=b)) for b in levels]
        # ENT first, PM later, and never back again.
        self.assertEqual(actions, sorted(actions, reverse=True))
        self.assertIn(ENT, actions)              # it really does switch early
        self.assertIn(PM, actions)               # and really does hold off

    def test_an_empty_buffer_halves_the_threshold(self):
        # Pins the FLOOR of the taper, not just its direction. Without this a
        # floor of 0.85 (or 1.0) passes every other test here while collapsing
        # this policy into GreedyThreshold -- and this policy is the bar the
        # Task 6 agent has to clear, so a silently flattened baseline would
        # manufacture a positive result for the study.
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        self.assertAlmostEqual(p.effective_threshold(0.0), 0.02, places=12)

    def test_the_taper_is_linear_between_empty_and_the_reserve(self):
        # Pins the SHAPE. Halfway to the reserve is halfway between the floor
        # (0.5 x threshold) and the nominal threshold, i.e. 0.75 x threshold.
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        for fullness, expected in ((0.0, 0.020),      # 0.50 x threshold
                                   (0.125, 0.025),    # 0.625 x
                                   (0.25, 0.030),     # 0.75  x
                                   (0.375, 0.035),    # 0.875 x
                                   (0.5, 0.040)):     # 1.00  x, at the reserve
            self.assertAlmostEqual(p.effective_threshold(fullness), expected,
                                   places=12, msg="fullness=%s" % fullness)

    def test_the_floor_shows_up_in_the_actions_too(self):
        # The same floor, observed through act() rather than the helper, so a
        # taper that is right in effective_threshold but unused by act() -- or
        # a floor raised toward 1.0 -- is caught either way.
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        # Just above half the nominal threshold: switches only at the floor.
        self.assertEqual(p.act(obs_with(qber=0.021, buffer=0.0)), ENT)
        # Just below it: even a bone-dry buffer holds on to PM.
        self.assertEqual(p.act(obs_with(qber=0.019, buffer=0.0)), PM)
        # A floor of 0.85 would put the empty-buffer switch point at 0.034,
        # so the 0.021 case above would read PM.

    def test_a_dying_link_switches_however_full_the_buffer(self):
        # Being buffer-aware must not become an excuse to sit on a dead link.
        p = BufferAwareThreshold(threshold=0.04, reserve=0.5)
        for buffer in (0.0, 0.5, 1.0):
            self.assertEqual(p.act(obs_with(qber=0.30, buffer=buffer)), ENT)

    def test_an_empty_reserve_degenerates_to_the_plain_threshold(self):
        # reserve=0 is a legal sweep point in Task 7 and must not divide by 0.
        p = BufferAwareThreshold(threshold=0.04, reserve=0.0)
        plain = GreedyThreshold(threshold=0.04)
        for buffer in (0.0, 0.5, 1.0):
            for qber in (0.0, 0.035, 0.05):
                obs = obs_with(qber=qber, buffer=buffer)
                self.assertEqual(p.act(obs), plain.act(obs))

    def test_it_reads_the_buffer_slot_a_real_environment_writes(self):
        env = HybridEnv(HybridParams(horizon=5, seed=1),
                        EveParams(f_levels=(0.0, 0.0, 0.0), seed=1))
        obs = env.reset()
        self.assertEqual(obs[I_BUFFER], 0.0)     # cold start, no key banked
        # Below PM's death QBER, so the plain threshold would stay on PM.
        obs[I_QBER] = 0.035
        self.assertEqual(GreedyThreshold(threshold=0.04).act(obs), PM)
        self.assertEqual(BufferAwareThreshold(0.04, 0.5).act(obs), ENT)


class TestRangeThreshold(unittest.TestCase):
    """PM close in, ENT far out. The SENSE of the comparison is the whole
    policy, so it is pinned from both sides."""

    def test_it_uses_pm_when_the_vessel_is_close(self):
        p = RangeThreshold(170.0)
        for d in (0.0, 55.0, 120.0, 169.9):
            self.assertEqual(p.act(obs_at_distance(d)), PM,
                             "should be PM at %.1f km" % d)

    def test_it_uses_ent_when_the_vessel_is_far(self):
        # MUTATION: flip the comparison, so the vessel uses ENT close in and
        # PM out past PM's death distance. Either half of this pair alone is
        # satisfied by a policy that returns one constant, so both are needed.
        p = RangeThreshold(170.0)
        for d in (170.1, 200.0, 249.0):
            self.assertEqual(p.act(obs_at_distance(d)), ENT,
                             "should be ENT at %.1f km" % d)

    def test_the_boundary_itself_belongs_to_ent(self):
        # Same convention as GreedyThreshold, for the same reason: the natural
        # setting is PM's death distance, at which PM already distils nothing.
        self.assertEqual(RangeThreshold(170.0).act(obs_at_distance(170.0)),
                         ENT)

    def test_it_reads_kilometres_not_the_normalised_feature(self):
        # MUTATION: compare obs[I_DIST] against d_switch without undoing the
        # scaling. Every distance in the study is then below every threshold
        # and the policy silently degenerates to AlwaysPM.
        self.assertEqual(RangeThreshold(0.5).act(obs_at_distance(100.0)), ENT)
        self.assertEqual(RangeThreshold(300.0).act(obs_at_distance(240.0)), PM)

    def test_it_ignores_the_qber(self):
        # Deliberate: paired with RangeOrQber it separates "what is knowing
        # the range worth" from "what does the inferred signal add on top".
        p = RangeThreshold(170.0)
        loud = obs_at_distance(120.0)
        loud[I_QBER] = 0.49
        self.assertEqual(p.act(obs_at_distance(120.0)), p.act(loud))

    def test_the_switch_distance_is_in_the_name(self):
        self.assertEqual(RangeThreshold(167.25).name, "range-167.2")
        self.assertNotEqual(RangeThreshold(150.0).name,
                            RangeThreshold(190.0).name)


class TestRangeOrQber(unittest.TestCase):
    def test_either_signal_alone_fires_it(self):
        p = RangeOrQber(170.0, 0.04)
        near_loud = obs_at_distance(120.0)
        near_loud[I_QBER] = 0.05
        self.assertEqual(p.act(obs_at_distance(180.0)), ENT)
        self.assertEqual(p.act(near_loud), ENT)

    def test_neither_signal_leaves_it_on_pm(self):
        # MUTATION: `and` for `or`, or a bare `return ENT`. Without this the
        # assertions above pass on a policy that never chooses PM at all.
        p = RangeOrQber(170.0, 0.04)
        near_quiet = obs_at_distance(120.0)
        near_quiet[I_QBER] = 0.01
        self.assertEqual(p.act(near_quiet), PM)

    def test_both_signals_together_still_fire_it(self):
        p = RangeOrQber(170.0, 0.04)
        far_loud = obs_at_distance(200.0)
        far_loud[I_QBER] = 0.05
        self.assertEqual(p.act(far_loud), ENT)

    def test_an_unreachable_distance_makes_it_exactly_greedy_threshold(self):
        # This is what guarantees the swept RangeOrQber row can never score
        # BELOW the swept greedy row because of the search: the QBER-only
        # policy is a point of its own grid.
        far_off = RangeOrQber(1e9, 0.035)
        greedy = GreedyThreshold(0.035)
        for d in (0.0, 55.0, 170.0, 240.0):
            for q in (0.0, 0.0349, 0.035, 0.2):
                obs = obs_at_distance(d)
                obs[I_QBER] = q
                self.assertEqual(far_off.act(obs), greedy.act(obs))

    def test_a_reachable_distance_is_not_greedy_threshold(self):
        # ...and that equivalence is a property of the SETTING, not of the
        # policy quietly ignoring its distance argument.
        obs = obs_at_distance(200.0)
        obs[I_QBER] = 0.0
        self.assertEqual(RangeOrQber(170.0, 0.035).act(obs), ENT)
        self.assertEqual(GreedyThreshold(0.035).act(obs), PM)

    def test_both_coordinates_are_in_the_name(self):
        self.assertEqual(RangeOrQber(167.25, 0.0355).name,
                         "range-qber-167.2-0.035")
        names = {RangeOrQber(d, t).name
                 for d in (150.0, 190.0) for t in (0.02, 0.05)}
        self.assertEqual(len(names), 4)


class TestObservedDistance(unittest.TestCase):
    def test_it_undoes_the_normalisation(self):
        self.assertAlmostEqual(observed_distance_km(obs_at_distance(137.5)),
                               137.5)

    def test_both_range_policies_go_through_it(self):
        # One place knows the scale, so a change to DIST_REF_KM cannot leave
        # one policy reading kilometres and the other a fraction.
        obs = obs_at_distance(200.0)
        self.assertEqual(RangeThreshold(199.0).act(obs), ENT)
        self.assertEqual(RangeOrQber(199.0, 1.0).act(obs), ENT)

    def test_it_tracks_a_real_environment(self):
        # The helper above builds observations by hand; this checks the
        # conversion against one the environment actually produced.
        env = HybridEnv(HybridParams(horizon=3, d_km=123.0))
        self.assertAlmostEqual(observed_distance_km(env.reset()), 123.0)

class TestBlockLatencyIsWhatMakesTimingMatter(unittest.TestCase):
    """Spec 2.1: ENT delivers nothing until a whole block completes, so
    switching with an empty buffer starves the queue while switching with a
    stocked buffer does not. If this property ever disappears, the buffer
    features in the observation become decoration and the problem collapses
    to a QBER threshold."""

    def _starved_fraction(self, prefill_slots):
        # Run PM for `prefill_slots` to build stock, then force ENT forever.
        heavy = EveParams(f_levels=(0.0, 0.0, 0.0), seed=1)
        env = HybridEnv(HybridParams(horizon=400, seed=1), heavy)
        obs = env.reset()
        starved = 0
        for t in range(400):
            action = PM if t < prefill_slots else ENT
            obs, _, done, info = env.step(action)
            if t >= prefill_slots and info["buffer"] < env.params.l0:
                starved += 1
            if done:
                break
        return starved

    def test_switching_cold_starves_the_queue(self):
        self.assertGreater(self._starved_fraction(prefill_slots=0), 0)

    def test_switching_with_stock_does_not(self):
        # 200 unjammed PM slots at ~23k *secret* bits/slot (kappa_PM; the
        # sifted rate is ~69k/slot, which is NOT what the buffer banks --
        # hybridenv.step deposits _key_length(), not the sifted count) is far
        # more than the ENT block gap needs, so the buffer carries the queue.
        self.assertEqual(self._starved_fraction(prefill_slots=200), 0)


class _Alternator(Policy):
    """Test double: flips mode every slot, so the switch count is known."""

    name = "alternator"

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.calls = 0

    def act(self, obs):
        action = PM if self.calls % 2 == 0 else ENT
        self.calls += 1
        return action


class TestRunEpisode(unittest.TestCase):
    def test_it_reports_eta_and_the_switch_count(self):
        env = HybridEnv(HybridParams(horizon=100, seed=1))
        out = run_episode(env, AlwaysPM())
        self.assertIn("eta", out)
        self.assertIn("switches", out)
        self.assertEqual(out["switches"], 0)
        self.assertGreaterEqual(out["eta"], 0.0)
        self.assertLessEqual(out["eta"], 1.0)

    def test_starting_on_ent_is_not_counted_as_a_switch(self):
        # The first slot has no previous mode. A runner that seeded its
        # bookkeeping with PM would report a phantom switch here while still
        # reporting 0 for AlwaysPM, so the AlwaysPM case alone cannot catch it.
        env = HybridEnv(HybridParams(horizon=100, seed=1))
        self.assertEqual(run_episode(env, AlwaysENT())["switches"], 0)

    def test_it_counts_one_switch_per_change_of_mode(self):
        # 20 slots alternating PM/ENT/PM/... = 19 boundaries, not 20 and not 18.
        env = HybridEnv(HybridParams(horizon=20, seed=1))
        self.assertEqual(run_episode(env, _Alternator())["switches"], 19)

    def test_the_same_seed_gives_the_same_eta(self):
        def once():
            return run_episode(HybridEnv(HybridParams(horizon=100, seed=9)),
                               AlwaysPM())["eta"]
        self.assertEqual(once(), once())

    def test_no_seed_replays_one_trajectory_on_the_same_env(self):
        # The default must stay byte-identical: Task 7 reuses a single env to
        # score several policies on the SAME jamming, which only holds if the
        # bare call keeps env.reset()'s no-argument semantics.
        env = HybridEnv(HybridParams(horizon=100, seed=3))
        self.assertEqual(run_episode(env, AlwaysENT())["eta"],
                         run_episode(env, AlwaysENT())["eta"])

    def test_a_seed_draws_a_different_episode_on_the_same_env(self):
        # Task 6 trains on one env instance (rebuilding HybridEnv re-runs the
        # expensive channel derivation) and must see a fresh trajectory per
        # episode, so the seed has to reach env.reset().
        env = HybridEnv(HybridParams(horizon=100, seed=3))
        etas = {run_episode(env, AlwaysENT(), seed=s)["eta"]
                for s in range(6)}
        self.assertGreater(len(etas), 1)

    def test_the_same_explicit_seed_replays_the_same_episode(self):
        env = HybridEnv(HybridParams(horizon=100, seed=3))
        self.assertEqual(run_episode(env, AlwaysENT(), seed=11)["eta"],
                         run_episode(env, AlwaysENT(), seed=11)["eta"])

    def test_the_seed_reaches_the_environment_not_just_the_policy(self):
        # An implementation that accepted `seed` and dropped it would still
        # pass the replay tests above. Compare against reset(seed) driven by
        # hand: the runner must produce exactly that trajectory's eta.
        env = HybridEnv(HybridParams(horizon=100, seed=3))
        env.reset(11)
        done = False
        while not done:
            _, _, done, _ = env.step(ENT)
        expected = env.eta
        self.assertEqual(run_episode(env, AlwaysENT(), seed=11)["eta"],
                         expected)
        # ... and the bare call must NOT reproduce it (different stream).
        self.assertNotEqual(run_episode(env, AlwaysENT())["eta"], expected)

    def test_a_switching_policy_records_its_switches(self):
        heavy = EveParams(f_levels=(0.08, 0.08, 0.08), seed=1)
        env = HybridEnv(HybridParams(horizon=300, seed=1), heavy)
        out = run_episode(env, GreedyThreshold(threshold=0.04))
        self.assertGreater(out["switches"], 0)
        # Exactly one: slot 0 has no QBER history yet so it opens on PM, and
        # from slot 1 the jammed estimate never falls back under 4%.
        self.assertEqual(out["switches"], 1)

    def test_eta_is_arrivals_served_not_slots_and_not_the_expiry_complement(self):
        # AlwaysENT at this channel leaves messages both expired AND still
        # queued at the horizon, so the three candidate formulas separate:
        # served/arrived = 9/93, served/horizon = 9/100, 1 - expired/arrived
        # = 1 - 61/93. Only the first is eq (24).
        env = HybridEnv(HybridParams(horizon=100, seed=1))
        out = run_episode(env, AlwaysENT())
        self.assertEqual(out["eta"], env.served_total / env.arrived_total)
        self.assertGreater(out["expired"], 0)
        self.assertNotAlmostEqual(out["eta"],
                                  env.served_total / env.params.horizon)
        self.assertNotAlmostEqual(out["eta"],
                                  1.0 - out["expired"] / env.arrived_total)

    def test_it_resets_the_policy_before_the_episode(self):
        # Task 6's agent carries per-episode state; a runner that skipped
        # reset() would leak the previous episode into the next one.
        env = HybridEnv(HybridParams(horizon=20, seed=1))
        policy = _Alternator()
        policy.calls = 7                       # dirt from a previous episode
        out = run_episode(env, policy)
        self.assertEqual(policy.resets, 1)
        self.assertEqual(out["switches"], 19)  # 19 only if the count restarted

    def test_the_diagnostics_describe_the_run_they_came_from(self):
        env = HybridEnv(HybridParams(horizon=100, seed=1))
        pm = run_episode(env, AlwaysPM())
        env = HybridEnv(HybridParams(horizon=100, seed=1))
        ent = run_episode(env, AlwaysENT())
        self.assertEqual(pm["ent_fraction"], 0.0)
        self.assertEqual(ent["ent_fraction"], 1.0)
        self.assertEqual(pm["policy"], AlwaysPM().name)
        self.assertEqual(ent["policy"], AlwaysENT().name)
        # PM at this channel banks key fast and never starves; ENT does both
        # the other way round. Constant diagnostics would fail this.
        self.assertGreater(pm["secret_bits"], ent["secret_bits"])
        self.assertGreater(pm["secret_bits"], 0)
        self.assertLess(pm["starved_fraction"], ent["starved_fraction"])
        self.assertEqual(pm["expired"], 0)
        self.assertGreater(ent["expired"], 0)


if __name__ == "__main__":
    unittest.main()


class RelaxedAlwaysENT(RelaxationBound):
    """AlwaysENT under the relaxation. Not shipped -- the study reports the
    PM control only -- but the Oracle has to beat BOTH constant modes to be
    a per-slot argmax rather than the better of the two."""

    name = "always-ent-relaxed"

    def choose(self, f, rates, obs):
        return ENT


# The regime the whole study lives in: QUIET most of the time, with HEAVY
# episodes above PM's death QBER and below ENT's.
MIXED_EVE = {"f_levels": (0.0, 0.03, 0.08)}
HEAVY_EVE = {"f_levels": (0.08, 0.08, 0.08)}


def mixed_env(horizon=400, seed=1, **over):
    params = HybridParams(horizon=horizon, seed=seed, lam=2.5,
                          b_max=1_000_000, **over)
    return HybridEnv(params, EveParams(seed=seed, **MIXED_EVE))


class TestTheOracleBound(unittest.TestCase):
    """Spec section 7's mandatory upper bound.

    The study's headline is NEGATIVE -- no switching policy beats AlwaysPM --
    and a negative result is only publishable with a bound on how much value
    switching could ever have had. These tests exist because a bound that is
    silently not a bound (not clairvoyant, paying a switch cost, delivering
    in blocks, or simply AlwaysPM renamed) still prints a plausible number
    and would understate exactly the quantity the paper turns on.
    """

    def test_the_oracle_takes_the_better_mode_slot_by_slot(self):
        oracle = Oracle()
        obs = obs_with()
        self.assertEqual(oracle.choose(0.0, (23332.0, 1137.0), obs), PM)
        self.assertEqual(oracle.choose(0.08, (0.0, 271.2), obs), ENT)
        # Both modes dead: switching buys nothing, so do not report a switch.
        self.assertEqual(oracle.choose(0.5, (0.0, 0.0), obs), PM)

    def test_the_relaxed_control_never_leaves_pm(self):
        control = RelaxedAlwaysPM()
        self.assertEqual(control.choose(0.08, (0.0, 271.2), obs_with()), PM)

    def test_the_oracle_switches_with_the_jamming(self):
        # Kills an "Oracle" that is really max(AlwaysPM, AlwaysENT): that one
        # reports zero switches and an ENT fraction of exactly 0 or 1.
        result = run_relaxed_episode(mixed_env(), Oracle(), seed=1)
        self.assertGreater(result["switches"], 0)
        self.assertGreater(result["ent_fraction"], 0.0)
        self.assertLess(result["ent_fraction"], 1.0)

    def test_the_oracle_out_distils_both_constant_modes_under_the_relaxation(self):
        # The same point, on the quantity the choice actually controls: a
        # per-slot argmax banks STRICTLY more key than either constant mode,
        # while "the better of the two" ties with one of them.
        env = mixed_env()
        oracle = run_relaxed_episode(env, Oracle(), seed=1)["secret_bits"]
        pm = run_relaxed_episode(env, RelaxedAlwaysPM(), seed=1)["secret_bits"]
        ent = run_relaxed_episode(env, RelaxedAlwaysENT(), seed=1)["secret_bits"]
        self.assertGreater(oracle, pm)
        self.assertGreater(oracle, ent)

    def test_the_oracle_acts_on_the_true_f_and_not_on_the_observed_qber(self):
        # Clairvoyance, made testable. In the first slot the QBER window is
        # EMPTY, so the observation reads q = 0 and any policy inferring the
        # channel from it starts on PM. Under permanent heavy jamming PM
        # distils nothing, and the Oracle must be on ENT from slot one.
        env = HybridEnv(HybridParams(horizon=5, seed=1, lam=0.0),
                        EveParams(seed=1, **HEAVY_EVE))
        env.reset()
        oracle = Oracle()
        done, slots = False, 0
        while not done:
            _, _, done, info = env.step_relaxed(oracle.choose)
            slots += 1
            self.assertEqual(info["mode"], ENT,
                             "slot %d fell back to PM" % slots)
            self.assertGreater(info["deposited"], 0.0)
        self.assertEqual(slots, 5)

    def test_a_bound_is_not_a_policy_and_cannot_be_run_as_one(self):
        # run_episode() would score it under the UNRELAXED dynamics and
        # produce a number that is neither the bound nor a policy.
        with self.assertRaises(AttributeError):
            run_episode(mixed_env(horizon=5), Oracle(), seed=1)

    def test_the_two_bounds_are_distinct_rows(self):
        self.assertNotEqual(Oracle.name, RelaxedAlwaysPM.name)
        env = mixed_env()
        oracle = run_relaxed_episode(env, Oracle(), seed=1)
        control = run_relaxed_episode(env, RelaxedAlwaysPM(), seed=1)
        self.assertNotEqual(oracle["ent_fraction"], control["ent_fraction"])
        self.assertNotEqual(oracle["secret_bits"], control["secret_bits"])

    def test_the_oracle_is_at_least_as_good_as_every_policy_on_every_seed(self):
        # Spec section 9's mandated property. If it ever fails, either the
        # environment leaks the future to a policy or the "bound" is not one.
        #
        # Two operating points, and the second is what gives this test teeth:
        # under permanent heavy jamming AlwaysPM scores 0 while AlwaysENT
        # does not, so an "Oracle" that is AlwaysPM under another name FAILS
        # here. Checked per seed, never on the mean -- a bound that lost on
        # one seed and won on another would still average out.
        policies = [AlwaysPM(), AlwaysENT(), GreedyThreshold(0.02),
                    GreedyThreshold(0.045), BufferAwareThreshold(0.045, 0.25)]
        for eve_kwargs in (MIXED_EVE, HEAVY_EVE):
            for seed in (11, 12, 13):
                params = HybridParams(horizon=200, seed=seed, lam=2.5,
                                      b_max=1_000_000)
                env = HybridEnv(params, EveParams(seed=seed, **eve_kwargs))
                bound = run_relaxed_episode(env, Oracle(), seed=seed)["eta"]
                for policy in policies:
                    got = run_episode(env, policy, seed=seed)["eta"]
                    self.assertGreaterEqual(
                        bound, got,
                        "%s scored %.6f above the oracle's %.6f (eve %s, "
                        "seed %d)" % (policy.name, got, bound,
                                      eve_kwargs["f_levels"], seed))

    def test_that_property_test_can_actually_fail(self):
        # The guard on the guard. Under permanent heavy jamming AlwaysENT
        # really does beat AlwaysPM, which is what makes the test above
        # capable of rejecting an Oracle that never leaves PM.
        params = HybridParams(horizon=200, seed=11, lam=2.5, b_max=1_000_000)
        env = HybridEnv(params, EveParams(seed=11, **HEAVY_EVE))
        pm = run_episode(env, AlwaysPM(), seed=11)["eta"]
        ent = run_episode(env, AlwaysENT(), seed=11)["eta"]
        self.assertEqual(pm, 0.0)
        self.assertGreater(ent, 0.0)


# ---------------------------------------------------------------------------
# The block-aware rule and its model-free cousin.
#
# GreedyThreshold compares the two modes and ignores what a switch DESTROYS:
# eq (5) throws the partial block away. At chapman_lowmu that is not academic
# -- greedy-0.045 switches 30 times and scores BELOW always-pm. These tests
# pin the arithmetic that stops that happening, so a future edit cannot
# quietly drop the cost term and still look like a switching policy.
# ---------------------------------------------------------------------------


class _RateModel:
    """The hardware model a BlockAwareGreedy is allowed to read.

    Exactly HybridEnv's kappa(mode, f), error_floor, sifted_per_slot and
    params.n_x -- nothing else, which is the point: every one of those is
    known at design time from the declared hardware, so a policy reading them
    is realisable. The true f_t and the future are NOT here.

    Hand-built rather than a real HybridEnv so a test can place the two modes
    at rates that make the switching arithmetic visible;
    TestBlockAwareGreedyOnARealEnvironment runs the rule against the real one.
    """

    def __init__(self, rates, floors=(0.01, 0.02), sifted=(1000, 1000),
                 n_x=10_000, t0_deadline=20, w_window=5):
        # rates is either (kappa_PM, kappa_ENT) or a callable (mode, f).
        self._rates = rates
        self.error_floor = tuple(floors)
        self.sifted_per_slot = tuple(sifted)
        self.params = HybridParams(n_x=int(n_x), t0_deadline=int(t0_deadline),
                                   w_window=int(w_window))

    def kappa(self, mode, f):
        if callable(self._rates):
            return float(self._rates(mode, f))
        return float(self._rates[mode])


class TestBindingAPolicyToItsEnvironment(unittest.TestCase):
    """A model-based policy needs the environment it is being scored in.

    eval_hybrid._score runs ONE policy object over every seed's env, so a
    policy that latched a model at construction would read the first env's
    distance while running in the last env's episode.
    """

    def test_the_base_policy_ignores_a_bind(self):
        # Every existing baseline must keep working unchanged.
        policy = AlwaysPM()
        policy.bind(_RateModel((1.0, 2.0)))
        self.assertEqual(policy.act(obs_with()), PM)

    def test_it_refuses_to_act_before_it_is_bound(self):
        with self.assertRaises(RuntimeError):
            BlockAwareGreedy().act(obs_with())

    def test_run_episode_binds_it_to_the_environment_it_runs_in(self):
        env = HybridEnv(HybridParams(horizon=5))
        policy = BlockAwareGreedy()
        run_episode(env, policy, seed=1)
        self.assertIs(policy.model, env)

    def test_one_object_rebinds_between_environments(self):
        # The failure this prevents: a policy answering for the previous
        # seed's hardware. Two models, opposite verdicts, one policy object.
        policy = BlockAwareGreedy(horizon=10)
        here = obs_with(mode=PM, block=0.0)
        policy.bind(_RateModel((1.0, 5.0)))
        self.assertEqual(policy.act(here), ENT)
        policy.bind(_RateModel((5.0, 1.0)))
        self.assertEqual(policy.act(here), PM)


class TestBlockAwareGreedyEstimatesTheJamming(unittest.TestCase):
    """f is unobservable; the windowed QBER is not. E_eff(f) is invertible."""

    def test_it_inverts_the_effective_qber(self):
        policy = BlockAwareGreedy(f_step=0.0)
        policy.bind(_RateModel((1.0, 1.0), floors=(0.01, 0.02)))
        # E_eff = e(1-f) + 0.5 f, with e = 0.01 and f = 0.2.
        q = 0.01 * 0.8 + 0.5 * 0.2
        self.assertAlmostEqual(policy.estimated_jamming(
            obs_with(qber=q, mode=PM)), 0.2, places=9)

    def test_it_uses_the_floor_of_the_mode_actually_running(self):
        # The same measured QBER means LESS jamming under the noisier mode.
        policy = BlockAwareGreedy(f_step=0.0)
        policy.bind(_RateModel((1.0, 1.0), floors=(0.01, 0.20)))
        q = 0.10
        self.assertGreater(
            policy.estimated_jamming(obs_with(qber=q, mode=PM)),
            policy.estimated_jamming(obs_with(qber=q, mode=ENT)))

    def test_a_qber_below_the_floor_estimates_no_jamming(self):
        # A lucky binomial draw must not produce a negative fraction and a
        # kappa lookup outside the model's domain.
        policy = BlockAwareGreedy(f_step=0.0)
        policy.bind(_RateModel((1.0, 1.0), floors=(0.05, 0.05)))
        self.assertEqual(policy.estimated_jamming(obs_with(qber=0.0)), 0.0)

    def test_a_dead_link_estimates_full_jamming_and_no_more(self):
        policy = BlockAwareGreedy(f_step=0.0)
        policy.bind(_RateModel((1.0, 1.0), floors=(0.01, 0.01)))
        self.assertEqual(policy.estimated_jamming(obs_with(qber=0.9)), 1.0)

    def test_it_snaps_the_estimate_to_a_grid(self):
        # Every distinct f costs a finite-key evaluation in HybridEnv.kappa;
        # an unquantised estimate would miss the memo on all 3,600 slots.
        policy = BlockAwareGreedy(f_step=0.01)
        policy.bind(_RateModel((1.0, 1.0), floors=(0.0, 0.0)))
        self.assertAlmostEqual(
            policy.estimated_jamming(obs_with(qber=0.5 * 0.2049)), 0.20,
            places=9)

    def test_the_grid_keeps_the_number_of_distinct_estimates_small(self):
        policy = BlockAwareGreedy(f_step=0.01)
        policy.bind(_RateModel((1.0, 1.0), floors=(0.0, 0.0)))
        seen = {policy.estimated_jamming(obs_with(qber=0.5 * i / 997.0))
                for i in range(998)}
        self.assertLessEqual(len(seen), 101)


class TestBlockAwareGreedyWeighsTheForfeitedBlock(unittest.TestCase):
    """Whole blocks retired inside the horizon H, staying versus switching:

        stay = l_m(f)  * floor((b + ns_m  * H) / n_X)
        go   = l_m'(f) * floor(     ns_m' * H  / n_X)      switch <=> go > stay

    Counted in BLOCKS, not in a continuous rate, and that is not a detail:
    continuous delivery is relaxation 3 of the Oracle bound, so a policy that
    priced its own decisions that way would be quietly borrowing the
    relaxation it is supposed to sit below. Measured, it also matters --
    at chapman_lowmu the continuous form switches 48 times to a mode that
    never finishes a block and lands BELOW always-pm.

    Every model here is 10,000 sifted bits to a block and 1,000 bits a slot,
    so a mode retires one whole block every ten slots and H = 19 buys exactly
    one -- which is what lets a half-filled block tip the count.
    """

    H = 19

    def model(self, rates, **kw):
        kw.setdefault("floors", (0.0, 0.0))
        kw.setdefault("sifted", (1000, 1000))
        kw.setdefault("n_x", 10_000)
        # No settling by default: these tests are about the switching
        # arithmetic, and a dwell would mask it.
        kw.setdefault("w_window", 0)
        return _RateModel(rates, **kw)

    def policy(self, rates, horizon=None, **kw):
        policy = BlockAwareGreedy(horizon=self.H if horizon is None else horizon)
        policy.bind(self.model(rates, **kw))
        return policy

    def test_it_switches_when_the_gain_outruns_the_forfeited_block(self):
        # b = 0.5 blocks: staying retires floor(0.5 + 1.9) = 2 blocks of 10
        # bits; switching retires floor(1.9) = 1 block of 30.
        policy = self.policy((1.0, 3.0))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.5)), ENT)

    def test_it_stays_when_the_forfeited_block_outweighs_the_gain(self):
        # THE DEFECT THIS POLICY EXISTS TO FIX. ENT is ahead on rate, so
        # GreedyThreshold switches; here staying retires 2 blocks x 10 bits
        # against switching's 1 x 15.
        policy = self.policy((1.0, 1.5))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.5)), PM)

    def test_the_same_slot_switches_once_the_block_is_cheap_enough(self):
        # Same rates, same horizon: only the sunk block differs, which is
        # what makes the forfeited work load-bearing rather than decorative.
        # floor(0.05 + 1.9) = 1 block either way, so the rates decide.
        policy = self.policy((1.0, 1.5))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.05)), ENT)

    def test_a_block_just_started_costs_nothing_to_abandon(self):
        policy = self.policy((1.0, 1.0 + 1e-9))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.0)), ENT)

    def test_it_never_switches_to_a_mode_that_earns_less(self):
        policy = self.policy((2.0, 1.0))
        for block in (0.0, 0.5, 0.99):
            self.assertEqual(policy.act(obs_with(mode=PM, block=block)), PM)

    def test_it_will_not_move_to_a_mode_that_cannot_finish_a_block_in_time(self):
        # THE SECOND DEFECT. ENT here is ten times slower, so in H = 19 slots
        # it retires floor(0.19) = 0 blocks and delivers NOTHING inside the
        # horizon however rich its rate looks. This is chapman_lowmu exactly:
        # 3,485 sifted bits a slot against a block of 100,000.
        policy = self.policy((0.0, 5.0), sifted=(1000, 100))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.5)), PM)

    def test_a_horizon_long_enough_for_a_block_does_move_to_it(self):
        # Same slow mode, same dead PM: with room for two whole ENT blocks
        # the switch is worth making. The pair is the point -- the rule is
        # about DELIVERY inside the horizon, not about rate.
        policy = self.policy((0.0, 5.0), horizon=200, sifted=(1000, 100))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.5)), ENT)

    def test_a_slow_mode_earns_more_from_each_block_it_does_finish(self):
        # l_m = kappa_m * n_X / ns_m -- a mode that takes ten times as long
        # to fill a block banks ten times as much key when it lands. Here ENT
        # retires 2 blocks to PM's 20 and still wins on total delivery;
        # comparing the per-SLOT rates instead would undercount it tenfold.
        policy = self.policy((1.0, 2.0), horizon=200, sifted=(1000, 100))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.0)), ENT)

    def test_it_leaves_a_dead_mode_at_once_however_full_the_block(self):
        # l_m = 0 makes the forfeited block worth nothing, so a dying link is
        # abandoned immediately -- the one case where a fixed dead band would
        # be actively harmful.
        policy = self.policy((0.0, 0.001))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.999)), ENT)

    def test_it_holds_when_both_modes_are_dead(self):
        # Nothing to gain: a switch here would show up in the tables as
        # thrashing that bought exactly zero bits.
        policy = self.policy((0.0, 0.0), horizon=100)
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.5)), PM)
        self.assertEqual(policy.act(obs_with(mode=ENT, block=0.5)), ENT)

    def test_it_works_the_other_way_round_too(self):
        # Sitting on ENT, PM recovers: same rule, mirrored.
        policy = self.policy((2.0, 1.0))
        self.assertEqual(policy.act(obs_with(mode=ENT, block=0.0)), PM)

    def test_a_longer_horizon_switches_where_a_shorter_one_holds(self):
        # H = 2 buys ENT floor(0.2) = 0 blocks while the sunk 0.8 of a block
        # still tips PM's count to 1; H = 200 buys both of them 20.
        here = obs_with(mode=PM, block=0.8)
        self.assertEqual(self.policy((1.0, 1.5), horizon=2).act(here), PM)
        self.assertEqual(self.policy((1.0, 1.5), horizon=200).act(here), ENT)

    def test_it_holds_a_new_mode_until_the_qber_window_has_flushed(self):
        # THE OSCILLATION THIS SETTLING TIME EXISTS TO STOP. The observed
        # QBER averages the last w_window slots, so for w_window slots after
        # a switch it is a BLEND of the two modes, and inverting that blend
        # against either floor is meaningless. Measured at ent_advanced
        # without it: the estimate reads "no jamming" on ENT (the window is
        # full of PM's 0.0102 against ENT's 0.0458 floor) and "f = 0.073" on
        # PM (the mirror image) -- a guaranteed two-cycle. 2,140 switches in
        # 3,600 slots and eta = 0.12 against always-pm's 0.61.
        policy = BlockAwareGreedy(horizon=self.H, settle=3)
        policy.bind(self.model((1.0, 5.0)))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.0)), ENT)
        # The environment has switched; now the model reverses its verdict.
        policy.bind(self.model((5.0, 1.0)))
        for _ in range(3):
            self.assertEqual(policy.act(obs_with(mode=ENT, block=0.0)), ENT)
        self.assertEqual(policy.act(obs_with(mode=ENT, block=0.0)), PM)

    def test_the_settling_time_does_not_delay_the_opening_move(self):
        # Slot 0 starts on PM by construction, which is not a switch this
        # policy made; holding a dead mode for w_window slots to honour it
        # would be pure loss.
        policy = BlockAwareGreedy(horizon=self.H, settle=5)
        policy.bind(self.model((1.0, 5.0)))
        self.assertEqual(policy.act(obs_with(mode=PM, block=0.0)), ENT)

    def test_the_default_settling_time_is_the_qber_window(self):
        policy = BlockAwareGreedy()
        policy.bind(self.model((1.0, 1.0), w_window=9))
        self.assertEqual(policy.settling_slots(), 9)

    def test_reset_clears_the_settling_clock(self):
        policy = BlockAwareGreedy(horizon=self.H, settle=4)
        policy.bind(self.model((1.0, 5.0)))
        policy.act(obs_with(mode=PM, block=0.0))          # -> ENT
        policy.bind(self.model((5.0, 1.0)))
        self.assertEqual(policy.act(obs_with(mode=ENT, block=0.0)), ENT)
        policy.reset()
        self.assertEqual(policy.act(obs_with(mode=ENT, block=0.0)), PM)

    def test_the_default_horizon_is_the_message_deadline(self):
        policy = BlockAwareGreedy()
        policy.bind(self.model((1.0, 1.0), t0_deadline=37))
        self.assertEqual(policy.planning_horizon(), 37)

    def test_an_explicit_horizon_is_in_the_name(self):
        self.assertEqual(BlockAwareGreedy().name, "block-aware")
        self.assertEqual(BlockAwareGreedy(horizon=40).name, "block-aware-40")
        self.assertNotEqual(BlockAwareGreedy(horizon=5).name,
                            BlockAwareGreedy(horizon=10).name)


class TestBlockAwareGreedyOnARealEnvironment(unittest.TestCase):
    def test_the_environment_supplies_everything_the_rule_reads(self):
        # Catches a rename of kappa/error_floor/sifted_per_slot: the stub
        # model above would happily go on satisfying an obsolete interface.
        env = HybridEnv(HybridParams(horizon=40))
        result = run_episode(env, BlockAwareGreedy(), seed=3)
        self.assertEqual(result["policy"], "block-aware")
        self.assertGreaterEqual(result["eta"], 0.0)
        self.assertLessEqual(result["eta"], 1.0)

    def test_it_does_not_flood_the_kappa_memo(self):
        # 3,600 unquantised estimates would mean 7,200 finite-key
        # evaluations per episode; the grid is what keeps this usable.
        env = HybridEnv(HybridParams(horizon=200))
        run_episode(env, BlockAwareGreedy(f_step=0.01), seed=3)
        self.assertLess(env.kappa_cache_size, 250)


class TestHysteresisThreshold(unittest.TestCase):
    """The model-free anti-thrash control: one threshold up, a lower one back.

    GreedyThreshold has a single threshold, so a QBER dithering around it
    switches on every crossing. This is the classical answer, and it needs no
    hardware model at all -- which is exactly what makes it the honest
    comparand for the derived rule.
    """

    def test_it_moves_to_ent_at_the_upper_threshold(self):
        policy = HysteresisThreshold(0.05, 0.03)
        policy.reset()
        self.assertEqual(policy.act(obs_with(qber=0.05, mode=PM)), ENT)

    def test_it_stays_on_pm_below_the_upper_threshold(self):
        policy = HysteresisThreshold(0.05, 0.03)
        policy.reset()
        self.assertEqual(policy.act(obs_with(qber=0.049, mode=PM)), PM)

    def test_it_holds_ent_inside_the_band(self):
        # The whole point. GreedyThreshold(0.05) would go straight back to PM.
        policy = HysteresisThreshold(0.05, 0.03)
        policy.reset()
        self.assertEqual(policy.act(obs_with(qber=0.04, mode=ENT)), ENT)
        self.assertEqual(GreedyThreshold(0.05).act(obs_with(qber=0.04)), PM)

    def test_it_returns_to_pm_at_the_lower_threshold(self):
        policy = HysteresisThreshold(0.05, 0.03)
        policy.reset()
        self.assertEqual(policy.act(obs_with(qber=0.03, mode=ENT)), PM)

    def test_a_zero_width_band_is_exactly_the_greedy_threshold(self):
        # A legal sweep point, and the one that must reproduce the baseline:
        # otherwise the grid cannot contain the policy it generalises.
        policy = HysteresisThreshold(0.05, 0.05)
        greedy = GreedyThreshold(0.05)
        policy.reset()
        for q, mode in ((0.049, PM), (0.05, PM), (0.06, ENT), (0.04, ENT)):
            self.assertEqual(policy.act(obs_with(qber=q, mode=mode)),
                             greedy.act(obs_with(qber=q)))

    def test_it_refuses_a_band_that_is_upside_down(self):
        with self.assertRaises(ValueError):
            HysteresisThreshold(0.03, 0.05)

    def test_the_dwell_blocks_a_reversal_that_comes_too_soon(self):
        policy = HysteresisThreshold(0.05, 0.03, min_dwell=3)
        policy.reset()
        self.assertEqual(policy.act(obs_with(qber=0.06, mode=PM)), ENT)
        # Now the QBER collapses; the dwell must hold ENT anyway.
        for _ in range(3):
            self.assertEqual(policy.act(obs_with(qber=0.0, mode=ENT)), ENT)
        self.assertEqual(policy.act(obs_with(qber=0.0, mode=ENT)), PM)

    def test_without_a_dwell_the_reversal_is_immediate(self):
        policy = HysteresisThreshold(0.05, 0.03, min_dwell=0)
        policy.reset()
        self.assertEqual(policy.act(obs_with(qber=0.06, mode=PM)), ENT)
        self.assertEqual(policy.act(obs_with(qber=0.0, mode=ENT)), PM)

    def test_the_dwell_clock_restarts_on_every_switch(self):
        policy = HysteresisThreshold(0.05, 0.03, min_dwell=2)
        policy.reset()
        policy.act(obs_with(qber=0.06, mode=PM))          # -> ENT
        policy.act(obs_with(qber=0.06, mode=ENT))
        policy.act(obs_with(qber=0.06, mode=ENT))
        self.assertEqual(policy.act(obs_with(qber=0.0, mode=ENT)), PM)
        # Back on PM the clock is fresh again, so a spike cannot bounce it.
        self.assertEqual(policy.act(obs_with(qber=0.9, mode=PM)), PM)

    def test_reset_clears_the_dwell_between_episodes(self):
        # run_episode resets; a dwell carried across episode boundaries would
        # make seed 2's first slots depend on seed 1's last ones.
        policy = HysteresisThreshold(0.05, 0.03, min_dwell=5)
        policy.reset()
        policy.act(obs_with(qber=0.06, mode=PM))         # -> ENT
        policy.act(obs_with(qber=0.06, mode=ENT))        # the dwell restarts
        # Mid-episode the dwell now blocks a return to PM ...
        self.assertEqual(policy.act(obs_with(qber=0.0, mode=ENT)), ENT)
        policy.reset()
        # ... and the next episode must not inherit it.
        self.assertEqual(policy.act(obs_with(qber=0.0, mode=ENT)), PM)

    def test_both_edges_and_the_dwell_are_in_the_name(self):
        names = {HysteresisThreshold(0.05, 0.03).name,
                 HysteresisThreshold(0.05, 0.04).name,
                 HysteresisThreshold(0.06, 0.03).name,
                 HysteresisThreshold(0.05, 0.03, min_dwell=4).name}
        self.assertEqual(len(names), 4)

    def test_it_runs_a_real_episode(self):
        env = HybridEnv(HybridParams(horizon=40))
        result = run_episode(env, HysteresisThreshold(0.05, 0.03), seed=3)
        self.assertGreaterEqual(result["eta"], 0.0)


class TestDecoyWatch(unittest.TestCase):
    """The controller that can actually see a photon-number-splitting attack.

    Every other policy here reacts to the QBER, and a PNS attacker moves the
    QBER by nothing at all: measured, greedy / hysteresis / block-aware all
    become bit-for-bit always-pm under one. The observable that does move is
    the decoy gain ratio, and this is the policy that watches it.

    The hard part is not the threshold, it is that the observable DISAPPEARS
    on ENT: while ENT runs, the decoy hardware is not transmitting, so
    obs[I_DECOY] holds its last PM reading for ever. A controller that
    switches away on an alarm and never looks again can never come back --
    which is why probing is a parameter and not an afterthought.
    """

    def watch(self, **kw):
        kw.setdefault("decoy_threshold", 0.20)
        policy = DecoyWatch(**kw)
        policy.bind(_RateModel((1.0, 1.0), w_window=3))
        policy.reset()
        return policy

    # ---------------- detection ----------------

    def test_it_stays_on_pm_while_the_decoy_ratio_looks_honest(self):
        policy = self.watch()
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.0)), PM)
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.19)), PM)

    def test_it_switches_on_the_decoy_excess_with_the_qber_at_zero(self):
        # THE point of the policy: an alarm from an observable that the rest
        # of the study does not look at, while the one they do look at is
        # flat.
        policy = self.watch()
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.20, qber=0.0)),
                         ENT)

    def test_the_threshold_itself_counts_as_crossed(self):
        # Same convention as GreedyThreshold, for the same reason.
        self.assertEqual(self.watch(decoy_threshold=0.2).act(
            obs_with(mode=PM, decoy=0.2)), ENT)

    def test_by_default_it_ignores_the_qber_entirely(self):
        # Deliberate: paired with a qber threshold it separates "what does
        # seeing the decoy ratio buy" from "what does the QBER add on top".
        policy = self.watch()
        self.assertEqual(policy.act(obs_with(mode=PM, qber=0.49, decoy=0.0)),
                         PM)

    def test_given_a_qber_threshold_it_reacts_to_jamming_too(self):
        # An adversary with two weapons needs a controller with two
        # observables; either alarm alone is enough, because either can
        # already have killed PM.
        policy = self.watch(qber_threshold=0.045)
        self.assertEqual(policy.act(obs_with(mode=PM, qber=0.05, decoy=0.0)),
                         ENT)

    # ---------------- the stale observable ----------------

    def test_without_probing_it_never_returns_to_pm(self):
        # Not a bug to be fixed silently -- it is the cost of the attack
        # being invisible from ENT, and it is what the probe parameter buys.
        policy = self.watch(probe_every=0)
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.5)), ENT)
        for _ in range(200):
            # The feature is frozen at its last PM reading, alarm and all.
            self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), ENT)

    def test_probing_goes_back_to_look_after_the_interval(self):
        policy = self.watch(probe_every=5)
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.5)), ENT)
        for _ in range(5):
            self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), ENT)
        self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), PM)

    def test_a_probe_holds_pm_until_the_decoy_window_has_flushed(self):
        # The window still holds the readings from before the probe, so the
        # first slots back on PM say whatever the attack used to be doing.
        # Exactly the contamination BlockAwareGreedy's settle exists for.
        policy = self.watch(probe_every=2, settle=3)
        policy.act(obs_with(mode=PM, decoy=0.5))          # -> ENT
        for _ in range(3):
            policy.act(obs_with(mode=ENT, decoy=0.5))     # ... -> PM
        # Now on PM with a stale alarm still in the window: it must NOT
        # bounce straight back to ENT.
        for _ in range(3):
            self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.5)), PM)
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.5)), ENT)

    def test_a_probe_that_finds_the_attack_gone_stays_on_pm(self):
        policy = self.watch(probe_every=2, settle=1)
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.5)), ENT)
        self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), ENT)
        self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), ENT)
        self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), PM)
        policy.act(obs_with(mode=PM, decoy=0.0))          # window flushing
        for _ in range(5):
            self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.0)), PM)

    def test_the_default_settle_is_the_observation_window(self):
        # The controller's own averaging window: a reading taken before it
        # has flushed is a reading of the previous mode.
        policy = DecoyWatch(decoy_threshold=0.2)
        policy.bind(_RateModel((1.0, 1.0), w_window=9))
        self.assertEqual(policy.settling_slots(), 9)

    def test_it_works_unbound_so_it_stays_a_model_FREE_comparand(self):
        # The contrast with BlockAwareGreedy is the whole point: this one
        # needs no kappa, no error floors and no sifted rates. Only its own
        # thresholds.
        policy = DecoyWatch(decoy_threshold=0.2, settle=0)
        policy.reset()
        self.assertEqual(policy.act(obs_with(mode=PM, decoy=0.5)), ENT)

    # ---------------- bookkeeping ----------------

    def test_reset_clears_the_probe_clock(self):
        policy = self.watch(probe_every=3)
        policy.act(obs_with(mode=PM, decoy=0.5))
        policy.act(obs_with(mode=ENT, decoy=0.5))
        policy.act(obs_with(mode=ENT, decoy=0.5))
        policy.reset()
        # A fresh episode must not inherit two thirds of a probe interval.
        for _ in range(3):
            self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), ENT)
        self.assertEqual(policy.act(obs_with(mode=ENT, decoy=0.5)), PM)

    def test_every_parameter_is_in_the_name(self):
        names = {DecoyWatch(0.2).name, DecoyWatch(0.3).name,
                 DecoyWatch(0.2, qber_threshold=0.05).name,
                 DecoyWatch(0.2, probe_every=50).name}
        self.assertEqual(len(names), 4)

    def test_it_refuses_a_negative_probe_interval(self):
        with self.assertRaises(ValueError):
            DecoyWatch(0.2, probe_every=-1)

    def test_it_runs_a_real_episode_and_reads_the_real_feature(self):
        # Catches a policy wired to the wrong observation index, which the
        # hand-built observations above cannot.
        env = HybridEnv(HybridParams(horizon=60),
                        EveParams(f_levels=(0.0, 0.0, 0.0),
                                  s_levels=(0.0, 0.0, 0.6), seed=3))
        result = run_episode(env, DecoyWatch(0.2, probe_every=20), seed=3)
        self.assertGreaterEqual(result["eta"], 0.0)


class TestTheDecoyWatchSeesWhatTheQberPoliciesCannot(unittest.TestCase):
    """The measured claim, as a test: under a stealthy adversary the
    QBER-threshold baselines do not switch at all and this one does."""

    ENV = dict(horizon=1200, lam=2.5, b_max=1_000_000,
               ent_preset="ent_advanced")
    EVE = dict(f_levels=(0.0, 0.0, 0.0), s_levels=(0.0, 0.0, 0.6), seed=3)

    def run_one(self, policy):
        env = HybridEnv(HybridParams(**self.ENV), EveParams(**self.EVE))
        return run_episode(env, policy, seed=5)

    def test_the_qber_baseline_never_fires(self):
        self.assertEqual(self.run_one(GreedyThreshold(0.045))["switches"], 0)

    def test_the_decoy_watch_does_fire(self):
        self.assertGreater(self.run_one(DecoyWatch(0.2))["switches"], 0)

    def test_and_it_delivers_more(self):
        self.assertGreater(self.run_one(DecoyWatch(0.2, probe_every=40))["eta"],
                           self.run_one(GreedyThreshold(0.045))["eta"])
