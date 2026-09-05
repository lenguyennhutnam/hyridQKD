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
from policies import (AlwaysENT, AlwaysPM, BufferAwareThreshold,
                      GreedyThreshold, Oracle, Policy, RangeOrQber,
                      RangeThreshold, RelaxationBound, RelaxedAlwaysPM,
                      observed_distance_km, run_episode, run_relaxed_episode)


def obs_with(qber=0.0, mode=PM, block=0.0, buffer=0.0, queue=0.0, age=0.0,
             d_km=0.0):
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
