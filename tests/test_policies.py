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
from hybridenv import (ENT, I_BUFFER, I_QBER, PM, HybridEnv, HybridParams)
from policies import (AlwaysENT, AlwaysPM, BufferAwareThreshold,
                      GreedyThreshold, Policy, run_episode)


def obs_with(qber=0.0, mode=PM, block=0.0, buffer=0.0, queue=0.0, age=0.0):
    """Build an observation by NAME, using the layout hybridenv owns.

    Positional literals would keep passing if the index constants ever drifted
    apart from hybridenv._observe(); routing through the real constants means
    a drift shows up here.
    """
    v = np.zeros(hybridenv.OBS_DIM)
    v[hybridenv.I_QBER] = qber
    v[hybridenv.I_MODE] = float(mode)
    v[hybridenv.I_BLOCK] = block
    v[hybridenv.I_BUFFER] = buffer
    v[hybridenv.I_QUEUE] = queue
    v[hybridenv.I_AGE] = age
    return v


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
                 BufferAwareThreshold(0.03, 0.8).name]
        self.assertEqual(len(set(names)), len(names))


class TestObservationConstantsAreShared(unittest.TestCase):
    def test_policies_reexports_the_layout_hybridenv_defines(self):
        # hybridenv._observe() fixes the order; policies must not keep a
        # second copy that can drift out of step with it.
        import policies
        for name in ("I_QBER", "I_MODE", "I_BLOCK", "I_BUFFER", "I_QUEUE",
                     "I_AGE"):
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
