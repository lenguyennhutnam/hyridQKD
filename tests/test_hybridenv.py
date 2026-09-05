"""Hybrid PM/ENT environment: eq (1)-(24) of system_model_v3_5."""
import unittest

from evemarkov import EveParams
from hybridenv import (DIST_REF_KM, ENT, I_AGE, I_BLOCK, I_BUFFER, I_DIST,
                       I_MODE, I_QBER, I_QUEUE, OBS_DIM, PM,
                       BlockAccumulator, HybridEnv, HybridParams, KeyBuffer,
                       MessageQueue)

QUIET_EVE = EveParams(f_levels=(0.0, 0.0, 0.0), seed=1)


class TestMessageQueue(unittest.TestCase):
    """FIFO with a common deadline T0 (eq 3-4). All messages expire at the
    same age, so the head of the queue is always the most urgent."""

    def test_serving_takes_from_the_oldest_first(self):
        q = MessageQueue(deadline=5)
        q.arrive(2)
        q.age_and_expire()          # those two are now age 1
        q.arrive(3)
        self.assertEqual(q.serve(2), 2)
        # The two oldest went; the survivors are the three that just arrived.
        self.assertEqual(q.length, 3)
        self.assertEqual(q.head_age, 0)

    def test_serving_more_than_present_serves_what_is_there(self):
        q = MessageQueue(deadline=5)
        q.arrive(3)
        self.assertEqual(q.serve(10), 3)
        self.assertEqual(q.length, 0)

    def test_messages_older_than_the_deadline_are_dropped(self):
        q = MessageQueue(deadline=3)
        q.arrive(4)
        for _ in range(3):
            self.assertEqual(q.age_and_expire(), 0)
        self.assertEqual(q.age_and_expire(), 4)   # age 4 > T0 = 3
        self.assertEqual(q.length, 0)

    def test_serving_crosses_a_bucket_boundary(self):
        # serve() must keep draining into younger buckets once the oldest is
        # empty. An implementation that stops at the first non-empty bucket
        # returns 2 here, silently capping PM's service rate and depressing
        # eta without failing any single-bucket test.
        q = MessageQueue(deadline=5)
        q.arrive(2)
        q.age_and_expire()          # those two are now age 1
        q.arrive(3)
        self.assertEqual(q.serve(4), 4)   # 2 from age 1, then 2 from age 0
        self.assertEqual(q.length, 1)
        self.assertEqual(q.head_age, 0)

    def test_only_the_expiring_cohort_is_dropped(self):
        # Two cohorts of different ages coexist and the older one times out.
        # An age_and_expire() that clears the whole queue is sum-conserving,
        # so eq (4) cannot see it -- only checking the survivors can.
        q = MessageQueue(deadline=3)
        q.arrive(2)                 # cohort A
        q.age_and_expire()          # A is age 1
        q.arrive(5)                 # cohort B, age 0
        self.assertEqual(q.age_and_expire(), 0)   # A age 2, B age 1
        self.assertEqual(q.age_and_expire(), 0)   # A age 3, B age 2
        self.assertEqual(q.age_and_expire(), 2)   # A expires, B age 3
        self.assertEqual(q.length, 5)             # all of B survives
        self.assertEqual(q.head_age, 3)
        self.assertEqual(q.age_and_expire(), 5)   # now B goes too
        self.assertEqual(q.length, 0)

    def test_head_age_tracks_the_oldest_waiting_message(self):
        q = MessageQueue(deadline=10)
        q.arrive(1)
        q.age_and_expire()
        q.age_and_expire()
        q.arrive(1)
        self.assertEqual(q.head_age, 2)

    def test_head_age_of_an_empty_queue_is_zero(self):
        self.assertEqual(MessageQueue(deadline=10).head_age, 0)

    def test_conservation_arrivals_equal_served_plus_expired_plus_left(self):
        # eq (4) as an invariant. A queue that silently loses messages would
        # inflate eta without anyone noticing.
        q = MessageQueue(deadline=4)
        served = expired = arrived = 0
        for t in range(50):
            q.arrive(2)
            arrived += 2
            served += q.serve(1)
            expired += q.age_and_expire()
        self.assertEqual(arrived, served + expired + q.length)


class TestBlockAccumulator(unittest.TestCase):
    """eq (5)-(7). Switching mode discards the partial block -- the intrinsic
    switching cost the paper relies on instead of a penalty constant."""

    def test_staying_in_one_mode_accumulates(self):
        acc = BlockAccumulator(n_x=1000)
        self.assertEqual(acc.add(0, 400, 4), [])
        self.assertEqual(acc.add(0, 400, 4), [])
        self.assertEqual(acc.progress, 800)

    def test_switching_mode_throws_the_partial_block_away(self):
        acc = BlockAccumulator(n_x=1000)
        acc.add(0, 900, 9)
        self.assertEqual(acc.progress, 900)
        acc.add(1, 100, 1)          # switched: the 900 are gone
        self.assertEqual(acc.progress, 100)
        self.assertEqual(acc.mode, 1)

    def test_a_completed_block_reports_its_size_and_errors(self):
        acc = BlockAccumulator(n_x=1000)
        acc.add(0, 600, 12)
        done = acc.add(0, 600, 12)
        self.assertEqual(len(done), 1)
        n, errors = done[0]
        self.assertEqual(n, 1000)
        # 24 errors over 1200 bits; the block keeps its proportional share.
        self.assertEqual(errors, 20)

    def test_the_overshoot_carries_into_the_next_block(self):
        acc = BlockAccumulator(n_x=1000)
        acc.add(0, 600, 12)
        acc.add(0, 600, 12)
        self.assertEqual(acc.progress, 200)     # 1200 - 1000
        self.assertEqual(acc.errors, 4)         # 24 - 20

    def test_a_block_completed_right_after_a_switch_is_pure(self):
        # eq (5)-(6): every bit in a finished block came from one protocol.
        acc = BlockAccumulator(n_x=1000)
        acc.add(0, 999, 10)
        done = acc.add(1, 1000, 50)
        self.assertEqual(done, [(1000, 50)])    # none of PM's errors leaked in

    def test_switching_mode_discards_the_partial_errors_too(self):
        # An implementation that zeroes `progress` but keeps `_errors` looks
        # correct to every progress-only assertion, then attributes the old
        # mode's errors to the new mode's first block -- a QBER, hence a key
        # length, that belongs to neither protocol.
        acc = BlockAccumulator(n_x=1000)
        acc.add(0, 900, 90)
        acc.add(1, 100, 1)
        self.assertEqual(acc.progress, 100)
        self.assertEqual(acc.errors, 1)

    def test_consecutive_blocks_stay_error_conserving(self):
        # Three slots of 700 bits complete two blocks. Errors must be
        # conserved across the hand-over: an implementation that recomputes
        # the leftover from scratch (or drops it) passes the single-block
        # tests and quietly biases every block after the first.
        acc = BlockAccumulator(n_x=1000)
        self.assertEqual(acc.add(0, 700, 70), [])
        first = acc.add(0, 700, 70)
        self.assertEqual(first, [(1000, 100)])  # 140 * 1000/1400
        self.assertEqual(acc.progress, 400)
        self.assertEqual(acc.errors, 40)
        second = acc.add(0, 700, 70)
        self.assertEqual(second, [(1000, 100)])  # 110 * 1000/1100 = 100
        self.assertEqual(acc.progress, 100)
        self.assertEqual(acc.errors, 10)

    def test_one_slot_can_retire_several_blocks(self):
        # A slot's harvest is not bounded by n_x, so add() must drain the pile
        # until it is short of a block. Retiring only the first (or collapsing
        # the surplus with progress %= n_x) destroys whole blocks of finished
        # key, and leaves progress/n_x far outside [0, 1).
        acc = BlockAccumulator(n_x=1000)
        done = acc.add(0, 3500, 350)
        self.assertEqual(done, [(1000, 100), (1000, 100), (1000, 100)])
        self.assertEqual(acc.progress, 500)     # the true remainder
        self.assertEqual(acc.errors, 50)

    def test_a_realistic_pm_slot_retires_every_block_it_earned(self):
        # The CRITICAL case: at d_km=0 PM harvests ~853,699 sifted per slot
        # against n_X = 100,000. Eight blocks are earned, so eight must come
        # back -- delivering one threw away ~88% of the key.
        acc = BlockAccumulator(n_x=100000)
        done = acc.add(0, 853699, 8537)         # ~1% QBER
        self.assertEqual(len(done), 8)
        self.assertEqual([n for n, _ in done], [100000] * 8)
        self.assertEqual(acc.progress, 53699)   # 853699 - 8 * 100000
        # Errors are conserved: what left in blocks plus what stayed behind is
        # exactly what came in. No error invented, none lost.
        self.assertEqual(sum(e for _, e in done) + acc.errors, 8537)
        for _, e in done:
            self.assertAlmostEqual(e, 1000, delta=2)
        # progress is back inside one block, so a later switch can only ever
        # discard a partial block.
        self.assertLess(acc.progress, acc.n_x)


class TestKeyBuffer(unittest.TestCase):
    """eq (11)/(22): finite capacity, never negative."""

    def test_deposits_accumulate(self):
        buf = KeyBuffer(b_max=1000)
        buf.deposit(300)
        buf.deposit(200)
        self.assertEqual(buf.level, 500)

    def test_it_saturates_at_b_max(self):
        # Overflow is discarded, which is why maximising raw key generation is
        # NOT the same as maximising the objective.
        buf = KeyBuffer(b_max=1000)
        buf.deposit(1500)
        self.assertEqual(buf.level, 1000)

    def test_withdrawal_reduces_the_level(self):
        buf = KeyBuffer(b_max=1000)
        buf.deposit(800)
        buf.withdraw(300)
        self.assertEqual(buf.level, 500)

    def test_it_refuses_to_go_negative(self):
        buf = KeyBuffer(b_max=1000)
        buf.deposit(100)
        with self.assertRaises(ValueError):
            buf.withdraw(200)

    def test_saturation_caps_the_level_not_the_single_deposit(self):
        # `self._level += min(b_max, bits)` caps each deposit instead of the
        # pool and passes both tests above, yet lets the buffer grow without
        # bound over many slots -- exactly the overflow the objective needs.
        buf = KeyBuffer(b_max=1000)
        buf.deposit(800)
        buf.deposit(800)
        self.assertEqual(buf.level, 1000)

    def test_withdrawing_the_whole_level_is_allowed(self):
        # Boundary: eq (2) sizes the withdrawal from `level`, so a `>=` guard
        # would reject the legal maximum and stall service at full buffer.
        buf = KeyBuffer(b_max=1000)
        buf.deposit(400)
        buf.withdraw(400)
        self.assertEqual(buf.level, 0)

    def test_the_pool_is_clamped_at_zero_from_below(self):
        # A negative deposit that drove level below zero would make every
        # later positive withdraw() raise, stopping service permanently and
        # silently: the pool must stay a physical quantity.
        buf = KeyBuffer(b_max=1000)
        buf.deposit(100)
        buf.deposit(-500)
        self.assertEqual(buf.level, 0)
        # Still usable afterwards -- no permanent lockout.
        buf.deposit(200)
        buf.withdraw(200)
        self.assertEqual(buf.level, 0)

    def test_a_refused_withdrawal_leaves_the_level_alone(self):
        # Raising after mutating would leave the pool corrupted for the
        # caller that catches the error.
        buf = KeyBuffer(b_max=1000)
        buf.deposit(100)
        with self.assertRaises(ValueError):
            buf.withdraw(200)
        self.assertEqual(buf.level, 100)


class TestHybridEnv(unittest.TestCase):
    def setUp(self):
        # Short horizon keeps the suite fast; dynamics are unchanged.
        self.params = HybridParams(horizon=200, seed=3)
        self.env = HybridEnv(self.params)

    def test_reset_returns_an_observation_of_the_right_shape(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (OBS_DIM,))

    def test_the_observation_carries_exactly_the_documented_features(self):
        # eq (16): the policy may not see f_t, A_t, or anything after slot
        # t-1. Pinning each index to a known quantity is what catches the
        # realistic regression -- someone appending f_t "just as a hint".
        # The distance IS observable (a vessel knows its own position); the
        # jamming fraction is not, and that asymmetry is the model.
        self.env.reset()
        obs, _, _, info = self.env.step(ENT)
        self.assertEqual(len(obs), OBS_DIM)
        self.assertEqual(obs[I_MODE], float(ENT))
        self.assertAlmostEqual(obs[I_BUFFER] * self.params.b_max,
                               info["buffer"], places=6)
        self.assertAlmostEqual(obs[I_DIST] * DIST_REF_KM, info["d_km"])
        # f_t and the arrivals are reported for analysis but never observed.
        self.assertIn("f", info)
        self.assertIn("arrivals", info)

    def test_every_observation_index_is_pinned_to_a_past_quantity(self):
        # The test above pins two of the six entries, so an _observe() that
        # writes f_t (or the arrival count) into any of the other four passes
        # it, passes every shape check, and quietly destroys the causality
        # constraint the whole research question rests on. Pin all six against
        # independently readable state and there is nowhere left to hide.
        #
        # The conditions matter as much as the assertions: jamming is held at
        # a distinctive non-zero level and messages are priced above B_max so
        # the queue never empties. Checked on a quiet slot with an empty queue
        # instead, four of the six entries are zero, f_t is zero, and a leak
        # matches by coincidence -- which is exactly how the first draft of
        # this test let that mutant through.
        # f = 0.025 is jamming that PM still survives, so the buffer fills and
        # that entry is non-zero too.
        p = HybridParams(horizon=30, seed=7, w_window=3, l0=20_000_000)
        env = HybridEnv(p, EveParams(f_levels=(0.025,) * 3, seed=1))
        env.reset()
        recent = []
        for t in range(30):
            action = ENT if t % 7 == 3 else PM
            obs, _, _, info = env.step(action)
            recent = (recent + [info["qber"]])[-3:]
            self.assertEqual(len(obs), OBS_DIM)
            self.assertAlmostEqual(obs[I_QBER], sum(recent) / len(recent))
            self.assertEqual(obs[I_MODE], float(action))
            self.assertAlmostEqual(obs[I_BLOCK] * p.n_x, env.block_progress)
            self.assertAlmostEqual(obs[I_BUFFER] * p.b_max, info["buffer"])
            self.assertAlmostEqual(obs[I_QUEUE] * (p.lam * p.t0_deadline),
                                   env.queue_length)
            self.assertAlmostEqual(obs[I_AGE] * p.t0_deadline,
                                   env.queue_head_age)
            self.assertAlmostEqual(obs[I_DIST] * DIST_REF_KM,
                                   env.distance_km)
        # The run really did exercise non-trivial values in every slot that a
        # leaked f_t could have impersonated.
        self.assertGreater(env.queue_length, 0)
        self.assertGreater(env.queue_head_age, 0)
        self.assertGreater(env.buffer_level, 0)

    def test_the_slot_qber_fluctuates_with_the_sample_size(self):
        # Spec section 5: the ONLY observation noise is the binomial error
        # draw. Replace it with its expected value and q_t becomes a noiseless
        # function of Eve's state -- the environment stops being partially
        # observable, which is the entire reason the problem needs a policy
        # rather than a lookup table.
        steady = EveParams(f_levels=(0.03, 0.03, 0.03), seed=1)
        env = HybridEnv(HybridParams(horizon=60, seed=17), steady)
        env.reset()
        done, seen = False, set()
        while not done:
            _, _, done, info = env.step(PM)
            seen.add(info["qber"])
        self.assertGreater(len(seen), 30)

    def test_the_qber_observation_averages_only_the_last_w_slots(self):
        # The W-slot window is what makes the observation lag Eve's state.
        # Drop the bound and q_t becomes a whole-episode mean that never
        # recovers after a jamming burst, so the policy stops being able to
        # see the channel clear -- and no shape or invariant test notices.
        env = HybridEnv(HybridParams(horizon=20, seed=13, w_window=3))
        env.reset()
        recent = []
        for _ in range(12):
            obs, _, _, info = env.step(PM)
            recent = (recent + [info["qber"]])[-3:]
            self.assertAlmostEqual(obs[I_QBER], sum(recent) / len(recent))

    def test_an_episode_ends_after_the_horizon(self):
        self.env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = self.env.step(0)
            steps += 1
        self.assertEqual(steps, self.params.horizon)

    def test_reward_is_the_number_of_messages_served(self):
        self.env.reset()
        _, reward, _, info = self.env.step(0)
        self.assertEqual(reward, info["served"])

    def test_eta_is_a_fraction(self):
        self.env.reset()
        done = False
        while not done:
            _, _, done, _ = self.env.step(0)
        self.assertGreaterEqual(self.env.eta, 0.0)
        self.assertLessEqual(self.env.eta, 1.0)

    def test_served_never_exceeds_arrivals(self):
        self.env.reset()
        done, served, arrived = False, 0, 0
        while not done:
            _, r, done, info = self.env.step(0)
            served += r
            arrived += info["arrivals"]
        self.assertLessEqual(served, arrived)

    def test_the_buffer_stays_within_bounds(self):
        self.env.reset()
        done = False
        while not done:
            _, _, done, info = self.env.step(0)
            self.assertGreaterEqual(info["buffer"], 0)
            self.assertLessEqual(info["buffer"], self.params.b_max)

    def test_same_seed_reproduces_the_run(self):
        def run():
            env = HybridEnv(HybridParams(horizon=100, seed=42))
            env.reset()
            done, total = False, 0
            while not done:
                _, r, done, _ = env.step(0)
                total += r
            return total
        self.assertEqual(run(), run())

    def test_pm_beats_ent_when_nobody_is_jamming(self):
        # Sanity check against the measured kappa table: with f == 0, PM
        # distils ~20x more key per slot, so it must serve more messages.
        quiet = EveParams(f_levels=(0.0, 0.0, 0.0), seed=1)
        pm = HybridEnv(HybridParams(horizon=600, seed=1), quiet)
        ent = HybridEnv(HybridParams(horizon=600, seed=1), quiet)
        self.assertGreater(_run_fixed(pm, 0), _run_fixed(ent, 1))

    def test_ent_beats_pm_under_heavy_jamming(self):
        # f = 0.08 is inside the switching window: PM's key length hits zero
        # while ENT still distils. This is the whole premise of the problem.
        heavy = EveParams(f_levels=(0.08, 0.08, 0.08), seed=1)
        pm = HybridEnv(HybridParams(horizon=1200, seed=1), heavy)
        ent = HybridEnv(HybridParams(horizon=1200, seed=1), heavy)
        self.assertGreater(_run_fixed(ent, 1), _run_fixed(pm, 0))

    def test_switching_every_slot_wastes_every_block(self):
        # eq (5): alternating modes means no block ever completes, so no key
        # is ever distilled -- the cost that makes the decision sequential.
        env = HybridEnv(HybridParams(horizon=400, seed=1))
        env.reset()
        done, action, served = False, 0, 0
        while not done:
            _, r, done, _ = env.step(action)
            action = 1 - action
            served += r
        self.assertEqual(served, 0)

    def test_holding_one_mode_accumulates_across_slots(self):
        # BlockAccumulator compares modes with == only, so feeding it "pm" /
        # "ent" instead of the integer constants reads as a switch on every
        # slot: no block ever completes and the run silently yields zero key
        # with nothing anywhere raising.
        env = HybridEnv(HybridParams(horizon=10, seed=9))
        env.reset()
        env.step(ENT)
        first = env.block_progress
        env.step(ENT)
        self.assertGreater(first, 0)
        self.assertEqual(env.block_progress, 2 * first)
        self.assertEqual(env.block_mode, ENT)

    def test_the_per_slot_sifted_rates_match_the_measured_design_point(self):
        # Spec section 4, measured at 55 km / 0.2 dB/km / 0 dB optics: PM
        # distils its block from the Z basis only (a quarter of clicks) and
        # BBM92 keys both sifted bases (half). Swap those two factors and
        # every dynamics test still passes while ENT's rate doubles, which
        # invalidates every PM-vs-ENT number the study reports.
        env = HybridEnv(HybridParams(seed=0))
        self.assertEqual(env.sifted_per_slot, (69422, 3485))
        # e_PM = 1.02%, e_ENT = 3.18%: the intrinsic floors of spec section 5.
        self.assertAlmostEqual(env.error_floor[PM], 0.0102, places=4)
        self.assertAlmostEqual(env.error_floor[ENT], 0.0318, places=4)

    def test_the_jamming_and_arrivals_do_not_depend_on_the_actions(self):
        # Policies are compared on a shared seed, so the exogenous processes
        # have to be common random numbers. Draw the errors and the arrivals
        # from one stream and numpy's binomial consumes a mode-dependent
        # number of uniforms, so every policy meets different traffic and a
        # worse policy can win on luck alone.
        def trace(actions):
            env = HybridEnv(HybridParams(horizon=len(actions), seed=5))
            env.reset()
            out = []
            for a in actions:
                _, _, _, info = env.step(a)
                out.append((info["f"], info["arrivals"]))
            return out
        n = 60
        self.assertEqual(trace([PM] * n), trace([ENT] * n))
        self.assertEqual(trace([PM] * n), trace([t % 2 for t in range(n)]))

    def test_reset_clears_the_queue_the_block_and_the_buffer(self):
        # Training reuses one env for thousands of episodes. State surviving
        # reset() carries one episode's banked key into the next, so the
        # learning curve measures the leak instead of the policy.
        env = HybridEnv(HybridParams(horizon=400, seed=6))
        env.reset()
        for _ in range(120):
            env.step(ENT)
        # ENT deliberately: PM drains the queue every slot, so a reset that
        # keeps the queue object would still read length 0 and survive.
        self.assertGreater(env.queue_length, 0)
        self.assertGreater(env.buffer_level, 0)
        self.assertGreater(env.block_progress, 0)
        obs = env.reset()
        # Every entry back to zero EXCEPT the distance, which is a position
        # and not an accumulator: at reset the vessel is back at the start of
        # its track, which on a fixed link is d_km.
        self.assertEqual(list(obs)[:I_DIST], [0.0] * I_DIST)
        self.assertAlmostEqual(obs[I_DIST], env.params.d_km / DIST_REF_KM)
        self.assertEqual(env.queue_length, 0)
        self.assertEqual(env.block_progress, 0)
        self.assertEqual(env.buffer_level, 0)
        # Both eta counters, asserted directly. eta == 0.0 is NOT a proxy for
        # this: drop the served reset and 0/A is still 0.0; drop the arrived
        # reset and the zero-denominator guard returns 0.0 as well. Either
        # deletion corrupts the only metric the study reports and would stay
        # green behind that assertion.
        self.assertEqual(env.served_total, 0)
        self.assertEqual(env.arrived_total, 0)
        self.assertEqual(env.eta, 0.0)
        # And the counters must still work after the reset: one slot in, the
        # arrivals of that slot are all that either counter has seen.
        _, r, _, info = env.step(PM)
        self.assertEqual(env.arrived_total, info["arrivals"])
        self.assertEqual(env.served_total, r)

    def test_reset_rewinds_the_episode_so_one_env_can_replay_it(self):
        # test_same_seed_reproduces_the_run builds a fresh env each time, so
        # it cannot see a reset() that forgets to rewind the RNG streams or
        # Eve's chain. Replaying on the SAME instance can.
        env = HybridEnv(HybridParams(horizon=40, seed=8))

        def episode():
            env.reset()
            out, done = [], False
            while not done:
                _, r, done, info = env.step(PM)
                out.append((info["f"], info["arrivals"], r))
            return out
        self.assertEqual(episode(), episode())

    def test_reset_with_a_seed_draws_a_different_episode(self):
        # Without this, every episode from one env is byte-identical: a
        # training loop calling reset() bare would fit 200 replays of a single
        # trajectory and then be scored on a seed set containing it. The
        # per-episode seed must move BOTH the sampling streams and Eve's
        # chain, or the agent still memorises one jamming schedule.
        env = HybridEnv(HybridParams(horizon=200, seed=1))

        def episode(seed):
            env.reset(seed)
            out, done = [], False
            while not done:
                _, r, done, info = env.step(PM)
                out.append((info["f"], info["arrivals"], r))
            return out
        first, again, other = episode(1234), episode(1234), episode(5678)
        self.assertEqual(first, again)          # same seed: exact replay
        self.assertNotEqual(first, other)       # different seed: new episode
        # Specifically the jamming, not just the traffic.
        self.assertNotEqual([f for f, _, _ in first],
                            [f for f, _, _ in other])

    def test_reset_without_a_seed_keeps_replaying_one_episode(self):
        # The default must stay exactly as it was: two envs sharing an
        # EveParams have to meet identical jamming, which is what makes
        # AlwaysPM and AlwaysENT comparable at all.
        env = HybridEnv(HybridParams(horizon=120, seed=1))

        def episode():
            env.reset()
            out, done = [], False
            while not done:
                _, r, done, info = env.step(ENT)
                out.append((info["f"], info["arrivals"], r))
            return out
        self.assertEqual(episode(), episode())
        twin = HybridEnv(HybridParams(horizon=120, seed=1))
        twin.reset()
        done, out = False, []
        while not done:
            _, r, done, info = twin.step(ENT)
            out.append((info["f"], info["arrivals"], r))
        self.assertEqual(episode(), out)

    def test_the_per_slot_key_rates_match_the_measured_kappa_table(self):
        # kappa_m of eq (12) -- secret bits per slot -- is what every result in
        # the study rests on, and what Task 8's "<2% against the real stack"
        # gate compares to. Spec section 4: 23,336 (PM) and 1,137 (ENT) at
        # 55 km with no jamming. Pinned only by the sifted rate and the error
        # floor today, a 20% drift in the block -> key mapping would pass the
        # whole suite in silence.
        #
        # Tolerance 2%, from two effects, both measured across seeds 1-3:
        #   - binomial noise on a block's QBER moves its key length ~1.5%,
        #     averaged down by sqrt(number of blocks) to ~0.1% here;
        #   - the unfinished block at the end of the run is not counted, worth
        #     up to one block (0.1% for PM, 0.7% for ENT at these horizons).
        # Observed spread was 23,333-23,364 and 1,135.7-1,137.8.
        quiet = EveParams(f_levels=(0.0, 0.0, 0.0), seed=1)
        for action, horizon, kappa in ((PM, 1200, 23336), (ENT, 4000, 1137)):
            env = HybridEnv(HybridParams(horizon=horizon, seed=1), quiet)
            env.reset()
            done, deposited = False, 0
            while not done:
                _, _, done, info = env.step(action)
                deposited += info["deposited"]
            measured = deposited / horizon
            self.assertAlmostEqual(measured / kappa, 1.0, delta=0.02,
                                   msg="kappa for action %d measured %.1f, "
                                       "expected %d" % (action, measured,
                                                        kappa))

    def test_every_arrival_is_served_expired_or_still_waiting(self):
        # eq (4) as an end-to-end invariant. ENT alone cannot keep up with the
        # traffic, so this run really does exercise the expiry path.
        env = HybridEnv(HybridParams(horizon=300, seed=11))
        env.reset()
        done, arrived, served, expired = False, 0, 0, 0
        while not done:
            _, r, done, info = env.step(ENT)
            arrived += info["arrivals"]
            served += r
            expired += info["expired"]
        self.assertGreater(expired, 0)
        self.assertEqual(arrived, served + expired + env.queue_length)

    def test_service_cannot_outrun_the_key_buffer(self):
        # eq (2): a message costs l0 bits of one-time pad. Price it above
        # B_max and nothing can ever be paid for, so an implementation that
        # serves the queue without charging the buffer shows up as eta > 0.
        env = HybridEnv(HybridParams(horizon=100, seed=2, l0=20_000_000))
        env.reset()
        done, arrived, served = False, 0, 0
        while not done:
            _, r, done, info = env.step(PM)
            arrived += info["arrivals"]
            served += r
        self.assertGreater(arrived, 0)
        self.assertEqual(served, 0)
        self.assertEqual(env.eta, 0.0)

    def test_distilled_key_is_either_banked_or_spent(self):
        # Serving without withdrawing lets one distilled bit pay for every
        # message ever sent -- eta goes to 1 and the objective stops meaning
        # anything. The run is short enough that B_max never saturates, so
        # the balance is an exact equality.
        env = HybridEnv(HybridParams(horizon=200, seed=4))
        env.reset()
        done, deposited, served = False, 0, 0
        while not done:
            _, r, done, info = env.step(PM)
            deposited += info["deposited"]
            served += r
        self.assertGreater(deposited, 0)
        self.assertGreater(served, 0)
        self.assertLess(deposited, env.params.b_max)
        self.assertEqual(env.buffer_level,
                         deposited - served * env.params.l0)

    def test_no_message_expires_while_the_buffer_can_still_pay(self):
        # The other half of the arrive -> serve -> age_and_expire order.
        # Expiring first kills a message in the slot the deadline still
        # allowed, even with key sitting unspent -- and every conservation
        # identity balances either way, so only this invariant sees it:
        # serve() drains the queue oldest-first up to what the buffer can
        # afford, so anything left to expire means the buffer ran out.
        #
        # lam = 0.14 puts the demand (1,147 bits/slot) within 1% of ENT's key
        # rate (1,137), so the buffer hovers around l0 and the queue stays a
        # handful of messages. That is the only regime where the two orders
        # differ: with a permanently starved buffer the expiring message is
        # simply replaced by a younger one and the totals match either way.
        env = HybridEnv(HybridParams(horizon=3000, seed=15, lam=0.14))
        env.reset()
        done, expired_total = False, 0
        while not done:
            _, _, done, info = env.step(ENT)
            if info["expired"]:
                self.assertLess(info["buffer"], env.params.l0)
            expired_total += info["expired"]
        self.assertGreater(expired_total, 0)

    def test_a_rich_slot_banks_every_block_it_earned(self):
        # At d_km = 0 -- a setting the panel actually uses -- PM sifts 853,699
        # bits in one slot against n_X = 100,000, so a single slot retires
        # eight blocks. Banking only the first throws away ~88% of the key,
        # and at the 55 km design point (69,422 per slot, never more than one
        # block per slot) nothing else in this file can tell the difference.
        env = HybridEnv(HybridParams(horizon=5, seed=16, d_km=0.0))
        env.reset()
        _, _, _, info = env.step(PM)
        self.assertEqual(env.sifted_per_slot[PM] // env.params.n_x, 8)
        self.assertEqual(info["blocks"], 8)
        # Eight blocks of real key, not one: a single block can never exceed
        # its own n_X bits.
        self.assertGreater(info["deposited"], 2 * env.params.n_x)

    def test_a_message_can_be_served_in_the_slot_it_arrives(self):
        # The per-slot order is arrive -> serve -> age_and_expire. Serving
        # before the arrivals land delays every message by one slot, which
        # shifts the effective deadline and depresses eta without breaking a
        # single conservation identity.
        env = HybridEnv(HybridParams(horizon=300, seed=12))
        env.reset()
        for _ in range(150):        # warm the buffer up past any shortage
            env.step(PM)
        self.assertEqual(env.queue_length, 0)
        for _ in range(100):
            _, r, _, info = env.step(PM)
            self.assertEqual(r, info["arrivals"])


def _run_fixed(env, action):
    """Run a whole episode with one fixed action; return messages served."""
    env.reset()
    done, total = False, 0
    while not done:
        _, r, done, _ = env.step(action)
        total += r
    return total


if __name__ == "__main__":
    unittest.main()


class TestTheRelaxedDynamics(unittest.TestCase):
    """HybridEnv.kappa / step_relaxed -- the mechanics of the Oracle bound.

    Everything here exists to stop the bound quietly becoming something else.
    A bound that is not actually relaxed is not an upper bound, and nothing
    about it looks wrong from the outside: it prints a smaller number and
    makes switching look more valuable than it really is.
    """

    def test_kappa_reproduces_the_measured_table(self):
        # Spec section 5's kappa table, which every conclusion of the study
        # rests on. 2%, the tolerance the block-rate test above already uses.
        env = HybridEnv(HybridParams(horizon=10, seed=1), QUIET_EVE)
        for f, k_pm, k_ent in ((0.00, 23332.0, 1137.0),
                               (0.03, 7642.0, 776.2)):
            self.assertAlmostEqual(
                env.kappa(PM, f) / k_pm, 1.0, delta=0.02,
                msg="kappa_PM(%.2f) = %.1f" % (f, env.kappa(PM, f)))
            self.assertAlmostEqual(
                env.kappa(ENT, f) / k_ent, 1.0, delta=0.02,
                msg="kappa_ENT(%.2f) = %.1f" % (f, env.kappa(ENT, f)))
        # PM is dead at f = 0.08 and ENT is not: that gap is the entire
        # reason a switching policy could have had any value at all.
        self.assertEqual(env.kappa(PM, 0.08), 0.0)
        self.assertAlmostEqual(env.kappa(ENT, 0.08) / 271.2, 1.0, delta=0.02)

    def test_kappa_is_the_rate_the_unrelaxed_environment_really_delivers(self):
        # The bound only bounds if its continuous rate matches the block
        # dynamics it replaces. A kappa scaled by, say, a basis factor would
        # inflate the Oracle without failing a single table lookup.
        for action, horizon in ((PM, 1200), (ENT, 4000)):
            env = HybridEnv(HybridParams(horizon=horizon, seed=1), QUIET_EVE)
            env.reset()
            done, deposited = False, 0
            while not done:
                _, _, done, info = env.step(action)
                deposited += info["deposited"]
            self.assertAlmostEqual(
                (deposited / horizon) / env.kappa(action, 0.0), 1.0,
                delta=0.02,
                msg="mode %d: blocks gave %.1f bit/slot, kappa says %.1f"
                    % (action, deposited / horizon, env.kappa(action, 0.0)))

    def test_the_relaxed_slot_banks_key_at_once_not_in_blocks(self):
        # Relaxation 3. ENT needs ~29 slots to fill a block at this channel,
        # so after ONE slot the unrelaxed environment has banked nothing
        # while the relaxed one has banked a slot's worth.
        params = HybridParams(horizon=5, seed=1, lam=0.0)
        real = HybridEnv(params, QUIET_EVE)
        real.reset()
        real.step(ENT)
        self.assertEqual(real.buffer_level, 0)

        relaxed = HybridEnv(params, QUIET_EVE)
        relaxed.reset()
        _, _, _, info = relaxed.step_relaxed(lambda f, rates, obs: ENT)
        self.assertAlmostEqual(info["deposited"], relaxed.kappa(ENT, 0.0))
        self.assertEqual(relaxed.buffer_level, int(relaxed.kappa(ENT, 0.0)))

    def test_the_relaxed_run_pays_nothing_for_a_switch(self):
        # Relaxation 2. Alternating mode every slot destroys every partial
        # block under eq (5); under the relaxation each slot must still bank
        # exactly kappa of the mode it ran, with no deduction anywhere.
        env = HybridEnv(HybridParams(horizon=40, seed=1, lam=0.0), QUIET_EVE)
        env.reset()
        slot = [0]

        def alternate(f, rates, obs):
            slot[0] += 1
            return ENT if slot[0] % 2 else PM

        # The buffer is tracked as well as the reported deposit: a switch
        # cost charged as a separate withdrawal would leave `deposited`
        # looking innocent. lam = 0, so nothing is ever spent.
        done, banked, switches, last, level = False, 0.0, 0, None, 0
        while not done:
            _, _, done, info = env.step_relaxed(alternate)
            self.assertAlmostEqual(info["deposited"],
                                   env.kappa(info["mode"], info["f"]))
            banked += info["deposited"]
            level += int(info["deposited"])
            self.assertEqual(env.buffer_level, level)
            switches += 1 if last is not None and info["mode"] != last else 0
            last = info["mode"]
        self.assertEqual(switches, 39)          # it really did thrash
        self.assertAlmostEqual(banked,
                               20 * (env.kappa(PM, 0.0) + env.kappa(ENT, 0.0)))

    def test_the_chooser_is_handed_the_true_jamming_of_this_slot(self):
        # Relaxation 1, structurally: whatever the chooser reads as `f` must
        # be the f the slot actually ran at, not a lagged or estimated one,
        # and `rates` must agree with kappa at that f for both modes.
        # 300 slots against a mean dwell of 40: long enough that the chain
        # really does leave QUIET, which the last assertion checks rather
        # than assumes.
        env = HybridEnv(HybridParams(horizon=300, seed=3, lam=0.0),
                        EveParams(f_levels=(0.0, 0.03, 0.08), seed=3))
        env.reset()
        seen = []

        def watch(f, rates, obs):
            seen.append((f, tuple(rates)))
            return PM

        done = False
        while not done:
            _, _, done, info = env.step_relaxed(watch)
            f_seen, rates_seen = seen[-1]
            self.assertEqual(f_seen, info["f"])
            self.assertEqual(rates_seen, (env.kappa(PM, info["f"]),
                                          env.kappa(ENT, info["f"])))
        self.assertGreater(len({f for f, _ in seen}), 1)   # f really moved

    def test_the_relaxed_observation_is_the_one_a_real_agent_would_have_had(self):
        # The bound must sit above a REALISABLE policy, so the observation
        # handed to the chooser has to be the noisy, lagged one that policy
        # would have seen. Otherwise "the Oracle reads f instead of obs" is a
        # distinction without a difference and no test of clairvoyance can
        # bite. obs[I_BLOCK] is the exception: relaxation 3 leaves no partial
        # block, so it is always 0.
        env = HybridEnv(HybridParams(horizon=12, seed=5, lam=0.0), QUIET_EVE)
        env.reset()
        obs_seen = []

        def record(f, rates, obs):
            obs_seen.append(obs)
            return PM

        done = False
        while not done:
            _, _, done, _ = env.step_relaxed(record)
        self.assertEqual(obs_seen[0][I_QBER], 0.0)     # empty window
        # By the end the window holds PM's floor -- measured, not assumed.
        self.assertAlmostEqual(obs_seen[-1][I_QBER], env.error_floor[PM],
                               delta=0.002)
        self.assertTrue(all(o[I_BLOCK] == 0.0 for o in obs_seen))

    def test_the_relaxed_run_faces_the_same_weather_as_the_real_one(self):
        # The bound is subtracted from policy rows, so it must run on the
        # identical jamming and arrival trajectories. Separate RNG streams
        # make that true; sharing one would silently break it.
        params = HybridParams(horizon=200, seed=7, lam=2.5)
        eve = EveParams(f_levels=(0.0, 0.03, 0.08), seed=7)
        real, relaxed = HybridEnv(params, eve), HybridEnv(params, eve)
        real.reset(7)
        relaxed.reset(7)
        for _ in range(200):
            _, _, _, a = real.step(PM)
            _, _, _, b = relaxed.step_relaxed(lambda f, rates, obs: PM)
            self.assertEqual((a["f"], a["arrivals"]), (b["f"], b["arrivals"]))
