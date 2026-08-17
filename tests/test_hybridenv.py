"""Hybrid PM/ENT environment: eq (1)-(24) of system_model_v3_5."""
import unittest

from hybridenv import BlockAccumulator, KeyBuffer, MessageQueue


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


if __name__ == "__main__":
    unittest.main()
