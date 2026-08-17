"""Hybrid PM/ENT environment: eq (1)-(24) of system_model_v3_5."""
import unittest

from hybridenv import MessageQueue


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


if __name__ == "__main__":
    unittest.main()
