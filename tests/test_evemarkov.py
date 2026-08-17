"""Eve's jamming process: a 3-state Markov chain the agent cannot observe."""
import unittest

from evemarkov import EveChain, EveParams


class TestEveChain(unittest.TestCase):
    def test_same_seed_gives_the_same_sequence(self):
        # Reproducibility is load-bearing: every policy must be compared on
        # the SAME jamming trajectory, otherwise the comparison is noise.
        a = EveChain(EveParams(seed=7))
        b = EveChain(EveParams(seed=7))
        a.reset()
        b.reset()
        self.assertEqual([a.step() for _ in range(200)],
                         [b.step() for _ in range(200)])

    def test_different_seeds_diverge(self):
        a = EveChain(EveParams(seed=1))
        b = EveChain(EveParams(seed=2))
        a.reset()
        b.reset()
        self.assertNotEqual([a.step() for _ in range(200)],
                            [b.step() for _ in range(200)])

    def test_every_emitted_f_is_one_of_the_levels(self):
        chain = EveChain(EveParams(seed=3))
        chain.reset()
        levels = set(EveParams().f_levels)
        for _ in range(500):
            self.assertIn(chain.step(), levels)

    def test_mean_dwell_time_matches_the_transition_matrix(self):
        # Self-transition 0.975 -> mean dwell 1/0.025 = 40 slots. If the chain
        # switched every slot the agent could never profit from switching mode,
        # so this property is what makes the problem well-posed.
        chain = EveChain(EveParams(seed=11))
        chain.reset()
        runs, current, length = [], chain.state, 1
        for _ in range(200000):
            chain.step()
            if chain.state == current:
                length += 1
            else:
                runs.append(length)
                current, length = chain.state, 1
        mean = sum(runs) / len(runs)
        self.assertGreater(mean, 20.0)
        self.assertLess(mean, 60.0)

    def test_all_three_states_are_visited(self):
        chain = EveChain(EveParams(seed=5))
        chain.reset()
        seen = {chain.state}
        for _ in range(20000):
            chain.step()
            seen.add(chain.state)
        self.assertEqual(seen, {0, 1, 2})

    def test_reset_returns_to_the_quiet_state(self):
        chain = EveChain(EveParams(seed=5))
        for _ in range(50):
            chain.step()
        self.assertEqual(chain.reset(), 0)
        self.assertAlmostEqual(chain.f, 0.0)


if __name__ == "__main__":
    unittest.main()
