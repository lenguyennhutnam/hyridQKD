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


class TestTheMechanismAndNotJustItsSymptoms(unittest.TestCase):
    """The tests above pin symptoms -- reproducibility, the level set, mean
    dwell. None of them pins the transition mechanism itself, so a transposed
    matrix (which leaves every diagonal, and therefore every dwell time,
    untouched) used to pass the whole suite while moving P(HEAVY) from 0.211
    to 0.397. These pin the mechanism."""

    def test_the_empirical_transitions_match_the_matrix_row_for_row(self):
        # THE defining property of a Markov chain. Kills three mutations at
        # once: a transposed transition matrix, a searchsorted on the raw row
        # instead of its cumulative sum, and an off-by-one in the state
        # update. Tolerance is set below the smallest disagreement a
        # transpose produces (0.005) and far above the sampling error
        # (worst-case sd 5e-4 at 400k steps).
        params = EveParams(seed=17)
        chain = EveChain(params)
        chain.reset()
        counts = [[0] * 3 for _ in range(3)]
        prev = chain.state
        for _ in range(400000):
            chain.step()
            counts[prev][chain.state] += 1
            prev = chain.state
        for i, row in enumerate(counts):
            total = sum(row)
            self.assertGreater(total, 10000, "state %d barely visited" % i)
            for j, n in enumerate(row):
                self.assertAlmostEqual(n / total, params.transition[i][j],
                                       delta=0.0025,
                                       msg="P(%d->%d)" % (i, j))

    def test_the_time_spent_in_each_state_matches_the_stationary_vector(self):
        # A second, independent grip on the same mutation: the exact
        # stationary distribution of the default matrix is 0.466/0.315/0.219,
        # and a transpose sends HEAVY to 0.397. How often the chain is in
        # HEAVY is what decides how often PM is dead, so this number is
        # load-bearing for every eta on the branch.
        chain = EveChain(EveParams(seed=23))
        chain.reset()
        seen = [0, 0, 0]
        for _ in range(400000):
            chain.step()
            seen[chain.state] += 1
        share = [n / sum(seen) for n in seen]
        for got, want in zip(share, (0.466, 0.315, 0.219)):
            self.assertAlmostEqual(got, want, delta=0.02)

    def test_reset_rewinds_the_random_stream_and_not_just_the_state(self):
        # run_episode calls reset() once per episode. A reset that rewinds
        # the state but leaves the RNG running makes every episode after the
        # first a CONTINUATION rather than a replay -- so two policies
        # "compared on the same trajectory" would not be, which is the one
        # assumption the whole comparison rests on. The existing
        # same-seed test cannot see this: it uses two fresh chains.
        chain = EveChain(EveParams(seed=4))
        chain.reset(99)
        first = [chain.step() for _ in range(100)]
        chain.reset(99)
        self.assertEqual([chain.step() for _ in range(100)], first)
        chain.reset(100)
        self.assertNotEqual([chain.step() for _ in range(100)], first)

    def test_f_reports_the_level_of_the_state_the_chain_just_entered(self):
        # Kills an f indexed by the PREVIOUS state: the jamming the policy
        # sees would lag the chain by one slot, quietly decorrelating the
        # QBER observation from the jamming that produced it.
        params = EveParams(seed=8)
        chain = EveChain(params)
        chain.reset()
        for _ in range(2000):
            returned = chain.step()
            self.assertEqual(returned, params.f_levels[chain.state])
            self.assertEqual(chain.f, params.f_levels[chain.state])


if __name__ == "__main__":
    unittest.main()
