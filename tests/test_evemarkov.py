"""Eve's jamming process: a 3-state Markov chain the agent cannot observe."""
import unittest

from evemarkov import QUIET, EveChain, EveParams


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


class TestTheSecondAttackChannel(unittest.TestCase):
    """Eve now has two levers on the same chain: jamming, which raises the
    QBER and hurts BOTH modes, and PNS, which certifies PM's key away while
    leaving the QBER and ENT untouched.

    They ride ONE chain deliberately. Eve commits her hardware to a posture
    per slot; giving PNS its own chain would also double the unobservable
    state the policy has to infer, for no modelled reason.
    """

    def test_the_default_adversary_does_no_pns(self):
        # Every result the study has published was measured without it, so
        # the default has to leave them alone.
        params = EveParams()
        self.assertEqual(set(params.s_levels), {0.0})
        self.assertEqual(len(params.s_levels), len(params.f_levels))
        self.assertEqual(EveChain(params).pns, 0.0)

    def test_both_levers_are_read_from_the_same_state(self):
        params = EveParams(f_levels=(0.0, 0.03, 0.0),
                           s_levels=(0.0, 0.0, 0.6))
        chain = EveChain(params)
        for _ in range(400):
            chain.step()
            self.assertEqual(chain.f, params.f_levels[chain.state])
            self.assertEqual(chain.pns, params.s_levels[chain.state])

    def test_a_state_can_carry_pns_without_any_jamming(self):
        # The interesting adversary: invisible to the QBER by construction.
        params = EveParams(f_levels=(0.0, 0.0, 0.0),
                           s_levels=(0.0, 0.0, 0.6))
        chain = EveChain(params)
        seen = set()
        for _ in range(2000):
            chain.step()
            self.assertEqual(chain.f, 0.0)
            seen.add(chain.pns)
        self.assertEqual(seen, {0.0, 0.6})

    def test_adding_pns_levels_does_not_disturb_the_jamming_trajectory(self):
        # THE regression guard. The chain must consume the same random draws
        # either way, or every episode in the study replays differently the
        # moment a PNS level is configured.
        plain = EveChain(EveParams(seed=7))
        armed = EveChain(EveParams(seed=7, s_levels=(0.0, 0.3, 0.6)))
        self.assertEqual([plain.step() for _ in range(500)],
                         [armed.step() for _ in range(500)])

    def test_reset_returns_both_levers_to_the_quiet_state(self):
        params = EveParams(s_levels=(0.0, 0.3, 0.6))
        chain = EveChain(params)
        for _ in range(300):
            chain.step()
        chain.reset()
        self.assertEqual(chain.f, params.f_levels[QUIET])
        self.assertEqual(chain.pns, params.s_levels[QUIET])

    def test_a_level_list_of_the_wrong_length_is_refused(self):
        # Silently padding would give the last states a PNS strength nobody
        # configured, which is the sort of thing that shows up as a number
        # that will not reproduce.
        with self.assertRaises(ValueError):
            EveChain(EveParams(s_levels=(0.0, 0.5)))

    def test_a_pns_level_outside_the_unit_interval_is_refused(self):
        for bad in ((-0.1, 0.0, 0.0), (0.0, 0.0, 1.5)):
            with self.assertRaises(ValueError):
                EveChain(EveParams(s_levels=bad))


class TestNamedPostures(unittest.TestCase):
    """The named adversaries live here, beside the chain they configure, so
    the visualiser can offer them without importing the whole study driver."""

    def test_the_four_postures_are_named(self):
        from evemarkov import ADVERSARIES

        self.assertEqual(set(ADVERSARIES), {"jam", "pns", "both", "none"})

    def test_pns_splits_photons_and_jams_nothing(self):
        from evemarkov import adversary

        eve = adversary("pns")
        self.assertEqual(set(eve.f_levels), {0.0})
        self.assertGreater(max(eve.s_levels), 0.0)

    def test_jam_is_the_historical_default_exactly(self):
        # Every number measured before PNS existed ran on this.
        from evemarkov import adversary

        self.assertEqual(adversary("jam"), EveParams())

    def test_a_posture_keeps_the_chain_it_is_given(self):
        from evemarkov import adversary

        base = EveParams(transition=((0.9, 0.1, 0.0), (0.1, 0.8, 0.1),
                                     (0.0, 0.2, 0.8)), seed=11)
        eve = adversary("pns", base)
        self.assertEqual(eve.transition, base.transition)
        self.assertEqual(eve.seed, 11)

    def test_an_unknown_posture_is_refused_not_defaulted(self):
        from evemarkov import adversary

        with self.assertRaises(KeyError):
            adversary("stealthy")

    def test_the_map_still_offers_the_same_postures(self):
        # optmap re-exports rather than redefining, so the two cannot drift.
        import evemarkov
        import optmap

        self.assertIs(optmap.ADVERSARIES, evemarkov.ADVERSARIES)
        self.assertIs(optmap.adversary, evemarkov.adversary)


if __name__ == "__main__":
    unittest.main()
