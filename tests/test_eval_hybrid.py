"""The evaluation driver: does switching mode actually pay?

Every test below names the wrong implementation it is there to kill. The
recurring hazard is a driver that flatters the learned policy: a sweep that
hands the baseline a bad operating point, a table whose rows are scored on
different jamming, or an agent scored on the seeds it trained on. None of
those show up as a crash -- they show up as a nicer number.
"""
import dataclasses
import statistics
import unittest

from dqn import DQNConfig, training_seed_range
from evemarkov import EveParams
from hybridenv import HybridEnv, HybridParams
from policies import (AlwaysENT, AlwaysPM, BufferAwareThreshold,
                      GreedyThreshold, run_episode)

# lam = 2.5 with B_max = 1e6. At the spec section 4 defaults AlwaysPM scores
# eta = 1.0000 and every threshold ties, so a sweep returning its WORST point
# would still satisfy `table[best] == max(table.values())`. In this regime
# the greedy thresholds measurably separate (0.7577 vs 0.8978), which is what
# gives the "best, not worst" assertions something to bite on.
NONDEGENERATE = {"lam": 2.5, "b_max": 1_000_000}

# Mean dwell keeps the chain in QUIET most of the time; the HEAVY level is
# above PM's death QBER and below ENT's.
MIXED_EVE = {"f_levels": (0.0, 0.03, 0.08)}

# Small enough that a training run costs milliseconds. seed=0 consumes
# training seeds {0, 1}, which every evaluation seed below stays clear of.
TINY_DQN = DQNConfig(episodes=2, hidden=8, warmup=10, batch_size=8,
                     capacity=200, seed=0)


def nondegenerate_params(horizon=200):
    return HybridParams(horizon=horizon, seed=1, **NONDEGENERATE)


def nondegenerate_factory(horizon=200):
    """seed -> HybridEnv in the regime where policies actually separate."""
    def factory(seed):
        return HybridEnv(
            dataclasses.replace(nondegenerate_params(horizon), seed=seed),
            EveParams(seed=seed, **MIXED_EVE))
    return factory


class TestThresholdSweep(unittest.TestCase):
    """The greedy baseline must be given its best threshold, otherwise beating
    it proves nothing about the learned policy."""

    def test_the_sweep_returns_the_best_threshold_it_tried(self):
        from eval_hybrid import sweep_threshold

        def factory(seed):
            heavy = EveParams(f_levels=(0.0, 0.03, 0.08), seed=seed)
            return HybridEnv(HybridParams(horizon=200, seed=seed), heavy)

        best, table = sweep_threshold(factory, (0.02, 0.04, 0.06), seeds=(1, 2))
        self.assertIn(best, (0.02, 0.04, 0.06))
        self.assertEqual(set(table), {0.02, 0.04, 0.06})
        self.assertEqual(table[best], max(table.values()))

    def test_the_sweep_returns_the_best_and_not_the_worst(self):
        # Kills `min(table, key=table.get)`. The test above cannot: at the
        # spec defaults every threshold scores exactly 1.0, so argmin and
        # argmax return the same key and the assertion passes either way.
        from eval_hybrid import sweep_threshold

        best, table = sweep_threshold(nondegenerate_factory(),
                                      (0.02, 0.04, 0.06), seeds=(1, 2))
        self.assertLess(min(table.values()), max(table.values()))
        self.assertEqual(best, max(table, key=table.get))
        self.assertNotEqual(best, min(table, key=table.get))

    def test_the_sweep_scores_every_seed_it_is_given(self):
        # Kills a sweep that ignores `seeds` (run_episode without a seed
        # replays ONE trajectory), that scores the wrong policy, or that
        # aggregates over the wrong axis. The expected numbers are built
        # here from the same factory, so only the real computation matches.
        from eval_hybrid import sweep_threshold

        factory = nondegenerate_factory(horizon=120)
        seeds = (3, 4)
        expected = {}
        for th in (0.02, 0.06):
            env = factory(seeds[0])
            expected[th] = statistics.mean(
                run_episode(env, GreedyThreshold(th), seed=s)["eta"]
                for s in seeds)
        _, table = sweep_threshold(factory, (0.02, 0.06), seeds=seeds)
        self.assertEqual(table, expected)
        # ... and a single-seed sweep is a different number, so "seeds" is
        # not being silently collapsed to its first element.
        _, one = sweep_threshold(factory, (0.02, 0.06), seeds=(seeds[0],))
        self.assertNotEqual(one, table)

    def test_thresholds_closer_than_the_policy_name_get_separate_rows(self):
        # GreedyThreshold.name renders %.3f, so 0.0405 and 0.0410 both become
        # "greedy-0.041". A table keyed on the name silently drops one row and
        # reports a sweep of two points as a sweep of one.
        from eval_hybrid import sweep_threshold

        self.assertEqual(GreedyThreshold(0.0405).name,
                         GreedyThreshold(0.0410).name)
        _, table = sweep_threshold(nondegenerate_factory(horizon=60),
                                   (0.0405, 0.0410), seeds=(1,))
        self.assertEqual(len(table), 2)
        self.assertEqual(set(table), {0.0405, 0.0410})

    def test_a_repeated_threshold_is_refused_rather_than_overwritten(self):
        # A caller widening the sweep must not lose a point in silence.
        from eval_hybrid import sweep_threshold

        with self.assertRaises(ValueError):
            sweep_threshold(nondegenerate_factory(horizon=60),
                            (0.02, 0.02), seeds=(1,))


class TestTheGridReachesThePlateau(unittest.TestCase):
    """A sweep whose argmax sits on its own edge has not been shown to contain
    the optimum -- "best threshold" then only means "largest one tried"."""

    def test_the_swept_optimum_is_not_the_largest_candidate(self):
        # Kills a truncated THRESHOLDS, which is what this module shipped
        # with: the grid stopped at 0.050 and reported it as the best greedy
        # point, while every larger threshold never fires at all (the highest
        # QBER PM can display under this Eve is E_PM(0.08) = 4.94%) and scores
        # strictly higher by being exactly AlwaysPM.
        from eval_hybrid import THRESHOLDS, sweep_threshold

        best, table = sweep_threshold(nondegenerate_factory(horizon=200),
                                      THRESHOLDS, seeds=(11, 12))
        self.assertEqual(table[best], max(table.values()))
        self.assertNotEqual(best, max(THRESHOLDS))

    def test_the_top_of_the_grid_is_the_never_switch_plateau(self):
        # The companion assertion: the argmax is interior because the grid
        # reaches a region where switching has stopped, not because a tie
        # happened to break leftwards. Pins the plateau by CONTENT -- the
        # largest candidate must score exactly AlwaysPM.
        from eval_hybrid import THRESHOLDS, sweep_threshold

        seeds = (11, 12)
        factory = nondegenerate_factory(horizon=200)
        best, table = sweep_threshold(factory, THRESHOLDS, seeds=seeds)
        env = factory(seeds[0])
        always_pm = statistics.mean(
            run_episode(env, AlwaysPM(), seed=s)["eta"] for s in seeds)
        self.assertAlmostEqual(table[max(THRESHOLDS)], always_pm)
        self.assertAlmostEqual(table[best], always_pm)
        # ...and the grid still contains points that DO switch, so it has not
        # degenerated into a sweep over four copies of AlwaysPM.
        self.assertLess(min(table.values()), always_pm)


class TestBufferAwareSweep(unittest.TestCase):
    def test_the_grid_covers_every_threshold_and_reserve_pair(self):
        # A coordinate sweep (best threshold first, then reserve) can miss the
        # grid optimum; denying the baseline its best point manufactures a
        # win for whatever it is compared against.
        from eval_hybrid import sweep_buffer_aware

        best, table = sweep_buffer_aware(nondegenerate_factory(horizon=120),
                                         thresholds=(0.02, 0.04),
                                         reserves=(0.0, 0.25, 0.5),
                                         seeds=(1, 2))
        self.assertEqual(len(table), 6)
        self.assertEqual(set(table),
                         {(0.02, 0.0), (0.02, 0.25), (0.02, 0.5),
                          (0.04, 0.0), (0.04, 0.25), (0.04, 0.5)})
        self.assertEqual(table[best], max(table.values()))


class TestEvaluationSeedsAreDisjointFromTraining(unittest.TestCase):
    """Scoring the agent on a trajectory it trained on is not a measurement."""

    def test_the_offered_evaluation_seeds_avoid_the_training_range(self):
        from eval_hybrid import evaluation_seeds

        # The third config is the awkward one: seed 990 trains on exactly the
        # block evaluation_seeds reserves, so returning that block blindly
        # would hand the agent its own training trajectories back.
        for cfg in (DQNConfig(), TINY_DQN, DQNConfig(seed=990, episodes=500)):
            trained = training_seed_range(cfg)
            seeds = evaluation_seeds(10, cfg)
            self.assertEqual(len(set(seeds)), 10)
            for s in seeds:
                self.assertNotIn(s, trained)

    def test_the_default_evaluation_seeds_would_have_collided_naively(self):
        # The brief's `range(1, n + 1)` lands inside DQNConfig()'s 0..199.
        # This pins WHY evaluation_seeds exists rather than trusting it.
        trained = training_seed_range(DQNConfig())
        self.assertIn(1, trained)
        self.assertIn(5, trained)

    def test_compare_refuses_a_seed_the_agent_trained_on(self):
        from eval_hybrid import compare

        with self.assertRaises(ValueError) as ctx:
            compare(seeds=(0, 1), params=nondegenerate_params(horizon=60),
                    dqn_config=TINY_DQN)
        self.assertIn("train", str(ctx.exception).lower())

    def test_compare_accepts_seeds_outside_the_training_range(self):
        from eval_hybrid import compare

        rows = compare(seeds=(11, 12), params=nondegenerate_params(horizon=60),
                       dqn_config=TINY_DQN, thresholds=(0.04,),
                       reserves=(0.0,))
        self.assertIn("dqn", [r["policy"] for r in rows])


class TestLabelRows(unittest.TestCase):
    def test_a_duplicate_label_is_refused_rather_than_overwritten(self):
        # DQNPolicy.name is a CLASS constant, so two trained agents would
        # collide on "dqn" and the second row would replace the first with no
        # symptom at all.
        from eval_hybrid import label_rows

        rows = [{"policy": "dqn", "eta": 0.1}, {"policy": "dqn", "eta": 0.9}]
        with self.assertRaises(ValueError):
            label_rows(rows)

    def test_distinct_labels_are_indexed(self):
        from eval_hybrid import label_rows

        rows = [{"policy": "a", "eta": 0.1}, {"policy": "b", "eta": 0.9}]
        self.assertEqual(label_rows(rows)["b"]["eta"], 0.9)


class TestCompare(unittest.TestCase):
    def test_every_policy_is_scored_on_the_same_seeds(self):
        # If the rows came from different seed sets the whole table would be
        # meaningless. Rebuild two rows by hand from the same seeds and
        # demand exact agreement -- this also pins the mean to the eta axis
        # rather than to switches or ent_fraction.
        from eval_hybrid import compare, label_rows

        seeds = (11, 12)
        params = nondegenerate_params(horizon=120)
        rows = label_rows(compare(seeds=seeds, params=params,
                                  dqn_config={}, thresholds=(0.04,),
                                  reserves=(0.0,)))
        for policy in (AlwaysPM(), AlwaysENT()):
            env = HybridEnv(params, EveParams())
            want = [run_episode(env, policy, seed=s) for s in seeds]
            self.assertAlmostEqual(rows[policy.name]["eta"],
                                   statistics.mean(r["eta"] for r in want))
            self.assertAlmostEqual(rows[policy.name]["switches"],
                                   statistics.mean(r["switches"]
                                                   for r in want))

    def test_the_greedy_row_carries_the_swept_best_threshold(self):
        # Kills a compare() that hardcodes THRESHOLDS[0] instead of sweeping.
        from eval_hybrid import compare, label_rows, sweep_threshold

        seeds = (11, 12)
        thresholds = (0.02, 0.04, 0.06)
        params = nondegenerate_params(horizon=200)
        best, _ = sweep_threshold(nondegenerate_factory(horizon=200),
                                  thresholds, seeds=seeds)
        rows = label_rows(compare(seeds=seeds, params=params,
                                  eve_params=EveParams(**MIXED_EVE),
                                  dqn_config={}, thresholds=thresholds,
                                  reserves=(0.0,)))
        self.assertIn(GreedyThreshold(best).name, rows)
        self.assertNotEqual(best, thresholds[0])

    def test_the_buffer_aware_row_is_the_best_point_of_its_grid(self):
        from eval_hybrid import compare, label_rows, sweep_buffer_aware

        seeds = (11, 12)
        thresholds, reserves = (0.02, 0.04), (0.0, 0.5)
        params = nondegenerate_params(horizon=120)
        _, table = sweep_buffer_aware(nondegenerate_factory(horizon=120),
                                      thresholds, reserves, seeds=seeds)
        rows = label_rows(compare(seeds=seeds, params=params,
                                  eve_params=EveParams(**MIXED_EVE),
                                  dqn_config={}, thresholds=thresholds,
                                  reserves=reserves))
        buffer_rows = [r for k, r in rows.items() if k.startswith("buffer-")]
        self.assertEqual(len(buffer_rows), 1)
        self.assertAlmostEqual(buffer_rows[0]["eta"], max(table.values()))

    def test_the_buffer_aware_row_beats_a_coordinate_descent(self):
        # Kills a compare() that searches "best greedy threshold, then best
        # reserve at that threshold" instead of the full grid -- the mutant
        # the previous review found alive, because compare() had its own copy
        # of the search and the 2x2 fixture above happened to make the two
        # answers agree.
        #
        # This fixture is chosen so they DISAGREE: the greedy sweep prefers
        # 0.045 while the buffer-aware grid's optimum sits at 0.050, so a
        # coordinate descent is locked out of the winning column.
        from eval_hybrid import (compare, label_rows, sweep_buffer_aware,
                                 sweep_threshold)

        seeds = (21, 22)
        thresholds, reserves = (0.045, 0.05), (0.1, 0.25)
        factory = nondegenerate_factory(horizon=300)
        best_th, _ = sweep_threshold(factory, thresholds, seeds=seeds)
        best_pair, table = sweep_buffer_aware(factory, thresholds, reserves,
                                              seeds=seeds)
        coord = max((k for k in table if k[0] == best_th), key=table.get)
        # The trap has to be real, or this test proves nothing: assert the
        # coordinate descent is STRICTLY worse before demanding compare avoid
        # it. If the fixture ever degenerates, this line fails loudly rather
        # than letting the test pass vacuously.
        self.assertNotEqual(coord, best_pair)
        self.assertLess(table[coord], table[best_pair])

        rows = label_rows(compare(seeds=seeds,
                                  params=nondegenerate_params(horizon=300),
                                  eve_params=EveParams(**MIXED_EVE),
                                  dqn_config={}, thresholds=thresholds,
                                  reserves=reserves))
        buffer_rows = [r for k, r in rows.items() if k.startswith("buffer-")]
        self.assertEqual(len(buffer_rows), 1)
        self.assertEqual(buffer_rows[0]["policy"],
                         BufferAwareThreshold(*best_pair).name)
        self.assertAlmostEqual(buffer_rows[0]["eta"], table[best_pair])

    def test_two_dqn_configs_produce_two_distinct_rows(self):
        # DQNPolicy.name is a class constant; keying the table on it would
        # collapse these two into one row.
        from eval_hybrid import compare

        configs = {"dqn-a": TINY_DQN,
                   "dqn-b": dataclasses.replace(TINY_DQN, seed=7)}
        rows = compare(seeds=(11, 12), params=nondegenerate_params(horizon=60),
                       dqn_config=configs, thresholds=(0.04,),
                       reserves=(0.0,))
        labels = [r["policy"] for r in rows]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn("dqn-a", labels)
        self.assertIn("dqn-b", labels)

    def test_the_table_can_report_ent_winning(self):
        # A driver rigged to favour PM would be useless as evidence that PM
        # wins. Under permanent heavy jamming PM distils nothing and ENT
        # still serves a little, and the table must say so.
        from eval_hybrid import compare, label_rows

        rows = label_rows(compare(
            seeds=(11, 12), params=nondegenerate_params(horizon=200),
            eve_params=EveParams(f_levels=(0.08, 0.08, 0.08)),
            dqn_config={}, thresholds=(0.04,), reserves=(0.0,)))
        self.assertGreater(rows["always-ent"]["eta"], rows["always-pm"]["eta"])
        self.assertEqual(rows["always-pm"]["eta"], 0.0)


class TestPresetFlagsReachTheEnvironment(unittest.TestCase):
    """The hardware assumption decides whether switching has any value at all,
    so a preset flag that parses but never reaches HybridParams is the worst
    kind of bug: every table would be labelled with a scenario it did not run
    at, and nothing would look wrong."""

    def _capture(self, argv):
        import eval_hybrid
        seen = {}

        def fake_compare(seeds=None, params=None, **kw):
            seen["params"] = params
            return [{"policy": "always-pm", "episodes": 1, "eta": 0.0,
                     "switches": 0.0, "expired": 0.0, "secret_bits": 0.0,
                     "ent_fraction": 0.0, "starved": 0.0}]

        real = eval_hybrid.compare
        eval_hybrid.compare = fake_compare
        try:
            eval_hybrid.main(argv)
        finally:
            eval_hybrid.compare = real
        return seen["params"]

    def test_the_preset_flags_are_not_silently_dropped(self):
        params = self._capture(["--no-dqn", "--seeds", "1",
                                "--ent-preset", "ent_advanced",
                                "--pm-preset", "rusca"])
        self.assertEqual(params.ent_preset, "ent_advanced")
        self.assertEqual(params.pm_preset, "rusca")

    def test_omitting_the_flags_keeps_the_documented_defaults(self):
        # Kills an override that writes None into HybridParams when the flag is
        # absent, which would break every existing run rather than leaving it
        # alone.
        params = self._capture(["--no-dqn", "--seeds", "1"])
        self.assertEqual(params.ent_preset, HybridParams().ent_preset)
        self.assertEqual(params.pm_preset, HybridParams().pm_preset)


if __name__ == "__main__":
    unittest.main()


class TestTheOracleRowsInTheTable(unittest.TestCase):
    """The bound is worthless if it never reaches the table, and worse than
    worthless if it reaches it as a duplicate of the row it is compared
    against."""

    def _rows(self, **over):
        from eval_hybrid import compare
        return compare(seeds=(11, 12),
                       params=nondegenerate_params(horizon=200),
                       eve_params=EveParams(**MIXED_EVE), dqn_config={},
                       thresholds=(0.04,), reserves=(0.0,), **over)

    def test_the_table_carries_the_oracle_and_its_relaxed_control(self):
        from eval_hybrid import label_rows

        rows = label_rows(self._rows())
        self.assertIn("oracle", rows)
        self.assertIn("always-pm-relaxed", rows)

    def test_the_control_row_is_not_the_oracle_row(self):
        # The failure this kills: scoring one bound and reporting it twice.
        # Both rows then agree exactly, the switching value collapses to 0
        # and the study understates its own headroom -- with nothing visibly
        # wrong in the table.
        rows = self._rows()
        oracle = [r for r in rows if r["policy"] == "oracle"][0]
        control = [r for r in rows if r["policy"] == "always-pm-relaxed"][0]
        self.assertIsNot(oracle, control)
        self.assertEqual(control["ent_fraction"], 0.0)
        self.assertGreater(oracle["ent_fraction"], 0.0)
        self.assertGreater(oracle["switches"], 0.0)
        self.assertEqual(control["switches"], 0.0)

    def test_the_bounds_are_scored_on_the_same_seeds_as_the_policies(self):
        # Rebuilt by hand from the same seeds: a bound run on a different
        # seed set is not comparable with the rows it is subtracted from.
        from eval_hybrid import label_rows
        from policies import Oracle, run_relaxed_episode

        seeds = (11, 12)
        params = nondegenerate_params(horizon=200)
        rows = label_rows(self._rows())
        want = []
        for s in seeds:
            env = HybridEnv(dataclasses.replace(params, seed=s),
                            EveParams(seed=s, **MIXED_EVE))
            want.append(run_relaxed_episode(env, Oracle(), seed=s))
        self.assertAlmostEqual(rows["oracle"]["eta"],
                               statistics.mean(r["eta"] for r in want))

    def test_the_oracle_row_is_at_least_as_good_as_every_other_row(self):
        # Spec section 9 again, at the level of the published table.
        from eval_hybrid import label_rows

        rows = label_rows(self._rows())
        for label, row in rows.items():
            if label == "oracle":
                continue
            self.assertGreaterEqual(rows["oracle"]["eta"], row["eta"], label)


class TestTheSwitchingValueDecomposition(unittest.TestCase):
    """Oracle - AlwaysPM measures the relaxation AND the switching together.
    Only Oracle - RelaxedAlwaysPM isolates switching, and at the measured
    operating points most of the headroom turns out to be the relaxation --
    so a driver that reported only the total would overstate the thing the
    study is about."""

    def test_it_splits_the_headroom_at_the_relaxed_control(self):
        from eval_hybrid import switching_value

        rows = [{"policy": "always-pm", "eta": 0.50},
                {"policy": "always-pm-relaxed", "eta": 0.60},
                {"policy": "oracle", "eta": 0.65}]
        split = switching_value(rows)
        self.assertAlmostEqual(split["relaxation"], 0.10)
        self.assertAlmostEqual(split["switching"], 0.05)
        self.assertAlmostEqual(split["total"], 0.15)

    def test_it_reports_nothing_rather_than_guessing_without_the_bounds(self):
        from eval_hybrid import switching_value

        self.assertIsNone(switching_value([{"policy": "always-pm",
                                            "eta": 0.5}]))

    def test_the_cli_prints_both_halves_of_the_split(self):
        # A decomposition computed and never printed is a decomposition the
        # reader does not have.
        import contextlib
        import io as _io

        import eval_hybrid

        rows = [{"policy": p, "episodes": 1, "eta": e, "switches": 0.0,
                 "expired": 0.0, "secret_bits": 0.0, "ent_fraction": 0.0,
                 "starved": 0.0}
                for p, e in (("always-pm", 0.50), ("always-pm-relaxed", 0.60),
                             ("oracle", 0.65))]
        real = eval_hybrid.compare
        eval_hybrid.compare = lambda **kw: rows
        try:
            out = _io.StringIO()
            with contextlib.redirect_stdout(out):
                eval_hybrid.main(["--seeds", "1", "--horizon", "10",
                                  "--no-dqn"])
        finally:
            eval_hybrid.compare = real
        text = out.getvalue()
        self.assertIn("relaxation alone  +0.100000", text)
        self.assertIn("switching itself  +0.050000", text)
        self.assertIn("total headroom    +0.150000", text)
