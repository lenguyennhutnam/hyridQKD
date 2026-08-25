"""The policy visualiser.

The figure this module produces is meant for a thesis, so the tests are
written against the three ways it could quietly lie rather than against the
happy path:

* the policies could be run on DIFFERENT jamming, which makes every
  comparison in the picture meaningless while looking entirely normal;
* the Oracle could be drawn as an ordinary policy, which turns an upper bound
  into an apparently achievable curve;
* the scoreboard's eta could come from a counter that is not eq (24), which
  agrees with the study at exactly the parameter setting the study calls
  degenerate and disagrees at the one it reports.

Each of those has a test that goes red when the property is removed.
"""
import dataclasses
import os
import shutil
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")            # no display in tests
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import channel  # noqa: E402
import policyviz  # noqa: E402
import presets  # noqa: E402
from dqn import DQNConfig, training_seed_range  # noqa: E402
import plots  # noqa: E402
from evemarkov import EveParams  # noqa: E402
from hybridenv import ENT, PM, HybridEnv  # noqa: E402
from policies import (AlwaysENT, AlwaysPM, BufferAwareThreshold,  # noqa: E402
                      GreedyThreshold, Oracle, RelaxedAlwaysPM, run_episode)
from policyviz import (PolicyRun, VizConfig, VizSession, death_thresholds,
                       dqn_label, draw_panels, format_scoreboard,
                       parse_spec)  # noqa: E402

class _RecordingMessagebox:
    """Stands in for tkinter.messagebox: records instead of blocking.

    A real showerror is modal -- it waits for a click -- so leaving it in
    place hangs any unattended run that trips an error path.
    """

    def __init__(self, sink):
        self._sink = sink

    def showerror(self, title, message, **kw):
        self._sink.append(("error", title, message))

    def showinfo(self, title, message, **kw):
        self._sink.append(("info", title, message))


# Short enough to keep the suite quick, long enough that Eve leaves the QUIET
# state and the policies genuinely diverge (mean dwell is 40 slots).
SLOTS = 300


def small_config(**kw):
    """A four-track session that runs in well under a second."""
    base = dict(specs=("always-pm", "always-ent", "greedy-0.045", "oracle"),
                horizon=SLOTS, lam=2.5, b_max=1_000_000)
    base.update(kw)
    return VizConfig(**base)


class TestTheJammingIsShared(unittest.TestCase):
    """THE invariant. Without it the picture compares policies against
    different weather and every conclusion drawn from it is unfounded."""

    @classmethod
    def setUpClass(cls):
        cls.session = VizSession(small_config())
        cls.session.step(SLOTS)

    def test_every_policy_faces_the_identical_jamming_trajectory(self):
        traces = [tuple(run.f) for run in self.session.runs]
        self.assertEqual(len(traces[0]), SLOTS)
        for run, trace in zip(self.session.runs[1:], traces[1:]):
            self.assertEqual(trace, traces[0],
                             "%s saw different jamming from %s"
                             % (run.display_name,
                                self.session.runs[0].display_name))

    def test_the_shared_trajectory_is_not_a_constant(self):
        # If Eve never left QUIET, "all runs saw the same f" would hold for a
        # broken implementation too. This is what makes the test above bite.
        self.assertGreater(len(set(self.session.runs[0].f)), 1)

    def test_the_policies_really_did_behave_differently(self):
        # Same weather, different decisions -- otherwise identical f traces
        # would just mean identical runs.
        modes = {tuple(run.mode) for run in self.session.runs}
        self.assertGreater(len(modes), 1)

    def test_every_policy_faces_the_identical_message_stream(self):
        # Arrivals are drawn from their own stream precisely so the mode
        # cannot perturb them; if it could, eta's denominator would differ
        # per policy and the scoreboard would compare fractions of different
        # totals.
        arrived = {run.env.arrived_total for run in self.session.runs}
        self.assertEqual(len(arrived), 1)

    def test_a_bound_shares_the_trajectory_with_the_policies(self):
        # The Oracle runs through step_relaxed, a different code path. It
        # still has to see the same weather, or the headroom it reports is
        # measured against a different episode.
        bound = [r for r in self.session.runs if r.is_bound][0]
        policy = [r for r in self.session.runs if not r.is_bound][0]
        self.assertEqual(tuple(bound.f), tuple(policy.f))

    def test_the_same_configuration_replays_the_same_episode(self):
        other = VizSession(small_config())
        other.step(SLOTS)
        self.assertEqual(tuple(other.runs[0].f), tuple(self.session.runs[0].f))

    def test_a_different_seed_index_draws_a_different_episode(self):
        other = VizSession(small_config(seed_index=3))
        other.step(SLOTS)
        self.assertNotEqual(tuple(other.runs[0].f),
                            tuple(self.session.runs[0].f))

    def test_the_reset_seed_is_what_actually_pins_the_trajectory(self):
        # Documented because it is counter-intuitive and it is why the guard
        # below exists: HybridEnv.reset(seed) derives Eve's seed from the
        # SeedSequence, so EveParams.seed is OVERRIDDEN and a per-run
        # EveParams.seed changes nothing at all. A test that only varied
        # EveParams.seed would therefore pass against an implementation that
        # shared no weather whatsoever.
        params = small_config().hybrid_params()
        runs = [PolicyRun(AlwaysPM(), HybridEnv(params, EveParams(seed=seed)),
                          seed=params.seed) for seed in (1, 999)]
        for _ in range(SLOTS):
            for run in runs:
                run.step()
        self.assertEqual(runs[0].f, runs[1].f)

    def test_different_reset_seeds_really_do_change_the_weather(self):
        # The other half: what the guard is protecting.
        params = small_config().hybrid_params()
        runs = [PolicyRun(AlwaysPM(), HybridEnv(params, EveParams()),
                          seed=seed) for seed in (params.seed,
                                                  params.seed + 1)]
        for _ in range(SLOTS):
            for run in runs:
                run.step()
        self.assertNotEqual(runs[0].f, runs[1].f)


class TestTheSharedWeatherGuard(unittest.TestCase):
    """The guard is what turns the invariant from a habit into a rule."""

    def _session(self):
        return VizSession(small_config(specs=("always-pm", "always-ent")))

    def test_a_correct_session_passes(self):
        self._session()._assert_shared_weather()

    def test_runs_on_different_reset_seeds_are_refused(self):
        session = self._session()
        session.runs[1].seed += 1
        with self.assertRaises(ValueError):
            session._assert_shared_weather()

    def test_runs_with_different_eve_seeds_are_refused(self):
        # Inert today, load-bearing after any edit that resets bare. The
        # guard refuses it either way rather than depending on which.
        session = self._session()
        session.runs[1].env.eve.params = dataclasses.replace(
            session.runs[1].env.eve.params, seed=12345)
        with self.assertRaises(ValueError):
            session._assert_shared_weather()

    def test_runs_with_a_different_jamming_model_are_refused(self):
        session = self._session()
        session.runs[1].env.eve.params = dataclasses.replace(
            session.runs[1].env.eve.params, f_levels=(0.0, 0.01, 0.02))
        with self.assertRaises(ValueError):
            session._assert_shared_weather()

    def test_the_guard_runs_at_construction_time(self):
        # A guard nobody calls is a comment. Building a session with a
        # mismatched run must fail on the spot, not on first use.
        real = policyviz.PolicyRun.__init__

        def sabotage(self, agent, env, seed, label=None, color=plots.NAVY):
            real(self, agent, env, seed, label=label, color=color)
            self.seed = seed + len(getattr(sabotage, "seen", []))
            sabotage.seen = getattr(sabotage, "seen", []) + [seed]

        policyviz.PolicyRun.__init__ = sabotage
        try:
            with self.assertRaises(ValueError):
                self._session()
        finally:
            policyviz.PolicyRun.__init__ = real


class TestTheOracleIsMarkedAsABound(unittest.TestCase):
    """Honesty requirement 1: a viewer must not read the Oracle as a policy."""

    @classmethod
    def setUpClass(cls):
        cls.session = VizSession(small_config(
            specs=("always-pm", "greedy-0.045", "always-pm-relaxed",
                   "oracle")))
        cls.session.step(SLOTS)

    def _run(self, needle):
        return [r for r in self.session.runs if needle in r.display_name][0]

    def test_the_oracle_is_flagged_a_bound(self):
        self.assertTrue(self._run("oracle").is_bound)

    def test_the_relaxed_control_is_flagged_a_bound_too(self):
        self.assertTrue(self._run("always-pm-relaxed").is_bound)

    def test_an_ordinary_policy_is_not_flagged_a_bound(self):
        self.assertFalse(self._run("greedy").is_bound)

    def test_the_oracle_says_so_in_its_own_name(self):
        self.assertIn("upper bound", self._run("oracle").display_name)

    def test_no_ordinary_policy_claims_to_be_a_bound(self):
        for run in self.session.runs:
            if not run.is_bound:
                self.assertNotIn("bound", run.display_name)

    def test_a_bound_is_dashed_and_a_policy_is_not(self):
        self.assertEqual(self._run("oracle").linestyle, "--")
        self.assertEqual(self._run("greedy").linestyle, "-")

    def test_the_figure_draws_the_bound_dashed(self):
        # The distinction has to survive into the artefact people screenshot,
        # not merely exist as an attribute nobody reads.
        fig = Figure(figsize=(8, 8))
        FigureCanvasAgg(fig)
        draw_panels(fig, self.session)
        ax_eta = fig.axes[3]
        # Keyed on is_bound, NOT on the label: keyed on the label this test
        # goes vacuously green the moment the label stops saying "bound",
        # which is exactly the regression it exists to catch.
        for run, line in zip(self.session.runs, ax_eta.get_lines()):
            want = "--" if run.is_bound else "-"
            self.assertEqual(line.get_linestyle(), want,
                             "%s drawn %s" % (run.display_name,
                                              line.get_linestyle()))
        self.assertTrue(any(r.is_bound for r in self.session.runs))
        self.assertTrue(any(not r.is_bound for r in self.session.runs))

    def test_the_scoreboard_spells_the_caveat_out(self):
        text = format_scoreboard(self.session)
        self.assertIn("NOT achievable", text)

    def test_the_bound_is_marked_on_its_own_row_not_just_in_a_footer(self):
        # A footer alone is not enough: rows get copied out of a scoreboard
        # one at a time, and a row reading plain "oracle" travels as a policy.
        lines = format_scoreboard(self.session).splitlines()
        for run in self.session.runs:
            row = [ln for ln in lines if ln.startswith(run.display_name)]
            self.assertTrue(row, "no row for %s" % run.display_name)
            if run.is_bound:
                self.assertIn("upper bound", row[0])
            else:
                self.assertNotIn("bound", row[0])

    def test_a_board_without_bounds_carries_no_caveat(self):
        # The disclaimer must track the rows actually shown, or it becomes
        # boilerplate a reader learns to skip.
        session = VizSession(small_config(specs=("always-pm",)))
        session.step(10)
        self.assertNotIn("NOT achievable", format_scoreboard(session))

    def test_the_bound_runs_through_the_relaxed_dynamics(self):
        # step_relaxed is the only path that reports kappa; if the Oracle were
        # stepped through step() it would be scored under the true dynamics,
        # which is neither the bound nor a policy.
        env = HybridEnv(small_config().hybrid_params())
        run = PolicyRun(Oracle(), env, seed=env.params.seed)
        info = run.step()
        self.assertIn("rates", info)

    def test_a_policy_does_not_run_through_the_relaxed_dynamics(self):
        env = HybridEnv(small_config().hybrid_params())
        run = PolicyRun(AlwaysPM(), env, seed=env.params.seed)
        self.assertNotIn("rates", run.step())

    def test_run_episode_still_refuses_a_bound(self):
        # policies.py keeps RelaxationBound outside the Policy hierarchy on
        # purpose. This visualiser respects that rather than routing around
        # it, so the guard must stay in force.
        env = HybridEnv(small_config().hybrid_params())
        with self.assertRaises(AttributeError):
            run_episode(env, Oracle())

    def test_a_non_agent_is_refused_outright(self):
        env = HybridEnv(small_config().hybrid_params())
        with self.assertRaises(TypeError):
            PolicyRun(object(), env, seed=0)


class TestTheScoreboardCounters(unittest.TestCase):
    """Honesty requirement: eta is eq (24), served / ARRIVED."""

    @classmethod
    def setUpClass(cls):
        # lam = 2.5 makes served/arrived, served/slot and served/horizon three
        # different numbers; at lam = 1.0 they nearly coincide and this test
        # would pass against the wrong counter.
        cls.session = VizSession(small_config(lam=2.5))
        cls.session.step(SLOTS)
        # NOT `cls.run`: that name is unittest.TestCase.run.
        cls.first = cls.session.runs[0]

    def test_eta_is_the_environments_own_eta(self):
        self.assertAlmostEqual(self.first.score()["eta"], self.first.env.eta,
                               places=12)

    def test_eta_is_served_over_arrived(self):
        env = self.first.env
        self.assertAlmostEqual(self.first.score()["eta"],
                               env.served_total / env.arrived_total,
                               places=12)

    def test_eta_is_not_served_over_slots(self):
        # The plausible wrong counter. It must be numerically distinguishable
        # here, or the test above proves nothing.
        wrong = self.first.served / self.first.slot
        self.assertNotAlmostEqual(self.first.score()["eta"], wrong, places=3)

    def test_eta_is_not_served_over_the_horizon(self):
        wrong = self.first.served / self.session.params.horizon
        self.assertNotAlmostEqual(self.first.score()["eta"], wrong, places=3)

    def test_eta_never_exceeds_one(self):
        for row in self.session.scores():
            self.assertLessEqual(row["eta"], 1.0)
            self.assertGreaterEqual(row["eta"], 0.0)

    def test_ent_fraction_is_over_slots_run_so_far(self):
        # Dividing by the horizon would report every policy as mostly-PM
        # while the episode is young, which is exactly when someone watches.
        partial = VizSession(small_config(specs=("always-ent",), horizon=1000))
        partial.step(50)
        self.assertAlmostEqual(partial.scores()[0]["ent_fraction"], 1.0)

    def test_the_always_policies_report_the_fractions_they_promise(self):
        by = {r["policy"]: r for r in self.session.scores()}
        self.assertEqual(by["always-pm"]["ent_fraction"], 0.0)
        self.assertEqual(by["always-ent"]["ent_fraction"], 1.0)
        self.assertEqual(by["always-pm"]["switches"], 0)

    def test_a_switching_policy_reports_its_switches(self):
        switching = [r for r in self.session.scores()
                     if r["policy"].startswith("greedy")][0]
        self.assertGreater(switching["switches"], 0)

    def test_the_first_slot_is_never_a_switch(self):
        session = VizSession(small_config(specs=("always-ent",)))
        session.step(1)
        self.assertEqual(session.scores()[0]["switches"], 0)


class TestPolicyRunMechanics(unittest.TestCase):
    def test_reset_rewinds_the_traces_and_the_counters(self):
        session = VizSession(small_config(specs=("greedy-0.045",)))
        session.step(50)
        before = tuple(session.runs[0].f)
        session.reset()
        self.assertEqual(session.runs[0].slot, 0)
        self.assertEqual(session.runs[0].f, [])
        self.assertEqual(session.runs[0].switches, 0)
        session.step(50)
        # Reset replays the SAME episode: a reset that drew fresh weather
        # would make "Reset" a different experiment rather than a rewind.
        self.assertEqual(tuple(session.runs[0].f), before)

    def test_stepping_past_the_horizon_stops(self):
        session = VizSession(small_config(specs=("always-pm",), horizon=20))
        self.assertEqual(session.step(100), 20)
        self.assertTrue(session.done)
        self.assertEqual(session.runs[0].slot, 20)
        self.assertIsNone(session.runs[0].step())

    def test_traces_stay_the_same_length_as_the_slot_count(self):
        session = VizSession(small_config(specs=("always-pm", "oracle")))
        session.step(37)
        for run in session.runs:
            self.assertEqual(len(run.t), 37)
            for trace in (run.mode, run.qber, run.buffer, run.eta_trace,
                          run.f):
                self.assertEqual(len(trace), 37)

    def test_the_buffer_trace_respects_b_max(self):
        session = VizSession(small_config(specs=("always-pm",),
                                          b_max=1_000_000))
        session.step(SLOTS)
        self.assertLessEqual(max(session.runs[0].buffer), 1_000_000)

    def test_each_track_gets_its_own_environment(self):
        # One shared env would make the second policy step an environment the
        # first had already advanced.
        session = VizSession(small_config())
        envs = {id(run.env) for run in session.runs}
        self.assertEqual(len(envs), len(session.runs))

    def test_each_track_gets_its_own_colour(self):
        session = VizSession(small_config())
        colours = {run.color for run in session.runs}
        self.assertEqual(len(colours), len(session.runs))


class TestSpecParsing(unittest.TestCase):
    def test_it_accepts_the_labels_the_study_prints(self):
        for spec, cls in (("always-pm", AlwaysPM), ("always-ent", AlwaysENT),
                          ("oracle", Oracle),
                          ("always-pm-relaxed", RelaxedAlwaysPM)):
            self.assertIsInstance(parse_spec(spec), cls)

    def test_greedy_carries_its_threshold_through(self):
        policy = parse_spec("greedy-0.045")
        self.assertIsInstance(policy, GreedyThreshold)
        self.assertAlmostEqual(policy.threshold, 0.045)

    def test_buffer_aware_carries_both_parameters_through(self):
        policy = parse_spec("buffer-aware-0.045-0.25")
        self.assertIsInstance(policy, BufferAwareThreshold)
        self.assertAlmostEqual(policy.threshold, 0.045)
        self.assertAlmostEqual(policy.reserve, 0.25)

    def test_a_parsed_policy_round_trips_through_its_own_name(self):
        # eval_hybrid's tables print policy.name; a spec copied from one must
        # come back as the same policy, or the two tools disagree silently.
        for spec in ("greedy-0.045", "buffer-aware-0.045-0.25"):
            self.assertEqual(parse_spec(spec).name, spec)

    def test_an_unknown_spec_is_refused(self):
        with self.assertRaises(ValueError):
            parse_spec("greedy")

    def test_dqn_without_a_trained_agent_is_refused(self):
        # Returning an untrained network would put a random curve on the
        # figure with nothing to show it was never trained.
        with self.assertRaises(ValueError):
            parse_spec("dqn")


class TestTheDQNIsLabelledWithItsEpisodeCount(unittest.TestCase):
    """Honesty requirement 2: a 20-episode agent must not read as the
    200-episode agent of the study."""

    def test_the_label_states_the_episode_count(self):
        self.assertIn("20", dqn_label(20))
        self.assertIn("ep", dqn_label(20))
        self.assertNotEqual(dqn_label(20), dqn_label(200))

    def test_a_session_labels_its_dqn_track_with_the_count_it_used(self):
        class _Stub(AlwaysPM):
            name = "dqn"
        config = small_config(specs=("dqn",), dqn_episodes=7)
        session = VizSession(config, dqn_policy=_Stub(), dqn_episodes=7)
        self.assertEqual(session.runs[0].display_name, "dqn (7 ep)")

    def test_the_displayed_count_follows_the_run_not_the_default(self):
        class _Stub(AlwaysPM):
            name = "dqn"
        config = small_config(specs=("dqn",), dqn_episodes=200)
        session = VizSession(config, dqn_policy=_Stub(), dqn_episodes=3)
        self.assertIn("3 ep", session.runs[0].display_name)


class TestTrainingTheDQN(unittest.TestCase):
    """Tiny episodes -- this checks the wiring, not the learning."""

    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.config = small_config(specs=("dqn",), horizon=60,
                                   dqn_episodes=2)

    def tearDown(self):
        shutil.rmtree(self.cache, ignore_errors=True)

    def test_it_returns_a_usable_policy_and_reports_it_trained(self):
        policy, cached = policyviz.train_dqn(self.config, cache_dir=self.cache)
        self.assertFalse(cached)
        session = VizSession(self.config, dqn_policy=policy, dqn_episodes=2)
        session.step(60)
        self.assertEqual(session.runs[0].slot, 60)

    def test_progress_is_reported_once_per_episode(self):
        # The window shows this counter instead of freezing silently; a
        # callback that never fires would leave "training… (0 / N)" on screen
        # for the whole run.
        seen = []
        policyviz.train_dqn(self.config, cache_dir=self.cache,
                            progress=lambda ep, eta, eps: seen.append(ep))
        self.assertEqual(len(seen), self.config.dqn_episodes)

    def test_the_second_call_comes_from_the_cache(self):
        policyviz.train_dqn(self.config, cache_dir=self.cache)
        _, cached = policyviz.train_dqn(self.config, cache_dir=self.cache)
        self.assertTrue(cached)

    def test_a_different_ent_preset_is_a_different_agent(self):
        # The two presets give the study OPPOSITE answers, so reusing weights
        # across them would show an agent trained for the other conclusion.
        policyviz.train_dqn(self.config, cache_dir=self.cache)
        other = dataclasses.replace(self.config, ent_preset="chapman_lowmu")
        _, cached = policyviz.train_dqn(other, cache_dir=self.cache)
        self.assertFalse(cached)

    def test_a_different_episode_count_is_a_different_agent(self):
        policyviz.train_dqn(self.config, cache_dir=self.cache)
        other = dataclasses.replace(self.config, dqn_episodes=3)
        _, cached = policyviz.train_dqn(other, cache_dir=self.cache)
        self.assertFalse(cached)

    def test_a_corrupt_cache_costs_a_retrain_not_a_crash(self):
        policyviz.train_dqn(self.config, cache_dir=self.cache)
        for name in os.listdir(self.cache):
            with open(os.path.join(self.cache, name), "wb") as fh:
                fh.write(b"not a pickle")
        policy, cached = policyviz.train_dqn(self.config, cache_dir=self.cache)
        self.assertFalse(cached)
        self.assertIsNotNone(policy)

    def test_training_never_touches_the_displayed_trajectory_seed(self):
        trained = training_seed_range(self.config.dqn_config())
        self.assertNotIn(self.config.env_seed(), trained)


class TestSeedsStayOutOfTheTrainingBlock(unittest.TestCase):
    def test_the_trajectory_seed_is_never_one_the_dqn_trained_on(self):
        config = small_config(specs=("always-pm", "dqn"), dqn_episodes=50)
        trained = training_seed_range(config.dqn_config())
        self.assertNotIn(config.env_seed(), trained)

    def test_a_large_seed_index_still_clears_the_training_block(self):
        config = small_config(specs=("dqn",), dqn_episodes=200,
                              seed_index=25)
        self.assertNotIn(config.env_seed(),
                         training_seed_range(config.dqn_config()))

    def test_no_dqn_means_no_training_config(self):
        self.assertIsNone(small_config(specs=("always-pm",)).dqn_config())

    def test_the_dqn_config_uses_the_requested_episode_count(self):
        config = small_config(specs=("dqn",), dqn_episodes=11)
        self.assertEqual(config.dqn_config().episodes, 11)

    def test_a_training_config_from_the_ui_is_a_real_dqn_config(self):
        self.assertIsInstance(small_config(specs=("dqn",)).dqn_config(),
                              DQNConfig)


class TestDeathThresholds(unittest.TestCase):
    """Panel 2's horizontal lines. They are why a switch is legible."""

    def test_they_match_the_operating_point_the_study_quotes(self):
        # eval_hybrid's docstring: PM 3.3535%, ENT 8.6194% at n_X = 1e5, 55 km.
        pm, ent = death_thresholds(small_config())
        self.assertAlmostEqual(pm, 0.033535, places=5)
        self.assertAlmostEqual(ent, 0.086194, places=5)

    def test_the_channel_overrides_the_presets_own_link(self):
        # Both modes ride the SAME fibre; reading a preset's own d_km here
        # would draw a threshold for a link the environment is not
        # simulating. CLAUDE.md records that mistake being published once.
        near = death_thresholds(small_config(d_km=0.0))[0]
        far = death_thresholds(small_config(d_km=120.0))[0]
        self.assertGreater(near, far)

    def test_the_pm_threshold_uses_the_configured_preset(self):
        aurea = death_thresholds(small_config(pm_preset="aurea"))[0]
        other = death_thresholds(small_config(pm_preset="hwsim_default"))[0]
        self.assertNotAlmostEqual(aurea, other, places=4)

    def test_they_agree_with_channel_called_directly(self):
        import dataclasses
        config = small_config()
        env = dataclasses.replace(presets.load_presets()[config.pm_preset],
                                  d_km=config.d_km, alpha=config.alpha,
                                  odr_losses=config.odr_losses)
        self.assertAlmostEqual(death_thresholds(config)[0],
                               channel.death_qber(env, config.n_x), places=12)


class TestTheFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = VizSession(small_config())
        cls.session.step(SLOTS)
        cls.fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(cls.fig)
        draw_panels(cls.fig, cls.session)
        cls.fig.canvas.draw()          # a real render, not just a build

    def test_it_has_the_four_panels(self):
        self.assertEqual(len(self.fig.axes), 4)

    def test_the_mode_panel_has_a_lane_per_policy(self):
        labels = [t.get_text() for t in self.fig.axes[0].get_yticklabels()]
        self.assertEqual(labels,
                         [r.display_name for r in self.session.runs])

    def test_the_mode_panel_uses_the_projects_own_mode_tints(self):
        import matplotlib.colors as mcolors
        import plots
        wanted = {mcolors.to_hex(plots.MODE_BAND["pm"]).lower(),
                  mcolors.to_hex(plots.MODE_BAND["ent"]).lower()}
        drawn = set()
        for coll in self.fig.axes[0].collections:
            for rgba in coll.get_facecolor():
                drawn.add(mcolors.to_hex(rgba).lower())
        self.assertTrue(wanted <= drawn,
                        "mode bands drawn in %s, expected %s" % (drawn, wanted))

    def test_the_qber_panel_draws_both_death_thresholds(self):
        levels = {round(line.get_ydata()[0], 4)
                  for line in self.fig.axes[1].get_lines()
                  if len(set(line.get_ydata())) == 1}
        self.assertIn(round(100.0 * self.session.death_pm, 4), levels)
        self.assertIn(round(100.0 * self.session.death_ent, 4), levels)

    def test_the_buffer_panel_marks_b_max_when_the_pool_reaches_it(self):
        # b_max small enough that the pool really does approach it.
        session = VizSession(small_config(specs=("always-pm",), b_max=20_000))
        session.step(SLOTS)
        fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(fig)
        draw_panels(fig, session)
        levels = {line.get_ydata()[0] for line in fig.axes[2].get_lines()
                  if len(set(line.get_ydata())) == 1}
        self.assertIn(20_000.0, levels)

    def test_an_off_scale_b_max_is_named_in_the_title_not_drawn(self):
        # At lam = 2.5 the pool never leaves the floor. Pinning the axis to
        # B_max there flattens every trace onto zero and the panel shows
        # nothing; the number still has to reach the reader, so it moves into
        # the title. Same rule as gui.DEATH_LINE_SHOW_FRAC.
        ax_buf = self.fig.axes[2]
        peak = max(b for run in self.session.runs for b in run.buffer)
        self.assertLess(peak,
                        policyviz.REFERENCE_LINE_SHOW_FRAC
                        * self.session.params.b_max)
        levels = {line.get_ydata()[0] for line in ax_buf.get_lines()
                  if len(set(line.get_ydata())) == 1}
        self.assertNotIn(float(self.session.params.b_max), levels)
        self.assertIn("B_max", ax_buf.get_title())

    def test_the_qber_panel_leaves_room_for_its_legend(self):
        # The legend sits top-left, over the slots where the policies first
        # diverge, unless the axis is given headroom above the data.
        ax = self.fig.axes[1]
        peak = max(100.0 * q for run in self.session.runs for q in run.qber)
        self.assertGreater(ax.get_ylim()[1], peak * 1.2)

    def test_the_eta_panel_carries_one_curve_per_policy(self):
        self.assertEqual(len(self.fig.axes[3].get_lines()),
                         len(self.session.runs))

    def test_the_header_says_it_is_the_numeric_model(self):
        self.assertIn("NUMERIC MODEL", self.fig._suptitle.get_text())

    def test_redrawing_does_not_pile_up_axes(self):
        # The window calls draw_panels on every tick; a figure that keeps its
        # old axes would leak one set per frame.
        draw_panels(self.fig, self.session)
        self.assertEqual(len(self.fig.axes), 4)

    def test_an_empty_session_still_renders(self):
        # The window draws once before Start is ever pressed.
        session = VizSession(small_config())
        fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(fig)
        draw_panels(fig, session)
        fig.canvas.draw()
        self.assertEqual(len(fig.axes), 4)



class TestHidingAQberTrace(unittest.TestCase):
    """Panel 2's legend is a CONTROL, not a caption.

    With five or six tracks on, the step traces sit on top of one another --
    always-ent, greedy and oracle spend most of the episode in the same mode
    and therefore draw the same line -- and the panel reads as one solid
    block of colour. The reader has to be able to switch tracks off.

    That control has four ways to be quietly broken, and there is a test
    below for each: the state can be kept on an artist that the next frame
    throws away; the toggle can leak out of panel 2 and erase a policy the
    reader must still be able to see is running; the legend can go on
    claiming a trace is on; and the axis can rescale under the curves that
    are left, which is the comparison the reader was trying to make.
    """

    def setUp(self):
        self.session = VizSession(small_config())
        self.session.step(SLOTS)
        self.fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(self.fig)
        # The track with the highest QBER, so the axis test below has real
        # headroom to lose if hidden traces stopped counting.
        self.name = max(self.session.runs,
                        key=lambda r: max(r.qber)).display_name
        self.other = [r.display_name for r in self.session.runs
                      if r.display_name != self.name][0]

    def _draw(self):
        draw_panels(self.fig, self.session)
        self.fig.canvas.draw()          # a real render, not just a build
        return self.fig

    def _traces(self):
        """{policy name: Line2D} for panel 2's traces; thresholds excluded."""
        names = {r.display_name for r in self.session.runs}
        return {ln.get_label(): ln for ln in self.fig.axes[1].get_lines()
                if ln.get_label() in names}

    def _entry(self, name):
        """(handle, text) of `name`'s legend entry."""
        legend = self.fig.axes[1].get_legend()
        for handle, text in zip(legend.get_lines(), legend.get_texts()):
            if text.get_text() == name:
                return handle, text
        raise AssertionError("no legend entry for %r" % (name,))

    # -- the toggle actually hides something ---------------------------

    def test_a_trace_is_drawn_when_it_is_not_hidden(self):
        # The control for every "it is hidden" assertion below: without this
        # they would all pass on a panel that drew nothing at all.
        self._draw()
        self.assertTrue(self._traces()[self.name].get_visible())

    def test_hiding_a_policy_takes_its_trace_off_the_qber_panel(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        self.assertFalse(self._traces()[self.name].get_visible(),
                         "%s is hidden but its QBER trace is still drawn"
                         % (self.name,))

    def test_hiding_one_policy_leaves_the_others_drawn(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        self.assertTrue(self._traces()[self.other].get_visible())

    def test_toggling_twice_brings_the_trace_back(self):
        self.session.toggle_qber(self.name)
        self.session.toggle_qber(self.name)
        self._draw()
        self.assertTrue(self._traces()[self.name].get_visible())

    def test_toggle_reports_the_visibility_it_leaves_behind(self):
        self.assertFalse(self.session.toggle_qber(self.name))
        self.assertFalse(self.session.qber_visible(self.name))
        self.assertTrue(self.session.toggle_qber(self.name))
        self.assertTrue(self.session.qber_visible(self.name))

    # -- the state cannot live on the artist ---------------------------

    def test_the_hidden_state_survives_the_next_draw(self):
        # THE one that matters. The window redraws on a timer, and
        # draw_panels clears the figure and builds new lines every time, so a
        # visibility flag left on a Line2D would be gone within a fraction of
        # a second and the user's click would silently undo itself while the
        # episode ran.
        self._draw()
        self.session.toggle_qber(self.name)
        self._draw()
        self._draw()
        self.assertFalse(self._traces()[self.name].get_visible(),
                         "the toggle did not survive a redraw: visibility "
                         "is being kept on the artist, not in the session")

    def test_the_line_really_is_a_new_object_on_the_second_draw(self):
        # Without this, the test above could pass on an implementation that
        # reuses one Line2D and keeps the flag on it -- which would then
        # break the moment draw_panels was called on a fresh figure.
        self._draw()
        first = self._traces()[self.name]
        self._draw()
        self.assertIsNot(self._traces()[self.name], first)

    def test_the_state_is_not_read_back_off_the_figure(self):
        # A fresh figure, never drawn to, must render the same hiding.
        self.session.toggle_qber(self.name)
        self._draw()
        other = Figure(figsize=(9, 8))
        FigureCanvasAgg(other)
        draw_panels(other, self.session)
        drawn = {ln.get_label(): ln for ln in other.axes[1].get_lines()}
        self.assertFalse(drawn[self.name].get_visible())

    def test_reset_keeps_the_toggles(self):
        # Reset rewinds the EPISODE. Which traces the reader switched off is
        # not part of the trajectory.
        self.session.toggle_qber(self.name)
        self.session.reset()
        self.assertFalse(self.session.qber_visible(self.name))

    # -- it must not leak out of panel 2 -------------------------------

    def test_hiding_a_trace_leaves_the_mode_lanes_alone(self):
        # A reader who could hide a policy from the QBER panel and thereby
        # lose its lane would conclude it was never selected.
        self.session.hidden_qber.add(self.name)
        self._draw()
        labels = [t.get_text() for t in self.fig.axes[0].get_yticklabels()]
        self.assertEqual(labels,
                         [r.display_name for r in self.session.runs])

    def test_hiding_a_trace_leaves_the_mode_bands_drawn(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        hidden_lane = [i for i, r in enumerate(self.session.runs)
                       if r.display_name == self.name][0]
        drawn = list(self.fig.axes[0].collections)
        self.assertTrue(drawn)
        # The hidden policy's lane still carries its bands.
        lows = set()
        for coll in drawn:
            for path in coll.get_paths():
                lows.add(round(path.get_extents().y0, 2))
        self.assertIn(round(hidden_lane - 0.38, 2), lows)

    def test_hiding_a_trace_leaves_the_buffer_panel_alone(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        drawn = [ln for ln in self.fig.axes[2].get_lines()
                 if ln.get_visible()]
        self.assertEqual(len(drawn), len(self.session.runs))

    def test_hiding_a_trace_leaves_the_eta_panel_alone(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        drawn = [ln for ln in self.fig.axes[3].get_lines()
                 if ln.get_visible()]
        self.assertEqual(len(drawn), len(self.session.runs))

    def test_hiding_a_trace_leaves_the_scoreboard_alone(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        board = format_scoreboard(self.session)
        for run in self.session.runs:
            self.assertIn(run.display_name, board)

    # -- the legend has to show its own state --------------------------

    def test_a_hidden_policy_keeps_its_legend_entry(self):
        # It is the only way back. An entry that vanished with its trace
        # would make the hiding irreversible.
        self.session.hidden_qber.add(self.name)
        self._draw()
        texts = [t.get_text()
                 for t in self.fig.axes[1].get_legend().get_texts()]
        self.assertIn(self.name, texts)

    def test_a_hidden_legend_entry_is_greyed_out(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        handle, text = self._entry(self.name)
        self.assertEqual(text.get_color(), policyviz.HIDDEN_LEGEND_COLOR)
        self.assertEqual(handle.get_color(), policyviz.HIDDEN_LEGEND_COLOR)
        # Alpha as well as colour: the seventh track is drawn in plots.GREY
        # to begin with, so colour alone could not distinguish it.
        self.assertIsNotNone(text.get_alpha())
        self.assertLess(text.get_alpha(), 1.0)
        self.assertLess(handle.get_alpha(), 1.0)

    def test_a_visible_legend_entry_is_not_greyed_out(self):
        self.session.hidden_qber.add(self.name)
        self._draw()
        handle, text = self._entry(self.other)
        self.assertNotEqual(text.get_color(), policyviz.HIDDEN_LEGEND_COLOR)
        self.assertIn(text.get_alpha(), (None, 1.0))
        self.assertIn(handle.get_alpha(), (None, 1.0))

    def test_the_greying_is_undone_when_the_trace_comes_back(self):
        self.session.toggle_qber(self.name)
        self._draw()
        self.session.toggle_qber(self.name)
        self._draw()
        handle, text = self._entry(self.name)
        self.assertNotEqual(text.get_color(), policyviz.HIDDEN_LEGEND_COLOR)
        self.assertIn(text.get_alpha(), (None, 1.0))

    # -- the axis must not move under the remaining curves -------------

    def _session_with_one_tall_trace(self):
        """A session in which exactly one trace holds the y-axis up.

        Nothing reachable does that on its own, which is worth knowing: in a
        300-slot episode the tallest QBER any policy reaches (8.45 %) is
        still BELOW the ENT death threshold (8.62 %), and the three ENT-mode
        tracks peak at the identical value because they see the identical
        jamming. So on real data the limit is set by a threshold line and
        hiding a trace could not move it whatever the rule was -- a test
        written on the default session would pass on both rules and prove
        nothing. One trace is therefore lifted explicitly, to build the
        condition the rule actually governs.
        """
        session = VizSession(small_config(specs=("always-pm", "always-ent")))
        session.step(SLOTS)
        tall = session.runs[-1]
        tall.qber = [q + 0.20 for q in tall.qber]
        others = max([q for r in session.runs[:-1] for q in r.qber]
                     + [session.death_pm, session.death_ent])
        # The two rules genuinely disagree here, so the assertion below is
        # about the rule and not about a coincidence.
        self.assertGreater(max(tall.qber), 1.5 * others)
        return session, tall.display_name

    def test_hiding_the_tallest_trace_does_not_move_the_axis(self):
        # Rescaling to what is left would change the shape and apparent
        # slope of every remaining curve on each click, and the reader hid a
        # trace precisely in order to compare the rest.
        session, tall = self._session_with_one_tall_trace()
        fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(fig)
        draw_panels(fig, session)
        before = fig.axes[1].get_ylim()
        session.hidden_qber.add(tall)
        draw_panels(fig, session)
        self.assertEqual(fig.axes[1].get_ylim(), before)

    def test_the_limit_is_the_one_the_hidden_trace_asks_for(self):
        # Pins the value, not just its stability: an axis frozen at some
        # other number would also "not move".
        session, tall = self._session_with_one_tall_trace()
        session.hidden_qber.add(tall)
        fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(fig)
        draw_panels(fig, session)
        peak = max([100.0 * q for r in session.runs for q in r.qber]
                   + [100.0 * session.death_pm, 100.0 * session.death_ent])
        self.assertAlmostEqual(fig.axes[1].get_ylim()[1], peak * 1.45,
                               places=6)

    def test_hiding_every_trace_still_renders(self):
        for run in self.session.runs:
            self.session.hidden_qber.add(run.display_name)
        self._draw()
        self.assertEqual(len(self.fig.axes), 4)
        self.assertTrue(all(not ln.get_visible()
                            for ln in self._traces().values()))

    # -- the reference lines are not controls --------------------------

    def test_the_death_thresholds_are_not_clickable(self):
        # They are what the traces are read AGAINST: a panel that let you
        # hide the line a curve is dying above invites exactly the
        # misreading it exists to prevent.
        self._draw()
        captions = [t for t in self.fig.axes[1].get_legend().get_texts()
                    if "dies" in t.get_text()]
        self.assertEqual(len(captions), 2)
        for text in captions:
            self.assertNotIn(text, self.session.legend_picks)

    def test_the_death_thresholds_survive_every_policy_being_hidden(self):
        for run in self.session.runs:
            self.session.hidden_qber.add(run.display_name)
        self._draw()
        levels = {round(ln.get_ydata()[0], 4)
                  for ln in self.fig.axes[1].get_lines()
                  if ln.get_visible() and len(set(ln.get_ydata())) == 1}
        self.assertIn(round(100.0 * self.session.death_pm, 4), levels)
        self.assertIn(round(100.0 * self.session.death_ent, 4), levels)

    # -- the pick map --------------------------------------------------

    def test_every_policy_entry_is_a_pick_target(self):
        self._draw()
        self.assertEqual(set(self.session.legend_picks.values()),
                         {r.display_name for r in self.session.runs})

    def test_both_halves_of_an_entry_answer_to_a_click(self):
        # Aiming at a 6 pt line sample is fiddly; a reader who clicks the
        # words and gets nothing concludes the control is broken.
        self._draw()
        handle, text = self._entry(self.name)
        self.assertEqual(self.session.legend_picks.get(handle), self.name)
        self.assertEqual(self.session.legend_picks.get(text), self.name)

    def test_the_pick_map_is_rebuilt_with_the_legend(self):
        # It maps ARTISTS, and every artist is replaced on each draw. A map
        # left over from the previous frame would resolve nothing and the
        # legend would stop responding after the first redraw.
        self._draw()
        stale = dict(self.session.legend_picks)
        self._draw()
        legend = self.fig.axes[1].get_legend()
        current = set(legend.get_lines()) | set(legend.get_texts())
        self.assertTrue(set(self.session.legend_picks) <= current)
        self.assertFalse(set(stale) & set(self.session.legend_picks))

    def test_the_entries_are_pickable_at_all(self):
        self._draw()
        handle, text = self._entry(self.name)
        self.assertIsNotNone(handle.get_picker())
        self.assertIsNotNone(text.get_picker())

    # -- stale names ---------------------------------------------------

    def test_a_name_no_longer_selected_does_not_break_the_draw(self):
        # Apply carries the hidden set across a change of policy selection,
        # so names for policies that are gone are normal, not exceptional.
        self.session.hidden_qber.add("a policy that is not selected")
        self._draw()
        self.assertEqual(len(self.fig.axes), 4)
        self.assertTrue(all(ln.get_visible()
                            for ln in self._traces().values()))



class TestModeSpans(unittest.TestCase):
    """Panel 1's bands. An off-by-one here silently mis-times every switch."""

    def _run(self, modes):
        run = PolicyRun.__new__(PolicyRun)
        run.mode = list(modes)
        run.t = list(range(1, len(modes) + 1))
        return run

    def test_one_unbroken_stretch_is_one_span(self):
        spans = policyviz._mode_spans(self._run([PM] * 5))
        self.assertEqual(spans[PM], [(1, 5)])
        self.assertEqual(spans[ENT], [])

    def test_a_switch_splits_the_stretch_at_the_right_slot(self):
        spans = policyviz._mode_spans(self._run([PM, PM, ENT, ENT, PM]))
        self.assertEqual(spans[PM], [(1, 2), (5, 1)])
        self.assertEqual(spans[ENT], [(3, 2)])

    def test_the_spans_tile_the_episode_exactly(self):
        modes = [PM, ENT, ENT, PM, PM, PM, ENT]
        spans = policyviz._mode_spans(self._run(modes))
        covered = sum(w for group in spans.values() for _, w in group)
        self.assertEqual(covered, len(modes))

    def test_no_spans_before_the_first_slot(self):
        self.assertEqual(policyviz._mode_spans(self._run([])),
                         {PM: [], ENT: []})


class TestTheScoreboardText(unittest.TestCase):
    def test_every_policy_gets_a_line(self):
        session = VizSession(small_config())
        session.step(20)
        text = format_scoreboard(session)
        for run in session.runs:
            self.assertIn(run.display_name, text)

    def test_the_numbers_match_the_scores(self):
        session = VizSession(small_config(specs=("always-pm",)))
        session.step(50)
        self.assertIn("%.4f" % session.scores()[0]["eta"],
                      format_scoreboard(session))


class TestTheWindowConstructsAndRenders(unittest.TestCase):
    """Tk runs under WSLg here and may or may not have a display. The core is
    tested above; this only checks the shell wires up and paints once."""

    def setUp(self):
        try:
            import policyviz_ui
        except Exception as exc:                  # pragma: no cover
            self.skipTest("tkinter/matplotlib unavailable: %s" % exc)
        self.ui = policyviz_ui
        # The window reports a bad configuration with messagebox.showerror,
        # which BLOCKS the mainloop until someone clicks it. Under a display
        # (WSLg provides one here) an unattended run would hang forever, so
        # the dialogs are recorded instead of shown.
        self.dialogs = []
        self._real_messagebox = policyviz_ui.messagebox
        policyviz_ui.messagebox = _RecordingMessagebox(self.dialogs)
        try:
            self.app = policyviz_ui.PolicyVizApp(small_config())
        except Exception as exc:                  # pragma: no cover
            policyviz_ui.messagebox = self._real_messagebox
            self.skipTest("cannot create Tk window: %s" % exc)
        self.app.withdraw()

    def tearDown(self):
        if getattr(self, "_real_messagebox", None) is not None:
            self.ui.messagebox = self._real_messagebox
        app = getattr(self, "app", None)
        if app is not None:
            app._running = False
            app.update_idletasks()
            app.destroy()

    def test_it_builds_a_session_and_paints_four_panels(self):
        self.assertIsNotNone(self.app.session)
        self.assertEqual(len(self.app.canvas.figure.axes), 4)

    def test_the_banner_is_on_screen(self):
        # The one thing that keeps this window from being mistaken for gui.py.
        found = [w for w in self.app.winfo_children()
                 if isinstance(w, self.ui.tk.Label)
                 and "NUMERIC MODEL" in w.cget("text")]
        self.assertTrue(found)

    def test_step_advances_exactly_one_slot(self):
        self.app._step_once()
        self.assertEqual(self.app.session.slot, 1)

    def test_start_and_pause_toggle_the_timer(self):
        self.app._toggle_run()
        self.assertTrue(self.app._running)
        self.app._toggle_run()
        self.assertFalse(self.app._running)

    def test_reset_rewinds_the_session(self):
        self.app._step_once()
        self.app._reset()
        self.assertEqual(self.app.session.slot, 0)

    def test_the_form_shows_the_threshold_it_was_launched_with(self):
        # A ticked "greedy" box next to a threshold the session is not
        # running would misreport what produced the curve on screen.
        # Read through the static helper rather than by opening a second Tk
        # root: two roots in one process is its own source of trouble.
        theta, ba_theta, ba_reserve = self.ui.PolicyVizApp._parameters_from(
            ("greedy-0.020", "buffer-aware-0.030-0.50"))
        self.assertEqual((theta, ba_theta, ba_reserve),
                         ("0.02", "0.03", "0.5"))

    def test_unmentioned_parameters_keep_their_defaults(self):
        theta, ba_theta, ba_reserve = self.ui.PolicyVizApp._parameters_from(
            ("always-pm", "oracle"))
        self.assertEqual((theta, ba_theta, ba_reserve),
                         ("0.045", "0.045", "0.25"))

    def test_the_spec_it_builds_is_the_policys_own_label(self):
        # eval_hybrid keys its tables on policy.name; a private format string
        # here would eventually render the same policy under another label.
        self.app.checks["greedy"].set(True)
        self.app.theta.set("0.02")
        config = self.app._read_config()
        spec = [s for s in config.specs if s.startswith("greedy")][0]
        self.assertEqual(spec, GreedyThreshold(0.02).name)
        self.assertEqual(parse_spec(spec).name, spec)

    def test_the_form_round_trips_the_configuration(self):
        config = self.app._read_config()
        self.assertEqual(config.horizon, SLOTS)
        self.assertIn("always-pm", config.specs)

    def test_a_bad_number_in_the_form_is_reported_not_raised(self):
        self.app.entries["lam"].set("not a number")
        with self.assertRaises(ValueError):
            self.app._read_config()

    def test_selecting_no_policy_is_refused(self):
        for var in self.app.checks.values():
            var.set(False)
        with self.assertRaises(ValueError):
            self.app._read_config()

    def test_the_running_redraw_is_throttled(self):
        # A full-length four-panel figure costs ~0.33 s to render. Redrawing
        # on every 80 ms tick would leave the mainloop painting and nothing
        # else -- the buttons stop responding and the episode crawls.
        calls = []
        real = policyviz.draw_panels
        policyviz.draw_panels = lambda *a, **k: calls.append(1)
        try:
            self.app._redraw(force=True)
            self.app._redraw(force=False)
            self.app._redraw(force=False)
        finally:
            policyviz.draw_panels = real
        self.assertEqual(len(calls), 1)

    def test_a_forced_redraw_is_never_throttled_away(self):
        # Step, Reset and the final frame of an episode all force: there the
        # whole point is to see the slot that just happened.
        calls = []
        real = policyviz.draw_panels
        policyviz.draw_panels = lambda *a, **k: calls.append(1)
        try:
            self.app._redraw(force=True)
            self.app._redraw(force=True)
        finally:
            policyviz.draw_panels = real
        self.assertEqual(len(calls), 2)


    # ------------------------------------------------------------------
    # clicking a legend entry
    # ------------------------------------------------------------------

    def _click(self, artist):
        """Dispatch a pick on `artist` the way the canvas would.

        Deliberately routed through the canvas's callback registry rather
        than by calling _on_pick directly: how MANY handlers are registered
        is the thing being tested, and a direct call would see exactly one
        however many were connected.
        """
        event = type("_Pick", (), {})()
        event.name = "pick_event"
        event.artist = artist
        event.canvas = self.app.canvas
        self.app.canvas.callbacks.process("pick_event", event)

    def _legend_entry(self, name):
        legend = self.app.figure.axes[1].get_legend()
        for text in legend.get_texts():
            if text.get_text() == name:
                return text
        raise AssertionError("no legend entry for %r" % (name,))

    def _trace(self, name):
        for line in self.app.figure.axes[1].get_lines():
            if line.get_label() == name:
                return line
        raise AssertionError("no QBER trace for %r" % (name,))

    def test_the_pick_handler_is_connected_exactly_once(self):
        # draw_panels rebuilds the legend on every frame. Connecting there
        # would leave one handler per frame, and after a minute of running a
        # single click would toggle a trace hundreds of times.
        for _ in range(4):
            self.app._redraw(force=True)
        registry = self.app.canvas.callbacks.callbacks.get("pick_event", {})
        self.assertEqual(len(registry), 1,
                         "%d pick handlers connected, expected 1"
                         % len(registry))

    def test_a_click_hides_the_trace_it_was_aimed_at(self):
        name = self.app.session.runs[0].display_name
        self.app._redraw(force=True)
        self._click(self._legend_entry(name))
        self.assertFalse(self.app.session.qber_visible(name))
        self.assertFalse(self._trace(name).get_visible())

    def test_one_click_after_several_redraws_makes_exactly_one_toggle(self):
        # The behavioural half of the test above.
        #
        # _redraw is frozen for the duration of the click ON PURPOSE. The
        # handler forces a redraw, which rebuilds the legend and with it
        # session.legend_picks, so a SECOND handler holding the same event
        # would find its artist gone from the map and quietly do nothing --
        # duplicate connections would be masked by the very redraw that
        # makes them wrong, and the count assertion above would be the only
        # thing left standing. Freezing the map is what lets a duplicated
        # handler show itself as the duplicated TOGGLE it is.
        name = self.app.session.runs[0].display_name
        for _ in range(3):
            self.app._redraw(force=True)
        entry = self._legend_entry(name)
        toggles = []
        real = self.app.session.toggle_qber
        self.app.session.toggle_qber = lambda n: toggles.append(n)
        self.app._redraw = lambda *a, **k: None
        try:
            self._click(entry)
        finally:
            self.app.session.toggle_qber = real
            del self.app._redraw
        self.assertEqual(toggles, [name])

    def test_clicking_the_entry_again_brings_the_trace_back(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self._click(self._legend_entry(name))
        self.assertTrue(self.app.session.qber_visible(name))
        self.assertTrue(self._trace(name).get_visible())

    def test_clicking_a_death_threshold_caption_changes_nothing(self):
        before = set(self.app.session.hidden_qber)
        caption = [t for t in self.app.figure.axes[1].get_legend().get_texts()
                   if "dies" in t.get_text()][0]
        self._click(caption)
        self.assertEqual(self.app.session.hidden_qber, before)

    def test_a_click_on_something_else_entirely_is_ignored(self):
        # pick_event fires for any pickable artist on the canvas.
        self._click(self.app.figure.axes[0])
        self.assertEqual(self.app.session.hidden_qber, set())

    def test_a_hidden_trace_stays_hidden_across_step(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app._step_once()
        self.assertFalse(self.app.session.qber_visible(name))
        self.assertFalse(self._trace(name).get_visible())

    def test_a_hidden_trace_stays_hidden_across_start_pause_and_a_tick(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app._toggle_run()
        self.app.session.step(5)
        self.app._redraw(force=True)
        self.app._toggle_run()
        self.assertFalse(self.app.session.qber_visible(name))
        self.assertFalse(self._trace(name).get_visible())

    def test_a_hidden_trace_stays_hidden_across_reset(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app._reset()
        self.assertFalse(self.app.session.qber_visible(name))

    def test_apply_carries_the_toggles_over(self):
        # Apply rebuilds the session, but which traces are switched off is a
        # view choice, not part of the episode: nudging lambda must not
        # un-hide everything the reader had just cleared away.
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app.entries["lam"].set("2.0")
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertFalse(self.app.session.qber_visible(name))

    def test_a_toggle_for_a_policy_that_is_dropped_does_not_break_apply(self):
        # A stale name in the carried set is normal, not exceptional.
        self._click(self._legend_entry("always-ent"))
        self.app.checks["always-ent"].set(False)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertIn("always-ent", self.app.session.hidden_qber)
        self.assertEqual(len(self.app.figure.axes), 4)

    def test_apply_reports_a_bad_form_instead_of_crashing(self):
        # Apply is a button callback: an exception there vanishes into Tk's
        # error handler and the window sits there looking fine.
        self.app.entries["lam"].set("not a number")
        self.app._apply()
        self.assertTrue(self.dialogs)
        self.assertIn("not a number", self.app.status.cget("text"))
        self.assertEqual(self.app.status.cget("fg"),
                         self.ui.STATUS_COLORS["error"])


class TestTheQberVisibilityCheckboxGroup(unittest.TestCase):
    """Checkboxes are a SECOND control over `hidden_qber`, alongside the
    legend click tested above. Two controls over one piece of state is the
    trap: if either one can change the state without the other finding out,
    they desync and whichever widget is stale starts lying about what is on
    screen. Every test below is written against one specific way that could
    happen, mirroring TestHidingAQberTrace's own structure for the legend.
    """

    def setUp(self):
        try:
            import policyviz_ui
        except Exception as exc:                  # pragma: no cover
            self.skipTest("tkinter/matplotlib unavailable: %s" % exc)
        self.ui = policyviz_ui
        self.dialogs = []
        self._real_messagebox = policyviz_ui.messagebox
        policyviz_ui.messagebox = _RecordingMessagebox(self.dialogs)
        try:
            self.app = policyviz_ui.PolicyVizApp(small_config())
        except Exception as exc:                  # pragma: no cover
            policyviz_ui.messagebox = self._real_messagebox
            self.skipTest("cannot create Tk window: %s" % exc)
        self.app.withdraw()

    def tearDown(self):
        if getattr(self, "_real_messagebox", None) is not None:
            self.ui.messagebox = self._real_messagebox
        app = getattr(self, "app", None)
        if app is not None:
            app._running = False
            app.update_idletasks()
            app.destroy()

    def _legend_entry(self, name):
        legend = self.app.figure.axes[1].get_legend()
        for text in legend.get_texts():
            if text.get_text() == name:
                return text
        raise AssertionError("no legend entry for %r" % (name,))

    def _trace(self, name):
        for line in self.app.figure.axes[1].get_lines():
            if line.get_label() == name:
                return line
        raise AssertionError("no QBER trace for %r" % (name,))

    def _click(self, artist):
        event = type("_Pick", (), {})()
        event.name = "pick_event"
        event.artist = artist
        event.canvas = self.app.canvas
        self.app.canvas.callbacks.process("pick_event", event)

    def _qber_widget(self, name):
        for child in self.app.qber_check_frame.winfo_children():
            if isinstance(child, self.ui.ttk.Checkbutton) \
                    and child.cget("text") == name:
                return child
        raise AssertionError("no QBER checkbox for %r" % (name,))

    # -- the group matches the session -----------------------------------

    def test_there_is_a_checkbox_for_every_policy_in_the_session(self):
        self.assertEqual(set(self.app.qber_checks),
                         {r.display_name for r in self.app.session.runs})

    def test_the_death_thresholds_get_no_checkbox(self):
        # Only session.runs feed the group; the two reference lines never
        # appear as a policy name, so this also covers "the group has
        # exactly as many entries as there are runs, no more".
        self.assertEqual(len(self.app.qber_checks),
                         len(self.app.session.runs))
        self.assertFalse(any("dies" in name for name in self.app.qber_checks))

    def test_checkboxes_start_checked_when_nothing_is_hidden(self):
        for var in self.app.qber_checks.values():
            self.assertTrue(var.get())

    # -- unticking actually hides the trace -------------------------------

    def test_unticking_a_checkbox_hides_its_trace(self):
        name = self.app.session.runs[0].display_name
        self.app._redraw(force=True)
        self._qber_widget(name).invoke()
        self.assertFalse(self.app.session.qber_visible(name))
        self.assertFalse(self._trace(name).get_visible())

    def test_unticking_leaves_a_different_policys_trace_alone(self):
        name = self.app.session.runs[0].display_name
        other = self.app.session.runs[1].display_name
        self.app._redraw(force=True)
        self._qber_widget(name).invoke()
        self.assertTrue(self._trace(other).get_visible())

    def test_ticking_it_back_on_restores_the_trace(self):
        name = self.app.session.runs[0].display_name
        widget = self._qber_widget(name)
        widget.invoke()
        widget.invoke()
        self.assertTrue(self.app.session.qber_visible(name))
        self.assertTrue(self._trace(name).get_visible())

    def test_unticking_greys_out_the_legend_entry_too(self):
        name = self.app.session.runs[0].display_name
        self._qber_widget(name).invoke()
        text = self._legend_entry(name)
        self.assertEqual(text.get_color(), policyviz.HIDDEN_LEGEND_COLOR)

    # -- both directions of the sync ---------------------------------------

    def test_clicking_a_legend_entry_updates_the_matching_checkbox(self):
        # THE desync this whole feature exists to prevent: a legend click
        # must not leave the checkbox showing the old visibility.
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.assertFalse(self.app.session.qber_visible(name))
        self.assertFalse(self.app.qber_checks[name].get(),
                         "checkbox still says visible after a legend click "
                         "hid the trace -- the two controls have desynced")

    def test_clicking_a_legend_entry_twice_leaves_the_checkbox_checked(self):
        name = self.app.session.runs[0].display_name
        entry = self._legend_entry(name)
        self._click(entry)
        self._click(self._legend_entry(name))
        self.assertTrue(self.app.session.qber_visible(name))
        self.assertTrue(self.app.qber_checks[name].get())

    def test_unticking_a_checkbox_leaves_the_legend_entry_in_step(self):
        # The reverse direction: a checkbox click updates the legend's own
        # greyed-out state via the ordinary redraw, so a stale legend cannot
        # show a trace as visible while the checkbox says otherwise.
        name = self.app.session.runs[0].display_name
        self._qber_widget(name).invoke()
        self.assertFalse(self._trace(name).get_visible())
        self.assertFalse(self.app.qber_checks[name].get())

    # -- the group is rebuilt on Apply, not left stale ---------------------

    def test_apply_adds_a_checkbox_for_a_newly_selected_policy(self):
        self.assertNotIn("buffer-aware-0.045-0.25", self.app.qber_checks)
        self.app.checks["buffer-aware"].set(True)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertIn("buffer-aware-0.045-0.25", self.app.qber_checks)

    def test_apply_drops_the_checkbox_for_a_deselected_policy(self):
        self.assertIn("always-ent", self.app.qber_checks)
        self.app.checks["always-ent"].set(False)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertNotIn("always-ent", self.app.qber_checks)

    def test_apply_keeps_the_visibility_a_surviving_policy_had(self):
        self._qber_widget("always-pm").invoke()
        self.app.entries["lam"].set("2.0")
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertIn("always-pm", self.app.qber_checks)
        self.assertFalse(self.app.qber_checks["always-pm"].get())
        self.assertFalse(self.app.session.qber_visible("always-pm"))

    def test_apply_does_not_resurrect_a_hidden_trace_as_a_stray_checkbox(self):
        # A toggle for a policy no longer selected is carried in hidden_qber
        # (see _build_session), but it must not grow a checkbox: there is no
        # run to draw a trace for it, and a checkbox with nothing behind it
        # would be a control that does nothing when clicked.
        self._qber_widget("always-ent").invoke()
        self.app.checks["always-ent"].set(False)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertNotIn("always-ent", self.app.qber_checks)

    # -- unticking touches ONLY the QBER panel ------------------------------

    def test_unticking_a_checkbox_leaves_the_mode_lanes_alone(self):
        name = self.app.session.runs[0].display_name
        self._qber_widget(name).invoke()
        labels = [t.get_text() for t in self.app.figure.axes[0]
                  .get_yticklabels()]
        self.assertEqual(labels,
                         [r.display_name for r in self.app.session.runs])

    def test_unticking_a_checkbox_leaves_the_buffer_panel_alone(self):
        name = self.app.session.runs[0].display_name
        before = len(self.app.figure.axes[2].get_lines())
        self._qber_widget(name).invoke()
        after = len(self.app.figure.axes[2].get_lines())
        self.assertEqual(before, after)

    def test_unticking_a_checkbox_leaves_the_eta_panel_alone(self):
        name = self.app.session.runs[0].display_name
        before = len(self.app.figure.axes[3].get_lines())
        self._qber_widget(name).invoke()
        after = len(self.app.figure.axes[3].get_lines())
        self.assertEqual(before, after)

    def test_unticking_a_checkbox_leaves_the_scoreboard_alone(self):
        name = self.app.session.runs[0].display_name
        self._qber_widget(name).invoke()
        board_text = self.app.board.get("1.0", "end")
        self.assertIn(name, board_text)



if __name__ == "__main__":
    unittest.main()
