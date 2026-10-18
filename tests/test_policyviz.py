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
import vizstate  # noqa: E402
import plots  # noqa: E402
from evemarkov import EveParams  # noqa: E402
from hybridenv import ENT, PM, HybridEnv  # noqa: E402
from policies import (AlwaysENT, AlwaysPM, BlockAwareGreedy,  # noqa: E402
                      BufferAwareThreshold, DecoyWatch, GreedyThreshold,
                      HysteresisThreshold, Oracle, RelaxedAlwaysPM,
                      run_episode)
from policyviz import (PolicyRun, VizConfig, VizSession, death_thresholds,
                       attack_spans, draw_panels, format_scoreboard,
                       parse_spec)  # noqa: E402

class _RecordingMessagebox:
    """Stands in for tkinter.messagebox: records instead of blocking.

    A real showerror is modal -- it waits for a click -- so leaving it in
    place hangs any unattended run that trips an error path.
    """

    def __init__(self, sink, answer=True):
        self._sink = sink
        self.answer = answer          # what askyesno returns

    def showerror(self, title, message, **kw):
        self._sink.append(("error", title, message))

    def showinfo(self, title, message, **kw):
        self._sink.append(("info", title, message))

    def askyesno(self, title, message, **kw):
        self._sink.append(("askyesno", title, message))
        return self.answer


class _RecordingSimpledialog:
    """Stands in for tkinter.simpledialog: answers instead of blocking.

    askstring is modal in exactly the way showerror is, and "Save as preset"
    reaches it on the happy path -- so an unstubbed one would hang the suite
    on a passing test rather than a failing one.
    """

    def __init__(self, sink, answer=None):
        self._sink = sink
        self.answer = answer

    def askstring(self, title, prompt, **kw):
        self._sink.append(("askstring", title, prompt))
        return self.answer


# Short enough to keep the suite quick, long enough that Eve leaves the QUIET
# state and the policies genuinely diverge (mean dwell is 40 slots).
SLOTS = 300


def small_config(**kw):
    """A four-track session that runs in well under a second.

    Pinned to the JAMMING adversary: every test below that was written before
    the window could choose its attacker asserts behaviour -- greedy switching,
    policies diverging under Eve -- that only the jammer produces. The window's
    own default is now the stealthy one; those tests say so explicitly.
    """
    base = dict(specs=("always-pm", "always-ent", "greedy-0.045", "oracle"),
                horizon=SLOTS, lam=2.5, b_max=1_000_000, adversary="jam")
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
        for spec in ("greedy-0.045", "buffer-aware-0.045-0.25",
                     "block-aware", "block-aware-80",
                     "hysteresis-0.050-0.030-20"):
            self.assertEqual(parse_spec(spec).name, spec)

    def test_block_aware_parses_with_and_without_a_horizon(self):
        bare = parse_spec("block-aware")
        self.assertIsInstance(bare, BlockAwareGreedy)
        self.assertIsNone(bare.horizon)
        self.assertEqual(parse_spec("block-aware-80").horizon, 80)

    def test_hysteresis_carries_all_three_parameters_through(self):
        policy = parse_spec("hysteresis-0.050-0.030-20")
        self.assertIsInstance(policy, HysteresisThreshold)
        self.assertAlmostEqual(policy.up, 0.05)
        self.assertAlmostEqual(policy.down, 0.03)
        self.assertEqual(policy.min_dwell, 20)

    def test_an_unknown_spec_is_refused(self):
        with self.assertRaises(ValueError):
            parse_spec("greedy")

    def test_dqn_is_no_longer_a_spec_the_window_accepts(self):
        # The DQN was taken out of the visualiser; a saved configuration or a
        # command line still naming it must be refused rather than quietly
        # drawn as something else.
        with self.assertRaises(ValueError):
            parse_spec("dqn")


class TestTheBlockAwarePolicyIsReachableFromTheWindow(unittest.TestCase):
    """The derived rule is the point of the exercise; a policy that can only
    be run from a script is not one the user can see working."""

    def test_it_is_offered_as_a_checkbox(self):
        self.assertIn("block-aware", policyviz.SPEC_CHOICES)

    def test_a_parameterised_spec_still_ticks_the_plain_checkbox(self):
        # --policies block-aware-80 must not open a window with nothing
        # ticked next to a track that is running.
        from policyviz_ui import PolicyVizApp

        self.assertIn("block-aware", PolicyVizApp._stems(("block-aware-80",)))

    def test_it_runs_in_a_session_alongside_the_bounds(self):
        # The model-based rule reads kappa()/error_floor/sifted_per_slot off
        # the environment; the session builds one env per run, so this is
        # where a policy bound to the wrong one would show up.
        session = VizSession(small_config(
            specs=("always-pm", "block-aware", "oracle")))
        for _ in range(12):
            session.step()
        names = [run.display_name for run in session.runs]
        self.assertIn("block-aware", names)
        for row in session.scores():
            self.assertGreaterEqual(row["eta"], 0.0)


class TestTheAdversaryIsChosenNotHardwired(unittest.TestCase):
    """The window used to build EveParams() unconditionally -- the jammer --
    so the attack the study now turns on could not be shown at all, whatever
    the form or the saved configuration said."""

    def test_the_session_faces_the_adversary_it_was_configured_with(self):
        session = VizSession(small_config(adversary="pns",
                                          specs=("always-pm",)))
        eve = session.runs[0].env.eve.params
        self.assertEqual(set(eve.f_levels), {0.0})
        self.assertGreater(max(eve.s_levels), 0.0)

    def test_jam_is_exactly_the_weather_the_window_always_used(self):
        session = VizSession(small_config(adversary="jam",
                                          specs=("always-pm",)))
        self.assertEqual(session.runs[0].env.eve.params,
                         EveParams(seed=session.seed))

    def test_an_unknown_adversary_is_refused_not_defaulted(self):
        with self.assertRaises(KeyError):
            VizSession(small_config(adversary="stealthy",
                                    specs=("always-pm",)))

    def test_every_run_still_faces_one_and_the_same_adversary(self):
        session = VizSession(small_config(adversary="pns"))
        first = session.runs[0].env.eve.params
        for run in session.runs[1:]:
            self.assertEqual(run.env.eve.params, first)

    def test_each_slot_records_the_attack_strength(self):
        # Without it the figure has nothing to draw the attack FROM: under PNS
        # the QBER is flat and f is zero on every slot.
        session = VizSession(small_config(adversary="pns",
                                          specs=("always-pm",), horizon=1200))
        session.step(1200)
        run = session.runs[0]
        self.assertEqual(len(run.pns), run.slot)
        self.assertTrue(any(level > 0 for level in run.pns))


class TestTheAttackIsDrawnWhereItHappens(unittest.TestCase):
    """Under PNS the QBER does not move, so a figure that only showed the QBER
    would show NOTHING happening while PM's key went to zero. The attack
    periods have to be drawn in their own right."""

    class _Run:
        t = [1, 2, 3, 4, 5, 6]
        f = [0.0, 0.0, 0.03, 0.03, 0.0, 0.0]
        pns = [0.0, 0.6, 0.6, 0.0, 0.0, 0.6]

    def test_pns_spans_cover_exactly_the_slots_it_ran(self):
        self.assertEqual(attack_spans(self._Run())["pns"], [(2, 2), (6, 1)])

    def test_jamming_spans_are_kept_apart_from_pns(self):
        self.assertEqual(attack_spans(self._Run())["jam"], [(3, 2)])

    def test_an_empty_run_has_no_spans(self):
        class _Empty:
            t, f, pns = [], [], []
        self.assertEqual(attack_spans(_Empty()), {"jam": [], "pns": []})

    def test_the_header_names_the_adversary(self):
        fig = Figure(figsize=(8, 7))
        FigureCanvasAgg(fig)
        session = VizSession(small_config(adversary="pns",
                                          specs=("always-pm",), horizon=20))
        session.step(20)
        draw_panels(fig, session)
        self.assertIn("pns", fig._suptitle.get_text())

    def test_the_figure_still_has_four_panels(self):
        fig = Figure(figsize=(8, 7))
        FigureCanvasAgg(fig)
        session = VizSession(small_config(adversary="pns",
                                          specs=("always-pm",), horizon=1200))
        session.step(1200)
        draw_panels(fig, session)
        self.assertEqual(len(fig.axes), 4)


class TestTheDecoyPolicyIsOffered(unittest.TestCase):
    def test_decoy_is_a_checkbox_and_dqn_is_not(self):
        self.assertIn("decoy", policyviz.SPEC_CHOICES)
        self.assertNotIn("dqn", policyviz.SPEC_CHOICES)

    def test_a_decoy_spec_ticks_the_decoy_checkbox(self):
        from policyviz_ui import PolicyVizApp

        self.assertEqual(PolicyVizApp._stems(("decoy-0.100-1.000-60",)),
                         ["decoy"])

    def test_the_decoy_boxes_are_seeded_from_an_incoming_spec(self):
        from policyviz_ui import PolicyVizApp

        self.assertEqual(
            PolicyVizApp._decoy_parameters_from(("decoy-0.250-0.018-90",)),
            ("0.25", "0.018", "90"))

    def test_absent_a_decoy_spec_the_boxes_keep_their_defaults(self):
        from policyviz_ui import PolicyVizApp

        self.assertEqual(PolicyVizApp._decoy_parameters_from(("always-pm",)),
                         ("0.1", "1", "60"))


class TestTheConfigHasNoDqnKnobs(unittest.TestCase):
    def test_there_is_no_episode_count_to_set(self):
        self.assertFalse(hasattr(VizConfig(), "dqn_episodes"))

    def test_episode_index_zero_is_the_first_evaluation_seed(self):
        # With no training block to steer around, the index maps straight onto
        # the study's evaluation seeds -- still well clear of any seed a
        # training run would draw.
        from eval_hybrid import evaluation_seeds

        self.assertEqual(VizConfig().env_seed(), evaluation_seeds(1)[0])


class TestTheDefaultsShowTheResultOnTheFirstApply(unittest.TestCase):
    """The request, as a test: open the window, press Apply without touching a
    box, and the episode on screen shows the finding.

    Measured on this exact default (seed index 0, 3600 slots):
        always-pm 0.5836   always-ent 0.6383   greedy 0.5836
        block-aware 0.5836   decoy 0.6807   oracle 0.7060
    """

    @classmethod
    def setUpClass(cls):
        cls.config = VizConfig()
        cls.session = VizSession(cls.config)
        cls.session.step(cls.session.params.horizon)
        cls.eta = {run.agent.name: run.env.eta for run in cls.session.runs}

    def test_the_defaults_are_the_stealthy_scenario(self):
        self.assertEqual(self.config.adversary, "pns")
        self.assertEqual(self.config.ent_preset, "ent_advanced")
        self.assertAlmostEqual(self.config.lam, 4.0)

    def test_the_default_policies_include_the_one_that_can_see_it(self):
        names = set(self.eta)
        self.assertTrue(any(n.startswith("decoy-") for n in names))
        self.assertIn("always-pm", names)
        self.assertIn("always-ent", names)
        self.assertIn("oracle", names)

    def test_decoy_beats_both_static_choices(self):
        decoy = max(v for k, v in self.eta.items() if k.startswith("decoy-"))
        self.assertGreater(decoy,
                           max(self.eta["always-pm"], self.eta["always-ent"]))

    def test_the_qber_policies_are_exactly_always_pm(self):
        # Blind, not merely worse: they never see anything to react to.
        for name, value in self.eta.items():
            if name.startswith("greedy-") or name.startswith("block-aware"):
                self.assertEqual(value, self.eta["always-pm"], name)

    def test_the_oracle_still_sits_above_everything_realisable(self):
        realisable = [v for k, v in self.eta.items() if k != "oracle"]
        self.assertGreater(self.eta["oracle"], max(realisable))


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
        self.session.hidden_traces.add(self.name)
        self._draw()
        self.assertFalse(self._traces()[self.name].get_visible(),
                         "%s is hidden but its QBER trace is still drawn"
                         % (self.name,))

    def _eta_lines(self):
        """The cumulative-eta panel's lines, keyed by run label."""
        runs = self.session.runs
        return {r.display_name: ln
                for r, ln in zip(runs, self.fig.axes[3].get_lines())}

    def test_hiding_a_policy_takes_its_line_off_the_eta_panel_too(self):
        # The eta curves bunch as tightly as the QBER traces, so the same
        # control has to reach both or the panel the reader cares about most
        # stays unreadable.
        self.session.hidden_traces.add(self.name)
        self._draw()
        self.assertFalse(self._eta_lines()[self.name].get_visible(),
                         "%s is hidden but its eta curve is still drawn"
                         % (self.name,))

    def test_hiding_one_policy_leaves_the_other_eta_curves_drawn(self):
        self.session.hidden_traces.add(self.name)
        self._draw()
        self.assertTrue(self._eta_lines()[self.other].get_visible())

    def test_hiding_a_line_never_touches_the_mode_lanes(self):
        # The mode lanes are the evidence that a policy is RUNNING. If the
        # visibility control reached them, a reader could hide a policy from
        # the line panels and conclude it had never been selected.
        self.session.hidden_traces.add(self.name)
        self._draw()
        labels = [t.get_text() for t in self.fig.axes[0].get_yticklabels()]
        self.assertIn(self.name, labels)

    def test_hiding_one_policy_leaves_the_others_drawn(self):
        self.session.hidden_traces.add(self.name)
        self._draw()
        self.assertTrue(self._traces()[self.other].get_visible())

    def test_toggling_twice_brings_the_trace_back(self):
        self.session.toggle_trace(self.name)
        self.session.toggle_trace(self.name)
        self._draw()
        self.assertTrue(self._traces()[self.name].get_visible())

    def test_toggle_reports_the_visibility_it_leaves_behind(self):
        self.assertFalse(self.session.toggle_trace(self.name))
        self.assertFalse(self.session.trace_visible(self.name))
        self.assertTrue(self.session.toggle_trace(self.name))
        self.assertTrue(self.session.trace_visible(self.name))

    # -- the state cannot live on the artist ---------------------------

    def test_the_hidden_state_survives_the_next_draw(self):
        # THE one that matters. The window redraws on a timer, and
        # draw_panels clears the figure and builds new lines every time, so a
        # visibility flag left on a Line2D would be gone within a fraction of
        # a second and the user's click would silently undo itself while the
        # episode ran.
        self._draw()
        self.session.toggle_trace(self.name)
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
        self.session.toggle_trace(self.name)
        self._draw()
        other = Figure(figsize=(9, 8))
        FigureCanvasAgg(other)
        draw_panels(other, self.session)
        drawn = {ln.get_label(): ln for ln in other.axes[1].get_lines()}
        self.assertFalse(drawn[self.name].get_visible())

    def test_reset_keeps_the_toggles(self):
        # Reset rewinds the EPISODE. Which traces the reader switched off is
        # not part of the trajectory.
        self.session.toggle_trace(self.name)
        self.session.reset()
        self.assertFalse(self.session.trace_visible(self.name))

    # -- it must not leak out of panel 2 -------------------------------

    def test_hiding_a_trace_leaves_the_mode_lanes_alone(self):
        # A reader who could hide a policy from the QBER panel and thereby
        # lose its lane would conclude it was never selected.
        self.session.hidden_traces.add(self.name)
        self._draw()
        labels = [t.get_text() for t in self.fig.axes[0].get_yticklabels()]
        self.assertEqual(labels,
                         [r.display_name for r in self.session.runs])

    def test_hiding_a_trace_leaves_the_mode_bands_drawn(self):
        self.session.hidden_traces.add(self.name)
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

    def test_hiding_a_trace_hides_its_buffer_line(self):
        # This test used to assert the opposite -- that the buffer panel was
        # untouched. The trace toggle was deliberately widened to cover it,
        # because unticking a policy left every buffer line on screen and
        # the panel overlapped unreadably with six tracks on it. What must
        # NOT widen is the mode lanes and the scoreboard: those are the
        # evidence a policy is running at all.
        self._draw()
        hidden = [ln for ln in self.fig.axes[2].get_lines()
                  if ln.get_visible()]
        self.assertEqual(len(hidden), len(self.session.runs))
        self.session.hidden_traces.add(self.name)
        self._draw()
        drawn = [ln for ln in self.fig.axes[2].get_lines()
                 if ln.get_visible()]
        self.assertEqual(len(drawn), len(self.session.runs) - 1)
        runs = self.session.runs
        hidden_index = [i for i, r in enumerate(runs)
                        if r.display_name == self.name][0]
        for i, ln in enumerate(self.fig.axes[2].get_lines()):
            expected = i != hidden_index
            self.assertEqual(ln.get_visible(), expected,
                             "run #%d (%s) visible=%s"
                             % (i, runs[i].display_name, ln.get_visible()))

    def test_hiding_a_trace_hides_exactly_one_eta_curve(self):
        # This test used to assert the opposite -- that the eta panel was
        # untouched. The control was deliberately widened to cover it, on the
        # grounds that the eta curves bunch as tightly as the QBER traces.
        # What must NOT widen is the mode lanes and the scoreboard: those are
        # the evidence a policy is running at all.
        self.session.hidden_traces.add(self.name)
        self._draw()
        drawn = [ln for ln in self.fig.axes[3].get_lines()
                 if ln.get_visible()]
        self.assertEqual(len(drawn), len(self.session.runs) - 1)

    def test_hiding_a_trace_leaves_the_scoreboard_alone(self):
        self.session.hidden_traces.add(self.name)
        self._draw()
        board = format_scoreboard(self.session)
        for run in self.session.runs:
            self.assertIn(run.display_name, board)

    # -- the legend has to show its own state --------------------------

    def test_a_hidden_policy_keeps_its_legend_entry(self):
        # It is the only way back. An entry that vanished with its trace
        # would make the hiding irreversible.
        self.session.hidden_traces.add(self.name)
        self._draw()
        texts = [t.get_text()
                 for t in self.fig.axes[1].get_legend().get_texts()]
        self.assertIn(self.name, texts)

    def test_a_hidden_legend_entry_is_greyed_out(self):
        self.session.hidden_traces.add(self.name)
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
        self.session.hidden_traces.add(self.name)
        self._draw()
        handle, text = self._entry(self.other)
        self.assertNotEqual(text.get_color(), policyviz.HIDDEN_LEGEND_COLOR)
        self.assertIn(text.get_alpha(), (None, 1.0))
        self.assertIn(handle.get_alpha(), (None, 1.0))

    def test_the_greying_is_undone_when_the_trace_comes_back(self):
        self.session.toggle_trace(self.name)
        self._draw()
        self.session.toggle_trace(self.name)
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
        session.hidden_traces.add(tall)
        draw_panels(fig, session)
        self.assertEqual(fig.axes[1].get_ylim(), before)

    def test_the_limit_is_the_one_the_hidden_trace_asks_for(self):
        # Pins the value, not just its stability: an axis frozen at some
        # other number would also "not move".
        session, tall = self._session_with_one_tall_trace()
        session.hidden_traces.add(tall)
        fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(fig)
        draw_panels(fig, session)
        peak = max([100.0 * q for r in session.runs for q in r.qber]
                   + [100.0 * session.death_pm, 100.0 * session.death_ent])
        self.assertAlmostEqual(fig.axes[1].get_ylim()[1], peak * 1.45,
                               places=6)

    def test_hiding_every_trace_still_renders(self):
        for run in self.session.runs:
            self.session.hidden_traces.add(run.display_name)
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
            self.session.hidden_traces.add(run.display_name)
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
        self.session.hidden_traces.add("a policy that is not selected")
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

    # ---- the attacker, the decoy policy, and no DQN ----------------------

    def test_the_form_starts_on_the_configured_adversary(self):
        # small_config pins the jammer; the form must not quietly show the
        # window's own default instead.
        self.assertEqual(self.app.adversary_var.get(), "jam")

    def test_the_adversary_on_the_form_reaches_the_configuration(self):
        self.app.adversary_var.set("pns")
        self.assertEqual(self.app._read_config().adversary, "pns")

    def test_every_named_adversary_is_offered(self):
        from evemarkov import ADVERSARIES

        self.assertEqual(set(self.app.adversary_choices), set(ADVERSARIES))

    def test_the_decoy_checkbox_builds_the_policys_own_label(self):
        self.app.checks["decoy"].set(True)
        self.app.dc_theta.set("0.25")
        self.app.dc_qber.set("0.018")
        self.app.dc_probe.set("90")
        config = self.app._read_config()
        spec = [s for s in config.specs if s.startswith("decoy-")][0]
        self.assertEqual(spec, DecoyWatch(0.25, qber_threshold=0.018,
                                          probe_every=90).name)
        self.assertEqual(parse_spec(spec).name, spec)

    def test_a_bad_decoy_number_is_refused(self):
        self.app.checks["decoy"].set(True)
        self.app.dc_probe.set("often")
        with self.assertRaises(ValueError):
            self.app._read_config()

    def test_there_is_no_dqn_anywhere_on_the_form(self):
        self.assertNotIn("dqn", self.app.checks)
        self.assertNotIn("dqn_episodes", self.app.entries)

    def test_save_captures_the_adversary_and_the_decoy_boxes(self):
        self.app.adversary_var.set("both")
        self.app.dc_theta.set("0.3")
        state = self.app._collect_state()
        self.assertEqual(state["adversary"], "both")
        self.assertAlmostEqual(state["dc_theta"], 0.3)
        self.assertIn("dc_probe", state)

    def test_restore_puts_the_adversary_and_the_decoy_boxes_back(self):
        self.app._apply_state({"fields": {}, "adversary": "none",
                               "dc_probe": 150.0})
        self.assertEqual(self.app.adversary_var.get(), "none")
        self.assertEqual(float(self.app.dc_probe.get()), 150.0)

    def test_a_saved_file_from_before_the_dqn_was_removed_still_restores(self):
        # It carries dqn_episodes and a "dqn" policy stem. Neither may stop
        # the window, and neither may come back.
        self.app._apply_state({"fields": {"dqn_episodes": 20, "lam": 3.0},
                               "policies": ["always-pm", "dqn"]})
        self.assertEqual(float(self.app.entries["lam"].get()), 3.0)
        self.assertNotIn("dqn", self.app._read_config().specs)

    def test_apply_builds_a_session_without_training_anything(self):
        self.app._apply()
        self.assertIsNotNone(self.app.session)
        self.assertEqual(self.app.session.config.adversary,
                         self.app.adversary_var.get())

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
        self.assertFalse(self.app.session.trace_visible(name))
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
        real = self.app.session.toggle_trace
        self.app.session.toggle_trace = lambda n: toggles.append(n)
        self.app._redraw = lambda *a, **k: None
        try:
            self._click(entry)
        finally:
            self.app.session.toggle_trace = real
            del self.app._redraw
        self.assertEqual(toggles, [name])

    def test_clicking_the_entry_again_brings_the_trace_back(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self._click(self._legend_entry(name))
        self.assertTrue(self.app.session.trace_visible(name))
        self.assertTrue(self._trace(name).get_visible())

    def test_clicking_a_death_threshold_caption_changes_nothing(self):
        before = set(self.app.session.hidden_traces)
        caption = [t for t in self.app.figure.axes[1].get_legend().get_texts()
                   if "dies" in t.get_text()][0]
        self._click(caption)
        self.assertEqual(self.app.session.hidden_traces, before)

    def test_a_click_on_something_else_entirely_is_ignored(self):
        # pick_event fires for any pickable artist on the canvas.
        self._click(self.app.figure.axes[0])
        self.assertEqual(self.app.session.hidden_traces, set())

    def test_a_hidden_trace_stays_hidden_across_step(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app._step_once()
        self.assertFalse(self.app.session.trace_visible(name))
        self.assertFalse(self._trace(name).get_visible())

    def test_a_hidden_trace_stays_hidden_across_start_pause_and_a_tick(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app._toggle_run()
        self.app.session.step(5)
        self.app._redraw(force=True)
        self.app._toggle_run()
        self.assertFalse(self.app.session.trace_visible(name))
        self.assertFalse(self._trace(name).get_visible())

    def test_a_hidden_trace_stays_hidden_across_reset(self):
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app._reset()
        self.assertFalse(self.app.session.trace_visible(name))

    def test_apply_carries_the_toggles_over(self):
        # Apply rebuilds the session, but which traces are switched off is a
        # view choice, not part of the episode: nudging lambda must not
        # un-hide everything the reader had just cleared away.
        name = self.app.session.runs[0].display_name
        self._click(self._legend_entry(name))
        self.app.entries["lam"].set("2.0")
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertFalse(self.app.session.trace_visible(name))

    def test_a_toggle_for_a_policy_that_is_dropped_does_not_break_apply(self):
        # A stale name in the carried set is normal, not exceptional.
        self._click(self._legend_entry("always-ent"))
        self.app.checks["always-ent"].set(False)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertIn("always-ent", self.app.session.hidden_traces)
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
    """Checkboxes are a SECOND control over `hidden_traces`, alongside the
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
        for child in self.app.trace_check_frame.winfo_children():
            if isinstance(child, self.ui.ttk.Checkbutton) \
                    and child.cget("text") == name:
                return child
        raise AssertionError("no QBER checkbox for %r" % (name,))

    # -- the group matches the session -----------------------------------

    def test_there_is_a_checkbox_for_every_policy_in_the_session(self):
        self.assertEqual(set(self.app.trace_checks),
                         {r.display_name for r in self.app.session.runs})

    def test_the_death_thresholds_get_no_checkbox(self):
        # Only session.runs feed the group; the two reference lines never
        # appear as a policy name, so this also covers "the group has
        # exactly as many entries as there are runs, no more".
        self.assertEqual(len(self.app.trace_checks),
                         len(self.app.session.runs))
        self.assertFalse(any("dies" in name for name in self.app.trace_checks))

    def test_checkboxes_start_checked_when_nothing_is_hidden(self):
        for var in self.app.trace_checks.values():
            self.assertTrue(var.get())

    # -- unticking actually hides the trace -------------------------------

    def test_unticking_a_checkbox_hides_its_trace(self):
        name = self.app.session.runs[0].display_name
        self.app._redraw(force=True)
        self._qber_widget(name).invoke()
        self.assertFalse(self.app.session.trace_visible(name))
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
        self.assertTrue(self.app.session.trace_visible(name))
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
        self.assertFalse(self.app.session.trace_visible(name))
        self.assertFalse(self.app.trace_checks[name].get(),
                         "checkbox still says visible after a legend click "
                         "hid the trace -- the two controls have desynced")

    def test_clicking_a_legend_entry_twice_leaves_the_checkbox_checked(self):
        name = self.app.session.runs[0].display_name
        entry = self._legend_entry(name)
        self._click(entry)
        self._click(self._legend_entry(name))
        self.assertTrue(self.app.session.trace_visible(name))
        self.assertTrue(self.app.trace_checks[name].get())

    def test_unticking_a_checkbox_leaves_the_legend_entry_in_step(self):
        # The reverse direction: a checkbox click updates the legend's own
        # greyed-out state via the ordinary redraw, so a stale legend cannot
        # show a trace as visible while the checkbox says otherwise.
        name = self.app.session.runs[0].display_name
        self._qber_widget(name).invoke()
        self.assertFalse(self._trace(name).get_visible())
        self.assertFalse(self.app.trace_checks[name].get())

    # -- the group is rebuilt on Apply, not left stale ---------------------

    def test_apply_adds_a_checkbox_for_a_newly_selected_policy(self):
        self.assertNotIn("buffer-aware-0.045-0.25", self.app.trace_checks)
        self.app.checks["buffer-aware"].set(True)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertIn("buffer-aware-0.045-0.25", self.app.trace_checks)

    def test_apply_drops_the_checkbox_for_a_deselected_policy(self):
        self.assertIn("always-ent", self.app.trace_checks)
        self.app.checks["always-ent"].set(False)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertNotIn("always-ent", self.app.trace_checks)

    def test_apply_keeps_the_visibility_a_surviving_policy_had(self):
        self._qber_widget("always-pm").invoke()
        self.app.entries["lam"].set("2.0")
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertIn("always-pm", self.app.trace_checks)
        self.assertFalse(self.app.trace_checks["always-pm"].get())
        self.assertFalse(self.app.session.trace_visible("always-pm"))

    def test_apply_does_not_resurrect_a_hidden_trace_as_a_stray_checkbox(self):
        # A toggle for a policy no longer selected is carried in hidden_traces
        # (see _build_session), but it must not grow a checkbox: there is no
        # run to draw a trace for it, and a checkbox with nothing behind it
        # would be a control that does nothing when clicked.
        self._qber_widget("always-ent").invoke()
        self.app.checks["always-ent"].set(False)
        self.app._apply()
        self.assertEqual(self.dialogs, [])
        self.assertNotIn("always-ent", self.app.trace_checks)

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
# ==========================================================================
# EDITABLE OPERATING POINTS
#
# The preset chooser used to be the whole story: a name in a dropdown, and
# the physics behind it fixed on disk. Now the numbers can be typed, which
# opens three ways for the window to be quietly wrong -- each has a class
# below, and each of those has at least one test that goes red when the
# property is removed (see the mutation table in the task report):
#
#   * the edits are COLLECTED but never REACH the environment, so the run
#     shows the old preset under a button that says "custom";
#   * the DQN cache is keyed on the preset NAME, so an agent trained on the
#     old mu is silently reused for the new one and the curve looks fine;
#   * a per-mode fibre reaches _setup_modes, which is the trap CLAUDE.md
#     records having been published once and retracted.
# ==========================================================================


def _ent_env(**kw):
    """ent_advanced with the visualiser's own fibre, plus overrides."""
    base = dataclasses.replace(presets.load_ent_presets()["ent_advanced"],
                               d_km=55.0, alpha=0.2, odr_losses=0.0)
    return dataclasses.replace(base, **kw)


def _pm_env(**kw):
    base = dataclasses.replace(presets.load_presets()["aurea"],
                               d_km=55.0, alpha=0.2, odr_losses=0.0)
    return dataclasses.replace(base, **kw)


class TestEditedParametersReachTheEnvironment(unittest.TestCase):
    """Collecting an edit and running it are two different things.

    A window that stores the numbers, labels its button "custom" and then
    builds the environment from the preset file would look completely correct
    on screen and report the wrong physics.
    """

    def test_an_edited_ent_parameter_changes_the_ent_physics(self):
        plain = HybridEnv(small_config().hybrid_params())
        edited = HybridEnv(small_config(
            ent_env=_ent_env(mu=0.015)).hybrid_params())
        self.assertNotEqual(plain.sifted_per_slot[ENT],
                            edited.sifted_per_slot[ENT])
        self.assertNotAlmostEqual(plain.error_floor[ENT],
                                  edited.error_floor[ENT], places=4)

    def test_an_edited_pm_parameter_changes_the_pm_physics(self):
        plain = HybridEnv(small_config().hybrid_params())
        edited = HybridEnv(small_config(
            pm_env=_pm_env(eta_bob=0.05)).hybrid_params())
        self.assertNotEqual(plain.sifted_per_slot[PM],
                            edited.sifted_per_slot[PM])

    def test_editing_one_mode_leaves_the_other_alone(self):
        plain = HybridEnv(small_config().hybrid_params())
        edited = HybridEnv(small_config(
            ent_env=_ent_env(mu=0.015)).hybrid_params())
        self.assertEqual(plain.sifted_per_slot[PM], edited.sifted_per_slot[PM])

    def test_an_env_equal_to_the_preset_reproduces_the_preset_run(self):
        # The normalisation the window relies on: OK-ing an untouched dialog
        # must not perturb anything.
        plain = HybridEnv(small_config().hybrid_params())
        same = HybridEnv(small_config(ent_env=_ent_env(),
                                      pm_env=_pm_env()).hybrid_params())
        self.assertEqual(plain.sifted_per_slot, same.sifted_per_slot)
        self.assertEqual(plain.error_floor, same.error_floor)

    def test_no_env_still_loads_the_named_preset(self):
        # Backwards compatibility: every caller that predates this feature
        # passes preset names only.
        from hybridenv import HybridParams
        env = HybridEnv(HybridParams(seed=0))
        self.assertEqual(env.sifted_per_slot, (69422, 3485))

    def test_the_session_runs_the_edited_operating_point(self):
        # End to end through VizSession, not just HybridParams: the config
        # has to carry the env all the way into the runs.
        session = VizSession(small_config(specs=("always-ent",),
                                          ent_env=_ent_env(mu=0.015)))
        self.assertEqual(session.runs[0].env.sifted_per_slot[ENT],
                         HybridEnv(small_config(ent_env=_ent_env(mu=0.015))
                                   .hybrid_params()).sifted_per_slot[ENT])

    def test_the_death_threshold_follows_the_edit(self):
        # Panel 2's line is what a trace is read AGAINST. Drawing it from the
        # preset file while the run uses edited parameters would show a dead
        # trace sitting comfortably under a threshold that does not apply.
        plain = death_thresholds(small_config())[0]
        edited = death_thresholds(small_config(
            pm_env=_pm_env(mu1=0.5, mu2=0.1, p1=0.7)))[0]
        self.assertNotAlmostEqual(plain, edited, places=4)


class TestTheSharedFibreIsNeverPerMode(unittest.TestCase):
    """Both modes ride the SAME fibre.

    CLAUDE.md records a comparison of two presets carrying different channels
    being published once and retracted: PM was handicapped ~20x and the
    conclusion "ENT matches PM" was wrong. Making the parameters editable
    reopens that door, so it is nailed shut in three places -- the dialog
    disables the fields, resolve_envs re-imposes the main window's link, and
    _setup_modes overrides whatever it is handed.
    """

    def test_a_per_mode_fibre_in_an_edited_env_is_overridden(self):
        honest = HybridEnv(small_config(pm_env=_pm_env()).hybrid_params())
        smuggled = HybridEnv(small_config(
            pm_env=_pm_env(d_km=0.0, alpha=0.0,
                           odr_losses=0.0)).hybrid_params())
        self.assertEqual(honest.sifted_per_slot, smuggled.sifted_per_slot)

    def test_it_is_the_main_windows_fibre_that_wins(self):
        near = HybridEnv(small_config(
            d_km=0.0, pm_env=_pm_env(d_km=200.0)).hybrid_params())
        far = HybridEnv(small_config(
            d_km=200.0, pm_env=_pm_env(d_km=0.0)).hybrid_params())
        self.assertGreater(near.sifted_per_slot[PM], far.sifted_per_slot[PM])

    def test_resolve_envs_stamps_the_configs_link_on_both_modes(self):
        config = small_config(d_km=17.0, pm_env=_pm_env(d_km=1.0),
                              ent_env=_ent_env(d_km=2.0))
        pm_env, ent_env = policyviz.resolve_envs(config)
        for env in (pm_env, ent_env):
            self.assertEqual(env.d_km, 17.0)
            self.assertEqual(env.alpha, config.alpha)
            self.assertEqual(env.odr_losses, config.odr_losses)

    def test_a_shared_field_typed_into_the_form_is_ignored(self):
        # parse_params takes the fibre from `shared`, never from the fields,
        # so even a disabled entry that somehow got edited cannot leak.
        env = policyviz.parse_params(
            "pm", {"mu1": "0.29", "d_km": "999"},
            {"d_km": 4.0, "alpha": 0.2, "odr_losses": 0.0})
        self.assertEqual(env.d_km, 4.0)

    def test_the_shared_fields_are_named_as_shared(self):
        for protocol in ("pm", "ent"):
            shared = {n for n, _, is_shared
                      in policyviz.param_fields(protocol) if is_shared}
            self.assertEqual(shared, set(policyviz.SHARED_CHANNEL_FIELDS))


class TestParsingTheParameterFields(unittest.TestCase):
    """Bad input must produce a message, never a traceback and never a
    half-applied configuration."""

    SHARED = {"d_km": 55.0, "alpha": 0.2, "odr_losses": 0.0}

    def _parse(self, protocol, **values):
        return policyviz.parse_params(protocol, values, self.SHARED)

    def test_an_empty_field_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            self._parse("ent", mu="")
        self.assertIn("mu", str(ctx.exception))

    def test_text_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            self._parse("ent", mu="quite a lot")
        self.assertIn("quite a lot", str(ctx.exception))

    def test_a_negative_intensity_is_refused(self):
        with self.assertRaises(ValueError):
            self._parse("ent", mu="-0.01")

    def test_an_efficiency_above_one_is_refused(self):
        with self.assertRaises(ValueError):
            self._parse("ent", eta_a="1.5")

    def test_a_zero_efficiency_is_refused(self):
        with self.assertRaises(ValueError):
            self._parse("ent", eta_bob="0")

    def test_an_error_rate_at_one_half_is_refused(self):
        # h2(0.5) = 1: there is no key there and no message worth sending.
        with self.assertRaises(ValueError):
            self._parse("pm", e_det="0.5")

    def test_infinity_is_refused(self):
        with self.assertRaises(ValueError):
            self._parse("ent", f_rep="inf")

    def test_a_decoy_at_or_above_the_signal_is_refused(self):
        # The 1-decoy bound reads the single-photon yield off the DIFFERENCE
        # between the intensities; at mu2 >= mu1 it bounds nothing.
        with self.assertRaises(ValueError) as ctx:
            self._parse("pm", mu1="0.1", mu2="0.2")
        self.assertIn("mu2", str(ctx.exception))
        with self.assertRaises(ValueError):
            self._parse("pm", mu1="0.1", mu2="0.1")

    def test_an_unknown_field_is_refused(self):
        with self.assertRaises(ValueError):
            self._parse("ent", not_a_parameter="1")

    def test_a_good_set_of_values_survives_verbatim(self):
        env = self._parse("ent", mu="0.0125", eta_a="0.81")
        self.assertAlmostEqual(env.mu, 0.0125)
        self.assertAlmostEqual(env.eta_a, 0.81)

    def test_every_preset_round_trips_through_the_dialogs_formatting(self):
        # "Save as preset" writes what the fields show, so a preset that does
        # not survive being displayed would be silently rewritten on save.
        for protocol, table in (("pm", presets.load_presets()),
                                ("ent", presets.load_ent_presets())):
            for name, env in table.items():
                values = {f: policyviz.format_param(getattr(env, f))
                          for f, _l, sh in policyviz.param_fields(protocol)
                          if not sh}
                shared = {f: getattr(env, f)
                          for f in policyviz.SHARED_CHANNEL_FIELDS}
                self.assertEqual(
                    policyviz.parse_params(protocol, values, shared), env,
                    "%s preset %s did not round-trip" % (protocol, name))


class TestPresetNameValidation(unittest.TestCase):
    def test_the_reserved_name_is_refused(self):
        # presets._load RAISES if presets/default.json exists, so accepting
        # this would not make a bad preset -- it would break every later
        # preset load in the process, this window's own redraws included.
        with self.assertRaises(ValueError):
            policyviz.validate_preset_name("default")

    def test_an_empty_name_is_refused(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                policyviz.validate_preset_name(bad)

    def test_a_path_separator_is_refused(self):
        for bad in ("../aurea", "a/b", "a\\b"):
            with self.assertRaises(ValueError):
                policyviz.validate_preset_name(bad)

    def test_surrounding_space_is_trimmed(self):
        self.assertEqual(policyviz.validate_preset_name("  my_point "),
                         "my_point")

    def test_ordinary_names_pass(self):
        for good in ("my_point", "ent-v2", "run.3", "A1"):
            self.assertEqual(policyviz.validate_preset_name(good), good)


class TestTheTrafficAndFibreFieldsReachTheEnvironment(unittest.TestCase):
    """l0, T0, alpha and odr_losses used to be unreachable from the window:
    l0 and w_window were not on VizConfig at all, and alpha/odr_losses were on
    it but not on the form, so they were pinned to defaults no matter what the
    experiment needed. A field that shows a number the run does not use is
    worse than a missing field."""

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
        self.ui.messagebox = self._real_messagebox
        app = getattr(self, "app", None)
        if app is not None:
            app.destroy()

    def test_every_traffic_and_fibre_field_reaches_hybrid_params(self):
        typed = {"l0": "4096", "t0_deadline": "7", "alpha": "0.25",
                 "odr_losses": "3.5", "w_window": "9", "lam": "1.25"}
        for name, text in typed.items():
            self.app.entries[name].set(text)
        params = self.app._read_config().hybrid_params()
        self.assertEqual(params.l0, 4096)
        self.assertEqual(params.t0_deadline, 7)
        self.assertAlmostEqual(params.alpha, 0.25)
        self.assertAlmostEqual(params.odr_losses, 3.5)
        self.assertEqual(params.w_window, 9)
        self.assertAlmostEqual(params.lam, 1.25)

    def test_the_fibre_fields_are_read_as_floats(self):
        # int(float("0.25")) is 0 -- a lossless fibre, silently.
        self.app.entries["alpha"].set("0.25")
        self.app.entries["odr_losses"].set("13.0")
        params = self.app._read_config().hybrid_params()
        self.assertNotEqual(params.alpha, 0)
        self.assertAlmostEqual(params.alpha, 0.25)
        self.assertAlmostEqual(params.odr_losses, 13.0)

    def test_a_heading_row_is_not_treated_as_a_field(self):
        self.assertTrue(any(n is None for n, _, _ in self.ui.FIELDS))
        for name, _, _ in self.ui.FIELDS:
            if name is not None:
                self.assertIn(name, self.app.entries)
        self.assertNotIn(None, self.app.entries)

    def test_a_bad_traffic_number_names_the_field(self):
        # The message names the field the way the FORM labels it, not the
        # way the dataclass spells it -- the reader is looking at the form.
        self.app.entries["l0"].set("nonsense")
        with self.assertRaises(ValueError) as ctx:
            self.app._read_config()
        self.assertIn("bits per message", str(ctx.exception))
        self.assertIn("nonsense", str(ctx.exception))


class TestCustomIsAReservedPresetName(unittest.TestCase):
    def test_a_preset_may_not_be_called_custom(self):
        # A preset named "custom" could not be told apart from the state that
        # word denotes, so the chooser would show one name for two things.
        with self.assertRaises(ValueError):
            policyviz.validate_preset_name(policyviz.CUSTOM_NAME)

    def test_an_ordinary_name_is_still_accepted(self):
        self.assertEqual(policyviz.validate_preset_name("my_point"),
                         "my_point")


class TestTheOperatingPointLabel(unittest.TestCase):
    """The button, and the figure header, must never hide an edit."""

    def setUp(self):
        self.table = presets.load_ent_presets()

    def test_an_unedited_preset_shows_its_name(self):
        self.assertEqual(
            policyviz.operating_point_label("ent", "ent_advanced", None,
                                            self.table),
            "ENT: ent_advanced")

    def test_an_env_equal_to_the_preset_is_not_marked_custom(self):
        self.assertEqual(
            policyviz.operating_point_label("ent", "ent_advanced", _ent_env(),
                                            self.table),
            "ENT: ent_advanced")

    def test_an_edited_env_is_marked(self):
        # The preset name is REPLACED, not annotated: once the numbers have
        # been edited the preset no longer describes the run, so naming it
        # would invite a reader to look up physics that is not in force.
        label = policyviz.operating_point_label(
            "ent", "ent_advanced", _ent_env(mu=0.015), self.table)
        self.assertEqual(label, "ENT: custom")
        self.assertNotIn("ent_advanced", label)

    def test_the_shared_fibre_alone_is_not_an_edit(self):
        # The dialog shows the MAIN WINDOW's link, which routinely differs
        # from the one the preset file ships (aurea has 13 dB of optics
        # losses; the visualiser's default link has none). Counting that as
        # an edit would mark every preset "custom" on sight.
        label = policyviz.operating_point_label(
            "pm", "aurea", _pm_env(d_km=999.0, odr_losses=1.0),
            presets.load_presets())
        self.assertNotIn("custom", label)

    def test_a_config_reports_both_of_its_operating_points(self):
        pm, ent = small_config(ent_env=_ent_env(mu=0.015)).operating_points()
        self.assertEqual(pm, "PM: aurea")
        self.assertIn("custom", ent)

    def test_the_figure_header_names_both_operating_points(self):
        session = VizSession(small_config(specs=("always-pm",),
                                          ent_env=_ent_env(mu=0.015)))
        fig = Figure(figsize=(9, 8))
        FigureCanvasAgg(fig)
        draw_panels(fig, session)
        head = fig._suptitle.get_text()
        self.assertIn("PM: aurea", head)
        self.assertIn("ENT: custom", head)


class TestTheParameterDialog(unittest.TestCase):
    """The editor itself. Nothing here is opened modally: `show()` is the
    only blocking call and no test calls it."""

    def setUp(self):
        try:
            import policyviz_ui
        except Exception as exc:                  # pragma: no cover
            self.skipTest("tkinter/matplotlib unavailable: %s" % exc)
        self.ui = policyviz_ui
        self.dialogs = []
        self._real_messagebox = policyviz_ui.messagebox
        self._real_simpledialog = policyviz_ui.simpledialog
        policyviz_ui.messagebox = _RecordingMessagebox(self.dialogs)
        policyviz_ui.simpledialog = _RecordingSimpledialog(self.dialogs)
        try:
            self.root = policyviz_ui.tk.Tk()
        except Exception as exc:                  # pragma: no cover
            policyviz_ui.messagebox = self._real_messagebox
            policyviz_ui.simpledialog = self._real_simpledialog
            self.skipTest("cannot create Tk window: %s" % exc)
        self.root.withdraw()
        self.shared = {"d_km": 55.0, "alpha": 0.2, "odr_losses": 0.0}

    def tearDown(self):
        self.ui.messagebox = self._real_messagebox
        self.ui.simpledialog = self._real_simpledialog
        root = getattr(self, "root", None)
        if root is not None:
            root.update_idletasks()
            root.destroy()

    def _dialog(self, protocol="ent", name="ent_advanced", env=None):
        table = policyviz.preset_table(protocol)
        dialog = self.ui.ParamsDialog(self.root, protocol, name,
                                      env if env is not None else table[name],
                                      self.shared)
        dialog.withdraw()
        return dialog

    def test_editing_a_field_flips_the_chooser_to_custom(self):
        # The chooser is the only thing in the dialog that names the physics.
        # Left showing "ent_advanced" over hand-edited numbers it is a lie.
        dialog = self._dialog()
        self.assertEqual(dialog.preset_var.get(), "ent_advanced")
        dialog.entries["mu"].set("0.015")
        self.assertEqual(dialog.preset_var.get(), policyviz.CUSTOM_NAME)

    def test_opening_on_an_already_edited_env_says_custom_at_once(self):
        dialog = self._dialog(env=_ent_env(mu=0.015))
        self.assertEqual(dialog.preset_var.get(), policyviz.CUSTOM_NAME)

    def test_refilling_from_a_preset_does_not_mark_it_custom(self):
        # fill() writes every field, which would trip the edit trace. If it
        # did, choosing a preset would immediately claim to be custom and the
        # chooser could never show a preset name again.
        dialog = self._dialog()
        dialog.preset_var.set("chapman_lowmu")
        dialog._on_preset()
        self.assertEqual(dialog.preset_var.get(), "chapman_lowmu")

    def test_ok_returns_the_base_preset_name_never_custom(self):
        # THE dangerous one. "custom" is a label, not a lookup key: handed
        # back as the preset name it reaches HybridParams.ent_preset and the
        # next presets.load_ent_presets()[name] raises KeyError.
        dialog = self._dialog()
        dialog.entries["mu"].set("0.015")
        self.assertEqual(dialog.preset_var.get(), policyviz.CUSTOM_NAME)
        dialog._ok()
        name, env = dialog.result
        self.assertEqual(name, "ent_advanced")
        self.assertIn(name, policyviz.preset_table("ent"))
        self.assertAlmostEqual(env.mu, 0.015)

    def test_choosing_custom_from_the_chooser_changes_no_field(self):
        # "custom" describes the fields; it cannot supply them.
        dialog = self._dialog()
        before = {k: v.get() for k, v in dialog.entries.items()}
        dialog.preset_var.set(policyviz.CUSTOM_NAME)
        dialog._on_preset()
        self.assertEqual({k: v.get() for k, v in dialog.entries.items()},
                         before)

    def test_the_chooser_offers_custom(self):
        dialog = self._dialog()
        self.assertIn(policyviz.CUSTOM_NAME,
                      list(dialog.preset_box.cget("values")))

    def test_it_shows_the_presets_actual_parameters(self):
        dialog = self._dialog()
        self.assertEqual(float(dialog.entries["mu"].get()),
                         presets.load_ent_presets()["ent_advanced"].mu)

    def test_choosing_a_preset_refills_the_fields(self):
        dialog = self._dialog()
        dialog.preset_var.set("chapman_lowmu")
        dialog._on_preset()
        self.assertEqual(float(dialog.entries["mu"].get()),
                         presets.load_ent_presets()["chapman_lowmu"].mu)

    def test_the_fibre_fields_are_shown_but_not_editable(self):
        dialog = self._dialog()
        self.assertEqual(set(dialog.shared_vars),
                         set(policyviz.SHARED_CHANNEL_FIELDS))
        for name in policyviz.SHARED_CHANNEL_FIELDS:
            self.assertNotIn(name, dialog.entries)
            self.assertEqual(float(dialog.shared_vars[name].get()),
                             self.shared[name])

    def test_the_fibre_shown_is_the_main_windows_not_the_presets(self):
        dialog = self._dialog("pm", "aurea")
        self.assertEqual(presets.load_presets()["aurea"].odr_losses, 13.0)
        self.assertEqual(float(dialog.shared_vars["odr_losses"].get()), 0.0)

    def test_ok_returns_what_the_fields_currently_show(self):
        dialog = self._dialog()
        dialog.entries["mu"].set("0.015")
        dialog._ok()
        name, env = dialog.result
        self.assertEqual(name, "ent_advanced")
        self.assertAlmostEqual(env.mu, 0.015)

    def test_cancel_returns_nothing_at_all(self):
        dialog = self._dialog()
        dialog.entries["mu"].set("0.015")
        dialog._cancel()
        self.assertIsNone(dialog.result)

    def test_the_window_manager_close_button_is_wired(self):
        # Without this the [x] destroys the dialog with `result` never set
        # and no way to tell a close from an OK.
        dialog = self._dialog()
        self.assertTrue(dialog.protocol("WM_DELETE_WINDOW"))

    def test_a_bad_number_reports_and_leaves_the_dialog_open(self):
        dialog = self._dialog()
        dialog.entries["mu"].set("banana")
        dialog._ok()
        self.assertIsNone(dialog.result)
        self.assertTrue(dialog.winfo_exists(),
                        "the dialog closed on a bad number, throwing away "
                        "every other edit in it")
        self.assertTrue([d for d in self.dialogs if d[0] == "error"])

    def test_a_bad_number_does_not_half_apply_the_other_fields(self):
        dialog = self._dialog()
        dialog.entries["eta_a"].set("0.5")
        dialog.entries["mu"].set("banana")
        dialog._ok()
        self.assertIsNone(dialog.result)


class TestSavingAPreset(unittest.TestCase):
    """"Save as preset" must write WHAT THE FIELDS SHOW.

    Writing the preset the dialog was opened on would produce a file that
    looks saved, loads cleanly and contains the wrong numbers -- with nothing
    on screen to say so.

    presets/ is redirected to a COPY for the whole class: a test that wrote
    into the real directory could overwrite an operating point the study
    depends on.
    """

    def setUp(self):
        try:
            import policyviz_ui
        except Exception as exc:                  # pragma: no cover
            self.skipTest("tkinter/matplotlib unavailable: %s" % exc)
        self.ui = policyviz_ui
        self.tmp = tempfile.mkdtemp()
        self.preset_dir = os.path.join(self.tmp, "presets")
        shutil.copytree(presets.PRESET_DIR, self.preset_dir)
        self._real_dir = presets.PRESET_DIR
        presets.PRESET_DIR = self.preset_dir
        self.dialogs = []
        self._real_messagebox = policyviz_ui.messagebox
        self._real_simpledialog = policyviz_ui.simpledialog
        self.messagebox = _RecordingMessagebox(self.dialogs)
        self.simpledialog = _RecordingSimpledialog(self.dialogs, "unit_point")
        policyviz_ui.messagebox = self.messagebox
        policyviz_ui.simpledialog = self.simpledialog
        try:
            self.root = policyviz_ui.tk.Tk()
        except Exception as exc:                  # pragma: no cover
            self._restore()
            self.skipTest("cannot create Tk window: %s" % exc)
        self.root.withdraw()
        self.dialog = policyviz_ui.ParamsDialog(
            self.root, "ent", "ent_advanced",
            presets.load_ent_presets()["ent_advanced"],
            {"d_km": 55.0, "alpha": 0.2, "odr_losses": 0.0})
        self.dialog.withdraw()

    def _restore(self):
        presets.PRESET_DIR = self._real_dir
        self.ui.messagebox = self._real_messagebox
        self.ui.simpledialog = self._real_simpledialog

    def tearDown(self):
        self._restore()
        root = getattr(self, "root", None)
        if root is not None:
            root.update_idletasks()
            root.destroy()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _saved(self, name="unit_point"):
        return presets.load_ent_presets()[name]

    def test_it_saves_the_edited_values_not_the_presets(self):
        original = presets.load_ent_presets()["ent_advanced"].mu
        self.dialog.entries["mu"].set("0.0123")
        self.dialog._save()
        self.assertAlmostEqual(self._saved().mu, 0.0123)
        self.assertNotAlmostEqual(self._saved().mu, original)

    def test_it_saves_every_edited_field(self):
        self.dialog.entries["mu"].set("0.0123")
        self.dialog.entries["eta_a"].set("0.55")
        self.dialog._save()
        saved = self._saved()
        self.assertAlmostEqual(saved.mu, 0.0123)
        self.assertAlmostEqual(saved.eta_a, 0.55)

    def test_the_saved_preset_round_trips_with_its_protocol_tag(self):
        # A missing "protocol": "ent" would file it under the PM presets,
        # where EnvParams has none of these fields and _load raises.
        self.dialog._save()
        self.assertIn("unit_point", presets.load_ent_presets())
        self.assertNotIn("unit_point", presets.load_presets())

    def test_the_saved_preset_still_carries_the_shared_fibre(self):
        # Dropping the read-only fields would leave a file that _load cannot
        # turn back into a complete dataclass.
        self.dialog._save()
        saved = self._saved()
        for name in policyviz.SHARED_CHANNEL_FIELDS:
            self.assertIsNotNone(getattr(saved, name))
        self.assertEqual(saved.d_km, 55.0)

    def test_the_new_preset_is_offered_without_a_restart(self):
        self.dialog._save()
        self.assertIn("unit_point", self.dialog.preset_box.cget("values"))
        self.assertEqual(self.dialog.preset_var.get(), "unit_point")

    def test_the_reserved_name_is_refused_with_a_message(self):
        self.simpledialog.answer = "default"
        self.dialog._save()
        self.assertTrue([d for d in self.dialogs if d[0] == "error"])
        self.assertFalse(os.path.exists(
            os.path.join(self.preset_dir, "default.json")))
        presets.load_ent_presets()          # would raise if the file existed

    def test_an_empty_name_is_refused_with_a_message(self):
        self.simpledialog.answer = "   "
        self.dialog._save()
        self.assertTrue([d for d in self.dialogs if d[0] == "error"])

    def test_cancelling_the_name_prompt_writes_nothing(self):
        self.simpledialog.answer = None
        before = sorted(os.listdir(self.preset_dir))
        self.dialog._save()
        self.assertEqual(sorted(os.listdir(self.preset_dir)), before)

    def test_an_existing_name_asks_before_overwriting(self):
        self.simpledialog.answer = "chapman_lowmu"
        self.messagebox.answer = False
        original = presets.load_ent_presets()["chapman_lowmu"].mu
        self.dialog.entries["mu"].set("0.0999")
        self.dialog._save()
        self.assertTrue([d for d in self.dialogs if d[0] == "askyesno"])
        self.assertAlmostEqual(
            presets.load_ent_presets()["chapman_lowmu"].mu, original)

    def test_a_confirmed_overwrite_goes_through(self):
        self.simpledialog.answer = "chapman_lowmu"
        self.messagebox.answer = True
        self.dialog.entries["mu"].set("0.0999")
        self.dialog._save()
        self.assertAlmostEqual(
            presets.load_ent_presets()["chapman_lowmu"].mu, 0.0999)

    def test_a_bad_number_is_never_written_to_disk(self):
        self.dialog.entries["mu"].set("banana")
        before = sorted(os.listdir(self.preset_dir))
        self.dialog._save()
        self.assertEqual(sorted(os.listdir(self.preset_dir)), before)
        self.assertTrue([d for d in self.dialogs if d[0] == "error"])
        self.assertFalse([d for d in self.dialogs if d[0] == "askstring"],
                         "asked for a name before checking the numbers")

    def test_saving_does_not_apply_anything(self):
        # Save writes a file; OK is what adopts the values. A save that also
        # closed the dialog would make "save then keep editing" impossible.
        self.dialog._save()
        self.assertIsNone(self.dialog.result)
        self.assertTrue(self.dialog.winfo_exists())


class TestTheOperatingPointButtons(unittest.TestCase):
    """The main window's half: two buttons that say what is in force."""

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
        self.ui.messagebox = self._real_messagebox
        app = getattr(self, "app", None)
        if app is not None:
            app._running = False
            app.update_idletasks()
            app.destroy()

    def _answer(self, protocol, result):
        """Make the next _edit_params(protocol) get `result`, no window."""
        self.app._params_dialog_result = (
            lambda p, _r=result: _r if p == protocol else None)

    def test_the_buttons_name_the_operating_points_in_force(self):
        self.assertEqual(self.app.mode_buttons["pm"].cget("text"),
                         "PM: aurea")
        self.assertEqual(self.app.mode_buttons["ent"].cget("text"),
                         "ENT: ent_advanced")

    def test_the_button_says_custom_after_an_edit(self):
        self._answer("ent", ("ent_advanced", _ent_env(mu=0.015)))
        self.app._edit_params("ent")
        self.assertIn("custom", self.app.mode_buttons["ent"].cget("text"))

    def test_the_edit_reaches_the_configuration_the_form_builds(self):
        self._answer("ent", ("ent_advanced", _ent_env(mu=0.015)))
        self.app._edit_params("ent")
        config = self.app._read_config()
        self.assertIsNotNone(config.ent_env)
        self.assertAlmostEqual(config.ent_env.mu, 0.015)

    def test_the_edit_reaches_the_session_on_apply(self):
        self._answer("ent", ("ent_advanced", _ent_env(mu=0.015)))
        self.app._edit_params("ent")
        self.app._apply()
        self.assertAlmostEqual(
            self.app.session.runs[0].env.params.ent_env.mu, 0.015)
        self.assertIn("custom", self.app.session.ent_label)

    def test_cancel_changes_nothing(self):
        self._answer("ent", ("ent_advanced", _ent_env(mu=0.015)))
        self.app._edit_params("ent")
        before = self.app.mode_buttons["ent"].cget("text")
        self.app._params_dialog_result = lambda p: None
        self.app._edit_params("ent")
        self.assertEqual(self.app.mode_buttons["ent"].cget("text"), before)
        self.assertAlmostEqual(self.app.mode_env["ent"].mu, 0.015)

    def test_cancel_on_an_untouched_mode_leaves_it_on_the_preset(self):
        self.app._params_dialog_result = lambda p: None
        self.app._edit_params("pm")
        self.assertIsNone(self.app.mode_env["pm"])
        self.assertEqual(self.app._read_config().pm_env, None)

    def test_choosing_another_preset_switches_without_marking_custom(self):
        self._answer("ent", ("chapman_lowmu",
                             presets.load_ent_presets()["chapman_lowmu"]))
        self.app._edit_params("ent")
        self.assertEqual(self.app.mode_buttons["ent"].cget("text"),
                         "ENT: chapman_lowmu")
        self.assertIsNone(self.app.mode_env["ent"])
        self.assertEqual(self.app._read_config().ent_preset, "chapman_lowmu")

    def test_ok_on_an_untouched_dialog_is_not_an_edit(self):
        table = presets.load_ent_presets()
        self._answer("ent", ("ent_advanced",
                             dataclasses.replace(table["ent_advanced"],
                                                 d_km=55.0, alpha=0.2,
                                                 odr_losses=0.0)))
        self.app._edit_params("ent")
        self.assertIsNone(self.app.mode_env["ent"])
        self.assertNotIn("custom",
                         self.app.mode_buttons["ent"].cget("text"))

    def test_the_dialog_is_opened_on_the_main_windows_fibre(self):
        self.app.entries["d_km"].set("12")
        self.assertEqual(self.app._shared_channel()["d_km"], 12.0)

    def test_a_half_typed_link_still_opens_the_dialog(self):
        self.app.entries["d_km"].set("")
        self.assertEqual(self.app._shared_channel()["d_km"],
                         float(self.app.config_obj.d_km))

    def test_the_status_line_says_an_apply_is_needed(self):
        self._answer("ent", ("ent_advanced", _ent_env(mu=0.015)))
        self.app._edit_params("ent")
        self.assertIn("Apply", self.app.status.cget("text"))


class TestTheSavedConfiguration(unittest.TestCase):
    """The Save button, and coming back up in exactly that state.

    The complaint this answers: load a preset, change one number, and the only
    way to keep it was to invent a preset name for it.

    `vizstate.DEFAULT_PATH` is redirected into a temp directory for the whole
    class. A test that wrote the real file would replace the developer's own
    startup configuration -- panelstate's tests take the same precaution, and
    its docstring records the trap being hit for real.
    """

    def setUp(self):
        try:
            import policyviz_ui
        except Exception as exc:                  # pragma: no cover
            self.skipTest("tkinter/matplotlib unavailable: %s" % exc)
        self.ui = policyviz_ui
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "policyviz_default.json")
        self._real_path = vizstate.DEFAULT_PATH
        vizstate.DEFAULT_PATH = self.path
        # Save reports a bad box with messagebox.showerror, which BLOCKS the
        # mainloop until someone clicks it; WSLg provides a display here, so
        # an unstubbed one would hang the run rather than fail it.
        self.dialogs = []
        self._real_messagebox = policyviz_ui.messagebox
        policyviz_ui.messagebox = _RecordingMessagebox(self.dialogs)
        self.apps = []
        self.app = self._open()

    def _open(self, saved=None, config=None):
        try:
            app = self.ui.PolicyVizApp(config or small_config(), saved=saved)
        except Exception as exc:                  # pragma: no cover
            self._restore()
            self.skipTest("cannot create Tk window: %s" % exc)
        app.withdraw()
        self.apps.append(app)
        return app

    def _reopen(self, saved=None):
        """Close the window and open a fresh one -- a restart, in one process.

        The old one is destroyed FIRST: two live Tk roots in one process is
        its own source of trouble, and this suite already avoids it elsewhere.
        """
        self._close(self.app)
        self.apps.remove(self.app)
        self.app = self._open(saved=saved)
        return self.app

    @staticmethod
    def _close(app):
        app._running = False
        app.update_idletasks()
        app.destroy()

    def _restore(self):
        vizstate.DEFAULT_PATH = self._real_path
        if getattr(self, "_real_messagebox", None) is not None:
            self.ui.messagebox = self._real_messagebox

    def tearDown(self):
        self._restore()
        for app in getattr(self, "apps", []):
            self._close(app)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --- what gets written -------------------------------------------------

    def test_it_saves_what_the_form_shows_not_what_was_applied(self):
        """THE point of the button.

        Saving from `config_obj` -- the configuration Apply last built --
        would write the previously applied numbers and silently discard the
        edit the user pressed Save to keep.
        """
        self.app.entries["lam"].set("3.75")
        self.app._save_state()
        self.assertAlmostEqual(vizstate.load_state()["fields"]["lam"], 3.75)
        # ... and the applied session is untouched by the save.
        self.assertAlmostEqual(self.app.session.params.lam, 2.5)

    def test_it_saves_every_field_on_the_form(self):
        state = self.app._collect_state()
        for name, _label, _width in self.ui.FIELDS:
            if name is None:                      # a heading row
                continue
            self.assertIn(name, state["fields"])

    def test_an_edited_operating_point_is_saved_as_its_numbers(self):
        """A preset that has been edited is "custom": the base preset name
        alone cannot describe it, so the dataclass itself is written."""
        self.app._set_operating_point("ent", "ent_advanced",
                                      _ent_env(mu=0.031))
        self.app._save_state()
        loaded = vizstate.load_state()
        self.assertEqual(loaded["ent_preset"], "ent_advanced")
        self.assertAlmostEqual(loaded["ent_env"].mu, 0.031)

    def test_an_unedited_operating_point_is_saved_as_none(self):
        """None means "run presets/aurea.json as it ships".

        Writing a frozen copy of the preset instead would load cleanly and
        quietly stop tracking the preset file: a later edit to
        presets/aurea.json would never be picked up again.
        """
        self.assertIsNone(self.app._collect_state()["pm_env"])
        self.app._save_state()
        loaded = vizstate.load_state()
        self.assertIn("pm_env", loaded)
        self.assertIsNone(loaded["pm_env"])

    # --- coming back up ----------------------------------------------------

    def test_the_edit_and_the_form_survive_a_restart(self):
        self.app._set_operating_point("ent", "ent_advanced",
                                      _ent_env(mu=0.031))
        self.app.entries["lam"].set("3.75")
        self.app._save_state()
        app = self._reopen(saved=vizstate.load_state())
        self.assertAlmostEqual(app.mode_env["ent"].mu, 0.031)
        self.assertEqual(app.entries["lam"].get(), "3.75")
        # Restored AND applied: the reopened window is running the physics it
        # is displaying, not merely showing the numbers on the form.
        self.assertAlmostEqual(app.session.params.ent_env.mu, 0.031)
        self.assertAlmostEqual(app.session.params.lam, 3.75)
        self.assertEqual(app.mode_buttons["ent"].cget("text"), "ENT: custom")

    def test_an_unedited_point_still_tracks_its_preset_after_a_restart(self):
        self.app._save_state()
        app = self._reopen(saved=vizstate.load_state())
        self.assertIsNone(app.mode_env["pm"])
        self.assertEqual(app.preset_var["pm"].get(), "aurea")
        self.assertEqual(app.mode_buttons["pm"].cget("text"), "PM: aurea")

    def test_the_policy_selection_comes_back(self):
        chosen = {"always-pm", "buffer-aware"}
        for stem, var in self.app.checks.items():
            var.set(stem in chosen)
        self.app._save_state()
        app = self._reopen(saved=vizstate.load_state())
        self.assertEqual({s for s, v in app.checks.items() if v.get()}, chosen)
        self.assertIn("buffer-aware-0.045-0.25", app.config_obj.specs)

    def test_a_threshold_comes_back_even_with_its_policy_unticked(self):
        # A number that was typed and then set aside is still a thing the
        # user chose; losing it means retyping it next session.
        self.app.checks["greedy"].set(False)
        self.app.theta.set("0.02")
        self.app._save_state()
        app = self._reopen(saved=vizstate.load_state())
        self.assertEqual(app.theta.get(), "0.02")
        self.assertFalse(app.checks["greedy"].get())

    def test_the_hidden_traces_come_back(self):
        name = self.app.session.runs[1].display_name
        self.app.session.toggle_trace(name)
        self.app._save_state()
        app = self._reopen(saved=vizstate.load_state())
        self.assertIn(name, app.session.hidden_traces)
        self.assertFalse(app.trace_checks[name].get())

    def test_the_speed_comes_back(self):
        self.app.speed.set(137)
        self.app._save_state()
        app = self._reopen(saved=vizstate.load_state())
        self.assertEqual(app.speed.get(), 137)

    def test_a_speed_off_the_scale_is_clamped_to_the_slider(self):
        self.app._apply_state({"fields": {}, "speed": 10000})
        self.assertEqual(self.app.speed.get(), self.ui.SPEED_MAX)

    # --- the startup order -------------------------------------------------

    def test_a_field_absent_from_an_old_saved_file_keeps_its_default(self):
        """THE ORDER: seed from the configuration first, overlay the saved
        file second.

        Overlaying first (or letting the saved file be the only thing that
        fills the form) leaves every control the file predates BLANK, so a
        default written before a field existed would break the form.
        """
        app = self._reopen(saved={"fields": {"lam": 9.5}})
        self.assertEqual(app.entries["lam"].get(), "9.5")
        self.assertEqual(app.entries["horizon"].get(), str(SLOTS))
        self.assertEqual(app.entries["n_x"].get(), str(VizConfig.n_x))

    def test_a_control_the_saved_file_omits_is_left_alone(self):
        app = self._reopen(saved={"fields": {}})
        self.assertEqual({s for s, v in app.checks.items() if v.get()},
                         set(self.ui.PolicyVizApp._stems(small_config().specs)))
        self.assertEqual(app.preset_var["pm"].get(), "aurea")
        self.assertIsNone(app.mode_env["ent"])
        self.assertEqual(app.theta.get(), "0.045")

    def test_the_window_does_not_read_the_file_on_its_own(self):
        # The caller reads it and hands it in. A constructor that read the
        # disk itself would make every OTHER test in this suite depend on
        # whether the developer happened to have saved a configuration.
        self.app.entries["lam"].set("9.9")
        self.app._save_state()
        app = self._reopen()                      # saved=None
        self.assertEqual(app.entries["lam"].get(), "2.5")

    def test_the_module_entry_point_reads_the_file(self):
        """`python3 policyviz_ui.py` is a second way in, and it must reach the
        saved configuration too -- the CALLER reads the file, so every caller
        has to. (`python3 policyviz.py` goes through policyviz.main, which the
        next class covers.)"""
        self.app._save_state()
        seen = {}

        class _Recorder:
            def __init__(self, config=None, saved=None):
                seen["config"], seen["saved"] = config, saved

            def mainloop(self):
                pass

        real = self.ui.PolicyVizApp
        self.ui.PolicyVizApp = _Recorder
        try:
            self.ui.main()
        finally:
            self.ui.PolicyVizApp = real
        self.assertIsNotNone(seen["saved"])
        self.assertAlmostEqual(seen["saved"]["fields"]["lam"], 2.5)

    # --- Save and Apply are separate ---------------------------------------

    def test_save_does_not_apply(self):
        """Apply rebuilds the session, which throws away the episode on
        screen and, with "dqn" ticked, starts a training thread. A Save that
        did that as a side effect would cost the user the run they were
        watching."""
        self.app.session.step(5)
        self.app.entries["lam"].set("3.75")
        self.app._save_state()
        self.assertAlmostEqual(self.app.config_obj.lam, 2.5)
        self.assertEqual(self.app.session.slot, 5)
        # And it says so, rather than leaving the user to guess.
        self.assertIn("Apply", self.app.status.cget("text"))

    def test_apply_does_not_save(self):
        # The reverse coupling is just as bad: it would overwrite the saved
        # configuration every time someone nudged a parameter to try it.
        self.app.entries["lam"].set("3.75")
        self.app._apply()
        self.assertFalse(os.path.exists(self.path))
        self.assertIsNone(vizstate.load_state())

    def test_save_refuses_a_form_it_could_not_apply(self):
        self.app.entries["lam"].set("")
        self.app._save_state()
        self.assertFalse(os.path.exists(self.path))
        self.assertTrue(self.app.status.cget("text").startswith("save failed"))
        self.assertTrue([d for d in self.dialogs if d[0] == "error"])

    def test_save_refuses_an_empty_policy_selection(self):
        for var in self.app.checks.values():
            var.set(False)
        self.app._save_state()
        self.assertFalse(os.path.exists(self.path))

    # --- Restore -----------------------------------------------------------

    def test_restore_puts_the_saved_form_back_mid_session(self):
        self.app._save_state()
        self.app.entries["lam"].set("9.9")
        self.app._restore_state()
        self.assertEqual(self.app.entries["lam"].get(), "2.5")

    def test_restore_does_not_apply_either(self):
        self.app._save_state()
        self.app.entries["lam"].set("9.9")
        self.app._apply()
        self.app._restore_state()
        self.assertEqual(self.app.entries["lam"].get(), "2.5")
        self.assertAlmostEqual(self.app.session.params.lam, 9.9)
        self.assertIn("Apply", self.app.status.cget("text"))

    def test_restore_puts_the_hidden_traces_back_on_the_live_session(self):
        # hidden_traces is the one restored thing Apply has nothing to do
        # with -- it is a view choice, not part of the configuration -- so it
        # is imposed on the running session straight away.
        name = self.app.session.runs[1].display_name
        self.app.session.toggle_trace(name)
        self.app._save_state()
        self.app.session.toggle_trace(name)
        self.assertNotIn(name, self.app.session.hidden_traces)
        self.app._restore_state()
        self.assertIn(name, self.app.session.hidden_traces)
        self.assertFalse(self.app.trace_checks[name].get())

    def test_restore_without_a_saved_file_says_so(self):
        self.app.entries["lam"].set("9.9")
        self.app._restore_state()
        self.assertIn("nothing saved", self.app.status.cget("text"))
        self.assertEqual(self.app.entries["lam"].get(), "9.9")

    def test_a_corrupt_saved_file_is_treated_as_nothing_saved(self):
        # It must not raise: the same read runs during startup, where an
        # exception would stop the window opening at all.
        with open(self.path, "w") as f:
            f.write("{not json at all")
        self.app.entries["lam"].set("9.9")
        self.app._restore_state()
        self.assertIn("nothing saved", self.app.status.cget("text"))
        self.assertEqual(self.app.entries["lam"].get(), "9.9")


class TestTheCommandLineBeatsTheSavedConfiguration(unittest.TestCase):
    """`--lam 3.0` must not be silently overlaid away by the saved file.

    The saved configuration is the startup state, so it has to be skipped
    whenever the user actually typed an option -- otherwise the option looks
    ignored, with nothing on screen to explain it.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._real_path = vizstate.DEFAULT_PATH
        vizstate.DEFAULT_PATH = os.path.join(self.tmp, "policyviz_default.json")
        vizstate.save_state({"fields": {"lam": 9.5}, "policies": ["always-pm"]})
        self.seen = {}
        self._real_build = policyviz._build_app

        class _Recorder:
            def __init__(_self, config, saved=None):
                self.seen["config"], self.seen["saved"] = config, saved

            def mainloop(_self):
                pass

        policyviz._build_app = lambda config, saved=None: _Recorder(config,
                                                                    saved)

    def tearDown(self):
        policyviz._build_app = self._real_build
        vizstate.DEFAULT_PATH = self._real_path
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_options_starts_on_the_saved_configuration(self):
        policyviz.main([])
        self.assertIsNotNone(self.seen["saved"])
        self.assertAlmostEqual(self.seen["saved"]["fields"]["lam"], 9.5)

    def test_a_typed_option_wins_and_the_saved_file_is_skipped(self):
        policyviz.main(["--lam", "3.0"])
        self.assertIsNone(self.seen["saved"])
        self.assertAlmostEqual(self.seen["config"].lam, 3.0)

    def test_an_untyped_option_still_gets_its_dataclass_default(self):
        policyviz.main(["--lam", "3.0"])
        self.assertEqual(self.seen["config"].horizon, VizConfig.horizon)
        self.assertEqual(self.seen["config"].specs, VizConfig.specs)

    def test_headless_never_reads_the_saved_configuration(self):
        # --no-gui is the reproducible path: a scoreboard printed for the
        # study must depend on the arguments and the presets, never on what
        # happens to be in one machine's policyviz_default.json.
        calls = []
        real = vizstate.load_state

        def _counting(*args, **kw):
            calls.append(1)
            return real(*args, **kw)

        vizstate.load_state = _counting
        try:
            policyviz.main(["--no-gui", "--policies", "always-pm",
                            "--horizon", "5"])
        finally:
            vizstate.load_state = real
        self.assertEqual(calls, [])
