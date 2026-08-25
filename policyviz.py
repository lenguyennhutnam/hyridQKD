#!/usr/bin/env python3
"""Watch the hybrid PM/ENT mode-selection policies run, side by side.

    python3 policyviz.py                      # the window
    python3 policyviz.py --no-gui             # the scoreboard, as text
    python3 policyviz.py --ent-preset chapman_lowmu --lam 2.5 --b-max 1e6

WHAT THIS DRIVES -- and what it does NOT
----------------------------------------
`gui.py` drives the REAL qline stack: an Apply there restarts hw_sim, gc and
qber, and a mode change costs the ~2.28 s those processes take to come back.
This window drives `HybridEnv` instead -- the numeric model of eq (1)-(24). No
process is started, no config file is written, no photon is simulated at the
event level. The two are different worlds and a figure that blurs them would
misrepresent the study, so the banner says so on screen and this module never
imports backend/stackctl.

WHY EVERY POLICY GETS ITS OWN ENVIRONMENT
-----------------------------------------
A policy changes the mode, the mode changes the error draw, and the error draw
consumes a different number of uniforms -- so two policies cannot share one
environment object and stay in step. Each `PolicyRun` therefore owns a
`HybridEnv`, and every one of them is constructed from the SAME `HybridParams`
seed and the SAME `EveParams` seed and reset with the same explicit seed.
`HybridEnv.reset(seed)` draws three independent streams (errors, arrivals,
jamming) from one SeedSequence, and jamming and arrivals advance exactly once
per slot whatever the mode is, so all runs face a bit-identical jamming
trajectory and a bit-identical message stream. That invariant is the only
reason comparing their curves means anything; `tests/test_policyviz.py` pins
it directly rather than trusting this paragraph.

TWO THINGS THE PICTURE MUST NOT LIE ABOUT
-----------------------------------------
1. `Oracle` is an UPPER BOUND, not a policy. It sees the true jamming
   fraction, switches for free and is paid continuously (`step_relaxed`), so a
   viewer who reads its curve as achievable throughput reads the study
   backwards. It is drawn dashed, labelled "oracle (upper bound)", and marked
   in the scoreboard. `policies.RelaxationBound` deliberately has no `act()`
   so that `run_episode` fails on it; `PolicyRun` dispatches on the type
   instead of working around that.
2. A DQN has to be TRAINED before it can be shown, and an agent trained in the
   seconds a UI can wait is not the agent in the study (200 episodes). Its
   label always carries the episode count it actually trained with -- "dqn
   (20 ep)" -- so no screenshot can pass a 20-episode agent off as the
   studied one.

LAYOUT
------
This module is the headless half: configuration, the runs, the scoreboard and
the drawing, none of which import Tk. `policyviz_ui.py` is the window, and it
is thin -- it reads the form, rebuilds a `VizSession`, and calls
`draw_panels`. Keeping the split at a file boundary is what lets the tests
render the real figure through the Agg backend on a machine with no display.

Spec of the underlying study: docs/superpowers/specs/2026-08-17-hybrid-policy-design.md
"""

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass

import channel
import dqn as dqn_module
import plots
import presets
from dqn import DQNConfig, training_seed_range
from eval_hybrid import evaluation_seeds
from evemarkov import EveParams
from hybridenv import ENT, PM, HybridEnv, HybridParams
from policies import (AlwaysENT, AlwaysPM, BufferAwareThreshold,
                      GreedyThreshold, Oracle, Policy, RelaxationBound,
                      RelaxedAlwaysPM)

__all__ = ["BANNER", "SPEC_CHOICES", "TRACK_COLORS",
           "REFERENCE_LINE_SHOW_FRAC", "VizConfig", "PolicyRun",
           "VizSession", "parse_spec", "death_thresholds", "train_dqn",
           "dqn_label", "draw_panels", "format_scoreboard", "main"]

HERE = os.path.dirname(os.path.abspath(__file__))
DQN_CACHE_DIR = os.path.join(HERE, "logs", "dqn_cache")

BANNER = ("NUMERIC MODEL (HybridEnv, eq 1-24) — this window does NOT drive "
          "the qline hardware. No hw_sim process is started here; use "
          "gui.py for the real stack.")

# What the policy chooser offers. `greedy` and `buffer-aware` are shown with
# their parameters filled in from the spinboxes, so the strings here are only
# the stems.
SPEC_CHOICES = ("always-pm", "always-ent", "greedy", "buffer-aware", "dqn",
                "oracle", "always-pm-relaxed")

# One colour per track, from the DecoyState palette plots.py already owns, so
# a screenshot of this window sits next to the analytic figures without a
# visual seam. Mode TINTS (panel 1) come from plots.MODE_BAND for the same
# reason.
TRACK_COLORS = (plots.NAVY, plots.GREEN, plots.AMBER, plots.TEAL, plots.RED,
                plots.BLUE, plots.GREY)

# Draw a reference line (B_max) only once the data climbs to this fraction of
# it. Borrowed from gui.DEATH_LINE_SHOW_FRAC, which solves the same problem
# for the death-QBER line: a reference far above the trace forces the axis and
# flattens the trace to a straight line at zero.
REFERENCE_LINE_SHOW_FRAC = 0.25


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class VizConfig:
    """Everything the window can change without restarting the process.

    `seed_index` is an EPISODE INDEX, not a raw seed: it is mapped through
    `eval_hybrid.evaluation_seeds`, which parks the trajectory in a seed block
    no DQN training run reaches. Typing a small number here therefore cannot
    accidentally score the agent on an episode it trained on -- the failure
    the study's whole seeding scheme exists to prevent.
    """

    specs: tuple = ("always-pm", "always-ent", "greedy-0.045", "oracle")
    pm_preset: str = "aurea"
    ent_preset: str = "ent_advanced"
    lam: float = 2.5
    b_max: int = 1_000_000
    horizon: int = 3600
    n_x: int = 100_000
    d_km: float = 55.0
    alpha: float = 0.2
    odr_losses: float = 0.0
    t0_deadline: int = 20
    seed_index: int = 0
    dqn_episodes: int = 20

    def dqn_config(self):
        """The DQNConfig a "dqn" track would train with, or None if unused."""
        if "dqn" not in self.specs:
            return None
        return DQNConfig(episodes=int(self.dqn_episodes))

    def env_seed(self):
        """The trajectory seed, guaranteed clear of the DQN training block."""
        idx = max(0, int(self.seed_index))
        return evaluation_seeds(idx + 1, self.dqn_config())[idx]

    def hybrid_params(self, seed=None):
        """`HybridParams` for this configuration at `seed`."""
        seed = self.env_seed() if seed is None else int(seed)
        return HybridParams(
            n_x=int(self.n_x), lam=float(self.lam), b_max=int(self.b_max),
            t0_deadline=int(self.t0_deadline), horizon=int(self.horizon),
            d_km=float(self.d_km), alpha=float(self.alpha),
            odr_losses=float(self.odr_losses), seed=seed,
            pm_preset=self.pm_preset, ent_preset=self.ent_preset)


def death_thresholds(config):
    """(PM, ENT) death QBER for `config`, in the units panel 2 plots.

    Both modes ride the same fibre, so the presets' own d_km/alpha/odr_losses
    are overridden exactly as `HybridEnv._setup_modes` overrides them --
    reading a preset's channel here instead would draw a threshold line for a
    link the environment is not simulating. (CLAUDE.md records that trap
    having been published once and retracted.)

    Costs ~25 finite-key evaluations per mode; call it once per session, not
    once per frame.
    """
    pm_env = dataclasses.replace(presets.load_presets()[config.pm_preset],
                                 d_km=config.d_km, alpha=config.alpha,
                                 odr_losses=config.odr_losses)
    ent_env = dataclasses.replace(
        presets.load_ent_presets()[config.ent_preset], d_km=config.d_km,
        alpha=config.alpha, odr_losses=config.odr_losses)
    return (channel.death_qber(pm_env, int(config.n_x)),
            channel.death_qber_ent(ent_env, int(config.n_x)))


# --------------------------------------------------------------------------
# building agents
# --------------------------------------------------------------------------

_GREEDY_RE = re.compile(r"^greedy-([0-9.]+)$")
_BUFFER_RE = re.compile(r"^buffer-aware-([0-9.]+)-([0-9.]+)$")


def parse_spec(spec, dqn_policy=None):
    """Turn a policy spec string into an agent.

    Accepts exactly the labels the study's own tables use, so a row copied out
    of `eval_hybrid.format_rows` can be pasted straight in:
    `always-pm`, `always-ent`, `greedy-<theta>`,
    `buffer-aware-<theta>-<reserve>`, `dqn`, `oracle`, `always-pm-relaxed`.

    `dqn` needs a trained agent handed in -- there is nothing sensible to
    return for it otherwise, and returning an untrained network would put a
    random-looking curve on the figure with no sign that it was never trained.
    """
    spec = str(spec).strip()
    if spec == "always-pm":
        return AlwaysPM()
    if spec == "always-ent":
        return AlwaysENT()
    if spec == "oracle":
        return Oracle()
    if spec == "always-pm-relaxed":
        return RelaxedAlwaysPM()
    if spec == "dqn":
        if dqn_policy is None:
            raise ValueError("spec 'dqn' needs a trained policy; call "
                             "train_dqn() first")
        return dqn_policy
    m = _GREEDY_RE.match(spec)
    if m:
        return GreedyThreshold(float(m.group(1)))
    m = _BUFFER_RE.match(spec)
    if m:
        return BufferAwareThreshold(float(m.group(1)), float(m.group(2)))
    raise ValueError("unknown policy spec %r" % (spec,))


def dqn_label(episodes):
    """The label a DQN track carries. The episode count is NOT decoration.

    The study trained 200 episodes; a UI that trains 20 while the user waits
    produces a different agent, and a screenshot with a bare "dqn" row invites
    the reader to compare it against the published number.
    """
    return "dqn (%d ep)" % int(episodes)


# --------------------------------------------------------------------------
# the headless core
# --------------------------------------------------------------------------

class PolicyRun:
    """One agent stepping its OWN environment, one slot at a time.

    No Tk anywhere in this path: the window polls this object and draws
    whatever it finds, so everything worth testing is testable without a
    display.

    Dispatch is on the TYPE of the agent, not on a flag the caller passes:
    `RelaxationBound` runs through `HybridEnv.step_relaxed`, `Policy` through
    `HybridEnv.step`. That is what keeps the Oracle out of the unrelaxed
    dynamics without anyone having to remember which it was.
    """

    def __init__(self, agent, env, seed, label=None, color=plots.NAVY):
        if isinstance(agent, RelaxationBound):
            self.is_bound = True
        elif isinstance(agent, Policy):
            self.is_bound = False
        else:
            raise TypeError("agent must be a policies.Policy or a "
                            "policies.RelaxationBound, got %r"
                            % (type(agent).__name__,))
        self.agent = agent
        self.env = env
        self.seed = int(seed)
        self.color = color
        self._label = label or agent.name
        self.reset()

    @property
    def display_name(self):
        """What the legend, the y-tick and the scoreboard all say.

        A bound is named as one. `Oracle.name` is the bare "oracle", which
        reads as just another policy in a legend; the suffix is added here
        rather than in policies.py because that name is a key in the study's
        result tables and renaming it there would break them.
        """
        if self.is_bound:
            return "%s (upper bound)" % self._label
        return self._label

    @property
    def linestyle(self):
        """Dashed for a bound, solid for a realisable policy."""
        return "--" if self.is_bound else "-"

    def reset(self):
        """Rewind to slot 0 on this run's own seed and clear every trace."""
        self.agent.reset()
        self._obs = self.env.reset(self.seed)
        self.t = []
        self.mode = []
        self.qber = []
        self.buffer = []
        self.eta_trace = []
        self.f = []
        self.slot = 0
        self.switches = 0
        self.ent_slots = 0
        self.expired = 0
        self.served = 0
        self.deposited = 0.0
        self.done = False
        self._last_mode = None

    def step(self):
        """Advance one slot; return the env's info dict (None when finished)."""
        if self.done:
            return None
        if self.is_bound:
            obs, _, done, info = self.env.step_relaxed(self.agent.choose)
        else:
            obs, _, done, info = self.env.step(self.agent.act(self._obs))
        self._obs = obs
        mode = info["mode"]
        if self._last_mode is not None and mode != self._last_mode:
            self.switches += 1
        self._last_mode = mode
        self.slot += 1
        self.t.append(self.slot)
        self.mode.append(mode)
        self.qber.append(info["qber"])
        self.buffer.append(info["buffer"])
        self.eta_trace.append(self.env.eta)
        self.f.append(info["f"])
        self.ent_slots += 1 if mode == ENT else 0
        self.expired += info["expired"]
        self.served += info["served"]
        self.deposited += info["deposited"]
        self.done = done
        return info

    def score(self):
        """The scoreboard row.

        `eta` is read from the ENVIRONMENT (eq 24: served / arrived), never
        recomputed here from a local counter. served/slot, served/horizon and
        served/arrived all coincide at lam = 1.0 and diverge everywhere else,
        so a locally recomputed figure would agree with the study at exactly
        the parameter setting the study calls degenerate and disagree at the
        one it reports.

        `ent_fraction` divides by the slots run SO FAR, not by the horizon:
        mid-episode the horizon denominator would read as "this policy is on
        PM" for every policy simply because the episode is young.
        """
        slots = max(self.slot, 1)
        return {"policy": self.display_name, "eta": self.env.eta,
                "switches": self.switches, "expired": self.expired,
                "ent_fraction": self.ent_slots / slots,
                "secret_bits": self.deposited, "slots": self.slot,
                "is_bound": self.is_bound}


class VizSession:
    """Every selected policy, stepping the same weather in lockstep.

    Construction is the expensive part (one `channel.derive`/`derive_ent` pair
    per run, plus the two death-QBER bisections), so the window rebuilds a
    session only when a parameter actually changes.
    """

    def __init__(self, config=None, dqn_policy=None, dqn_episodes=None):
        self.config = config or VizConfig()
        self.seed = self.config.env_seed()
        self.params = self.config.hybrid_params(self.seed)
        # ONE EveParams seed for every run. Each HybridEnv builds its own
        # EveChain from it -- sharing a chain object would have them take
        # turns drawing from it and see different jamming, which is the exact
        # failure this class exists to prevent.
        self.eve_params = EveParams(seed=self.seed)
        self.runs = []
        for i, spec in enumerate(self.config.specs):
            agent = parse_spec(spec, dqn_policy)
            label = (dqn_label(dqn_episodes if dqn_episodes is not None
                               else self.config.dqn_episodes)
                     if spec == "dqn" else agent.name)
            self.runs.append(PolicyRun(
                agent, HybridEnv(self.params, self.eve_params), self.seed,
                label=label, color=TRACK_COLORS[i % len(TRACK_COLORS)]))
        self._assert_shared_weather()
        self.death_pm, self.death_ent = death_thresholds(self.config)

    def _assert_shared_weather(self):
        """Refuse a session whose runs would face different jamming.

        This is checked rather than assumed because the mechanism is subtle
        enough to be broken by a plausible edit. The trajectory is pinned by
        the seed each run RESETS on: `HybridEnv.reset(seed)` derives Eve's
        seed from that SeedSequence and overrides whatever `EveParams.seed`
        said, so giving each run its own `EveParams.seed` is inert TODAY --
        a test written against that alone would pass on an implementation
        that shares nothing. One bare `reset()` anywhere and it stops being
        inert.

        So both conditions are enforced: one reset seed, and one Eve
        configuration (levels, transition matrix and seed alike). A
        comparison silently run on different weather is the one failure that
        would invalidate every conclusion drawn from this window, and it has
        no visible symptom.
        """
        seeds = {run.seed for run in self.runs}
        if len(seeds) > 1:
            raise ValueError(
                "runs reset on different seeds %s: they would face different "
                "jamming and their curves could not be compared" % (sorted(seeds),))
        first = self.runs[0].env.eve.params if self.runs else None
        for run in self.runs[1:]:
            if run.env.eve.params != first:
                raise ValueError(
                    "%s has a different Eve configuration (%r) from %s (%r)"
                    % (run.display_name, run.env.eve.params,
                       self.runs[0].display_name, first))

    @property
    def done(self):
        return bool(self.runs) and all(r.done for r in self.runs)

    @property
    def slot(self):
        return max((r.slot for r in self.runs), default=0)

    def reset(self):
        for run in self.runs:
            run.reset()

    def step(self, n=1):
        """Advance every run by `n` slots. Returns slots actually advanced."""
        advanced = 0
        for _ in range(int(n)):
            if self.done:
                break
            for run in self.runs:
                run.step()
            advanced += 1
        return advanced

    def scores(self):
        return [r.score() for r in self.runs]


# --------------------------------------------------------------------------
# the DQN, and being honest about it
# --------------------------------------------------------------------------

def _cache_key(config, cfg):
    """Identity of a trained agent: the training config AND the environment.

    An agent trained at `chapman_lowmu` is a different agent from one trained
    at `ent_advanced` -- those two presets give the study opposite answers --
    so the preset has to be in the key or flipping the dropdown would silently
    reuse the wrong weights.
    """
    payload = {
        "episodes": int(cfg.episodes), "seed": int(cfg.seed),
        "hidden": int(cfg.hidden), "lr": float(cfg.lr),
        "gamma": float(cfg.gamma), "batch": int(cfg.batch_size),
        "pm_preset": config.pm_preset, "ent_preset": config.ent_preset,
        "lam": float(config.lam), "b_max": int(config.b_max),
        "horizon": int(config.horizon), "n_x": int(config.n_x),
        "d_km": float(config.d_km), "alpha": float(config.alpha),
        "odr": float(config.odr_losses), "t0": int(config.t0_deadline),
        # dqn.py is under active development in a parallel session (an
        # observation normaliser). A pickle from before that change would
        # unpickle into the new class without __init__ ever running, leaving
        # an agent missing the attribute its act() needs -- so the module's
        # public surface goes into the key and the cache invalidates itself.
        "dqn_api": sorted(dqn_module.__all__),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def train_dqn(config, progress=None, cache_dir=DQN_CACHE_DIR, use_cache=True):
    """Train (or load) the agent for `config`; return (policy, from_cache).

    `progress(episode, eta, eps)` is `dqn.train`'s own callback, forwarded so
    the window can show "training… (episode k of N)" instead of freezing with
    no explanation.

    The training environment is built from a seed in the config's OWN training
    block, never from the trajectory seed the session displays: `train` resets
    per episode, so the construction seed never reaches a trajectory, but
    keeping the two blocks visibly separate is what makes "evaluated on an
    episode it never trained on" checkable rather than asserted.
    """
    cfg = config.dqn_config()
    if cfg is None:
        cfg = DQNConfig(episodes=int(config.dqn_episodes))
    key = _cache_key(config, cfg)
    path = os.path.join(cache_dir, "dqn_%s.pkl" % key)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh), True
        except Exception:
            # A stale or truncated cache must cost a retrain, never a crash.
            pass
    train_seed = training_seed_range(cfg).start
    env = HybridEnv(config.hybrid_params(train_seed),
                    EveParams(seed=train_seed))
    policy = dqn_module.train(env, cfg, progress=progress)
    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(policy, fh)
        except Exception:
            pass
    return policy, False


# --------------------------------------------------------------------------
# drawing -- no Tk here either, so a Figure can be rendered headless
# --------------------------------------------------------------------------

def _mode_spans(run):
    """{PM: [(start, width), ...], ENT: [...]} for broken_barh."""
    spans = {PM: [], ENT: []}
    if not run.mode:
        return spans
    start = 0
    current = run.mode[0]
    for i, mode in enumerate(run.mode[1:], start=1):
        if mode != current:
            spans[current].append((run.t[start], run.t[i - 1] - run.t[start] + 1))
            start = i
            current = mode
    spans[current].append((run.t[start], run.t[-1] - run.t[start] + 1))
    return spans


def draw_panels(fig, session, title=None):
    """Render the four panels of `session` into `fig`.

    Takes a bare `matplotlib.figure.Figure`, so the window and a headless
    Agg render go through exactly one drawing implementation -- a separate
    "export" path is how a live panel and its published figure drift apart.
    """
    fig.clear()
    axes = fig.subplots(4, 1, sharex=True,
                        gridspec_kw={"height_ratios": [1.1, 1, 1, 1]})
    ax_mode, ax_qber, ax_buf, ax_eta = axes
    fig.patch.set_facecolor(plots.FACE)
    for ax in axes:
        ax.set_facecolor("white")
    runs = session.runs
    horizon = session.params.horizon

    # --- panel 1: which mode each policy is in ---------------------------
    for i, run in enumerate(runs):
        spans = _mode_spans(run)
        for mode, name in ((PM, "pm"), (ENT, "ent")):
            if spans[mode]:
                ax_mode.broken_barh(spans[mode], (i - 0.38, 0.76),
                                    facecolor=plots.MODE_BAND[name],
                                    edgecolor="none")
    ax_mode.set_yticks(range(len(runs)))
    ax_mode.set_yticklabels([r.display_name for r in runs], fontsize=7)
    ax_mode.set_ylim(-0.6, len(runs) - 0.4)
    plots.style_axis(ax_mode, title="mode per policy  (light blue = PM, "
                                    "light green = ENT)")
    # A bound's lane must not read as a schedule anyone could run.
    for i, run in enumerate(runs):
        if run.is_bound:
            ax_mode.annotate("bound", (0.995, i), xycoords=("axes fraction",
                                                            "data"),
                             ha="right", va="center", fontsize=6,
                             color=plots.RED, style="italic")

    # --- panel 2: observed QBER, against both death thresholds -----------
    for run in runs:
        ax_qber.plot([t for t in run.t], [100.0 * q for q in run.qber],
                     color=run.color, lw=1.0, ls=run.linestyle,
                     label=run.display_name)
    ax_qber.axhline(100.0 * session.death_pm, color=plots.MODE_LINE["pm"],
                    lw=1.0, ls=":",
                    label="PM dies %.2f%%" % (100.0 * session.death_pm))
    ax_qber.axhline(100.0 * session.death_ent, color=plots.MODE_LINE["ent"],
                    lw=1.0, ls=":",
                    label="ENT dies %.2f%%" % (100.0 * session.death_ent))
    plots.style_axis(ax_qber, title="observed QBER (windowed)",
                     ylabel="QBER [%]")
    # Headroom for the legend, which would otherwise sit on top of the first
    # few hundred slots -- the stretch where the policies first diverge.
    peak = max([100.0 * q for run in runs for q in run.qber]
               + [100.0 * session.death_pm, 100.0 * session.death_ent])
    ax_qber.set_ylim(0.0, peak * 1.45)
    ax_qber.legend(fontsize=6, ncol=3, loc="upper left", framealpha=0.9)

    # --- panel 3: the key buffer ----------------------------------------
    for run in runs:
        ax_buf.plot(run.t, run.buffer, color=run.color, lw=1.0,
                    ls=run.linestyle)
    # The B_max line is drawn only once the pool climbs near it. At the
    # operating point the study reports (lam = 2.5) demand outruns supply and
    # the buffer never leaves the floor; pinning the axis to B_max there
    # flattens every trace onto zero and the panel shows nothing at all.
    # gui.py hides its death-QBER line on the same rule and for the same
    # reason, so the two panels behave alike.
    peak_buffer = max([b for run in runs for b in run.buffer] or [0])
    b_max = float(session.params.b_max)
    if peak_buffer >= REFERENCE_LINE_SHOW_FRAC * b_max:
        ax_buf.axhline(b_max, color=plots.RED, lw=1.0, ls=":",
                       label="B_max = %.3g" % b_max)
        plots.style_axis(ax_buf, title="key buffer", ylabel="bits")
        ax_buf.legend(fontsize=6, loc="upper left", framealpha=0.9)
    else:
        plots.style_axis(
            ax_buf, ylabel="bits",
            title="key buffer  (B_max = %.3g, off scale: peak %.3g)"
                  % (b_max, peak_buffer))

    # --- panel 4: cumulative eta ----------------------------------------
    for run in runs:
        ax_eta.plot(run.t, run.eta_trace, color=run.color, lw=1.2,
                    ls=run.linestyle)
    plots.style_axis(ax_eta, title="cumulative η  (served / arrived, eq 24)",
                     xlabel="slot", ylabel="η")
    ax_eta.set_ylim(0.0, 1.02)
    ax_eta.set_xlim(0, horizon)

    head = title or ("%s   ent=%s  λ=%.2f  B_max=%.3g  n_X=%.0e  seed=%d"
                     % (BANNER.split("—")[0].strip(), session.config.ent_preset,
                        session.params.lam, session.params.b_max,
                        session.params.n_x, session.seed))
    fig.suptitle(head, fontsize=8, color=plots.NAVY, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def format_scoreboard(session):
    """The live scoreboard as text. Bounds are marked, not silently listed."""
    # 32 columns: "always-pm-relaxed (upper bound)" is 31 characters and
    # a narrower field would push its numbers out of alignment.
    head = ("%-32s %8s %9s %9s %9s %8s"
            % ("policy", "eta", "switches", "ENT frac", "expired", "slots"))
    lines = [head, "-" * len(head)]
    for row in session.scores():
        lines.append("%-32s %8.4f %9d %9.3f %9d %8d"
                     % (row["policy"], row["eta"], row["switches"],
                        row["ent_fraction"], row["expired"], row["slots"]))
    if any(r["is_bound"] for r in session.scores()):
        lines.append("")
        lines.append("(upper bound) = clairvoyant, free switching, key "
                     "delivered continuously — NOT achievable.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the Tk shell -- thin on purpose
# --------------------------------------------------------------------------

def _build_app(config):
    """Import Tk and construct the window. Kept behind a function so that
    importing policyviz costs nothing on a machine with no display."""
    from policyviz_ui import PolicyVizApp
    return PolicyVizApp(config)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ent-preset", default=VizConfig.ent_preset)
    ap.add_argument("--pm-preset", default=VizConfig.pm_preset)
    ap.add_argument("--lam", type=float, default=VizConfig.lam)
    ap.add_argument("--b-max", type=float, default=VizConfig.b_max)
    ap.add_argument("--horizon", type=int, default=VizConfig.horizon)
    ap.add_argument("--seed-index", type=int, default=VizConfig.seed_index)
    ap.add_argument("--dqn-episodes", type=int, default=VizConfig.dqn_episodes)
    ap.add_argument("--policies", default=",".join(VizConfig.specs),
                    help="comma-separated policy specs, e.g. "
                         "'always-pm,greedy-0.045,oracle'")
    ap.add_argument("--no-gui", action="store_true",
                    help="run the episode headless and print the scoreboard")
    ap.add_argument("--figure", default=None,
                    help="with --no-gui, also write the four panels here")
    args = ap.parse_args(argv)

    config = VizConfig(
        specs=tuple(s.strip() for s in args.policies.split(",") if s.strip()),
        pm_preset=args.pm_preset, ent_preset=args.ent_preset, lam=args.lam,
        b_max=int(args.b_max), horizon=args.horizon,
        seed_index=args.seed_index, dqn_episodes=args.dqn_episodes)

    if not args.no_gui:
        _build_app(config).mainloop()
        return

    trained = None
    if "dqn" in config.specs:
        print("training dqn (%d episodes)..." % config.dqn_episodes)
        trained, cached = train_dqn(config)
        print("  %s" % ("loaded from cache" if cached else "trained"))
    session = VizSession(config, dqn_policy=trained)
    session.step(config.horizon)
    print(BANNER)
    print("seed %d (episode index %d)" % (session.seed, config.seed_index))
    print()
    print(format_scoreboard(session))
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        fig = Figure(figsize=(10, 9), dpi=130)
        draw_panels(fig, session)
        fig.savefig(args.figure, facecolor=fig.get_facecolor())
        print("\nwrote %s" % args.figure)


if __name__ == "__main__":
    main()
