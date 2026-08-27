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
import panelstate
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
           "REFERENCE_LINE_SHOW_FRAC", "HIDDEN_LEGEND_COLOR",
           "HIDDEN_LEGEND_ALPHA", "LEGEND_PICK_RADIUS",
           "SHARED_CHANNEL_FIELDS", "PARAM_CLASSES", "PARAM_RULES",
           "PARAM_LABELS", "param_fields", "format_param", "parse_params",
           "strip_channel", "is_modified", "operating_point_label",
           "preset_table", "validate_preset_name", "resolve_envs",
           "VizConfig", "PolicyRun", "VizSession", "parse_spec",
           "death_thresholds", "train_dqn", "dqn_label", "draw_panels",
           "format_scoreboard", "main"]

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

# How a switched-off QBER trace shows in the legend. A toggle whose only
# effect is somewhere else on the figure is a trap: the reader sees a legend
# entry, looks for its curve, and concludes the policy produced no data. The
# entry therefore goes grey AND translucent -- grey alone would be ambiguous
# for the seventh track, which is drawn in plots.GREY to begin with.
HIDDEN_LEGEND_COLOR = plots.GREY
HIDDEN_LEGEND_ALPHA = 0.35
# Pixels of slack around a legend line for a click to count as hitting it.
LEGEND_PICK_RADIUS = 5


# --------------------------------------------------------------------------
# operating points: the two modes' physics, editable
# --------------------------------------------------------------------------
#
# A preset NAME is not enough to identify an operating point any more. The
# window can hand `VizConfig` an `EnvParams`/`EntEnvParams` it edited, so two
# sessions can carry the same preset name and different physics -- which is
# why `_cache_key` below hashes VALUES and never the name, and why
# `operating_point_label` exists to say "(modified)" on screen.

# The three parameters BOTH modes share, because both ride the same fibre.
#
# `HybridEnv._setup_modes` overrides them on both envs from the main window's
# values, so a per-mode copy is not merely redundant -- it is a lie: the
# dialog would show a link the environment is not simulating. CLAUDE.md
# records a comparison of two presets carrying different channels having been
# published once and retracted (PM was handicapped ~20x and the conclusion
# "ENT matches PM" was wrong). They are therefore displayed READ-ONLY in the
# dialog, and re-imposed by _setup_modes rather than trusted.
SHARED_CHANNEL_FIELDS = ("d_km", "alpha", "odr_losses")

PARAM_CLASSES = {"pm": channel.EnvParams, "ent": channel.EntEnvParams}

# field -> (minimum, maximum, minimum is exclusive, maximum is exclusive).
# None for a bound means unbounded on that side. Anything not listed is only
# required to be a finite number.
PARAM_RULES = {
    "d_km":       (0.0, None, False, False),
    "alpha":      (0.0, None, False, False),
    "odr_losses": (0.0, None, False, False),
    "eta_bob":    (0.0, 1.0, True, False),
    "eta_a":      (0.0, 1.0, True, False),
    "pdc":        (0.0, 1.0, False, False),
    "xi_a":       (0.0, 1.0, False, False),
    "xi_b":       (0.0, 1.0, False, False),
    # An "error rate" at or above 1/2 is not an error rate; the finite-key
    # routines return nothing usable there and the message should say why.
    "e_det":      (0.0, 0.5, False, True),
    "e_intr":     (0.0, 0.5, False, True),
    "dead_us":    (0.0, None, False, False),
    "coeff":      (0.0, None, True, False),
    "f_rep":      (0.0, None, True, False),
    "mu1":        (0.0, None, True, False),
    "mu2":        (0.0, None, True, False),
    "mu":         (0.0, None, True, False),
    # p1 = 1 means no decoy pulses at all, which the 1-decoy bound cannot be
    # evaluated at.
    "p1":         (0.0, 1.0, True, True),
    "qber_sigma": (0.0, None, False, False),
}

# Short captions for the dialog. Field names alone ("coeff", "xi_a") do not
# tell a reader what they are looking at.
PARAM_LABELS = {
    "d_km": "fibre length d [km]",
    "alpha": "fibre attenuation alpha [dB/km]",
    "odr_losses": "optics/receiver losses [dB]",
    "eta_bob": "eta_bob detector efficiency",
    "eta_a": "eta_A Alice arm efficiency",
    "pdc": "dark-count prob / gate",
    "e_det": "e_det intrinsic optical error",
    "e_intr": "e_intr intrinsic basis error",
    "xi_a": "xi_A background prob / pulse",
    "xi_b": "xi_B background prob / pulse",
    "dead_us": "detector dead time [us]",
    "coeff": "dead-time correction coeff",
    "f_rep": "repetition rate f_rep [Hz]",
    "mu1": "mu1 signal intensity",
    "mu2": "mu2 decoy intensity",
    "mu": "mu mean pairs / pump pulse",
    "p1": "p1 = P(signal)",
    "qber_sigma": "QBER jitter sigma",
}


def param_fields(protocol):
    """[(name, label, is_shared)] for one mode's dialog, in dataclass order."""
    cls = PARAM_CLASSES[str(protocol)]
    return [(f.name, PARAM_LABELS.get(f.name, f.name),
             f.name in SHARED_CHANNEL_FIELDS)
            for f in dataclasses.fields(cls)]


def format_param(value):
    """How a parameter is written INTO the dialog.

    repr() renders 80000000.0 in full and 1.33e-06 in exponent form; %g keeps
    every one of them short and round-trips through float() unchanged, which
    matters because "Save as preset" writes back exactly what the field shows.
    """
    return "%g" % float(value)


def _check_range(name, label, number):
    rule = PARAM_RULES.get(name)
    if rule is None:
        return
    lo, hi, lo_open, hi_open = rule
    if lo is not None and (number <= lo if lo_open else number < lo):
        raise ValueError("%s: must be %s %g (got %g)"
                         % (label, "greater than" if lo_open else "at least",
                            lo, number))
    if hi is not None and (number >= hi if hi_open else number > hi):
        raise ValueError("%s: must be %s %g (got %g)"
                         % (label, "less than" if hi_open else "at most",
                            hi, number))


def _check_consistency(protocol, env):
    """Cross-field rules a per-field range check cannot see."""
    if protocol == "pm" and env.mu2 >= env.mu1:
        # The 1-decoy bound estimates the single-photon yield from the
        # DIFFERENCE between the two intensities; at mu2 >= mu1 it is solving
        # the wrong inequality and returns a length that bounds nothing.
        raise ValueError("mu2 decoy intensity (%g) must be below mu1 signal "
                         "intensity (%g)" % (env.mu2, env.mu1))


def parse_params(protocol, values, shared):
    """Build one mode's params dataclass from the dialog's text fields.

    `values` is {field: text} as typed; `shared` is {field: number} for the
    three fibre parameters, taken from the MAIN WINDOW rather than from the
    dialog (they are shown read-only there). Anything absent from `values`
    keeps the dataclass default.

    Raises ValueError NAMING THE FIELD, so the window has a message it can put
    in front of the user instead of a traceback. Nothing is applied anywhere
    until this returns, which is what keeps a rejected edit from leaving a
    half-written configuration behind.
    """
    cls = PARAM_CLASSES[str(protocol)]
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError("unknown parameter(s): %s" % sorted(unknown))
    out = {}
    for name, label, is_shared in param_fields(protocol):
        if is_shared:
            out[name] = float(shared[name])
            continue
        if name not in values:
            continue
        text = str(values[name]).strip()
        if not text:
            raise ValueError("%s: no value given" % label)
        try:
            number = float(text)
        except ValueError:
            raise ValueError("%s: %r is not a number" % (label, text))
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError("%s: %r is not a finite number" % (label, text))
        _check_range(name, label, number)
        out[name] = number
    env = cls(**out)
    _check_consistency(protocol, env)
    return env


def strip_channel(env):
    """`env` as a dict with the three shared fibre fields removed.

    Two operating points are "the same" for labelling purposes when they
    agree on everything the dialog can actually edit. The fibre is excluded
    because the dialog shows the MAIN WINDOW's link, which routinely differs
    from the one the preset file happens to carry (aurea ships
    odr_losses = 13 dB; the visualiser's default link has 0), and comparing it
    would report every preset as "modified" the moment it was opened.
    """
    data = dataclasses.asdict(env)
    for name in SHARED_CHANNEL_FIELDS:
        data.pop(name, None)
    return data


def is_modified(env, preset_env):
    """Has this operating point been edited away from its preset?"""
    if env is None or preset_env is None:
        return False
    return strip_channel(env) != strip_channel(preset_env)


def operating_point_label(protocol, preset_name, env, table):
    """What the main window's button (and the figure header) says.

    "PM: aurea" when the preset is running unchanged, "PM: custom" when the
    numbers have been edited away from it. A reader must never be unable to
    tell which physics produced the curve in front of them, and a bare preset
    name on an edited configuration is exactly that failure.
    """
    tag = str(protocol).upper()
    name = str(preset_name)
    preset_env = (table or {}).get(name)
    if env is not None and (preset_env is None
                            or is_modified(env, preset_env)):
        return "%s: %s" % (tag, CUSTOM_NAME)
    return "%s: %s" % (tag, name)


def preset_table(protocol):
    """{name: params} for one protocol, read fresh from presets/.

    Read fresh on every call, not cached: "Save as preset" writes a new file
    and the chooser has to offer it without the app being restarted.
    """
    return (presets.load_presets() if str(protocol) == "pm"
            else presets.load_ent_presets())


# Shown wherever an operating point has been edited away from its preset.
# It is a DISPLAY name, never a lookup key: the configuration keeps the base
# preset name so the physics can still be resolved, and only the label says
# "custom". Reserved in validate_preset_name for the same reason `default` is
# -- a preset actually called "custom" would be indistinguishable from the
# edited state it denotes.
CUSTOM_NAME = "custom"


def validate_preset_name(name, existing=()):
    """Clean and check a name typed into "Save as preset"; returns it.

    `default` is reserved for the saved panel state, and `presets._load`
    RAISES if a presets/default.json exists -- so accepting it here would not
    merely produce a bad preset, it would break every later preset load in the
    process, including the ones this window needs in order to redraw itself.

    `existing` is accepted so a caller can ask whether it would overwrite; the
    decision to overwrite is the caller's (the window asks first).
    """
    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("give the preset a name")
    if cleaned == panelstate.RESERVED_NAME:
        raise ValueError("%r is reserved for the saved panel state -- choose "
                         "another name" % (panelstate.RESERVED_NAME,))
    if cleaned == CUSTOM_NAME:
        raise ValueError("%r is reserved: it is what an edited operating "
                         "point is called, so a preset of that name could "
                         "not be told apart from one -- choose another name"
                         % (CUSTOM_NAME,))
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", cleaned):
        raise ValueError("preset names may use letters, digits, '_', '-' and "
                         "'.', and must start with a letter or digit (got %r)"
                         % (cleaned,))
    return cleaned


def resolve_envs(config):
    """(pm_env, ent_env) ACTUALLY in force for `config`, fibre included.

    One place resolves "edited env, else named preset", and one place imposes
    the shared fibre, so the death-threshold lines, the DQN cache key and
    `HybridEnv._setup_modes` cannot end up describing three slightly different
    experiments.
    """
    pm_env = (config.pm_env if config.pm_env is not None
              else preset_table("pm")[config.pm_preset])
    ent_env = (config.ent_env if config.ent_env is not None
               else preset_table("ent")[config.ent_preset])
    shared = dict(d_km=float(config.d_km), alpha=float(config.alpha),
                  odr_losses=float(config.odr_losses))
    return (dataclasses.replace(pm_env, **shared),
            dataclasses.replace(ent_env, **shared))


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
    l0: int = 8192              # key bits to encrypt one message (OTP)
    w_window: int = 5           # slots the observed QBER averages over
    seed_index: int = 0
    dqn_episodes: int = 20
    # EDITED OPERATING POINTS. None = "run the named preset as it ships".
    #
    # The preset NAME is kept even when an env is present: it is what the
    # button and the figure header are labelled with, and "aurea (modified)"
    # tells a reader far more than a bare list of numbers would.
    #
    # These are `channel.EnvParams` / `channel.EntEnvParams`, i.e. MUTABLE
    # dataclasses on a value object. Nothing here ever mutates them in place
    # -- the dialog builds a fresh one on OK -- because a VizConfig is
    # compared, replaced and hashed into the DQN cache key.
    pm_env: object = None
    ent_env: object = None

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
            t0_deadline=int(self.t0_deadline), l0=int(self.l0),
            w_window=int(self.w_window), horizon=int(self.horizon),
            d_km=float(self.d_km), alpha=float(self.alpha),
            odr_losses=float(self.odr_losses), seed=seed,
            pm_preset=self.pm_preset, ent_preset=self.ent_preset,
            pm_env=self.pm_env, ent_env=self.ent_env)

    def operating_points(self):
        """(pm label, ent label) as the button and the header say them."""
        return (operating_point_label("pm", self.pm_preset, self.pm_env,
                                      preset_table("pm")),
                operating_point_label("ent", self.ent_preset, self.ent_env,
                                      preset_table("ent")))


def death_thresholds(config):
    """(PM, ENT) death QBER for `config`, in the units panel 2 plots.

    Goes through `resolve_envs`, so an EDITED operating point moves the line
    with it: a threshold drawn from the preset file while the environment runs
    edited parameters would tell the reader a dead trace is alive.

    Both modes ride the same fibre, so the presets' own d_km/alpha/odr_losses
    are overridden exactly as `HybridEnv._setup_modes` overrides them --
    reading a preset's channel here instead would draw a threshold line for a
    link the environment is not simulating. (CLAUDE.md records that trap
    having been published once and retracted.)

    Costs ~25 finite-key evaluations per mode; call it once per session, not
    once per frame.
    """
    pm_env, ent_env = resolve_envs(config)
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
        # Resolved ONCE, at construction, from the same `resolve_envs` the
        # environment and the cache key use -- so the label on screen cannot
        # describe one operating point while the traces come from another.
        self.pm_label, self.ent_label = self.config.operating_points()
        self.death_pm, self.death_ent = death_thresholds(self.config)
        # WHICH QBER TRACES ARE SWITCHED OFF, by display name.
        #
        # It lives here and not on the Line2D objects because the window
        # redraws the whole figure several times a second: `draw_panels`
        # calls `fig.clear()` and builds brand-new lines and a brand-new
        # legend every frame, so a visibility flag set on an artist would be
        # thrown away on the next tick and the user's click would silently
        # undo itself mid-episode. `draw_panels` READS this set on every draw
        # instead.
        #
        # It lives on the session and not on `VizConfig` because VizConfig is
        # a value object describing the EPISODE: it is compared, replaced and
        # hashed into the DQN cache key, and two configurations that differ
        # only in which curve is on screen are not two different experiments.
        # The session is also the only object `draw_panels` is handed.
        self.hidden_traces = set()
        # {artist: policy name} for this frame's clickable legend entries.
        # Refreshed by every `draw_panels`; it is a lookup table, not state.
        self.legend_picks = {}

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

    def trace_visible(self, name):
        """Is this policy's line switched on in the QBER and eta panels?"""
        return name not in self.hidden_traces

    def toggle_trace(self, name):
        """Flip one policy's QBER trace. Returns its new visibility.

        Only panel 2 consults this. The mode lanes, the buffer panel, the
        cumulative-eta panel and the scoreboard all ignore it, deliberately:
        a policy the reader has cleared off the QBER panel is still running,
        and a figure that let a click erase it everywhere would let someone
        conclude a policy was never selected.
        """
        if name in self.hidden_traces:
            self.hidden_traces.discard(name)
            return True
        self.hidden_traces.add(name)
        return False

    def reset(self):
        # Rewinding the EPISODE, not the view: which traces the reader has
        # switched off is not part of the trajectory and survives a Reset,
        # exactly as it survives Start, Pause and Step.
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
    so the operating point has to be in the key or flipping the chooser would
    silently reuse the wrong weights.

    THE KEY IS BUILT FROM VALUES, NOT FROM PRESET NAMES. Once the parameters
    are editable a name no longer identifies an operating point: two runs can
    both say "ent_advanced" and differ in mu, and a cache keyed on the name
    would hand back an agent trained on the other physics -- with a perfectly
    plausible-looking curve and nothing on screen to say so. `resolve_envs`
    returns exactly what `HybridEnv._setup_modes` will build from, shared
    fibre included, so the key covers every number the environment can see.
    """
    pm_env, ent_env = resolve_envs(config)
    payload = {
        "episodes": int(cfg.episodes), "seed": int(cfg.seed),
        "hidden": int(cfg.hidden), "lr": float(cfg.lr),
        "gamma": float(cfg.gamma), "batch": int(cfg.batch_size),
        "pm_env": dataclasses.asdict(pm_env),
        "ent_env": dataclasses.asdict(ent_env),
        "lam": float(config.lam), "b_max": int(config.b_max),
        "horizon": int(config.horizon), "n_x": int(config.n_x),
        "d_km": float(config.d_km), "alpha": float(config.alpha),
        "odr": float(config.odr_losses), "t0": int(config.t0_deadline),
        "l0": int(config.l0), "w_window": int(config.w_window),
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


def _wire_qber_legend(legend, session, names):
    """Make panel 2's policy entries clickable, and show which are off.

    `names` runs parallel to the legend's entries, carrying None where an
    entry is a reference line rather than a policy.

    THE DEATH THRESHOLDS ARE DELIBERATELY NOT TOGGLEABLE. They are what the
    traces are read AGAINST -- the whole question the panel answers is
    "which policy is above the line" -- so a control that could take the line
    away would let a reader look at a dead trace and see nothing wrong. They
    are also two thin dotted lines, not the six overlapping step traces the
    toggle exists to thin out. Their entries keep full colour, so it is
    visible that they are captions and not controls.

    Returns {artist: policy name} for the window's pick handler. The map is
    rebuilt here on every draw because the legend itself is; the window
    connects its handler ONCE, at construction, and looks names up through
    this map rather than through the artists it happens to be holding.
    """
    picks = {}
    for handle, text, name in zip(legend.get_lines(), legend.get_texts(),
                                  names):
        if name is None:
            continue
        # Both halves of the entry answer to a click: aiming at a 6 pt line
        # sample is fiddly, and a reader who clicks the words and gets
        # nothing concludes the control is broken.
        handle.set_picker(LEGEND_PICK_RADIUS)
        text.set_picker(True)
        picks[handle] = name
        picks[text] = name
        if not session.trace_visible(name):
            handle.set_color(HIDDEN_LEGEND_COLOR)
            handle.set_alpha(HIDDEN_LEGEND_ALPHA)
            text.set_color(HIDDEN_LEGEND_COLOR)
            text.set_alpha(HIDDEN_LEGEND_ALPHA)
    return picks


def draw_panels(fig, session, title=None):
    """Render the four panels of `session` into `fig`.

    Takes a bare `matplotlib.figure.Figure`, so the window and a headless
    Agg render go through exactly one drawing implementation -- a separate
    "export" path is how a live panel and its published figure drift apart.

    Every artist here is thrown away and rebuilt on each call (`fig.clear()`
    below), which is why panel 2 reads `session.hidden_traces` rather than
    trusting any flag left on a line. As a side effect the call refreshes
    `session.legend_picks`, the map the window's pick handler resolves a
    clicked legend entry through.
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
    # A bound's lane must not read as a schedule anyone could run. The tag
    # goes at the LEFT edge: pinned to the right it sits exactly where the
    # last slot is drawn and reads as a terminus -- a real reader took it to
    # mean the run had stopped there, when the episode had simply reached
    # its horizon.
    for i, run in enumerate(runs):
        if run.is_bound:
            ax_mode.annotate("bound", (0.005, i), xycoords=("axes fraction",
                                                            "data"),
                             ha="left", va="center", fontsize=6,
                             color=plots.RED, style="italic")

    # --- panel 2: observed QBER, against both death thresholds -----------
    # Handles are collected and passed to legend() explicitly, so the entries
    # keep the order of `runs` and `_wire_qber_legend` can line each one up
    # with the policy it belongs to. `None` marks an entry that is a
    # reference line rather than a policy.
    handles, labels, names = [], [], []
    for run in runs:
        line, = ax_qber.plot([t for t in run.t], [100.0 * q for q in run.qber],
                             color=run.color, lw=1.0, ls=run.linestyle,
                             label=run.display_name)
        # READ from the session, every frame. See VizSession.hidden_traces for
        # why the flag cannot be left on this line.
        line.set_visible(session.trace_visible(run.display_name))
        handles.append(line)
        labels.append(run.display_name)
        names.append(run.display_name)
    for mode, death in (("pm", session.death_pm), ("ent", session.death_ent)):
        label = "%s dies %.2f%%" % (mode.upper(), 100.0 * death)
        handles.append(ax_qber.axhline(100.0 * death,
                                       color=plots.MODE_LINE[mode], lw=1.0,
                                       ls=":", label=label))
        labels.append(label)
        names.append(None)
    plots.style_axis(ax_qber,
                     title="observed QBER (windowed)  —  click a policy in "
                           "the legend to hide/show it here and in η",
                     ylabel="QBER [%]")
    # Headroom for the legend, which would otherwise sit on top of the first
    # few hundred slots -- the stretch where the policies first diverge.
    #
    # HIDDEN TRACES STILL SET THE LIMIT. Rescaling to whatever is left on
    # screen would change the shape and apparent slope of the remaining
    # curves on every click, so two screenshots of the same episode would
    # carry two different vertical scales -- and the reader was hiding a
    # trace in order to COMPARE the rest, which a moving axis defeats. The
    # complaint this control answers is overlap, not scale.
    peak = max([100.0 * q for run in runs for q in run.qber]
               + [100.0 * session.death_pm, 100.0 * session.death_ent])
    ax_qber.set_ylim(0.0, peak * 1.45)
    legend = ax_qber.legend(handles, labels, fontsize=6, ncol=3,
                            loc="upper left", framealpha=0.9)
    session.legend_picks = _wire_qber_legend(legend, session, names)

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
    # Same visibility set as panel 2: with six tracks the eta curves bunch as
    # tightly as the QBER traces do. The MODE LANES and the SCOREBOARD still
    # ignore it, deliberately -- a policy cleared off the line panels is still
    # running, and hiding it everywhere would let someone conclude it was
    # never selected.
    for run in runs:
        line, = ax_eta.plot(run.t, run.eta_trace, color=run.color, lw=1.2,
                            ls=run.linestyle)
        line.set_visible(session.trace_visible(run.display_name))
    plots.style_axis(ax_eta, title="cumulative η  (served / arrived, eq 24)",
                     xlabel="slot", ylabel="η")
    ax_eta.set_ylim(0.0, 1.02)
    ax_eta.set_xlim(0, horizon)

    # The operating-point labels, not the bare preset names: an edited
    # configuration that printed "ent=ent_advanced" would put a figure in a
    # thesis under the name of physics it was not run with.
    head = title or ("%s   %s  %s  λ=%.2f  B_max=%.3g  n_X=%.0e  seed=%d"
                     % (BANNER.split("—")[0].strip(), session.pm_label,
                        session.ent_label, session.params.lam,
                        session.params.b_max, session.params.n_x,
                        session.seed))
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
