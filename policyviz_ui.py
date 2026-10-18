#!/usr/bin/env python3
"""The window for policyviz.py. Thin on purpose.

Everything worth testing lives in `policyviz` and never imports Tk; this file
reads the form, rebuilds a `policyviz.VizSession`, drives a timer and calls
`policyviz.draw_panels`. If a behaviour is being written here that a test
would want to make an assertion about, it belongs in `policyviz` instead.

One state this window owes the viewer an explanation for:

* **bound** -- the Oracle row is dashed and labelled, in the figure and in the
  scoreboard both. See policyviz's module docstring.

The DQN was taken out of this window: there is no checkbox for it, no episode
count and no training thread. It is still in dqn.py for the study's tables.

THE SAVED CONFIGURATION
-----------------------
Save snapshots the panel to `vizstate`; Restore puts it back; the startup
overlay happens in `__init__`, AFTER `_build()` has seeded the form from the
incoming `VizConfig`. SAVE DOES NOT APPLY AND APPLY DOES NOT SAVE -- see
`_save_state` for why either coupling would cost the user something they did
not ask to give up. Neither the module nor the class reads the file on its
own: the two entry points (`policyviz._build_app` and `main()` here) read it
and hand the state in, which is what keeps the test suite independent of
whether the developer has saved a configuration of their own.

THE PARAMETER DIALOGS
---------------------
`ParamsDialog` edits one mode's operating point. It owns no physics: every
number it produces goes through `policyviz.parse_params`, every name through
`policyviz.validate_preset_name`, and the fibre it shows is read-only because
`HybridEnv._setup_modes` overrides it on both modes anyway (see
`policyviz.SHARED_CHANNEL_FIELDS`).

It is written so that NOTHING in it has to be opened modally to be tested:
`show()` is the only blocking call, and the window reaches it through the one
seam `_params_dialog_result`. Every modal in this file -- showerror, showinfo,
askyesno, askstring -- is reached through the module-level `messagebox` and
`simpledialog` names so a test can record them instead of waiting for a click
that will never come. CLAUDE.md records an unattended run hanging eight
minutes on exactly that.
"""

import dataclasses
import os
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import presets as preset_store  # noqa: E402
import policyviz  # noqa: E402
import vizstate  # noqa: E402
from evemarkov import ADVERSARIES  # noqa: E402
from policies import (BufferAwareThreshold, DecoyWatch,  # noqa: E402
                      GreedyThreshold)

TICK_MS = 80            # timer period; `speed` sets slots advanced per tick
# Redrawing is throttled SEPARATELY from stepping. A full 3,600-slot figure
# with six tracks costs ~0.33 s to render, so redrawing on every 80 ms tick
# would leave the mainloop doing nothing but paint -- the buttons would go
# unresponsive and the episode would crawl. Stepping stays on TICK_MS (that
# is what the speed slider controls); the picture refreshes at most this
# often. Step and Reset force a redraw regardless, because there the whole
# point is to see the single slot that just happened.
REDRAW_MS = 400
STATUS_COLORS = {"idle": "#9e9e9e",
                 "running": "#5cb85c", "paused": "#31708f",
                 "finished": "#2E75B6", "error": "#d9534f"}

# (field, label, width) for the plain numeric entries.
# (name, label, entry width). A row whose name is None is a heading, not a
# field: the form had grown to the point where a flat list of nine numbers
# gave no clue which described the traffic, which the fibre and which the run.
FIELDS = [
    (None, "traffic", None),
    ("lam", "λ arrivals/slot", 8),
    ("l0", "bits per message  l₀", 10),
    ("t0_deadline", "deadline T₀ [slots]", 8),
    ("b_max", "B_max [bits]", 10),
    (None, "fibre (shared by both modes)", None),
    ("d_km", "d [km]", 8),
    ("alpha", "α [dB/km]", 8),
    ("odr_losses", "optics losses [dB]", 8),
    (None, "run", None),
    ("horizon", "horizon [slots]", 8),
    ("n_x", "n_X block [bits]", 10),
    ("w_window", "QBER window [slots]", 8),
    ("seed_index", "seed (episode #)", 8),
]

# Everything else is read as an int. Getting this wrong turns 0.2 dB/km into
# 0 dB/km -- a lossless fibre -- without raising anything.
FLOAT_FIELDS = frozenset(("lam", "d_km", "alpha", "odr_losses"))

# How tall the scrolling control column is allowed to be. Chosen to leave a
# usable strip of the desktop (a terminal, usually) visible under the window
# on a 1080-line screen.
LEFT_VIEW_HEIGHT = 620

# The same rows, split into the two side-by-side columns the form draws them
# in. Stacked in one column the panel grew taller than the screen and hid the
# terminal behind it. FIELDS stays the flat concatenation, so read-back and
# every test that walks it are unaffected by the layout.
FIELD_COLUMNS = [FIELDS[:FIELDS.index((None, "run", None))],
                 FIELDS[FIELDS.index((None, "run", None)):]]

# The policy-parameter boxes, as (saved key, attribute name, label).
THRESHOLD_FIELDS = (("theta", "theta", "greedy θ"),
                    ("ba_theta", "ba_theta", "buffer-aware θ"),
                    ("ba_reserve", "ba_reserve", "buffer-aware reserve"))

# The decoy controller's three boxes, same shape. Kept apart from
# THRESHOLD_FIELDS so `_parameters_from` keeps its three-value contract.
# The QBER alarm defaults to 1.0 -- never fires -- because against the default
# adversary (PNS) the QBER does not move, and measured, reacting to light
# jamming only pays at small blocks (it HURTS from n_X = 3e5 up).
DECOY_FIELDS = (("dc_theta", "dc_theta", "decoy excess threshold"),
                ("dc_qber", "dc_qber", "decoy QBER alarm (1 = off)"),
                ("dc_probe", "dc_probe", "decoy probe every [slots]"))

# One line per adversary for the dropdown, so the choice explains itself.
ADVERSARY_HINTS = {
    "pns": "PNS: kills PM's key, QBER does NOT move",
    "jam": "jamming: raises QBER, hurts both modes",
    "both": "both weapons (on 53% of slots: ENT wins)",
    "none": "no attack (contrast)",
}

# The speed slider's range. A saved value is clamped into it: a file holding
# 10000 would leave the slider pinned off its own scale, showing a number the
# widget cannot represent.
SPEED_MIN, SPEED_MAX = 1, 400


def _parse_or_none(spec):
    """The policy for `spec`, or None if this version cannot build it.

    A saved configuration may name a policy that no longer exists -- "dqn",
    from before the DQN was taken out of the window. Reading parameters out of
    the spec list must skip it rather than stop the window from opening.
    """
    try:
        return policyviz.parse_spec(spec)
    except ValueError:
        return None


def format_field(name, value):
    """How a saved number is written back INTO a form field.

    The int-valued boxes are restored as integers, and through str() rather
    than "%g": `repr(3600.0)` would put "3600.0" in the horizon box and "%g"
    would put "1e+06" in B_max. Both survive `_read_config` (it reads every
    box through float()), but both read as a typo to the user, who then
    "corrects" a number that was already right.
    """
    if name in FLOAT_FIELDS:
        return "%g" % float(value)
    return str(int(float(value)))


class ParamsDialog(tk.Toplevel):
    """Edit ONE mode's operating point: choose a preset, or type the numbers.

    `result` is (preset name, params dataclass) after OK and None after
    Cancel -- so a caller that forgets to check it gets None, not a
    half-applied configuration.

    The three fibre parameters are shown but DISABLED. Both modes ride the
    same fibre and `HybridEnv._setup_modes` overrides them from the main
    window on both envs; an editable copy here would let the reader believe
    they had set a per-mode link that the environment then ignored. They are
    still written into a saved preset (the dataclass has no hole in it), so a
    preset saved from this dialog still round-trips through presets.load_*.
    """

    def __init__(self, master, protocol, preset_name, env, shared):
        super().__init__(master)
        self.protocol_name = str(protocol)
        self.shared = {k: float(v) for k, v in dict(shared).items()}
        self.table = policyviz.preset_table(self.protocol_name)
        self.result = None
        self.title("%s operating point" % self.protocol_name.upper())
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)
        row = 0
        ttk.Label(frame, text="preset").grid(row=row, column=0, sticky="w")
        # The chooser DISPLAYS "custom" once the fields no longer match a
        # preset, but the dialog keeps the real preset name in _base_preset:
        # that is what resolves to physics, and "custom" is not a lookup key.
        self._base_preset = str(preset_name)
        self._filling = False
        self.preset_var = tk.StringVar(value=str(preset_name))
        self.preset_box = ttk.Combobox(
            frame, textvariable=self.preset_var, width=20, state="readonly",
            values=self._chooser_values())
        self.preset_box.grid(row=row, column=1, sticky="w")
        self.preset_box.bind("<<ComboboxSelected>>", self._on_preset)
        row += 1
        ttk.Label(frame, text="(choosing a preset refills the fields below; "
                              "edit them freely afterwards)",
                  font=("TkDefaultFont", 7), foreground="#666666",
                  wraplength=300, justify="left"
                  ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        self.entries = {}          # editable field -> StringVar
        self.shared_vars = {}      # read-only fibre field -> StringVar
        for name, label, is_shared in policyviz.param_fields(
                self.protocol_name):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar()
            entry = ttk.Entry(frame, textvariable=var, width=14)
            entry.grid(row=row, column=1, sticky="w")
            if is_shared:
                entry.state(["disabled"])
                self.shared_vars[name] = var
            else:
                self.entries[name] = var
                # A hand edit means the fields stop describing any preset.
                # Guarded by _filling so refilling from a preset (or the
                # initial fill) does not mark the dialog custom on sight.
                var.trace_add("write", self._mark_custom)
            row += 1
        ttk.Label(frame, text="fibre length, attenuation and optics losses "
                              "are SHARED by both modes and are set in the "
                              "main window",
                  font=("TkDefaultFont", 7), foreground="#8A6D3B",
                  wraplength=300, justify="left"
                  ).grid(row=row, column=0, columnspan=2, sticky="w",
                         pady=(4, 0))
        row += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(buttons, text="Save as preset…", command=self._save
                   ).pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self._cancel
                   ).pack(side="right")
        ttk.Button(buttons, text="OK", command=self._ok
                   ).pack(side="right", padx=4)

        self.fill(env)
        # Opened on an already-edited operating point: say so straight away
        # rather than showing the preset name the numbers no longer match.
        if policyviz.is_modified(env, self.table.get(self._base_preset)):
            self.preset_var.set(policyviz.CUSTOM_NAME)

    # ------------------------------------------------------------------

    def _chooser_values(self):
        """Preset names with "custom" first -- it is a state, not a source."""
        return [policyviz.CUSTOM_NAME] + sorted(self.table)

    def _mark_custom(self, *_args):
        """Any hand edit means the fields no longer describe a preset."""
        if self._filling:
            return
        if self.preset_var.get() != policyviz.CUSTOM_NAME:
            self.preset_var.set(policyviz.CUSTOM_NAME)

    def fill(self, env):
        """Write `env` into the fields. The fibre always comes from `shared`.

        Filling the fibre from `env` instead would be the quiet way to
        reintroduce a per-mode channel: a preset carrying odr_losses = 13 dB
        would display 13 while the environment ran the main window's 0.
        """
        data = dataclasses.asdict(env)
        self._filling = True
        try:
            for name, var in self.entries.items():
                var.set(policyviz.format_param(data[name]))
            for name, var in self.shared_vars.items():
                var.set(policyviz.format_param(self.shared[name]))
        finally:
            self._filling = False

    def read_env(self):
        """The params dataclass the fields currently describe.

        Raises ValueError with a field-named message. Used by BOTH OK and
        Save as preset, so what gets saved is what the dialog is showing and
        what OK applies -- never the preset the dialog was opened on.
        """
        return policyviz.parse_params(
            self.protocol_name,
            {name: var.get() for name, var in self.entries.items()},
            self.shared)

    def _on_preset(self, *_event):
        name = self.preset_var.get()
        if name == policyviz.CUSTOM_NAME:
            # "custom" describes the fields; it cannot supply them.
            return
        env = self.table.get(name)
        if env is not None:
            self._base_preset = name
            self.fill(env)

    def _ok(self):
        try:
            env = self.read_env()
        except ValueError as exc:
            # The dialog STAYS OPEN: closing it on a bad number would throw
            # away every other edit the user had made.
            messagebox.showerror("%s operating point"
                                 % self.protocol_name.upper(), str(exc))
            return
        # The base preset name, never the chooser's text: "custom" is a
        # label, and handing it back would make the next preset lookup fail.
        self.result = (self._base_preset, env)
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

    def _save(self):
        """Write the CURRENT FIELDS to presets/<name>.json.

        Validated first, so a preset file can never hold numbers the dialog
        would refuse to apply.
        """
        title = "%s operating point" % self.protocol_name.upper()
        try:
            env = self.read_env()
        except ValueError as exc:
            messagebox.showerror(title, str(exc))
            return
        raw = simpledialog.askstring(title, "Save these parameters as preset:",
                                     parent=self)
        if raw is None:
            return                      # the name prompt was cancelled
        try:
            name = policyviz.validate_preset_name(raw)
        except ValueError as exc:
            messagebox.showerror(title, str(exc))
            return
        if name in policyviz.preset_table(self.protocol_name):
            if not messagebox.askyesno(
                    title, "Preset %r already exists. Overwrite it?" % name):
                return
        try:
            path = preset_store.save_preset(name, env)
        except Exception as exc:        # disk full, read-only tree, ...
            messagebox.showerror(title, "could not save preset: %s" % exc)
            return
        # Reload from disk rather than inserting into the copy in hand: that
        # is the only version that proves the file we just wrote can be read
        # back, and it is what the chooser must offer without a restart.
        self.table = policyviz.preset_table(self.protocol_name)
        self.preset_box.configure(values=self._chooser_values())
        self._base_preset = name
        self.preset_var.set(name)
        messagebox.showinfo(title, "saved %s" % path)

    def show(self):
        """Open modally and return `result`. The ONLY blocking call here."""
        self.grab_set()
        self.wait_window(self)
        return self.result


class PolicyVizApp(tk.Tk):
    """The window.

    `saved` is a `vizstate.load_state()` dict overlaid on the form once it has
    been built, or None for "start on `config` alone". THE FILE IS NOT READ
    HERE: the callers that own a real session (`policyviz._build_app` and
    `main()` below) read it and hand the result in. A constructor that read
    the disk by default would make every test in the suite depend on whether
    the developer happened to have saved a configuration.
    """

    def __init__(self, config=None, saved=None):
        super().__init__()
        self.title("Hybrid policy visualiser — NUMERIC MODEL (not the stack)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.config_obj = config or policyviz.VizConfig()
        # THE EDITED OPERATING POINTS LIVE HERE, one per mode, and nowhere
        # else. None means "run the named preset as it ships"; a dataclass
        # means the dialog produced numbers that differ from it. `_read_config`
        # copies them into the VizConfig it builds, `VizConfig.hybrid_params`
        # copies them into HybridParams, and `HybridEnv._setup_modes` runs
        # them -- so a value typed into the dialog reaches the physics by one
        # path with no branch in it.
        #
        # They are normalised back to None by `_set_operating_point` whenever
        # they match the preset, so `is None` is a reliable "vanilla preset"
        # test everywhere downstream.
        self.mode_env = {"pm": self.config_obj.pm_env,
                         "ent": self.config_obj.ent_env}
        self.session = None
        self._running = False
        self._last_draw = 0.0
        # Traces to switch off in the session that has not been built yet.
        # A restored `hidden_traces` cannot be written to `self.session`
        # (there is none during startup), so it waits here and
        # `_build_session` folds it in.
        self._pending_hidden = set()
        self._build()
        # ORDER: the form is SEEDED from `config` in _build() and the saved
        # state is overlaid on top of it here. The other way round -- overlay
        # first, seed second -- would have the seeding wipe the restored
        # values; and seeding first is also what lets a file written before a
        # control existed still load, because a control the file says nothing
        # about is left holding its seeded value rather than being blanked.
        if saved is not None:
            self._apply_state(saved)
        self._apply()
        self.after(TICK_MS, self._tick)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------

    def _build(self):
        banner = tk.Label(self, text=policyviz.BANNER, bg="#FFF3CD",
                          fg="#8A6D3B", font=("TkDefaultFont", 9, "bold"),
                          wraplength=1180, justify="left", padx=8, pady=4)
        banner.grid(row=0, column=0, columnspan=2, sticky="ew")

        # The control column is TALLER THAN THE SCREEN on a 1080-line display
        # (it asked for 965 px of a 1080 px screen, so the window covered the
        # terminal behind it). Rather than keep trimming rows every time a
        # parameter is added, it lives in a scrolling viewport: the window can
        # then be any height and the column simply scrolls inside it.
        viewport = ttk.Frame(self)
        viewport.grid(row=1, column=0, sticky="ns")
        canvas = tk.Canvas(viewport, borderwidth=0, highlightthickness=0,
                           height=LEFT_VIEW_HEIGHT)
        bar = ttk.Scrollbar(viewport, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(canvas, padding=8)
        canvas.create_window((0, 0), window=left, anchor="nw")

        def _fit(_event=None):
            # Width follows the content so nothing is cut off sideways; height
            # stays at the viewport and the scrollbar covers the rest.
            canvas.configure(scrollregion=canvas.bbox("all"),
                             width=left.winfo_reqwidth())

        left.bind("<Configure>", _fit)
        # Button-4/5 are X11's wheel events, which is what WSLg delivers.
        for seq, delta in (("<Button-4>", -1), ("<Button-5>", 1)):
            canvas.bind_all(seq, lambda e, d=delta: canvas.yview_scroll(d, "units"))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1,
                                                      "units"))
        self._left_canvas = canvas
        right = ttk.Frame(self, padding=4)
        right.grid(row=1, column=1, sticky="nsew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(left, text="policies", font=("TkDefaultFont", 9, "bold")
                  ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.checks = {}
        selected = set(self._stems(self.config_obj.specs))
        for stem in policyviz.SPEC_CHOICES:
            var = tk.BooleanVar(value=stem in selected)
            text = stem
            if stem == "oracle":
                text = "oracle  (UPPER BOUND, not achievable)"
            elif stem == "always-pm-relaxed":
                text = "always-pm-relaxed  (bound control)"
            ttk.Checkbutton(left, text=text, variable=var).grid(
                row=row, column=0, columnspan=2, sticky="w")
            self.checks[stem] = var
            row += 1

        # Seeded from the incoming specs where they carry parameters, so a
        # session launched with --policies greedy-0.02 does not show a
        # checkbox ticked next to a threshold it is not running.
        theta, ba_theta, ba_reserve = self._parameters_from(
            self.config_obj.specs)
        self.theta = tk.StringVar(value=theta)
        self.ba_theta = tk.StringVar(value=ba_theta)
        self.ba_reserve = tk.StringVar(value=ba_reserve)
        dc_theta, dc_qber, dc_probe = self._decoy_parameters_from(
            self.config_obj.specs)
        self.dc_theta = tk.StringVar(value=dc_theta)
        self.dc_qber = tk.StringVar(value=dc_qber)
        self.dc_probe = tk.StringVar(value=dc_probe)
        for _key, attr, label in THRESHOLD_FIELDS + DECOY_FIELDS:
            var = getattr(self, attr)
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(left, textvariable=var, width=8).grid(row=row, column=1,
                                                            sticky="w")
            row += 1

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        # THE ATTACKER. Chosen here rather than hard-wired: the window used to
        # build Eve as the jammer unconditionally, so the attack the study
        # turns on could not be shown at all.
        ttk.Label(left, text="adversary",
                  font=("TkDefaultFont", 9, "bold")
                  ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.adversary_choices = sorted(ADVERSARIES)
        self.adversary_var = tk.StringVar(value=self.config_obj.adversary)
        combo = ttk.Combobox(left, textvariable=self.adversary_var,
                             values=self.adversary_choices, state="readonly",
                             width=8)
        combo.grid(row=row, column=0, sticky="w")
        self.adversary_hint = ttk.Label(left, font=("TkDefaultFont", 7),
                                        foreground="#666666", wraplength=170,
                                        justify="left")
        self.adversary_hint.grid(row=row, column=1, sticky="w")
        self.adversary_var.trace_add(
            "write", lambda *_: self._refresh_adversary_hint())
        self._refresh_adversary_hint()
        row += 1

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        # OPERATING POINTS. One button per mode, and the button IS the
        # readout: it carries the preset name and, when the numbers have been
        # edited away from it, the word "(modified)". A control that only said
        # "PM parameters…" would leave a reader unable to tell which physics
        # produced the curve on screen -- which is the whole reason the preset
        # name used to be visible in a dropdown here.
        ttk.Label(left, text="operating points",
                  font=("TkDefaultFont", 9, "bold")
                  ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(left, text="(click to choose a preset or type the "
                             "parameters; takes effect on Apply)",
                  font=("TkDefaultFont", 7), foreground="#666666",
                  wraplength=200, justify="left"
                  ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.preset_var = {"pm": tk.StringVar(value=self.config_obj.pm_preset),
                           "ent": tk.StringVar(
                               value=self.config_obj.ent_preset)}
        self.mode_buttons = {}
        for protocol in ("pm", "ent"):
            btn = ttk.Button(left, text=protocol.upper(),
                             command=lambda m=protocol: self._edit_params(m))
            btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=1)
            self.mode_buttons[protocol] = btn
            row += 1
        self._refresh_mode_buttons()

        self.entries = {}
        fields = ttk.Frame(left)
        fields.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        for col, group in enumerate(FIELD_COLUMNS):
            base = col * 2
            if base:
                # A little air between the two columns of entries.
                fields.grid_columnconfigure(base, minsize=14)
            frow = 0
            for name, label, width in group:
                if name is None:
                    ttk.Label(fields, text=label,
                              font=("TkDefaultFont", 8, "bold")
                              ).grid(row=frow, column=base, columnspan=2,
                                     sticky="w", pady=(6, 1))
                    frow += 1
                    continue
                ttk.Label(fields, text=label).grid(row=frow, column=base,
                                                   sticky="w", padx=(0, 4))
                var = tk.StringVar(value=str(getattr(self.config_obj, name)))
                ttk.Entry(fields, textvariable=var, width=width).grid(
                    row=frow, column=base + 1, sticky="w")
                self.entries[name] = var
                frow += 1

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        buttons = ttk.Frame(left)
        buttons.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.start_btn = ttk.Button(buttons, text="Start",
                                    command=self._toggle_run)
        self.start_btn.pack(side="left")
        ttk.Button(buttons, text="Step", command=self._step_once
                   ).pack(side="left", padx=3)
        ttk.Button(buttons, text="Reset", command=self._reset
                   ).pack(side="left")
        ttk.Button(buttons, text="Apply", command=self._apply
                   ).pack(side="left", padx=3)

        # SAVE DOES NOT APPLY, AND APPLY DOES NOT SAVE. See _save_state.
        saved_row = ttk.Frame(left)
        saved_row.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Button(saved_row, text="Save", command=self._save_state
                   ).pack(side="left")
        ttk.Button(saved_row, text="Restore", command=self._restore_state
                   ).pack(side="left", padx=3)
        ttk.Label(saved_row, text="(this form, reloaded at startup)",
                  font=("TkDefaultFont", 7), foreground="#666666"
                  ).pack(side="left", padx=2)

        ttk.Label(left, text="speed [slots/tick]").grid(row=row, column=0,
                                                        sticky="w")
        self.speed = tk.IntVar(value=20)
        tk.Scale(left, from_=1, to=400, orient="horizontal",
                 variable=self.speed, showvalue=True, length=150
                 ).grid(row=row, column=1, sticky="w")
        row += 1

        self.status = tk.Label(left, text="idle", anchor="w",
                               fg=STATUS_COLORS["idle"],
                               font=("TkDefaultFont", 9, "bold"))
        self.status.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1

        # QBER PANEL VISIBILITY -- deliberately far from the "policies" group
        # at the top of this column and under its own heading. That group
        # picks which policies RUN; this one only picks which already-running
        # policy's trace is drawn on panel 2. The two look alike (a stack of
        # checkboxes with policy-shaped labels) and a reader who conflates
        # them would think unticking one here drops a policy from the study.
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Label(left, text="show lines for  (QBER + η panels)",
                  font=("TkDefaultFont", 9, "bold")
                  ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(left, text="(view only -- does not change which policies "
                             "run, see \"policies\" above)",
                  font=("TkDefaultFont", 7), foreground="#666666",
                  wraplength=200, justify="left"
                  ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        # Populated by _rebuild_trace_checks, never here: at this point there
        # is no session yet (_apply() runs after _build()), and the set of
        # policies it will need checkboxes for is not known until then.
        self.trace_check_frame = ttk.Frame(left)
        self.trace_check_frame.grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        self.trace_checks = {}   # display name -> tk.BooleanVar

        # Sieve untuk panel buffer: hanya gambar buffer saat salah satu senjata
        # aktif (jamming atau PNS). Muatan visual panel buffer yang padat biasanya
        # berasal dari rentang "cuaca baik" saat buffer berayun penuh; memotong
        # rentang itu menjadikan periode serangan lebih mudah dibaca. Tanda cek
        # ini murni tampilan -- session menjalankan episode yang sama dan
        # scoreboard tidak boleh berubah.
        self.buffer_sieve_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left, text="sieve buffer: attack slots only",
            variable=self.buffer_sieve_var,
            command=self._on_buffer_sieve
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        # Ẩn TOÀN BỘ panel buffer khỏi figure. Chỉ là view: session không chạy
        # lại, `run.buffer` vẫn ghi mỗi slot và vẫn đi vào công thức
        # eq (2)-(24) lẫn η -- figure chỉ bớt một panel.
        self.buffer_hide_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left, text="hide buffer panel",
            variable=self.buffer_hide_var,
            command=self._on_buffer_hide
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.board = tk.Text(left, height=12, width=52, font=("TkFixedFont", 8),
                             state="disabled", wrap="none")
        self.board.grid(row=row, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(row, weight=1)

        # 8.4 in at 100 dpi asked for 840 px of plot, which put the window
        # at 990 px on a 1080 px screen and buried the terminal behind it.
        # The exported figure (policyviz.py's --figure path) keeps its own,
        # larger size: this number is about fitting a screen, not print.
        self.figure = Figure(figsize=(9.2, 6.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # ONE connection, for the life of the window.
        #
        # The obvious place to bind this is next to the legend it listens
        # for, inside draw_panels -- and that is the trap. draw_panels runs
        # on the redraw timer, so binding there would add a handler per
        # frame: after a minute of running, one click would toggle a trace
        # hundreds of times and appear to do nothing at all (or something,
        # depending on the parity of the frame count). The handler is bound
        # here instead and resolves the clicked artist through
        # `session.legend_picks`, which draw_panels refreshes each frame.
        self._pick_cid = self.canvas.mpl_connect("pick_event", self._on_pick)

    # ------------------------------------------------------------------
    # form <-> config
    # ------------------------------------------------------------------

    @staticmethod
    def _parameters_from(specs, theta="0.045", ba_theta="0.045",
                         ba_reserve="0.25"):
        """(greedy theta, buffer-aware theta, reserve) read out of `specs`.

        Anything the specs do not mention keeps its default, so a partial
        selection does not blank the other boxes.
        """
        for spec in specs:
            policy = _parse_or_none(spec)
            if isinstance(policy, GreedyThreshold):
                theta = "%g" % policy.threshold
            elif isinstance(policy, BufferAwareThreshold):
                ba_theta = "%g" % policy.threshold
                ba_reserve = "%g" % policy.reserve
        return theta, ba_theta, ba_reserve

    @staticmethod
    def _decoy_parameters_from(specs, dc_theta="0.1", dc_qber="1",
                               dc_probe="60"):
        """(excess threshold, QBER alarm, probe interval) out of `specs`.

        The defaults are the operating point measured to beat both static
        choices under the default adversary.
        """
        for spec in specs:
            policy = _parse_or_none(spec)
            if isinstance(policy, DecoyWatch):
                dc_theta = "%g" % policy.decoy_threshold
                dc_qber = "%g" % policy.qber_threshold
                dc_probe = "%g" % policy.probe_every
        return dc_theta, dc_qber, dc_probe

    def _refresh_adversary_hint(self):
        self.adversary_hint.config(
            text=ADVERSARY_HINTS.get(self.adversary_var.get(), ""))

    @staticmethod
    def _stems(specs):
        """Map full specs back to the checkbox stems they came from."""
        out = []
        for spec in specs:
            if spec.startswith("buffer-aware"):
                out.append("buffer-aware")
            elif spec.startswith("decoy-"):
                out.append("decoy")
            elif spec.startswith("block-aware"):
                # "block-aware-80" from the command line still ticks the
                # plain checkbox; the horizon comes from the configuration.
                out.append("block-aware")
            elif spec.startswith("greedy"):
                out.append("greedy")
            else:
                out.append(spec)
        return out

    # ------------------------------------------------------------------
    # operating points
    # ------------------------------------------------------------------

    def _shared_channel(self):
        """The fibre BOTH modes ride, as the MAIN WINDOW currently has it.

        Only d_km is on the form; alpha and odr_losses ride along on the
        config object. A half-typed d_km falls back to the applied value
        rather than raising: this is called to POPULATE a dialog, and refusing
        to open it because a box is mid-edit would be a worse answer than
        showing the link that is actually running.
        """
        try:
            d_km = float(self.entries["d_km"].get().strip())
        except (ValueError, KeyError):
            d_km = float(self.config_obj.d_km)
        return {"d_km": d_km, "alpha": float(self.config_obj.alpha),
                "odr_losses": float(self.config_obj.odr_losses)}

    def _current_env(self, protocol):
        """What the dialog for `protocol` should open showing."""
        env = self.mode_env.get(protocol)
        if env is not None:
            return env
        table = policyviz.preset_table(protocol)
        name = self.preset_var[protocol].get()
        return table.get(name) or policyviz.PARAM_CLASSES[protocol]()

    def _set_operating_point(self, protocol, name, env):
        """Adopt what a dialog returned, and normalise it.

        An env equal to its preset is stored as None. Otherwise every OK --
        including one on a dialog nobody touched -- would freeze a snapshot of
        the preset file into the configuration, and the button would go on
        saying "aurea" while later edits to presets/aurea.json were ignored.
        """
        table = policyviz.preset_table(protocol)
        self.preset_var[protocol].set(name)
        self.mode_env[protocol] = (
            env if policyviz.is_modified(env, table.get(name)) else None)
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self):
        for protocol, btn in self.mode_buttons.items():
            btn.config(text=policyviz.operating_point_label(
                protocol, self.preset_var[protocol].get(),
                self.mode_env.get(protocol),
                policyviz.preset_table(protocol)))

    def _params_dialog_result(self, protocol):
        """Open the editor modally; (preset name, env) on OK, None on Cancel.

        The ONE seam between this window and a blocking dialog. A test drives
        `_edit_params` by replacing this method, so nothing in the suite ever
        has to wait for a click.
        """
        dialog = ParamsDialog(self, protocol,
                              self.preset_var[protocol].get(),
                              self._current_env(protocol),
                              self._shared_channel())
        return dialog.show()

    def _edit_params(self, protocol):
        """The two operating-point buttons.

        Cancel (a None result) changes NOTHING -- not the preset name, not the
        parameters, not the status line.
        """
        result = self._params_dialog_result(protocol)
        if result is None:
            return
        name, env = result
        self._set_operating_point(protocol, name, env)
        # The form has changed but the session has not; every other control
        # here behaves the same way, and silently running the old physics
        # under a button that reads "(modified)" would be the worse failure.
        self._set_status("%s changed — press Apply"
                         % self.mode_buttons[protocol].cget("text"), "paused")

    # ------------------------------------------------------------------

    def _read_config(self):
        specs = []
        for stem in policyviz.SPEC_CHOICES:
            if not self.checks[stem].get():
                continue
            # The spec string comes from the policy's OWN name, not from a
            # second format string here: policies.py renders thresholds at
            # fixed precision, and a private format would eventually disagree
            # with the label the study's tables use.
            if stem == "greedy":
                specs.append(GreedyThreshold(float(self.theta.get())).name)
            elif stem == "buffer-aware":
                specs.append(BufferAwareThreshold(
                    float(self.ba_theta.get()),
                    float(self.ba_reserve.get())).name)
            elif stem == "decoy":
                try:
                    specs.append(DecoyWatch(
                        float(self.dc_theta.get()),
                        qber_threshold=float(self.dc_qber.get()),
                        probe_every=int(float(self.dc_probe.get()))).name)
                except ValueError:
                    raise ValueError("decoy: every box needs a number, got "
                                     "%r / %r / %r"
                                     % (self.dc_theta.get(), self.dc_qber.get(),
                                        self.dc_probe.get()))
            else:
                specs.append(stem)
        if not specs:
            raise ValueError("select at least one policy")
        values = {}
        for name, label, _ in FIELDS:
            if name is None:
                continue
            text = self.entries[name].get().strip()
            try:
                values[name] = (float(text) if name in FLOAT_FIELDS
                                else int(float(text)))
            except ValueError:
                raise ValueError("%s: %r is not a number" % (label, text))
        return policyviz.VizConfig(
            specs=tuple(specs), pm_preset=self.preset_var["pm"].get(),
            ent_preset=self.preset_var["ent"].get(),
            pm_env=self.mode_env.get("pm"), ent_env=self.mode_env.get("ent"),
            adversary=self.adversary_var.get(), **values)

    # ------------------------------------------------------------------
    # the saved configuration
    # ------------------------------------------------------------------

    def _collect_state(self):
        """Snapshot the panel EXACTLY AS IT STANDS, for vizstate.save_state.

        Read from the WIDGETS, not from `self.config_obj`: the config object
        is the last thing Apply built, so saving from it would write the
        previously applied configuration and quietly discard the edit the user
        pressed Save to keep -- which is the whole complaint this exists to
        answer.

        The thresholds are read even when their policy is unticked, so a
        threshold that was typed and then set aside is still there next time.

        Parsed here, so a malformed box raises BEFORE anything is written and
        the saved file can never hold text the form would refuse to apply.
        """
        fields = {}
        for name, label, _width in FIELDS:
            if name is None:
                continue
            text = self.entries[name].get().strip()
            try:
                fields[name] = (float(text) if name in FLOAT_FIELDS
                                else int(float(text)))
            except ValueError:
                raise ValueError("%s: %r is not a number" % (label, text))
        policies = [stem for stem in policyviz.SPEC_CHOICES
                    if self.checks[stem].get()]
        if not policies:
            # Same refusal as Apply: a saved state with nothing ticked would
            # restore a form that cannot be run.
            raise ValueError("select at least one policy")
        state = {"fields": fields, "policies": policies,
                 "adversary": self.adversary_var.get()}
        for key, attr, label in THRESHOLD_FIELDS + DECOY_FIELDS:
            text = getattr(self, attr).get().strip()
            try:
                state[key] = float(text)
            except ValueError:
                raise ValueError("%s: %r is not a number" % (label, text))
        for protocol in ("pm", "ent"):
            state["%s_preset" % protocol] = self.preset_var[protocol].get()
            # THE DATACLASS, not the preset name -- and None when the mode is
            # tracking its preset file unedited. `_set_operating_point` keeps
            # `mode_env` normalised to exactly that, and both halves of the
            # distinction have to survive (see vizstate's docstring).
            state["%s_env" % protocol] = self.mode_env.get(protocol)
        hidden = (set(self.session.hidden_traces)
                  if self.session is not None else set(self._pending_hidden))
        state["hidden_traces"] = sorted(hidden)
        state["speed"] = int(self.speed.get())
        return state

    def _apply_state(self, state):
        """Overlay a saved snapshot onto the form. Absent keys are LEFT ALONE.

        Only the form is touched: nothing here rebuilds the session, so a
        Restore behaves like every other control in this column and takes
        effect on Apply. The exception is `hidden_traces`, which is a view
        choice with no place in `VizConfig` -- there is nothing for Apply to
        do with it, so it is imposed on the live session immediately.
        """
        for name, value in state.get("fields", {}).items():
            if name in self.entries:      # a field this version dropped
                self.entries[name].set(format_field(name, value))
        policies = state.get("policies")
        if policies is not None:
            chosen = set(policies)
            for stem, var in self.checks.items():
                var.set(stem in chosen)
        for key, attr, _label in THRESHOLD_FIELDS + DECOY_FIELDS:
            if key in state:
                getattr(self, attr).set("%g" % state[key])
        if state.get("adversary") in ADVERSARIES:
            self.adversary_var.set(state["adversary"])
        for protocol in ("pm", "ent"):
            name_key, env_key = "%s_preset" % protocol, "%s_env" % protocol
            name = state.get(name_key, self.preset_var[protocol].get())
            if env_key in state:
                # Through _set_operating_point, so a saved env that now
                # matches its preset (the file having been edited meanwhile)
                # normalises back to None exactly as an OK from the dialog
                # would. `is None` stays a reliable "vanilla preset" test.
                if state[env_key] is None:
                    self.preset_var[protocol].set(name)
                    self.mode_env[protocol] = None
                else:
                    self._set_operating_point(protocol, name, state[env_key])
            elif name_key in state:
                self.preset_var[protocol].set(name)
        self._refresh_mode_buttons()
        if "hidden_traces" in state:
            self._pending_hidden = set(state["hidden_traces"])
            if self.session is not None:
                self.session.hidden_traces.clear()
                self.session.hidden_traces.update(self._pending_hidden)
                self._sync_trace_checks()
                self._redraw(force=True)
        if "speed" in state:
            self.speed.set(max(SPEED_MIN, min(SPEED_MAX, int(state["speed"]))))

    def _save_state(self):
        """The Save button: RECORD THE FORM. It does not Apply.

        Apply rebuilds the session, which throws away the episode running on
        screen. Making Save do that as a side effect would mean a user who
        wanted to keep a number lost the run they were watching. The reverse
        coupling is just as bad: an Apply that saved would overwrite the saved
        configuration every time someone nudged a parameter to try something.

        So the two stay separate, and the status line says which one just
        happened -- the button must not silently do the other one.
        """
        try:
            path = vizstate.save_state(self._collect_state())
        except Exception as exc:      # bad box, unwritable file, full disk
            self._set_status("save failed: %s" % exc, "error")
            messagebox.showerror("policy visualiser",
                                 "could not save the configuration:\n%s" % exc)
            return
        self._set_status("saved → %s — press Apply to run it"
                         % os.path.basename(path), "idle")

    def _restore_state(self):
        """The Restore button: put the saved configuration back on the form.

        The saved state is otherwise only reachable by restarting the app,
        which is a poor answer to "I have been experimenting and want my
        settings back". (gui.py solves the same problem by adding a "default"
        entry to its preset dropdown; this window's dropdown lives in the
        per-mode dialog and lists ONE mode's operating points, so a whole-panel
        snapshot has no honest place in it.)
        """
        state = vizstate.load_state()
        if state is None:
            self._set_status("nothing saved yet — press Save first", "error")
            return
        self._apply_state(state)
        self._set_status("restored %s — press Apply to run it"
                         % os.path.basename(vizstate.DEFAULT_PATH), "paused")

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _set_status(self, text, kind):
        self.status.config(text=text, fg=STATUS_COLORS.get(kind, "#000000"))

    def _apply(self):
        """Rebuild the session from the form."""
        self._running = False
        self.start_btn.config(text="Start")
        try:
            config = self._read_config()
        except ValueError as exc:
            self._set_status(str(exc), "error")
            messagebox.showerror("policy visualiser", str(exc))
            return
        self.config_obj = config
        self._build_session(config)

    def _on_pick(self, event):
        """Click a QBER legend entry: hide or show that policy's trace.

        Anything that is not a policy entry -- the two death-threshold
        captions, or a click that lands on some other artist -- resolves to
        None and is ignored. The redraw is forced: throttling it would leave
        the click apparently unanswered for up to REDRAW_MS.
        """
        if self.session is None:
            return
        name = self.session.legend_picks.get(event.artist)
        if name is None:
            return
        self.session.toggle_trace(name)
        # The checkbox group is the OTHER control over this same bit of
        # state (see the group's heading). Without this line a legend click
        # would leave that policy's checkbox showing the visibility it had
        # a moment ago -- ticked while the trace is actually hidden -- and
        # the checkbox would be lying about what is on screen.
        self._sync_trace_checks()
        self._redraw(force=True)

    def _rebuild_trace_checks(self):
        """One checkbox per policy in the CURRENT session, keyed by its
        display name -- the same string draw_panels labels its legend and
        y-ticks with, and the same string `hidden_traces` stores.

        Called every time a session is (re)built, i.e. on every Apply,
        because the policy set can change: a policy just ticked off in the
        "policies" group needs a new checkbox here, and one just ticked on
        needs its checkbox to disappear. Rebuilding from scratch rather than
        diffing the old group against the new is what guarantees a stale
        checkbox for a dropped policy can never linger.

        `hidden_traces` is the only source of truth for visibility -- this
        method reads it once to seed each fresh BooleanVar and never keeps
        a second copy that could drift out of agreement with it.
        """
        for child in self.trace_check_frame.winfo_children():
            child.destroy()
        self.trace_checks = {}
        if self.session is None:
            return
        for run in self.session.runs:
            name = run.display_name
            var = tk.BooleanVar(value=self.session.trace_visible(name))
            ttk.Checkbutton(
                self.trace_check_frame, text=name, variable=var,
                command=lambda n=name: self._on_trace_check(n)
                ).pack(anchor="w")
            self.trace_checks[name] = var

    def _sync_trace_checks(self):
        """Make every checkbox var agree with `hidden_traces`.

        This is a ONE-WAY read: `hidden_traces` is never inferred from a
        checkbox's value, only written to it. That is what keeps the legend
        and the checkbox group from being two copies of the state that
        happen to agree most of the time -- whichever control was used last,
        this is the step that reconciles the other one.
        """
        if self.session is None:
            return
        for name, var in self.trace_checks.items():
            var.set(self.session.trace_visible(name))

    def _on_trace_check(self, name):
        """A checkbox was clicked: flip the shared state and redraw.

        This always TOGGLES `hidden_traces` rather than reading the
        checkbox's own post-click value and writing that in. A Tk
        Checkbutton flips its variable before invoking `command`, so the two
        are equivalent when the checkbox and `hidden_traces` already agree --
        but toggling is also exactly what `_on_pick` does, and having both
        entry points agree on "flip the shared bit" rather than "impose
        whatever this widget now shows" is what keeps them from disagreeing
        about which one wins if they were ever out of step to begin with.
        `_sync_trace_checks` right after is what actually reconciles the
        widget with `hidden_traces`, not this method's belief about the click.
        """
        if self.session is None:
            return
        self.session.toggle_trace(name)
        self._sync_trace_checks()
        self._redraw(force=True)

    def _on_buffer_sieve(self):
        """The "sieve buffer" checkbox was clicked: update the view and redraw.

        This is strictly a view bit: the episode does not change and no
        policy is reselected, so there is no need to rebuild the session.
        The bool lives on VizSession only because draw_panels is handed just
        that object.
        """
        if self.session is None:
            self._redraw(force=True)
            return
        self.session.buffer_sieve_attacks_only = bool(self.buffer_sieve_var.get())
        self._redraw(force=True)

    def _on_buffer_hide(self):
        """The "hide buffer panel" checkbox was clicked: redraw, episode unchanged.

        Like the sieve, strictly a view bit -- the buffer is still accumulated
        and still enters eq (2)-(24); only the figure stops drawing that panel.
        """
        if self.session is None:
            self._redraw(force=True)
            return
        self.session.buffer_hidden = bool(self.buffer_hide_var.get())
        self._redraw(force=True)

    def _build_session(self, config):
        # Which QBER traces are switched off is a VIEW choice, not part of
        # the episode, so it rides across an Apply: otherwise nudging lambda
        # would silently un-hide everything the reader had just cleared away,
        # and Apply is the button they press most.
        #
        # Names of policies that are no longer selected are carried too, and
        # simply never match -- draw_panels only ever asks "is THIS run
        # hidden", so a stale name cannot reach the draw. Keeping them means
        # re-ticking a policy restores the visibility it had rather than
        # springing back on top of the traces the reader was comparing.
        #
        # `_pending_hidden` is the same set arriving from the SAVED
        # configuration, which was read before any session existed. It is
        # consumed here rather than left in place: keeping it would make a
        # trace the user switched back on spring off again on the next Apply.
        carried = set(self._pending_hidden)
        self._pending_hidden = set()
        if self.session is not None:
            carried |= set(self.session.hidden_traces)
        try:
            self.session = policyviz.VizSession(config)
        except Exception as exc:
            self.session = None
            self._rebuild_trace_checks()
            self._set_status("cannot build session: %s" % exc, "error")
            messagebox.showerror("policy visualiser", str(exc))
            return
        self.session.hidden_traces.update(carried)
        if hasattr(self, "buffer_sieve_var"):
            self.session.buffer_sieve_attacks_only = bool(self.buffer_sieve_var.get())
        if hasattr(self, "buffer_hide_var"):
            self.session.buffer_hidden = bool(self.buffer_hide_var.get())
        # Rebuilt AFTER the carried toggles land in hidden_traces, and not
        # before: a policy that survived Apply must show the visibility it
        # already had, which _rebuild_trace_checks can only read correctly
        # once hidden_traces reflects it.
        self._rebuild_trace_checks()
        self._set_status("ready — seed %d, %d slots"
                         % (self.session.seed, config.horizon), "idle")
        self._redraw()

    def _toggle_run(self):
        if self.session is None:
            return
        self._running = not self._running
        self.start_btn.config(text="Pause" if self._running else "Start")
        self._set_status("running" if self._running else "paused",
                         "running" if self._running else "paused")

    def _step_once(self):
        if self.session is None:
            return
        self._running = False
        self.start_btn.config(text="Start")
        self.session.step(1)
        self._redraw()

    def _reset(self):
        if self.session is None:
            return
        self._running = False
        self.start_btn.config(text="Start")
        self.session.reset()
        self._set_status("reset — seed %d" % self.session.seed, "idle")
        self._redraw()

    # ------------------------------------------------------------------
    # the timer
    # ------------------------------------------------------------------

    def _tick(self):
        if self._running and self.session is not None:
            self.session.step(self.speed.get())
            done = self.session.done
            if done:
                self._running = False
                self.start_btn.config(text="Start")
                self._set_status("finished (%d slots)" % self.session.slot,
                                 "finished")
            # The last frame is never throttled away: an episode that ended
            # between two redraws would otherwise show a figure stopping
            # short of its own horizon.
            self._redraw(force=done)
        self.after(TICK_MS, self._tick)

    def _redraw(self, force=True):
        if self.session is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_draw) * 1000.0 < REDRAW_MS:
            return
        self._last_draw = now
        policyviz.draw_panels(self.figure, self.session)
        self.canvas.draw_idle()
        self.board.config(state="normal")
        self.board.delete("1.0", "end")
        self.board.insert("1.0", policyviz.format_scoreboard(self.session))
        self.board.config(state="disabled")

    def _on_close(self):
        self._running = False
        self.destroy()


def main():
    # Reading the saved configuration is the CALLER's job, never the
    # constructor's (see PolicyVizApp's docstring). policyviz._build_app does
    # the same for the `python3 policyviz.py` entry point.
    PolicyVizApp(saved=vizstate.load_state()).mainloop()


if __name__ == "__main__":
    main()
