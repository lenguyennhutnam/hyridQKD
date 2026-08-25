#!/usr/bin/env python3
"""The window for policyviz.py. Thin on purpose.

Everything worth testing lives in `policyviz` and never imports Tk; this file
reads the form, rebuilds a `policyviz.VizSession`, drives a timer and calls
`policyviz.draw_panels`. If a behaviour is being written here that a test
would want to make an assertion about, it belongs in `policyviz` instead.

Two states this window owes the viewer an explanation for:

* **training** -- a DQN track cannot be shown before it is trained, and even a
  short run takes seconds. Training therefore happens on a worker thread with
  the episode counter on screen, rather than blocking the mainloop with a
  frozen window. (`gui.py` blocks deliberately during Apply; that is a
  different trade-off, made because there the user is waiting on real
  hardware and there is nothing to animate.)
* **bound** -- the Oracle row is dashed and labelled, in the figure and in the
  scoreboard both. See policyviz's module docstring.
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import policyviz  # noqa: E402
from policies import BufferAwareThreshold, GreedyThreshold  # noqa: E402
from presets import load_ent_presets, load_presets  # noqa: E402

TICK_MS = 80            # timer period; `speed` sets slots advanced per tick
TRAIN_POLL_MS = 200
# Redrawing is throttled SEPARATELY from stepping. A full 3,600-slot figure
# with six tracks costs ~0.33 s to render, so redrawing on every 80 ms tick
# would leave the mainloop doing nothing but paint -- the buttons would go
# unresponsive and the episode would crawl. Stepping stays on TICK_MS (that
# is what the speed slider controls); the picture refreshes at most this
# often. Step and Reset force a redraw regardless, because there the whole
# point is to see the single slot that just happened.
REDRAW_MS = 400
STATUS_COLORS = {"idle": "#9e9e9e", "training": "#f0ad4e",
                 "running": "#5cb85c", "paused": "#31708f",
                 "finished": "#2E75B6", "error": "#d9534f"}

# (field, label, width) for the plain numeric entries.
FIELDS = [("lam", "λ arrivals/slot", 8), ("b_max", "B_max [bits]", 10),
          ("horizon", "horizon [slots]", 8), ("n_x", "n_X block [bits]", 10),
          ("d_km", "d [km]", 8), ("seed_index", "seed (episode #)", 8),
          ("dqn_episodes", "DQN episodes", 8)]


class PolicyVizApp(tk.Tk):
    def __init__(self, config=None):
        super().__init__()
        self.title("Hybrid policy visualiser — NUMERIC MODEL (not the stack)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.config_obj = config or policyviz.VizConfig()
        self.session = None
        self._running = False
        self._trainer = None
        self._last_draw = 0.0
        self._train_state = {"episode": 0, "total": 0, "policy": None,
                             "error": None, "cached": False}
        self._build()
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

        left = ttk.Frame(self, padding=8)
        left.grid(row=1, column=0, sticky="ns")
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
        for label, var in (("greedy θ", self.theta),
                           ("buffer-aware θ", self.ba_theta),
                           ("buffer-aware reserve", self.ba_reserve)):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(left, textvariable=var, width=8).grid(row=row, column=1,
                                                            sticky="w")
            row += 1

        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        ttk.Label(left, text="ENT preset").grid(row=row, column=0, sticky="w")
        self.ent_preset = tk.StringVar(value=self.config_obj.ent_preset)
        ent_box = ttk.Combobox(left, textvariable=self.ent_preset, width=16,
                               state="readonly",
                               values=sorted(load_ent_presets()))
        ent_box.grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Label(left, text="PM preset").grid(row=row, column=0, sticky="w")
        self.pm_preset = tk.StringVar(value=self.config_obj.pm_preset)
        ttk.Combobox(left, textvariable=self.pm_preset, width=16,
                     state="readonly", values=sorted(load_presets())
                     ).grid(row=row, column=1, sticky="w")
        row += 1

        self.entries = {}
        for name, label, width in FIELDS:
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(getattr(self.config_obj, name)))
            ttk.Entry(left, textvariable=var, width=width).grid(
                row=row, column=1, sticky="w")
            self.entries[name] = var
            row += 1

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

        self.board = tk.Text(left, height=12, width=52, font=("TkFixedFont", 8),
                             state="disabled", wrap="none")
        self.board.grid(row=row, column=0, columnspan=2, sticky="nsew")
        left.rowconfigure(row, weight=1)

        self.figure = Figure(figsize=(9.2, 8.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

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
            policy = None if spec == "dqn" else policyviz.parse_spec(spec)
            if isinstance(policy, GreedyThreshold):
                theta = "%g" % policy.threshold
            elif isinstance(policy, BufferAwareThreshold):
                ba_theta = "%g" % policy.threshold
                ba_reserve = "%g" % policy.reserve
        return theta, ba_theta, ba_reserve

    @staticmethod
    def _stems(specs):
        """Map full specs back to the checkbox stems they came from."""
        out = []
        for spec in specs:
            if spec.startswith("buffer-aware"):
                out.append("buffer-aware")
            elif spec.startswith("greedy"):
                out.append("greedy")
            else:
                out.append(spec)
        return out

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
            else:
                specs.append(stem)
        if not specs:
            raise ValueError("select at least one policy")
        values = {}
        for name, label, _ in FIELDS:
            text = self.entries[name].get().strip()
            try:
                values[name] = (float(text) if name in ("lam", "d_km")
                                else int(float(text)))
            except ValueError:
                raise ValueError("%s: %r is not a number" % (label, text))
        return policyviz.VizConfig(
            specs=tuple(specs), pm_preset=self.pm_preset.get(),
            ent_preset=self.ent_preset.get(), **values)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _set_status(self, text, kind):
        self.status.config(text=text, fg=STATUS_COLORS.get(kind, "#000000"))

    def _apply(self):
        """Rebuild the session from the form. Trains first if a DQN is on."""
        self._running = False
        self.start_btn.config(text="Start")
        try:
            config = self._read_config()
        except ValueError as exc:
            self._set_status(str(exc), "error")
            messagebox.showerror("policy visualiser", str(exc))
            return
        self.config_obj = config
        if "dqn" in config.specs:
            self._start_training(config)
            return
        self._build_session(config, None)

    def _start_training(self, config):
        """Train on a worker thread; the mainloop keeps redrawing meanwhile.

        A frozen window with no explanation is the failure mode this avoids:
        even 20 episodes is several seconds, and the user has no way to tell
        that from a hang.
        """
        state = {"episode": 0, "total": int(config.dqn_episodes),
                 "policy": None, "error": None, "cached": False}
        self._train_state = state

        def progress(episode, eta, eps):
            state["episode"] = episode + 1

        def work():
            try:
                policy, cached = policyviz.train_dqn(config, progress=progress)
                state["policy"] = policy
                state["cached"] = cached
            except Exception as exc:          # surfaced, never swallowed
                state["error"] = exc

        self._trainer = threading.Thread(target=work, daemon=True)
        self._trainer.start()
        self._set_status("training… (0 / %d episodes)" % state["total"],
                         "training")
        self.after(TRAIN_POLL_MS, self._poll_training, config)

    def _poll_training(self, config):
        state = self._train_state
        if self._trainer is not None and self._trainer.is_alive():
            self._set_status("training… (%d / %d episodes)"
                             % (state["episode"], state["total"]), "training")
            self.after(TRAIN_POLL_MS, self._poll_training, config)
            return
        self._trainer = None
        if state["error"] is not None:
            self._set_status("training failed: %s" % state["error"], "error")
            messagebox.showerror("policy visualiser",
                                 "DQN training failed:\n%s" % state["error"])
            return
        self._build_session(config, state["policy"])

    def _build_session(self, config, dqn_policy):
        try:
            self.session = policyviz.VizSession(
                config, dqn_policy=dqn_policy,
                dqn_episodes=config.dqn_episodes)
        except Exception as exc:
            self.session = None
            self._set_status("cannot build session: %s" % exc, "error")
            messagebox.showerror("policy visualiser", str(exc))
            return
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
    PolicyVizApp().mainloop()


if __name__ == "__main__":
    main()
