#!/usr/bin/env python3
"""Tkinter control panel. Pure UI: every action goes through backend.Backend.

Layout (spec §6): left = env-param form + presets + Apply/Stop; right =
3 matplotlib axes (QBER, sifted rate, secret bits), status light, forecast
line, log pane. Closing the window stops every child process.
"""

import tkinter as tk
from collections import deque
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from channel import EnvParams, death_qber  # noqa: E402
from presets import load_presets  # noqa: E402

POLL_MS = 2000
HISTORY = 300  # plot points kept (~10 min at 2 s)
STATUS_COLORS = {"down": "#9e9e9e", "starting": "#f0ad4e",
                 "streaming": "#5cb85c", "error": "#d9534f"}

# (field, label) in display order; must match EnvParams field names.
FIELDS = [
    ("d_km", "distance d [km]"), ("alpha", "alpha [dB/km]"),
    ("odr_losses", "optics losses [dB]"), ("eta_bob", "detector eff."),
    ("pdc", "dark count pdc"), ("e_det", "optical error e_det"),
    ("dead_us", "dead time [us]"), ("coeff", "dead-time coeff"),
    ("f_rep", "rep. rate [Hz]"), ("mu1", "mu1 (signal)"),
    ("mu2", "mu2 (decoy)"), ("p1", "P(signal)"),
]


class App(tk.Tk):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.title("QKD Control Panel (PM / BB84 decoy)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.hist_t = deque(maxlen=HISTORY)
        self.hist_qber = deque(maxlen=HISTORY)
        self.hist_rate = deque(maxlen=HISTORY)
        self.hist_secret = deque(maxlen=HISTORY)
        self._death_qber = None  # set once per Apply (expensive), see _apply()
        self._n_z_block = 10_000_000  # block size in effect, for the status line
        self._build_left()
        self._build_right()
        self._load_preset("aurea")
        self.after(POLL_MS, self._poll)

    # ---------------- left: form ----------------
    def _build_left(self):
        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        self.entries = {}
        for i, (field, label) in enumerate(FIELDS):
            ttk.Label(left, text=label).grid(row=i, column=0, sticky="w")
            e = ttk.Entry(left, width=12)
            e.grid(row=i, column=1, pady=1)
            e.bind("<KeyRelease>", lambda *_: self._schedule_forecast())
            self.entries[field] = e
        row = len(FIELDS)
        ttk.Label(left, text="preset").grid(row=row, column=0, sticky="w")
        self.preset_box = ttk.Combobox(left, values=list(load_presets()),
                                       state="readonly", width=11)
        self.preset_box.grid(row=row, column=1, pady=4)
        self.preset_box.bind("<<ComboboxSelected>>",
                             lambda *_: self._load_preset(self.preset_box.get()))
        self.apply_btn = ttk.Button(left, text="Apply & Restart", command=self._apply)
        self.apply_btn.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(left, text="Stop", command=self._stop).grid(
            row=row + 2, column=0, columnspan=2, sticky="ew")
        self.forecast_var = tk.StringVar(value="forecast: -")
        self.forecast_lbl = ttk.Label(left, textvariable=self.forecast_var,
                                      wraplength=180, foreground="#333")
        self.forecast_lbl.grid(row=row + 3, column=0, columnspan=2,
                               sticky="w", pady=6)

    # ---------------- right: plots + status + log ----------------
    def _build_right(self):
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        fig = Figure(figsize=(7, 5), dpi=90)
        self.ax_qber = fig.add_subplot(311)
        self.ax_rate = fig.add_subplot(312)
        self.ax_secret = fig.add_subplot(313)
        fig.subplots_adjust(hspace=0.55)
        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="down")
        self.status_lbl = tk.Label(right, textvariable=self.status_var,
                                   fg="white", bg=STATUS_COLORS["down"],
                                   font=("TkDefaultFont", 11, "bold"))
        self.status_lbl.pack(fill="x", pady=3)
        # Sticky error area: set by _apply()/_stop() on exception and by
        # _poll()'s except branch; _poll() never touches it on the normal
        # path, so an Apply failure stays visible instead of being
        # overwritten by the next 2 s poll tick.
        self.error_var = tk.StringVar(value="")
        self.error_lbl = tk.Label(right, textvariable=self.error_var,
                                  fg=STATUS_COLORS["error"], anchor="w",
                                  wraplength=500, justify="left")
        self.error_lbl.pack(fill="x")
        self.log = tk.Text(right, height=8, state="disabled",
                           font=("TkFixedFont", 8))
        self.log.pack(fill="x")

    # ---------------- helpers ----------------
    def _load_preset(self, name):
        env = load_presets()[name]
        for field, _ in FIELDS:
            self.entries[field].delete(0, "end")
            self.entries[field].insert(0, repr(getattr(env, field)))
        self._schedule_forecast()

    def _read_env(self):
        kwargs = {}
        for field, _ in FIELDS:
            kwargs[field] = float(self.entries[field].get())
        return EnvParams(**kwargs)

    def _schedule_forecast(self):
        # Debounce keystrokes: recompute at most every 300 ms.
        if getattr(self, "_forecast_job", None):
            self.after_cancel(self._forecast_job)
        self._forecast_job = self.after(300, self._forecast)

    def _forecast(self):
        self._forecast_job = None
        try:
            d = self.backend.predict(self._read_env())
        except Exception as exc:  # bad input or broken bound: show, don't crash
            self.forecast_var.set("forecast: ERROR %s" % exc)
            return
        verdict = ("l~%s bits/block" % format(d.l_predict, ",")
                   if d.l_predict else "NO KEY (over decoy bound)")
        self.forecast_var.set(
            "forecast: eta=%.3e  E=%.2f%%  rate~%.0f clicks/s  %s"
            % (d.eta, 100 * d.e_avg, d.clicks_per_s, verdict))
        # Spec §8: warn red when the forecast says the block will yield no key.
        self.forecast_lbl.config(
            foreground=STATUS_COLORS["error"] if d.l_predict == 0 else "#333")

    def _apply(self):
        self.apply_btn.state(["disabled"])
        self.status_var.set("applying...")
        self.error_var.set("")
        self.update_idletasks()
        try:
            env = self._read_env()
            self.backend.apply(env, self._n_z_block)
            # Expensive (~25 finite-key evaluations) -- computed once here,
            # never from the per-keystroke _forecast().
            self._death_qber = death_qber(env, self._n_z_block)
            for h in (self.hist_t, self.hist_qber, self.hist_rate,
                      self.hist_secret):
                h.clear()
        except Exception as exc:
            self.status_var.set("error")
            self.error_var.set("Apply failed: %s" % exc)
            self.status_lbl.config(bg=STATUS_COLORS["error"])
        finally:
            self.apply_btn.state(["!disabled"])

    def _stop(self):
        self.status_var.set("stopping...")
        self.update_idletasks()
        try:
            self.backend.stop()
        except Exception as exc:
            self.status_var.set("error")
            self.error_var.set("Stop failed: %s" % exc)
            self.status_lbl.config(bg=STATUS_COLORS["error"])

    def _poll(self):
        try:
            snap = self.backend.refresh()
            self.status_var.set(
                "%s   QBER %.2f%%   keys %d   block %s/%s   secret %s b"
                % (snap.status, 100 * snap.qber, snap.blocks,
                   format(snap.block_progress, ","), format(self._n_z_block, ","),
                   format(snap.secret_bits, ",")))
            self.status_lbl.config(bg=STATUS_COLORS.get(snap.status, "#777"))
            if snap.status == "streaming":
                self.hist_t.append(snap.unix_time)
                self.hist_qber.append(100 * snap.qber)
                self.hist_rate.append(snap.sifted)
                self.hist_secret.append(snap.secret_bits)
                self._redraw()
            self.log.config(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("end", "\n".join(snap.log_lines))
            self.log.see("end")
            self.log.config(state="disabled")
        except Exception as exc:
            self.status_var.set("error")
            self.error_var.set("Poll failed: %s" % exc)
            self.status_lbl.config(bg=STATUS_COLORS["error"])
            self.log.config(state="normal")
            self.log.delete("1.0", "end")
            self.log.insert("end", "Poll error: %s" % exc)
            self.log.config(state="disabled")
        finally:
            self.after(POLL_MS, self._poll)

    def _redraw(self):
        t0 = self.hist_t[0]
        xs = [t - t0 for t in self.hist_t]
        self.ax_qber.clear()
        self.ax_qber.plot(xs, list(self.hist_qber))
        if self._death_qber is not None:
            self.ax_qber.axhline(100 * self._death_qber, color="red",
                                 linestyle="--", linewidth=1)
        self.ax_qber.set_ylabel("QBER %")
        self.ax_rate.clear()
        self.ax_rate.plot(xs, list(self.hist_rate))
        self.ax_rate.set_ylabel("sifted/win")
        self.ax_secret.clear()
        self.ax_secret.plot(xs, list(self.hist_secret))
        self.ax_secret.set_ylabel("secret bits")
        self.ax_secret.set_xlabel("t [s]")
        self.canvas.draw_idle()

    def _on_close(self):
        self.backend.stop()
        self.destroy()
