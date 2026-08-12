#!/usr/bin/env python3
"""Tkinter control panel. Pure UI: every action goes through backend.Backend.

Layout (spec §6): left = protocol radio (PM/ENT) + a static three-region
env-param form (shared channel / PM-only / ENT-only, all always visible) +
presets + Apply/Stop; right = 3 matplotlib axes (QBER, sifted rate, key
buffer), status light, forecast line, log pane. Closing the window stops
every child process.
"""

import os
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import panelstate  # noqa: E402
import plots  # noqa: E402
import runlog  # noqa: E402
import sessionio  # noqa: E402
from channel import EntEnvParams, EnvParams, death_qber, death_qber_ent  # noqa: E402
# Per-mode background tints for the session timeline, plus the grey used for
# stack-down stretches (whose width is the switch cost t_switch). Owned by
# plots.py so the live panel and the exported figures cannot drift apart.
from plots import DOWNTIME_BAND, MODE_BAND, MODE_LINE, band_end  # noqa: E402
from presets import load_ent_presets, load_presets  # noqa: E402
from timeline import Timeline  # noqa: E402

POLL_MS = 2000
# Only draw the death-QBER line once the window's QBER climbs to this fraction
# of it; below that the line would flatten a low-QBER trace, so we hide it and
# let matplotlib autoscale to the data (jitter/Eve variation becomes visible).
DEATH_LINE_SHOW_FRAC = 0.5
BUFFER_FULL = "#d9534f"       # key-buffer bar fill (at B_max)
BUFFER_TRACK = "#e0e0e0"      # key-buffer bar background track
NOTICE_COLOR = "#31708f"      # transient confirmations (not failures)
STATUS_COLORS = {"down": "#9e9e9e", "starting": "#f0ad4e",
                 "streaming": "#5cb85c", "error": "#d9534f"}
# Names sessionio.export_session writes into the sidecar itself. It merges
# extra_meta AFTER setting them, so anything the panel sends under one of these
# would silently overwrite the file's own honesty flags. The export's payload
# (b_max / n_z_block / panel) avoids them, and _export_session checks that it
# still does if that payload ever grows.
SIDECAR_RESERVED_KEYS = ("kind", "exported_at", "exported_at_iso",
                         "secret_bits_are_cumulative",
                         "per_intensity_columns_are_zero", "segments", "gaps",
                         "secret_by_protocol")

# (field, label) per region; names must match the dataclass field names
# (ENT keys carry an "ent_" prefix in self.entries because eta_bob exists
# in both EnvParams and EntEnvParams -- stripped in _read_ent_env()).
FIELDS_SHARED = [
    ("d_km", "d [km]"), ("alpha", "α [dB/km]"),
    ("odr_losses", "optics losses [dB]"), ("f_rep", "f_rep [Hz]"),
    ("qber_sigma", "QBER jitter σ"),
]
FIELDS_PM = [
    ("eta_bob", "η detector"), ("pdc", "p_dc (dark count)"),
    ("e_det", "e_det (optical err)"), ("dead_us", "τ dead [µs]"),
    ("coeff", "dead-time coeff"), ("mu1", "μ₁ (signal)"),
    ("mu2", "μ₂ (decoy)"), ("p1", "P(μ₁)"),
]
FIELDS_ENT = [
    ("ent_eta_a", "η_A (Alice)"), ("ent_eta_bob", "η_B detector"),
    ("ent_xi_a", "ξ_A (bg/pulse)"), ("ent_xi_b", "ξ_B (bg/pulse)"),
    ("ent_e_intr", "e₂ intrinsic"), ("ent_mu", "μ (pairs/pulse)"),
]


class App(tk.Tk):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.title("QKD Control Panel (PM / ENT)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Session-long history: kept across Apply/Stop and mode switches, and
        # only ever reset by exiting the app (spec: 2026-08-11-gui-timeline).
        self.timeline = Timeline()
        self._death_qber = None  # set once per Apply (expensive), see _apply()
        self._n_z_block = 10_000_000  # block size in effect, for the status line
        self._b_max = 10_000_000  # key-buffer capacity [bits], from the B_max input
        # Where "Export session…" writes. Tests point these elsewhere; None
        # means the real logs/ directories (logs/runs and logs/figures).
        self._export_runs_dir = None
        self._export_figs_dir = None
        self._build_left()
        self._build_right()
        # Seed the ENT region too, so its entries are never empty if the user
        # flips the radio without loading an ENT preset first. Order matters:
        # the last _load_preset() also selects the starting mode (PM).
        self._load_preset("chapman_current")
        self._load_preset("aurea")
        # Then overlay the user's saved panel state, if any. Seeding first
        # means a default written before some field existed still leaves that
        # field with a sane preset value rather than blank.
        saved = panelstate.load_default()
        if saved is not None:
            self._apply_panel_state(saved)
            self.preset_box.set(panelstate.RESERVED_NAME)
        self.after(POLL_MS, self._poll)

    # ---------------- left: form ----------------
    def _build_left(self):
        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        # Protocol selector. Layout is STATIC (spec §6): all three param
        # regions stay visible in both modes; the radio only decides which
        # region feeds predict()/apply().
        self.proto_var = tk.StringVar(value="pm")
        proto = ttk.Frame(left)
        proto.grid(row=0, column=0, columnspan=2, sticky="w")
        for val, txt in (("pm", "PM (BB84 decoy)"), ("ent", "ENT (BBM92)")):
            ttk.Radiobutton(proto, text=txt, value=val,
                            variable=self.proto_var,
                            command=self._schedule_forecast).pack(side="left")
        self.entries = {}
        row = 1
        for title, fields in (("channel", FIELDS_SHARED),
                              ("PM", FIELDS_PM), ("ENT", FIELDS_ENT)):
            ttk.Separator(left, orient="horizontal").grid(
                row=row, column=0, columnspan=2, sticky="ew", pady=3)
            ttk.Label(left, text=title, font=("TkDefaultFont", 9, "bold")
                      ).grid(row=row + 1, column=0, columnspan=2, sticky="w")
            row += 2
            for field, label in fields:
                ttk.Label(left, text=label).grid(row=row, column=0, sticky="w")
                e = ttk.Entry(left, width=12)
                e.grid(row=row, column=1, pady=1)
                e.bind("<KeyRelease>", lambda *_: self._schedule_forecast())
                self.entries[field] = e
                row += 1
        # Block size (n_z) is not an EnvParams field -- kept separate from
        # FIELDS_*/entries so it stays out of the FIELDS<->dataclass invariant.
        ttk.Label(left, text="block [bits]").grid(row=row, column=0, sticky="w")
        self.block_entry = ttk.Entry(left, width=12)
        self.block_entry.grid(row=row, column=1, pady=1)
        self.block_entry.insert(0, "10000000")
        self.block_entry.bind("<KeyRelease>", lambda *_: self._schedule_forecast())
        row += 1
        ttk.Label(left, text="preset").grid(row=row, column=0, sticky="w")
        self.preset_box = ttk.Combobox(left, state="readonly", width=11)
        self.preset_box.grid(row=row, column=1, pady=4)
        self.preset_box.bind("<<ComboboxSelected>>",
                             lambda *_: self._load_preset(self.preset_box.get()))
        self._refresh_preset_list()
        self.apply_btn = ttk.Button(left, text="Apply & Restart", command=self._apply)
        self.apply_btn.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(left, text="Stop", command=self._stop).grid(
            row=row + 2, column=0, columnspan=2, sticky="ew")
        # Saves the whole panel (both regions + block + B_max + mode) so the
        # next launch starts where this one left off. Not an Apply: it touches
        # no config and no process.
        ttk.Button(left, text="Save as default", command=self._save_default).grid(
            row=row + 3, column=0, columnspan=2, sticky="ew", pady=2)
        # Exports the whole session (every mode) as a run file and renders it
        # with plots.py. The live panel stays plain on purpose; the pretty
        # figure is produced from the file, by the one drawing implementation.
        ttk.Button(left, text="Export session…",
                   command=self._export_session).grid(
            row=row + 4, column=0, columnspan=2, sticky="ew", pady=2)
        # Eve controls: fraction input + toggle. NOT an EnvParams field; acts
        # live (no stack restart), so it is separate from Apply.
        ttk.Label(left, text="Eve f (0..1)").grid(row=row + 5, column=0, sticky="w")
        self.eve_entry = ttk.Entry(left, width=12)
        self.eve_entry.grid(row=row + 5, column=1, pady=1)
        self.eve_entry.insert(0, "0.1")
        self.eve_entry.bind("<KeyRelease>", lambda *_: self._schedule_forecast())
        self.eve_btn = tk.Button(left, text="Eve: OFF", command=self._toggle_eve)
        self.eve_btn.grid(row=row + 6, column=0, columnspan=2, sticky="ew", pady=2)
        # Capture the platform default colors once so the OFF branch in
        # _poll() can restore them instead of hardcoding a Windows-only
        # name like "SystemButtonFace" (this app runs on Linux/X11 via WSLg).
        self._eve_btn_defaults = (self.eve_btn.cget("bg"), self.eve_btn.cget("fg"))
        self.forecast_var = tk.StringVar(value="forecast: -")
        self.forecast_lbl = ttk.Label(left, textvariable=self.forecast_var,
                                      wraplength=180, foreground="#333")
        self.forecast_lbl.grid(row=row + 7, column=0, columnspan=2,
                               sticky="w", pady=6)
        # Key-buffer config, separated from the system params above by a rule.
        # B_max is the buffer capacity [bits]; NOT an EnvParams field (display
        # only, updates the bar live, no Apply needed).
        ttk.Separator(left, orient="horizontal").grid(
            row=row + 8, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(left, text="B_max [bits]").grid(row=row + 9, column=0, sticky="w")
        self.bmax_entry = ttk.Entry(left, width=12)
        self.bmax_entry.grid(row=row + 9, column=1, pady=1)
        self.bmax_entry.insert(0, "10000000")

    # ---------------- right: plots + status + log ----------------
    def _build_right(self):
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        fig = Figure(figsize=(7, 5), dpi=90)
        # Buffer row is half-height of the two time-series plots (it's just a
        # bar, not a curve) so the filled bar looks slim without top/bottom pad.
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.5], hspace=0.6)
        self.ax_qber = fig.add_subplot(gs[0])
        self.ax_rate = fig.add_subplot(gs[1])
        self.ax_buffer = fig.add_subplot(gs[2])  # key-buffer fill bar, not a time series
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
        self._draw_buffer()      # render an empty buffer bar before the first run
        self.canvas.draw_idle()

    # ---------------- helpers ----------------
    def _refresh_preset_list(self):
        """(Re)fill the preset dropdown. The saved panel state leads the list,
        but only once it exists -- offering it before the first save would just
        produce an error on selection."""
        names = list(load_presets()) + list(load_ent_presets())
        if panelstate.default_exists():
            names.insert(0, panelstate.RESERVED_NAME)
        self.preset_box.config(values=names)

    def _set_entry(self, widget, value):
        widget.delete(0, "end")
        widget.insert(0, value)

    def _load_preset(self, name):
        """Fill the shared region + the preset's own region, and select its mode.

        Only the matching protocol's region is touched, so loading a PM preset
        leaves the ENT entries at their last (seeded) values and vice versa.
        The saved panel state is the exception: it restores the WHOLE form.
        """
        if name == panelstate.RESERVED_NAME:
            state = panelstate.load_default()
            if state is None:
                self.error_var.set("no saved default to load")
                self._refresh_preset_list()   # it vanished; stop offering it
                return
            self._apply_panel_state(state)
            return
        pm, ent = load_presets(), load_ent_presets()
        if name in ent:
            env = ent[name]
            fields = FIELDS_SHARED + FIELDS_ENT
            self.proto_var.set("ent")
        else:
            env = pm[name]
            fields = FIELDS_SHARED + FIELDS_PM
            self.proto_var.set("pm")
        for field, _ in fields:
            attr = field[4:] if field.startswith("ent_") else field
            self._set_entry(self.entries[field], repr(getattr(env, attr)))
        self._schedule_forecast()

    # ---------------- saved panel state ----------------
    def _collect_panel_state(self):
        """Snapshot the whole form as a panelstate dict.

        Values are parsed to float/int here so a malformed entry raises before
        anything is written -- the saved file never holds unusable text.
        """
        fields = {}
        for field, _ in FIELDS_SHARED + FIELDS_PM + FIELDS_ENT:
            fields[field] = float(self.entries[field].get())
        return {"protocol": self.proto_var.get(),
                "block": self._read_n_z_block(),
                "b_max": self._read_b_max(),
                "fields": fields}

    def _apply_panel_state(self, state):
        """Restore a saved snapshot. Keys absent from `state` are left alone,
        so an older file written before a field existed still loads."""
        for field, value in state.get("fields", {}).items():
            if field in self.entries:      # dropped/renamed field: skip it
                self._set_entry(self.entries[field], repr(value))
        if "protocol" in state:
            self.proto_var.set(state["protocol"])
        if "block" in state:
            self._set_entry(self.block_entry, str(state["block"]))
        if "b_max" in state:
            self._set_entry(self.bmax_entry, str(state["b_max"]))
        self._schedule_forecast()

    def _save_default(self):
        try:
            path = panelstate.save_default(self._collect_panel_state())
        except Exception as exc:   # bad entry or unwritable file: say so
            self.error_var.set("Save as default failed: %s" % exc)
            return
        self._refresh_preset_list()   # the "default" entry may be new
        self._flash_notice("saved as default -> %s" % os.path.basename(path))

    def _export_session(self, name=None, confirm_overwrite=True):
        """Write the whole session as a run file and render it with plots.py.

        Returns (csv_path, json_path, png_path), or None when nothing was
        written. `name`/`confirm_overwrite` exist so tests can drive this
        without a modal dialog.
        """
        if not len(self.timeline):
            self.error_var.set("nothing to export yet: no samples recorded")
            return None
        if name is None:
            name = simpledialog.askstring(
                "Export session", "File name:",
                initialvalue=time.strftime("session_%Y%m%d-%H%M%S"),
                parent=self)
            if name is None:
                return None            # cancelled
        runs_dir = self._export_runs_dir or runlog.RUNS_DIR
        figs_dir = self._export_figs_dir or os.path.join(runlog.LOG_DIR,
                                                         "figures")
        try:
            csv_path, json_path = sessionio.session_paths(name, runs_dir)
        except ValueError as exc:
            self.error_var.set("Export failed: %s" % exc)
            return None
        if confirm_overwrite and os.path.exists(csv_path):
            if not messagebox.askyesno(
                    "Export session",
                    "%s already exists. Overwrite?"
                    % os.path.basename(csv_path), parent=self):
                return None
        try:
            # _collect_panel_state() raises on a malformed entry box; that is
            # caught below and reported, so a bad form value costs the export,
            # not the app.
            extra_meta = {"b_max": self._read_b_max(),
                          "n_z_block": self._n_z_block,
                          "panel": self._collect_panel_state()}
            clash = sorted(set(extra_meta) & set(SIDECAR_RESERVED_KEYS))
            if clash:
                raise ValueError("panel metadata would overwrite the sidecar's "
                                 "own key(s): %s" % ", ".join(clash))
            csv_path, json_path = sessionio.export_session(
                self.timeline, name, runs_dir=runs_dir, extra_meta=extra_meta)
            run = runlog.load_run(csv_path)
            # Same overlays the CLI draws by default, so `plots.py <file>` and
            # this button never disagree about the same run: the buffer
            # capacity the panel is holding, and any jamming episode that
            # explains a QBER excursion in the trace.
            png_path = plots.save_run_figures(
                run, figs_dir, b_max=extra_meta["b_max"],
                eve=runlog.load_eve_episodes(run.t_start, run.t_end))[0]
        except Exception as exc:
            self.error_var.set("Export failed: %s" % exc)
            return None
        # Report the RESOLVED filename: sanitize_name may have rewritten what
        # the user typed (".." becomes "dot"), and the notice must name the
        # file that actually exists.
        self._flash_notice("exported %s + %s"
                           % (os.path.basename(csv_path),
                              os.path.basename(png_path)))
        return csv_path, json_path, png_path

    def _flash_notice(self, text, ms=4000):
        """Transient, non-alarming message in the sticky error area."""
        self.error_lbl.config(fg=NOTICE_COLOR)
        self.error_var.set(text)
        if getattr(self, "_notice_job", None):
            self.after_cancel(self._notice_job)

        def clear():
            self._notice_job = None
            # Only clear our own message; an error raised meanwhile wins.
            if self.error_var.get() == text:
                self.error_var.set("")
            self.error_lbl.config(fg=STATUS_COLORS["error"])

        self._notice_job = self.after(ms, clear)

    def _read_env(self):
        kwargs = {}
        for field, _ in FIELDS_SHARED + FIELDS_PM:
            kwargs[field] = float(self.entries[field].get())
        return EnvParams(**kwargs)

    def _read_ent_env(self):
        kwargs = {}
        for field, _ in FIELDS_SHARED:
            kwargs[field] = float(self.entries[field].get())
        for field, _ in FIELDS_ENT:
            kwargs[field[4:]] = float(self.entries[field].get())
        return EntEnvParams(**kwargs)

    def _read_proto_env(self):
        """(protocol, env) for the currently selected mode."""
        proto = self.proto_var.get()
        return proto, (self._read_ent_env() if proto == "ent"
                       else self._read_env())

    def _read_n_z_block(self):
        # int(float(...)) so scientific-notation input like "1e7" also works.
        n = int(float(self.block_entry.get()))
        if n <= 0:
            raise ValueError("block [bits] must be a positive integer")
        return n

    def _read_eve_f(self):
        f = float(self.eve_entry.get())
        if not 0.0 <= f <= 1.0:
            raise ValueError("Eve f must be in [0, 1]")
        return f

    def _read_b_max(self):
        # int(float(...)) so "1e7" also works; falls back to the last good value.
        try:
            n = int(float(self.bmax_entry.get()))
            if n > 0:
                self._b_max = n
        except (ValueError, tk.TclError):
            pass
        return self._b_max

    def _toggle_eve(self):
        try:
            if self.backend.eve.active:
                self.backend.stop_eve()
            else:
                self.backend.start_eve(self._read_eve_f())
        except Exception as exc:
            self.error_var.set("Eve toggle failed: %s" % exc)

    def _schedule_forecast(self):
        # Debounce keystrokes: recompute at most every 300 ms.
        if getattr(self, "_forecast_job", None):
            self.after_cancel(self._forecast_job)
        self._forecast_job = self.after(300, self._forecast)

    def _forecast(self):
        self._forecast_job = None
        try:
            proto, env = self._read_proto_env()
            d = self.backend.predict(env, self._read_n_z_block(),
                                     protocol=proto)
        except Exception as exc:  # bad input or broken bound: show, don't crash
            self.forecast_var.set("forecast: ERROR %s" % exc)
            return
        verdict = ("l~%s bits/block" % format(d.l_predict, ",")
                   if d.l_predict else "NO KEY (over finite-key bound)")
        # e_base = the mode's operating QBER; feeds the Eve preview below.
        if proto == "ent":
            base_txt = ("forecast: η_B=%.3e  R=%.3e  Q₂=%.2f%%  "
                        "rate~%.0f clicks/s  %s"
                        % (d.eta_b, d.r_coinc, 100 * d.q2, d.clicks_per_s,
                           verdict))
            e_base = d.q2
        else:
            base_txt = ("forecast: η=%.3e  E=%.2f%%  rate~%.0f clicks/s  %s"
                        % (d.eta, 100 * d.e_avg, d.clicks_per_s, verdict))
            e_base = d.e_avg
        # Eve preview: E_eff at the entered f, vs the death threshold. Do NOT
        # compute a fresh death_qber() here -- it's ~25 finite-key evaluations
        # (see channel.death_qber docstring) and _forecast() runs on every
        # debounced keystroke. Before the first Apply self._death_qber is
        # still None, so show a neutral verdict instead of running the bisection.
        try:
            from eve import effective_qber
            f = self._read_eve_f()
            e_eff = effective_qber(e_base, f)
            if self._death_qber is not None:
                verdict = "NO KEY" if e_eff >= self._death_qber else "key ok"
            else:
                verdict = "(Apply to compute threshold)"
            eve_txt = "\nEve f=%.2f → E=%.1f%% → %s" % (f, 100 * e_eff, verdict)
        except Exception:
            eve_txt = ""
        self.forecast_var.set(base_txt + eve_txt)
        self.forecast_lbl.config(
            foreground=STATUS_COLORS["error"] if d.l_predict == 0 else "#333")

    def _apply(self):
        self.apply_btn.state(["disabled"])
        self.status_var.set("applying...")
        self.error_var.set("")
        self.update_idletasks()
        try:
            proto, env = self._read_proto_env()
            n_z_block = self._read_n_z_block()
            # Mark the stack going down before the restart: the width of this
            # gap is the real switch cost, so it must start now, not after
            # apply() returns 5-15 s later.
            self.timeline.begin_downtime(time.time(), proto, None)
            self.backend.apply(env, n_z_block, protocol=proto)
            self._n_z_block = n_z_block
            # Expensive (~25 finite-key evaluations) -- computed once here,
            # never from the per-keystroke _forecast().
            if proto == "ent":
                self._death_qber = death_qber_ent(env, self._n_z_block)
            else:
                self._death_qber = death_qber(env, self._n_z_block)
            # The threshold is only known after the (expensive) bisection
            # above, so hand it to the pending segment now.
            self.timeline.set_pending_death_qber(self._death_qber)
        except Exception as exc:
            self._cancel_downtime_if_still_up()
            self.status_var.set("error")
            self.error_var.set("Apply failed: %s" % exc)
            self.status_lbl.config(bg=STATUS_COLORS["error"])
        finally:
            self.apply_btn.state(["!disabled"])

    def _stop(self):
        self.status_var.set("stopping...")
        self.update_idletasks()
        # The stack goes down whether or not stop() reports an error, so open
        # the gap unconditionally, before the call.
        self.timeline.begin_downtime(time.time())
        try:
            self.backend.stop()
        except Exception as exc:
            self._cancel_downtime_if_still_up()
            self.status_var.set("error")
            self.error_var.set("Stop failed: %s" % exc)
            self.status_lbl.config(bg=STATUS_COLORS["error"])

    def _cancel_downtime_if_still_up(self):
        """Withdraw an optimistically recorded downtime after Apply/Stop failed.

        backend.apply() validates params and rewrites configs BEFORE stopping
        the stack, so a bad entry raises while the old stack is still
        streaming. Keeping the gap then invents a switch episode and makes the
        next segment double-count the running collector's cumulative key
        count. A LATE failure (stack already down) does not report streaming,
        so the gap correctly stays.
        """
        try:
            if self.backend.refresh().status == "streaming":
                self.timeline.cancel_downtime()
        except Exception:
            pass          # cannot tell: keep the gap, it is the safer lie

    def _poll(self):
        """Timer tick: refresh once, then queue the next tick."""
        try:
            self._poll_once()
        finally:
            self.after(POLL_MS, self._poll)

    def _poll_once(self):
        """One refresh + repaint. Split out of _poll() so tests can drive a
        single tick without scheduling real Tk timers."""
        try:
            snap = self.backend.refresh()
            # Lead with the protocol the STACK is actually running (from the
            # backend), not the radio: with the static layout the radio can be
            # flipped after an Apply, and without this the panel would show PM
            # forecast text over an ENT trace with nothing to tell them apart.
            self.status_var.set(
                "[%s] %s   QBER %.2f%%   keys %d   block %s/%s   secret %s b (run)"
                % (snap.protocol.upper(), snap.status, 100 * snap.qber,
                   snap.blocks, format(snap.block_progress, ","),
                   format(self._n_z_block, ","), format(snap.secret_bits, ",")))
            self.status_lbl.config(bg=STATUS_COLORS.get(snap.status, "#777"))
            if snap.eve_active:
                self.eve_btn.config(text="Eve: ON  f=%.2f" % snap.eve_f,
                                    bg=STATUS_COLORS["error"], fg="white")
            else:
                self.eve_btn.config(text="Eve: OFF",
                                    bg=self._eve_btn_defaults[0],
                                    fg=self._eve_btn_defaults[1])
            if snap.status == "streaming":
                # Forward the WHOLE row, not just the four plotted values: the
                # session export writes a collector-shaped CSV, and
                # runlog.load_run drops any row missing a column, so anything
                # not forwarded here would export as zeros.
                self.timeline.add_sample(
                    snap.unix_time, 100 * snap.qber, snap.sifted,
                    snap.secret_bits, clicks=snap.clicks,
                    window_s=snap.window_s, errors=snap.errors,
                    total_sifted=snap.total_sifted, blocks=snap.blocks,
                    block_progress=snap.block_progress)
                self._redraw()
            else:
                # Keep the buffer bar rendered (empty/last state) before the
                # first stream and while down, so it never shows a blank axis.
                self._draw_buffer()
                self.canvas.draw_idle()
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

    def _redraw(self):
        tl = self.timeline
        if not len(tl):
            # Nothing sampled yet (e.g. Stop before the first stream): keep
            # the buffer bar rendered instead of touching an empty history.
            self._draw_buffer()
            self.canvas.draw_idle()
            return
        t0 = tl.t0
        xs = [t - t0 for t in tl.t]
        for ax in (self.ax_qber, self.ax_rate):
            ax.clear()
            self._draw_bands(ax, tl, t0, xs[-1])
        # One line PER SEGMENT: a single line over the whole history would
        # draw a straight fake trace across every downtime.
        for seg in tl.segments:
            sl = slice(seg.i0, seg.i1 + 1)
            color = MODE_LINE.get(seg.protocol, MODE_LINE["pm"])
            self.ax_qber.plot(xs[sl], tl.qber[sl], color=color, lw=1.5,
                              marker=".", markersize=3)
            self.ax_rate.plot(xs[sl], tl.sifted[sl], color=color, lw=1.5,
                              marker=".", markersize=3)
            self._draw_segment_threshold(seg, tl, xs)
        self._shade_eve(self.ax_qber, tl, t0, xs)
        self._mark_current(self.ax_qber, xs, tl.qber, "%.2f%%")
        self.ax_qber.set_ylabel("QBER %")
        self._mark_current(self.ax_rate, xs, tl.sifted,
                           lambda v: format(int(v), ","))
        self.ax_rate.set_ylabel("sifted/win")
        self._draw_buffer()
        self.canvas.draw_idle()

    def _draw_bands(self, ax, tl, t0, x_last):
        """Background: one tint per mode stretch, grey while the stack is down.

        Every span is clamped to x >= 0: t0 is the first SAMPLE, but the first
        Apply's downtime starts before any sample exists, so that gap's raw
        x0 is negative and would drag the axis into negative time.
        """
        # A segment ends at its last SAMPLE, but the mode kept running until
        # Apply was pressed; band_end() carries the tint to that moment so no
        # untinted sliver appears before the grey band (shared with plots.py
        # so the panel and the exported figure agree).
        gap_starts = sorted(gap.t_start for gap in tl.gaps)
        for seg in tl.segments:
            ax.axvspan(max(seg.t_start - t0, 0.0),
                       max(band_end(seg.t_end, gap_starts) - t0, 0.0),
                       color=MODE_BAND.get(seg.protocol, MODE_BAND["pm"]),
                       alpha=0.55, lw=0)
        for gap in tl.gaps:
            x0 = max(gap.t_start - t0, 0.0)
            # t_end == 0.0 means "still down": run the band out to the latest
            # sample rather than back to the origin.
            x1 = (gap.t_end - t0) if gap.t_end > 0 else x_last
            if x1 <= x0:
                continue          # entirely before the first sample, or empty
            ax.axvspan(x0, x1, color=DOWNTIME_BAND, alpha=0.9, lw=0)
            if ax is self.ax_qber and gap.t_end > 0:
                ax.annotate("%s %.0fs" % ("switch" if gap.is_switch else "down",
                                          gap.t_end - gap.t_start),
                            xy=((x0 + x1) / 2, 1.0),
                            xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=6.5,
                            color="#666666")

    def _draw_segment_threshold(self, seg, tl, xs):
        """Death-QBER line for one segment only, so the step at a mode change
        is visible (PM ~5.2% vs ENT ~9.9%)."""
        if seg.death_qber is None:
            return
        death_pct = 100 * seg.death_qber
        window = tl.qber[seg.i0:seg.i1 + 1]
        # Same "don't flatten a quiet trace" rule as before, judged per
        # segment rather than over the whole session.
        if not window or max(window) < death_pct * DEATH_LINE_SHOW_FRAC:
            return
        self.ax_qber.hlines(death_pct, xs[seg.i0], xs[seg.i1],
                            color="red", linestyle="--", linewidth=1)

    def _shade_eve(self, ax, tl, t0, xs):
        """Red bands for Eve's jamming episodes (unchanged behaviour).

        Episodes are in unix time, same as tl.t; an open episode (t_end == 0)
        extends to the latest sample.
        """
        try:
            now = tl.t[-1]
            for ep in self.backend.eve.episodes():
                end = ep.t_end if ep.t_end > 0 else now
                x0, x1 = ep.t_start - t0, end - t0
                if x1 >= xs[0] and x0 <= xs[-1]:
                    ax.axvspan(max(x0, xs[0]), min(x1, xs[-1]),
                               color="red", alpha=0.12)
        except Exception:
            pass

    def _mark_current(self, ax, xs, ys, fmt):
        """Mark the latest value: a faint horizontal dashed line at its level
        (traceable back to the Oy axis) plus the value labelled at the point."""
        if not xs or not ys:
            return
        y = ys[-1]
        text = fmt(y) if callable(fmt) else fmt % y
        ax.axhline(y, color="#888", linestyle="--", linewidth=0.8)
        ax.annotate(text, xy=(xs[-1], y), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=8, color="#c00", clip_on=False)

    def _draw_buffer(self):
        """Key buffer as a horizontal fill bar, split by which protocol
        distilled each part; capped and recoloured at B_max."""
        b_max = self._read_b_max()
        total = self.timeline.total_secret
        by_proto = self.timeline.secret_by_protocol
        full = total >= b_max
        ax = self.ax_buffer
        ax.clear()
        ax.barh(0, b_max, height=1.0, color=BUFFER_TRACK)                # track
        left = 0
        for proto in ("pm", "ent"):
            width = min(by_proto.get(proto, 0), max(b_max - left, 0))
            if width <= 0:
                continue
            ax.barh(0, width, left=left, height=1.0,
                    color=BUFFER_FULL if full else MODE_LINE[proto],
                    label=proto.upper())
            left += width
        if 0 < left < b_max:
            ax.axvline(left, color="#333", linestyle="--", linewidth=0.8)
        ax.set_xlim(0, b_max)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("key buffer [bits]")
        label = "%s / %s%s" % (format(total, ","), format(b_max, ","),
                               "  FULL" if full else "")
        ax.text(0.99, 1.08, label, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9,
                color=BUFFER_FULL if full else "#333")
        if by_proto.get("pm") and by_proto.get("ent"):
            ax.text(0.01, 1.08, "PM %s  •  ENT %s"
                    % (format(by_proto["pm"], ","), format(by_proto["ent"], ",")),
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7.5, color="#555555")

    def _on_close(self):
        self.backend.stop()
        self.destroy()
