#!/usr/bin/env python3
"""Glue between the GUI and everything else. The GUI never touches subprocess,
files or sockets directly — it calls predict/apply/stop/refresh here.
(Same split as VeriQloud's local/gui/backend.py.)"""

import csv
import dataclasses
import os
import time
from dataclasses import dataclass, field

import channel
import configio
import evectl
import runlog
import stackctl

STALE_S = 10.0  # newest CSV row older than this while running -> error


@dataclass
class Snapshot:
    """One GUI refresh tick worth of state."""

    status: str = "down"        # down | starting | streaming | error
    unix_time: float = 0.0
    window_s: float = 0.0       # length of the last window, in seconds
    qber: float = float("nan")
    clicks: int = 0             # clicks in the last window
    sifted: int = 0             # sifted bits in the last window
    errors: int = 0             # sifted bits that disagreed, in the last window
    total_sifted: int = 0
    blocks: int = 0
    block_progress: int = 0
    secret_bits: int = 0
    log_lines: list = field(default_factory=list)
    eve_active: bool = False
    eve_f: float = 0.0
    protocol: str = "pm"


class Backend:
    """predict() is pure; apply() has side effects (configs + processes)."""

    def __init__(self, csv_path=None):
        self.csv_path = csv_path or os.path.join(stackctl.LOG_DIR, "live.csv")
        self._started = False
        self._protocol = "pm"
        self.eve = evectl.EveController()

    def predict(self, env, n_z_block=10_000_000, protocol="pm"):
        """Forecast hw_sim knobs/key length without touching anything.

        env must be channel.EnvParams for pm, channel.EntEnvParams for ent.
        """
        if protocol == "ent":
            return channel.derive_ent(env, n_z_block)
        return channel.derive(env, n_z_block)

    def apply(self, env, n_z_block=10_000_000, protocol="pm"):
        """Write configs from env, restart stack + collector. Returns forecast.

        n_z_block is threaded into both the forecast (predict) and the
        collector's --key-block-bits so the two provably agree on block size
        instead of relying on matching defaults in two separate modules.

        protocol selects pm (decoy BB84, default) or ent (BBM92): pm writes
        the decoy configs and starts the decoy stack; ent writes the plain
        (non-decoy) configs, starts the plain stack, and tells the collector
        to parse the stream as ENT via --protocol ent.
        """
        self.eve.stop()  # stack restart changes simulator PIDs; drop any jam first
        derived = self.predict(env, n_z_block, protocol)
        if protocol == "ent":
            configio.update_hw_sim_plain_configs(derived.r_coinc, derived.q2,
                                                 derived.pulse_distance,
                                                 qber_sigma=env.qber_sigma)
            mode = "plain"
            collector_args = ["--protocol", "ent",
                              "--key-block-bits", str(n_z_block)]
        else:
            configio.update_hw_sim_configs(derived.eta, derived.e_avg,
                                           env.mu1, env.mu2, env.p1,
                                           derived.pulse_distance,
                                           qber_sigma=env.qber_sigma)
            mode = "decoy"
            collector_args = ["--mu1", str(env.mu1), "--mu2", str(env.mu2),
                              "--p1", str(env.p1), "--eta", str(derived.eta),
                              "--key-block-bits", str(n_z_block)]
        stackctl.stop_all()
        # Park the run that just ended before wiping it -- the order matters,
        # archiving after the delete would silently destroy it. The run is
        # labelled from its own live sidecar (written at the bottom of this
        # method when it started), not from the parameters being applied now.
        runlog.archive_run(self.csv_path)
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)  # fresh run, fresh CSV
        # Truncate the log tail too, so the log pane doesn't show the
        # previous run's output after a restart.
        os.makedirs(stackctl.LOG_DIR, exist_ok=True)
        open(os.path.join(stackctl.LOG_DIR, "collector.log"), "w").close()
        stackctl.start_stack(mode)
        stackctl.start_collector(self.csv_path, collector_args)
        # Describe the run now starting, so that whoever archives it later
        # (the next apply, possibly in a different session) can label it.
        runlog.write_live_meta(self.csv_path, {
            "protocol": protocol,
            "n_z_block": n_z_block,
            "env": dataclasses.asdict(env),
            "forecast": dataclasses.asdict(derived),
        })
        self._started = True
        self._protocol = protocol
        return derived

    def stop(self):
        """Idempotent full stop (used by Stop button, atexit, window close)."""
        self.eve.stop()
        stackctl.stop_all()
        self._started = False

    def start_eve(self, f):
        """Begin fractional jamming at fraction f (0..1)."""
        self.eve.start(f)

    def stop_eve(self):
        """Stop jamming; leaves the simulator unjammed."""
        self.eve.stop()

    def _last_row(self):
        try:
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            return rows[-1] if rows else None
        except OSError:
            return None

    def _log_tail(self, n=15):
        path = os.path.join(stackctl.LOG_DIR, "collector.log")
        try:
            with open(path, errors="replace") as f:
                return [l.rstrip("\n") for l in f.readlines()[-n:]]
        except OSError:
            return []

    def refresh(self):
        """Build a Snapshot from collector CSV + process liveness."""
        row = self._last_row()
        running = stackctl.collector_running()
        snap = Snapshot(log_lines=self._log_tail())
        snap.eve_active = self.eve.active
        snap.eve_f = self.eve.current_f
        snap.protocol = self._protocol
        if row is None:
            snap.status = "starting" if running else "down"
            return snap
        snap.unix_time = float(row["unix_time"])
        # window_s/errors are not shown in the panel, but the GUI forwards the
        # whole row into the session Timeline so an export can be written as a
        # complete collector-shaped CSV.
        snap.window_s = float(row["window_s"])
        snap.qber = float(row["qber"])
        snap.clicks = int(row["clicks"])
        snap.sifted = int(row["sifted"])
        snap.errors = int(row["errors"])
        snap.total_sifted = int(row["total_sifted"])
        snap.blocks = int(row["blocks"])
        snap.block_progress = int(row["block_progress"])
        snap.secret_bits = int(row["secret_bits"])
        fresh = (time.time() - snap.unix_time) < STALE_S
        if running and fresh:
            snap.status = "streaming"
        elif running:
            snap.status = "error"      # collector alive but stream stalled
        elif self._started:
            snap.status = "error"      # we started it and it died on its own
        else:
            snap.status = "down"       # deliberately stopped (or never started
                                        # by this instance, e.g. stale live.csv)
        return snap
