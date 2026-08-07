#!/usr/bin/env python3
"""Glue between the GUI and everything else. The GUI never touches subprocess,
files or sockets directly — it calls predict/apply/stop/refresh here.
(Same split as VeriQloud's local/gui/backend.py.)"""

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import channel
import configio
import stackctl

STALE_S = 10.0  # newest CSV row older than this while running -> error


@dataclass
class Snapshot:
    """One GUI refresh tick worth of state."""

    status: str = "down"        # down | starting | streaming | error
    unix_time: float = 0.0
    qber: float = float("nan")
    clicks: int = 0             # clicks in the last window
    sifted: int = 0             # sifted bits in the last window
    total_sifted: int = 0
    blocks: int = 0
    block_progress: int = 0
    secret_bits: int = 0
    log_lines: list = field(default_factory=list)


class Backend:
    """predict() is pure; apply() has side effects (configs + processes)."""

    def __init__(self, csv_path=None):
        self.csv_path = csv_path or os.path.join(stackctl.LOG_DIR, "live.csv")
        self._started = False

    def predict(self, env, n_z_block=10_000_000):
        """Forecast eta/qberr/key length without touching anything."""
        return channel.derive(env, n_z_block)

    def apply(self, env):
        """Write configs from env, restart stack + collector. Returns forecast."""
        derived = self.predict(env)
        configio.update_hw_sim_configs(derived.eta, derived.e_avg,
                                       env.mu1, env.mu2, env.p1)
        stackctl.stop_all()
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)  # fresh run, fresh CSV
        stackctl.start_stack()
        stackctl.start_collector(self.csv_path, [
            "--mu1", str(env.mu1), "--mu2", str(env.mu2),
            "--p1", str(env.p1), "--eta", str(derived.eta),
        ])
        self._started = True
        return derived

    def stop(self):
        """Idempotent full stop (used by Stop button, atexit, window close)."""
        stackctl.stop_all()
        self._started = False

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
        if row is None:
            snap.status = "starting" if running else "down"
            return snap
        snap.unix_time = float(row["unix_time"])
        snap.qber = float(row["qber"])
        snap.clicks = int(row["clicks"])
        snap.sifted = int(row["sifted"])
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
