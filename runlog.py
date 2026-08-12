#!/usr/bin/env python3
"""Read and archive collector run logs.

The collector appends one CSV row per window to logs/live.csv, and
backend.apply() wipes that file on every restart. Without archiving, only the
most recent run would ever be plottable -- so apply() hands the finished run
to archive_run() first, which parks it in logs/runs/ next to a JSON sidecar
recording which parameters produced it.

Eve episodes live in a separate, ACCUMULATING file (logs/eve.csv is never
wiped), so load_eve_episodes() filters it down to the window of one run.

I/O only: no matplotlib here, so plots.py stays a pure rendering layer and
this module can be unit-tested without a display.

ERROR POLICY: logs are machine-written and read by a plotting command, so a
damaged row is skipped rather than aborting the figure (the same reasoning as
panelstate.py, the opposite of presets.py where hand-written typos must fail
loudly).
"""

import csv
import glob
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, "logs")
LIVE_CSV = os.path.join(LOG_DIR, "live.csv")
RUNS_DIR = os.path.join(LOG_DIR, "runs")
EVE_CSV = os.path.join(LOG_DIR, "eve.csv")

# Columns written by collector.py, in file order.
_COLUMNS = ("unix_time", "window_s", "clicks", "sifted", "errors", "qber",
            "n_sig", "m_sig", "n_dec", "m_dec", "total_sifted", "blocks",
            "block_progress", "secret_bits")

_INT_COLUMNS = ("clicks", "sifted", "errors", "n_sig", "m_sig", "n_dec",
                "m_dec", "total_sifted", "blocks", "block_progress",
                "secret_bits")


@dataclass
class EveEpisode:
    """One interval during which Eve was jamming at fraction f."""

    t_start: float
    t_end: float
    f: float


@dataclass
class RunData:
    """One collector run, as arrays ready to plot.

    `t` is seconds since the first sample; `unix_t` keeps absolute time so Eve
    episodes (logged in unix time) can be lined up against it.
    """

    path: str = ""
    meta: dict = field(default_factory=dict)
    unix_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    window_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    clicks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    sifted: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    errors: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    qber: np.ndarray = field(default_factory=lambda: np.empty(0))
    total_sifted: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    blocks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    block_progress: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    secret_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    # Session exports (sessionio.py) carry these; auto-archived collector runs
    # do not, and then they stay empty so plots.py draws exactly as before.
    protocol: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    gaps: list = field(default_factory=list)

    def __len__(self):
        return len(self.t)

    @property
    def t_start(self):
        return float(self.unix_t[0]) if len(self.unix_t) else 0.0

    @property
    def t_end(self):
        return float(self.unix_t[-1]) if len(self.unix_t) else 0.0

    @property
    def duration_s(self):
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def name(self):
        return os.path.basename(self.path) if self.path else "run"


def _sidecar(csv_path):
    """Metadata path that pairs with a run CSV."""
    return os.path.splitext(csv_path)[0] + ".json"


def _discard_sidecar(csv_path):
    """Drop a live sidecar whose CSV is gone, so it cannot mislabel the next
    run archived from the same path."""
    try:
        os.remove(_sidecar(csv_path))
    except OSError:
        pass


def _has_samples(path):
    """True if the CSV holds at least one data row (not just the header)."""
    try:
        with open(path) as f:
            f.readline()               # header
            return bool(f.readline().strip())
    except OSError:
        return False


def write_live_meta(csv_path=LIVE_CSV, meta=None):
    """Record the parameters of the run that is starting now.

    Written beside the live CSV so archive_run() can label the run with the
    settings that PRODUCED it. apply() only ever holds the parameters of the
    run it is about to start, while the run it archives is the previous one --
    the sidecar is what carries that description across, and it survives the
    app being closed and reopened in between.
    """
    payload = dict(meta or {})
    payload["started_at"] = time.time()
    payload["started_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(_sidecar(csv_path), "w") as f:
        json.dump(payload, f, indent=4, sort_keys=True, default=str)
        f.write("\n")


def archive_run(csv_path=LIVE_CSV, meta=None, runs_dir=RUNS_DIR):
    """Move a finished run into runs_dir and write its metadata sidecar.

    Metadata comes from `meta` when given, otherwise from the live sidecar
    written by write_live_meta() when the run started -- so the archived run
    is labelled with its OWN parameters, not the ones being applied now.

    Returns the archived CSV path, or None when there was nothing worth
    keeping. A header-only file (Apply pressed twice with no stream in
    between) is deleted rather than archived, so runs_dir stays free of empty
    runs.

    Called by backend.apply() BEFORE it deletes live.csv -- reversing that
    order silently destroys the run.
    """
    if meta is None:
        meta = _load_meta(csv_path)
    if not os.path.exists(csv_path):
        _discard_sidecar(csv_path)
        return None
    if not _has_samples(csv_path):
        try:
            os.remove(csv_path)
        except OSError:
            pass
        _discard_sidecar(csv_path)
        return None

    os.makedirs(runs_dir, exist_ok=True)
    stamp = time.strftime("run_%Y%m%d-%H%M%S")
    dest = os.path.join(runs_dir, stamp + ".csv")
    if os.path.exists(dest):
        # Two archives inside the same second: keep both.
        n = 2
        while os.path.exists(os.path.join(runs_dir, "%s_%d.csv" % (stamp, n))):
            n += 1
        dest = os.path.join(runs_dir, "%s_%d.csv" % (stamp, n))
    os.replace(csv_path, dest)

    payload = dict(meta or {})
    payload["archived_at"] = time.time()
    payload["archived_at_iso"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(_sidecar(dest), "w") as f:
        json.dump(payload, f, indent=4, sort_keys=True, default=str)
        f.write("\n")
    _discard_sidecar(csv_path)      # consumed; must not label the next run
    return dest


def load_run(path=LIVE_CSV):
    """Parse a run CSV (plus its sidecar, if any) into a RunData.

    A missing file, an empty file, or an unparsable row yields fewer samples
    rather than an exception -- plotting a partial run beats refusing to plot.
    """
    rows = {name: [] for name in _COLUMNS}
    protocol = []
    try:
        with open(path, newline="") as f:
            for raw in csv.DictReader(f):
                try:
                    parsed = {name: float(raw[name]) for name in _COLUMNS}
                except (TypeError, ValueError, KeyError):
                    continue           # damaged/truncated row: skip it
                for name, value in parsed.items():
                    rows[name].append(value)
                protocol.append(raw.get("protocol") or "")
    except OSError:
        pass

    run = RunData(path=path, meta=_load_meta(path))
    if not rows["unix_time"]:
        return run

    unix_t = np.asarray(rows["unix_time"], dtype=float)
    run.unix_t = unix_t
    run.t = unix_t - unix_t[0]
    run.window_s = np.asarray(rows["window_s"], dtype=float)
    run.qber = np.asarray(rows["qber"], dtype=float)
    for name in _INT_COLUMNS:
        setattr(run, name, np.asarray(rows[name], dtype=float).astype(np.int64))
    run.protocol = protocol
    run.segments = _clean_dicts(run.meta.get("segments"))
    run.gaps = _clean_dicts(run.meta.get("gaps"))
    return run


def _load_meta(csv_path):
    try:
        with open(_sidecar(csv_path)) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _clean_dicts(value):
    """Keep only the dict entries of a sidecar list.

    Sidecars are machine-written but may be hand-edited or truncated; a
    malformed entry must cost that entry, not the whole figure.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def list_runs(runs_dir=RUNS_DIR):
    """Archived run CSVs, newest first (names are timestamps, so sorting the
    names sorts by time)."""
    return sorted(glob.glob(os.path.join(runs_dir, "run_*.csv")), reverse=True)


def load_eve_episodes(t0, t1, path=EVE_CSV):
    """Eve episodes overlapping the unix-time window [t0, t1].

    eve.csv accumulates across every run ever made, so a run's figure must
    filter it. An episode still open when it was logged (t_end == 0) is drawn
    as lasting to the end of the window.
    """
    out = []
    try:
        with open(path, newline="") as f:
            for raw in csv.DictReader(f):
                try:
                    start = float(raw["t_start"])
                    end = float(raw["t_end"])
                    frac = float(raw["f"])
                except (TypeError, ValueError, KeyError):
                    continue
                if end <= 0.0:
                    end = t1
                if end >= t0 and start <= t1:
                    out.append(EveEpisode(t_start=start, t_end=end, f=frac))
    except OSError:
        pass
    return out
