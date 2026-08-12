#!/usr/bin/env python3
"""Write a GUI session out as a collector-shaped run file.

The live panel deliberately draws a plain, cheap view; plots.py is the single
place that draws report-quality figures, and it works from files. So the
export button does not render anything itself -- it serialises the session and
lets plots.py read it back.

Direction of travel is why this is not part of runlog.py: runlog READS run
files and archives what the collector produced, while this WRITES one out of
in-memory GUI state. Keeping the writer here also keeps runlog free of any
Timeline import.

Two honesty flags go in the sidecar because the file looks like a collector
run but is not one in two respects:
  * secret_bits holds the SESSION-cumulative total (live.csv restarts it per
    collector process), so it never steps backwards at a mode switch;
  * the per-intensity columns (n_sig/m_sig/n_dec/m_dec) are zero -- Snapshot
    does not carry them, and inventing numbers would be worse than a flag.
"""

import csv
import json
import os
import re
import time

import runlog

# The 14 columns collector.py writes, plus the protocol that produced each row.
CSV_COLUMNS = ("unix_time", "window_s", "clicks", "sifted", "errors", "qber",
               "n_sig", "m_sig", "n_dec", "m_dec", "total_sifted", "blocks",
               "block_progress", "secret_bits", "protocol")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_TRAVERSAL = re.compile(r"\.{2,}")   # "..", "...", etc: no name content, and
                                     # ".." is the parent-directory token


def sanitize_name(name):
    """Turn a hand-typed name into a safe bare filename.

    Names come from a text prompt, so this must never let one escape the runs
    directory: separators are neutralised, and a run of 2+ dots -- ".." is the
    parent-directory token -- is replaced by a word instead of being trimmed
    away, so a lone ".." sanitizes to something usable rather than to "" (an
    empty result is reserved for names with no salvageable content at all,
    like "///").
    """
    cleaned = _UNSAFE.sub("_", (name or "").strip())
    cleaned = _TRAVERSAL.sub("dot", cleaned)
    cleaned = cleaned.strip(".")            # drop a single leading/trailing dot
    if cleaned.lower().endswith(".csv"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("name is empty after removing unsafe characters")
    return cleaned


def session_paths(name, runs_dir=None):
    """(csv_path, json_path) for a session name, without writing anything."""
    runs_dir = runs_dir or runlog.RUNS_DIR
    stem = sanitize_name(name)
    return (os.path.join(runs_dir, stem + ".csv"),
            os.path.join(runs_dir, stem + ".json"))


def _protocol_at(timeline, index):
    """Which mode produced sample `index` (empty string if unknown)."""
    for seg in timeline.segments:
        if seg.i0 <= index <= seg.i1:
            return seg.protocol
    return ""


def export_session(timeline, name, extra_meta=None, runs_dir=None):
    """Write the session as <name>.csv + <name>.json; returns both paths.

    Raises ValueError for an empty timeline or an unusable name -- callers
    should surface that rather than write a file nobody can plot.
    """
    if not len(timeline):
        raise ValueError("nothing to export: the session has no samples")
    runs_dir = runs_dir or runlog.RUNS_DIR
    csv_path, json_path = session_paths(name, runs_dir)
    os.makedirs(runs_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for i in range(len(timeline)):
            writer.writerow([
                "%.3f" % timeline.t[i],
                "%.3f" % timeline.window_s[i],
                timeline.clicks[i],
                timeline.sifted[i],
                timeline.errors[i],
                "%.6f" % (timeline.qber[i] / 100.0),   # stored as percent
                0, 0, 0, 0,                            # per-intensity: unknown
                timeline.total_sifted[i],
                timeline.blocks[i],
                timeline.block_progress[i],
                timeline.secret_total[i],
                _protocol_at(timeline, i),
            ])

    meta = {
        "kind": "session",
        "exported_at": time.time(),
        "exported_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "secret_bits_are_cumulative": True,
        "per_intensity_columns_are_zero": True,
        "segments": [{"protocol": s.protocol, "t_start": s.t_start,
                      "t_end": s.t_end, "death_qber": s.death_qber,
                      "i0": s.i0, "i1": s.i1} for s in timeline.segments],
        "gaps": [{"t_start": g.t_start, "t_end": g.t_end,
                  "is_switch": g.is_switch} for g in timeline.gaps],
        "secret_by_protocol": timeline.secret_by_protocol,
    }
    meta.update(extra_meta or {})
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=4, sort_keys=True, default=str)
        f.write("\n")
    return csv_path, json_path
