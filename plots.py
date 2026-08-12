#!/usr/bin/env python3
"""Figures from a recorded collector run.

Pure rendering: runlog.py does the reading, this module only draws. Each panel
is its own function taking an Axes, so the same code serves the combined
figure, the per-panel exports, and (later) a GUI "export figure" button.

The look follows DecoyState/qkd_1decoy_analysis_v13_compact.py -- same palette,
same de-cluttered axes, same "title + parameter subtitle" header -- so figures
from the simulator sit next to the analytic ones in a report without a visual
seam.

Standalone use:
    python3 plots.py                        # logs/live.csv (current run)
    python3 plots.py --run last             # newest archived run
    python3 plots.py <run.csv> -o figs/ --split --b-max 1e7
    python3 plots.py --list                 # what has been archived
"""

import argparse
import os
import sys

import matplotlib
if __name__ == "__main__":       # writing files, not showing windows
    matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import runlog  # noqa: E402

# Palette lifted from the DecoyState figures so both figure families match.
NAVY = "#1F3864"
BLUE = "#2E75B6"
TEAL = "#008080"
AMBER = "#D4A017"
GREEN = "#70AD47"
RED = "#C00000"
GREY = "#888888"
FACE = "#FAFAFA"
EVE_SHADE = "#C00000"

# Per-mode tints for multi-mode session figures, shared with the live panel
# (gui.py imports these) so the two never drift apart.
MODE_BAND = {"pm": "#D6E4F7", "ent": "#E2F0D9"}
MODE_LINE = {"pm": "#2E75B6", "ent": "#70AD47"}
DOWNTIME_BAND = "#DDDDDD"


def style_axis(ax, title=None, xlabel=None, ylabel=None,
               fontsize_title=9, fontsize_label=8, title_pad=5):
    """De-cluttered axes matching the DecoyState figures.

    title_pad is raised by panels that annotate the downtime gaps just above
    the axes, so the label and the title do not sit in the same strip.
    """
    if title:
        ax.set_title(title, fontsize=fontsize_title, fontweight="bold",
                     color=NAVY, pad=title_pad)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize_label)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _shade_eve(ax, run, eve):
    """Red bands over the windows Eve was jamming.

    Episodes are logged in unix time; the panels use seconds-since-start, so
    they are shifted by the run's own start and clipped to it.
    """
    if not eve or len(run) == 0:
        return
    t0, t1 = run.t_start, run.t_end
    labelled = False
    for ep in eve:
        x0 = max(ep.t_start, t0) - t0
        x1 = min(ep.t_end, t1) - t0
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, color=EVE_SHADE, alpha=0.12,
                   label=None if labelled else "Eve jamming")
        labelled = True


def _draw_mode_bands(ax, run, label_gaps=False):
    """Tint each mode's stretch and grey out the stack-down intervals.

    Only session exports carry segments; a plain collector run has none and
    this is a no-op, leaving those figures byte-identical to before.

    Returns the number of gap labels drawn, so the caller can make room for
    them above the axes.
    """
    if not run.segments:
        return 0
    t0 = run.t_start
    for seg in run.segments:
        x0 = float(seg.get("t_start", t0)) - t0
        x1 = float(seg.get("t_end", t0)) - t0
        if x1 <= x0:
            continue
        ax.axvspan(x0, x1, lw=0, alpha=0.55,
                   color=MODE_BAND.get(seg.get("protocol"), MODE_BAND["pm"]))
    x_last = float(run.t[-1]) if len(run.t) else 0.0
    labels = 0
    for gap in run.gaps:
        x0 = max(float(gap.get("t_start", t0)) - t0, 0.0)
        # t_end == 0.0 means "still down": run the band out to the last sample.
        # The sentinel must be tested on the RAW value -- the first Apply's
        # downtime ends exactly at the first sample, whose SHIFTED value is
        # 0.0 and would be misread as "still down", greying the whole figure.
        raw_end = float(gap.get("t_end", 0.0))
        x1 = (raw_end - t0) if raw_end > 0.0 else x_last
        if x1 <= x0:
            continue          # entirely before the first sample, or empty
        ax.axvspan(x0, x1, color=DOWNTIME_BAND, alpha=0.9, lw=0)
        if label_gaps and raw_end > 0.0:
            ax.annotate("%s %.0fs" % ("switch" if gap.get("is_switch")
                                      else "down", raw_end - float(
                                          gap.get("t_start", t0))),
                        xy=((x0 + x1) / 2, 1.0),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=6.5,
                        color="#666666")
            labels += 1
    return labels


def _segment_slices(run):
    """(protocol, slice, death_qber) per segment, clamped to the samples.

    A segment without both sample indices is skipped rather than defaulted to
    the whole run: defaulting would draw a trace bridging the downtime, the
    very artefact the per-segment split exists to prevent.
    """
    out = []
    n = len(run.t)
    for seg in run.segments:
        if seg.get("i0") is None or seg.get("i1") is None:
            continue
        i0 = max(int(seg["i0"]), 0)
        i1 = min(int(seg["i1"]), n - 1)
        if i1 < i0:
            continue
        out.append((seg.get("protocol", "pm"), slice(i0, i1 + 1),
                    seg.get("death_qber")))
    return out


def _once(seen, protocol):
    """Legend label for a protocol the first time it is drawn, else None.

    A session alternates modes (PM -> ENT -> PM), so labelling every segment
    would repeat entries in the legend.
    """
    key = str(protocol).lower()
    if key in seen:
        return None
    seen.add(key)
    return str(protocol).upper()


def panel_qber(ax, run, death_qber=None, eve=()):
    """Measured QBER over time, with the no-key threshold and Eve's episodes.

    A session export (run.segments non-empty) is drawn one trace per mode with
    a stepped threshold; a plain run keeps the single trace and single line it
    has always had.
    """
    n_labels = _draw_mode_bands(ax, run, label_gaps=True)
    _shade_eve(ax, run, eve)
    if len(run) and run.segments:
        seen = set()
        for protocol, sl, seg_death in _segment_slices(run):
            ax.plot(run.t[sl], 100.0 * run.qber[sl], lw=1.6, marker=".",
                    markersize=3,
                    color=MODE_LINE.get(protocol, MODE_LINE["pm"]),
                    label=_once(seen, protocol))
            if seg_death:
                ax.hlines(100.0 * float(seg_death), run.t[sl][0],
                          run.t[sl][-1], color=RED, lw=1.0, ls="--")
    elif len(run):
        ax.plot(run.t, 100.0 * run.qber, color=NAVY, lw=1.6, label="measured")
    if death_qber:
        ax.axhline(100.0 * death_qber, color=RED, lw=1.0, ls="--",
                   label=r"death threshold %.2f%%" % (100.0 * death_qber))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, loc="best")
    style_axis(ax, title="1. QBER", xlabel="time (s)", ylabel="QBER (%)",
               title_pad=14 if n_labels else 5)


def panel_rate(ax, run):
    """Sifted bits per window, annotated with the mean detection rate."""
    _draw_mode_bands(ax, run)
    if len(run) and run.segments:
        seen = set()
        for protocol, sl, _ in _segment_slices(run):
            ax.plot(run.t[sl], run.sifted[sl], lw=1.6, marker=".",
                    markersize=3,
                    color=MODE_LINE.get(protocol, MODE_LINE["pm"]),
                    label=_once(seen, protocol))
    elif len(run):
        ax.plot(run.t, run.sifted, color=BLUE, lw=1.6, label="sifted / window")
    if len(run):
        with_time = run.window_s > 0
        if with_time.any():
            mean_clicks = float((run.clicks[with_time]
                                 / run.window_s[with_time]).mean())
            ax.text(0.02, 0.06, "mean %.2f M clicks/s" % (mean_clicks / 1e6),
                    transform=ax.transAxes, fontsize=7, color="#555555",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5",
                              ec="#DDDDDD", lw=0.5))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, loc="best")
    style_axis(ax, title="2. Sifted rate", xlabel="time (s)",
               ylabel="bits / window")


def panel_secret(ax, run, b_max=None):
    """Cumulative distilled secret bits; flat stretches are blocks that died.

    A session export is stepped once per segment, coloured by the mode that
    produced those bits, so the panel says WHICH protocol filled the buffer;
    a plain run keeps the single green step it has always had.
    """
    _draw_mode_bands(ax, run)
    if len(run) and run.segments:
        seen = set()
        for protocol, sl, _ in _segment_slices(run):
            ax.step(run.t[sl], run.secret_bits[sl], where="post", lw=1.8,
                    color=MODE_LINE.get(protocol, MODE_LINE["pm"]),
                    label=_once(seen, protocol))
        _mark_dead_blocks(ax, run)
    elif len(run):
        ax.step(run.t, run.secret_bits, where="post", color=GREEN, lw=1.8,
                label="secret bits")
        _mark_dead_blocks(ax, run)
    if b_max:
        ax.axhline(b_max, color=AMBER, lw=1.0, ls=":",
                   label=r"$B_{max}$ = %s" % format(int(b_max), ","))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, loc="best")
    style_axis(ax, title=r"3. Secret key buffer", xlabel="time (s)",
               ylabel="bits")


def _mark_dead_blocks(ax, run):
    """Cross where a block completed but distilled nothing (QBER over bound).

    On a session export the difference is taken WITHIN each segment. Every
    collector process restarts the blocks counter from zero, so the pair
    straddling a downtime compares two unrelated counters: it reads as "a
    block completed and produced nothing" whenever the new collector's first
    count happens to exceed the old one's last, planting a phantom cross at
    the first sample after the gap.
    """
    spans = ([sl for _, sl, _ in _segment_slices(run)] if run.segments
             else [slice(0, len(run))])
    idx = []
    for sl in spans:
        blocks = run.blocks[sl]
        secret = run.secret_bits[sl]
        dead = (blocks[1:] > blocks[:-1]) & (secret[1:] == secret[:-1])
        idx.extend(sl.start + i + 1 for i, flag in enumerate(dead) if flag)
    if not idx:
        return
    ax.plot(run.t[idx], run.secret_bits[idx], "x", color=RED, ms=7, mew=1.6,
            ls="none", label="block with no key")


def _blocks_caption(ax, n_blocks, total):
    """Corner box counting the finished blocks and the bits they distilled."""
    ax.text(0.02, 0.90,
            "%d block(s) completed — %s secret bits"
            % (n_blocks, format(total, ",")),
            transform=ax.transAxes, fontsize=7, color="#555555", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#DDDDDD",
                      lw=0.5))


def panel_blocks(ax, run):
    """Fill level of the block being accumulated: each drop is a finished block.

    Split per segment on a session export, like the panels above it: one trace
    over the whole run interpolates the fill level straight across the grey
    downtime band, asserting the block kept filling while the stack was down.
    """
    _draw_mode_bands(ax, run)
    if len(run) and run.segments:
        slices = _segment_slices(run)
        for protocol, sl, _ in slices:
            ax.plot(run.t[sl], run.block_progress[sl], lw=1.4,
                    color=MODE_LINE.get(protocol, MODE_LINE["pm"]))
        # Each collector restarts the blocks counter, so the session total is
        # the sum of the per-segment finals, not the last value in the column.
        n_blocks = sum(int(run.blocks[sl][-1]) for _, sl, _ in slices)
        total = int(run.secret_bits[-1])
        _blocks_caption(ax, n_blocks, total)
    elif len(run):
        ax.plot(run.t, run.block_progress, color=TEAL, lw=1.4)
        _blocks_caption(ax, int(run.blocks[-1]), int(run.secret_bits[-1]))
    style_axis(ax, title="4. Block fill progress", xlabel="time (s)",
               ylabel="bits in block")


# (metadata key, LaTeX label, plain label) for the parameter subtitle.
_ENV_LABELS = (
    ("d_km", r"$d=%g$ km", "d=%g km"),
    ("mu1", r"$\mu_1=%g$", "mu1=%g"),
    ("mu2", r"$\mu_2=%g$", "mu2=%g"),
    ("p1", r"$p_{\mu_1}=%g$", "p1=%g"),
    ("mu", r"$\mu=%g$", "mu=%g"),
    ("eta_a", r"$\eta_A=%g$", "eta_A=%g"),
    ("e_det", r"$e_{det}=%g$", "e_det=%g"),
    ("e_intr", r"$e_2=%g$", "e2=%g"),
)


def _subtitle(run, latex=True):
    """One line of the parameters that produced the run, from the sidecar.

    latex=False gives the same content without math markup, for terminal
    output where "$\\mu=0.045$" would just be noise.
    """
    meta = run.meta or {}
    bits = []
    proto = str(meta.get("protocol", "")).upper()
    if proto:
        bits.append(proto)
    env = meta.get("env") or {}
    if isinstance(env, dict):
        for key, tex, plain in _ENV_LABELS:
            if isinstance(env.get(key), (int, float)):
                bits.append((tex if latex else plain) % env[key])
    if isinstance(meta.get("n_z_block"), (int, float)):
        block = format(int(meta["n_z_block"]), ",")
        bits.append((r"block $=%s$" if latex else "block=%s") % block)
    forecast = meta.get("forecast") or {}
    if isinstance(forecast, dict) and isinstance(
            forecast.get("l_predict"), (int, float)):
        ell = format(int(forecast["l_predict"]), ",")
        bits.append((r"$\ell_{predict}=%s$" if latex else "l_predict=%s") % ell)
    if run.segments:
        # "->" for the terminal listing: a non-UTF-8 console cannot print "→".
        modes = (" → " if latex else " -> ").join(
            "%s %.0fs" % (str(s.get("protocol", "?")).upper(),
                          float(s.get("t_end", 0)) - float(s.get("t_start", 0)))
            for s in run.segments)
        bits.append(modes)
        switches = [g for g in run.gaps if g.get("is_switch")]
        if switches:
            total = sum(float(g.get("t_end", 0)) - float(g.get("t_start", 0))
                        for g in switches)
            bits.append("%d switch (%.0fs down)" % (len(switches), total))
    if len(run):
        bits.append("%.0f s, %d windows" % (run.duration_s, len(run)))
    return "   ".join(bits)


def figure_run(run, death_qber=None, b_max=None, eve=(), figsize=(9, 10)):
    """The four panels stacked on a shared time axis, as one figure."""
    fig = Figure(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(FACE)
    fig.text(0.5, 0.975, "QKD run — %s" % run.name, ha="center",
             fontsize=13, fontweight="bold", color=NAVY)
    subtitle = _subtitle(run)
    if subtitle:
        fig.text(0.5, 0.957, subtitle, ha="center", fontsize=8.5,
                 color="#444444")
    # A session pads every panel title upwards to clear the gap labels
    # (style_axis's title_pad), which pushes the FIRST panel's title into the
    # parameter subtitle above it. Drop the grid to give that pad somewhere to
    # go; a plain run has no labels, no pad, and keeps its original geometry.
    gs = fig.add_gridspec(4, 1, hspace=0.42, left=0.10, right=0.97,
                          top=0.915 if run.segments else 0.93, bottom=0.06)
    for i, (_, draw) in enumerate(_panels_for(run, death_qber, b_max, eve)):
        draw(fig.add_subplot(gs[i]))
    return fig


def _panels_for(run, death_qber=None, b_max=None, eve=()):
    """(name, draw(ax)) for each panel, with run and overlays already bound.

    One list of panels serves both the combined figure and the split export,
    so the two can never drift apart.
    """
    return (
        ("qber", lambda ax: panel_qber(ax, run, death_qber=death_qber,
                                       eve=eve)),
        ("rate", lambda ax: panel_rate(ax, run)),
        ("secret", lambda ax: panel_secret(ax, run, b_max=b_max)),
        ("blocks", lambda ax: panel_blocks(ax, run)),
    )


def save_run_figures(run, out_dir, split=False, death_qber=None, b_max=None,
                     eve=(), dpi=150):
    """Write the figures; returns the paths written.

    split=False -> one combined PNG. split=True -> one PNG per panel, drawn by
    the very same panel functions.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(run.name)[0] or "run"
    if not split:
        path = os.path.join(out_dir, "%s.png" % stem)
        figure_run(run, death_qber=death_qber, b_max=b_max,
                   eve=eve).savefig(path, dpi=dpi, bbox_inches="tight",
                                    facecolor=FACE)
        return [path]

    written = []
    for name, draw in _panels_for(run, death_qber, b_max, eve):
        fig = Figure(figsize=(7, 3.2), dpi=110)
        fig.patch.set_facecolor(FACE)
        draw(fig.add_subplot(1, 1, 1))
        path = os.path.join(out_dir, "%s_%s.png" % (stem, name))
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=FACE)
        written.append(path)
    return written


def _print_run_list():
    runs = runlog.list_runs()
    if not runs:
        print("no archived runs in %s" % runlog.RUNS_DIR)
        return
    print("%-30s %8s %7s %11s  %s" % ("run", "windows", "dur[s]", "secret",
                                      "params"))
    for path in runs:
        run = runlog.load_run(path)
        secret = int(run.secret_bits[-1]) if len(run) else 0
        print("%-30s %8d %7.0f %11s  %s"
              % (os.path.basename(path), len(run), run.duration_s,
                 format(secret, ","), _subtitle(run, latex=False)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", default=None,
                    help="run CSV to plot (default: logs/live.csv)")
    ap.add_argument("--run", default=None,
                    help="'last' to plot the newest archived run")
    ap.add_argument("--list", action="store_true",
                    help="list archived runs and exit")
    ap.add_argument("-o", "--out-dir", default=os.path.join(runlog.LOG_DIR,
                                                            "figures"),
                    help="directory for the PNGs")
    ap.add_argument("--split", action="store_true",
                    help="one PNG per panel instead of one combined figure")
    ap.add_argument("--b-max", type=float, default=None,
                    help="buffer capacity line on the secret-bits panel")
    ap.add_argument("--death-qber", type=float, default=None,
                    help="no-key QBER threshold line, e.g. 0.0522")
    ap.add_argument("--no-eve", action="store_true",
                    help="do not shade Eve's jamming episodes")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    if args.list:
        _print_run_list()
        return 0

    if args.run == "last":
        runs = runlog.list_runs()
        if not runs:
            print("no archived runs in %s" % runlog.RUNS_DIR, file=sys.stderr)
            return 1
        path = runs[0]
    else:
        path = args.csv or runlog.LIVE_CSV

    run = runlog.load_run(path)
    if not len(run):
        print("no samples in %s" % path, file=sys.stderr)
        return 1

    eve = () if args.no_eve else runlog.load_eve_episodes(run.t_start,
                                                          run.t_end)
    written = save_run_figures(run, args.out_dir, split=args.split,
                               death_qber=args.death_qber, b_max=args.b_max,
                               eve=eve, dpi=args.dpi)
    print("%d window(s), %.0f s, %s secret bits%s"
          % (len(run), run.duration_s,
             format(int(run.secret_bits[-1]), ","),
             ", %d Eve episode(s)" % len(eve) if eve else ""))
    for p in written:
        print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
