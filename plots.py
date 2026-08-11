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


def style_axis(ax, title=None, xlabel=None, ylabel=None,
               fontsize_title=9, fontsize_label=8):
    """De-cluttered axes matching the DecoyState figures."""
    if title:
        ax.set_title(title, fontsize=fontsize_title, fontweight="bold",
                     color=NAVY, pad=5)
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


def panel_qber(ax, run, death_qber=None, eve=()):
    """Measured QBER over time, with the no-key threshold and Eve's episodes."""
    _shade_eve(ax, run, eve)
    if len(run):
        ax.plot(run.t, 100.0 * run.qber, color=NAVY, lw=1.6, label="measured")
    if death_qber:
        ax.axhline(100.0 * death_qber, color=RED, lw=1.0, ls="--",
                   label=r"death threshold %.2f%%" % (100.0 * death_qber))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, loc="best")
    style_axis(ax, title="1. QBER", xlabel="time (s)", ylabel="QBER (%)")


def panel_rate(ax, run):
    """Sifted bits per window, annotated with the mean detection rate."""
    if len(run):
        ax.plot(run.t, run.sifted, color=BLUE, lw=1.6, label="sifted / window")
        with_time = run.window_s > 0
        if with_time.any():
            mean_clicks = float((run.clicks[with_time]
                                 / run.window_s[with_time]).mean())
            ax.text(0.02, 0.06, "mean %.2f M clicks/s" % (mean_clicks / 1e6),
                    transform=ax.transAxes, fontsize=7, color="#555555",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5",
                              ec="#DDDDDD", lw=0.5))
        ax.legend(fontsize=7.5, loc="best")
    style_axis(ax, title="2. Sifted rate", xlabel="time (s)",
               ylabel="bits / window")


def panel_secret(ax, run, b_max=None):
    """Cumulative distilled secret bits; flat stretches are blocks that died."""
    if len(run):
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
    """Cross where a block completed but distilled nothing (QBER over bound)."""
    new_block = run.blocks[1:] > run.blocks[:-1]
    no_key = run.secret_bits[1:] == run.secret_bits[:-1]
    dead = new_block & no_key
    if not dead.any():
        return
    idx = [i + 1 for i, flag in enumerate(dead) if flag]
    ax.plot(run.t[idx], run.secret_bits[idx], "x", color=RED, ms=7, mew=1.6,
            ls="none", label="block with no key")


def panel_blocks(ax, run):
    """Fill level of the block being accumulated: each drop is a finished block."""
    if len(run):
        ax.plot(run.t, run.block_progress, color=TEAL, lw=1.4)
        n_blocks = int(run.blocks[-1])
        total = int(run.secret_bits[-1])
        ax.text(0.02, 0.90,
                "%d block(s) completed — %s secret bits"
                % (n_blocks, format(total, ",")),
                transform=ax.transAxes, fontsize=7, color="#555555",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5",
                          ec="#DDDDDD", lw=0.5))
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
    gs = fig.add_gridspec(4, 1, hspace=0.42, left=0.10, right=0.97,
                          top=0.93, bottom=0.06)
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
