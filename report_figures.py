#!/usr/bin/env python3
"""Export the thesis result figures: the visualiser's panels, tidied for print.

`policyviz.draw_panels` draws for a live window -- long titles, a hint about
clicking the legend, raw policy specs as labels, and a key-buffer panel that is
solid noise at the loads where mode selection matters. None of that belongs in
a report, so this script redraws the same session and then:

  * drops the key-buffer panel and re-lays out the three that remain,
  * puts the figure on a white canvas, to match the page it is printed on,
  * shortens every title and gives the policies their report names,
  * puts the run's parameters in one compact line under the title.

Nothing here recomputes anything: the numbers are the visualiser's own, so a
figure cannot disagree with the scoreboard it came from.

    python3 report_figures.py [-o figs]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

import plots  # noqa: E402
from policyviz import VizConfig, VizSession, draw_panels  # noqa: E402

# Report names for the policy specs used below. Keys are what
# `PolicyRun.display_name` produces.
SHORT = {
    "always-pm": "AlwaysPM",
    "always-ent": "AlwaysENT",
    "block-aware": "BlockAware",
    "greedy-0.045": "Greedy",
    "decoy-0.100-1.000-60": "DecoyWatch",
    "decoy-0.100-0.018-60": "DecoyWatch",
    "oracle (upper bound)": "Oracle (bound)",
}

# (file stem, adversary, lam, horizon, policies, what the bands mean)
FIGURES = (
    ("fig_jam", "jam", 2.5, 600,
     ("always-pm", "always-ent", "block-aware"),
     "amber band = jamming"),
    ("fig_pns", "pns", 5.0, 600,
     ("always-pm", "always-ent", "decoy-0.100-1.000-60"),
     "red band = PNS attack"),
    ("fig_both", "both", 5.0, 600,
     ("always-pm", "block-aware", "decoy-0.100-0.018-60"),
     "amber = jamming, red = PNS attack"),
)

# Panel geometry, bottom-up, once the buffer panel is gone.
BOXES = ((0.10, 0.70, 0.88, 0.20),      # mode lanes
         (0.10, 0.40, 0.88, 0.23),      # QBER
         (0.10, 0.07, 0.88, 0.24))      # cumulative eta


def short(label):
    return SHORT.get(label, label)


def tidy(fig, session, config, bands):
    """Turn the live panels into a printable figure."""
    ax_mode, ax_qber, ax_buf, ax_eta = fig.axes
    fig.delaxes(ax_buf)
    # White, not the live panel's off-white: the report's tables and page are
    # white, and a tinted canvas reads as a screenshot next to them.
    fig.patch.set_facecolor("white")
    for ax, box in zip((ax_mode, ax_qber, ax_eta), BOXES):
        ax.set_position(box)

    ax_mode.set_yticklabels([short(r.display_name) for r in session.runs],
                            fontsize=8)
    plots.style_axis(ax_mode,
                     title="Mode per slot  (blue = PM, green = ENT;  %s)"
                           % bands)
    plots.style_axis(ax_qber, title="Observed QBER")
    ax_qber.set_ylabel("QBER [%]")
    plots.style_axis(ax_eta, title="Cumulative goodput η")
    ax_eta.set_ylabel("η")
    ax_eta.set_xlabel("slot")

    handles = [ln for ln in ax_qber.get_lines() if ln.get_label()[0] != "_"]
    for line in handles:
        line.set_label(short(line.get_label()))
    ax_qber.legend(handles=handles, fontsize=7, ncol=len(handles),
                   loc="upper left", framealpha=0.9)
    if ax_eta.get_legend() is not None:
        ax_eta.get_legend().remove()

    fig.suptitle("%s adversary   λ = %.1f   n_X = %.0e   d = %.0f km   "
                 "T₀ = %d slots   seed index %d"
                 % (config.adversary, config.lam, float(config.n_x),
                    config.d_km, config.t0_deadline, config.seed_index),
                 fontsize=9, color=plots.NAVY, y=0.975)


def build(stem, adversary, lam, horizon, specs, bands, out_dir):
    config = VizConfig(adversary=adversary, lam=lam, horizon=horizon,
                       specs=specs)
    session = VizSession(config)
    session.step(horizon)
    fig = Figure(figsize=(9.5, 6.2), dpi=200)
    draw_panels(fig, session)
    tidy(fig, session, config, bands)
    path = os.path.join(out_dir, stem + ".png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    etas = "  ".join("%s %.4f" % (short(r.agent.name), r.env.eta)
                     for r in session.runs)
    print("%-10s %s\n           %s" % (stem, path, etas))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out-dir", default="figs")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    for stem, adversary, lam, horizon, specs, bands in FIGURES:
        build(stem, adversary, lam, horizon, specs, bands, args.out_dir)


if __name__ == "__main__":
    main()
