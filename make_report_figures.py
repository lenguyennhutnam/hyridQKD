#!/usr/bin/env python3
"""Create report figures from the repository's measured simulation outputs.

Figures:
  1. Two-tier platform and validation flow (schematic; values are measured).
  2. ENT/PM throughput ratios for the hardware scenarios in the report.
  3. Goodput vs controlled PNS duty cycle (reported 8-seed experiment).
  4. Winning policy map from logs/optmap_pns.csv (4-seed experiment).
  5. Example policy episode for PNS (seed 0; illustrative, not an average).

Run from qkd_app/ in WSL:
  python3 make_report_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figs" / "chapter4"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1F3864"
BLUE = "#2E75B6"
TEAL = "#008080"
AMBER = "#D4A017"
RED = "#B4443E"
INK = "#263445"
MUTED = "#64748B"
GRID = "#DCE3EA"
PALE = "#F4F7FA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=NAVY, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)


def save(fig, name):
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


def platform_flow():
    """Draw a schematic distinguishing measured and numerical-model stages."""
    fig, ax = plt.subplots(figsize=(12, 6.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.suptitle("Simulation platform and validation boundary", x=0.07, y=0.97,
                 ha="left", color=NAVY, fontsize=17, fontweight="bold")
    ax.text(0.07, 0.91,
            "The event-stream stack anchors physical-layer quantities; the slot model evaluates policies and message delivery.",
            transform=ax.transAxes, color=MUTED, fontsize=10)

    def box(x, y, w, h, title, body, edge, face="white", title_color=None):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.03,rounding_size=0.12",
                               linewidth=1.6, edgecolor=edge, facecolor=face)
        ax.add_patch(patch)
        ax.text(x + 0.18, y + h - 0.3, title, fontsize=11, fontweight="bold",
                color=title_color or edge, va="top")
        ax.text(x + 0.18, y + h - 0.72, body, fontsize=9, color=INK,
                va="top", linespacing=1.45)

    box(0.4, 4.15, 3.1, 1.85, "TIER 1 · EVENT-STREAM STACK",
        "hw_sim (Alice/Bob)\nGC synchronisation\nQBER collector + sifting",
        BLUE, "#F3F8FC")
    box(4.15, 4.15, 3.25, 1.85, "MEASURED OUTPUTS",
        "QBER by window\nSifted bits per slot\nKey bits per finite block",
        TEAL, "#F1F8F7")
    box(8.1, 4.15, 3.45, 1.85, "ANALYTIC KEY BOUNDS",
        "PM: Rusca 1-decoy\nENT: Chapman BBM92\nPNS effects: analytic model",
        AMBER, "#FFFAEB")
    box(0.4, 0.85, 3.1, 1.95, "TIER 2 · SLOT-BASED MODEL",
        "HybridEnv: Eve + channel\nFinite-key blocks + buffer\nMessage queue + deadlines",
        NAVY, "#F4F6FA")
    box(4.15, 0.85, 3.25, 1.95, "POLICY EVALUATION",
        "Always / threshold baselines\nBlockAwareGreedy\nDecoyWatch + Oracle reference",
        NAVY, "#F4F6FA")
    box(8.1, 0.85, 3.45, 1.95, "SYSTEM-LEVEL OUTPUT",
        "Messages served on time\nGoodput η across seeds\nSwitches, buffer, expiries",
        TEAL, "#F1F8F7")

    def arrow(x1, y1, x2, y2, color, text=None, rad=0):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                              mutation_scale=15, linewidth=1.7, color=color,
                              connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arr)
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.16, text, color=color,
                    fontsize=8, ha="center", va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    arrow(3.5, 5.05, 4.15, 5.05, BLUE)
    arrow(7.4, 5.05, 8.1, 5.05, AMBER, "finite-key analysis")
    arrow(9.82, 4.15, 1.95, 2.8, AMBER, "calibrates mode physics: sifted rate and key/block",
          rad=-0.12)
    arrow(3.5, 1.82, 4.15, 1.82, NAVY)
    arrow(7.4, 1.82, 8.1, 1.82, TEAL)

    ax.text(0.45, 3.45,
            "Validation boundary: physical layer compared with stack (≤0.4% in Table 4.1);\n"
            "classical queue/buffer/deadline layer is unit-tested, not stack-validated.",
            color=RED, fontsize=9, ha="left", va="center",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF5F4",
                  "edgecolor": "#E8C5C1"})
    save(fig, "fig4_0_platform_validation.png")


def hardware_ratio():
    """Plot reported ENT/PM throughput ratios at 55 km."""
    labels = ["Chapman\ncurrent", "SNSPD\ndetector", "Next-gen\nENT", "High source\ncollection"]
    values = [10.7, 24.7, 91.3, 122.5]
    colors = [MUTED, BLUE, TEAL, NAVY]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.64,
                  edgecolor="white", linewidth=1.0)
    ax.axhline(100, color=AMBER, linestyle="--", linewidth=1.6,
               label="ENT = PM throughput")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 140)
    style_axis(ax, "ENT throughput depends strongly on hardware efficiency",
               "ENT hardware scenario (55 km)", "ENT throughput / PM throughput (%)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val:.1f}%",
                ha="center", va="bottom", color=INK, fontweight="bold", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.12, 0.015,
             "Same channel and block size; μ re-optimised for each hardware configuration. "
             "The 80% source-efficiency scenario is an assumed next-generation case.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig4_3_ent_hardware_ratio.png")


def pns_duty():
    """Plot the reported 8-seed mean goodput values against PNS duty cycle."""
    duty = np.array([2, 5, 10, 20, 40])
    series = {
        "AlwaysPM": ([0.5562, 0.5357, 0.4976, 0.4396, 0.3388], NAVY, "o"),
        "AlwaysENT": ([0.5034, 0.5034, 0.5034, 0.5034, 0.5034], MUTED, "s"),
        "DecoyWatch": ([0.5655, 0.5592, 0.5476, 0.5304, 0.5077], TEAL, "o"),
        "Oracle (reference bound)": ([0.5681, 0.5657, 0.5613, 0.5545, 0.5428], AMBER, "^"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    for label, (vals, color, marker) in series.items():
        ls = "--" if label == "Oracle (reference bound)" else "-"
        ax.plot(duty, vals, color=color, marker=marker, linewidth=2.1,
                markersize=6.5, linestyle=ls, label=label)
    ax.axvline(10, color=RED, linewidth=1.1, linestyle=":")
    ax.annotate("largest measured gain\nnear mode crossover",
                xy=(10, 0.5476), xytext=(15, 0.572),
                arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.2},
                color=RED, fontsize=9, ha="left")
    ax.set_xticks(duty)
    ax.set_ylim(0.30, 0.60)
    ax.set_xlim(0, 43)
    style_axis(ax, "DecoyWatch responds to the stealthy PNS attack",
               "PNS attack duty cycle (%)", "Goodput η (fraction of messages served on time)")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.text(0.12, 0.015,
             "8-seed means; λ = 5 messages/slot, n_X = 10⁵, 55 km, ent_advanced. "
             "Oracle is a relaxed reference, not an implementable policy.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig4_7_pns_duty_goodput.png")


def policy_map():
    """Draw winner and goodput heatmaps from the archived PNS optmap CSV."""
    path = ROOT / "logs" / "optmap_pns.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")

    ds = sorted({float(r["d_km"]) for r in rows})
    ns = sorted({float(r["n_x"]) for r in rows})
    winners = {}
    for r in rows:
        if int(r["is_winner"]) == 1 and int(r["is_bound"]) == 0:
            winners[(float(r["d_km"]), float(r["n_x"]))] = r
    expected = len(ds) * len(ns)
    if len(winners) != expected:
        raise RuntimeError(f"Expected {expected} winner cells, found {len(winners)}")

    winner_names = sorted({r["policy"] for r in winners.values()})
    colors = {"always-ent": "#CBD5E1"}
    decoy_colors = [TEAL, BLUE, AMBER, RED]
    for i, name in enumerate(n for n in winner_names if n.startswith("decoy-")):
        colors[name] = decoy_colors[i % len(decoy_colors)]
    other_colors = [NAVY, "#7C3AED", "#059669"]
    for i, name in enumerate(n for n in winner_names
                             if n != "always-ent" and not n.startswith("decoy-")):
        colors[name] = other_colors[i % len(other_colors)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("Winning policy changes with distance and finite-key block length",
                 x=0.07, y=0.98, ha="left", color=NAVY, fontsize=16,
                 fontweight="bold")
    ax1.set_xlim(0, len(ns))
    ax1.set_ylim(0, len(ds))
    ax1.invert_yaxis()
    for iy, d in enumerate(ds):
        for ix, n in enumerate(ns):
            r = winners[(d, n)]
            policy = r["policy"]
            eta = float(r["eta"])
            rect = plt.Rectangle((ix, iy), 1, 1, facecolor=colors[policy],
                                 edgecolor="white", linewidth=2)
            ax1.add_patch(rect)
            label = ("AlwaysENT" if policy == "always-ent" else
                     "DecoyWatch" if policy.startswith("decoy-") else policy)
            ax1.text(ix+0.5, iy+0.43, label, ha="center", va="center",
                     fontsize=8.2, fontweight="bold",
                     color="white" if colors[policy] not in ("#CBD5E1", AMBER) else INK)
            ax1.text(ix+0.5, iy+0.72, f"η={eta:.3f}", ha="center", va="center",
                     fontsize=8.1, color=INK if colors[policy] == "#CBD5E1" else "white")
    ax1.set_xticks(np.arange(len(ns))+0.5,
                   ["10⁵", "3×10⁵", "10⁶", "3×10⁶"])
    ax1.set_yticks(np.arange(len(ds))+0.5, [f"{d:g} km" for d in ds])
    ax1.set_xlabel("Finite-key block length n_X")
    ax1.set_ylabel("Link distance")
    ax1.set_title("Best feasible policy per cell", loc="left", color=NAVY,
                  fontweight="bold", pad=12)
    ax1.tick_params(length=0, colors=MUTED)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    matrix = np.array([[float(winners[(d, n)]["eta"]) for n in ns] for d in ds])
    im = ax2.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(np.arange(len(ns)), ["10⁵", "3×10⁵", "10⁶", "3×10⁶"])
    ax2.set_yticks(np.arange(len(ds)), [f"{d:g} km" for d in ds])
    ax2.set_xlabel("Finite-key block length n_X")
    ax2.set_ylabel("Link distance")
    ax2.set_title("Winning goodput η", loc="left", color=NAVY,
                  fontweight="bold", pad=12)
    for iy in range(len(ds)):
        for ix in range(len(ns)):
            v = matrix[iy, ix]
            ax2.text(ix, iy, f"{v:.3f}", ha="center", va="center",
                     fontsize=9, color="white" if v > 0.52 else INK,
                     fontweight="bold")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.04)
    cbar.set_label("η")
    fig.text(0.07, 0.015,
             "PNS-only adversary; λ = 5; 4 seeds. Values are archived optmap results. "
             "At 20 km η saturates at 1, so the winning-policy label is less informative.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save(fig, "fig4_9_policy_map_pns.png")


def episode_trace():
    """Run one PNS episode and plot observations with the selected mode."""
    import sys
    sys.path.insert(0, str(ROOT))
    from hybridenv import HybridEnv, HybridParams, ENT
    from evemarkov import EveParams
    from policies import AlwaysPM, AlwaysENT, DecoyWatch

    params = HybridParams(d_km=55, lam=4.0, n_x=100_000,
                          l0=8192, t0_deadline=20, b_max=10_000_000,
                          horizon=3600, w_window=5,
                          ent_preset="ent_advanced")
    eve = EveParams(s_levels=(0.0, 0.0, 0.6), f_levels=(0.0, 0.0, 0.0))
    env = HybridEnv(params, eve)

    policies = [
        ("AlwaysPM", AlwaysPM()),
        ("AlwaysENT", AlwaysENT()),
        ("DecoyWatch", DecoyWatch(decoy_threshold=0.1,
                                  qber_threshold=1.0,
                                  probe_every=60)),
    ]
    tracks = []
    for name, policy in policies:
        policy.bind(env)
        obs = env.reset(0)
        times, modes, qbers, pns_levels = [], [], [], []
        done = False
        while not done:
            obs, _, done, info = env.step(policy.act(obs))
            times.append(len(times) + 1)
            modes.append(info["mode"])
            qbers.append(100.0 * info["qber"])
            pns_levels.append(100.0 * info.get("pns", 0.0))
        tracks.append((name, np.asarray(times), np.asarray(modes),
                       np.asarray(qbers), np.asarray(pns_levels)))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.2]})
    fig.suptitle("One PNS episode: the QBER stays flat while decoy monitoring detects the attack",
                 x=0.08, y=0.985, ha="left", color=NAVY, fontsize=15,
                 fontweight="bold")
    d = tracks[2][1]
    pns = tracks[2][4]
    active = pns > 0
    if np.any(active):
        axes[0].fill_between(d, 0, 1, where=active, step="post",
                             color=RED, alpha=0.12, transform=axes[0].get_xaxis_transform(),
                             label="PNS active")
    axes[0].plot(tracks[0][1], tracks[0][3], color=BLUE, lw=1.25,
                 label="AlwaysPM QBER")
    axes[0].plot(tracks[1][1], tracks[1][3], color=MUTED, lw=1.1, alpha=0.9,
                 label="AlwaysENT QBER")
    axes[0].plot(tracks[2][1], tracks[2][3], color=TEAL, lw=1.25,
                 label="DecoyWatch QBER")
    axes[0].set_ylabel("QBER (%)")
    axes[0].set_title("QBER observation (does not reveal PNS)", loc="left",
                      color=NAVY, fontsize=10, fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")
    axes[0].grid(axis="y", color=GRID, linewidth=0.7)

    axes[1].fill_between(d, 0, pns, step="post", color=RED, alpha=0.30,
                         label="PNS strength s")
    axes[1].step(d, pns, where="post", color=RED, lw=1.1)
    axes[1].set_ylabel("PNS strength (%)")
    axes[1].set_ylim(-3, 70)
    axes[1].set_title("Hidden attack state (shown only for explanation)", loc="left",
                      color=NAVY, fontsize=10, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].grid(axis="y", color=GRID, linewidth=0.7)

    mode_color = {0: BLUE, 1: TEAL}
    for idx, (name, t, modes, _, _) in enumerate(tracks):
        y = 2 - idx
        for start in range(len(t)):
            axes[2].axvspan(t[start]-0.5, t[start]+0.5,
                            color=mode_color[int(modes[start])], alpha=0.85,
                            linewidth=0)
        axes[2].text(-0.012, (y+0.5)/3, name, transform=axes[2].transAxes,
                     ha="right", va="center", fontsize=8.5, color=INK)
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color=BLUE, label="PM"),
                            Patch(color=TEAL, label="ENT")],
                   frameon=False, fontsize=8, loc="upper right", ncol=2)
    axes[2].set_yticks([])
    axes[2].set_ylim(0, 3)
    axes[2].set_xlabel("Slot (1 s per slot)")
    axes[2].set_title("Mode selected by policy", loc="left", color=NAVY,
                      fontsize=10, fontweight="bold")
    axes[2].grid(axis="x", color=GRID, linewidth=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[-1].set_xlim(0, 3600)
    fig.text(0.08, 0.012,
             "Illustrative single run (seed 0), PNS-only, 55 km, λ = 4, n_X = 10⁵. "
             "Not an average; quantitative policy comparison is in Table 4.7.",
             fontsize=8.3, color=MUTED)
    fig.tight_layout(rect=(0.08, 0.055, 1, 0.95), h_pad=1.2)
    save(fig, "fig4_8_pns_episode.png")


def main():
    platform_flow()
    hardware_ratio()
    pns_duty()
    policy_map()
    episode_trace()


if __name__ == "__main__":
    main()
