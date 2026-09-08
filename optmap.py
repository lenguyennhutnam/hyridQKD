#!/usr/bin/env python3
"""The optimisation map: which policy wins at which channel and block length.

The thesis question is "optimise the transmission channel and the key length in
a hybrid QKD model under a jamming attack", and those are literally the two
axes here: `d_km` (the channel) and `n_X` (the finite-key block), swept against
each other while Eve's Markov chain runs. Each cell reports the best REALISABLE
policy, what it gained over always-pm, and how much of the Oracle's headroom it
left behind.

WHY THE MAP AND NOT A SINGLE OPERATING POINT
--------------------------------------------
Every conclusion the study has published so far is a statement about one cell
of this grid (d = 55 km, n_X = 1e5), and two of them have already turned out to
be properties of the cell rather than of the problem:

  * "switching does not pay" is true at chapman_lowmu and false at
    ent_advanced -- a hardware axis;
  * "eta is exactly proportional to the total key distilled" is true at
    n_X = 1e5 and false by n_X = 1e6 -- a block-length axis, and it decides
    whether the queue and the deadline do anything at all.

A map is the only honest form for a result whose sign depends on where it is
measured.

WHAT A CELL IS ALLOWED TO CLAIM
-------------------------------
`winner` is the best row that a real controller could run: the two reference
bounds are excluded by name, because the Oracle wins every cell by construction
and a map reporting it would be a map of something nobody can build.

`margin` is measured against always-pm, the control the rest of the study is
written against, so a cell here can be read next to any table already
published. `switching_value` is measured against always-pm-RELAXED, for the
reason CLAUDE.md records having got wrong once: oracle - always-pm mixes the
value of the relaxation into the value of switching.

The parameterised policies are swept per cell, so every row is a best-case
in-sample number -- the same caveat the study's own tables carry.

REGIME IS MEASURED, NOT ASSUMED
-------------------------------
Three regimes, read off the control row's STARVATION -- the fraction of slots
whose buffer held less than one message's worth of key:

  supply     the buffer is empty essentially always. eta is then a linear
             function of the total key distilled and no scheduling decision
             can matter; the problem degenerates to "maximise total key".
  timing     key comes and goes, so WHEN a block lands against the deadline
             decides. This is the only regime in which the queue and the
             deadline of eq (2)-(4) do any work at all.
  saturated  key is nearly always in hand and eta is at its ceiling; more key
             buys almost nothing.

`coupling_spread` -- the spread of (eta/eta_pm)/(key/key_pm) across policies
-- is reported beside it as the corroborating evidence, and in supply cells it
comes out at 0.0001-0.0007. It is NOT what decides the label, and an earlier
version of this module that used it alone mislabelled every saturated cell as
"timing": a saturated cell spreads too, because eta is pinned at its ceiling
while the key keeps growing. Starvation is the mechanism, the spread is the
symptom, and the symptom has two causes.

The block fill times are reported as well, because they are what EXPLAINS the
starvation -- but deciding from them would be a guess: n_X = 1e6 fills in 14.4
slots against a T0 of 20 and is already timing-limited.

Usage:
    python3 optmap.py --seeds 4                       # the default grid
    python3 optmap.py --quiet-eve                     # the no-jamming contrast
    python3 optmap.py --d-grid 55,170 --n-x-grid 1e5,1e6 --csv figs/map.csv
"""

import argparse
import csv
import dataclasses
import os
import sys

from evemarkov import EveParams
from eval_hybrid import (BLOCK_HORIZONS, compare, evaluation_seeds)
from hybridenv import HybridEnv, HybridParams
from policies import AlwaysPM, Oracle, RelaxedAlwaysPM

__all__ = ["BOUND_NAMES", "CONTROL_NAME", "SUPPLY_SPREAD", "D_GRID",
           "N_X_GRID", "MAP_THRESHOLDS", "MAP_RESERVES", "MAP_SHAPES",
           "map_switches", "quiet_eve", "coupling_spread", "summarise_cell",
           "run_cell",
           "sweep_map", "format_map", "write_rows_csv", "main"]

# Rows that are bounds rather than policies. Taken from the classes, not
# retyped: a renamed bound must not silently become eligible to win a cell.
BOUND_NAMES = (RelaxedAlwaysPM.name, Oracle.name)
CONTROL_NAME = AlwaysPM.name

# How far the per-policy ratio (eta/eta_pm)/(key/key_pm) may spread before a
# cell stops counting as supply-limited. Measured at ent_advanced, lam = 2.5:
# 0.0007 at n_X = 1e5, 0.0485 at 3e5, 0.098 at 1e6. The threshold sits in the
# gap between the first two rather than at a round number chosen in advance.
SUPPLY_SPREAD = 0.01

# The regime is read off the CONTROL row's starvation -- the fraction of slots
# whose buffer held less than one message's worth of key. That is the
# MECHANISM; the coupling spread above is its symptom, and the symptom has a
# second cause: a SATURATED cell also spreads, because eta is pinned at its
# ceiling while the key keeps growing. Measured over the 24 cells of the
# d = 20..100 km map, the three clusters are well separated and these two
# cut-offs sit in the gaps between them:
#
#   starved 0.95 - 1.00  supply    -- the buffer is empty essentially always,
#                                     so only the TOTAL key matters
#                                     (coupling spread 0.0001 - 0.0007)
#   starved 0.27 - 0.88  timing    -- key comes and goes, so WHEN it lands
#                                     decides (spread 0.055 - 0.19)
#   starved 0.07 - 0.20  saturated -- key is nearly always in hand; more of it
#                                     buys almost nothing (spread 0.06 - 0.12,
#                                     for the wrong reason)
SUPPLY_STARVED = 0.90
SATURATED_STARVED = 0.25

# The default axes.
#
# d_km spans PM's death distance: at ent_advanced with both modes on the same
# detector PM stops distilling at 199.8 km and ENT at 247.3 km, so the grid has
# to have points on both sides of 200 or the ENT-only band -- the one place
# where mode choice is forced rather than optimised -- never appears.
D_GRID = (55.0, 120.0, 170.0, 195.0, 220.0)
# n_X spans the two regimes: 1e5 is where the study's published numbers live
# and eta is proportional to the key; by 1e6 the block fill time is comparable
# with T0 and the ranking changes.
N_X_GRID = (100_000, 300_000, 1_000_000, 3_000_000)

# Sweep grids for a MAP rather than for a single reported point. Coarser than
# eval_hybrid's own defaults on purpose: a cell costs one full compare(), and
# the map's claim is "which policy wins here", which a 3-point threshold grid
# settles about as well as a 12-point one. A headline number for one cell
# should still be taken from eval_hybrid.
MAP_THRESHOLDS = (0.02, 0.04, 0.06)
MAP_RESERVES = (0.0, 0.25)
MAP_SHAPES = ((0.0, 0), (0.010, 0), (0.010, 20))


def map_switches(d_km):
    """Switch distances for the range policies at a FIXED link of `d_km`.

    eval_hybrid builds this grid from the patrol band, which is right when a
    vessel is moving: a switch distance only means something relative to where
    it actually goes. The map runs a fixed link, where every RangeThreshold
    collapses to always-pm (switch beyond the link) or always-ent (switch at
    or before it) -- two behaviours, whatever the grid size. Nine points, and
    their twenty-seven-point product with the thresholds, would be over half
    the cost of a cell for nothing.
    """
    return (0.5 * float(d_km), 1.5 * float(d_km))


def quiet_eve(eve=None):
    """The same adversary with every jamming level set to zero.

    The contrast case for "under a jamming attack" has to be no jamming at
    all, not a gentler HEAVY level -- and only the LEVELS are zeroed, so the
    chain's dwell structure is untouched and the two maps differ in exactly
    one thing.
    """
    eve = eve or EveParams()
    return dataclasses.replace(eve, f_levels=tuple(0.0 for _ in eve.f_levels))


def coupling_spread(rows, control=CONTROL_NAME):
    """How far eta departs from being a linear function of the total key.

    Returns max - min of (eta/eta_ctl) / (key/key_ctl) over the rows that
    delivered anything. 0 means every policy converted key into served
    messages at exactly the same exchange rate, which is what a supply-limited
    cell looks like: nothing a policy does with TIMING shows up in eta.

    Returns None when the control delivered no key or no messages, because
    then there is no exchange rate to compare against.
    """
    by = {r["policy"]: r for r in rows}
    if control not in by:
        raise KeyError("no %r row to normalise against" % (control,))
    base_eta = float(by[control]["eta"])
    base_key = float(by[control]["secret_bits"])
    if base_eta <= 0.0 or base_key <= 0.0:
        return None
    ratios = [(float(r["eta"]) / base_eta) / (float(r["secret_bits"]) / base_key)
              for r in rows
              if float(r["secret_bits"]) > 0.0 and float(r["eta"]) > 0.0]
    if len(ratios) < 2:
        return None
    return max(ratios) - min(ratios)


def _regime(eta_best, starved):
    """Which of the three regimes a cell is in, from the control's starvation.

    "unknown" when the control carries no starvation figure at all, rather
    than a guess: a mislabelled cell is worse than an unlabelled one, and
    this is the column a reader uses to decide whether the scheduling half of
    the model did anything.
    """
    if eta_best <= 0.0:
        return "dead"
    if starved is None:
        return "unknown"
    starved = float(starved)
    if starved >= SUPPLY_STARVED:
        return "supply"
    if starved <= SATURATED_STARVED:
        return "saturated"
    return "timing"


def summarise_cell(d_km, n_x, rows, fill, t0_deadline=None):
    """One cell of the map: the winner, the margins, and what explains them.

    `fill` is (n_X/ns_PM, n_X/ns_ENT) in slots -- reported, never used to
    decide the regime; see the module docstring.

    `t0_deadline` is carried through for the reader (a fill time only means
    something next to the deadline it is racing) and is not part of any
    decision here.
    """
    by = {r["policy"]: r for r in rows}
    if CONTROL_NAME not in by:
        raise KeyError("cell at d=%s n_X=%s has no %r row"
                       % (d_km, n_x, CONTROL_NAME))
    realisable = [r for r in rows if r["policy"] not in BOUND_NAMES]
    best = max(realisable, key=lambda r: r["eta"])
    eta_pm = float(by[CONTROL_NAME]["eta"])
    relaxed = by.get(RelaxedAlwaysPM.name)
    oracle = by.get(Oracle.name)
    eta_relaxed = float(relaxed["eta"]) if relaxed else None
    eta_oracle = float(oracle["eta"]) if oracle else None

    headroom = (eta_oracle - eta_pm) if eta_oracle is not None else None
    margin = float(best["eta"]) - eta_pm
    spread = coupling_spread(rows)
    starved = by[CONTROL_NAME].get("starved")
    return {
        "d_km": float(d_km),
        "n_x": int(n_x),
        "rows": list(rows),
        "winner": best["policy"],
        "eta_winner": float(best["eta"]),
        "eta_pm": eta_pm,
        "margin": margin,
        "switches": float(best.get("switches", 0.0)),
        "ent_fraction": float(best.get("ent_fraction", 0.0)),
        "relaxation_value": (None if eta_relaxed is None
                             else eta_relaxed - eta_pm),
        "switching_value": (None if eta_oracle is None or eta_relaxed is None
                            else eta_oracle - eta_relaxed),
        "headroom": headroom,
        "headroom_left": (None if eta_oracle is None
                          else eta_oracle - float(best["eta"])),
        # 0.0 rather than None when there was no headroom: "the winner took
        # none of nothing" is the honest reading, and it keeps the column
        # sortable.
        "headroom_taken": (0.0 if not headroom else margin / headroom),
        "fill_pm": float(fill[0]),
        "fill_ent": float(fill[1]),
        "t0_deadline": t0_deadline,
        "coupling_spread": spread,
        "starved": None if starved is None else float(starved),
        "regime": _regime(float(best["eta"]), starved),
        "dead": float(best["eta"]) <= 0.0,
    }


def run_cell(d_km, n_x, seeds, params=None, eve=None, thresholds=None,
             reserves=None, horizons=None, shapes=None, switches=None):
    """Score every policy at one (d_km, n_X) and summarise the result.

    Goes through eval_hybrid.compare() rather than scoring here, so the map
    and the study's own tables cannot disagree about what a policy scored --
    including the two bounds, which compare() always appends.
    """
    params = dataclasses.replace(params or HybridParams(),
                                 d_km=float(d_km), n_x=int(n_x))
    eve = eve or EveParams()
    # One probe environment for the fill times. It costs a channel.derive
    # pair, which compare() is about to pay len(seeds) times anyway.
    probe = HybridEnv(params, eve)
    ns_pm, ns_ent = probe.sifted_per_slot
    fill = (params.n_x / ns_pm if ns_pm else float("inf"),
            params.n_x / ns_ent if ns_ent else float("inf"))
    rows = compare(seeds=seeds, params=params, eve_params=eve, dqn_config={},
                   switches=(map_switches(params.d_km) if switches is None
                             else switches),
                   thresholds=MAP_THRESHOLDS if thresholds is None
                   else thresholds,
                   reserves=MAP_RESERVES if reserves is None else reserves,
                   horizons=BLOCK_HORIZONS if horizons is None else horizons,
                   shapes=MAP_SHAPES if shapes is None else shapes)
    return summarise_cell(d_km=params.d_km, n_x=params.n_x, rows=rows,
                          fill=fill, t0_deadline=params.t0_deadline)


def sweep_map(d_grid=D_GRID, n_x_grid=N_X_GRID, seeds=(1, 2, 3),
              cell_runner=None, progress=None, **cell_kwargs):
    """Every (d_km, n_X) of the product grid, in grid order.

    `cell_runner` is the injection point the tests drive: the real one costs a
    full compare() per cell, and a map that skipped a point would otherwise
    only be visible as a hole in a figure.
    """
    runner = cell_runner or run_cell
    cells = []
    for d_km in d_grid:
        for n_x in n_x_grid:
            if progress is not None:
                progress(d_km, n_x)
            cells.append(runner(d_km=d_km, n_x=int(n_x), seeds=seeds,
                                **cell_kwargs))
    return cells


def _grid_table(cells, title, value, width=16):
    """Render one quantity over the (d_km down, n_X across) grid."""
    ds = sorted({c["d_km"] for c in cells})
    xs = sorted({c["n_x"] for c in cells})
    at = {(c["d_km"], c["n_x"]): c for c in cells}
    out = [title, "%8s" % "d_km \\ n_X"
           + "".join("%*s" % (width, "%.3g" % x) for x in xs)]
    for d in ds:
        line = "%8.1f" % d
        for x in xs:
            cell = at.get((d, x))
            line += "%*s" % (width, "-" if cell is None else value(cell))
        out.append(line)
    return "\n".join(out)


def format_map(cells):
    """The map as text: winners, margins, and the per-cell detail below."""
    if not cells:
        return "(empty map)"
    blocks = [
        _grid_table(cells, "WINNER (best realisable policy; bounds excluded)",
                    lambda c: c["winner"], width=18),
        "",
        _grid_table(cells, "MARGIN over always-pm",
                    lambda c: "%+.4f" % c["margin"], width=12),
        "",
        _grid_table(cells, "REGIME (measured: how often was the buffer empty?)",
                    lambda c: c["regime"], width=12),
        "",
        _grid_table(cells, "HEADROOM TAKEN (margin / (oracle - always-pm))",
                    lambda c: ("n/a" if c["headroom"] is None
                               else "%.0f%%" % (100 * c["headroom_taken"])),
                    width=12),
        "",
        "%8s %10s %9s %9s %8s %9s %9s %9s %9s"
        % ("d_km", "n_X", "fill_PM", "fill_ENT", "eta_pm", "eta_win",
           "margin", "switchval", "left"),
    ]
    for c in sorted(cells, key=lambda c: (c["d_km"], c["n_x"])):
        blocks.append(
            "%8.1f %10.3g %9.1f %9.1f %8.4f %9.4f %+9.4f %9s %9s"
            % (c["d_km"], c["n_x"], c["fill_pm"], c["fill_ent"], c["eta_pm"],
               c["eta_winner"], c["margin"],
               "n/a" if c["switching_value"] is None
               else "%+.4f" % c["switching_value"],
               "n/a" if c["headroom_left"] is None
               else "%+.4f" % c["headroom_left"]))
    return "\n".join(blocks)


CSV_FIELDS = ("d_km", "n_x", "regime", "fill_pm", "fill_ent",
              "coupling_spread", "policy", "is_winner", "is_bound", "eta",
              "switches", "expired", "secret_bits", "ent_fraction", "starved",
              "mean_d_km")


def write_rows_csv(cells, path):
    """Every policy row of every cell, one line each.

    The whole table rather than the winners: a map is a summary, and the
    question a reader asks next is always "by how much did it beat the one I
    would have picked", which needs the losing rows too.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for cell in cells:
            for row in cell["rows"]:
                writer.writerow({
                    "d_km": cell["d_km"], "n_x": cell["n_x"],
                    "regime": cell["regime"], "fill_pm": cell["fill_pm"],
                    "fill_ent": cell["fill_ent"],
                    "coupling_spread": cell["coupling_spread"],
                    "policy": row["policy"],
                    "is_winner": int(row["policy"] == cell["winner"]),
                    "is_bound": int(row["policy"] in BOUND_NAMES),
                    "eta": row["eta"], "switches": row.get("switches"),
                    "expired": row.get("expired"),
                    "secret_bits": row.get("secret_bits"),
                    "ent_fraction": row.get("ent_fraction"),
                    "starved": row.get("starved"),
                    "mean_d_km": row.get("mean_d_km")})
    return path


def _floats(text):
    return tuple(float(part) for part in text.split(",") if part.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=4,
                    help="evaluation episodes per policy per cell")
    ap.add_argument("--horizon", type=int, default=3600)
    ap.add_argument("--lam", type=float, default=2.5)
    ap.add_argument("--b-max", type=float, default=1e6)
    ap.add_argument("--t0", type=int, default=None)
    ap.add_argument("--ent-preset", default="ent_advanced")
    ap.add_argument("--pm-preset", default=None)
    ap.add_argument("--d-grid", default=None,
                    help="comma-separated link lengths [km]")
    ap.add_argument("--n-x-grid", default=None,
                    help="comma-separated block sizes, e.g. 1e5,1e6")
    ap.add_argument("--quiet-eve", action="store_true",
                    help="run the contrast case with every jamming level at "
                         "zero; the chain itself is unchanged")
    ap.add_argument("--csv", default=None,
                    help="write every policy row of every cell here")
    args = ap.parse_args(argv)

    overrides = {"horizon": args.horizon, "lam": args.lam,
                 "b_max": int(args.b_max), "ent_preset": args.ent_preset}
    if args.t0 is not None:
        overrides["t0_deadline"] = args.t0
    if args.pm_preset is not None:
        overrides["pm_preset"] = args.pm_preset
    params = HybridParams(**overrides)
    eve = quiet_eve() if args.quiet_eve else EveParams()
    seeds = evaluation_seeds(args.seeds)

    d_grid = D_GRID if args.d_grid is None else _floats(args.d_grid)
    n_x_grid = (N_X_GRID if args.n_x_grid is None
                else tuple(int(v) for v in _floats(args.n_x_grid)))

    print("presets: pm=%s ent=%s | lam=%.3f B_max=%.3g T0=%d horizon=%d"
          % (params.pm_preset, params.ent_preset, params.lam, params.b_max,
             params.t0_deadline, params.horizon))
    print("jamming: %s" % ("OFF (contrast case)" if args.quiet_eve
                           else "f_levels=%s" % (eve.f_levels,)))
    print("grid: %d distances x %d block sizes = %d cells, %d seeds each"
          % (len(d_grid), len(n_x_grid), len(d_grid) * len(n_x_grid),
             len(seeds)))

    def progress(d_km, n_x):
        sys.stderr.write("  cell d=%.1f km n_X=%.3g ...\n" % (d_km, n_x))
        sys.stderr.flush()

    cells = sweep_map(d_grid=d_grid, n_x_grid=n_x_grid, seeds=seeds,
                      progress=progress, params=params, eve=eve)
    print()
    print(format_map(cells))
    if args.csv:
        print("\nrows written to %s" % write_rows_csv(cells, args.csv))
    return cells


if __name__ == "__main__":
    main()
