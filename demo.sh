#!/usr/bin/env bash
# Reproduce the three defence-slide numbers with the numerical model.
#
#   bash demo.sh 1   # Slide 1  — Results under Jamming          (~4 min)
#   bash demo.sh 2   # Slide 2  — Results under the PNS Attack    (~5 s)
#   bash demo.sh 3   # Slide 3  — Block length + policy map       (minutes)
#   bash demo.sh 23  # Slides 2 and 3 without the slow one        (~1 min)
#   bash demo.sh all
#
# Run from WSL. Everything below drives `HybridEnv` (the eq 1-24 numeric
# model); no hw_sim process is started -- this is NOT the real QKD stack.
# Use gui.py for that.
#
# ---------------------------------------------------------------------
# THE TWO KINDS OF NUMBER IN THESE SLIDES, AND WHICH PRODUCES WHICH
# ---------------------------------------------------------------------
# The slides report 8-seed averages. `policyviz.py --no-gui` runs ONE episode
# (seed index 0), so its digits are NOT the slide digits -- they are there to
# be watched live, where the point is the ORDERING and the SHAPE, not the
# third decimal:
#
#   * `policyviz.py`    one episode, seconds  -> show the mechanism
#   * `eval_hybrid.py`  8-seed average,  ~4 min -> the slide numbers
#
# Slides 1 and 2 print both. Slide 3's fill-time column is arithmetic, not a
# measurement (see demo 3). Companion narration: demo_script_en.md
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
OUT=${DEMO_OUT:-figs}
PM=demo_pm
ENT=demo_ent_advanced

banner () {
  echo
  echo "=============================================================="
  echo "$1"
  echo "=============================================================="
}

# ---------------------------------------------------------------------
# Slide 1 -- Results under Jamming
# ---------------------------------------------------------------------
demo_1 () {
  banner "SLIDE 1 -- Jamming: BlockAwareGreedy's goodput"

  cat <<'TXT'
  LIVE (what to show on screen) -- one episode, seed 0:
    adversary=jam   lam=2.5   B_max=1e6   n_X=1e5   d=55 km
    ticks: always-pm, always-ent, greedy-0.045, block-aware, oracle
    un-tick: buffer-aware, decoy, always-pm-relaxed

  lam MUST be 2.5, not the window's default 4: at 4 the buffer never runs
  dry, every policy saturates and nothing separates.

  Expected ranking (watch for the order, not the digits):
    always-pm   ~0.678    (lowest -- stays in PM through every jam)
    greedy      ~0.700
    always-ent  ~0.704
    block-aware ~0.753    (highest realisable -- leaves PM inside amber bands)
    oracle      ~0.770    (UPPER BOUND, dashed, not achievable)

TXT

  $PY policyviz.py --no-gui --adversary jam --lam 2.5 --b-max 1000000 \
      --pm-preset "$PM" --ent-preset "$ENT" \
      --policies always-pm,always-ent,greedy-0.045,block-aware,oracle

  cat <<'TXT'

  SLIDE DIGITS -- 8-seed average, this is what belongs on the slide:
TXT
  $PY eval_hybrid.py --seeds 8 --no-dqn --lam 2.5 --b-max 1e6 \
      --pm-preset "$PM" --ent-preset "$ENT" --d-km 55

  cat <<'TXT'

  Reported on the slide (Table 4.4):
    always-pm 0.6284 | always-ent 0.6616 | best threshold 0.6628
    block-aware 0.7035 | oracle 0.7228
    vs always-pm   +11.9%     (0.7035/0.6284 - 1)
    vs always-ent  + 6.3%     (0.7035/0.6616 - 1)
    headroom taken  79.6%     (0.7035-0.6284)/(0.7228-0.6284)

  The eval_hybrid call above is slow (~4 min): run it before the defence
  and paste its table into the slide. Only the policyviz block above is
  meant to run live.
TXT
}

# ---------------------------------------------------------------------
# Slide 2 -- Results under the PNS Attack
# ---------------------------------------------------------------------
demo_2 () {
  banner "SLIDE 2 -- PNS attack: why QBER-based controllers are blind"

  cat <<'TXT'
  LIVE -- the window's defaults already are this configuration:
    adversary=pns   lam=4   B_max=1e6   n_X=1e5   d=55 km
    ticks: always-pm, always-ent, greedy-0.045, block-aware,
           decoy-0.100-1.000-60, oracle

  TWO THINGS TO POINT AT ON SCREEN:

    1. "Why QBER is not enough" -- the QBER panel stays flat at ~1% straight
       through the red attack bands, because PNS removes key without adding
       errors. always-pm, greedy and block-aware sit on one identical line.

    2. "Why DecoyWatch works" -- in the mode lane, only decoy-0.100 leaves PM
       and runs ENT inside the red bands. It watches the decoy gain ratio,
       and its QBER alarm is set to 1.000, i.e. OFF: it reacts to the decoy
       ratio alone.

TXT

  mkdir -p "$OUT"
  $PY policyviz.py --no-gui --adversary pns --lam 4 --b-max 1000000 \
      --pm-preset "$PM" --ent-preset "$ENT" \
      --policies always-pm,always-ent,greedy-0.045,block-aware,decoy-0.100-1.000-60,oracle \
      --figure "$OUT/demo_pns.png"
  echo
  echo "wrote $OUT/demo_pns.png"

  cat <<'TXT'

  Expected ranking (seed 0 -- ordering is the message, digits are not the
  slide's):
    always-pm / greedy / block-aware  0.5836   (bit-for-bit identical)
    always-ent                       0.6383
    decoy-0.100-1.000-60             0.6807   <- the only line that moves
    oracle (upper bound)             0.7060

  The duty-cycle table on that slide (Table 4.7: 2 / 5 / 10 / 20 / 40 %) is
  a SWEEP over the adversary's Markov chain. The window has no control for
  it, so that table is produced by code, not shown live. The break-even duty
  where always-pm and always-ent cross is ~10% in simulation and d* = 10.9%
  analytically (eq 3.1); at 10% duty the 8-seed figures are
  PM 0.4976 / ENT 0.5034 / DecoyWatch 0.5476 / Oracle 0.5613 (+8.8% over the
  better fixed mode).

TXT
}

# ---------------------------------------------------------------------
# Slide 3 -- Block length + policy map
# ---------------------------------------------------------------------
demo_3 () {
  banner "SLIDE 3 -- Finite-key block length, and the policy map"

  cat <<'TXT'
  The left-hand table is ARITHMETIC, not a measurement:

    fill_time = n_X / ns_PM      with ns_PM ~ 69 422 sifted bits per slot
                                  at d = 55 km on demo_pm

    1e5 / 69422 =  1.4 slots  ->  supply-limited   (goodput ~ proportional
    3e5 / 69422 =  4.3 slots  ->  transition         to key distilled)
    1e6 / 69422 = 14.4 slots  ->  timing-limited    (deadline T0 = 20 slots)
    3e6 / 69422 = 43.2 slots  ->  timing-dominated

  To SHOW one of those cells live, change ONE box and press Apply:
    n_X block [bits] = 1000000   (timing-limited cell), everything else as
    slide 2. Watch the key buffer arrive in large late chunks instead of
    smoothly, and note that each PM probe now discards a much bigger partial
    block (eq 5) -- which is why the decoy controller's advantage shrinks as
    the block grows.

TXT

  $PY policyviz.py --no-gui --adversary pns --lam 4 --b-max 1000000 \
      --pm-preset "$PM" --ent-preset "$ENT" \
      --policies always-pm,always-ent,decoy-0.100-1.000-60,oracle

  cat <<'TXT'

  The policy map (Table 4.9) is a grid the window cannot draw -- it runs one
  seed at one (distance, block) -- so it is produced by optmap.py. This is a
  full grid: run it BEFORE the defence, do not run it live.

TXT

  $PY optmap.py --seeds 4 --adversary pns --lam 5 \
      --d-grid 20,55,85 --n-x-grid 1e5,3e5,1e6,3e6 \
      --ent-preset "$ENT" --pm-preset "$PM"

  cat <<'TXT'

  What the map should show (Table 4.9):
    55 km and 85 km, n_X up to 1e6 -> decoy-0.100-1.000-60 wins
    20 km (saturated: ENT alone supplies enough key) -> always-ent
    n_X = 3e6 (blocks so large that probing is too expensive) -> always-ent

  Reading of the slide: no single policy is always optimal -- the right choice
  depends on link distance, finite-key block length and the attack. Probing
  every 60 slots wins at small blocks and probing never wins at large ones,
  because each probe throws away a partial ENT block.
TXT
}

case "${1:-all}" in
  1)    demo_1 ;;
  2)    demo_2 ;;
  3)    demo_3 ;;
  23)   demo_2; demo_3 ;;
  all)  demo_1; demo_2; demo_3 ;;
  *)    echo "usage: bash demo.sh [1|2|3|23|all]" >&2; exit 2 ;;
esac
