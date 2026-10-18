# qkd_app — Hybrid QKD policy study

Numerical study of Hybrid QKD mode selection (PM/BB84-decoy vs ENT/BBM92) under a
jamming / photon-number-splitting adversary. The window is a **numeric model**
(`HybridEnv`, eq 1–24); it does not drive the real qline hardware.

The real stack (`hw_sim` + `gc` + `qber`) lives alongside this folder and is only
needed for `gui.py` / `collector.py`.

---

## 1. Requirements

* **WSL2 Ubuntu 20.04** (Linux-only: FIFOs, Unix sockets, `mmap`, crate `nix`)
  + Python 3.8.10, Rust 1.97 — already present in the lab WSL image
  + Python packages: `numpy`, `matplotlib`, `tkinter` (TkAgg), `scipy`
* On Windows, `tkinter`/WSLg is required for the two Tk windows.
* The sibling folder `../DecoyState` must be present (Rusca 2018 finite-key bound
  `compute_final_length`); `keyrate.py` imports it via a relative path. Keep the
  workspace layout:

```
e:\qkdsim\
  DecoyState/compute_final_length.py
  hw_sim/  kiwi_hw_control/           # VeriQloud repos (Rust)
  qkd_app/                            # ← this folder
```

No other dependency is introduced by this app — do not add one without asking.

---

## 2. Quick start — no hardware, no build

The most useful entry point needs **nothing built**:

```bash
cd /mnt/e/qkdsim/qkd_app          # or the cloned path

# a) the hybrid policy visualiser (numeric model, eq 1-24)
python3 policyviz.py              # window — press Apply, then Start
python3 policyviz.py --no-gui     # same episode as text scoreboard
python3 policyviz.py --no-gui --adversary pns --lam 5 --figure figs/x.png

# b) the study's answer "does switching pay", headless
python3 eval_hybrid.py --no-dqn --lam 2.5 --b-max 1e6 --seeds 4 --show-sweep
python3 eval_hybrid.py --ent-preset ent_advanced --lam 2.5 --b-max 1e6 --seeds 8
python3 optmap.py --adversary pns --lam 5 --seeds 4   # winner per (d_km, n_X) cell
```

The window's **Save as default** writes `policyviz_default.json` next to this README
(git-ignored); `--no-gui` deliberately never reads it so a printed scoreboard is
reproducible. View checkboxes (hide a trace, sieve/hide the buffer panel) are view-only
— `run.buffer` and every formula still run on the same episode.

---

## 3. Hardware stack — only for `gui.py` / `collector.py`

This is **not** needed for `policyviz.py`, `eval_hybrid.py` or `optmap.py`.

```bash
cd /mnt/e/qkdsim/qkd_app

bash build.sh          # ~2-3 min with JOBS=1 (9p/drvfs needs serial cargo)
bash run_stack.sh      # MODE=decoy (PM) by default; MODE=plain for ENT
python3 collector.py   # live QBER + sifted rate; --detail / --csv / --protocol ent
bash stop_stack.sh     # teardown (also kills Eve if active)

# The Tk control panel on top of the stack
python3 main.py
# emergency (GUI hung/crashed):  bash kill_all.sh
```

`build.sh` stages `bin/{simulator,gc_alice,gc_bob,qber_alice,qber_bob}` under `qkd_app/`;
`run_stack.sh` picks `hw_sim_decoy_config.json` (`MODE=decoy`) or `hw_sim_config.json`
(`MODE=plain`) — there are two copies per player, editing the wrong one has no effect.
Every `Apply` in the GUI rewrites that pair and restarts the stack (~2.3 s to the first
CSV line: `wait_ready.sh` gates on `/tmp/fpga_*` ≥ 77824 bytes and on `:58000`).

Configuration presets under `presets/` (e.g. `aurea`, `chapman_current`,
`ent_advanced`) are JSON overlays on the Rust configs. GUI edits are kept as dataclass
overrides in-memory and written through `configio.py`.

---

## 4. Repo layout

```
qkd_app/
  channel.py         env params → eta/qber/l_predict + death_qber
  keyrate.py         PM 1-decoy finite-key bridge (gain-modelled split)
  keyrate_ent.py     ENT/BBM92 finite-key bridge (Chapman 2022, d=2)
  hybridenv.py       eq 1-24 HybridEnv (queue/buffer/deadline, per-slot)
  policies.py        always-pm/ent, greedy, buffer-aware, block-aware, decoy, oracle
  evemarkov.py       3-state Eve Markov chain (jam + PNS, per adversary)
  eval_hybrid.py     headless sweep + comparison driver
  optmap.py          (d_km, n_X, adversary) winner map
  policyviz.py       VizConfig/VizSession + draw_panels (headless + figure export)
  policyviz_ui.py    Tk window (thin: form → VizSession → draw_panels)
  gui.py / backend.py / stackctl.py / configio.py  real stack plumbing
  collector.py       sifting + finite-key live extractor
  presets/*.json     operating points (pm/ent)
  tests/             1074 tests — PYTHONDONTWRITEBYTECODE=1 on /mnt/e
  figs/  logs/  bin/  build/   git-ignored
```

`qkd_app` is its own Git repo; the sibling `hw_sim` / `kiwi_hw_control` /
`DecoyState` repos are independent. The scratch scripts `_wsl_*`, `_poc_bridge.py`
and `_thesis_extract/` at the workspace root are not part of this app.

---

## 5. Tests

```bash
# on WSL
cd /mnt/e/qkdsim/qkd_app
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -q 2>&1 | tail -5
# expected:  Ran 1074 tests in ~180s  —  OK
# (single module:  python3 -m unittest tests.test_policyviz -q)
```

Run with `PYTHONDONTWRITEBYTECODE=1` on `/mnt/e` (drvfs/9p) — `__pycache__` mtimes are
coarse there and stale bytecode causes flaky failures after `git checkout`.

The scoreboard printed by `policyviz.py --no-gui` is pinned by
`tests/test_policyviz.py` (seed 99000000, adversary `pns`, lam=5).

---

## 6. Notes for the thesis figure

* The visualiser is **NUMERIC MODEL** — stated on screen. `gui.py` drives the real
  qline; `policyviz.py` drives only `HybridEnv`.
* `Oracle` / `always-pm-relaxed` are **upper bounds** (clairvoyant, free switching,
  continuous key delivery) — labelled as such, never counted as a "winner".
* Under PNS the QBER does not move; attack periods are drawn as red (PNS) / amber
  (jamming) bands so a flat QBER does not read as "nothing happened".

Spec of the underlying study:
`docs/superpowers/specs/2026-08-17-hybrid-policy-design.md`
