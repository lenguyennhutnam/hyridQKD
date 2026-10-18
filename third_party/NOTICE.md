# Third-party code

## `compute_final_length.py`

* **Upstream:** https://github.com/VeriQloud/DecoyState — file `compute_final_length.py`
* **Copied at:** `d177a75` (branch tip as cloned into the workspace)
* **What it is:** the Rusca 2018 one-decoy finite-key bound. `compute_final_length()`
  returns the number of secret bits a block of raw key yields after error
  correction and privacy amplification; `hbin()` is its binary entropy helper.
* **Why it is vendored:** this app must run from a clone of this repo alone. The
  bound is the physics of the whole study — it is not reimplemented here, because
  a reimplementation would be a second, silently-diverging copy of the same
  formula.
* **Modified before vendoring:** yes, in the upstream workspace copy, in two
  places, both bug fixes rather than changes to the bound itself:
  1. `compute_pap()` loops over however many afterpulse components a config
     declares. It previously hardcoded three and raised `IndexError` on
     two-component configs such as `params_aurea.json`.
  2. the `__main__` block honours the config's `disable_pap` flag. It was
     previously ignored.

  Both fixes are in the vendored file because the app needs them; neither
  touches `compute_final_length()` itself.
* **Licence:** the upstream repository ships no licence file, so its terms are
  those of the VeriQloud project rather than of a public open-source grant.
  Check with them before redistributing further. The citation the bound
  implements is:

  > A. Rusca et al., *Finite-key analysis of a decoy-state quantum key
  > distribution protocol*, Phys. Rev. A **97**, 032330 (2018).