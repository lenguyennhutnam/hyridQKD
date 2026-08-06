#!/usr/bin/env python3
"""Read/update hw_sim decoy config files for Alice and Bob.

Bob's eta/qberr drive the observed physics (Alice's copies are ignored by the
simulator), but seed/angles/decoy_states MUST be identical on both sides or the
preshared-seed correlation desyncs (measured: QBER jumps to 50%). We therefore
always write both files identically and re-verify after writing.
"""

import json
import os

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ALICE = os.path.join(_WORKSPACE, "hw_sim", "config_files", "alice",
                             "hw_sim_decoy_config.json")
DEFAULT_BOB = os.path.join(_WORKSPACE, "hw_sim", "config_files", "bob",
                           "hw_sim_decoy_config.json")

# Fields whose mismatch breaks Alice-Bob correlation.
_MATCH_KEYS = ("seed", "angles")


def read_config(path):
    """Load one hw_sim config file as a dict."""
    with open(path) as f:
        return json.load(f)


def check_matched(alice_cfg, bob_cfg):
    """Raise ValueError if the two configs would desync the preshared-seed RNG."""
    a, b = alice_cfg["backend_config"], bob_cfg["backend_config"]
    for key in _MATCH_KEYS:
        if a.get(key) != b.get(key):
            raise ValueError("alice/bob mismatch on %r: %r != %r"
                             % (key, a.get(key), b.get(key)))
    if ("decoy_states" in a) != ("decoy_states" in b):
        raise ValueError("decoy mode enabled on one side only")
    if a.get("decoy_states") != b.get("decoy_states"):
        raise ValueError("decoy_states differ: %r != %r"
                         % (a.get("decoy_states"), b.get("decoy_states")))


def update_hw_sim_configs(eta, qberr, mu1, mu2, p1,
                          alice_path=DEFAULT_ALICE, bob_path=DEFAULT_BOB):
    """Write eta/qberr/decoy_states into BOTH configs, preserving other fields.

    Validates in-memory before writing to prevent partial updates on error.
    """
    # Read both configs first
    alice_cfg = read_config(alice_path)
    bob_cfg = read_config(bob_path)

    # Apply updates in memory
    alice_cfg["backend_config"]["eta"] = eta
    alice_cfg["backend_config"]["qberr"] = qberr
    alice_cfg["backend_config"]["decoy_states"] = {"mu1": mu1, "mu2": mu2, "p1": p1}

    bob_cfg["backend_config"]["eta"] = eta
    bob_cfg["backend_config"]["qberr"] = qberr
    bob_cfg["backend_config"]["decoy_states"] = {"mu1": mu1, "mu2": mu2, "p1": p1}

    # Validate in memory before writing (fail-safe)
    check_matched(alice_cfg, bob_cfg)

    # Write both files only after validation passes
    for path, cfg in [(alice_path, alice_cfg), (bob_path, bob_cfg)]:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")

    # Final guard: verify after writing
    check_matched(read_config(alice_path), read_config(bob_path))
