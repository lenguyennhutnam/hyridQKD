#!/usr/bin/env python3
"""Persist the whole control-panel state as the user's "default".

One snapshot of everything the left-hand form holds: the protocol radio, all
three parameter regions (shared / PM / ENT), the finite-key block size and the
key-buffer capacity. That spans BOTH protocol dataclasses, which is why it
cannot live in presets.py -- a preset is one protocol's parameters, this is a
picture of the panel.

File format (qkd_app/panel_default.json):

    {"protocol": "pm", "block": 10000000, "b_max": 10000000,
     "fields": {"d_km": 0.0, "alpha": 0.23, ..., "ent_mu": 0.045}}

LOADING NEVER RAISES -- deliberately the opposite of presets.py. Presets are
hand-written, so a typo there must fail loudly. This file is machine-written
and read during startup, so a corrupt or outdated copy must degrade to "no
default" rather than stop the app from opening. Anything unrecognised is
dropped; anything missing is simply absent from the returned dict, and the
caller keeps whatever the form already shows.

Every function resolves DEFAULT_PATH at CALL time (path=None) rather than
binding it as a default argument, so a test can redirect the module constant
and never touch a developer's real panel_default.json.
"""

import json
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(APP_DIR, "panel_default.json")

# Reserved so the preset dropdown can never show two entries with this name.
RESERVED_NAME = "default"

_PROTOCOLS = ("pm", "ent")


def save_default(state, path=None):
    """Write a panel-state dict; returns the path written.

    The caller is responsible for having parsed/validated the form -- this
    only serialises. Unlike load_default(), errors here DO propagate: a failed
    save must be visible, not silently swallowed.
    """
    path = path or DEFAULT_PATH
    with open(path, "w") as f:
        json.dump(state, f, indent=4, sort_keys=True)
        f.write("\n")
    return path


def default_exists(path=None):
    """True if a default has been saved (used to decide whether the preset
    dropdown offers the "default" entry)."""
    return os.path.exists(path or DEFAULT_PATH)


def _clean_positive_int(value):
    """Return int(value) if it is a positive integer-ish number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return int(value)


def load_default(path=None):
    """Load the saved panel state, or None if there isn't a usable one.

    Returns a dict with only the keys that survived validation: "fields"
    (always present, possibly empty) plus any of "protocol"/"block"/"b_max"
    that were valid. Absent keys mean "leave the form as it is".
    """
    try:
        with open(path or DEFAULT_PATH) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    fields = {}
    raw_fields = raw.get("fields")
    if isinstance(raw_fields, dict):
        for name, value in raw_fields.items():
            # bool is an int subclass; a stray true/false is not a parameter.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            fields[name] = float(value)

    out = {"fields": fields}
    if raw.get("protocol") in _PROTOCOLS:
        out["protocol"] = raw["protocol"]
    for key in ("block", "b_max"):
        cleaned = _clean_positive_int(raw.get(key))
        if cleaned is not None:
            out[key] = cleaned
    return out
