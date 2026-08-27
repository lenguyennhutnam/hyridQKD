#!/usr/bin/env python3
"""Persist the POLICY VISUALISER's panel as the user's saved configuration.

The problem this solves: the visualiser's operating points come from
presets/*.json, so the only way to keep "aurea but with mu1 = 0.31" used to be
to invent a preset name for it. Everything else on the form -- the ticked
policies, the thresholds, lambda, the horizon, which traces are switched off --
had no home at all. This module gives the whole panel ONE file, written by a
Save button and read back at startup.

File format (qkd_app/policyviz_default.json):

    {"fields": {"lam": 2.5, "horizon": 3600, ...},
     "policies": ["always-pm", "greedy", "oracle"],
     "theta": 0.045, "ba_theta": 0.045, "ba_reserve": 0.25,
     "pm_preset": "aurea",  "pm_env": null,
     "ent_preset": "ent_advanced", "ent_env": {"mu": 0.031, ..., "protocol": "ent"},
     "hidden_traces": ["always-ent"], "speed": 20}

WHY THIS IS NOT panelstate.py
-----------------------------
`panelstate` holds `gui.py`'s form: one protocol radio, three parameter
regions, a block size and a buffer cap. This window holds a different thing
entirely -- a policy selection, two INDEPENDENT operating points that are live
at the same time, and a set of hidden traces. Sharing one file between them
would mean one panel's Save silently rewriting the other panel's startup
state, so they have separate files and separate modules. The two DECISIONS
below are copied from panelstate deliberately; the schema is not.

`pm_env`/`ent_env` ARE TRI-STATE, and all three states matter:

    absent   -- nothing was saved about this mode; the form keeps what it has.
    null     -- the mode is running its named preset AS THE FILE SHIPS IT.
                Later edits to presets/aurea.json are picked up.
    a dict   -- the numbers were edited away from the preset ("custom").

Writing a dict where null belongs freezes a copy of the preset into the saved
state and quietly stops tracking the preset file; writing the preset NAME
where a dict belongs throws the user's edit away. Both round-trip without
error and both are wrong, which is why `save_state` serialises the dataclass
it is handed rather than a name, and why `load_state` distinguishes an absent
key from a null one.

LOADING NEVER RAISES -- deliberately the opposite of presets.py. A preset is
hand-written, so a typo there must fail loudly. This file is machine-written
and read during startup, so a corrupt, truncated or outdated copy must degrade
to "nothing saved" rather than stop the window from opening. Anything
unrecognised is dropped and anything missing is simply absent from the
returned dict, which the caller reads as "leave that control alone".

Every function resolves DEFAULT_PATH at CALL time (path=None) rather than
binding it as a default argument, so a test can redirect the module constant
and never touch a developer's real policyviz_default.json. panelstate's
docstring records that trap being hit for real.
"""

import dataclasses
import json
import math
import os

import channel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
# NOT panel_default.json: that name belongs to gui.py's panel (panelstate.py)
# and the two panels hold completely different state.
DEFAULT_PATH = os.path.join(APP_DIR, "policyviz_default.json")

_CLASSES = {"pm": channel.EnvParams, "ent": channel.EntEnvParams}

# saved key -> which mode's dataclass it must deserialise as.
ENV_KEYS = {"pm_env": "pm", "ent_env": "ent"}
_PRESET_KEYS = ("pm_preset", "ent_preset")
# The three policy-parameter spinboxes, saved as numbers rather than as the
# text in the box: a saved file holding "0.o45" would restore a form that
# cannot be applied.
_NUMBER_KEYS = ("theta", "ba_theta", "ba_reserve")


def _is_number(value):
    """True for a real, finite JSON number. bool is an int subclass, and a
    stray true/false is not a parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _clean_positive_int(value):
    """int(value) if it is a positive integer-ish number, else None."""
    if not _is_number(value) or value <= 0:
        return None
    return int(value)


def env_to_dict(env):
    """One operating point as JSON: a dict, or None when it has no edits.

    Tagged with "protocol" exactly as `presets.save_preset` tags a preset
    file, so a reader can tell an ENT point from a PM one from the CONTENT
    instead of trusting the key it happened to be filed under.

    None is passed straight through: it is the "this mode tracks its preset
    file" state, not a missing value (see the module docstring).
    """
    if env is None:
        return None
    data = {name: float(value)
            for name, value in dataclasses.asdict(env).items()}
    data["protocol"] = "ent" if isinstance(env, channel.EntEnvParams) else "pm"
    return data


def env_from_dict(protocol, data):
    """Rebuild one operating point, or None if `data` is not usable as one.

    A field that cannot be read discards the WHOLE operating point rather
    than just that field. Dropping one field would substitute the dataclass
    default for a number the user saved -- different physics, running under
    the label the user chose, with nothing on screen to say so. Returning
    None falls back to the named preset instead, which the mode button
    reports honestly (it says "aurea", not "custom").
    """
    cls = _CLASSES.get(str(protocol))
    if cls is None or not isinstance(data, dict):
        return None
    if data.get("protocol", protocol) != protocol:
        # Filed under pm_env but tagged ent (or the reverse): whichever of
        # the two is right, the pair cannot both be, so trust neither.
        return None
    known = {f.name for f in dataclasses.fields(cls)}
    values = {}
    for name, value in data.items():
        if name == "protocol":
            continue
        if name not in known:
            continue            # a field the dataclass has since dropped
        if not _is_number(value):
            return None
        values[name] = float(value)
    try:
        return cls(**values)
    except TypeError:           # pragma: no cover - `known` already filters
        return None


def save_state(state, path=None):
    """Write a panel-state dict; returns the path written.

    The caller has already parsed the form -- this only serialises -- but the
    two operating points are handed over AS DATACLASSES (or None) and
    serialised here, so `save_state` and `load_state` are each other's inverse
    and a round-trip test is meaningful.

    Unlike load_state(), errors here DO propagate: a save that failed must be
    visible, not silently swallowed.
    """
    payload = dict(state)
    for key in ENV_KEYS:
        if key in payload:
            payload[key] = env_to_dict(payload[key])
    path = path or DEFAULT_PATH
    with open(path, "w") as f:
        json.dump(payload, f, indent=4, sort_keys=True)
        f.write("\n")
    return path


def state_exists(path=None):
    """True once a configuration has been saved."""
    return os.path.exists(path or DEFAULT_PATH)


def load_state(path=None):
    """The saved panel state, or None if there is no usable one.

    Returns a dict holding only the keys that survived validation. "fields"
    is always present (possibly empty); everything else is present only if it
    was saved and readable. An ABSENT key means "leave that control as it is",
    which is what lets a file written before a control existed still load.
    """
    try:
        with open(path or DEFAULT_PATH) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    out = {"fields": {}}
    raw_fields = raw.get("fields")
    if isinstance(raw_fields, dict):
        for name, value in raw_fields.items():
            if _is_number(value):
                # int stays int: the horizon box should read "3600", not
                # "3600.0", when the window fills it back in.
                out["fields"][name] = value

    raw_policies = raw.get("policies")
    if isinstance(raw_policies, list):
        # Names are NOT checked against the window's checkbox list here: a
        # spec this version has never heard of simply never matches a
        # checkbox, which is the same forward-compatibility rule the fields
        # follow. An empty selection cannot be applied, so it is dropped
        # rather than restored.
        chosen = [p.strip() for p in raw_policies
                  if isinstance(p, str) and p.strip()]
        if chosen:
            out["policies"] = chosen

    for key in _NUMBER_KEYS:
        if _is_number(raw.get(key)):
            out[key] = float(raw[key])

    for key in _PRESET_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()

    for key, protocol in ENV_KEYS.items():
        if key not in raw:
            continue                    # absent: the form keeps what it has
        value = raw[key]
        if value is None:
            out[key] = None             # explicitly "tracking the preset"
            continue
        env = env_from_dict(protocol, value)
        if env is not None:
            out[key] = env

    hidden = raw.get("hidden_traces")
    if isinstance(hidden, list):
        # Sorted and de-duplicated so the round trip is stable; the window
        # keeps them in a set anyway.
        out["hidden_traces"] = sorted({h for h in hidden
                                       if isinstance(h, str) and h})

    speed = _clean_positive_int(raw.get("speed"))
    if speed is not None:
        out["speed"] = speed
    return out
