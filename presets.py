#!/usr/bin/env python3
"""Load/save parameter presets stored as JSON files in presets/.

A preset file may carry a "protocol" key: "pm" (default when absent) or
"ent". PM presets deserialise to channel.EnvParams, ENT presets to
channel.EntEnvParams; each loader silently skips the other protocol's files
so both can share the presets/ directory.

Chapman Table I -> field mapping used by the ENT presets: Bob's arm folds
detector efficiency x measurement-system transmission into eta_bob
(0.4*0.6=0.24 current, 0.9*0.7=0.63 future) with odr_losses=0; eta_a=0.3 is
Alice's total efficiency including detection (Chapman Fig. 8); alpha=0.2
dB/km matches the paper's fiber statement.
"""

import dataclasses
import json
import os
from collections import OrderedDict

import panelstate
from channel import EntEnvParams, EnvParams

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

_CLASSES = {"pm": EnvParams, "ent": EntEnvParams}


def _load(protocol):
    cls = _CLASSES[protocol]
    known = {f.name for f in dataclasses.fields(cls)}
    out = OrderedDict()
    for fname in sorted(os.listdir(PRESET_DIR)):
        if not fname.endswith(".json"):
            continue
        if fname[:-5] == panelstate.RESERVED_NAME:
            # The panel dropdown lists this name for the saved panel state
            # (panelstate.py); a preset sharing it would appear twice.
            raise ValueError("preset name %r is reserved for the saved panel "
                             "state -- rename presets/%s"
                             % (panelstate.RESERVED_NAME, fname))
        with open(os.path.join(PRESET_DIR, fname)) as f:
            data = json.load(f)
        if data.pop("protocol", "pm") != protocol:
            continue
        unknown = set(data) - known
        if unknown:
            raise ValueError("preset %s has unknown keys: %s"
                             % (fname, sorted(unknown)))
        out[fname[:-5]] = cls(**data)
    return out


def load_presets():
    """Return OrderedDict {name: EnvParams} for PM presets, sorted by name."""
    return _load("pm")


def load_ent_presets():
    """Return OrderedDict {name: EntEnvParams} for ENT presets."""
    return _load("ent")


def save_preset(name, env):
    """Write a params dataclass as presets/<name>.json; returns the path."""
    data = dataclasses.asdict(env)
    if isinstance(env, EntEnvParams):
        data["protocol"] = "ent"
    path = os.path.join(PRESET_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    return path
