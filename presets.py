#!/usr/bin/env python3
"""Load/save EnvParams presets stored as JSON files in presets/."""

import dataclasses
import json
import os
from collections import OrderedDict

from channel import EnvParams

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


def load_presets():
    """Return OrderedDict {name: EnvParams} from presets/*.json, sorted by name.

    Unknown JSON keys are rejected loudly (typo protection), missing keys fall
    back to EnvParams defaults.
    """
    out = OrderedDict()
    known = {f.name for f in dataclasses.fields(EnvParams)}
    for fname in sorted(os.listdir(PRESET_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(PRESET_DIR, fname)) as f:
            data = json.load(f)
        unknown = set(data) - known
        if unknown:
            raise ValueError("preset %s has unknown keys: %s" % (fname, sorted(unknown)))
        out[fname[:-5]] = EnvParams(**data)
    return out


def save_preset(name, env):
    """Write EnvParams as presets/<name>.json; returns the path."""
    path = os.path.join(PRESET_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(env), f, indent=4)
        f.write("\n")
    return path
