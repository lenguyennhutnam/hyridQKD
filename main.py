#!/usr/bin/env python3
"""Entry point for the QKD control panel. Run inside WSL:  python3 main.py

Wires all modules together (channel/presets/configio/stackctl via backend,
gui on top) and guarantees child processes die with the app: Stop button,
window close and interpreter exit all funnel into backend.stop(); a hard
crash is covered by kill_all.sh and by startup orphan cleanup here.
"""

import argparse
import atexit

from backend import Backend
from gui import App
import stackctl


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=None,
                        help="collector CSV path (default logs/live.csv)")
    args = parser.parse_args()

    stackctl.stop_all()          # clean orphans from a previous crash
    backend = Backend(csv_path=args.csv)
    atexit.register(backend.stop)
    App(backend).mainloop()


if __name__ == "__main__":
    main()
