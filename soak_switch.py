#!/usr/bin/env python3
"""Soak test for mode switching: restart the stack N times and prove it is
neither slow nor flaky.

run_stack.sh gates on readiness artifacts instead of sleeping. That is fast,
but a gate that is too weak produces failures that appear only occasionally --
gc mmaps /tmp/fpga_* and would take a SIGBUS on a file that exists but has not
been sized yet. One successful run proves nothing about that; this does.

Each iteration alternates PM and ENT, then checks:
  * the stack came up (apply did not raise)
  * the collector produced a row
  * QBER is in the mode's expected band -- a desynchronised Alice/Bob RNG
    stream shows up as ~50%, which is the loudest symptom of a bad gate
  * no panic/SIGBUS in the gc or simulator logs

Usage:  python3 soak_switch.py [iterations]
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backend as backend_mod  # noqa: E402
import channel  # noqa: E402
import stackctl  # noqa: E402

QBER_CEILING = 0.25   # anything near 0.5 means Alice/Bob lost correlation
ROW_TIMEOUT_S = 20.0
CRASH_PATTERN = re.compile(r"panic|SIGBUS|Segmentation fault|core dumped",
                           re.IGNORECASE)
LOGS = ("gc_alice.log", "gc_bob.log", "hwsim_alice.log", "hwsim_bob.log")


def first_row(csv_path, timeout=ROW_TIMEOUT_S):
    """Wait for the collector's first data row; return it parsed, or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            if len(lines) >= 2:
                header = lines[0].strip().split(",")
                return dict(zip(header, lines[1].strip().split(",")))
        except OSError:
            pass
        time.sleep(0.02)
    return None


def scan_logs():
    """Return crash lines found in the stack logs, if any."""
    hits = []
    for name in LOGS:
        path = os.path.join(stackctl.LOG_DIR, name)
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    if CRASH_PATTERN.search(line):
                        hits.append("%s: %s" % (name, line.strip()))
        except OSError:
            pass
    return hits


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    b = backend_mod.Backend()
    modes = [("pm", channel.EnvParams(mu1=0.29, mu2=0.103, p1=0.62), 10_000_000),
             ("ent", channel.EntEnvParams(), 1_000_000)]

    times, failures = [], []
    print("soaking %d switches (alternating PM/ENT)..." % n)
    for i in range(n):
        protocol, env, block = modes[i % 2]
        # Clear the logs each iteration so scan_logs() reports THIS bring-up.
        for name in LOGS:
            open(os.path.join(stackctl.LOG_DIR, name), "w").close()

        t0 = time.time()
        try:
            b.apply(env, block, protocol=protocol)
        except Exception as e:                      # noqa: BLE001 - report any
            failures.append("#%d %s: apply raised: %s" % (i, protocol, e))
            print("  #%-3d %-3s FAILED to start: %s" % (i, protocol, e))
            continue
        row = first_row(b.csv_path)
        dt = time.time() - t0
        times.append(dt)

        if row is None:
            failures.append("#%d %s: no CSV row within %.0fs"
                            % (i, protocol, ROW_TIMEOUT_S))
            status = "NO ROW"
        else:
            qber = float(row["qber"])
            if qber > QBER_CEILING:
                failures.append("#%d %s: QBER %.4f -- RNG desync?"
                                % (i, protocol, qber))
                status = "QBER %.2f%% BAD" % (100 * qber)
            else:
                status = "QBER %.2f%%" % (100 * qber)
        crashes = scan_logs()
        if crashes:
            failures.extend("#%d %s: %s" % (i, protocol, c) for c in crashes)
            status += " + CRASH"
        print("  #%-3d %-3s %5.2fs  %s" % (i, protocol, dt, status))

    b.stop()
    print("\n=== %d switches ===" % n)
    if times:
        print("apply -> first row:  min %.2fs  mean %.2fs  max %.2fs"
              % (min(times), sum(times) / len(times), max(times)))
    if failures:
        print("\nFAILURES (%d):" % len(failures))
        for f in failures:
            print("  " + f)
        return 1
    print("no failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
