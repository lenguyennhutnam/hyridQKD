#!/usr/bin/env python3
"""Start/stop the qline stack and the collector as managed subprocesses.

The stack binaries themselves are handled by the existing run_stack.sh /
stop_stack.sh (which pkill by exact binary name and clear /tmp IPC). This
module adds: collector lifecycle with its own session + pid file, and an
idempotent stop_all() used both by the GUI (atexit / window close / startup
orphan cleanup) and by kill_all.sh.
"""

import os
import signal
import subprocess
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, "logs")
PID_DIR = os.path.join(LOG_DIR, "pids")
COLLECTOR_PID = os.path.join(PID_DIR, "collector.pid")


def _ensure_dirs():
    os.makedirs(PID_DIR, exist_ok=True)


def start_stack(mode="decoy"):
    """Run run_stack.sh (blocks a few seconds); raise if it fails."""
    _ensure_dirs()
    env = dict(os.environ, MODE=mode)
    with open(os.path.join(LOG_DIR, "run_stack.log"), "ab") as log:
        ret = subprocess.run(["bash", os.path.join(APP_DIR, "run_stack.sh")],
                             cwd=APP_DIR, env=env, stdout=log,
                             stderr=subprocess.STDOUT).returncode
    if ret != 0:
        raise RuntimeError("run_stack.sh failed (exit %d), see logs/run_stack.log" % ret)


def stack_running():
    """True if at least one simulator process is alive."""
    return subprocess.run(["pgrep", "-x", "simulator"],
                          stdout=subprocess.DEVNULL).returncode == 0


def start_collector(csv_path, extra_args=()):
    """Spawn collector.py in its own session; return its pid."""
    _ensure_dirs()
    cmd = ["python3", os.path.join(APP_DIR, "collector.py"),
           "--csv", csv_path] + list(extra_args)
    log = open(os.path.join(LOG_DIR, "collector.log"), "ab", buffering=0)
    proc = subprocess.Popen(cmd, cwd=APP_DIR, stdout=log, stderr=log,
                            start_new_session=True)
    with open(COLLECTOR_PID, "w") as f:
        f.write(str(proc.pid))
    return proc.pid


def _read_pid():
    try:
        with open(COLLECTOR_PID) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def collector_running():
    pid = _read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_collector(timeout=5.0):
    """SIGTERM the collector's process group (lets it flush + send gc Stop),
    escalate to SIGKILL after timeout. Idempotent."""
    pid = _read_pid()
    if pid is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not collector_running():
                break
            time.sleep(0.1)
        if not collector_running():
            break
    try:
        os.remove(COLLECTOR_PID)
    except OSError:
        pass


def stop_all():
    """Stop collector + stack + stragglers. Safe to call any time."""
    stop_collector()
    subprocess.run(["bash", os.path.join(APP_DIR, "stop_stack.sh")],
                   cwd=APP_DIR, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    # Safety net for orphans whose pid file was lost.
    subprocess.run(["pkill", "-f", "collector.py --csv"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
