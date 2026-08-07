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

# Popen handle for a collector we spawned in this interpreter. When set and
# its pid matches the pid file, we can reap it with poll()/waitpid() instead
# of relying on kill(pid, 0), which reports zombies as alive.
_collector_proc = None


def _ensure_dirs():
    os.makedirs(PID_DIR, exist_ok=True)


def start_stack(mode="decoy", health_timeout=3.0):
    """Run run_stack.sh (blocks a few seconds); raise if it fails.

    run_stack.sh backgrounds all 5 binaries and always exits 0 (it never waits
    on them), so a dead simulator would otherwise surface only as a confusing
    starting->error transition later. Poll for a live simulator process for up
    to health_timeout seconds and raise here instead, naming the log that
    explains why (e.g. a bad hw_sim config).
    """
    _ensure_dirs()
    env = dict(os.environ, MODE=mode)
    with open(os.path.join(LOG_DIR, "run_stack.log"), "ab") as log:
        ret = subprocess.run(["bash", os.path.join(APP_DIR, "run_stack.sh")],
                             cwd=APP_DIR, env=env, stdout=log,
                             stderr=subprocess.STDOUT).returncode
    if ret != 0:
        raise RuntimeError("run_stack.sh failed (exit %d), see logs/run_stack.log" % ret)

    deadline = time.time() + health_timeout
    while time.time() < deadline:
        if stack_running():
            return
        time.sleep(0.1)
    raise RuntimeError("qline stack did not come up (no simulator process "
                       "after %.1fs), see logs/hwsim_bob.log" % health_timeout)


def stack_running():
    """True if at least one simulator process is alive."""
    return subprocess.run(["pgrep", "-x", "simulator"],
                          stdout=subprocess.DEVNULL).returncode == 0


def start_collector(csv_path, extra_args=()):
    """Spawn collector.py in its own session; return its pid."""
    global _collector_proc
    _ensure_dirs()
    cmd = ["python3", os.path.join(APP_DIR, "collector.py"),
           "--csv", csv_path] + list(extra_args)
    log = open(os.path.join(LOG_DIR, "collector.log"), "ab", buffering=0)
    proc = subprocess.Popen(cmd, cwd=APP_DIR, stdout=log, stderr=log,
                            start_new_session=True)
    log.close()  # child holds its own fd copy; our handle is no longer needed
    _collector_proc = proc
    with open(COLLECTOR_PID, "w") as f:
        f.write(str(proc.pid))
    return proc.pid


def _read_pid():
    try:
        with open(COLLECTOR_PID) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _pid_is_collector(pid):
    """True if pid is alive and its cmdline looks like our collector.

    Guards against pid reuse: a stale pid file could point at an unrelated
    process that happens to have been assigned the same pid later.
    """
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            cmdline = f.read()
    except OSError:
        return False
    return b"collector.py" in cmdline


def collector_running():
    pid = _read_pid()
    if pid is None:
        return False
    if _collector_proc is not None and _collector_proc.pid == pid:
        # We own this child: poll() reaps it via waitpid(WNOHANG), so a dead
        # child is correctly reported dead instead of showing up as a zombie
        # under kill(pid, 0).
        return _collector_proc.poll() is None
    # Not our child (e.g. orphan pid file from a previous app run). An
    # orphan gets reparented to init and reaped there, so kill(pid, 0)
    # failing does mean dead. Guard against pid reuse before trusting it.
    if not _pid_is_collector(pid):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_collector(timeout=5.0):
    """SIGTERM the collector's process group (lets it flush + send gc Stop),
    escalate to SIGKILL after timeout. Idempotent."""
    global _collector_proc
    pid = _read_pid()
    if pid is None:
        return
    owned = _collector_proc is not None and _collector_proc.pid == pid
    if not owned and not _pid_is_collector(pid):
        # Stale pid file: pid is dead, or reused by an unrelated process.
        try:
            os.remove(COLLECTOR_PID)
        except OSError:
            pass
        _collector_proc = None
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        deadline = time.time() + timeout
        while time.time() < deadline:
            if owned:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            if not collector_running():
                break
            time.sleep(0.1)
        if not collector_running():
            break
    try:
        os.remove(COLLECTOR_PID)
    except OSError:
        pass
    _collector_proc = None


def stop_all():
    """Stop collector + stack + stragglers. Safe to call any time."""
    stop_collector()
    subprocess.run(["bash", os.path.join(APP_DIR, "stop_stack.sh")],
                   cwd=APP_DIR, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    # Safety net for orphans whose pid file was lost.
    subprocess.run(["pkill", "-f", "collector.py --csv"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
