#!/usr/bin/env bash
# Tear down the qline stack started by run_stack.sh and clear its IPC objects.
# Pass --keep-ipc to leave /tmp alone (e.g. to inspect the fpga register dumps).
KEEP_IPC=0
[ "${1:-}" = "--keep-ipc" ] && KEEP_IPC=1

pkill -x qber_bob 2>/dev/null
pkill -x gc_alice 2>/dev/null; pkill -x gc_bob 2>/dev/null
pkill -x simulator 2>/dev/null

if [ "$KEEP_IPC" = 0 ]; then
    # Wait for the writers to exit before unlinking, otherwise a still-running
    # simulator keeps the old inode alive and the next run reuses a stale pipe.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        pgrep -x simulator > /dev/null || break
        sleep 0.1
    done
    # Same set run_stack.sh clears on startup: fifos, unix sockets, the mmap'd
    # fpga register files and the hw parameter files.
    rm -f /tmp/fpga_alice /tmp/fpga_bob /tmp/gc_alice_* /tmp/gc_bob_* \
          /tmp/hw_params_alice.txt /tmp/hw_params_bob.txt
    echo "qline stack stopped, /tmp IPC cleared."
else
    echo "qline stack stopped (/tmp IPC kept)."
fi
