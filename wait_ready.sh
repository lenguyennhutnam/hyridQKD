#!/usr/bin/env bash
# Readiness gates for the qline stack, sourced by run_stack.sh.
#
# These replace fixed sleeps. A sleep long enough to be safe is ~30x longer
# than the stack actually needs (measured: whole stack ready in 0.12s), and a
# sleep short enough to be quick is a race. Polling for the artifact the NEXT
# stage consumes is both fast and correct.
#
# Every gate returns 0 as soon as the condition holds, or 1 with a message on
# stderr naming what never turned up.

_WAIT_POLL_S=0.01

_wait_tries() {  # <timeout_s> -> number of polls, at least 1
    awk -v t="$1" -v p="$_WAIT_POLL_S" 'BEGIN { n = int(t / p); print (n < 1) ? 1 : n }'
}

wait_file_size() {  # <path> <min_bytes> <timeout_s>
    # Existence is NOT enough for the mmap'd fpga register files: the simulator
    # creates them with O_TRUNC and set_len()s them afterwards, so there is a
    # window where the file exists at 0 bytes. gc mmaps len 0x1000 at offset
    # 0x1000 and would take a SIGBUS on a short file, which is exactly the
    # intermittent crash the old sleep was hiding.
    local path="$1" min="$2" timeout="$3"
    local n size i
    n=$(_wait_tries "$timeout")
    for ((i = 0; i < n; i++)); do
        size=$(stat -c %s "$path" 2>/dev/null) || size=-1
        if [ "$size" -ge "$min" ] 2>/dev/null; then
            return 0
        fi
        sleep "$_WAIT_POLL_S"
    done
    echo "timeout: $path did not reach $min bytes within ${timeout}s (size: ${size})" >&2
    return 1
}

wait_exists() {  # <path> <timeout_s>
    # For FIFOs and unix sockets, which are always 0 bytes: the inode appearing
    # IS the readiness signal, since mkfifo/bind are atomic.
    local path="$1" timeout="$2"
    local n i
    n=$(_wait_tries "$timeout")
    for ((i = 0; i < n; i++)); do
        if [ -e "$path" ]; then
            return 0
        fi
        sleep "$_WAIT_POLL_S"
    done
    echo "timeout: $path did not appear within ${timeout}s" >&2
    return 1
}

wait_tcp_listen() {  # <port> <timeout_s>
    # Observed passively via ss rather than by connecting: qber_bob serves a
    # single expected client (the collector), so a probe connection could
    # disturb the very handshake we are waiting for.
    local port="$1" timeout="$2"
    local n i
    n=$(_wait_tries "$timeout")
    for ((i = 0; i < n; i++)); do
        if ss -ltn 2>/dev/null | grep -qE "[:.]${port}([[:space:]]|\$)"; then
            return 0
        fi
        sleep "$_WAIT_POLL_S"
    done
    echo "timeout: nothing listening on port $port within ${timeout}s" >&2
    return 1
}
