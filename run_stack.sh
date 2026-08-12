#!/usr/bin/env bash
# Bring up the qline stack WITHOUT qber_alice, ready for collector.py to attach.
# Binaries come from ./bin (see build.sh).
#
#   MODE=decoy (default)  -> hw_sim_decoy_config.json   (PM/BB84 with mu1/mu2)
#   MODE=plain            -> hw_sim_config.json         (no decoy states)
#
# There are two config files per player; editing the one the stack is NOT running
# has no effect, so the actual file and its qberr/eta are echoed on startup.
set -u

MODE="${MODE:-decoy}"
case "$MODE" in
    decoy) HW_CFG=hw_sim_decoy_config.json ;;
    plain) HW_CFG=hw_sim_config.json ;;
    *) echo "MODE must be 'decoy' or 'plain', got '$MODE'" >&2; exit 1 ;;
esac

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$APP_DIR/bin"
CFG="$(dirname "$APP_DIR")/hw_sim/config_files"
LOG="$APP_DIR/logs"
mkdir -p "$LOG"

if [ ! -x "$BIN/simulator" ]; then
    echo "binaries missing in $BIN — run: bash build.sh" >&2
    exit 1
fi

source "$APP_DIR/wait_ready.sh"

# Size the simulator gives /tmp/fpga_* (hw_sim/configs/src/ipc/mod.rs:
# 0x12000 + 0x1000). gc mmaps len 0x1000 at offset 0x1000, so starting gc
# against a shorter file is a SIGBUS, not a retry.
FPGA_BYTES=77824
QBER_PORT=58000   # matches ip_listen in bob/qber_config.json

# teardown leftovers -- wait for them to actually die before unlinking, or the
# old inode stays alive and the next run reuses a stale pipe
pkill -x simulator 2>/dev/null
pkill -x gc_alice 2>/dev/null; pkill -x gc_bob 2>/dev/null
pkill -x qber_alice 2>/dev/null; pkill -x qber_bob 2>/dev/null
for _ in $(seq 1 200); do
    pgrep -x 'simulator|gc_alice|gc_bob|qber_bob' > /dev/null || break
    sleep 0.01
done
rm -f /tmp/fpga_alice /tmp/fpga_bob /tmp/gc_alice_* /tmp/gc_bob_* \
      /tmp/hw_params_alice.txt /tmp/hw_params_bob.txt

"$BIN/simulator" --config-path "$CFG/alice/$HW_CFG" > "$LOG/hwsim_alice.log" 2>&1 &
"$BIN/simulator" --config-path "$CFG/bob/$HW_CFG"   > "$LOG/hwsim_bob.log"   2>&1 &
# The fpga file is the LAST thing setup_ipc creates on both sides (fifos and
# hw_params come first), so gating on its size covers the whole setup.
wait_file_size /tmp/fpga_alice "$FPGA_BYTES" 10 || exit 1
wait_file_size /tmp/fpga_bob   "$FPGA_BYTES" 10 || exit 1
wait_exists /tmp/hw_params_alice.txt 10 || exit 1
wait_exists /tmp/hw_params_bob.txt   10 || exit 1

"$BIN/gc_alice" -c "$CFG/alice/gc_config.json" > "$LOG/gc_alice.log" 2>&1 &
"$BIN/gc_bob"   -c "$CFG/bob/gc_config.json"   > "$LOG/gc_bob.log"   2>&1 &
wait_exists /tmp/gc_alice_command.socket 10 || exit 1
wait_exists /tmp/gc_bob_command.socket   10 || exit 1

"$BIN/qber_bob" -c "$CFG/bob/qber_config.json" > "$LOG/qber_bob.log" 2>&1 &
wait_tcp_listen "$QBER_PORT" 10 || exit 1

# Show the config actually in use. Bob's qberr/eta drive the observed physics;
# Alice's are ignored, but seed/angles/decoy-mode must match on both sides.
echo "qline stack up (MODE=$MODE, no qber_alice). Logs: $LOG"
echo "  config: $HW_CFG"
for who in alice bob; do
    echo "    $who: $(grep -oE '"(seed|eta|qberr)"[[:space:]]*:[[:space:]]*[0-9.e-]+' \
        "$CFG/$who/$HW_CFG" | tr -d '"' | tr '\n' ' ')"
done
echo "now run:  python3 collector.py"
