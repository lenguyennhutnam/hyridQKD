#!/usr/bin/env bash
# Build the qline stack binaries. Every artifact stays under qkd_app/:
#   qkd_app/build/<project>/release/...   intermediate cargo output
#   qkd_app/bin/...                       the binaries run_stack.sh launches
#
# One target dir per project (same isolation cargo gives by default), so gc's and
# qber's identically-named `alice`/`bob` binaries never overwrite each other.
#
# JOBS defaults to 1 on purpose: build/ lives on /mnt/e, a 9p/drvfs mount, and
# parallel cargo writes race there — the build dies with "Permission denied
# (os error 13)" on a different crate each run (reproduced at both -j16 and -j2).
# Serial is ~6x slower but reliable. Override with JOBS=n if you point BUILD at
# the Linux filesystem instead.
set -euo pipefail

JOBS="${JOBS:-1}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$APP_DIR")"
BUILD="$APP_DIR/build"
BIN="$APP_DIR/bin"
mkdir -p "$BIN"

build() {  # build <name> <path-under-root> <cargo args...>
    local name=$1 path=$2
    shift 2
    echo "### building $name (-j$JOBS) ###"
    ( cd "$ROOT/$path" && CARGO_TARGET_DIR="$BUILD/$name" cargo build --release -j"$JOBS" "$@" )
}

build hw_sim hw_sim                --bin simulator
build gc     kiwi_hw_control/gc    --bin alice --bin bob
build qber   kiwi_hw_control/qber  --bin alice --bin bob

echo "### staging -> $BIN ###"
cp -f "$BUILD/hw_sim/release/simulator" "$BIN/simulator"
cp -f "$BUILD/gc/release/alice"         "$BIN/gc_alice"
cp -f "$BUILD/gc/release/bob"           "$BIN/gc_bob"
cp -f "$BUILD/qber/release/alice"       "$BIN/qber_alice"
cp -f "$BUILD/qber/release/bob"         "$BIN/qber_bob"
ls -la "$BIN"
