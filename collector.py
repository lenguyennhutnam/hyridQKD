#!/usr/bin/env python3
"""Realtime extractor for the qline QKD stack (PM/BB84-decoy or ENT/BBM92).

This is the first building block of the main app. It takes the role of
`qber_alice` in the VeriQloud stack: it talks to gc-alice (Start/Stop over its
unix command socket), asks `qber_bob` to stream Bob's angles + detection
results over TCP, reads Alice's angle+decoy stream from the hw_sim FIFO, and
every --interval seconds prints one line:

    total bits generated (sifted), keys completed, leftover bits of the key
    in progress, and the window QBER.

A "key" is plain bookkeeping for now: every KEY_BLOCK_BITS sifted bits = 1 key
(e.g. 10e6+1000 bits -> 1 key, 1000 bits leftover). Tune the knobs in the
PARAMETERS block below, or override per-run with the CLI flags.

--protocol picks the finite-key bound and what a block is filled from: `pm`
(default) distils the Z-basis subset with the Rusca 1-decoy bound, `ent` keys
BOTH sifted bases with the Chapman BBM92 bound (no decoy states involved).

Run inside WSL with the stack already up (see run_stack.sh):
    python3 collector.py [--protocol pm|ent] [--interval 2.0]
                         [--key-block-bits 10000000] [--csv out.csv]
Stop with Ctrl-C (sends Stop to gc and prints a cumulative summary).
Pass --detail for the extra per-intensity line (mu1/mu2 split) each window.
"""

import argparse
import signal
import socket
import struct
import sys
import time

import numpy as np

from keyrate import KeyParams, secret_key_length
from keyrate_ent import secret_key_length_ent

# ==================== PARAMETERS (chinh o day) ====================
KEY_BLOCK_BITS = 10_000_000   # so sifted bit cho 1 block finite-key (PM: basis Z; ENT: ca hai basis)
PRINT_INTERVAL_S = 2.0        # chu ky in ra terminal (giay)

# Tham so giao thuc dung cho cong thuc key length (phai khop hw_sim config).
MU1, MU2, P1 = 0.5, 0.1, 0.7  # cuong do signal/decoy va P(signal)
ETA = 0.1                     # transmittance x detector efficiency
# Luu y: voi bo nay bound 1-decoy tat o QBER ~4.2%. Bo toi uu hon
# (mu1=0.29, mu2=0.103, p1=0.62) day nguong len ~5.2%.
# CLI flags --key-block-bits / --interval / --mu1 ... override cac gia tri nay.
# ==================================================================

# --- qber_bob stream framing (kiwi_hw_control/qber/src/lib.rs: BATCHSIZE) ---
BATCH = 256                # detection events per batch
ANGLE_BYTES = BATCH // 2   # 2 events per byte: low nibble = even event
RESULT_BYTES = BATCH       # 1 byte per event (0 or 1)

# --- comm crate wire protocol (kiwi_hw_control/comm/src/lib.rs) ---
# Frame: 4-byte LE length prefix + bincode(standard) payload.
# A unit enum variant encodes as its variant index (single byte for idx < 251).
GC_START, GC_STOP = 0, 1
GC_RESPONSES = {0: "Done", 1: "DidNothing", 2: "HwReady", 3: "HwNotReady"}
QBER_SEND_ANGLES = 0


def recv_exact_sock(sock, n):
    """Read exactly n bytes from a socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed by peer")
        buf += chunk
    return bytes(buf)


def read_exact_file(f, n):
    """Read exactly n bytes from a file object (blocking FIFO)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise ConnectionError("fifo closed by writer")
        buf += chunk
    return bytes(buf)


def write_message(sock, variant):
    """comm::write_message for a unit enum variant."""
    payload = bytes([variant])
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def read_message(sock):
    """comm::read_message; returns the raw payload bytes."""
    (length,) = struct.unpack("<I", recv_exact_sock(sock, 4))
    return recv_exact_sock(sock, length)


def build_expect_table(angles):
    """exp[i, j] = deterministic outcome (0/1) for Alice state i, Bob state j,
    or -1 when the outcome is random (unmatched bases, discarded by sifting).

    Mirrors hw_sim (sim_lib/src/simulation/mod.rs): P(result=0) =
    cos^2(pi*offset/128) with offset = (A[i] + A[j] + 32) & 127, so the
    outcome is deterministic iff offset == 0 (-> 0) or offset == 64 (-> 1).
    """
    exp = np.full((4, 4), -1, dtype=np.int16)
    for i, ai in enumerate(angles):
        for j, aj in enumerate(angles):
            off = (ai + aj + 32) & 127
            if off == 0:
                exp[i, j] = 0
            elif off == 64:
                exp[i, j] = 1
    return exp


def expand_nibbles(packed):
    """Unpack a byte array into per-event nibbles (low nibble = even event)."""
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = packed >> 4
    return out


class Stats:
    """Counters for one window (or the cumulative run).

    n_z/m_z are the key-basis subset: with hw_sim's angles [0,32,96,64] Alice's
    states {0,3} only ever correlate with Bob's {1,2} and vice versa, so {0,3}
    is taken as the key basis (Z) and {1,2} as the check basis (X).
    """

    FIELDS = ("clicks", "sifted", "errors", "n_sig", "m_sig", "n_dec", "m_dec",
              "n_z", "m_z")

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, 0)

    def add(self, other):
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @property
    def qber(self):
        return self.errors / self.sifted if self.sifted else float("nan")


def block_attrs(protocol):
    """Which Stats counters feed the finite-key block.

    PM keys only the Z basis (n_z/m_z). ENT/BBM92 keys BOTH bases
    (Chapman: bases chosen with p=1/2 each, both generate key), so the
    block fills from the full sifted count.
    """
    if protocol == "ent":
        return "sifted", "errors"
    return "n_z", "m_z"


def crunch(aa, ab, rr, exp):
    """Turn raw window bytes into Stats.

    aa: Alice nibbles `0[decoy][b1][b0]` (from the hw_sim angle FIFO),
    ab: Bob nibbles `00[b1][b0]` (over TCP from qber_bob),
    rr: result bytes, one per event.
    """
    alice_n = expand_nibbles(np.frombuffer(aa, dtype=np.uint8))
    bob = expand_nibbles(np.frombuffer(ab, dtype=np.uint8)) & 3
    res = (np.frombuffer(rr, dtype=np.uint8) != 0).astype(np.int16)

    decoy = (alice_n >> 2) & 1          # 1 = decoy pulse (mu2)
    alice = alice_n & 3

    e = exp[alice, bob]                 # -1 / 0 / 1
    sift = e >= 0
    err = sift & (res != e)

    s = Stats()
    s.clicks = alice.size
    s.sifted = int(np.count_nonzero(sift))
    s.errors = int(np.count_nonzero(err))
    sig = decoy == 0
    s.n_sig = int(np.count_nonzero(sift & sig))
    s.m_sig = int(np.count_nonzero(err & sig))
    s.n_dec = s.sifted - s.n_sig
    s.m_dec = s.errors - s.m_sig
    z = (alice == 0) | (alice == 3)     # key basis
    s.n_z = int(np.count_nonzero(sift & z))
    s.m_z = int(np.count_nonzero(err & z))
    return s


def fmt_key_row(t, s, cum, block_fill, block_bits, blocks, secret_bits):
    """Main line: sifted totals, progress of the block being filled, and the
    secret key distilled so far.

    block_fill is the active protocol's fill counter (see block_attrs).
    """
    return (
        f"[{t:7.1f}s] sifted +{s.sifted:>9,} (tot {cum.sifted:>11,}) | "
        f"block {block_fill:>10,}/{block_bits:,} | QBER {100*s.qber:5.2f}% | "
        f"keys {blocks:>2} secret {secret_bits:>10,} b"
    )


def fmt_detail_row(dt, s):
    """Optional extra line: click rate and per-intensity (mu1/mu2) split."""
    rate = s.clicks / dt if dt > 0 else 0.0
    e_sig = 100 * s.m_sig / s.n_sig if s.n_sig else float("nan")
    e_dec = 100 * s.m_dec / s.n_dec if s.n_dec else float("nan")
    dec_pct = 100 * (s.n_dec / s.sifted) if s.sifted else float("nan")
    return (
        f"          clicks {s.clicks:>9,} ({rate/1e6:5.2f}M/s) | "
        f"mu1 n={s.n_sig:>9,} E={e_sig:5.2f}% | "
        f"mu2 n={s.n_dec:>8,} E={e_dec:5.2f}% | decoy {dec_pct:4.1f}%"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gc-socket", default="/tmp/gc_alice_command.socket",
                    help="gc-alice unix command socket")
    ap.add_argument("--bob", default="127.0.0.1:58000",
                    help="qber_bob TCP address host:port")
    ap.add_argument("--fifo", default="/tmp/gc_alice_angle.fifo",
                    help="Alice angle FIFO written by hw_sim")
    ap.add_argument("--angles", default="0,32,96,64",
                    help="angles table, must match hw_sim backend_config.angles")
    ap.add_argument("--interval", type=float, default=PRINT_INTERVAL_S,
                    help="print period in seconds")
    ap.add_argument("--key-block-bits", type=int, default=KEY_BLOCK_BITS,
                    help="sifted bits per finite-key block (default 10e6); "
                         "key basis only for pm, both bases for ent")
    ap.add_argument("--protocol", choices=("pm", "ent"), default="pm",
                    help="finite-key bound: pm=Rusca 1-decoy (Z basis), "
                         "ent=Chapman BBM92 (both bases)")
    ap.add_argument("--mu1", type=float, default=MU1, help="signal intensity")
    ap.add_argument("--mu2", type=float, default=MU2, help="decoy intensity")
    ap.add_argument("--p1", type=float, default=P1, help="P(signal intensity)")
    ap.add_argument("--eta", type=float, default=ETA,
                    help="channel transmittance x detector efficiency")
    ap.add_argument("--detail", action="store_true",
                    help="also print the per-intensity mu1/mu2 line each window")
    ap.add_argument("--csv", default=None,
                    help="optional: append per-window stats to this CSV file")
    args = ap.parse_args()

    angles = [int(x) for x in args.angles.split(",")]
    exp = build_expect_table(angles)
    host, port = args.bob.rsplit(":", 1)

    # SIGTERM behaves like Ctrl-C so the finally-block still sends Stop.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    csv = None
    if args.csv:
        csv = open(args.csv, "a", buffering=1)
        if csv.tell() == 0:
            csv.write("unix_time,window_s,clicks,sifted,errors,qber,"
                      "n_sig,m_sig,n_dec,m_dec,total_sifted,blocks,"
                      "block_progress,secret_bits\n")

    # Connection order mirrors qber_alice: bob TCP, gc socket, Start, SendAngles.
    bob = socket.create_connection((host, int(port)), timeout=10)
    gc = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    gc.settimeout(5.0)
    gc.connect(args.gc_socket)

    print(f"-> gc Start ({args.gc_socket})")
    write_message(gc, GC_START)
    reply = read_message(gc)
    print(f"<- gc: {GC_RESPONSES.get(reply[0], reply.hex())}")

    write_message(bob, QBER_SEND_ANGLES)
    bob.settimeout(None)

    params = KeyParams(mu1=args.mu1, mu2=args.mu2, p1=args.p1, eta=args.eta)
    bits_attr, err_attr = block_attrs(args.protocol)
    if args.protocol == "ent":
        def compute_len(n, q):
            return secret_key_length_ent(n, q)
    else:
        def compute_len(n, q):
            return secret_key_length(n, q, params)

    cum = Stats()
    t0 = time.monotonic()
    win_t0 = t0
    blk = Stats()          # bits accumulated toward the current block
    blocks = 0
    secret_bits = 0
    aa_buf, ab_buf, rr_buf = bytearray(), bytearray(), bytearray()

    if args.protocol == "ent":
        proto_line = ("protocol ENT/BBM92 (Chapman), 1 block = "
                      f"{args.key_block_bits:,} sifted bits (both bases)")
    else:
        proto_line = ("protocol PM/BB84 (Rusca 1-decoy), 1 block = "
                      f"{args.key_block_bits:,} key-basis sifted bits, "
                      f"mu1={args.mu1} mu2={args.mu2} p1={args.p1} "
                      f"eta={args.eta}")
    print(f"streaming... {proto_line}, "
          f"printing every {args.interval:.1f}s (Ctrl-C to stop)")
    try:
        with open(args.fifo, "rb") as fifo:
            while True:
                # One batch, same order as qber_alice: own fifo, bob angles, results.
                aa_buf += read_exact_file(fifo, ANGLE_BYTES)
                ab_buf += recv_exact_sock(bob, ANGLE_BYTES)
                rr_buf += recv_exact_sock(bob, RESULT_BYTES)

                now = time.monotonic()
                if now - win_t0 >= args.interval:
                    s = crunch(bytes(aa_buf), bytes(ab_buf), bytes(rr_buf), exp)
                    cum.add(s)
                    blk.add(s)

                    # A block is ready once enough bits have piled up (key
                    # basis for PM, both bases for ENT — see block_attrs).
                    # The overshoot (and its share of the errors) carries over to
                    # the next block so no bit is counted twice.
                    while getattr(blk, bits_attr) >= args.key_block_bits:
                        fill = getattr(blk, bits_attr)
                        fill_err = getattr(blk, err_attr)
                        n_blk = args.key_block_bits
                        over = fill - n_blk
                        m_carry = round(fill_err * over / fill)
                        m_blk = fill_err - m_carry
                        q_blk = m_blk / n_blk
                        ell = compute_len(n_blk, q_blk)
                        blocks += 1
                        secret_bits += ell
                        verdict = (f"{ell:,} secret bits" if ell
                                   else "NO KEY (QBER above the finite-key bound)")
                        print(f"          *** block #{blocks}: "
                              f"QBER {100*q_blk:.2f}% -> {verdict} ***")
                        setattr(blk, bits_attr, over)
                        setattr(blk, err_attr, m_carry)

                    print(fmt_key_row(now - t0, s, cum, getattr(blk, bits_attr),
                                      args.key_block_bits, blocks, secret_bits),
                          flush=True)
                    if args.detail:
                        print(fmt_detail_row(now - win_t0, s), flush=True)
                    if csv:
                        csv.write(f"{time.time():.3f},{now - win_t0:.3f},"
                                  f"{s.clicks},{s.sifted},{s.errors},"
                                  f"{s.qber:.6f},{s.n_sig},{s.m_sig},"
                                  f"{s.n_dec},{s.m_dec},{cum.sifted},{blocks},"
                                  f"{getattr(blk, bits_attr)},{secret_bits}\n")
                    aa_buf, ab_buf, rr_buf = bytearray(), bytearray(), bytearray()
                    win_t0 = now
    except (KeyboardInterrupt, SystemExit):
        pass
    except ConnectionError as e:
        print(f"stream ended: {e}", file=sys.stderr)
    finally:
        # Flush the partial window into the cumulative totals. Ctrl-C can land
        # between the three reads of a batch, leaving the buffers one batch
        # apart — trim them to the same number of complete events first.
        n_ev = min(len(aa_buf) * 2, len(ab_buf) * 2, len(rr_buf))
        if n_ev:
            cum.add(crunch(bytes(aa_buf[:n_ev // 2]),
                           bytes(ab_buf[:n_ev // 2]),
                           bytes(rr_buf[:n_ev]), exp))
        print("-> gc Stop")
        try:
            write_message(gc, GC_STOP)
            reply = read_message(gc)
            print(f"<- gc: {GC_RESPONSES.get(reply[0], reply.hex())}")
        except OSError as e:
            print(f"(gc stop failed: {e})", file=sys.stderr)
        gc.close()
        bob.close()
        if csv:
            csv.close()
        dt = time.monotonic() - t0
        print(f"=== total {dt:.1f}s: clicks {cum.clicks:,}  "
              f"sifted {cum.sifted:,} (key basis {cum.n_z:,})  "
              f"QBER {100*cum.qber:.2f}% ===")
        print(f"=== blocks {blocks} -> secret {secret_bits:,} bits"
              f"{f' ({secret_bits/dt/1000:.1f} kbit/s)' if dt > 0 else ''}, "
              f"block in progress {getattr(blk, bits_attr):,}"
              f"/{args.key_block_bits:,} ===")


if __name__ == "__main__":
    main()
