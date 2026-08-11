#!/usr/bin/env python3
"""Continuous panel history across Apply/Stop and PM<->ENT mode switches.

The GUI used to wipe its plots on every Apply, so switching protocol threw
away everything the previous mode had produced. This module keeps one running
history for the whole session (reset only when the app exits) and records:

  * SEGMENTS -- one per stretch of streaming, tagged with the protocol that
    produced it and that protocol's own death-QBER threshold, so the panel can
    colour each stretch and step the threshold line at the boundary.
  * GAPS -- the stack-down intervals between segments. Their width IS the
    switch cost t_switch, which the hybrid policy problem needs as a real
    measured number rather than a guess.

THE ACCUMULATION PROBLEM: the collector's secret_bits column counts from zero
for each collector process, and backend.apply() restarts the collector every
time. A naive "stop clearing the history" would therefore make the key buffer
drop back to zero at each switch. Each segment carries a `secret_offset` (the
session total before it began) and the exported `secret_total` adds the two,
so the buffer only ever grows.

No tkinter, no matplotlib: this is plain state, unit-tested headless, in the
same spirit as channel.py / eve.py / panelstate.py.
"""

from dataclasses import dataclass

PROTOCOLS = ("pm", "ent")
_DEFAULT_PROTOCOL = "pm"


@dataclass
class Segment:
    """One uninterrupted stretch of streaming under a single protocol."""

    protocol: str
    t_start: float                 # unix time of the segment's first sample
    t_end: float                   # unix time of its latest sample
    death_qber: float = None       # threshold of THIS segment's mode (may be None)
    secret_offset: int = 0         # session total before this segment began
    secret_final: int = 0          # last per-run secret_bits seen here
    i0: int = 0                    # index of its first sample
    i1: int = 0                    # index of its latest sample (inclusive)
    secret_banked: int = 0         # production from runs before a mid-segment respawn


@dataclass
class Gap:
    """A stretch with no stream: Apply restarting the stack, or after Stop."""

    t_start: float                 # when Apply/Stop was pressed
    t_end: float = 0.0             # first sample of the next segment (0 = open)
    is_switch: bool = False        # True only when the protocol actually changed


@dataclass
class _Pending:
    """What the next segment will be, recorded when downtime begins."""

    protocol: str = None
    death_qber: float = None


class Timeline:
    """Session-long history. Feed it begin_downtime() and add_sample()."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget everything (the app only does this by exiting)."""
        self.t = []
        self.qber = []             # PERCENT, as the GUI already stores it
        self.sifted = []
        self.secret_total = []
        self._segments = []
        self._gaps = []
        self._pending = _Pending()

    def __len__(self):
        return len(self.t)

    # ---------------- inputs ----------------
    def begin_downtime(self, now, next_protocol=None, death_qber=None):
        """Called on Apply (with the mode being started) and on Stop (without).

        Closes the running segment and opens a gap. Stop passes no protocol,
        so is_switch is decided later, when the next segment actually opens.
        """
        if next_protocol is not None:
            self._pending = _Pending(protocol=next_protocol,
                                     death_qber=death_qber)
        if self._gaps and self._gaps[-1].t_end == 0.0:
            return                 # already down; keep the earlier start time
        self._gaps.append(Gap(t_start=now))

    def add_sample(self, unix_t, qber_pct, sifted, secret_bits):
        """One streaming poll tick."""
        seg = self._current_segment()
        if seg is None:
            seg = self._open_segment(unix_t)

        # A collector that died and respawned restarts its counter mid-segment;
        # bank what it had already produced so the buffer cannot go backwards.
        # secret_banked mirrors that into secret_by_protocol's per-segment
        # sum, which reads secret_banked + secret_final instead of the raw
        # (now-zeroed) secret_final alone.
        if secret_bits < seg.secret_final:
            seg.secret_offset += seg.secret_final
            seg.secret_banked += seg.secret_final
            seg.secret_final = 0

        self.t.append(unix_t)
        self.qber.append(qber_pct)
        self.sifted.append(sifted)
        self.secret_total.append(seg.secret_offset + int(secret_bits))
        seg.secret_final = int(secret_bits)
        seg.t_end = unix_t
        seg.i1 = len(self.t) - 1

    # ---------------- internals ----------------
    def _current_segment(self):
        """The segment still accepting samples, or None if we are down."""
        if not self._segments:
            return None
        if self._gaps and self._gaps[-1].t_end == 0.0:
            return None            # downtime open: the old segment is closed
        return self._segments[-1]

    def _open_segment(self, unix_t):
        previous = self._segments[-1] if self._segments else None
        protocol = self._pending.protocol or (
            previous.protocol if previous else _DEFAULT_PROTOCOL)
        if self._gaps and self._gaps[-1].t_end == 0.0:
            gap = self._gaps[-1]
            gap.t_end = unix_t
            gap.is_switch = bool(previous) and previous.protocol != protocol
        seg = Segment(protocol=protocol, t_start=unix_t, t_end=unix_t,
                      death_qber=self._pending.death_qber,
                      secret_offset=self.total_secret,
                      i0=len(self.t), i1=len(self.t))
        self._segments.append(seg)
        return seg

    # ---------------- views ----------------
    @property
    def segments(self):
        return list(self._segments)

    @property
    def gaps(self):
        return list(self._gaps)

    @property
    def t0(self):
        """Unix time of the session's first sample (0.0 when empty)."""
        return self.t[0] if self.t else 0.0

    @property
    def total_secret(self):
        return self.secret_total[-1] if self.secret_total else 0

    @property
    def latest_protocol(self):
        return self._segments[-1].protocol if self._segments else None

    @property
    def secret_by_protocol(self):
        """Session secret bits split by which protocol distilled them.

        An unrecognised protocol string is bucketed under 'pm' rather than
        raising -- this feeds a progress bar, not a security claim.

        A segment's own production is secret_banked + secret_final: bits from
        runs before a mid-segment collector respawn (banked into
        secret_offset, see add_sample) plus the current run's latest count.
        Using secret_final alone would drop the banked bits, so this sum
        stays consistent with total_secret across a respawn.
        """
        out = {name: 0 for name in PROTOCOLS}
        for seg in self._segments:
            key = seg.protocol if seg.protocol in out else _DEFAULT_PROTOCOL
            out[key] += seg.secret_banked + seg.secret_final
        return out
