#!/usr/bin/env python3
"""Hybrid PM/ENT mode-switching environment -- eq (1)-(24) of
system_model_v3_5.docx.

Analytic, not the qline stack: training needs ~1e6 slots and a slot is one
second of wall clock on the real hardware. Key lengths still come from the
verified keyrate/keyrate_ent modules, and channel parameters from channel.py,
so no security formula is reimplemented here.

Spec: docs/superpowers/specs/2026-08-17-hybrid-policy-design.md
"""

from collections import deque


class MessageQueue:
    """FIFO of messages that expire T0 slots after arrival (eq 3-4).

    Every message carries the same deadline, so ages form a sorted run and
    the queue can be stored as counts bucketed by age rather than as
    individual messages -- O(T0) per slot instead of O(queue length).
    """

    def __init__(self, deadline):
        self.deadline = int(deadline)
        # _ages[i] = number of waiting messages whose age is i.
        self._ages = deque([0] * (self.deadline + 1), maxlen=self.deadline + 1)

    def arrive(self, n):
        """n new messages enter at age 0."""
        self._ages[0] += int(n)

    def serve(self, n):
        """Serve up to n messages, oldest first; return how many were served."""
        want = int(n)
        served = 0
        for i in range(len(self._ages) - 1, -1, -1):
            if want <= 0:
                break
            take = min(self._ages[i], want)
            self._ages[i] -= take
            served += take
            want -= take
        return served

    def age_and_expire(self):
        """End of slot: everyone ages one slot; those past T0 leave (eq 3).

        Returns how many expired.
        """
        expired = self._ages[self.deadline]
        self._ages.appendleft(0)      # maxlen drops the oldest bucket
        return int(expired)

    @property
    def length(self):
        return int(sum(self._ages))

    @property
    def head_age(self):
        for i in range(len(self._ages) - 1, -1, -1):
            if self._ages[i] > 0:
                return i
        return 0


class BlockAccumulator:
    """Sifted bits piling up toward finite-key blocks (eq 5-7).

    A block must be homogeneous in protocol, so changing mode discards
    whatever was accumulated. That is the switching cost of the whole
    problem: it is intrinsic, not a penalty term bolted on.

    One slot's harvest is not bounded by n_x -- at short range PM sifts
    hundreds of thousands of bits per slot against an n_X of 1e5 -- so `add`
    returns a *list* of every block the pile can pay for, and keeps only the
    sub-block remainder. `progress / n_x` therefore always lies in [0, 1),
    which is what makes "a switch discards a partial block" true rather than
    "a switch discards fifteen finished blocks".
    """

    def __init__(self, n_x):
        self.n_x = int(n_x)
        self._progress = 0
        self._errors = 0
        self._mode = None

    def add(self, mode, sifted, errors):
        """Add one slot's harvest under `mode`.

        Returns a list [(n_x, block_errors), ...] of the blocks this slot
        completed -- empty when none did, and longer than one entry when the
        slot was rich enough to pay for several. The remainder and its
        proportional share of the errors carry over to the next block.
        """
        if self._mode is not None and mode != self._mode:
            self._progress = 0        # eq (5): indicator 1{m_t == m_{t-1}}
            self._errors = 0
        self._mode = mode
        self._progress += int(sifted)
        self._errors += int(errors)
        blocks = []
        while self._progress >= self.n_x:
            # Split errors in proportion to bits, so no block is flattered.
            block_errors = int(round(self._errors * self.n_x / self._progress))
            self._progress -= self.n_x
            self._errors -= block_errors
            blocks.append((self.n_x, block_errors))
        return blocks

    @property
    def progress(self):
        return self._progress

    @property
    def errors(self):
        return self._errors

    @property
    def mode(self):
        return self._mode


class KeyBuffer:
    """The key pool (eq 11/22): saturates at B_max, never goes negative."""

    def __init__(self, b_max):
        self.b_max = int(b_max)
        self._level = 0

    def deposit(self, bits):
        """Add distilled key; anything above B_max is lost.

        Clamped at 0 from below too: a pool driven negative would make every
        later withdrawal raise, stopping service for good.
        """
        self._level = min(self.b_max, max(0, self._level + int(bits)))

    def withdraw(self, bits):
        """Consume key. Callers must size the withdrawal from `level` first
        (eq 2 does exactly that), so asking for more is a caller bug."""
        bits = int(bits)
        if bits > self._level:
            raise ValueError("withdraw %d from a buffer holding %d"
                             % (bits, self._level))
        self._level -= bits

    @property
    def level(self):
        return self._level
