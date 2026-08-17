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
