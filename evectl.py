#!/usr/bin/env python3
"""EveController: duty-cycle hw_sim's SIGUSR1/2 jamming to realise a fractional
jamming rate f, and log the attack episodes.

hw_sim only has a binary jam switch (SIGUSR1 = QBER->50%, SIGUSR2 = off). To get
an average jamming fraction f in (0,1) we toggle it: each period of length
period_s we hold the jam on for f*period_s and off for the rest. A collector CSV
row spans ~20 periods, so the row-averaged QBER lands on eve.effective_qber(f).

The controller owns a daemon thread. stop() always leaves the simulator UNjammed
(a final SIGUSR2) and joins the thread, so no jam state or stray timer survives.
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "logs", "eve.csv")


@dataclass
class Episode:
    t_start: float
    t_end: float   # 0.0 while the episode is still open (Eve currently ON)
    f: float


def _default_sender(sig):
    """Send `sig` to every running simulator process (both Alice and Bob)."""
    out = subprocess.run(["pgrep", "-x", "simulator"],
                         stdout=subprocess.PIPE).stdout.decode()
    for pid in out.split():
        try:
            os.kill(int(pid), sig)
        except (ProcessLookupError, ValueError):
            pass


class EveController:
    """Fractional jamming via a background duty-cycle thread."""

    def __init__(self, signal_sender=None, period_s=0.1, csv_path=None):
        self._send = signal_sender or _default_sender
        self.period_s = period_s
        self.csv_path = csv_path or DEFAULT_CSV
        self._thread = None
        self._stop_evt = threading.Event()
        self._f = 0.0
        self._lock = threading.Lock()
        self._episodes = []        # list[Episode]; last may be open (t_end 0)

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_f(self):
        return self._f

    def episodes(self):
        with self._lock:
            return list(self._episodes)

    def start(self, f):
        """Begin (or adjust) jamming at fraction f. Opens a new episode."""
        # Validate through eve.EveParams-equivalent range check.
        if not 0.0 <= f <= 1.0:
            raise ValueError("jamming fraction f must be in [0, 1], got %r" % f)
        if self.active:
            # Adjust in place: close the old episode, open a new one at new f.
            self._close_episode()
        self._f = f
        with self._lock:
            self._episodes.append(Episode(time.time(), 0.0, f))
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        on = self.period_s * self._f
        off = self.period_s - on
        while not self._stop_evt.is_set():
            if on > 0:
                self._send(signal.SIGUSR1)
                if self._stop_evt.wait(on):
                    break
            if off > 0:
                self._send(signal.SIGUSR2)
                if self._stop_evt.wait(off):
                    break

    def _close_episode(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            if self._episodes and self._episodes[-1].t_end == 0.0:
                ep = self._episodes[-1]
                ep.t_end = time.time()
                self._append_csv(ep)

    def stop(self):
        """Stop jamming: leave the simulator unjammed, join the thread. Idempotent."""
        was_active = self.active
        self._close_episode()
        # Always end in the clean state, even if we were never active (harmless).
        self._send(signal.SIGUSR2)
        self._f = 0.0
        return was_active

    def _append_csv(self, ep):
        new = not os.path.exists(self.csv_path)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "a") as f:
            if new:
                f.write("t_start,t_end,f\n")
            f.write("%.3f,%.3f,%.4f\n" % (ep.t_start, ep.t_end, ep.f))
