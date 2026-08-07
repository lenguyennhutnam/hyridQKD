#!/usr/bin/env python3
"""EveController: duty-cycle hw_sim's SIGUSR1/2 jamming to realise a fractional
jamming rate f, and log the attack episodes.

hw_sim only has a binary jam switch (SIGUSR1 = QBER->50%, SIGUSR2 = off). To get
an average jamming fraction f in (0,1) we toggle it: each period of length
period_s we hold the jam on for f*period_s and off for the rest. A collector CSV
row spans ~20 periods, so the row-averaged QBER lands on eve.effective_qber(f).

The controller owns a daemon thread. stop() always leaves the simulator UNjammed
(a final SIGUSR2) and joins the thread, so no jam state or stray timer survives.

Concurrency notes:
- Each duty-cycle thread gets its OWN threading.Event (created in start() and
  passed into _run as an argument), instead of sharing one Event on self. This
  means a thread that fails to join within the timeout (e.g. because
  signal_sender is slow/blocked) is waiting on an Event nobody else touches
  anymore: a later start() creates a brand-new Event for the new thread, so it
  can never accidentally "resurrect" the stale thread by clearing a shared
  flag it's still waiting on.
- Each thread also receives its `f` as an argument rather than reading
  self._f, so it is fully independent of self._f once started; self._f is
  purely a status field for current_f() and can be safely written under the
  lock without racing the thread's behaviour.
- All mutation of self._f / self._thread / self._stop_evt is done while
  holding self._lock (blocking calls like Event.set()/thread.join() are done
  OUTSIDE the lock, using locally-captured references, so slow joins never
  block other callers of active()/current_f()/episodes()).
"""

import os
import signal
import subprocess
import sys
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
                         stdout=subprocess.PIPE, timeout=2.0).stdout.decode()
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
        self._stop_evt = None      # this thread's OWN Event, or None if idle
        self._zombies = []         # threads that didn't join within timeout
        self._f = 0.0
        self._lock = threading.Lock()
        self._episodes = []        # list[Episode]; last may be open (t_end 0)

    @property
    def active(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def current_f(self):
        with self._lock:
            return self._f

    def episodes(self):
        with self._lock:
            return list(self._episodes)

    def start(self, f):
        """Begin (or adjust) jamming at fraction f. Opens a new episode."""
        # Validate through eve.EveParams-equivalent range check.
        if not 0.0 <= f <= 1.0:
            raise ValueError("jamming fraction f must be in [0, 1], got %r" % f)
        # Always close out whatever's currently running first (no-op if idle).
        # This also avoids a check-then-act race on `active`.
        self._close_episode()
        stop_evt = threading.Event()
        thread = threading.Thread(target=self._run, args=(f, stop_evt),
                                  daemon=True)
        with self._lock:
            self._f = f
            self._stop_evt = stop_evt
            self._thread = thread
            self._episodes.append(Episode(time.time(), 0.0, f))
        thread.start()

    def _run(self, f, stop_evt):
        """Duty-cycle loop for one episode. Reads only its own args, never
        shared mutable controller state, so it can't be disturbed by a
        concurrent start()/stop() touching self._f or self._stop_evt."""
        on = self.period_s * f
        off = self.period_s - on
        while not stop_evt.is_set():
            if on > 0:
                self._send(signal.SIGUSR1)
                if stop_evt.wait(on):
                    break
            if off > 0:
                self._send(signal.SIGUSR2)
                if stop_evt.wait(off):
                    break

    def _close_episode(self):
        with self._lock:
            thread = self._thread
            stop_evt = self._stop_evt
            self._thread = None
            self._stop_evt = None
        if stop_evt is not None:
            stop_evt.set()
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                # The thread is still running (e.g. blocked in a slow
                # signal_sender). It owns its own Event, already set, so it
                # will exit as soon as its current send/wait returns -- it
                # just won't be reflected in self._thread anymore. Keep a
                # reference instead of silently dropping it.
                with self._lock:
                    self._zombies.append(thread)
                print("evectl: warning: duty-cycle thread did not join within "
                     "2.0s timeout; it will keep exiting on its own private "
                     "stop event and will not affect future start()/stop() "
                     "calls", file=sys.stderr)
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
        with self._lock:
            self._f = 0.0
        return was_active

    def _append_csv(self, ep):
        new = not os.path.exists(self.csv_path)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "a") as f:
            if new:
                f.write("t_start,t_end,f\n")
            f.write("%.3f,%.3f,%.4f\n" % (ep.t_start, ep.t_end, ep.f))
