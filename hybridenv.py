#!/usr/bin/env python3
"""Hybrid PM/ENT mode-switching environment -- eq (1)-(24) of
system_model_v3_5.docx.

Analytic, not the qline stack: training needs ~1e6 slots and a slot is one
second of wall clock on the real hardware. Key lengths still come from the
verified keyrate/keyrate_ent modules, and channel parameters from channel.py,
so no security formula is reimplemented here.

Spec: docs/superpowers/specs/2026-08-17-hybrid-policy-design.md
"""

import dataclasses
from collections import deque
from dataclasses import dataclass

import numpy as np

import channel
import keyrate
import keyrate_ent
import presets
from evemarkov import EveChain, EveParams

PM, ENT = 0, 1

OBS_DIM = 6
# Layout of the observation vector (eq 16). Defined here because _observe()
# is what fixes the order; policies.py re-exports these so no policy ever
# indexes the state with a bare integer.
I_QBER, I_MODE, I_BLOCK, I_BUFFER, I_QUEUE, I_AGE = range(OBS_DIM)


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


@dataclass
class HybridParams:
    """Everything the numeric evaluation can sweep. Spec section 4."""

    delta_t: float = 1.0            # slot length [s]
    n_x: int = 100_000              # finite-key block size [sifted bits]
    l0: int = 8192                  # key bits to encrypt one message (OTP)
    lam: float = 1.0                # mean arrivals per slot (Poisson)
    t0_deadline: int = 20           # message deadline [slots]
    b_max: int = 10_000_000         # key buffer capacity [bits]
    w_window: int = 5               # QBER estimation window [slots]
    horizon: int = 3600             # slots per episode
    d_km: float = 55.0              # shared fibre, both modes
    alpha: float = 0.2
    odr_losses: float = 0.0
    seed: int = 0
    # Which measured operating point each mode runs at. chapman_lowmu (mu =
    # 0.015) rather than chapman_current: at mu = 0.045 ENT dies *before* PM
    # does, so eq (14)'s crossover q* would not exist (spec section 2, #2).
    pm_preset: str = "aurea"
    ent_preset: str = "chapman_lowmu"


class HybridEnv:
    """eq (1)-(24). step() advances one slot and returns the usual RL tuple.

    Per-mode physics is resolved ONCE at construction from the verified
    channel/keyrate modules: the sifted rate ns_m and the intrinsic error
    floor e_m. Per slot only the jamming fraction changes, so a step costs a
    binomial draw plus (rarely) one finite-key evaluation.

    Deliberate divergence from the qline stack: switching mode costs only the
    partial block (eq 20), not the ~2.28 s the stack needs to restart. The
    document's cost model has no dead time, and the measured 2.28 s is an
    artifact of restarting Python/Rust processes rather than a property of a
    hybrid device (spec section 8).
    """

    def __init__(self, params=None, eve_params=None):
        self.params = params or HybridParams()
        self.eve = EveChain(eve_params or EveParams())
        self._setup_modes()
        self.reset()

    def _setup_modes(self):
        """Resolve both modes' per-slot physics from the verified modules.

        Both modes ride the SAME fibre, so d_km/alpha/odr_losses override
        whatever the presets carried. Comparing presets with different
        channels is the trap CLAUDE.md records having already been published
        once and retracted.
        """
        p = self.params
        pm_env = dataclasses.replace(presets.load_presets()[p.pm_preset],
                                     d_km=p.d_km, alpha=p.alpha,
                                     odr_losses=p.odr_losses)
        ent_env = dataclasses.replace(
            presets.load_ent_presets()[p.ent_preset],
            d_km=p.d_km, alpha=p.alpha, odr_losses=p.odr_losses)
        d_pm = channel.derive(pm_env, p.n_x)
        d_ent = channel.derive_ent(ent_env, p.n_x)
        # pdc is passed explicitly: KeyParams defaults it, and a preset with a
        # different dark-count rate would otherwise be silently ignored here
        # while channel.derive() honoured it.
        self._key_params = keyrate.KeyParams(mu1=pm_env.mu1, mu2=pm_env.mu2,
                                             p1=pm_env.p1, eta=d_pm.eta,
                                             pdc=pm_env.pdc)
        # PM keys the Z basis only (a quarter of clicks); ENT keys both
        # sifted bases (half). Getting this wrong doubles ENT's apparent rate.
        self._sifted = (int(d_pm.clicks_per_s * 0.25 * p.delta_t),
                        int(d_ent.clicks_per_s * 0.5 * p.delta_t))
        self._e_floor = (d_pm.e_avg, d_ent.q2)

    def reset(self):
        """Start a fresh episode; return the first observation."""
        p = self.params
        # Two independent streams. Sharing one would make the arrival process
        # depend on how many uniforms numpy's binomial happened to consume for
        # this slot's error draw -- which varies with the mode -- so two
        # policies on the same seed would face different message traffic and
        # could not be compared (nor could an Oracle bound them).
        err_seed, arr_seed = np.random.SeedSequence(p.seed).generate_state(2)
        self._rng = np.random.RandomState(err_seed)
        self._arrivals_rng = np.random.RandomState(arr_seed)
        self.eve.reset()
        self._queue = MessageQueue(p.t0_deadline)
        self._block = BlockAccumulator(p.n_x)
        self._buffer = KeyBuffer(p.b_max)
        self._slot = 0
        self._served_total = 0
        self._arrived_total = 0
        self._last_mode = PM
        self._qber_hist = deque(maxlen=max(1, p.w_window))
        return self._observe()

    def _observe(self):
        """s_t of eq (16) -- nothing here may depend on slot t or later."""
        p = self.params
        q = (sum(self._qber_hist) / len(self._qber_hist)
             if self._qber_hist else 0.0)
        q_ref = max(p.lam * p.t0_deadline, 1.0)
        # T0 = 0 (serve on arrival or never) is a legal sweep point; keep it
        # from turning the urgency feature into a division by zero.
        age_ref = max(p.t0_deadline, 1)
        return np.array([
            q,                                        # sliding-window QBER
            float(self._last_mode),                   # m_{t-1}
            self._block.progress / p.n_x,             # b_t / n_X
            self._buffer.level / p.b_max,             # B_t / B_max
            min(self._queue.length / q_ref, 2.0),     # queue pressure
            self._queue.head_age / age_ref,           # urgency
        ], dtype=float)

    def _key_length(self, mode, n, errors):
        """Secret bits from one finished block -- eq (8) for PM, (9) for ENT."""
        qber = errors / n if n else 0.0
        if mode == ENT:
            return keyrate_ent.secret_key_length_ent(n, qber) or 0
        return keyrate.secret_key_length(n, qber, self._key_params) or 0

    def step(self, action):
        """Advance one slot under `action` (PM or ENT).

        Returns (obs, reward, done, info): reward is U_t, the messages served
        before their deadline in this slot (eq 17), and info carries the
        quantities the policy may NOT see -- f_t above all.
        """
        p = self.params
        mode = ENT if int(action) == ENT else PM

        # --- quantum side: this slot's harvest under the current jamming ---
        f = self.eve.step()
        e_eff = self._e_floor[mode] * (1.0 - f) + 0.5 * f
        sifted = self._sifted[mode]
        # Errors are a binomial draw, so the QBER estimate is noisy exactly in
        # proportion to the sample size -- no invented observation noise.
        errors = int(self._rng.binomial(sifted, min(e_eff, 1.0)))
        self._qber_hist.append(errors / sifted if sifted else 0.0)

        # A rich slot can pay for several blocks at once, so add() returns
        # every block it retired (empty list when none did).
        deposited = 0
        blocks = self._block.add(mode, sifted, errors)
        for n_block, block_errors in blocks:
            bits = self._key_length(mode, n_block, block_errors)
            deposited += bits
            self._buffer.deposit(bits)

        # --- classical side: arrivals, service, expiry (eq 2-4) ---
        # This order is mandatory: arrive -> serve -> age_and_expire. Any
        # other shifts the effective deadline by a slot.
        arrivals = int(self._arrivals_rng.poisson(p.lam))
        self._queue.arrive(arrivals)
        capacity = self._buffer.level // p.l0
        served = self._queue.serve(min(self._queue.length, capacity))
        self._buffer.withdraw(served * p.l0)
        expired = self._queue.age_and_expire()

        self._served_total += served
        self._arrived_total += arrivals
        self._last_mode = mode
        self._slot += 1
        done = self._slot >= p.horizon
        info = {"f": f, "eve_state": self.eve.state, "arrivals": arrivals,
                "served": served, "expired": expired,
                "buffer": self._buffer.level, "deposited": deposited,
                "blocks": len(blocks), "mode": mode,
                "qber": self._qber_hist[-1]}
        return self._observe(), float(served), done, info

    @property
    def eta(self):
        """eq (24): fraction of arrivals served before their deadline."""
        if self._arrived_total == 0:
            return 0.0
        return self._served_total / self._arrived_total

    # Read-only views of the internal state. Tests and the analysis scripts
    # need these; nothing outside may mutate them.

    @property
    def sifted_per_slot(self):
        """(PM, ENT) sifted bits harvested per slot at this channel."""
        return self._sifted

    @property
    def error_floor(self):
        """(PM, ENT) intrinsic error rates e_m with no jamming."""
        return self._e_floor

    @property
    def buffer_level(self):
        return self._buffer.level

    @property
    def queue_length(self):
        return self._queue.length

    @property
    def queue_head_age(self):
        return self._queue.head_age

    @property
    def block_progress(self):
        return self._block.progress

    @property
    def block_mode(self):
        return self._block.mode
