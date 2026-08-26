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
    # EDITED OPERATING POINTS, when the caller has some.
    #
    # None means "load the named preset", which is what every caller did
    # before the visualiser let the parameters be edited, and is still what
    # eval_hybrid and the CLI do. A `channel.EnvParams` /
    # `channel.EntEnvParams`
    # here REPLACES the preset lookup -- the preset NAME is kept alongside it
    # so labels can still say which preset the values started from.
    #
    # The shared fibre is overridden either way (see _setup_modes): an env
    # handed in here does not get to carry its own link.
    pm_env: object = None
    ent_env: object = None


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
        # An explicitly supplied env wins over the preset name; the channel
        # override below applies to BOTH paths, which is what stops an edited
        # operating point from smuggling in its own fibre.
        pm_base = (p.pm_env if p.pm_env is not None
                   else presets.load_presets()[p.pm_preset])
        ent_base = (p.ent_env if p.ent_env is not None
                    else presets.load_ent_presets()[p.ent_preset])
        pm_env = dataclasses.replace(pm_base, d_km=p.d_km, alpha=p.alpha,
                                     odr_losses=p.odr_losses)
        ent_env = dataclasses.replace(ent_base, d_km=p.d_km, alpha=p.alpha,
                                      odr_losses=p.odr_losses)
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
        # kappa() memo: f takes one of three values, so the finite-key
        # routine runs a handful of times per episode instead of 3,600.
        # Keyed on params, not on the episode, so reset() must NOT clear it.
        self._kappa_cache = {}

    def reset(self, seed=None):
        """Start a fresh episode; return the first observation.

        seed=None reseeds from params.seed and leaves Eve on her own
        configured seed, so repeated calls replay ONE trajectory byte for
        byte. That is what lets two envs sharing an EveParams be compared on
        identical jamming, and every determinism guarantee in the tests rests
        on it -- but it also means a training loop that calls reset() bare
        sees the same episode over and over.

        Pass an integer to draw a different episode deterministically: both
        RNG streams and Eve's chain are reseeded from it. Task 6 MUST call
        reset(episode_seed) with a fresh seed per episode, and evaluation MUST
        draw its seeds from a set disjoint from training's -- otherwise the
        agent is scored on trajectories it memorised.

        reset(params.seed) is deliberately NOT identical to reset(): the bare
        form is the only one that leaves Eve's trajectory fixed by EveParams.
        """
        p = self.params
        # Independent streams for errors, arrivals and jamming. Sharing one
        # would make the arrival process depend on how many uniforms numpy's
        # binomial happened to consume for this slot's error draw -- which
        # varies with the mode -- so two policies on the same seed would face
        # different message traffic and could not be compared (nor could an
        # Oracle bound them). SeedSequence(s).generate_state(3) has
        # generate_state(2) as its prefix, so the None path is bit-for-bit
        # what it was before the seed argument existed.
        streams = np.random.SeedSequence(
            p.seed if seed is None else int(seed)).generate_state(3)
        self._rng = np.random.RandomState(streams[0])
        self._arrivals_rng = np.random.RandomState(streams[1])
        if seed is None:
            self.eve.reset()
        else:
            self.eve.reset(int(streams[2]))
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

    def _classical_slot(self):
        """eq (2)-(4) for one slot: arrivals, service, expiry.

        Shared by step() and step_relaxed(). The Oracle relaxation touches
        the QUANTUM side only, so the two must run the same classical code on
        the same arrival stream -- a second copy here is how a bound quietly
        stops bounding the thing it is compared against.

        The order is mandatory: arrive -> serve -> age_and_expire. Any other
        shifts the effective deadline by a slot.
        """
        p = self.params
        arrivals = int(self._arrivals_rng.poisson(p.lam))
        self._queue.arrive(arrivals)
        capacity = self._buffer.level // p.l0
        served = self._queue.serve(min(self._queue.length, capacity))
        self._buffer.withdraw(served * p.l0)
        expired = self._queue.age_and_expire()
        self._served_total += served
        self._arrived_total += arrivals
        return arrivals, served, expired

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
        arrivals, served, expired = self._classical_slot()

        self._last_mode = mode
        self._slot += 1
        done = self._slot >= p.horizon
        info = {"f": f, "eve_state": self.eve.state, "arrivals": arrivals,
                "served": served, "expired": expired,
                "buffer": self._buffer.level, "deposited": deposited,
                "blocks": len(blocks), "mode": mode,
                "qber": self._qber_hist[-1]}
        return self._observe(), float(served), done, info

    # ---------------- the Oracle relaxation (spec section 7) ----------------
    #
    # A negative result -- "switching does not pay" -- is only publishable
    # with a bound on how much it could ever have paid. These two methods are
    # the mechanics of that bound; policies.py supplies the decision rule.
    # Three constraints are dropped, EVERY one of them in the switcher's
    # favour, so no realisable switching policy can exceed what they produce:
    #
    #   1. the mode is chosen from the true f_t, not from the noisy windowed
    #      QBER a real controller sees (eq 16's causality constraint lifted);
    #   2. switching is free -- eq (5) does not destroy the partial block;
    #   3. key arrives continuously at kappa_m(f_t) instead of in n_x-sized
    #      blocks, which can only make key available earlier, never later.
    #
    # The classical side is NOT relaxed: same _classical_slot(), same arrival
    # stream, same B_max clamp.

    def kappa(self, mode, f):
        """eq (12): mean secret bits per slot for `mode` at jamming `f`.

        ns_m * l_m(E_m(f)) / n_x -- one block's key spread over the slots
        that fill the block.

        The error count handed to the finite-key routine is ROUNDED to an
        integer, because that is what it receives at every other call site: a
        finished block carries a whole number of errors. Feeding it the
        fractional expectation instead moves eta in the fourth decimal, which
        would leave the bound on a slightly different footing from the blocks
        it is meant to bound.

        The one thing the relaxation cannot wave away: this is an
        EXPECTATION, while a real block's key length comes from its realised
        binomial error count, and l(q) is convex -- so a lucky block can
        out-earn kappa. Measured over 4,000 (PM) and 20,000 (ENT) slot runs
        at f = 0.00 and 0.03, the block rate lands 0.001-0.05% BELOW kappa
        (the uncounted final partial block dominates), so the bound holds
        comfortably in practice. It is not an almost-sure inequality, and a
        report that needs one should say so.
        """
        key = (int(mode), float(f))
        if key not in self._kappa_cache:
            n = self.params.n_x
            e_eff = self._e_floor[mode] * (1.0 - f) + 0.5 * f
            errors = int(round(min(e_eff, 1.0) * n))
            bits = self._key_length(mode, n, errors)
            self._kappa_cache[key] = self._sifted[mode] * bits / n
        return self._kappa_cache[key]

    def step_relaxed(self, chooser):
        """One slot under the relaxation above; same (obs, reward, done, info).

        `chooser(f, rates, obs) -> mode` decides the slot AFTER f_t is drawn.
        That signature is what makes clairvoyance structural rather than a
        promise in a comment: relaxation 1 is the `f` argument, and a chooser
        that ignores it and reads `obs` instead is exactly the realisable
        policy this bound is supposed to sit above. `rates` is
        (kappa_PM, kappa_ENT) at this slot's f.

        The QBER window is still filled from a real binomial draw, so `obs`
        is the observation an unrelaxed agent would have had at this slot and
        such a chooser is a fair comparand. `obs[I_BLOCK]` is always 0:
        under relaxation 3 there is no partial block to lose.
        """
        p = self.params
        obs = self._observe()
        f = self.eve.step()
        rates = (self.kappa(PM, f), self.kappa(ENT, f))
        mode = ENT if int(chooser(f, rates, obs)) == ENT else PM

        sifted = self._sifted[mode]
        e_eff = self._e_floor[mode] * (1.0 - f) + 0.5 * f
        errors = int(self._rng.binomial(sifted, min(e_eff, 1.0)))
        self._qber_hist.append(errors / sifted if sifted else 0.0)

        # Relaxations 2 and 3 together: no BlockAccumulator at all, so a
        # switch costs nothing and this slot's key is banked in this slot.
        deposited = rates[mode]
        self._buffer.deposit(deposited)

        arrivals, served, expired = self._classical_slot()

        self._last_mode = mode
        self._slot += 1
        done = self._slot >= p.horizon
        info = {"f": f, "eve_state": self.eve.state, "arrivals": arrivals,
                "served": served, "expired": expired,
                "buffer": self._buffer.level, "deposited": deposited,
                "blocks": 0, "mode": mode, "qber": self._qber_hist[-1],
                "rates": rates}
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
    def served_total(self):
        """Messages served so far this episode (numerator of eq 24)."""
        return self._served_total

    @property
    def arrived_total(self):
        """Messages that arrived so far this episode (denominator of eq 24)."""
        return self._arrived_total

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
