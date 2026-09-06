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
from mobility import MobilityParams, VesselTrack

PM, ENT = 0, 1

OBS_DIM = 7
# Layout of the observation vector (eq 16). Defined here because _observe()
# is what fixes the order; policies.py re-exports these so no policy ever
# indexes the state with a bare integer.
I_QBER, I_MODE, I_BLOCK, I_BUFFER, I_QUEUE, I_AGE, I_DIST = range(OBS_DIM)

# Divisor for the distance feature. A fixed constant, not the mobility band:
# the band is a sweep parameter, and normalising by it would make the same
# physical distance read as a different number in two runs -- and would be
# degenerate (0/0) with mobility off, where the distance is a constant. 250 km
# is just past the furthest range at which ANY mode in this study still
# distils key (ENT/ent_advanced dies at 247.3 km), so the feature sits in
# roughly [0, 1] like the other bounded entries. Distances beyond it are NOT
# clipped: a feature that saturates would hide the very region the mobility
# study is about.
DIST_REF_KM = 250.0


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
    # A MOBILE SSV, when the caller wants one.
    #
    # None (and MobilityParams(enabled=False)) means the link length is fixed
    # at d_km, which is what every result before this existed assumed. With a
    # track the distance changes each slot and BOTH modes' physics follow it.
    mobility: object = None
    # Distance quantisation for the per-mode physics cache, in km. Resolving a
    # slot's physics means channel.derive()/derive_ent() plus a finite-key
    # evaluation; at 3,600 slots x hundreds of episodes that is the dominant
    # cost of the whole study, so distances are snapped to a grid and each
    # grid point is resolved once (see _physics_at). DECLARED APPROXIMATION:
    # the vessel gets the physics of a point at most d_grid_km/2 away.
    # <= 0 disables snapping (exact physics, every distinct distance
    # recomputed) and is a legal, slow sweep point.
    #
    # Snapping applies ONLY when a track is running. With mobility off the
    # exact d_km is used, so a fixed-distance env is bit-for-bit what it was
    # before this field existed, whatever d_km happens to be.
    d_grid_km: float = 0.5


class HybridEnv:
    """eq (1)-(24). step() advances one slot and returns the usual RL tuple.

    Per-mode physics comes from the verified channel/keyrate modules: the
    sifted rate ns_m and the intrinsic error floor e_m. On a FIXED link it is
    resolved once at construction and a slot costs a binomial draw plus
    (rarely) one finite-key evaluation. With a MOBILE SSV (params.mobility) it
    depends on the distance too, so it is resolved once per DISTANCE GRID
    POINT and memoised -- see _physics_at and HybridParams.d_grid_km for the
    approximation that buys.

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
        """Resolve the per-mode physics machinery.

        Both modes ride the SAME fibre, so d_km/alpha/odr_losses override
        whatever the presets carried. Comparing presets with different
        channels is the trap CLAUDE.md records having already been published
        once and retracted.

        With a mobile SSV the distance is no longer a constant, so what is
        fixed here is the pair of *base* operating points (everything except
        the link length) plus the caches; the distance-dependent quantities
        are resolved per grid point by _physics_at() and installed by
        _set_distance().
        """
        p = self.params
        # An explicitly supplied env wins over the preset name; the channel
        # override below applies to BOTH paths, which is what stops an edited
        # operating point from smuggling in its own fibre.
        pm_base = (p.pm_env if p.pm_env is not None
                   else presets.load_presets()[p.pm_preset])
        ent_base = (p.ent_env if p.ent_env is not None
                    else presets.load_ent_presets()[p.ent_preset])
        self._pm_base = dataclasses.replace(pm_base, alpha=p.alpha,
                                            odr_losses=p.odr_losses)
        self._ent_base = dataclasses.replace(ent_base, alpha=p.alpha,
                                             odr_losses=p.odr_losses)
        mobility = p.mobility
        if isinstance(mobility, MobilityParams) and not mobility.enabled:
            mobility = None
        self._track = (VesselTrack(mobility, d_start=p.d_km)
                       if mobility is not None else None)
        # Both caches are keyed on the PARAMETERS, never on the episode, so
        # reset() must NOT clear either of them: a 200-episode training run
        # would otherwise re-derive the same grid points 200 times over.
        self._physics_cache = {}
        # kappa() memo: f takes one of three values and the distance one of a
        # handful of grid points, so the finite-key routine runs a few dozen
        # times per episode instead of 3,600.
        self._kappa_cache = {}
        self._physics_key = None
        self._distance = None
        # The track's start is d_km CLAMPED into the patrol band, so ask the
        # track rather than assuming d_km: otherwise construction resolves --
        # and caches -- a grid point the vessel may never visit.
        self._set_distance(p.d_km if self._track is None
                           else self._track.distance)

    def _grid_key(self, d_km):
        """Cache key for `d_km`, and the distance that key stands for.

        Returns (key, d_used). With no track running the key is the exact
        distance and d_used is it unchanged -- the fixed-link case must not
        move by so much as a metre. With a track, the key is the integer grid
        index, so two distances in the same cell share one physics.
        """
        step = self.params.d_grid_km
        if self._track is None or step <= 0:
            return float(d_km), float(d_km)
        index = int(round(float(d_km) / step))
        return index, index * step

    def _physics_at(self, d_km):
        """(sifted, e_floor, key_params) for the link at `d_km`, memoised.

        This is the expensive call the grid exists to amortise:
        channel.derive() runs the decoy gains and a finite-key evaluation,
        channel.derive_ent() the BBM92 coincidence model.
        """
        key, d_used = self._grid_key(d_km)
        cached = self._physics_cache.get(key)
        if cached is None:
            p = self.params
            pm_env = dataclasses.replace(self._pm_base, d_km=d_used)
            ent_env = dataclasses.replace(self._ent_base, d_km=d_used)
            d_pm = channel.derive(pm_env, p.n_x)
            d_ent = channel.derive_ent(ent_env, p.n_x)
            # pdc is passed explicitly: KeyParams defaults it, and a preset
            # with a different dark-count rate would otherwise be silently
            # ignored here while channel.derive() honoured it.
            key_params = keyrate.KeyParams(mu1=pm_env.mu1, mu2=pm_env.mu2,
                                           p1=pm_env.p1, eta=d_pm.eta,
                                           pdc=pm_env.pdc)
            # PM keys the Z basis only (a quarter of clicks); ENT keys both
            # sifted bases (half). Getting this wrong doubles ENT's apparent
            # rate.
            sifted = (int(d_pm.clicks_per_s * 0.25 * p.delta_t),
                      int(d_ent.clicks_per_s * 0.5 * p.delta_t))
            cached = (sifted, (d_pm.e_avg, d_ent.q2), key_params)
            self._physics_cache[key] = cached
        return key, cached

    def _set_distance(self, d_km):
        """Install the physics for `d_km`.

        Cheap when the cell has not changed, which is the common case: at
        0.0129 km/slot the vessel crosses a 0.5 km cell boundary about once
        every 39 slots.
        """
        self._distance = float(d_km)
        key, (sifted, e_floor, key_params) = self._physics_at(d_km)
        self._physics_key = key
        self._sifted = sifted
        self._e_floor = e_floor
        self._key_params = key_params

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
        # generate_state(n) is a prefix-stable expansion, so widening this
        # from 3 to 4 leaves the error, arrival and jamming streams bit-for-
        # bit what they were before the vessel existed: every trajectory the
        # earlier study recorded still replays exactly.
        streams = np.random.SeedSequence(
            p.seed if seed is None else int(seed)).generate_state(4)
        self._rng = np.random.RandomState(streams[0])
        self._arrivals_rng = np.random.RandomState(streams[1])
        if seed is None:
            self.eve.reset()
        else:
            self.eve.reset(int(streams[2]))
        # The track rewinds with the episode; the physics CACHE does not -- it
        # is keyed on the distance grid and outlives every episode.
        if self._track is not None:
            self._set_distance(self._track.reset(
                None if seed is None else int(streams[3])))
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
        """s_t of eq (16) -- nothing here may depend on slot t or later.

        The distance is the one exogenous quantity the policy DOES see (a
        vessel knows its own position); Eve's f_t is not, and never appears
        here. The value reported is the distance the slot just ended at, so a
        policy acting on it is at most one slot -- 12.9 m at the default speed
        limit -- behind the vessel.
        """
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
            self._distance / DIST_REF_KM,             # d_t / 250 km
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
        # The vessel moves first, so this slot's physics is the physics of
        # where it is now -- exactly as f_t is drawn before it is applied.
        if self._track is not None:
            self._set_distance(self._track.step())
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
                "qber": self._qber_hist[-1], "d_km": self._distance}
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
        # Keyed on the distance grid point as well as (mode, f): with a
        # mobile SSV the same jamming level buys a different number of bits at
        # 40 km and at 200 km, and a two-part key would hand the bound the
        # physics of the first distance the episode happened to visit.
        key = (int(mode), float(f), self._physics_key)
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
        if self._track is not None:
            self._set_distance(self._track.step())
        # _observe() AFTER the move: the relaxed chooser is already
        # clairvoyant about f_t, and handing it a stale distance as well would
        # make the bound weaker than the policies it has to sit above.
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
                "rates": rates, "d_km": self._distance}
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
    def distance_km(self):
        """The link length this slot ran at [km]. Constant unless a
        MobilityParams with enabled=True was supplied."""
        return self._distance

    @property
    def track(self):
        """The VesselTrack driving the distance, or None when the link is
        fixed. A read-only view: nothing outside may step it."""
        return self._track

    @property
    def physics_cache_size(self):
        """Distinct distance grid points resolved so far. The whole point of
        the grid is that this stays small; a test asserts it."""
        return len(self._physics_cache)

    @property
    def kappa_cache_size(self):
        """Distinct (mode, f, distance) points kappa() has evaluated.

        A model-based policy calls kappa() with an ESTIMATED f, so unlike the
        Oracle's three exact levels its keys are only as few as its
        quantisation makes them; a test watches this rather than the wall
        clock, because a miss costs a finite-key evaluation.
        """
        return len(self._kappa_cache)

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
