"""A mobile SSV inside the hybrid environment.

Three things have to be true at once and each is pinned separately:

1. with mobility OFF nothing changed -- the fixed-link results predating the
   vessel still mean what they meant;
2. with mobility ON the per-mode physics really follows the distance;
3. the distance GRID that makes (2) affordable costs a bounded, declared
   amount of accuracy -- and is actually used.
"""
import dataclasses
import unittest

import numpy as np

import channel
import hybridenv
from evemarkov import EveParams
from hybridenv import (DIST_REF_KM, ENT, I_DIST, PM, HybridEnv, HybridParams)
from mobility import MobilityParams

QUIET_EVE = EveParams(f_levels=(0.0, 0.0, 0.0), seed=1)

# A band around PM's death distance for the default operating point (PM/aurea
# stops distilling at 167.2 km with alpha = 0.2, odr = 0, n_X = 1e5), and a
# speed fast enough that a short test episode crosses it.
CROSSING = MobilityParams(enabled=True, d_min=150.0, d_max=190.0, v_max=0.5,
                          loiter_min=0, loiter_max=10, seed=5)


def mobile_params(**kw):
    base = dict(horizon=300, seed=3, mobility=CROSSING, d_km=170.0)
    base.update(kw)
    return HybridParams(**base)


class TestMobilityIsOffByDefault(unittest.TestCase):
    def test_the_default_environment_has_no_track(self):
        self.assertIsNone(HybridEnv(HybridParams(horizon=5)).track)

    def test_an_explicitly_disabled_track_is_no_track_at_all(self):
        # MobilityParams(enabled=False) must be as inert as None, or a caller
        # that builds the params object and leaves the flag alone starts the
        # vessel moving by accident.
        env = HybridEnv(HybridParams(horizon=5,
                                     mobility=MobilityParams(enabled=False)))
        self.assertIsNone(env.track)

    def test_a_fixed_link_never_changes_distance_or_physics(self):
        p = HybridParams(horizon=200, seed=3, d_km=55.0)
        env = HybridEnv(p)
        physics = (env.sifted_per_slot, env.error_floor)
        for _ in range(200):
            _, _, _, info = env.step(PM)
            self.assertEqual(info["d_km"], 55.0)
        self.assertEqual((env.sifted_per_slot, env.error_floor), physics)
        # One distance visited, so one physics resolved, for the whole run.
        self.assertEqual(env.physics_cache_size, 1)

    def test_a_fixed_link_uses_the_exact_d_km_not_a_grid_point(self):
        # MUTATION: quantise unconditionally. 55.3 km snaps to 55.5 on the
        # default 0.5 km grid, so every fixed-link number in the study would
        # move -- silently, and only for distances that miss the grid.
        d = 55.3
        env = HybridEnv(HybridParams(horizon=5, d_km=d, d_grid_km=0.5))
        pm = channel.derive(dataclasses.replace(
            env._pm_base, d_km=d), env.params.n_x)
        ent = channel.derive_ent(dataclasses.replace(
            env._ent_base, d_km=d), env.params.n_x)
        self.assertEqual(env.distance_km, d)
        self.assertEqual(env.sifted_per_slot,
                         (int(pm.clicks_per_s * 0.25), int(ent.clicks_per_s
                                                           * 0.5)))
        self.assertEqual(env.error_floor, (pm.e_avg, ent.q2))

    def test_a_fixed_link_run_is_identical_with_and_without_a_disabled_track(
            self):
        def trace(mobility):
            env = HybridEnv(HybridParams(horizon=120, seed=4, d_km=55.0,
                                         mobility=mobility), QUIET_EVE)
            env.reset(17)
            out = []
            for t in range(120):
                obs, r, _, info = env.step(ENT if t % 5 == 0 else PM)
                out.append((tuple(obs), r, info["qber"], info["buffer"],
                            info["deposited"]))
            return out
        self.assertEqual(trace(None), trace(MobilityParams(enabled=False)))

    def test_the_fourth_random_stream_did_not_disturb_the_first_three(self):
        # reset() widened SeedSequence.generate_state(3) to (4) to seed the
        # track. generate_state is prefix-stable, so every trajectory the
        # earlier study recorded still replays -- but that is a property of
        # numpy, not of our wishes, so assert it.
        for seed in (0, 3, 99_000_007):
            wide = np.random.SeedSequence(seed).generate_state(4)
            narrow = np.random.SeedSequence(seed).generate_state(3)
            np.testing.assert_array_equal(wide[:3], narrow)


class TestTheVesselDrivesThePhysics(unittest.TestCase):
    def test_the_distance_changes_over_an_episode(self):
        env = HybridEnv(mobile_params(), QUIET_EVE)
        env.reset(1)
        seen = [env.step(PM)[3]["d_km"] for _ in range(300)]
        self.assertGreater(max(seen) - min(seen), 5.0)

    def test_the_sifted_rate_follows_the_distance(self):
        # MUTATION: resolve the physics once and keep it (the "stale cache"
        # failure). The rate then never moves while the vessel does.
        env = HybridEnv(mobile_params(horizon=600), QUIET_EVE)
        env.reset(1)
        samples = []
        for _ in range(600):
            _, _, _, info = env.step(PM)
            samples.append((info["d_km"], env.sifted_per_slot[PM]))
        far = max(samples)[1]
        near = min(samples)[1]
        self.assertGreater(near, far)          # closer is always faster
        self.assertGreater(near / max(far, 1), 1.5)

    def test_the_physics_matches_an_independent_calculation_every_slot(self):
        # The strongest form: not "it changes" but "it is the RIGHT physics
        # for where the vessel is". A cache returning a neighbouring cell's
        # values changes with distance and still fails here.
        env = HybridEnv(mobile_params(horizon=400), QUIET_EVE)
        env.reset(2)
        step = env.params.d_grid_km
        for _ in range(400):
            env.step(PM)
            d_cell = round(env.distance_km / step) * step
            pm = channel.derive(dataclasses.replace(env._pm_base,
                                                    d_km=d_cell),
                                env.params.n_x)
            ent = channel.derive_ent(dataclasses.replace(env._ent_base,
                                                         d_km=d_cell),
                                     env.params.n_x)
            self.assertEqual(env.sifted_per_slot,
                             (int(pm.clicks_per_s * 0.25),
                              int(ent.clicks_per_s * 0.5)))
            self.assertEqual(env.error_floor, (pm.e_avg, ent.q2))

    def test_the_key_params_follow_the_distance_too(self):
        # eta inside KeyParams is the decoy analysis' transmittance. Leaving
        # it at the construction distance yields blocks distilled against the
        # wrong channel -- an error invisible in the sifted rate.
        env = HybridEnv(mobile_params(horizon=600), QUIET_EVE)
        env.reset(1)
        etas = set()
        for _ in range(600):
            env.step(PM)
            etas.add(env._key_params.eta)
        self.assertGreater(len(etas), 5)

    def test_kappa_depends_on_the_distance(self):
        # MUTATION: key the kappa memo on (mode, f) only. The bound then
        # reports the first distance the episode happened to visit for the
        # whole run.
        env = HybridEnv(mobile_params(), QUIET_EVE)
        env._set_distance(150.0)
        near = env.kappa(PM, 0.0)
        env._set_distance(190.0)
        far = env.kappa(PM, 0.0)
        self.assertGreater(near, far)
        env._set_distance(150.0)
        self.assertEqual(env.kappa(PM, 0.0), near)

    def test_step_relaxed_moves_the_vessel_as_well(self):
        # The Oracle bound runs through step_relaxed. A vessel that only
        # moves in step() would leave the bound sitting at one distance while
        # the policies it bounds sail away from it.
        env = HybridEnv(mobile_params(horizon=300), QUIET_EVE)
        env.reset(1)
        seen = [env.step_relaxed(lambda f, rates, obs: PM)[3]["d_km"]
                for _ in range(300)]
        self.assertGreater(max(seen) - min(seen), 5.0)

    def test_reset_rewinds_the_track_but_keeps_the_physics_cache(self):
        env = HybridEnv(mobile_params(horizon=300), QUIET_EVE)
        env.reset(1)
        for _ in range(300):
            env.step(PM)
        resolved = env.physics_cache_size
        self.assertGreater(resolved, 3)
        env.reset(1)
        # The vessel is back at the start...
        self.assertEqual(env.distance_km, env.track.distance)
        # ...but the grid points already paid for are still paid for. Clearing
        # the cache here would re-derive them every episode of a 200-episode
        # training run.
        self.assertEqual(env.physics_cache_size, resolved)

    def test_two_envs_on_the_same_seed_sail_the_same_track(self):
        def trace(seed):
            env = HybridEnv(mobile_params(horizon=200), QUIET_EVE)
            env.reset(seed)
            return [env.step(PM)[3]["d_km"] for _ in range(200)]
        self.assertEqual(trace(31), trace(31))
        self.assertNotEqual(trace(31), trace(32))


class TestTheDistanceGrid(unittest.TestCase):
    def _count_derives(self, params, slots):
        """Run `slots` slots, counting calls into channel.derive."""
        calls = []
        real = channel.derive

        def counting(env, n):
            calls.append(env.d_km)
            return real(env, n)

        channel.derive = counting
        try:
            env = HybridEnv(params, QUIET_EVE)
            env.reset(1)
            distances = [env.step(PM)[3]["d_km"] for _ in range(slots)]
        finally:
            channel.derive = real
        return calls, distances, env

    def test_the_grid_is_used_instead_of_recomputing_every_slot(self):
        # MUTATION: drop the memo (or the snapping) so every slot resolves its
        # own physics. 800 slots then cost 800 derives instead of ~35.
        slots = 800
        calls, distances, env = self._count_derives(
            mobile_params(horizon=slots), slots)
        cells = {round(d / env.params.d_grid_km) for d in distances}
        self.assertLess(len(calls), slots / 10)
        # Exactly one derive per distinct cell, plus the construction call.
        self.assertLessEqual(len(calls), len(cells) + 1)
        self.assertEqual(len(set(calls)), len(calls))   # never resolved twice
        self.assertEqual(env.physics_cache_size, len(calls))

    def test_turning_the_grid_off_resolves_every_distinct_distance(self):
        # The other half of the previous test: without it, an implementation
        # that ignores d_grid_km entirely and always caches on the raw
        # distance would still look "efficient" whenever the vessel loiters.
        slots = 400
        calls, distances, _ = self._count_derives(
            mobile_params(horizon=slots, d_grid_km=0.0), slots)
        self.assertEqual(len(set(calls)), len(calls))   # still memoised
        # ...but the memo now has one entry per distinct DISTANCE, which on a
        # moving vessel is nearly one per slot.
        self.assertEqual(set(distances) - set(calls), set())
        self.assertGreater(len(calls), 100)

    def test_the_cache_survives_across_episodes(self):
        env = HybridEnv(mobile_params(horizon=200), QUIET_EVE)
        env.reset(1)
        for _ in range(200):
            env.step(PM)
        first = env.physics_cache_size
        real = channel.derive
        calls = []

        def counting(env_, n):
            calls.append(env_.d_km)
            return real(env_, n)

        channel.derive = counting
        try:
            env.reset(1)
            for _ in range(200):
                env.step(PM)
        finally:
            channel.derive = real
        self.assertEqual(env.physics_cache_size, first)
        self.assertEqual(calls, [])

    def test_the_quantisation_error_it_costs_is_bounded(self):
        # THE DECLARED APPROXIMATION. At the default 0.5 km grid the vessel
        # is given the physics of a point at most 0.25 km away, i.e. 0.05 dB
        # of fibre: 10^(0.005) - 1 = 1.16% of transmittance. Measured over
        # 1..200 km the worst case is 1.18% on the sifted rate and 0.87% on
        # the error floor.
        #
        # The sifted rate is compared only where it exceeds 1000 bits/slot:
        # below that the int() truncation in _physics_at dominates (2 vs 1
        # is a 50% "error" that has nothing to do with the grid), and that
        # truncation predates this work.
        exact = HybridEnv(mobile_params(d_grid_km=0.0), QUIET_EVE)
        grid = HybridEnv(mobile_params(d_grid_km=0.5), QUIET_EVE)
        worst_rate = worst_floor = 0.0
        for d in np.arange(1.0, 200.0, 0.113):
            exact._set_distance(float(d))
            grid._set_distance(float(d))
            for mode in (PM, ENT):
                a, b = exact.sifted_per_slot[mode], grid.sifted_per_slot[mode]
                if a >= 1000:
                    worst_rate = max(worst_rate, abs(b - a) / a)
                a, b = exact.error_floor[mode], grid.error_floor[mode]
                worst_floor = max(worst_floor, abs(b - a) / a)
        self.assertLess(worst_rate, 0.015)
        self.assertLess(worst_floor, 0.010)
        # And the bound is tight enough to be meaningful: a grid four times
        # coarser really is four times worse, so these numbers are measuring
        # the grid and not some unrelated ceiling.
        coarse = HybridEnv(mobile_params(d_grid_km=2.0), QUIET_EVE)
        worst_coarse = 0.0
        for d in np.arange(1.0, 200.0, 0.113):
            exact._set_distance(float(d))
            coarse._set_distance(float(d))
            for mode in (PM, ENT):
                a, b = exact.error_floor[mode], coarse.error_floor[mode]
                worst_coarse = max(worst_coarse, abs(b - a) / a)
        self.assertGreater(worst_coarse, 2.5 * worst_floor)

    def test_the_grid_displaces_pms_death_distance_by_at_most_half_a_cell(
            self):
        # The one place the bounded relative error above does NOT apply: near
        # a death distance the key length falls off a cliff, so the honest
        # statement is about WHERE the cliff is, not how tall it is.
        step = 0.5
        env = HybridEnv(mobile_params(d_grid_km=step), QUIET_EVE)

        def alive(distance, quantise):
            d = round(distance / step) * step if quantise else distance
            return channel.derive(dataclasses.replace(env._pm_base, d_km=d),
                                  env.params.n_x).l_predict > 0

        def death(quantise):
            lo, hi = 100.0, 250.0
            for _ in range(40):
                mid = (lo + hi) / 2.0
                lo, hi = (mid, hi) if alive(mid, quantise) else (lo, mid)
            return lo

        self.assertLessEqual(abs(death(True) - death(False)), step / 2 + 1e-6)


class TestTheDistanceIsObservable(unittest.TestCase):
    """A vessel knows its own position; Eve's jamming it does not."""

    def test_the_observation_reports_the_distance_every_slot(self):
        # MUTATION: report a constant, or the starting distance, or d_km. All
        # of them keep the observation the right shape and the right scale.
        env = HybridEnv(mobile_params(horizon=400), QUIET_EVE)
        env.reset(1)
        reported = []
        for _ in range(400):
            obs, _, _, info = env.step(PM)
            self.assertAlmostEqual(obs[I_DIST] * DIST_REF_KM, info["d_km"])
            self.assertAlmostEqual(obs[I_DIST] * DIST_REF_KM,
                                   env.distance_km)
            reported.append(obs[I_DIST])
        # ...and it really varied, so a constant could not have passed.
        self.assertGreater(max(reported) - min(reported),
                           5.0 / DIST_REF_KM)

    def test_the_feature_is_normalised_by_a_declared_constant(self):
        env = HybridEnv(HybridParams(horizon=5, d_km=125.0))
        self.assertAlmostEqual(env.reset()[I_DIST], 125.0 / DIST_REF_KM)
        self.assertEqual(DIST_REF_KM, 250.0)

    def test_the_jamming_fraction_is_still_invisible(self):
        # The whole point of the asymmetry: adding an observable exogenous
        # signal must not have made the hidden one leak.
        # No level is 0.0: a jamming fraction that happens to equal an
        # empty queue's feature would make assertNotIn fail for a reason that
        # has nothing to do with leaking.
        env = HybridEnv(mobile_params(horizon=120),
                        EveParams(f_levels=(0.017, 0.041, 0.083), seed=2))
        env.reset(1)
        levels = set()
        for _ in range(120):
            obs, _, _, info = env.step(PM)
            self.assertNotIn(info["f"], list(obs))
            levels.add(info["f"])
        # The run really did jam, so there was something to leak.
        self.assertGreater(len(levels), 1)

    def test_obs_dim_and_the_index_agree(self):
        self.assertEqual(hybridenv.OBS_DIM, 7)
        self.assertEqual(I_DIST, hybridenv.OBS_DIM - 1)
        self.assertEqual(len(HybridEnv(HybridParams(horizon=2)).reset()),
                         hybridenv.OBS_DIM)


if __name__ == "__main__":
    unittest.main()
