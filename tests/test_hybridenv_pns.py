"""The PNS attack inside the slot-by-slot environment.

The attack's whole character is that it is SELECTIVE and INVISIBLE: it
certifies PM's key away, leaves ENT untouched, and moves neither the QBER nor
(much) the count rate. The only observable it disturbs is the ratio of the two
decoy intensities' gains, so the environment has to carry that ratio in the
observation or no realisable policy can ever react to the attack.

These tests are written against the ways that can silently go wrong: an
attack that leaks into ENT, a QBER that moves after all, an observation
feature with invented noise instead of the finite-sample noise a real
estimate has, and -- above all -- an environment whose UNATTACKED behaviour
drifted while the attack was being added.
"""
import unittest

import hybridenv
from evemarkov import EveParams
from hybridenv import ENT, PM, HybridEnv, HybridParams
from policies import (AlwaysENT, AlwaysPM, GreedyThreshold, run_episode)


def params(**kw):
    kw.setdefault("horizon", 400)
    kw.setdefault("lam", 2.5)
    kw.setdefault("b_max", 1_000_000)
    kw.setdefault("ent_preset", "ent_advanced")
    return HybridParams(**kw)


# A posture that jams NOTHING and only splits photons, so every observation
# except the decoy ratio is exactly the unattacked one.
STEALTH = EveParams(f_levels=(0.0, 0.0, 0.0), s_levels=(0.0, 0.0, 0.6),
                    seed=3)
# The same chain doing nothing at all.
IDLE = EveParams(f_levels=(0.0, 0.0, 0.0), s_levels=(0.0, 0.0, 0.0), seed=3)


class TestTheUnattackedEnvironmentDidNotMove(unittest.TestCase):
    """Measured BEFORE the attack was added. Every eta the study has
    published rests on these three trajectories replaying exactly."""

    GOLDEN = {"always-pm": (0.6463414634, 5213542, 297),
              "always-ent": (0.6981707317, 5635468, 246),
              "greedy-0.045": (0.7022357724, 5665379, 243)}

    def test_the_default_adversary_replays_the_recorded_episodes(self):
        for agent in (AlwaysPM(), AlwaysENT(), GreedyThreshold(0.045)):
            env = HybridEnv(params(), EveParams(seed=3))
            result = run_episode(env, agent, seed=5)
            eta, bits, expired = self.GOLDEN[agent.name]
            self.assertAlmostEqual(result["eta"], eta, places=10)
            self.assertEqual(result["secret_bits"], bits)
            self.assertEqual(result["expired"], expired)


class TestTheAttackIsSelective(unittest.TestCase):
    def test_it_certifies_pms_key_away(self):
        env = HybridEnv(params(), STEALTH)
        self.assertGreater(env.kappa(PM, 0.0, 0.0), 0.0)
        self.assertEqual(env.kappa(PM, 0.0, 0.6), 0.0)

    def test_it_does_not_touch_ent_at_all(self):
        env = HybridEnv(params(), STEALTH)
        self.assertEqual(env.kappa(ENT, 0.0, 0.6), env.kappa(ENT, 0.0, 0.0))

    def test_a_heavier_attack_never_earns_pm_more(self):
        env = HybridEnv(params(), STEALTH)
        rates = [env.kappa(PM, 0.0, s) for s in (0.0, 0.1, 0.2, 0.3, 0.4)]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_kappa_defaults_to_the_honest_channel(self):
        # Two positional arguments is how every existing caller spells it.
        env = HybridEnv(params(), IDLE)
        self.assertEqual(env.kappa(PM, 0.03), env.kappa(PM, 0.03, 0.0))


class TestTheAttackIsInvisibleWhereItMatters(unittest.TestCase):
    def run_slots(self, eve, action, n=300):
        env = HybridEnv(params(horizon=n), eve)
        obs = env.reset(11)
        out = []
        for _ in range(n):
            obs, _, done, info = env.step(action)
            out.append((obs, info))
            if done:
                break
        return env, out

    def test_the_qber_does_not_move(self):
        # Same seed, same mode, jamming off in both: the only difference is
        # the photon splitting, and the QBER must not notice it.
        _, idle = self.run_slots(IDLE, PM)
        _, hit = self.run_slots(STEALTH, PM)
        mean_idle = sum(i["qber"] for _, i in idle) / len(idle)
        mean_hit = sum(i["qber"] for _, i in hit) / len(hit)
        self.assertAlmostEqual(mean_idle, mean_hit, places=4)

    def test_the_key_stops_arriving_anyway(self):
        idle_env, idle = self.run_slots(IDLE, PM)
        hit_env, hit = self.run_slots(STEALTH, PM)
        self.assertLess(sum(i["deposited"] for _, i in hit),
                        0.8 * sum(i["deposited"] for _, i in idle))

    def test_the_count_rate_barely_falls(self):
        # An operator watching the sifted rate would see a few per cent.
        env = HybridEnv(params(), STEALTH)
        honest = env.sifted_per_slot[PM]
        attacked = env.sifted_under_attack(0.6)[PM]
        self.assertLess(attacked, honest)
        self.assertGreater(attacked, 0.9 * honest)

    def test_ent_harvests_exactly_as_much_under_attack(self):
        env = HybridEnv(params(), STEALTH)
        self.assertEqual(env.sifted_under_attack(0.6)[ENT],
                         env.sifted_per_slot[ENT])


class TestTheDecoyRatioIsInTheObservation(unittest.TestCase):
    def test_the_layout_grew_by_exactly_one_entry(self):
        self.assertEqual(hybridenv.OBS_DIM, 8)
        self.assertEqual(hybridenv.I_DECOY, 7)

    def test_it_reads_near_zero_on_an_unattacked_link(self):
        # The feature is an EXCESS over the honest ratio, so an idle channel
        # sits at 0 whatever mu1, mu2 and the distance happen to be -- which
        # is what lets one threshold work across the whole map.
        env = HybridEnv(params(), IDLE)
        obs = env.reset(11)
        for _ in range(60):
            obs, _, _, _ = env.step(PM)
        self.assertLess(abs(obs[hybridenv.I_DECOY]), 0.05)

    def test_it_rises_under_an_attack_the_qber_cannot_see(self):
        env = HybridEnv(params(), STEALTH)
        obs = env.reset(11)
        seen = []
        for _ in range(300):
            obs, _, _, info = env.step(PM)
            if info["pns"] > 0.0:
                seen.append(obs[hybridenv.I_DECOY])
        self.assertTrue(seen, "the chain never reached the attacking state")
        self.assertGreater(max(seen), 0.3)

    def test_the_slot_reports_the_attack_strength_the_policy_may_not_see(self):
        # info carries it for the analysis, exactly as it carries f.
        env = HybridEnv(params(), STEALTH)
        env.reset(11)
        _, _, _, info = env.step(PM)
        self.assertIn("pns", info)

    def test_it_goes_STALE_on_ent_rather_than_lying(self):
        # Running ENT means the decoy hardware is not transmitting, so no new
        # decoy statistics exist. The feature holds its last PM reading --
        # which is the real observability asymmetry a controller has to live
        # with, and the reason it must occasionally go back and look.
        env = HybridEnv(params(), STEALTH)
        env.reset(11)
        obs = None
        for _ in range(200):
            obs, _, _, info = env.step(PM)
            if info["pns"] > 0.0 and obs[hybridenv.I_DECOY] > 0.3:
                break
        held = obs[hybridenv.I_DECOY]
        for _ in range(50):
            obs, _, _, _ = env.step(ENT)
        self.assertEqual(obs[hybridenv.I_DECOY], held)

    def test_the_noise_is_the_finite_sample_kind_and_not_invented(self):
        # The estimate comes from a binomial split of the slot's own sifted
        # bits, so a bigger harvest is a quieter estimate -- in proportion,
        # not by a tuned constant. A short link harvests ~20x what a long one
        # does, so its feature must be materially steadier.
        def spread(d_km):
            env = HybridEnv(params(d_km=d_km), IDLE)
            obs = env.reset(11)
            vals = []
            for _ in range(150):
                obs, _, _, _ = env.step(PM)
                vals.append(obs[hybridenv.I_DECOY])
            mean = sum(vals) / len(vals)
            return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        self.assertLess(spread(20.0), spread(85.0))

    def test_the_window_starts_empty_rather_than_at_a_guess(self):
        env = HybridEnv(params(), STEALTH)
        obs = env.reset(11)
        self.assertEqual(obs[hybridenv.I_DECOY], 0.0)


class TestTheAttackReachesAWholeEpisode(unittest.TestCase):
    def test_always_pm_loses_badly_while_always_ent_does_not(self):
        # The scenario the whole exercise is for: a stealthy adversary makes
        # the mode choice matter, and no QBER-based rule can see why.
        pm = run_episode(HybridEnv(params(), STEALTH), AlwaysPM(), seed=5)
        ent = run_episode(HybridEnv(params(), STEALTH), AlwaysENT(), seed=5)
        self.assertLess(pm["eta"], ent["eta"])

    def test_a_qber_threshold_policy_behaves_exactly_like_always_pm(self):
        # It never fires, because there is nothing in the QBER to fire on.
        greedy = run_episode(HybridEnv(params(), STEALTH),
                             GreedyThreshold(0.045), seed=5)
        pm = run_episode(HybridEnv(params(), STEALTH), AlwaysPM(), seed=5)
        self.assertEqual(greedy["switches"], 0)
        self.assertEqual(greedy["eta"], pm["eta"])


if __name__ == "__main__":
    unittest.main()
