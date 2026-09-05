"""DQN in plain numpy -- no new dependency, 6 inputs and 2 actions.

The tests here are written to kill the *plausible weakened* implementation,
not to exercise the happy path. A DQN is unusually good at looking healthy
while being broken: a flipped gradient sign, a target network that is never
synced, a Bellman target bootstrapped off the online net, or a replay buffer
handing back slots that were never written all produce a loop that runs to
completion and returns a policy. Every one of those is pinned below, and each
test names the mutation it exists to catch.
"""
import unittest

import numpy as np

import dqn
from dqn import (SEED_STRIDE, DQNConfig, DQNPolicy, ObsNormalizer, QNetwork,
                 ReplayBuffer, bellman_target, episode_seed, epsilon_at,
                 train, training_seed_range)
from hybridenv import ENT, OBS_DIM, PM, HybridEnv, HybridParams
from policies import Policy, run_episode


class _BanditEnv:
    """Two-armed bandit dressed as an environment.

    The observation is constant, so Q(a) must converge to the mean reward of
    arm a and nothing about the state representation can rescue a broken
    update. `good` is which arm pays, so the same test runs both ways round
    and an agent that always answers "1" fails one of them.
    """

    def __init__(self, good, horizon=25):
        self.good = int(good)
        self.horizon = int(horizon)
        self.seeds = []
        self._t = 0
        self._paid = 0.0

    def reset(self, seed=None):
        self.seeds.append(seed)
        self._t = 0
        self._paid = 0.0
        return np.full(OBS_DIM, 0.5)

    def step(self, action):
        reward = 1.0 if int(action) == self.good else 0.0
        self._paid += reward
        self._t += 1
        return np.full(OBS_DIM, 0.5), reward, self._t >= self.horizon, {}

    @property
    def eta(self):
        return self._paid / self.horizon


class _ChainEnv:
    """A three-state chain whose payoff is one step away from the decision.

    `_BanditEnv` above is deliberately degenerate -- constant observation,
    `gamma = 0` -- which makes it a clean probe of the *update* but blind to
    most of `train`'s wiring: with one observation and no discounting, `obs`
    and `next_obs` are interchangeable and gamma multiplies nothing. Three
    corruptions of the loop survive it (measured, see the report): storing the
    transition backwards, hardwiring gamma to 0, and regressing on `next_obs`
    instead of `obs`.

    So this fixture is built to make each of those matter:

        A --good--> PAY --any--> PAY   reward 1 per slot, forever
        A --bad --> DEAD --any--> DEAD  reward 0, forever

    * The observation genuinely varies over time: a one-hot of the state plus
      a normalised slot clock, so `obs` and `next_obs` are never equal.
    * The reward at A is 0 for BOTH actions. The right action is identifiable
      only by what it leads to, so the signal has to travel through the
      bootstrap -- at gamma = 0 both actions at A are worth exactly 0 and the
      chain becomes unlearnable. `test_the_fixture_really_needs_a_bootstrap`
      pins that, so the fixture cannot quietly lose its teeth.
    * A is reachable only by reset(), never as a `next_obs`. That is what
      makes the backwards-storage corruptions bite deterministically rather
      than by luck: they leave Q(A, .) with no training signal at all.

    `good` runs the chain both ways round, so an agent that always answers the
    same action fails one of the two.
    """

    A, PAY, DEAD = 0, 1, 2

    def __init__(self, good, horizon=12):
        self.good = int(good)
        self.horizon = int(horizon)
        self.seeds = []
        self._state = self.A
        self._t = 0
        self._paid = 0.0

    def _obs(self):
        v = np.zeros(OBS_DIM)
        v[self._state] = 1.0
        v[3] = self._t / self.horizon
        return v

    def reset(self, seed=None):
        self.seeds.append(seed)
        self._state = self.A
        self._t = 0
        self._paid = 0.0
        return self._obs()

    def step(self, action):
        if self._state == self.A:
            self._state = self.PAY if int(action) == self.good else self.DEAD
            reward = 0.0
        else:
            reward = 1.0 if self._state == self.PAY else 0.0
        self._paid += reward
        self._t += 1
        return self._obs(), reward, self._t >= self.horizon, {}

    @property
    def eta(self):
        return self._paid / self.horizon


def _chain_cfg(good_seed=0, **kw):
    """Budget that learns the chain in ~50 ms. Verified over 6 config seeds."""
    base = dict(episodes=40, hidden=16, lr=0.05, gamma=0.9, batch_size=32,
                capacity=1000, warmup=64, target_sync=50, eps_start=1.0,
                eps_end=0.1, eps_decay_episodes=20, seed=good_seed)
    base.update(kw)
    return DQNConfig(**base)


def _tiny_cfg(**kw):
    """A config small enough that a whole training run is milliseconds."""
    base = dict(episodes=4, hidden=8, batch_size=8, capacity=200, warmup=8,
                target_sync=10, eps_decay_episodes=2, seed=0)
    base.update(kw)
    return DQNConfig(**base)


class _RecordingBuffer(dqn.ReplayBuffer):
    """A replay buffer that keeps every (action, reward) it was handed.

    The shaped reward is written into the buffer and nowhere else, so this is
    the only place a test can see whether the penalty was applied, applied to
    the right slots, or applied at all.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    def add(self, obs, action, reward, next_obs, done):
        self.seen.append((int(action), float(reward)))
        return super().add(obs, action, reward, next_obs, done)


def _record_training(cfg, env):
    """Train, and return every (action, reward) that reached the buffer."""
    real = dqn.ReplayBuffer
    box = {}

    def factory(*a, **kw):
        buf = _RecordingBuffer(*a, **kw)
        box["buf"] = buf
        return buf

    dqn.ReplayBuffer = factory
    try:
        policy = dqn.train(env, cfg)
    finally:
        dqn.ReplayBuffer = real
    return policy, box["buf"].seen


class TestTheSwitchPenalty(unittest.TestCase):
    """The unshaped agent flips mode hundreds of times an episode. eq (5)
    already charges a switch, but only as a partial block whose cost surfaces
    many slots later; the penalty makes it immediate. It is a TRAINING device
    -- the environment's reward, and therefore every policy's eta, is
    untouched."""

    def test_zero_penalty_reproduces_the_unshaped_agent_exactly(self):
        # The default must be a no-op, or every result recorded before this
        # feature silently stops being reproducible.
        a, _ = _record_training(_tiny_cfg(switch_penalty=0.0), _BanditEnv(1))
        b, _ = _record_training(_tiny_cfg(), _BanditEnv(1))
        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            np.testing.assert_array_equal(getattr(a.net, name),
                                          getattr(b.net, name))

    def test_only_a_slot_that_changed_action_is_charged(self):
        # Kills a penalty subtracted from every step, which would merely
        # shift the reward baseline and change no preference at all.
        # ONE episode: prev_action is re-seeded to PM at every reset, so a
        # test that tracked `prev` across episode boundaries would disagree
        # with the code at exactly the first slot of episodes 2..N.
        cfg = _tiny_cfg(switch_penalty=0.25, eps_start=1.0, eps_end=1.0,
                        episodes=1)
        _, seen = _record_training(cfg, _BanditEnv(1, horizon=30))
        prev = 0                      # PM: what reset() leaves behind
        for action, reward in seen:
            base = 1.0 if action == 1 else 0.0
            want = base - (0.25 if action != prev else 0.0)
            self.assertAlmostEqual(reward, want)
            prev = action

    def test_the_first_slot_of_an_episode_is_charged_against_pm(self):
        # reset() leaves the environment in PM, so choosing ENT on slot 1
        # really is a switch and eq (5) really does bill it. Treating the
        # first slot as free would teach the agent a discount that the
        # environment does not give.
        cfg = _tiny_cfg(switch_penalty=0.5, eps_start=1.0, eps_end=1.0,
                        episodes=1)
        _, seen = _record_training(cfg, _BanditEnv(1, horizon=6))
        first_action, first_reward = seen[0]
        base = 1.0 if first_action == 1 else 0.0
        if first_action != 0:
            self.assertAlmostEqual(first_reward, base - 0.5)
        else:
            self.assertAlmostEqual(first_reward, base)

    def test_a_bigger_penalty_lowers_the_reward_it_stores(self):
        small = _record_training(
            _tiny_cfg(switch_penalty=0.1, eps_start=1.0, eps_end=1.0),
            _BanditEnv(1))[1]
        big = _record_training(
            _tiny_cfg(switch_penalty=2.0, eps_start=1.0, eps_end=1.0),
            _BanditEnv(1))[1]
        self.assertEqual([a for a, _ in small], [a for a, _ in big])
        self.assertLess(sum(r for _, r in big), sum(r for _, r in small))

    def test_the_environment_reward_is_never_shaped(self):
        # The penalty must not reach eta. If it did, the DQN would be scored
        # on a different objective from the baselines it is compared against
        # and the whole comparison would be meaningless.
        env = _BanditEnv(1, horizon=10)
        cfg = _tiny_cfg(switch_penalty=5.0, eps_start=1.0, eps_end=1.0)
        _record_training(cfg, env)
        self.assertGreaterEqual(env.eta, 0.0)
        self.assertLessEqual(env.eta, 1.0)
        _, r, _, _ = _BanditEnv(1).step(1)
        self.assertEqual(r, 1.0)

    def test_a_negative_penalty_is_refused(self):
        with self.assertRaises(ValueError):
            dqn.validate_config(_tiny_cfg(switch_penalty=-0.1))


class TestQNetwork(unittest.TestCase):
    def test_forward_returns_one_value_per_action(self):
        net = QNetwork(OBS_DIM, 16, 2, seed=0)
        out = net.forward(np.zeros((1, OBS_DIM)))
        self.assertEqual(out.shape, (1, 2))

    def test_a_batch_is_handled_in_one_call(self):
        net = QNetwork(OBS_DIM, 16, 2, seed=0)
        self.assertEqual(net.forward(np.zeros((7, OBS_DIM))).shape, (7, 2))

    def test_a_bare_observation_is_promoted_to_a_batch_of_one(self):
        # DQNPolicy.act() hands in a 1-D vector; without atleast_2d the matmul
        # would silently produce a 1-D result and [0] would index an action.
        net = QNetwork(OBS_DIM, 16, 2, seed=0)
        self.assertEqual(net.forward(np.zeros(OBS_DIM)).shape, (1, 2))

    def test_the_same_seed_builds_the_same_weights(self):
        a = QNetwork(OBS_DIM, 8, 2, seed=3)
        b = QNetwork(OBS_DIM, 8, 2, seed=3)
        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name))

    def test_different_seeds_build_different_weights(self):
        # An implementation that ignored `seed` would make every "two runs
        # differ" test below vacuous.
        a = QNetwork(OBS_DIM, 8, 2, seed=3)
        b = QNetwork(OBS_DIM, 8, 2, seed=4)
        self.assertFalse(np.allclose(a.w1, b.w1))

    def test_training_reduces_the_loss_on_a_fixed_target(self):
        # The gradient must actually point downhill; a sign error here trains
        # silently and forever without learning anything.
        net = QNetwork(OBS_DIM, 16, 2, seed=0)
        x = np.ones((8, OBS_DIM))
        target = np.full((8, 2), 3.0)
        first = net.train_step(x, target, lr=0.05)
        for _ in range(200):
            last = net.train_step(x, target, lr=0.05)
        self.assertLess(last, first)
        self.assertLess(last, 0.05)

    def test_the_gradient_matches_finite_differences(self):
        # The sharpest test in the file. `assertLess(last, first)` above only
        # senses the SIGN; this pins the magnitude and the shape of every
        # backward term, so a missing ReLU mask, a transposed matmul, or a
        # gradient scaled by the wrong batch factor all die here.
        x = np.random.RandomState(7).randn(5, OBS_DIM)
        target = np.random.RandomState(8).randn(5, 2)
        names = ("w1", "b1", "w2", "b2", "w3", "b3")

        def fresh():
            # The biases must not stay at their zero init here. With b2 = 0 a
            # sample whose whole first hidden layer is dead lands EXACTLY on
            # the ReLU kink, where the analytic subgradient (0) and the
            # one-sided finite difference legitimately disagree -- that would
            # make this test fail on a correct implementation. Perturbing to a
            # generic point is deterministic and keeps the check meaningful.
            net = QNetwork(OBS_DIM, 4, 2, seed=5)
            rs = np.random.RandomState(9)
            net.b1 = rs.randn(4) * 0.5
            net.b2 = rs.randn(4) * 0.5
            net.b3 = rs.randn(2) * 0.5
            return net

        net = fresh()
        # Guard: no pre-activation may sit within the finite-difference step of
        # the kink, or the comparison below would be measuring the kink rather
        # than the gradient.
        z1 = x @ net.w1 + net.b1
        z2 = np.maximum(0.0, z1) @ net.w2 + net.b2
        self.assertGreater(min(np.abs(z1).min(), np.abs(z2).min()), 1e-3)
        before = {n: getattr(net, n).copy() for n in names}
        net.train_step(x, target, lr=1.0)     # p -= 1.0 * grad
        for name in names:
            analytic = before[name] - getattr(net, name)
            numeric = np.zeros_like(analytic)
            eps = 1e-6
            for idx in np.ndindex(analytic.shape):
                up, down = fresh(), fresh()
                getattr(up, name)[idx] += eps
                getattr(down, name)[idx] -= eps
                loss_up = np.mean((up.forward(x) - target) ** 2)
                loss_down = np.mean((down.forward(x) - target) ** 2)
                numeric[idx] = (loss_up - loss_down) / (2 * eps)
            np.testing.assert_allclose(analytic, numeric, rtol=2e-4,
                                       atol=1e-9, err_msg=name)

    def test_the_reported_loss_is_the_one_before_the_update(self):
        # A train_step that measured the loss AFTER stepping would report a
        # rosier curve than the weights justify, and would make the
        # "loss goes down" test above pass even with a tiny backwards step.
        x = np.random.RandomState(1).randn(6, OBS_DIM)
        target = np.random.RandomState(2).randn(6, 2)
        net = QNetwork(OBS_DIM, 8, 2, seed=6)
        expected = float(np.mean((net.forward(x) - target) ** 2))
        self.assertAlmostEqual(net.train_step(x, target, lr=0.01), expected,
                               places=12)

    def test_copy_from_makes_the_outputs_identical(self):
        a = QNetwork(OBS_DIM, 16, 2, seed=1)
        b = QNetwork(OBS_DIM, 16, 2, seed=2)
        x = np.random.RandomState(0).randn(4, OBS_DIM)
        b.copy_from(a)
        np.testing.assert_allclose(a.forward(x), b.forward(x))

    def test_copy_from_copies_rather_than_aliases(self):
        # If copy_from shared the arrays, the "target network" would BE the
        # online network: the Bellman target would chase itself and the whole
        # point of a target net would be gone, with every other test passing.
        a = QNetwork(OBS_DIM, 8, 2, seed=1)
        b = QNetwork(OBS_DIM, 8, 2, seed=2)
        b.copy_from(a)
        x = np.random.RandomState(0).randn(4, OBS_DIM)
        frozen = b.forward(x).copy()
        for _ in range(20):
            a.train_step(x, np.full((4, 2), 9.0), lr=0.1)
        np.testing.assert_allclose(b.forward(x), frozen)
        self.assertFalse(np.allclose(a.forward(x), frozen))


class TestReplayBuffer(unittest.TestCase):
    def test_it_reports_its_length(self):
        buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, seed=0)
        for i in range(4):
            buf.add(np.zeros(OBS_DIM), 0, 1.0, np.zeros(OBS_DIM), False)
        self.assertEqual(len(buf), 4)

    def test_the_length_saturates_at_the_capacity(self):
        buf = ReplayBuffer(capacity=3, obs_dim=OBS_DIM, seed=0)
        for i in range(9):
            buf.add(np.zeros(OBS_DIM), 0, 1.0, np.zeros(OBS_DIM), False)
        self.assertEqual(len(buf), 3)

    def test_it_overwrites_oldest_when_full(self):
        buf = ReplayBuffer(capacity=3, obs_dim=OBS_DIM, seed=0)
        for i in range(5):
            buf.add(np.full(OBS_DIM, i), 0, float(i), np.zeros(OBS_DIM), False)
        self.assertEqual(len(buf), 3)
        obs, _, _, _, _ = buf.sample(3)
        self.assertNotIn(0.0, obs[:, 0])          # the first two are gone

    def test_the_ring_keeps_exactly_the_most_recent_capacity_entries(self):
        # Stronger than "the oldest is gone": pins the whole surviving SET, so
        # a ring that overwrote the wrong slot (or stopped writing once full)
        # is caught.
        buf = ReplayBuffer(capacity=4, obs_dim=OBS_DIM, seed=0)
        for i in range(10):
            buf.add(np.full(OBS_DIM, i), 0, float(i), np.zeros(OBS_DIM), False)
        _, _, rew, _, _ = buf.sample(400)
        self.assertEqual(set(rew.tolist()), {6.0, 7.0, 8.0, 9.0})

    def test_sample_never_returns_a_slot_that_was_never_written(self):
        # `randint(0, capacity)` instead of `randint(0, len)` hands back rows
        # of zeros: a reward of 0 and an all-zero next state, which is a
        # perfectly plausible transition and so poisons learning silently.
        buf = ReplayBuffer(capacity=500, obs_dim=OBS_DIM, seed=0)
        for i in range(3):
            buf.add(np.full(OBS_DIM, i + 1), 1, float(i + 1),
                    np.zeros(OBS_DIM), False)
        _, _, rew, _, _ = buf.sample(300)
        self.assertEqual(set(rew.tolist()), {1.0, 2.0, 3.0})

    def test_sample_returns_matching_batch_shapes(self):
        buf = ReplayBuffer(capacity=10, obs_dim=OBS_DIM, seed=0)
        for i in range(10):
            buf.add(np.zeros(OBS_DIM), i % 2, 1.0, np.ones(OBS_DIM), False)
        obs, act, rew, nxt, done = buf.sample(4)
        self.assertEqual(obs.shape, (4, OBS_DIM))
        self.assertEqual(act.shape, (4,))
        self.assertEqual(rew.shape, (4,))
        self.assertEqual(nxt.shape, (4, OBS_DIM))
        self.assertEqual(done.shape, (4,))

    def test_the_five_columns_of_a_sampled_row_belong_together(self):
        # Sampling each column with its own index draw would give correctly
        # shaped batches of transitions that never happened -- the single most
        # destructive silent bug a replay buffer can have.
        buf = ReplayBuffer(capacity=64, obs_dim=OBS_DIM, seed=0)
        for i in range(64):
            buf.add(np.full(OBS_DIM, i), i % 2, float(i),
                    np.full(OBS_DIM, i + 1000), i % 3 == 0)
        obs, act, rew, nxt, done = buf.sample(64)
        for k in range(64):
            i = obs[k, 0]
            self.assertEqual(rew[k], i)
            self.assertEqual(act[k], i % 2)
            self.assertEqual(nxt[k, 0], i + 1000)
            self.assertEqual(done[k], 1.0 if i % 3 == 0 else 0.0)

    def test_sampling_decorrelates_rather_than_returning_a_run(self):
        # Returning the last n transitions (or index 0 n times) keeps the very
        # correlation experience replay exists to break, and would leave every
        # shape assertion above green.
        buf = ReplayBuffer(capacity=1000, obs_dim=OBS_DIM, seed=0)
        for i in range(1000):
            buf.add(np.full(OBS_DIM, i), 0, float(i), np.zeros(OBS_DIM), False)
        _, _, rew, _, _ = buf.sample(200)
        self.assertGreater(len(set(rew.tolist())), 100)   # not one repeat
        self.assertGreater(rew.max() - rew.min(), 500)    # not a tail slice

    def test_done_is_stored_as_a_number_the_bellman_mask_can_use(self):
        buf = ReplayBuffer(capacity=4, obs_dim=OBS_DIM, seed=0)
        buf.add(np.zeros(OBS_DIM), 0, 0.0, np.zeros(OBS_DIM), True)
        buf.add(np.zeros(OBS_DIM), 0, 0.0, np.zeros(OBS_DIM), False)
        _, _, _, _, done = buf.sample(50)
        self.assertEqual(set(done.tolist()), {0.0, 1.0})

    def test_the_same_seed_draws_the_same_batch(self):
        def once():
            buf = ReplayBuffer(capacity=50, obs_dim=OBS_DIM, seed=4)
            for i in range(50):
                buf.add(np.zeros(OBS_DIM), 0, float(i), np.zeros(OBS_DIM),
                        False)
            return buf.sample(10)[2].tolist()
        self.assertEqual(once(), once())

    def test_sampling_an_empty_buffer_says_so(self):
        buf = ReplayBuffer(capacity=8, obs_dim=OBS_DIM, seed=0)
        with self.assertRaises(ValueError):
            buf.sample(4)


class TestBellmanTarget(unittest.TestCase):
    """y = r + gamma (1 - done) max_a' Q_target(s', a'), written into the
    chosen action's slot only. Three separate mutations live in these lines."""

    def _pinned(self, values):
        """A network whose output is `values` for every input."""
        net = QNetwork(OBS_DIM, 4, 2, seed=0)
        net.w3[:] = 0.0
        net.b3[:] = np.asarray(values, dtype=float)
        return net

    def _batch(self):
        obs = np.zeros((2, OBS_DIM))
        act = np.array([0, 1])
        rew = np.array([1.0, 2.0])
        nxt = np.zeros((2, OBS_DIM))
        done = np.array([0.0, 1.0])
        return obs, act, rew, nxt, done

    def test_it_bootstraps_off_the_target_net_not_the_online_net(self):
        # Online says 20 is the best next value, target says 7. Using the
        # online net is the classic copy-paste slip and makes the agent chase
        # its own moving estimate.
        online = self._pinned([10.0, 20.0])
        target = self._pinned([1.0, 7.0])
        obs, act, rew, nxt, done = self._batch()
        y = bellman_target(online, target, obs, act, rew, nxt, done, gamma=0.5)
        self.assertAlmostEqual(y[0, 0], 1.0 + 0.5 * 7.0)   # 4.5, not 11.0

    def test_a_terminal_transition_bootstraps_nothing(self):
        # Dropping (1 - done) adds phantom value past the horizon; with a
        # fixed-length episode that is every 3600th sample.
        online = self._pinned([10.0, 20.0])
        target = self._pinned([1.0, 7.0])
        obs, act, rew, nxt, done = self._batch()
        y = bellman_target(online, target, obs, act, rew, nxt, done, gamma=0.5)
        self.assertAlmostEqual(y[1, 1], 2.0)               # not 2.0 + 3.5

    def test_the_action_that_was_not_taken_keeps_its_online_value(self):
        # Writing y into both slots trains the agent on a reward it never
        # collected for an action it never took.
        online = self._pinned([10.0, 20.0])
        target = self._pinned([1.0, 7.0])
        obs, act, rew, nxt, done = self._batch()
        y = bellman_target(online, target, obs, act, rew, nxt, done, gamma=0.5)
        self.assertAlmostEqual(y[0, 1], 20.0)
        self.assertAlmostEqual(y[1, 0], 10.0)

    def test_gamma_scales_the_bootstrap(self):
        online = self._pinned([0.0, 0.0])
        target = self._pinned([1.0, 7.0])
        obs, act, rew, nxt, done = self._batch()
        for gamma in (0.0, 0.5, 0.99):
            y = bellman_target(online, target, obs, act, rew, nxt, done, gamma)
            self.assertAlmostEqual(y[0, 0], 1.0 + gamma * 7.0)

    def test_it_leaves_the_target_network_alone(self):
        online = self._pinned([10.0, 20.0])
        target = self._pinned([1.0, 7.0])
        before = target.b3.copy()
        obs, act, rew, nxt, done = self._batch()
        bellman_target(online, target, obs, act, rew, nxt, done, gamma=0.9)
        np.testing.assert_allclose(target.b3, before)


class TestEpsilonSchedule(unittest.TestCase):
    def test_it_starts_at_eps_start(self):
        cfg = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_episodes=10)
        self.assertAlmostEqual(epsilon_at(0, cfg), 1.0)

    def test_it_actually_reaches_eps_end(self):
        # An off-by-one denominator (episode / (decay + 1)) leaves the agent
        # permanently more random than configured -- invisible except here.
        cfg = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_episodes=10)
        self.assertAlmostEqual(epsilon_at(10, cfg), 0.05)

    def test_it_stays_at_eps_end_afterwards(self):
        # Without the clamp epsilon keeps falling and goes negative, which
        # means no exploration at all for the rest of training.
        cfg = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_episodes=10)
        for episode in (11, 50, 10_000):
            self.assertAlmostEqual(epsilon_at(episode, cfg), 0.05)

    def test_it_never_increases(self):
        cfg = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_episodes=37)
        values = [epsilon_at(e, cfg) for e in range(80)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreaterEqual(min(values), cfg.eps_end)

    def test_a_zero_length_decay_is_legal_and_lands_on_eps_end(self):
        cfg = DQNConfig(eps_start=1.0, eps_end=0.05, eps_decay_episodes=0)
        self.assertAlmostEqual(epsilon_at(1, cfg), 0.05)


class TestSeedScheme(unittest.TestCase):
    """Training must draw a fresh trajectory per episode, and Task 7 must be
    able to name a seed set disjoint from the one training consumed."""

    def test_episode_seeds_are_distinct_within_a_run(self):
        cfg = DQNConfig(episodes=200, seed=0)
        seeds = [episode_seed(cfg, e) for e in range(cfg.episodes)]
        self.assertEqual(len(set(seeds)), cfg.episodes)

    def test_two_configs_never_share_an_episode_seed(self):
        # The stride is what makes the disjointness claim to Task 7 true.
        a = {episode_seed(DQNConfig(episodes=200, seed=0), e)
             for e in range(200)}
        b = {episode_seed(DQNConfig(episodes=200, seed=1), e)
             for e in range(200)}
        self.assertEqual(a & b, set())

    def test_the_advertised_range_is_exactly_what_training_consumes(self):
        cfg = DQNConfig(episodes=17, seed=2)
        advertised = list(training_seed_range(cfg))
        consumed = [episode_seed(cfg, e) for e in range(cfg.episodes)]
        self.assertEqual(advertised, consumed)

    def test_more_episodes_than_the_stride_is_refused(self):
        # Past the stride, run seed=0 and run seed=1 would overlap and the
        # disjointness promise would quietly become false.
        with self.assertRaises(ValueError):
            train(_BanditEnv(good=1), _tiny_cfg(episodes=SEED_STRIDE + 1))


class TestConfigValidation(unittest.TestCase):
    def test_a_warmup_the_buffer_can_never_reach_is_refused(self):
        # len(buf) saturates at capacity, so warmup > capacity means the
        # gradient step never fires: training "succeeds" and learns nothing.
        with self.assertRaises(ValueError):
            train(_BanditEnv(good=1), _tiny_cfg(warmup=500, capacity=100))

    def test_a_non_positive_target_sync_is_refused(self):
        with self.assertRaises(ValueError):
            train(_BanditEnv(good=1), _tiny_cfg(target_sync=0))


class TestDQNPolicy(unittest.TestCase):
    def test_it_picks_the_action_with_the_higher_q(self):
        # Force the output layer so Q = [1, 5] for any input; the greedy
        # choice must be action 1.
        net = QNetwork(OBS_DIM, 4, 2, seed=0)
        net.w3[:] = 0.0
        net.b3[:] = np.array([1.0, 5.0])
        self.assertEqual(
            DQNPolicy(net, ObsNormalizer()).act(np.zeros(OBS_DIM)), 1)

    def test_it_picks_the_other_action_when_that_one_is_higher(self):
        # `return 1` passes the test above. It does not pass this one.
        net = QNetwork(OBS_DIM, 4, 2, seed=0)
        net.w3[:] = 0.0
        net.b3[:] = np.array([5.0, 1.0])
        self.assertEqual(
            DQNPolicy(net, ObsNormalizer()).act(np.zeros(OBS_DIM)), 0)

    def test_it_returns_a_plain_int_the_environment_accepts(self):
        # np.int64 compares equal to 0/1 but is not what `info["mode"]` and
        # the CSV writers downstream expect.
        net = QNetwork(OBS_DIM, 4, 2, seed=0)
        self.assertIs(
            type(DQNPolicy(net, ObsNormalizer()).act(np.zeros(OBS_DIM))), int)

    def test_it_is_a_policy_the_episode_runner_accepts(self):
        self.assertIsInstance(
            DQNPolicy(QNetwork(OBS_DIM, 4, 2, seed=0), ObsNormalizer()),
            Policy)

    def test_acting_does_not_change_the_network(self):
        net = QNetwork(OBS_DIM, 8, 2, seed=0)
        before = net.w1.copy()
        for _ in range(5):
            DQNPolicy(net, ObsNormalizer()).act(
                np.random.RandomState(0).randn(OBS_DIM))
        np.testing.assert_allclose(net.w1, before)

    def test_it_carries_a_label_the_result_tables_can_key_on(self):
        self.assertTrue(
            DQNPolicy(QNetwork(OBS_DIM, 4, 2, seed=0), ObsNormalizer()).name)


class TestTrainingLoop(unittest.TestCase):
    def test_training_runs_and_returns_a_usable_policy(self):
        # Tiny budget: this asserts the loop is wired, not that it converges.
        env = HybridEnv(HybridParams(horizon=50, seed=0))
        cfg = DQNConfig(episodes=3, hidden=8, batch_size=8, warmup=16, seed=0)
        policy = train(env, cfg)
        action = policy.act(env.reset())
        self.assertIn(action, (0, 1))

    def test_the_returned_policy_drives_a_real_episode(self):
        env = HybridEnv(HybridParams(horizon=30, seed=0))
        policy = train(env, _tiny_cfg(episodes=2))
        out = run_episode(env, policy, seed=999)
        self.assertIn(out["policy"], (policy.name,))
        self.assertGreaterEqual(out["eta"], 0.0)
        self.assertLessEqual(out["eta"], 1.0)

    def test_every_episode_gets_its_own_seed(self):
        # `env.reset()` with no argument replays ONE trajectory for the whole
        # run (hybridenv.reset docstring), so the agent would train on a
        # single episode and then be evaluated on a set containing it.
        env = _BanditEnv(good=1)
        cfg = _tiny_cfg(episodes=5)
        train(env, cfg)
        self.assertEqual(len(env.seeds), 5)
        self.assertNotIn(None, env.seeds)
        self.assertEqual(len(set(env.seeds)), 5)
        self.assertEqual(env.seeds, [episode_seed(cfg, e) for e in range(5)])

    def test_two_config_seeds_give_different_training_trajectories(self):
        # Both the environment seeds consumed and the resulting weights must
        # move with cfg.seed; a scheme that ignored it would make repeated
        # runs of the study identical and hide the variance.
        env_a, env_b = _BanditEnv(good=1), _BanditEnv(good=1)
        pol_a = train(env_a, _tiny_cfg(episodes=4, seed=0))
        pol_b = train(env_b, _tiny_cfg(episodes=4, seed=1))
        self.assertEqual(set(env_a.seeds) & set(env_b.seeds), set())
        probe = np.random.RandomState(0).randn(3, OBS_DIM)
        self.assertFalse(np.allclose(pol_a.net.forward(probe),
                                     pol_b.net.forward(probe)))

    def test_the_same_config_seed_reproduces_the_run(self):
        probe = np.random.RandomState(0).randn(3, OBS_DIM)

        def once():
            return train(_BanditEnv(good=1),
                         _tiny_cfg(episodes=4, seed=5)).net.forward(probe)
        np.testing.assert_allclose(once(), once())

    def test_the_target_net_is_synced_on_schedule_and_not_every_step(self):
        # Never syncing and syncing every step are both silent: the loop runs,
        # the loss falls, and only the learning dynamics change. Count the
        # copies instead of hoping the outcome differs.
        calls = []
        real = QNetwork.copy_from

        def spy(self, other):
            calls.append(1)
            return real(self, other)

        env = _BanditEnv(good=1, horizon=20)
        cfg = _tiny_cfg(episodes=2, warmup=5, target_sync=10, capacity=100)
        original = QNetwork.copy_from
        QNetwork.copy_from = spy
        try:
            train(env, cfg)
        finally:
            QNetwork.copy_from = original
        # 2 x 20 = 40 env steps; the gradient step fires once the buffer holds
        # warmup=5 entries, i.e. 40 - 5 + 1 = 36 times. Syncs at 10, 20, 30,
        # plus the one copy that initialises the target net.
        self.assertEqual(sum(calls), 1 + 36 // 10)

    def test_the_target_net_is_initialised_from_the_online_net(self):
        # Starting the target from an independent init makes the first
        # target_sync steps regress toward noise.
        seen = {}
        real = QNetwork.copy_from

        def spy(self, other):
            seen.setdefault("first", True)
            return real(self, other)

        original = QNetwork.copy_from
        QNetwork.copy_from = spy
        try:
            train(_BanditEnv(good=1, horizon=4),
                  _tiny_cfg(episodes=1, warmup=1000, capacity=2000))
        finally:
            QNetwork.copy_from = original
        # warmup is unreachable within 4 steps, so no gradient step and no
        # scheduled sync can have happened: any copy_from at all is the init.
        self.assertTrue(seen.get("first"))

    def test_no_gradient_step_happens_before_warmup(self):
        # Learning from a 3-sample buffer is not a crash, it is a quietly
        # worse agent, so the guard has to be pinned directly.
        steps = []
        real = QNetwork.train_step

        def spy(self, x, target, lr):
            steps.append(1)
            return real(self, x, target, lr)

        original = QNetwork.train_step
        QNetwork.train_step = spy
        try:
            train(_BanditEnv(good=1, horizon=10),
                  _tiny_cfg(episodes=1, warmup=1000, capacity=2000))
        finally:
            QNetwork.train_step = original
        self.assertEqual(steps, [])

    def test_the_progress_hook_is_called_once_per_episode(self):
        seen = []
        train(_BanditEnv(good=1, horizon=5), _tiny_cfg(episodes=3),
              progress=lambda ep, eta, eps: seen.append((ep, eps)))
        self.assertEqual([ep for ep, _ in seen], [0, 1, 2])
        self.assertEqual([eps for _, eps in seen],
                         sorted([eps for _, eps in seen], reverse=True))

    def test_it_learns_which_arm_pays(self):
        # End to end: a flipped gradient, a target written into the wrong
        # action slot, a replay buffer whose action column is misaligned, or a
        # loop that never stores the action it actually took all leave this
        # failing for at least one value of `good` while every unit test above
        # could still pass.
        for good in (0, 1):
            env = _BanditEnv(good=good, horizon=25)
            cfg = DQNConfig(episodes=20, hidden=8, lr=0.05, gamma=0.0,
                            batch_size=16, capacity=1000, warmup=32,
                            target_sync=25, eps_start=1.0, eps_end=0.1,
                            eps_decay_episodes=10, seed=0)
            policy = train(env, cfg)
            self.assertEqual(policy.act(np.full(OBS_DIM, 0.5)), good,
                             "failed to learn arm %d" % good)

    def test_the_action_space_is_the_environments_two_modes(self):
        # PM/ENT are 0/1; a net built with a different output width would make
        # argmax return an action hybridenv silently coerces to PM.
        policy = train(_BanditEnv(good=1, horizon=5), _tiny_cfg(episodes=1))
        self.assertEqual(policy.net.forward(np.zeros(OBS_DIM)).shape, (1, 2))
        self.assertIn(policy.act(np.zeros(OBS_DIM)), (PM, ENT))

    def test_a_missing_config_falls_back_to_the_defaults(self):
        self.assertIsInstance(dqn.DQNConfig(), DQNConfig)
        env = _BanditEnv(good=1, horizon=2)
        cfg = _tiny_cfg(episodes=1)
        self.assertIsInstance(train(env, cfg), DQNPolicy)


class TestTrainingLoopWiring(unittest.TestCase):
    """The loop itself, on a fixture that can tell its parts apart.

    Everything in TestTrainingLoop above runs against `_BanditEnv`, where a
    constant observation and gamma = 0 mean most of `train`'s plumbing cannot
    matter. These run against `_ChainEnv`, where it does.
    """

    def _learn(self, good, **kw):
        env = _ChainEnv(good=good)
        policy = train(env, _chain_cfg(**kw))
        # Through q_values, not net.forward: the net is trained on
        # standardised inputs, so reading it with a raw observation reports
        # an argmax the policy would never take.
        q = policy.q_values(_ChainEnv(good).reset())
        return int(np.argmax(q)), float(q[good] - q[1 - good])

    def test_it_learns_an_action_whose_payoff_is_one_step_away(self):
        # The reward at A is 0 whichever action is taken, so this can only be
        # solved by carrying value back from PAY through the bootstrap. It
        # fails if the transition is stored backwards, if the regression is
        # run on next_obs, or if gamma is not applied -- none of which the
        # bandit fixture can see.
        for good in (0, 1):
            picked, gap = self._learn(good)
            self.assertEqual(picked, good, "chain good=%d" % good)
            # A margin, not a coin flip: the observed gap is 3.6-5.0 over six
            # config seeds, while a loop that learns nothing about A sits at
            # |gap| < 0.02. Anything in between is not "learned".
            self.assertGreater(gap, 1.0, "chain good=%d gap=%.3f"
                               % (good, gap))

    def test_the_fixture_really_needs_a_bootstrap(self):
        # Guards the guard. If a future edit made the chain solvable from the
        # immediate reward alone, the test above would keep passing while
        # silently losing its power over gamma. At gamma = 0 both actions at A
        # are worth exactly 0, so no margin can exist.
        for good in (0, 1):
            _, gap = self._learn(good, gamma=0.0)
            self.assertLess(abs(gap), 0.5, "gamma=0 gap=%.3f" % gap)

    def test_the_observation_really_changes_from_slot_to_slot(self):
        # If obs == next_obs the swap corruptions are no-ops by construction,
        # so the fixture's premise is asserted rather than assumed.
        env = _ChainEnv(good=1)
        obs = env.reset(0)
        nxt, _, _, _ = env.step(1)
        self.assertFalse(np.array_equal(obs, nxt))
        later, _, _, _ = env.step(1)
        self.assertFalse(np.array_equal(nxt, later))   # the clock keeps moving

    def test_the_decision_state_is_never_a_successor(self):
        # The property that makes the backwards-storage corruptions bite
        # deterministically: nothing transitions INTO A, so storing (s', a, r,
        # s) leaves Q(A, .) with no training signal at all.
        env = _ChainEnv(good=1)
        start = env.reset(0)
        done = False
        while not done:
            nxt, _, done, _ = env.step(1)
            self.assertFalse(np.array_equal(nxt, start))


def _training_trace(cfg, env):
    """Snapshot (online.w1, target.w1) at the start of every gradient step.

    `train` looks `bellman_target` up in the module globals at call time, so
    patching it here yields a per-gradient-step window onto both networks
    without `train` exposing anything for testing. The snapshot is taken
    BEFORE this step's train_step and AFTER the previous step's sync, so
    trace[i][0] is exactly the online net as it stood when a sync at the end
    of step i-1 would have copied it.
    """
    trace = []
    real = dqn.bellman_target

    def spy(online, target, *args, **kwargs):
        trace.append((online.w1.copy(), target.w1.copy()))
        return real(online, target, *args, **kwargs)

    dqn.bellman_target = spy
    try:
        policy = train(env, cfg)
    finally:
        dqn.bellman_target = real
    return trace, policy


class TestTargetNetworkWeights(unittest.TestCase):
    """The target network, asserted on WEIGHTS rather than call counts.

    Counting copy_from calls pins the schedule but says nothing about what was
    copied where. Two corruptions slip through a count: copying in the wrong
    direction (which clobbers the ONLINE net every sync and destroys learning
    outright), and a target built from an independent initialisation.
    """

    # 2 episodes x 20 slots = 40 env steps; learning starts once the buffer
    # holds warmup=5, so there are 40 - 5 + 1 = 36 gradient steps and syncs
    # land at the end of steps 10, 20 and 30 -- i.e. trace indices 10, 20, 30
    # are the snapshots taken immediately after a sync.
    CFG = dict(episodes=2, hidden=8, batch_size=8, capacity=200, warmup=5,
               target_sync=10, lr=0.05, gamma=0.9, eps_decay_episodes=1,
               seed=0)
    SYNC_AT = (10, 20, 30)

    def setUp(self):
        cfg = DQNConfig(**self.CFG)
        self.cfg = cfg
        self.trace, self.policy = _training_trace(cfg,
                                                  _ChainEnv(good=1,
                                                            horizon=20))
        self.assertEqual(len(self.trace), 36)     # the arithmetic above holds

    def test_the_target_starts_out_equal_to_the_online_net(self):
        # A target built from its own seed regresses the first target_sync
        # steps toward noise. Only a weight comparison sees it: the call count
        # is unchanged and training still "works".
        online, target = self.trace[0]
        np.testing.assert_allclose(target, online)

    def test_the_target_is_frozen_between_syncs(self):
        # Syncing every step would make the target move with the online net
        # here, which is the whole difference between DQN and the divergent
        # version of it.
        first_target = self.trace[1][1]
        for i in range(1, self.SYNC_AT[0]):
            np.testing.assert_allclose(self.trace[i][1], first_target,
                                       err_msg="target moved at step %d" % i)
        # ... while the online net is moving the whole time.
        self.assertFalse(np.allclose(self.trace[self.SYNC_AT[0] - 1][0],
                                     self.trace[1][0]))

    def test_a_sync_copies_the_online_weights_into_the_target(self):
        for i in self.SYNC_AT:
            online, target = self.trace[i]
            np.testing.assert_allclose(target, online,
                                       err_msg="no sync visible at step %d" % i)
            # and it really is a change, not a target that was already equal
            self.assertFalse(np.allclose(target, self.trace[i - 1][1]),
                             "target did not move at step %d" % i)

    def test_a_sync_never_overwrites_the_online_net(self):
        # `online.copy_from(target)` -- the arguments the wrong way round --
        # resets the online net to the target's stale weights at every sync.
        # The call count is identical and every component test still passes,
        # but learning is destroyed. Here the online net would snap back to
        # its initial weights at each sync, so pin that it never does.
        init = QNetwork(OBS_DIM, self.cfg.hidden, 2, seed=self.cfg.seed).w1
        for i in self.SYNC_AT:
            self.assertFalse(np.allclose(self.trace[i][0], init),
                             "online net was reset to its init at step %d" % i)

    def test_the_online_net_moves_monotonically_away_from_its_init(self):
        # A second angle on the same corruption, independent of where the
        # syncs fall: training must leave the returned net somewhere the
        # initialisation is not.
        init = QNetwork(OBS_DIM, self.cfg.hidden, 2, seed=self.cfg.seed).w1
        self.assertFalse(np.allclose(self.policy.net.w1, init))
        early = np.linalg.norm(self.trace[1][0] - init)
        late = np.linalg.norm(self.trace[-1][0] - init)
        self.assertGreater(late, early)

    def test_the_returned_policy_holds_the_online_net_not_the_target(self):
        # Returning DQNPolicy(target) would hand back a net that is up to
        # target_sync gradient steps stale. Training ends on step 36, which is
        # not a sync point, so the two nets are genuinely apart here.
        self.assertFalse(np.allclose(self.policy.net.w1, self.trace[-1][1]))
        # The returned net is one gradient step past the last online snapshot,
        # so it must sit far closer to that than to the target.
        to_online = np.linalg.norm(self.policy.net.w1 - self.trace[-1][0])
        to_target = np.linalg.norm(self.policy.net.w1 - self.trace[-1][1])
        self.assertLess(to_online, to_target)


class _OffScaleEnv:
    """An environment whose six features live on wildly different scales.

    This is the defect under test, reproduced small enough to assert on. The
    real HybridEnv's per-feature standard deviations span two orders of
    magnitude (module docstring of dqn); here they span six by construction,
    and one feature is CONSTANT so the epsilon guard is exercised by the
    training loop itself and not only by a unit test on the normaliser.

    Feature 0 is the only one that says anything: it is +SIGNAL when the
    paying arm is 0 and -SIGNAL when it is 1, and it is deliberately the
    SMALLEST-amplitude feature in the vector, exactly as QBER is the smallest
    real feature while being the one the decision turns on.
    """

    SIGNAL = 0.02          # the informative feature's amplitude
    LOUD = 500.0           # an irrelevant feature that shouts

    def __init__(self, good, horizon=12, seed=0):
        self.good = int(good)
        self.horizon = int(horizon)
        self.seeds = []
        self._rng = np.random.RandomState(seed)
        self._t = 0
        self._paid = 0.0

    def _obs(self):
        r = self._rng.randn(OBS_DIM)
        v = np.zeros(OBS_DIM)
        v[0] = (self.SIGNAL if self.good == 0 else -self.SIGNAL) + 1e-4 * r[0]
        v[1] = self.LOUD * r[1]              # loud noise
        v[2] = 7.0 + self.LOUD * r[2]        # loud noise, offset
        v[3] = 3.0                           # constant: zero variance
        v[4] = 1e-6 * r[4]                   # nearly constant
        v[5] = self._t / self.horizon
        return v

    def reset(self, seed=None):
        self.seeds.append(seed)
        self._rng = np.random.RandomState(0 if seed is None else int(seed))
        self._t = 0
        self._paid = 0.0
        return self._obs()

    def step(self, action):
        reward = 1.0 if int(action) == self.good else 0.0
        self._paid += reward
        self._t += 1
        return self._obs(), reward, self._t >= self.horizon, {}

    @property
    def eta(self):
        return self._paid / self.horizon


class _StubNet:
    """Q = [x0, -x0]: the greedy action is the SIGN of the first feature.

    A real QNetwork cannot be forced to a known decision boundary without
    solving for its weights, and a boundary that is known exactly is what
    turns "does inference normalise" from a statistical claim into an
    equality.
    """

    def forward(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.column_stack([x[:, 0], -x[:, 0]])


class TestObsNormalizer(unittest.TestCase):
    """The scaling itself. Every test names the mutation it exists to kill."""

    def test_an_unfitted_normaliser_is_the_identity(self):
        # So ObsNormalizer() is a usable "no statistics yet" object and no
        # separate identity class exists for DQNPolicy to be handed by
        # accident.
        x = np.random.RandomState(0).randn(4, OBS_DIM)
        np.testing.assert_allclose(
            ObsNormalizer(OBS_DIM, clip=1e9).transform(x), x)

    def test_it_centres_and_scales_to_the_sample_statistics(self):
        # Welford must agree with the two-pass formula it replaces; an
        # off-by-one in the update leaves a plausible but wrong variance.
        # One scale per observation feature; the list must track OBS_DIM
        # (7 since the distance became observable) or numpy broadcasts it
        # against the wrong axis and the test errors out.
        data = np.random.RandomState(1).randn(500, OBS_DIM) * [
            1.0, 0.02, 300.0, 5.0, 0.5, 7.0, 0.3] + 3.0
        norm = ObsNormalizer(OBS_DIM)
        for row in data:
            norm.update(row)
        np.testing.assert_allclose(norm.mean, data.mean(axis=0), rtol=1e-10)
        np.testing.assert_allclose(norm.var, data.var(axis=0), rtol=1e-8)

    def test_it_equalises_features_that_differ_by_orders_of_magnitude(self):
        # The defect in one assertion: raw, the loud feature outweighs the
        # informative one by a factor of 1e4; standardised, they weigh the same.
        data = np.random.RandomState(2).randn(2000, OBS_DIM) * [
            0.02, 300.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        norm = ObsNormalizer(OBS_DIM)
        for row in data:
            norm.update(row)
        z = norm.transform(data)
        self.assertGreater(data.std(axis=0)[1] / data.std(axis=0)[0], 1e4)
        np.testing.assert_allclose(z.std(axis=0), np.ones(OBS_DIM), rtol=0.05)

    def test_a_zero_variance_feature_survives_instead_of_becoming_nan(self):
        # MUTATION: divide by sqrt(var) with no epsilon. A constant feature
        # then yields 0/0 = nan, which poisons every weight in the first layer
        # on the next gradient step and shows up only as a policy that picks
        # PM forever.
        data = np.random.RandomState(3).randn(100, OBS_DIM)
        data[:, 2] = 4.0
        norm = ObsNormalizer(OBS_DIM)
        for row in data:
            norm.update(row)
        z = norm.transform(data)
        self.assertTrue(np.isfinite(z).all())
        np.testing.assert_allclose(z[:, 2], 0.0, atol=1e-12)

    def test_the_epsilon_guard_does_not_squash_a_real_feature(self):
        # The other half of the same mutation: an epsilon large enough to feel
        # "safe" (1e-2, say, or a plain +1 on the variance) collapses the QBER
        # feature -- std 0.0155 in the study -- to nearly a constant, which is
        # the defect again wearing a guard's clothes.
        data = np.random.RandomState(4).randn(4000, OBS_DIM) * 0.0155
        norm = ObsNormalizer(OBS_DIM)
        for row in data:
            norm.update(row)
        z = norm.transform(data)
        np.testing.assert_allclose(z.std(axis=0), np.ones(OBS_DIM), rtol=0.05)

    def test_it_clips_a_far_outlier_without_touching_the_decision_region(self):
        # A fully jammed slot puts QBER some 30 sigma out; unclipped, one such
        # row dominates a batch's gradient.
        data = np.random.RandomState(5).randn(500, OBS_DIM)
        norm = ObsNormalizer(OBS_DIM, clip=5.0)
        for row in data:
            norm.update(row)
        far = norm.mean + 40.0 * norm.std
        np.testing.assert_allclose(norm.transform(far), 5.0)
        near = norm.mean + 1.4 * norm.std
        np.testing.assert_allclose(norm.transform(near), 1.4, rtol=1e-6)

    def test_transform_never_moves_the_statistics(self):
        # MUTATION: a transform() that also updates. The policy would keep
        # adapting while being scored, and two evaluation runs of the SAME
        # policy would differ by the order they were run in.
        norm = ObsNormalizer(OBS_DIM)
        for row in np.random.RandomState(6).randn(50, OBS_DIM):
            norm.update(row)
        before = (norm.count, norm.mean.copy(), norm.var.copy())
        for _ in range(10):
            norm.transform(np.random.RandomState(7).randn(OBS_DIM))
        self.assertEqual(norm.count, before[0])
        np.testing.assert_allclose(norm.mean, before[1])
        np.testing.assert_allclose(norm.var, before[2])

    def test_a_snapshot_does_not_move_when_the_original_keeps_learning(self):
        # MUTATION: return DQNPolicy(online, norm) holding the LIVE object. An
        # issued policy's scaling would then depend on training that happened
        # after it was issued.
        norm = ObsNormalizer(OBS_DIM)
        for row in np.random.RandomState(8).randn(30, OBS_DIM):
            norm.update(row)
        frozen = norm.snapshot()
        np.testing.assert_allclose(frozen.mean, norm.mean)
        for row in np.random.RandomState(9).randn(30, OBS_DIM) + 100.0:
            norm.update(row)
        self.assertFalse(np.allclose(frozen.mean, norm.mean))

    def test_it_reports_whether_it_was_fitted(self):
        self.assertFalse(ObsNormalizer(OBS_DIM).fitted)
        norm = ObsNormalizer(OBS_DIM)
        norm.update(np.zeros(OBS_DIM))
        self.assertFalse(norm.fitted)      # one sample estimates no variance
        norm.update(np.ones(OBS_DIM))
        self.assertTrue(norm.fitted)


class TestInferenceNormalises(unittest.TestCase):
    """MUTATION: fitted during training, not applied at inference.

    The classic form of this bug. It does not raise; it scores a network on
    inputs from a distribution it never saw, which reads as a mediocre policy.
    """

    def _shifted(self, mean0):
        norm = ObsNormalizer(OBS_DIM)
        # Two samples straddling mean0 give a fitted normaliser with a known
        # mean and a nonzero variance, without a loop.
        norm.update(np.full(OBS_DIM, mean0 - 1.0))
        norm.update(np.full(OBS_DIM, mean0 + 1.0))
        return norm

    def test_act_uses_the_normaliser_and_not_the_raw_observation(self):
        # Q = [x0, -x0], so the action is the sign of feature 0. Raw, the
        # observation is +0.5 -> action 0. Centred on 1.0 it is negative ->
        # action 1. A DQNPolicy that skipped the transform answers 0.
        policy = DQNPolicy(_StubNet(), self._shifted(1.0))
        self.assertEqual(policy.act(np.full(OBS_DIM, 0.5)), 1)

    def test_the_same_network_answers_the_other_way_on_the_raw_input(self):
        # Guards the guard: the test above is only evidence if the raw answer
        # really differs.
        raw = int(np.argmax(_StubNet().forward(np.full(OBS_DIM, 0.5))[0]))
        self.assertEqual(raw, 0)

    def test_q_values_are_the_networks_values_at_the_normalised_input(self):
        norm = self._shifted(1.0)
        obs = np.full(OBS_DIM, 0.5)
        np.testing.assert_allclose(
            DQNPolicy(_StubNet(), norm).q_values(obs),
            _StubNet().forward(norm.transform(obs))[0])

    def test_a_trained_policy_carries_a_fitted_normaliser(self):
        # MUTATION: train() returns DQNPolicy(online, ObsNormalizer()) -- an
        # unfitted normaliser IS the identity, so this is the "absent at
        # evaluation" bug with a normaliser-shaped object in the slot.
        policy = train(_OffScaleEnv(good=1), _tiny_cfg(episodes=2))
        self.assertTrue(policy.norm.fitted)
        # ... and its statistics are the environment's, not zeros and ones.
        self.assertGreater(abs(policy.norm.mean[2]), 1.0)   # obs[2] ~ 7 + noise
        self.assertGreater(policy.norm.std[1], 10.0)        # the loud feature

    def test_the_normaliser_a_policy_carries_changes_what_it_reports(self):
        # A policy handed a FRESH (identity) normaliser reads the SAME network
        # at a different point, so the shipped statistics demonstrably reach
        # the output. Asserted on Q values rather than on the argmax: a tiny
        # training budget can leave both readings on the same side of the
        # boundary, which would make an action comparison flaky rather than
        # sharp.
        env = _OffScaleEnv(good=0)
        policy = train(env, _tiny_cfg(episodes=3))
        probe = np.array([env.reset(1)] + [env.step(0)[0] for _ in range(20)])
        naive = DQNPolicy(policy.net, ObsNormalizer(OBS_DIM))
        gaps = [float(np.max(np.abs(policy.q_values(o) - naive.q_values(o))))
                for o in probe]
        self.assertGreater(max(gaps), 1e-6)


class TestTrainingNormalises(unittest.TestCase):
    """MUTATION: normalised at inference but raw during training -- and the
    obs/next_obs half-application.

    Asserted on the arrays that actually reach the network, captured by
    spying on `bellman_target` and `QNetwork.train_step`, rather than on the
    outcome: an outcome test would be statistical, and each of these
    mutations leaves a loop that runs to completion and returns a policy.
    """

    def _capture(self, env, cfg):
        seen = {"obs": [], "next": [], "x": []}
        real_bellman = dqn.bellman_target
        real_step = QNetwork.train_step

        def spy_bellman(online, target, obs, act, rew, next_obs, done, gamma):
            seen["obs"].append(np.asarray(obs, dtype=float).copy())
            seen["next"].append(np.asarray(next_obs, dtype=float).copy())
            return real_bellman(online, target, obs, act, rew, next_obs,
                                done, gamma)

        def spy_step(self_net, x, target, lr):
            seen["x"].append(np.asarray(x, dtype=float).copy())
            return real_step(self_net, x, target, lr)

        dqn.bellman_target = spy_bellman
        QNetwork.train_step = spy_step
        try:
            policy = train(env, cfg)
        finally:
            dqn.bellman_target = real_bellman
            QNetwork.train_step = real_step
        return seen, policy

    def setUp(self):
        self.env = _OffScaleEnv(good=1, horizon=40)
        cfg = _tiny_cfg(episodes=6, warmup=32, batch_size=16)
        self.seen, self.policy = self._capture(self.env, cfg)
        self.assertGreater(len(self.seen["obs"]), 50)

    def _assert_standardised(self, batches, what):
        # Pooled over every gradient step: each feature must arrive centred
        # and at unit scale. Raw, feature 1 has std ~500 and feature 2 a mean
        # of 7, so either half-application is a factor of hundreds out.
        pooled = np.concatenate(batches, axis=0)
        self.assertLess(np.abs(pooled.mean(axis=0)).max(), 0.35,
                        "%s not centred: %s" % (what, pooled.mean(axis=0)))
        spread = pooled.std(axis=0)
        for i in (0, 1, 2, 5):      # 3 is constant, 4 is numerical dust
            self.assertGreater(spread[i], 0.5, "%s feature %d flat: %.4g"
                               % (what, i, spread[i]))
            self.assertLess(spread[i], 2.0, "%s feature %d loud: %.4g"
                            % (what, i, spread[i]))

    def test_the_obs_batch_reaching_the_bellman_target_is_standardised(self):
        self._assert_standardised(self.seen["obs"], "obs")

    def test_the_next_obs_batch_is_standardised_too(self):
        # MUTATION: normalise obs but not next_obs. Q(s) and the bootstrap
        # max Q(s') would then be read off two different input scales, which
        # is a Bellman equation between two different functions.
        self._assert_standardised(self.seen["next"], "next_obs")

    def test_the_regression_input_is_the_batch_the_target_was_built_on(self):
        # MUTATION: normalise for bellman_target but regress on the raw batch,
        # or the reverse. The target would be Q of one input while the
        # gradient step moved Q of another.
        self.assertEqual(len(self.seen["x"]), len(self.seen["obs"]))
        for i, (x, obs) in enumerate(zip(self.seen["x"], self.seen["obs"])):
            np.testing.assert_allclose(x, obs, err_msg="step %d" % i)

    def test_the_environment_really_hands_out_off_scale_observations(self):
        # Guards the guard: if the fixture were already standardised, every
        # assertion above would pass on an implementation that normalises
        # nothing at all.
        env = _OffScaleEnv(good=1, horizon=40)
        raw = np.array([env.reset(0)] + [env.step(1)[0] for _ in range(200)])
        self.assertGreater(raw.std(axis=0)[1], 100.0)
        self.assertGreater(abs(raw.mean(axis=0)[2]), 5.0)
        self.assertLess(raw.std(axis=0)[0], 0.01)

    def test_the_greedy_branch_of_training_sees_the_same_scaling(self):
        # MUTATION: normalise the gradient step but act on raw obs while
        # collecting. The behaviour policy and the learned policy would then
        # be two different functions of the state, so the buffer would fill
        # with transitions the agent itself would never generate.
        calls = []
        real_forward = QNetwork.forward

        def spy(self_net, x):
            calls.append(np.asarray(x, dtype=float).copy())
            return real_forward(self_net, x)

        QNetwork.forward = spy
        try:
            train(_OffScaleEnv(good=1, horizon=20),
                  _tiny_cfg(episodes=3, warmup=10_000, capacity=10_000,
                            eps_start=0.0, eps_end=0.0))
        finally:
            QNetwork.forward = real_forward
        # warmup is unreachable here, so every forward IS an action choice.
        single = np.array([c.reshape(-1) for c in calls
                           if np.asarray(c).reshape(-1).size == OBS_DIM])
        self.assertGreater(len(single), 20)
        self.assertLess(np.abs(single[:, 1]).max(), 20.0,
                        "acting on raw observations: feature 1 reaches %.4g"
                        % np.abs(single[:, 1]).max())


class TestNormalisationMakesTheQuietFeatureUsable(unittest.TestCase):
    """The end-to-end reason the fix exists.

    _OffScaleEnv's only informative feature has amplitude 0.02 against noise
    of amplitude 500 -- the study's QBER-against-block situation, exaggerated
    enough that a millisecond budget can resolve it. An unnormalised agent
    cannot learn it; a normalised one must.
    """

    def _learn(self, good):
        env = _OffScaleEnv(good=good, horizon=25)
        policy = train(env, _chain_cfg(episodes=60, gamma=0.0))
        probe = np.array([env.reset(5)]
                         + [env.step(good)[0] for _ in range(199)])
        picked = [policy.act(o) for o in probe]
        return sum(1 for a in picked if a == good) / len(picked)

    def test_it_learns_the_quiet_feature_both_ways_round(self):
        for good in (0, 1):
            share = self._learn(good)
            self.assertGreater(share, 0.9,
                               "good=%d chose it only %.0f%% of the time"
                               % (good, 100 * share))


if __name__ == "__main__":
    unittest.main()
