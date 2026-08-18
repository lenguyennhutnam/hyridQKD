#!/usr/bin/env python3
"""A small DQN written in numpy.

Six state features and two actions do not justify a deep-learning framework,
and the project rule is no new dependencies. Two hidden layers of ReLU with
plain SGD is enough, and it keeps every step of the update auditable: the
Bellman target and the epsilon schedule are ordinary module-level functions
rather than three lines buried in a loop, so each can be pinned by a test on
its own.

Seeding scheme
--------------
`HybridEnv.reset()` with no argument replays ONE trajectory byte for byte.
Training against that would show the agent a single episode `episodes` times
and then let Task 7 score it on a set containing that very episode. So every
episode is reset with its own seed:

    episode_seed(config, episode) = config.seed * SEED_STRIDE + episode

with SEED_STRIDE = 100_000. Training therefore consumes exactly the seeds in
`training_seed_range(config)`, i.e.

    [config.seed * 100_000, config.seed * 100_000 + config.episodes)

The default `DQNConfig()` (seed=0, episodes=200) consumes seeds 0..199.
Task 7 MUST evaluate on seeds outside that half-open interval -- the safe
habit is to reserve a whole stride block that no training run uses, e.g.
seeds 99_000_000+, which corresponds to config.seed = 990 and is far outside
any seed a study would train with. `episodes > SEED_STRIDE` is refused rather
than allowed to wrap, because wrapping would make two different config seeds
share trajectories and quietly break that disjointness promise.

Spec: docs/superpowers/specs/2026-08-17-hybrid-policy-design.md
"""

from dataclasses import dataclass

import numpy as np

from hybridenv import OBS_DIM
from policies import Policy

# Distance between the seed blocks of two config seeds. Also the hard cap on
# episodes per run: past it, block n and block n+1 would overlap.
SEED_STRIDE = 100_000

N_ACTIONS = 2       # PM and ENT; hybridenv defines nothing else.

__all__ = ["SEED_STRIDE", "N_ACTIONS", "DQNConfig", "QNetwork", "ReplayBuffer",
           "DQNPolicy", "bellman_target", "epsilon_at", "episode_seed",
           "training_seed_range", "validate_config", "train"]


@dataclass
class DQNConfig:
    episodes: int = 200
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    capacity: int = 100_000
    warmup: int = 1_000
    target_sync: int = 500          # gradient steps between target copies
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 150
    seed: int = 0


class QNetwork:
    """obs -> ReLU -> ReLU -> Q(a). Weights are public so tests can pin them."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.RandomState(seed)
        # He initialisation: ReLU halves the variance at each layer.
        self.w1 = rng.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_hidden)
        self.w3 = rng.randn(n_hidden, n_out) * np.sqrt(2.0 / n_hidden)
        self.b3 = np.zeros(n_out)

    def forward(self, x):
        """Q values for a batch, or for a single observation.

        atleast_2d is what lets DQNPolicy.act() hand in a bare (OBS_DIM,)
        vector and still index the result as [0].
        """
        x = np.atleast_2d(x)
        self._h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        self._h2 = np.maximum(0.0, self._h1 @ self.w2 + self.b2)
        self._x = x
        return self._h2 @ self.w3 + self.b3

    def train_step(self, x, target, lr):
        """One SGD step on mean squared error; returns the loss BEFORE it.

        Reporting the pre-update loss is deliberate: it is the loss the
        gradient was taken at, so a learning curve built from it cannot
        flatter the step that produced it.

        The output gradient is divided by `diff.size`, not by the batch
        length, so it is exactly d(mean squared error)/d(output). Dividing by
        the batch length only would leave the analytic gradient a factor
        n_out too large -- harmless as a rescaled learning rate, but it would
        make a finite-difference check of this function impossible, and that
        check is the only thing standing between us and a subtly wrong
        backward pass.
        """
        out = self.forward(x)
        diff = out - target
        loss = float(np.mean(diff ** 2))
        g_out = 2.0 * diff / diff.size
        g_w3 = self._h2.T @ g_out
        g_b3 = g_out.sum(axis=0)
        g_h2 = (g_out @ self.w3.T) * (self._h2 > 0)
        g_w2 = self._h1.T @ g_h2
        g_b2 = g_h2.sum(axis=0)
        g_h1 = (g_h2 @ self.w2.T) * (self._h1 > 0)
        g_w1 = self._x.T @ g_h1
        g_b1 = g_h1.sum(axis=0)
        # w3 is read by g_h2 above, so every gradient must be computed before
        # any parameter is written.
        for param, grad in ((self.w3, g_w3), (self.b3, g_b3),
                            (self.w2, g_w2), (self.b2, g_b2),
                            (self.w1, g_w1), (self.b1, g_b1)):
            param -= lr * grad
        return loss

    def copy_from(self, other):
        """Deep-copy `other`'s weights.

        The copy is what makes a target network a target network: sharing the
        arrays instead would leave the Bellman target chasing the online net's
        own moving estimate, with no visible symptom.
        """
        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            setattr(self, name, getattr(other, name).copy())


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions.

    Sampling draws uniformly over the entries actually written -- never over
    the whole capacity, which would hand back all-zero rows that look like
    perfectly ordinary transitions.
    """

    def __init__(self, capacity, obs_dim=OBS_DIM, seed=0):
        self.capacity = int(capacity)
        self._obs = np.zeros((self.capacity, obs_dim))
        self._act = np.zeros(self.capacity, dtype=int)
        self._rew = np.zeros(self.capacity)
        self._next = np.zeros((self.capacity, obs_dim))
        self._done = np.zeros(self.capacity)
        self._n = 0
        self._i = 0
        self._rng = np.random.RandomState(seed)

    def add(self, obs, action, reward, next_obs, done):
        i = self._i
        self._obs[i] = obs
        self._act[i] = action
        self._rew[i] = reward
        self._next[i] = next_obs
        self._done[i] = 1.0 if done else 0.0
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def sample(self, n):
        """A batch of `n` transitions drawn with replacement.

        One index draw feeds all five columns, so a sampled row is a
        transition that really happened.
        """
        if self._n == 0:
            raise ValueError("sample from an empty replay buffer")
        idx = self._rng.randint(0, self._n, size=n)
        return (self._obs[idx], self._act[idx], self._rew[idx],
                self._next[idx], self._done[idx])

    def __len__(self):
        return self._n


class DQNPolicy(Policy):
    """Greedy w.r.t. a trained Q-network."""

    name = "dqn"

    def __init__(self, net):
        self.net = net

    def act(self, obs):
        return int(np.argmax(self.net.forward(obs)[0]))


def bellman_target(online, target, obs, act, rew, next_obs, done, gamma):
    """The regression target matrix for one batch.

        y = r + gamma (1 - done) max_a' Q_target(s', a')

    written into the slot of the action that was actually taken; every other
    slot keeps the online net's current value so no gradient flows through
    it. The bootstrap reads `target`, never `online` -- that separation is
    the whole reason a target network exists.
    """
    q_next = target.forward(next_obs).max(axis=1)
    y = rew + gamma * (1.0 - done) * q_next
    out = online.forward(obs).copy()
    out[np.arange(len(act)), np.asarray(act, dtype=int)] = y
    return out


def epsilon_at(episode, config):
    """Linear epsilon decay, clamped at `eps_end`.

    Clamped at both ends: the fraction saturates at 1.0 so a run longer than
    eps_decay_episodes keeps exploring at eps_end instead of sliding to a
    negative epsilon (i.e. no exploration at all).
    """
    frac = min(1.0, episode / max(config.eps_decay_episodes, 1))
    return config.eps_start + frac * (config.eps_end - config.eps_start)


def episode_seed(config, episode):
    """The environment seed training uses for `episode`. See module docstring."""
    return config.seed * SEED_STRIDE + int(episode)


def training_seed_range(config):
    """Exactly the seeds a run of `config` consumes -- Task 7 must avoid these."""
    start = config.seed * SEED_STRIDE
    return range(start, start + config.episodes)


def validate_config(config):
    """Refuse configurations that would train silently but pointlessly."""
    if config.episodes > SEED_STRIDE:
        raise ValueError(
            "episodes=%d exceeds SEED_STRIDE=%d: episode seeds would wrap "
            "into the next config seed's block" % (config.episodes,
                                                   SEED_STRIDE))
    if config.target_sync < 1:
        raise ValueError("target_sync must be >= 1, got %d"
                         % config.target_sync)
    if config.warmup > config.capacity:
        raise ValueError(
            "warmup=%d exceeds capacity=%d: the buffer saturates at capacity "
            "so no gradient step would ever fire" % (config.warmup,
                                                     config.capacity))


def train(env, config=None, progress=None):
    """Train on `env`; return a greedy DQNPolicy.

    Each episode is reset with its own seed (see the module docstring), so
    the agent sees `config.episodes` distinct trajectories rather than one
    trajectory `episodes` times.

    Rewards are already in served-messages units (one served message = 1), so
    no reward normalisation is needed.
    """
    cfg = config or DQNConfig()
    validate_config(cfg)
    rng = np.random.RandomState(cfg.seed)
    online = QNetwork(OBS_DIM, cfg.hidden, N_ACTIONS, seed=cfg.seed)
    target = QNetwork(OBS_DIM, cfg.hidden, N_ACTIONS, seed=cfg.seed)
    target.copy_from(online)
    buf = ReplayBuffer(cfg.capacity, OBS_DIM, seed=cfg.seed)
    steps = 0
    for episode in range(cfg.episodes):
        eps = epsilon_at(episode, cfg)
        obs = env.reset(episode_seed(cfg, episode))
        done = False
        while not done:
            if rng.rand() < eps:
                action = int(rng.randint(N_ACTIONS))
            else:
                action = int(np.argmax(online.forward(obs)[0]))
            next_obs, reward, done, _ = env.step(action)
            buf.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if len(buf) >= cfg.warmup:
                batch = buf.sample(cfg.batch_size)
                y = bellman_target(online, target, batch[0], batch[1],
                                   batch[2], batch[3], batch[4], cfg.gamma)
                online.train_step(batch[0], y, cfg.lr)
                steps += 1
                if steps % cfg.target_sync == 0:
                    target.copy_from(online)
        if progress is not None:
            progress(episode, env.eta, eps)
    return DQNPolicy(online)
