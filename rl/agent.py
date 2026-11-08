"""
rl/agent.py  (Task T-08)
========================
Independent tabular Q-learning for one predator, plus the epsilon schedule and
a small wrapper that runs a trained team greedily.

Each predator owns its own Q-table (n_states x 5), initially zero:

    Q(o, a) <- Q(o, a) + alpha * [ r + gamma * max_a' Q(o', a') * (1 - done) - Q(o, a) ]

No Q-table, experience or policy is shared between predators.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402


def get_epsilon(
    episode: int,
    total_episodes: int = config.TOTAL_EPISODES,
    eps_start: float = config.EPS_START,
    eps_end: float = config.EPS_END,
    decay_frac: float = config.EPS_DECAY_FRAC,
) -> float:
    """Linear decay eps_start -> eps_end over the first `decay_frac` of training, then flat."""
    decay_episodes = decay_frac * total_episodes
    if episode >= decay_episodes:
        return eps_end
    return eps_start - (eps_start - eps_end) * episode / decay_episodes


class PredatorAgent:
    """One predator's Q-table, action selection and TD update."""

    def __init__(
        self,
        n_states: int = config.NUM_OBSERVATIONS,
        n_actions: int = config.NUM_ACTIONS,
        seed: Optional[int] = None,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    # -- acting -------------------------------------------------------------
    def act(self, obs: int, epsilon: float) -> int:
        """Epsilon-greedy action. Ties in the greedy choice are broken uniformly at random."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(obs)

    def greedy_action(self, obs: int) -> int:
        q = self.q_table[obs]
        best = q.max()
        ties = np.flatnonzero(q == best)
        if len(ties) == 1:
            return int(ties[0])
        return int(ties[self.rng.integers(len(ties))])

    # -- learning -----------------------------------------------------------
    def update(
        self,
        obs: int,
        action: int,
        reward: float,
        next_obs: int,
        done: bool,
        alpha: float = config.ALPHA,
        gamma: float = config.GAMMA,
    ) -> float:
        """One TD update. The bootstrap term is zero when `done` (capture or timeout). Returns the TD error."""
        target = reward + gamma * float(self.q_table[next_obs].max()) * (1.0 - float(done))
        td_error = target - float(self.q_table[obs, action])
        self.q_table[obs, action] += alpha * td_error
        return td_error

    # -- stats / io ---------------------------------------------------------
    def visited_states(self) -> int:
        """Number of states with at least one non-zero Q-value (table coverage)."""
        return int(np.count_nonzero(np.any(self.q_table != 0.0, axis=1)))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            np.save(f, self.q_table)

    def load(self, path: str) -> None:
        table = np.load(path)
        if table.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Q-table shape {table.shape} does not match agent "
                f"({self.n_states}, {self.n_actions})"
            )
        self.q_table = table.astype(np.float32, copy=True)


class TeamPolicy:
    """
    Greedy (epsilon = 0), frozen team: callable `policy(env_obs) -> [a0, a1, a2]`.

    `encode_team` maps the env observation dict to one state index per predator
    (rl.observation.encode_team_observations for the full model).
    """

    def __init__(self, agents: Sequence[PredatorAgent], encode_team: Callable[[dict], List[int]], seed: int = 0):
        self.agents = list(agents)
        self.encode_team = encode_team
        self._seed = seed
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the tie-breaking RNGs so each evaluation episode is reproducible."""
        base = self._seed if seed is None else seed
        for i, agent in enumerate(self.agents):
            agent.rng = np.random.default_rng([base, i])

    def __call__(self, env_obs: dict) -> List[int]:
        states = self.encode_team(env_obs)
        return [agent.greedy_action(s) for agent, s in zip(self.agents, states)]

    @classmethod
    def from_dir(
        cls,
        directory: str,
        encode_team: Callable[[dict], List[int]],
        n_states: int = config.NUM_OBSERVATIONS,
    ) -> "TeamPolicy":
        """Load predator_0.npy .. predator_2.npy from a checkpoint directory."""
        agents = []
        for i in range(config.NUM_PREDATORS):
            a = PredatorAgent(n_states=n_states)
            a.load(os.path.join(directory, f"predator_{i}.npy"))
            agents.append(a)
        return cls(agents, encode_team)


def save_team(agents: Sequence[PredatorAgent], directory: str) -> None:
    """Write predator_0.npy .. predator_2.npy into `directory`."""
    for i, agent in enumerate(agents):
        agent.save(os.path.join(directory, f"predator_{i}.npy"))
