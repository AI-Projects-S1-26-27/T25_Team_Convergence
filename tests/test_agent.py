"""tests/test_agent.py -- rl/agent.py (T-08)"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from rl.agent import PredatorAgent, TeamPolicy, get_epsilon, save_team
from rl.observation import encode_team_observations


def test_q_table_shape_and_zero_init():
    a = PredatorAgent()
    assert a.q_table.shape == (16200, 5)
    assert not a.q_table.any()


def test_td_update_matches_formula():
    a = PredatorAgent()
    a.q_table[7] = [0.0, 1.0, 0.5, 0.0, 0.0]   # max_a' Q(next) = 1.0
    a.q_table[3, 2] = 0.4
    a.update(3, 2, reward=-0.01, next_obs=7, done=False, alpha=0.2, gamma=0.99)
    target = -0.01 + 0.99 * 1.0
    assert a.q_table[3, 2] == pytest.approx(0.4 + 0.2 * (target - 0.4), rel=1e-5)


def test_td_update_no_bootstrap_when_done():
    a = PredatorAgent()
    a.q_table[7] = [9.0] * 5
    a.update(3, 2, reward=10.0, next_obs=7, done=True, alpha=0.2, gamma=0.99)
    assert a.q_table[3, 2] == pytest.approx(0.2 * 10.0)  # next state's value ignored


def test_greedy_picks_argmax_and_ties_are_random():
    a = PredatorAgent(seed=0)
    a.q_table[5] = [0.0, 0.0, 3.0, 0.0, 0.0]
    assert all(a.act(5, 0.0) == 2 for _ in range(20))
    picks = {a.act(9, 0.0) for _ in range(200)}  # all-zero row -> 5-way tie
    assert picks == {0, 1, 2, 3, 4}


def test_epsilon_one_is_uniform_random_and_zero_is_greedy():
    a = PredatorAgent(seed=1)
    a.q_table[5, 3] = 1.0
    assert all(a.act(5, 0.0) == 3 for _ in range(20))
    picks = [a.act(5, 1.0) for _ in range(500)]
    assert len(set(picks)) == 5


def test_epsilon_schedule():
    assert get_epsilon(0) == 1.0
    assert get_epsilon(4000) == pytest.approx(0.525)
    assert get_epsilon(8000) == 0.05
    assert get_epsilon(9999) == 0.05


def test_save_load_roundtrip(tmp_path):
    a = PredatorAgent(seed=0)
    a.update(1, 1, 1.0, 2, False)
    p = str(tmp_path / "q.npy")
    a.save(p)
    b = PredatorAgent()
    b.load(p)
    assert np.array_equal(a.q_table, b.q_table)
    with pytest.raises(ValueError):
        PredatorAgent(n_states=10).load(p)


def test_visited_states_counts_nonzero_rows():
    a = PredatorAgent()
    assert a.visited_states() == 0
    a.update(4, 0, 1.0, 5, True)
    a.update(8, 1, 1.0, 5, True)
    assert a.visited_states() == 2


def test_team_policy_is_greedy_and_loads_from_dir(tmp_path):
    agents = [PredatorAgent(seed=i) for i in range(3)]
    env_obs = {"predators": [(5, 5), (6, 6), (9, 9)], "prey": (5, 8), "prev_prey": None, "step": 0}
    states = encode_team_observations(env_obs)
    for ag, s in zip(agents, states):
        ag.q_table[s, 2] = 1.0
    team = TeamPolicy(agents, encode_team_observations)
    assert team(env_obs) == [2, 2, 2]

    save_team(agents, str(tmp_path))
    loaded = TeamPolicy.from_dir(str(tmp_path), encode_team_observations)
    assert loaded(env_obs) == [2, 2, 2]
