"""
tests/test_rewards.py  --  unit tests for rl/rewards.py (Task T-09)
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from rl.rewards import base_reward, potential, shaped_reward

PREY = (10, 10)
FAR = [(0, 0), (0, 19), (19, 0)]
NEAR = [(1, 0), (1, 19), (18, 0)]  # each predator one step closer than FAR


def test_base_reward_values():
    assert base_reward(True, 5) == 10.0
    assert base_reward(False, 5) == -0.01


def test_potential_is_bounded_between_minus_one_and_zero():
    assert potential([(0, 0)] * 3, (19, 19)) == pytest.approx(-1.0)  # max distance
    assert potential([PREY] * 3, PREY) == 0.0
    for p in (FAR, NEAR):
        assert -1.0 <= potential(p, PREY) <= 0.0


def test_potential_increases_as_predators_close_in():
    assert potential(NEAR, PREY) > potential(FAR, PREY)


def test_shaped_reward_gives_positive_bonus_when_closing_in():
    bonus = shaped_reward(FAR, PREY, False, NEAR, PREY) - base_reward(False)
    assert bonus > 0
    expected = config.GAMMA * potential(NEAR, PREY) - potential(FAR, PREY)
    assert bonus == pytest.approx(expected)


def test_shaped_reward_penalises_moving_away():
    assert shaped_reward(NEAR, PREY, False, FAR, PREY) < base_reward(False)


def test_shaped_reward_zero_potential_at_capture():
    r = shaped_reward(NEAR, PREY, True, NEAR, PREY)
    assert r == pytest.approx(10.0 - potential(NEAR, PREY))


def test_shaped_reward_zero_potential_at_timeout_when_done_passed():
    r = shaped_reward(NEAR, PREY, False, NEAR, PREY, done=True)
    assert r == pytest.approx(-0.01 - potential(NEAR, PREY))


def test_standing_still_costs_only_the_step_penalty_plus_discount_leak():
    # Phi(s') == Phi(s): shaping reduces to (gamma - 1) * Phi, a tiny positive term.
    r = shaped_reward(FAR, PREY, False, FAR, PREY)
    assert r == pytest.approx(-0.01 + (config.GAMMA - 1.0) * potential(FAR, PREY))
