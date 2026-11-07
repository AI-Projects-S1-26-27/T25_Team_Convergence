"""
tests/test_observation.py  --  unit tests for rl/observation.py (Task T-07)
"""

from __future__ import annotations

import itertools
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from env.environment import Environment
from rl.observation import (
    bearing_bin,
    combine_features,
    decode_observation,
    encode_features,
    encode_observation,
    encode_team_observations,
    nearest_teammate,
    prev_prey_move_bin,
    prey_range_bin,
    range_bin,
    teammate_quadrant_bin,
    teammate_range_bin,
    wall_context_bin,
)


def test_state_space_is_16200():
    assert config.NUM_OBSERVATIONS == 16200
    assert combine_features((7, 2, 4, 2, 4, 8)) == 16199
    assert combine_features((0, 0, 0, 0, 0, 0)) == 0


# ---- Feature 1: bearing ---------------------------------------------------
@pytest.mark.parametrize(
    "prey, expected",
    [
        ((5, 10), 0),   # N
        ((5, 15), 1),   # NE
        ((10, 15), 2),  # E
        ((15, 15), 3),  # SE
        ((15, 10), 4),  # S
        ((15, 5), 5),   # SW
        ((10, 5), 6),   # W
        ((5, 5), 7),    # NW
        ((10, 10), 0),  # same cell -> default N
        ((2, 11), 1),   # sign-based: far NNE still NE
    ],
)
def test_bearing_bins(prey, expected):
    assert bearing_bin((10, 10), prey) == expected


# ---- Feature 2 / 4: range bins -------------------------------------------
@pytest.mark.parametrize(
    "dist, expected",
    [(0, 0), (2, 0), (3, 1), (6, 1), (7, 2), (38, 2)],
)
def test_range_bin_boundaries(dist, expected):
    assert range_bin(dist) == expected


def test_prey_and_teammate_range_use_manhattan():
    assert prey_range_bin((0, 0), (1, 1)) == 0      # dist 2
    assert prey_range_bin((0, 0), (3, 3)) == 1      # dist 6
    assert prey_range_bin((0, 0), (4, 3)) == 2      # dist 7
    assert teammate_range_bin((5, 5), (5, 8)) == 1  # dist 3


# ---- Feature 3: nearest-teammate quadrant --------------------------------
@pytest.mark.parametrize(
    "mate, expected",
    [
        ((7, 10), 0),   # N   (dy=-3, dx=0)
        ((10, 13), 1),  # E
        ((13, 10), 2),  # S
        ((10, 7), 3),   # W
        ((10, 10), 4),  # same cell
        ((8, 8), 4),    # exact diagonal -> overlap
        ((8, 12), 4),   # exact diagonal -> overlap
        ((6, 9), 0),    # |dy|=4 > |dx|=1 -> N
        ((9, 14), 1),   # |dx|=4 > |dy|=1 -> E
    ],
)
def test_teammate_quadrant(mate, expected):
    assert teammate_quadrant_bin((10, 10), mate) == expected


def test_nearest_teammate_picks_closest_and_breaks_ties_by_order():
    assert nearest_teammate((0, 0), [(5, 5), (1, 1)]) == (1, 1)
    assert nearest_teammate((0, 0), [(0, 3), (3, 0)]) == (0, 3)
    with pytest.raises(ValueError):
        nearest_teammate((0, 0), [])


# ---- Feature 5: previous prey move ---------------------------------------
@pytest.mark.parametrize(
    "prey, prev, expected",
    [
        ((5, 5), None, 0),
        ((5, 5), (5, 5), 0),
        ((4, 5), (5, 5), 1),  # moved N
        ((5, 6), (5, 5), 2),  # moved E
        ((6, 5), (5, 5), 3),  # moved S
        ((5, 4), (5, 5), 4),  # moved W
    ],
)
def test_prev_prey_move(prey, prev, expected):
    assert prev_prey_move_bin(prey, prev) == expected


def test_prev_prey_move_rejects_teleport():
    with pytest.raises(ValueError):
        prev_prey_move_bin((5, 5), (9, 9))


# ---- Feature 6: wall context ---------------------------------------------
@pytest.mark.parametrize(
    "pos, expected",
    [
        ((0, 0), 0),     # top-left
        ((1, 1), 0),     # still within 2 cells
        ((2, 2), 4),     # interior
        ((17, 17), 4),   # interior
        ((18, 18), 8),   # bottom-right
        ((0, 10), 1),    # top, interior column
        ((10, 0), 3),    # interior row, left
        ((10, 19), 5),   # interior row, right
        ((19, 10), 7),   # bottom, interior column
        ((0, 19), 2),    # top-right
        ((19, 0), 6),    # bottom-left
    ],
)
def test_wall_context(pos, expected):
    assert wall_context_bin(pos) == expected


# ---- Whole encoder --------------------------------------------------------
def test_features_map_to_expected_index():
    # predator (10,10), prey SE at dist 10 (far), mate N close, prey last moved E, interior
    feats = encode_features((10, 10), (15, 15), [(8, 10)], (15, 14))
    assert feats == (3, 2, 0, 0, 2, 4)
    expected = 3 + 8 * 2 + 24 * 0 + 120 * 0 + 360 * 2 + 1800 * 4
    assert encode_observation((10, 10), (15, 15), [(8, 10)], (15, 14)) == expected


def test_encoding_is_bijective_over_all_16200_states():
    seen = set()
    for f in itertools.product(range(8), range(3), range(5), range(3), range(5), range(9)):
        idx = combine_features(f)
        assert 0 <= idx < config.NUM_OBSERVATIONS
        assert decode_observation(idx) == f
        seen.add(idx)
    assert len(seen) == config.NUM_OBSERVATIONS


def test_decode_rejects_out_of_range():
    with pytest.raises(ValueError):
        decode_observation(config.NUM_OBSERVATIONS)
    with pytest.raises(ValueError):
        decode_observation(-1)


def test_every_grid_position_encodes_in_range():
    mates = [(3, 4), (17, 2)]
    for r in range(config.GRID_SIZE):
        for c in range(config.GRID_SIZE):
            obs = encode_observation((r, c), (10, 10), mates, (10, 9))
            assert 0 <= obs < config.NUM_OBSERVATIONS


def test_random_episode_states_stay_in_range():
    env = Environment(seed=7)
    obs = env.reset()
    for _ in range(100):
        for o in encode_team_observations(obs):
            assert 0 <= o < config.NUM_OBSERVATIONS
        obs, done, _ = env.step([1, 2, 3], 4)
        if done:
            break


def test_team_encoding_uses_other_predators_as_teammates():
    env_obs = {
        "predators": [(10, 10), (8, 10), (10, 20 - 1)],
        "prey": (15, 15),
        "prev_prey": None,
        "step": 0,
    }
    team = encode_team_observations(env_obs)
    assert len(team) == 3
    assert team[0] == encode_observation((10, 10), (15, 15), [(8, 10), (10, 19)], None)
    assert team[1] == encode_observation((8, 10), (15, 15), [(10, 10), (10, 19)], None)
    # Different vantage points give different observations.
    assert len(set(team)) == 3
