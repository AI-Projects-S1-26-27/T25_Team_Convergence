"""tests/test_observation_no_teammate.py -- rl/observation_no_teammate.py (T-12A)"""

from __future__ import annotations

import itertools
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from rl import observation as full
from rl import observation_no_teammate as nt


def test_state_space_is_1080():
    assert nt.NUM_OBSERVATIONS == 1080


def test_ignores_teammates():
    a = nt.encode_observation((10, 10), (15, 15), [(8, 10), (3, 3)], (15, 14))
    b = nt.encode_observation((10, 10), (15, 15), [(19, 19), (0, 0)], (15, 14))
    assert a == b


def test_index_range_and_bijection():
    seen = set()
    for r, c in itertools.product(range(config.GRID_SIZE), repeat=2):
        for prey in [(10, 10), (0, 0), (19, 19), (3, 17)]:
            for prev in [None, (prey[0], prey[1]), (min(prey[0] + 1, 19), prey[1]) if prey[0] < 19 else (prey[0] - 1, prey[1])]:
                o = nt.encode_observation((r, c), prey, None, prev)
                assert 0 <= o < nt.NUM_OBSERVATIONS
                seen.add(o)
    assert len(seen) > 100  # many distinct states are reachable


def test_team_encoding_length_and_range():
    obs = {"predators": [(1, 1), (10, 10), (18, 18)], "prey": (5, 5), "prev_prey": (5, 4), "step": 3}
    team = nt.encode_team_observations(obs)
    assert len(team) == 3 and all(0 <= o < 1080 for o in team)


def test_agrees_with_full_features_it_keeps():
    f = full.encode_features((10, 10), (15, 15), [(8, 10)], (15, 14))
    expected = f[0] + 8 * f[1] + 24 * f[4] + 120 * f[5]
    assert nt.encode_observation((10, 10), (15, 15), None, (15, 14)) == expected
