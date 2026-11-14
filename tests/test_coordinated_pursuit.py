"""tests/test_coordinated_pursuit.py -- baselines/coordinated_pursuit.py (T-06)"""

from __future__ import annotations

import os
import sys
from itertools import permutations

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common import apply_action, manhattan
import pytest

from baselines.coordinated_pursuit import (
    CoordinatedPursuit,
    CoordinatedPursuitPolicy,
    coordinated_pursuit_actions,
    assign_targets,
    candidate_targets,
    coordinated_targets,
)
from experiments.evaluate import evaluate
from experiments.generate_scenarios import make_val_scenarios

STAY, N, E, S, W = 0, 1, 2, 3, 4


def obs(predators, prey, prev_prey=None, step=1):
    return {"predators": predators, "prey": prey, "prev_prey": prev_prey, "step": step}


def rng(seed=0):
    return np.random.default_rng(seed)


def test_candidate_cells_open_edge_corner():
    assert sorted(candidate_targets((10, 10))) == [(9, 10), (10, 9), (10, 10), (10, 11), (11, 10)]
    assert len(candidate_targets((0, 10))) == 4      # top edge
    assert sorted(candidate_targets((0, 0))) == [(0, 0), (0, 1), (1, 0)]  # corner: still 3
    assert candidate_targets((10, 10))[0] == (10, 10)  # projected cell comes first


def test_three_predators_always_get_three_different_targets():
    # Work-division spec: no two predators share a target.
    g = rng(1)
    for _ in range(500):
        preds = [tuple(int(x) for x in g.integers(0, 20, 2)) for _ in range(3)]
        prey = tuple(int(x) for x in g.integers(0, 20, 2))
        prev = prey if g.random() < 0.2 else apply_action(prey, int(g.integers(1, 5)))
        prev = (min(max(prev[0], 0), 19), min(max(prev[1], 0), 19))
        targets = coordinated_targets(preds, prey, prev, g)
        assert len(set(targets)) == 3


def test_assignment_has_minimum_total_distance():
    g = rng(2)
    for _ in range(300):
        preds = [tuple(int(x) for x in g.integers(0, 20, 2)) for _ in range(3)]
        cands = candidate_targets(tuple(int(x) for x in g.integers(0, 20, 2)))
        chosen = assign_targets(preds, cands, g)
        best = min(sum(manhattan(p, t) for p, t in zip(preds, c)) for c in permutations(cands, 3))
        assert sum(manhattan(p, t) for p, t in zip(preds, chosen)) == best


def test_surround_example_from_three_sides():
    # Prey still at (10,10); predators west, north, east of it.
    # Cheapest split (total 12): west->(10,9), north->(9,10), east->(10,11).
    preds = [(10, 5), (5, 10), (10, 15)]
    assert coordinated_targets(preds, (10, 10), (10, 10), rng()) == [(10, 9), (9, 10), (10, 11)]
    assert CoordinatedPursuit()(obs(preds, (10, 10), (10, 10))) == [E, S, W]


def test_uses_projected_cell_as_centre():
    # Prey moving East from (10,9) to (10,10) -> centre (10,12).
    targets = coordinated_targets([(10, 0), (0, 12), (19, 12)], (10, 10), (10, 9), rng())
    assert all(manhattan(t, (10, 12)) <= 1 for t in targets)


def test_step_zero_centres_on_current_prey_cell():
    targets = coordinated_targets([(0, 0), (19, 19), (0, 19)], (7, 7), None, rng())
    assert all(manhattan(t, (7, 7)) <= 1 for t in targets)


def test_each_predator_moves_toward_its_own_target():
    g = rng(3)
    for _ in range(200):
        preds = [tuple(int(x) for x in g.integers(0, 20, 2)) for _ in range(3)]
        prey = tuple(int(x) for x in g.integers(0, 20, 2))
        seed = int(g.integers(1_000_000))
        targets = coordinated_targets(preds, prey, None, rng(seed))
        policy = CoordinatedPursuit()
        policy.rng = rng(seed)  # same RNG stream -> same assignment inside the policy
        acts = policy(obs(preds, prey, None))
        for p, t, a in zip(preds, targets, acts):
            assert manhattan(apply_action(p, a), t) == max(manhattan(p, t) - 1, 0)


def test_full_games_are_valid_and_reproducible():
    scenarios = make_val_scenarios()[:10]
    r1 = evaluate(CoordinatedPursuit(), scenarios)
    assert r1 == evaluate(CoordinatedPursuit(), scenarios)
    assert all(1 <= r["steps_to_capture"] <= 300 for r in r1)


def test_compatible_with_group_c_interface():
    # Group C's robustness script uses CoordinatedPursuitPolicy(seed=0) with the evaluator,
    # and the old function could be called without prev_prey / rng.
    assert CoordinatedPursuitPolicy is CoordinatedPursuit
    scenarios = make_val_scenarios()[:5]
    assert evaluate(CoordinatedPursuitPolicy(seed=0), scenarios) == evaluate(CoordinatedPursuit(seed=0), scenarios)
    acts = coordinated_pursuit_actions([(0, 0), (19, 19), (0, 19)], (10, 10))
    assert len(acts) == 3 and all(0 <= a <= 4 for a in acts)
    with pytest.raises(ValueError):
        coordinated_pursuit_actions([(0, 0), (1, 1)], (10, 10))
