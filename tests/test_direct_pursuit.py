"""tests/test_direct_pursuit.py -- baselines/direct_pursuit.py (T-04)"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common import apply_action, manhattan
from baselines.direct_pursuit import DirectPursuit, direct_pursuit_actions
from experiments.evaluate import evaluate, run_scenario
from experiments.generate_scenarios import make_val_scenarios

STAY, N, E, S, W = 0, 1, 2, 3, 4


def obs(predators, prey, prev_prey=None, step=0):
    return {"predators": predators, "prey": prey, "prev_prey": prev_prey, "step": step}


def test_corner_predator_moves_east_or_south_toward_centre_prey():
    # Work-division spec: predator at (0,0), prey at (10,10) -> East or South.
    for seed in range(20):
        a = direct_pursuit_actions([(0, 0)], (10, 10), np.random.default_rng(seed))[0]
        assert a in (E, S)


def test_every_predator_gets_one_step_closer():
    preds, prey = [(0, 0), (19, 19), (10, 3)], (10, 10)
    acts = DirectPursuit()(obs(preds, prey))
    assert len(acts) == 3
    for p, a in zip(preds, acts):
        assert manhattan(apply_action(p, a), prey) == manhattan(p, prey) - 1


def test_adjacent_predator_steps_onto_prey_cell():
    assert DirectPursuit()(obs([(5, 4), (0, 0), (19, 19)], (5, 5)))[0] == E


def test_ignores_previous_prey_move():
    preds, prey = [(2, 2), (17, 3), (9, 18)], (10, 10)
    a, b = DirectPursuit(), DirectPursuit()
    a.reset(1)
    b.reset(1)
    assert a(obs(preds, prey, None)) == b(obs(preds, prey, (10, 9), step=5))


def test_predators_act_independently_and_may_share_a_cell():
    # Two predators on the same cell both chase; nothing stops them stacking.
    acts = DirectPursuit()(obs([(10, 5), (10, 5), (0, 0)], (10, 10)))
    assert acts[0] == E and acts[1] == E


def test_full_games_are_valid_and_reproducible():
    scenarios = make_val_scenarios()[:10]          # validation set, not the test set
    r1 = evaluate(DirectPursuit(), scenarios)
    r2 = evaluate(DirectPursuit(), scenarios)
    assert r1 == r2
    for r in r1:
        assert 1 <= r["steps_to_capture"] <= 300
        assert r["captured"] == (r["capture_step"] is not None)


def test_trajectory_moves_one_cell_at_most_per_step():
    res = run_scenario(DirectPursuit(), make_val_scenarios()[0], record=True)
    traj = res["trajectory"]
    for (p0, _), (p1, _) in zip(traj, traj[1:]):
        for a, b in zip(p0, p1):
            assert manhattan(a, b) <= 1
