"""tests/test_predictive_interception.py -- baselines/predictive_interception.py (T-05)"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.direct_pursuit import DirectPursuit
from baselines.predictive_interception import PredictiveInterception, project_prey
from experiments.evaluate import evaluate
from experiments.generate_scenarios import make_val_scenarios

STAY, N, E, S, W = 0, 1, 2, 3, 4


def obs(predators, prey, prev_prey=None, step=1):
    return {"predators": predators, "prey": prey, "prev_prey": prev_prey, "step": step}


def test_projection_extrapolates_two_steps():
    assert project_prey((10, 10), (10, 9)) == (10, 12)   # moved East
    assert project_prey((10, 10), (11, 10)) == (8, 10)   # moved North
    assert project_prey((10, 10), (10, 10)) == (10, 10)  # stood still


def test_projection_is_clipped_to_the_arena():
    assert project_prey((10, 19), (10, 18)) == (10, 19)  # would be col 21
    assert project_prey((1, 5), (2, 5)) == (0, 5)        # would be row -1
    assert project_prey((0, 0), (0, 1)) == (0, 0)


def test_no_previous_move_means_current_cell():
    assert project_prey((4, 7), None) == (4, 7)


def test_targets_cell_ahead_of_eastward_prey_not_current_cell():
    # Work-division spec: prey moving East steadily -> aim ahead of it.
    # A predator already sitting on the projected cell (10,12) waits there;
    # Direct Pursuit would instead step West onto the prey's current column.
    preds = [(10, 12), (0, 0), (19, 0)]
    pi = PredictiveInterception()(obs(preds, (10, 10), (10, 9)))
    dp = DirectPursuit()(obs(preds, (10, 10), (10, 9)))
    assert pi[0] == STAY
    assert dp[0] == W


def test_predator_in_line_moves_toward_projection():
    # Projection (10,12) is directly below (5,12): only South helps.
    assert PredictiveInterception()(obs([(5, 12), (0, 0), (19, 0)], (10, 10), (10, 9)))[0] == S


def test_step_zero_matches_direct_pursuit():
    preds, prey = [(0, 0), (19, 19), (3, 15)], (10, 10)
    for seed in range(10):
        pi, dp = PredictiveInterception(), DirectPursuit()
        pi.reset(seed)
        dp.reset(seed)
        assert pi(obs(preds, prey, None, step=0)) == dp(obs(preds, prey, None, step=0))


def test_full_games_are_valid_and_reproducible():
    scenarios = make_val_scenarios()[:10]
    r1 = evaluate(PredictiveInterception(), scenarios)
    assert r1 == evaluate(PredictiveInterception(), scenarios)
    assert all(1 <= r["steps_to_capture"] <= 300 for r in r1)
