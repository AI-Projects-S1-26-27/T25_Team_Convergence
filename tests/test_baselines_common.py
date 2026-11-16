"""tests/test_baselines_common.py -- baselines/common.py (Group B shared helpers)"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines.common import (
    BaselinePolicy,
    apply_action,
    legal_actions,
    manhattan,
    step_toward,
)
from experiments.evaluate import run_scenario
from experiments.generate_scenarios import make_scenario

STAY, N, E, S, W = 0, 1, 2, 3, 4


def rng(seed=0):
    return np.random.default_rng(seed)


def test_manhattan_and_apply_action():
    assert manhattan((0, 0), (3, 4)) == 7
    assert apply_action((5, 5), N) == (4, 5)
    assert apply_action((5, 5), E) == (5, 6)
    assert apply_action((5, 5), S) == (6, 5)
    assert apply_action((5, 5), W) == (5, 4)
    assert apply_action((5, 5), STAY) == (5, 5)


def test_legal_actions_respect_walls():
    assert sorted(legal_actions((0, 0))) == [STAY, E, S]
    assert sorted(legal_actions((19, 19))) == [STAY, N, W]
    assert sorted(legal_actions((10, 10))) == [STAY, N, E, S, W]


def test_step_toward_diagonal_target_picks_a_helpful_move():
    # From the top-left corner toward the centre: only East or South help.
    for seed in range(20):
        assert step_toward((0, 0), (10, 10), rng(seed)) in (E, S)


def test_step_toward_straight_line_is_unique():
    assert step_toward((5, 5), (5, 12), rng()) == E
    assert step_toward((5, 5), (1, 5), rng()) == N


def test_step_toward_stays_when_on_target():
    assert step_toward((7, 7), (7, 7), rng()) == STAY


def test_step_toward_always_closes_distance_and_stays_in_bounds():
    g = rng(123)
    for _ in range(2000):
        pos = tuple(int(x) for x in g.integers(0, 20, 2))
        tgt = tuple(int(x) for x in g.integers(0, 20, 2))
        a = step_toward(pos, tgt, g)
        new = apply_action(pos, a)
        assert 0 <= new[0] < 20 and 0 <= new[1] < 20
        assert manhattan(new, tgt) == max(manhattan(pos, tgt) - 1, 0)


def test_tie_break_is_reproducible_and_uses_both_options():
    g1, g2 = rng(7), rng(7)                           # same seed -> same tie-breaks
    assert [step_toward((0, 0), (10, 10), g1) for _ in range(50)] == \
           [step_toward((0, 0), (10, 10), g2) for _ in range(50)]
    g = rng(0)
    seen = {step_toward((0, 0), (10, 10), g) for _ in range(200)}
    assert seen == {E, S}


class _Chase(BaselinePolicy):
    """Tiny test policy: every predator steps toward the prey."""

    def choose_actions(self, predators, prey, prev_prey):
        return [step_toward(p, prey, self.rng) for p in predators]


def test_base_policy_plugs_into_shared_evaluator_and_is_reproducible():
    sc = make_scenario(999, 0)
    r1 = run_scenario(_Chase(), sc, record=True)
    r2 = run_scenario(_Chase(), sc, record=True)
    assert r1 == r2                                   # same scenario -> identical episode
    assert len(r1["trajectory"]) >= 2
    acts = _Chase()({"predators": [(0, 0), (5, 5), (19, 19)], "prey": (10, 10),
                     "prev_prey": None, "step": 0})
    assert len(acts) == 3 and all(isinstance(a, int) and 0 <= a <= 4 for a in acts)
