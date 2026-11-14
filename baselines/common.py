"""
baselines/common.py  (Group B — shared helpers for all scripted baselines)
=========================================================================
Every scripted baseline in this folder works the same way:

    1. pick a TARGET cell for each predator
         - Direct Pursuit:           the prey's current cell
         - Predictive Interception:  the prey's projected cell 2 turns ahead
         - Coordinated Pursuit:      a different cell near the projection per predator
    2. move each predator ONE step toward its target  ->  step_toward()

This module holds the pieces they share, plus `BaselinePolicy`, a small base
class that plugs straight into the team's shared evaluator
(experiments/evaluate.py), which expects:

    policy(env_obs) -> [a0, a1, a2]      # env_obs = dict from env.reset()/env.step()
    policy.reset(seed)                   # optional; called before every scenario

Randomness: baselines only use randomness to break ties between equally good
moves. They use their OWN generator (never the prey's), re-seeded from the
scenario seed in reset(), so every run of a scenario is exactly reproducible.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from agents.prey import get_legal_moves  # noqa: E402  (same wall rule as the prey)

Pos = Tuple[int, int]


def manhattan(a: Pos, b: Pos) -> int:
    """Grid distance |dr| + |dc| (agents can't move diagonally)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def apply_action(pos: Pos, action: int) -> Pos:
    """Cell reached by taking `action` from `pos` (no bounds check)."""
    dr, dc = config.ACTIONS[action]
    return (pos[0] + dr, pos[1] + dc)


def legal_actions(pos: Pos, grid_size: int = config.GRID_SIZE) -> List[int]:
    """Actions that keep an agent inside the grid. Stay (0) is always included."""
    return get_legal_moves(pos, grid_size)


def step_toward(
    pos: Pos,
    target: Pos,
    rng: np.random.Generator,
    grid_size: int = config.GRID_SIZE,
) -> int:
    """
    Greedy move: the legal action whose resulting cell is closest (Manhattan)
    to `target`. Out-of-bounds moves are never considered. If several actions
    are equally close (e.g. target is diagonal, so North and East both help),
    one of them is picked at random with `rng`.
    """
    best: List[int] = []
    best_d: Optional[int] = None
    for a in legal_actions(pos, grid_size):
        d = manhattan(apply_action(pos, a), target)
        if best_d is None or d < best_d:
            best, best_d = [a], d
        elif d == best_d:
            best.append(a)
    if len(best) == 1:
        return best[0]
    return int(rng.choice(best))


class BaselinePolicy:
    """
    Base class for scripted teams. Subclasses implement `choose_actions`.

    Usage with the shared evaluator:
        from experiments.evaluate import evaluate
        results = evaluate(DirectPursuit(), scenarios)
    """

    name = "baseline"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the tie-break generator (called by evaluate.py per scenario)."""
        self.rng = np.random.default_rng([self._seed, 0 if seed is None else seed])

    def __call__(self, env_obs: dict) -> List[int]:
        actions = self.choose_actions(
            list(env_obs["predators"]), tuple(env_obs["prey"]), env_obs.get("prev_prey")
        )
        return [int(a) for a in actions]

    def choose_actions(
        self, predators: Sequence[Pos], prey: Pos, prev_prey: Optional[Pos]
    ) -> List[int]:
        raise NotImplementedError
