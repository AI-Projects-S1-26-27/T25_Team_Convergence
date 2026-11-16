"""
baselines/predictive_interception.py  (Task T-05 — Baseline 2: Predictive Interception)
=======================================================================================
Rule (proposal, Section 4):
    "Each moves toward the prey's projected position two turns ahead,
     extrapolating its previous move and clipping to the arena."

    last move   delta  = prey - prev_prey          e.g. (0, +1) = moved East
    projection  target = prey + 2 * delta          clipped to rows/cols 0..19

Every predator, independently, takes one greedy step toward `target`
(same step_toward helper as Direct Pursuit — only the target differs).

On the very first step of an episode there is no previous prey position
(prev_prey is None), so the projection is just the prey's current cell —
i.e. it behaves exactly like Direct Pursuit for step 0. The same happens
whenever the prey stood still last turn (delta = 0).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

import config
from baselines.common import BaselinePolicy, Pos, step_toward

HORIZON = 2  # turns ahead to project (proposal: "two turns ahead")


def _clip(v: int, grid_size: int) -> int:
    return max(0, min(grid_size - 1, v))


def project_prey(
    prey: Pos,
    prev_prey: Optional[Pos],
    horizon: int = HORIZON,
    grid_size: int = config.GRID_SIZE,
) -> Pos:
    """Where the prey will be in `horizon` turns if it repeats its last move."""
    if prev_prey is None:
        return tuple(prey)
    dr, dc = prey[0] - prev_prey[0], prey[1] - prev_prey[1]
    return (_clip(prey[0] + horizon * dr, grid_size), _clip(prey[1] + horizon * dc, grid_size))


def predictive_interception_actions(
    predator_positions: Sequence[Pos],
    prey_position: Pos,
    prev_prey_position: Optional[Pos],
    rng: np.random.Generator,
) -> List[int]:
    """One greedy step toward the projected prey cell, for each predator."""
    target = project_prey(prey_position, prev_prey_position)
    return [step_toward(p, target, rng) for p in predator_positions]


class PredictiveInterception(BaselinePolicy):
    """Baseline 2. Target = prey's projected cell 2 turns ahead, for every predator."""

    name = "predictive_interception"

    def choose_actions(
        self, predators: Sequence[Pos], prey: Pos, prev_prey: Optional[Pos]
    ) -> List[int]:
        return predictive_interception_actions(predators, prey, prev_prey, self.rng)
