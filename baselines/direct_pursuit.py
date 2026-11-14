"""
baselines/direct_pursuit.py  (Task T-04 — Baseline 1: Direct Pursuit)
=====================================================================
Rule (proposal, Section 4):
    "Each predator moves to minimize distance to the current prey cell."

Every predator, independently, takes the legal move (Stay/N/E/S/W, never off
the grid) that brings it closest — in Manhattan distance — to where the prey
is RIGHT NOW. Ties are broken at random with the policy's own seeded RNG.
It ignores teammates and ignores where the prey is heading, so the three
predators tend to bunch up and chase from behind.

    from baselines.direct_pursuit import DirectPursuit
    from experiments.evaluate import evaluate
    results = evaluate(DirectPursuit(), scenarios)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from baselines.common import BaselinePolicy, Pos, step_toward


def direct_pursuit_actions(
    predator_positions: Sequence[Pos],
    prey_position: Pos,
    rng: np.random.Generator,
) -> List[int]:
    """One greedy step toward the prey's current cell, for each predator."""
    return [step_toward(p, prey_position, rng) for p in predator_positions]


class DirectPursuit(BaselinePolicy):
    """Baseline 1. Target = the prey's current cell, for every predator."""

    name = "direct_pursuit"

    def choose_actions(
        self, predators: Sequence[Pos], prey: Pos, prev_prey: Optional[Pos]
    ) -> List[int]:
        return direct_pursuit_actions(predators, prey, self.rng)
