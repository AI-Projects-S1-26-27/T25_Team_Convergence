"""
baselines/coordinated_pursuit.py  (Task T-06 — Baseline 3: Coordinated Pursuit)
===============================================================================
Rule (proposal, Section 4):
    "Assign predators to distinct targets among the projected prey cell and its
     legal adjacent cells, minimizing total travel distance; each moves toward
     its assigned target."

Each turn:
  1. Candidates — the prey's projected cell 2 turns ahead (same projection as
     Predictive Interception) plus its in-bounds N/E/S/W neighbours.
     That is 5 cells in the open, 4 on an edge, 3 in a corner — always >= 3.
  2. Assignment — try every way of giving the 3 predators 3 DIFFERENT
     candidates (at most 5*4*3 = 60 ways). Keep the one with the smallest total
     Manhattan distance. Ties are broken at random with the policy's seeded RNG.
     (Brute force; equivalent to the Hungarian algorithm at this size.)
  3. Movement — each predator takes one greedy step toward its own target.

This file replaces an earlier stand-in version written by Group C (which used the
work-division doc's "projected cell + widest adjacent pair" targets); the team chose
the proposal's rule above as the official Coordinated Pursuit. The old class name
`CoordinatedPursuitPolicy` is kept as an alias (see the bottom of this file).

This is the only baseline with explicit role assignment: predators never share
a target, so they stop stacking on the same cell. It does not force them to
spread around all four sides — if all three approach from the west they take
the three western-most candidates.
"""

from __future__ import annotations

from itertools import permutations
from typing import List, Optional, Sequence, Tuple

import numpy as np

import config
from baselines.common import BaselinePolicy, Pos, apply_action, legal_actions, manhattan, step_toward
from baselines.predictive_interception import project_prey


def candidate_targets(center: Pos, grid_size: int = config.GRID_SIZE) -> List[Pos]:
    """The projected cell followed by its in-bounds N/E/S/W neighbours."""
    return [apply_action(center, a) for a in legal_actions(center, grid_size)]  # Stay first


def assign_targets(
    predator_positions: Sequence[Pos],
    candidates: Sequence[Pos],
    rng: np.random.Generator,
) -> List[Pos]:
    """Distinct target per predator, minimising total Manhattan distance."""
    best: List[Tuple[Pos, ...]] = []
    best_cost: Optional[int] = None
    for combo in permutations(candidates, len(predator_positions)):
        cost = sum(manhattan(p, t) for p, t in zip(predator_positions, combo))
        if best_cost is None or cost < best_cost:
            best, best_cost = [combo], cost
        elif cost == best_cost:
            best.append(combo)
    chosen = best[0] if len(best) == 1 else best[int(rng.integers(len(best)))]
    return list(chosen)


def coordinated_targets(
    predator_positions: Sequence[Pos],
    prey_position: Pos,
    prev_prey_position: Optional[Pos],
    rng: np.random.Generator,
) -> List[Pos]:
    """Steps 1+2: the target cell assigned to each predator this turn."""
    center = project_prey(prey_position, prev_prey_position)
    return assign_targets(predator_positions, candidate_targets(center), rng)


def coordinated_pursuit_actions(
    predator_positions: Sequence[Pos],
    prey_position: Pos,
    prev_prey_position: Optional[Pos] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Steps 1-3: one greedy step toward each predator's assigned target."""
    if len(predator_positions) != config.NUM_PREDATORS:
        raise ValueError(f"expected {config.NUM_PREDATORS} predator positions")
    if rng is None:
        rng = np.random.default_rng(0)
    targets = coordinated_targets(predator_positions, prey_position, prev_prey_position, rng)
    return [step_toward(p, t, rng) for p, t in zip(predator_positions, targets)]


class CoordinatedPursuit(BaselinePolicy):
    """Baseline 3. Distinct targets around the projected prey cell, cheapest total travel."""

    name = "coordinated_pursuit"

    def choose_actions(
        self, predators: Sequence[Pos], prey: Pos, prev_prey: Optional[Pos]
    ) -> List[int]:
        return coordinated_pursuit_actions(predators, prey, prev_prey, self.rng)


# Compatibility name: Group C's experiments/run_remaining_evaluations.py imports
# `CoordinatedPursuitPolicy` (the name used by their earlier stand-in version of this
# file). It is the same class, so their script runs unchanged with this baseline.
CoordinatedPursuitPolicy = CoordinatedPursuit
