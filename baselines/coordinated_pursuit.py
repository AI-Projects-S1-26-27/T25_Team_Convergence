"""Coordinated pursuit baseline for the predator-prey grid world."""

from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import numpy as np

import config

Pos = Tuple[int, int]


def _projected_prey(prey: Pos, previous_prey: Optional[Pos]) -> Pos:
    if previous_prey is None:
        return prey
    dy, dx = prey[0] - previous_prey[0], prey[1] - previous_prey[1]
    return (
        min(config.GRID_SIZE - 1, max(0, prey[0] + 2 * dy)),
        min(config.GRID_SIZE - 1, max(0, prey[1] + 2 * dx)),
    )


def _targets(prey: Pos, previous_prey: Optional[Pos]) -> list[Pos]:
    """Projected prey cell plus the widest valid pair of adjacent cells."""
    target = _projected_prey(prey, previous_prey)
    r, c = target
    adjacent = []
    for dr, dc in config.ACTIONS[1:]:
        cell = (r + dr, c + dc)
        if 0 <= cell[0] < config.GRID_SIZE and 0 <= cell[1] < config.GRID_SIZE:
            adjacent.append(cell)
    # At every grid position there are at least two distinct valid neighbors.
    pair = max(itertools.combinations(adjacent, 2), key=lambda xy: _manhattan(*xy))
    return [target, pair[0], pair[1]]


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move_toward(position: Pos, target: Pos, rng: np.random.Generator) -> int:
    distances = []
    for action, (dr, dc) in enumerate(config.ACTIONS):
        candidate = (position[0] + dr, position[1] + dc)
        if not (0 <= candidate[0] < config.GRID_SIZE and 0 <= candidate[1] < config.GRID_SIZE):
            candidate = position
        distances.append(_manhattan(candidate, target))
    best = np.flatnonzero(np.asarray(distances) == min(distances))
    return int(rng.choice(best))


def coordinated_pursuit_actions(
    predator_positions: Sequence[Pos],
    prey_position: Pos,
    prev_prey_position: Optional[Pos] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[int]:
    """Assign distinct surround targets to three predators, then move greedily."""
    if len(predator_positions) != config.NUM_PREDATORS:
        raise ValueError(f"expected {config.NUM_PREDATORS} predator positions")
    rng = rng or np.random.default_rng(0)
    targets = _targets(prey_position, prev_prey_position)
    assignment = min(
        itertools.permutations(range(config.NUM_PREDATORS)),
        key=lambda order: sum(_manhattan(predator_positions[i], targets[order[i]]) for i in range(config.NUM_PREDATORS)),
    )
    return [
        _move_toward(predator_positions[i], targets[assignment[i]], rng)
        for i in range(config.NUM_PREDATORS)
    ]


class CoordinatedPursuitPolicy:
    """Evaluator-compatible wrapper with reproducible tie breaking."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(self._seed if seed is None else seed)

    def __call__(self, env_obs: dict) -> list[int]:
        return coordinated_pursuit_actions(
            env_obs["predators"], env_obs["prey"], env_obs.get("prev_prey"), self.rng
        )
