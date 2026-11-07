"""
rl/observation.py  (Task T-07)
==============================
Maps one predator's local view of the world to a single integer that indexes
its Q-table.

State space (matches the project proposal and config.NUM_OBSERVATIONS = 16,200):

    f1  prey bearing              8 bins  N, NE, E, SE, S, SW, W, NW
    f2  prey range                3 bins  Manhattan dist 0-2 | 3-6 | >=7
    f3  nearest-teammate quadrant 5 bins  N, E, S, W, overlap
    f4  nearest-teammate range    3 bins  0-2 | 3-6 | >=7
    f5  previous prey move        5 bins  Stay, N, E, S, W
    f6  wall context              9 bins  3 row zones x 3 col zones
                                          (within 2 cells of an edge, or interior)

Mixed-radix combination (f1 is the least-significant digit):

    obs = f1 + 8*f2 + 8*3*f3 + 8*3*5*f4 + 8*3*5*3*f5 + 8*3*5*3*5*f6
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402

Pos = Tuple[int, int]

# Range bins: 0-2 -> 0, 3-6 -> 1, >=7 -> 2   (project proposal, Section 4)
RANGE_NEAR_MAX = 2
RANGE_MID_MAX = 6

# Wall zones: within WALL_MARGIN cells of an edge counts as "near" that edge.
WALL_MARGIN = 2

# Bearing bin for each (sign(dy), sign(dx)).
#   N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
_BEARING_TABLE = {
    (-1, 0): 0,
    (-1, 1): 1,
    (0, 1): 2,
    (1, 1): 3,
    (1, 0): 4,
    (1, -1): 5,
    (0, -1): 6,
    (-1, -1): 7,
    (0, 0): 0,  # same cell -> default to N (episode ends on capture anyway)
}

# Previous prey move bin: mirrors the action encoding (Stay, N, E, S, W).
_PREV_MOVE_TABLE = {
    (0, 0): 0,
    (-1, 0): 1,
    (0, 1): 2,
    (1, 0): 3,
    (0, -1): 4,
}

# Radix multipliers, derived from config so a bin-count change cannot drift.
_M1 = 1
_M2 = _M1 * config.OBS_BEARING_BINS
_M3 = _M2 * config.OBS_RANGE_BINS
_M4 = _M3 * config.OBS_TEAMMATE_QUAD_BINS
_M5 = _M4 * config.OBS_TEAMMATE_RANGE_BINS
_M6 = _M5 * config.OBS_PREV_PREY_BINS


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def range_bin(dist: int) -> int:
    """Bin a Manhattan distance: 0-2 -> 0, 3-6 -> 1, >=7 -> 2."""
    if dist <= RANGE_NEAR_MAX:
        return 0
    if dist <= RANGE_MID_MAX:
        return 1
    return 2


def bearing_bin(predator_pos: Pos, prey_pos: Pos) -> int:
    """Feature 1: compass direction (8 bins) from predator to prey."""
    dy = prey_pos[0] - predator_pos[0]
    dx = prey_pos[1] - predator_pos[1]
    return _BEARING_TABLE[(_sign(dy), _sign(dx))]


def prey_range_bin(predator_pos: Pos, prey_pos: Pos) -> int:
    """Feature 2: Manhattan distance to prey (3 bins)."""
    return range_bin(_manhattan(predator_pos, prey_pos))


def nearest_teammate(predator_pos: Pos, teammate_positions: Sequence[Pos]) -> Pos:
    """Closest teammate by Manhattan distance; ties go to the earlier listed one."""
    if len(teammate_positions) == 0:
        raise ValueError("teammate_positions must contain at least one position")
    best = teammate_positions[0]
    best_d = _manhattan(predator_pos, best)
    for t in teammate_positions[1:]:
        d = _manhattan(predator_pos, t)
        if d < best_d:
            best, best_d = t, d
    return best


def teammate_quadrant_bin(predator_pos: Pos, teammate_pos: Pos) -> int:
    """
    Feature 3: which side the nearest teammate is on (5 bins).

    0 = N, 1 = E, 2 = S, 3 = W, 4 = overlap.
    Overlap is checked first: it covers the same cell and exact diagonals
    (|dy| == |dx| != 0), where no single cardinal side dominates.
    """
    dy = teammate_pos[0] - predator_pos[0]
    dx = teammate_pos[1] - predator_pos[1]
    if abs(dy) == abs(dx):  # includes dy == dx == 0
        return 4
    if abs(dy) > abs(dx):
        return 0 if dy < 0 else 2
    return 1 if dx > 0 else 3


def teammate_range_bin(predator_pos: Pos, teammate_pos: Pos) -> int:
    """Feature 4: Manhattan distance to the nearest teammate (3 bins)."""
    return range_bin(_manhattan(predator_pos, teammate_pos))


def prev_prey_move_bin(prey_pos: Pos, prev_prey_pos: Optional[Pos]) -> int:
    """Feature 5: direction the prey moved last step (5 bins). None -> Stay."""
    if prev_prey_pos is None:
        return 0
    delta = (prey_pos[0] - prev_prey_pos[0], prey_pos[1] - prev_prey_pos[1])
    try:
        return _PREV_MOVE_TABLE[delta]
    except KeyError:
        raise ValueError(
            f"prey moved {delta} in one step; expected one of Stay/N/E/S/W"
        ) from None


def _zone(coord: int, grid_size: int) -> int:
    if coord < WALL_MARGIN:
        return 0
    if coord >= grid_size - WALL_MARGIN:
        return 2
    return 1


def wall_context_bin(predator_pos: Pos, grid_size: int = config.GRID_SIZE) -> int:
    """
    Feature 6: where the predator sits relative to the walls (9 bins).

    row zone (0 = near top, 1 = interior, 2 = near bottom) * 3
    + col zone (0 = near left, 1 = interior, 2 = near right).
    """
    return _zone(predator_pos[0], grid_size) * 3 + _zone(predator_pos[1], grid_size)


def encode_features(
    predator_pos: Pos,
    prey_pos: Pos,
    teammate_positions: Sequence[Pos],
    prev_prey_pos: Optional[Pos],
    grid_size: int = config.GRID_SIZE,
) -> Tuple[int, int, int, int, int, int]:
    """Return the six feature bins (f1..f6) for one predator."""
    mate = nearest_teammate(predator_pos, teammate_positions)
    return (
        bearing_bin(predator_pos, prey_pos),
        prey_range_bin(predator_pos, prey_pos),
        teammate_quadrant_bin(predator_pos, mate),
        teammate_range_bin(predator_pos, mate),
        prev_prey_move_bin(prey_pos, prev_prey_pos),
        wall_context_bin(predator_pos, grid_size),
    )


def combine_features(features: Sequence[int]) -> int:
    """Mixed-radix combination of (f1..f6) into a single Q-table row index."""
    f1, f2, f3, f4, f5, f6 = features
    return f1 * _M1 + f2 * _M2 + f3 * _M3 + f4 * _M4 + f5 * _M5 + f6 * _M6


def decode_observation(obs: int) -> Tuple[int, int, int, int, int, int]:
    """Inverse of combine_features: split an index back into (f1..f6)."""
    if not 0 <= obs < config.NUM_OBSERVATIONS:
        raise ValueError(f"obs index {obs} outside [0, {config.NUM_OBSERVATIONS})")
    f6, rest = divmod(obs, _M6)
    f5, rest = divmod(rest, _M5)
    f4, rest = divmod(rest, _M4)
    f3, rest = divmod(rest, _M3)
    f2, f1 = divmod(rest, _M2)
    return (f1, f2, f3, f4, f5, f6)


def encode_observation(
    predator_pos: Pos,
    prey_pos: Pos,
    teammate_positions: Sequence[Pos],
    prev_prey_pos: Optional[Pos],
    grid_size: int = config.GRID_SIZE,
) -> int:
    """Encode one predator's view as an int in [0, config.NUM_OBSERVATIONS)."""
    return combine_features(
        encode_features(predator_pos, prey_pos, teammate_positions, prev_prey_pos, grid_size)
    )


def encode_team_observations(env_obs: dict, grid_size: int = config.GRID_SIZE) -> list:
    """
    Convenience for the training loop: encode all predators from the dict that
    env.reset() / env.step() return (keys 'predators', 'prey', 'prev_prey').

    Returns a list of 3 ints, one per predator, in predator order.
    """
    preds = env_obs["predators"]
    prey = env_obs["prey"]
    prev = env_obs["prev_prey"]
    return [
        encode_observation(
            preds[i], prey, [p for j, p in enumerate(preds) if j != i], prev, grid_size
        )
        for i in range(len(preds))
    ]
