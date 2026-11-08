"""
rl/observation_no_teammate.py  (Task T-12A, ablation A)
=======================================================
Same as rl/observation.py but WITHOUT the two teammate features (f3, f4).

State space: 8 (bearing) x 3 (range) x 5 (prev prey move) x 9 (wall) = 1,080.

    obs = f1 + 8*f2 + 8*3*f5 + 8*3*5*f6

Used by `python train.py --obs no_teammate` to test whether knowing where the
other predators are actually helps.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from rl.observation import (  # noqa: E402
    bearing_bin,
    prev_prey_move_bin,
    prey_range_bin,
    wall_context_bin,
)

Pos = Tuple[int, int]

NUM_OBSERVATIONS = (
    config.OBS_BEARING_BINS
    * config.OBS_RANGE_BINS
    * config.OBS_PREV_PREY_BINS
    * config.OBS_WALL_BINS
)  # = 1,080

_M1 = 1
_M2 = _M1 * config.OBS_BEARING_BINS
_M5 = _M2 * config.OBS_RANGE_BINS
_M6 = _M5 * config.OBS_PREV_PREY_BINS


def encode_observation(
    predator_pos: Pos,
    prey_pos: Pos,
    teammate_positions: Optional[Sequence[Pos]],
    prev_prey_pos: Optional[Pos],
    grid_size: int = config.GRID_SIZE,
) -> int:
    """Encode one predator's view as an int in [0, 1080). `teammate_positions` is ignored."""
    return (
        bearing_bin(predator_pos, prey_pos) * _M1
        + prey_range_bin(predator_pos, prey_pos) * _M2
        + prev_prey_move_bin(prey_pos, prev_prey_pos) * _M5
        + wall_context_bin(predator_pos, grid_size) * _M6
    )


def encode_team_observations(env_obs: dict, grid_size: int = config.GRID_SIZE) -> list:
    """Encode all predators from an env observation dict (same interface as rl.observation)."""
    prey = env_obs["prey"]
    prev = env_obs["prev_prey"]
    return [encode_observation(p, prey, None, prev, grid_size) for p in env_obs["predators"]]
