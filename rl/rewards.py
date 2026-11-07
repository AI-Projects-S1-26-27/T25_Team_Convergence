"""
rl/rewards.py  (Task T-09)
==========================
Reward functions for the shared team reward.

    base_reward   : +10 on capture, -0.01 otherwise.
    shaped_reward : base_reward + gamma * Phi(s') - Phi(s)   (potential-based)

Phi(s) = -(mean Manhattan distance from the 3 predators to the prey) / 38,
so Phi lies in [-1, 0] and rises toward 0 as the predators close in.
Phi(s') is taken as 0 at a terminal state (capture OR timeout).

No term rewards flanking, encirclement, roles or interception.
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


def base_reward(capture: bool, step_count: int = 0) -> float:
    """+10.0 on capture, otherwise the -0.01 per-step penalty.

    `step_count` is accepted for interface compatibility and is unused.
    """
    return config.REWARD_CAPTURE if capture else config.REWARD_STEP


def potential(predator_positions: Sequence[Pos], prey_pos: Pos) -> float:
    """Phi(s) = -mean Manhattan distance(predators, prey) / 38, in [-1, 0]."""
    total = sum(
        abs(p[0] - prey_pos[0]) + abs(p[1] - prey_pos[1]) for p in predator_positions
    )
    return -(total / len(predator_positions)) / config.MAX_MANHATTAN_DIST


def shaped_reward(
    predator_positions: Sequence[Pos],
    prey_pos: Pos,
    capture: bool,
    next_predator_positions: Sequence[Pos],
    next_prey_pos: Pos,
    gamma: float = config.GAMMA,
    done: Optional[bool] = None,
    step_count: int = 0,
) -> float:
    """
    base_reward + gamma * Phi(s') - Phi(s).

    `done` marks a terminal s'. It defaults to `capture`; the training loop
    should pass the env's `done` so a timeout also gets Phi(s') = 0.
    """
    if done is None:
        done = capture
    phi_now = potential(predator_positions, prey_pos)
    phi_next = 0.0 if (done or capture) else potential(next_predator_positions, next_prey_pos)
    return base_reward(capture, step_count) + gamma * phi_next - phi_now


if __name__ == "__main__":
    # Sanity check: closing in gives a positive shaping bonus.
    prey = (10, 10)
    far = [(0, 0), (0, 19), (19, 0)]
    near = [(1, 0), (1, 19), (18, 0)]  # each predator one step closer
    print(f"Phi(far)  = {potential(far, prey):+.4f}")
    print(f"Phi(near) = {potential(near, prey):+.4f}  (should be larger / less negative)")
    r_base = base_reward(False)
    r_shaped = shaped_reward(far, prey, False, near, prey)
    print(f"base = {r_base:+.4f}, shaped = {r_shaped:+.4f}, bonus = {r_shaped - r_base:+.4f}")
    print(f"capture: base = {base_reward(True):+.2f}, "
          f"shaped = {shaped_reward(near, prey, True, near, prey):+.4f}")
