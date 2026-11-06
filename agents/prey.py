"""
agents/prey.py
==============
Prey (evader) policy for the 20x20 predator-prey simulation.

The prey uses a greedy-flee strategy: it evaluates legal moves and selects the
action that maximizes the minimum Manhattan distance to any predator.
With probability PREY_RANDOM_PROB (0.1) it takes a stochastic exploration move,
picking uniformly at random from legal moves.

Team Convergence (Team 25) - Predator-Prey Dynamics.
Owner: Group A (Environment & Visualisation) - Ansh & Lohit.
"""

from __future__ import annotations

import sys
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Ensure the repository root is importable regardless of working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from config import GRID_SIZE, ACTIONS, PREY_RANDOM_PROB


# ---------------------------------------------------------------------------
# Distance Metric
# ---------------------------------------------------------------------------

def manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    Compute and return the Manhattan distance between two (row, col) positions.

    Parameters
    ----------
    p1 : (row, col)
    p2 : (row, col)

    Returns
    -------
    int
        |p1[0] - p2[0]| + |p1[1] - p2[1]|
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# ---------------------------------------------------------------------------
# Legal Moves Calculation
# ---------------------------------------------------------------------------

def get_legal_moves(
    prey_pos: Tuple[int, int],
    grid_size: int = GRID_SIZE,
) -> List[int]:
    """
    Compute all actions from config.ACTIONS that keep the prey within [0, grid_size - 1].

    Parameters
    ----------
    prey_pos : (row, col)
        Current prey position.
    grid_size : int
        Size of the square grid (default GRID_SIZE = 20).

    Returns
    -------
    List[int]
        Subset of action indices {0, 1, 2, 3, 4} that remain in bounds.
        Action 0 (Stay) is always legal, so the returned list is never empty.
    """
    r, c = prey_pos
    legal: List[int] = []
    for action_idx, (dr, dc) in enumerate(ACTIONS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            legal.append(action_idx)
    return legal


# ---------------------------------------------------------------------------
# Prey Policy
# ---------------------------------------------------------------------------

def prey_action(
    prey_pos: Tuple[int, int],
    predator_positions: Sequence[Tuple[int, int]],
    legal_moves: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
    random_prob: float = PREY_RANDOM_PROB,
    pre_sampled_rand: Optional[float] = None,
    pre_sampled_choice: Optional[int] = None,
) -> int:
    """
    Select an action for the prey evader agent.

    Parameters
    ----------
    prey_pos : (row, col)
        Current position of the prey.
    predator_positions : sequence of (row, col)
        Current positions of all predators.
    legal_moves : sequence of int or None
        Pre-computed legal actions. If None, computed using get_legal_moves(prey_pos).
    rng : np.random.Generator or None
        NumPy random generator. If None, np.random.default_rng() is used.
    random_prob : float
        Probability of selecting a random legal action (defaults to PREY_RANDOM_PROB = 0.1).
    pre_sampled_rand : float or None
        Optional pre-drawn uniform random number in [0, 1) for the stochastic trigger roll.
    pre_sampled_choice : int or None
        Optional pre-drawn index used in the random branch or tie-break for deterministic replay.

    Returns
    -------
    int
        Selected action index in {0, 1, 2, 3, 4}.

    Raises
    ------
    ValueError
        If predator_positions is empty.
    """
    if not predator_positions:
        raise ValueError("predator_positions must not be empty.")

    if rng is None:
        rng = np.random.default_rng()

    if legal_moves is None:
        legal_moves = get_legal_moves(prey_pos)
    legal_moves = list(legal_moves)

    # Edge Case: If legal_moves is empty or only [0], return 0 (Stay)
    if not legal_moves or legal_moves == [0]:
        return 0

    # Stochastic Move (Exploration / Panic):
    if pre_sampled_rand is not None:
        take_random = pre_sampled_rand < random_prob
    else:
        take_random = bool(rng.random() < random_prob)

    if take_random:
        if pre_sampled_choice is not None:
            idx = int(pre_sampled_choice) % len(legal_moves)
            return int(legal_moves[idx])
        return int(rng.choice(legal_moves))

    # Deterministic Move (Maximizing Minimum Safety Distance):
    # An evader flees from predators, considering all moving candidate moves.
    candidates = [a for a in legal_moves if a != 0]
    if not candidates:
        return 0

    best_actions: List[int] = []
    best_min_dist: int = -1
    best_sum_dist: int = -1

    for action in candidates:
        dr, dc = ACTIONS[action]
        cand_pos = (prey_pos[0] + dr, prey_pos[1] + dc)
        dists = [manhattan_dist(cand_pos, pred_pos) for pred_pos in predator_positions]
        m_dist = min(dists)
        s_dist = sum(dists)

        if m_dist > best_min_dist:
            best_min_dist = m_dist
            best_sum_dist = s_dist
            best_actions = [action]
        elif m_dist == best_min_dist:
            # Secondary tie-break: prefer moves with higher total distance from predators
            if s_dist > best_sum_dist:
                best_sum_dist = s_dist
                best_actions = [action]
            elif s_dist == best_sum_dist:
                best_actions.append(action)

    if len(best_actions) == 1:
        return int(best_actions[0])

    # Tie-breaking among tied moves
    if pre_sampled_choice is not None:
        return int(best_actions[int(pre_sampled_choice) % len(best_actions)])
    return int(rng.choice(best_actions))
