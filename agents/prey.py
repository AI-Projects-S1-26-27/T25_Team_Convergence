"""
agents/prey.py
==============
Prey (evader) policy for the 20×20 predator-prey simulation.

The prey uses a **greedy-flee** strategy: it picks the legal move that
maximises its Manhattan distance from the nearest predator. With probability
``random_prob`` it overrides this and picks a uniformly-random legal action
instead (enabling stochastic diversity during training and robustness tests).

Two random-draw modes are supported so the same policy can be called both
during live training (pass ``rng``) and during deterministic matched-scenario
replay (pass ``pre_sampled_rand`` / ``pre_sampled_choice``):

* **Live mode** — supply ``rng`` (a ``np.random.Generator``); the function
  draws internally for both the random-trigger roll and the tie/random-move
  selection.
* **Pre-sampled mode** — supply ``pre_sampled_rand`` (float in [0, 1)) for
  the trigger roll and optionally ``pre_sampled_choice`` (int index) for the
  move selection; no internal draws are made.  Used by Task T-11 (matched
  test scenarios) and Task T-16 (robustness tests).

Dependencies: Python standard library, numpy, config.py (repo root).

Team Convergence (Team 25) — Predator-Prey Dynamics.
Owner: Group A — Lohit & Ansh.
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

import config  # noqa: E402


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_legal_moves(
    prey_pos: Tuple[int, int],
    grid_size: int = config.GRID_SIZE,
) -> List[int]:
    """
    Return the list of action integers whose resulting cell is within the grid.

    Parameters
    ----------
    prey_pos : (row, col)
        Current prey position.  Must satisfy ``0 <= row, col < grid_size``.
    grid_size : int
        Side length of the square grid (default ``config.GRID_SIZE = 20``).

    Returns
    -------
    List[int]
        Subset of ``{0, 1, 2, 3, 4}`` that keep the prey inside the grid.
        Action ``0`` (Stay) is always included, so the list is never empty.

    Notes
    -----
    Action encoding (mirrors ``config.ACTIONS``):

    ===  =====  ====  ====
    Int  Name   Δrow  Δcol
    ===  =====  ====  ====
    0    Stay     0     0
    1    North   −1     0
    2    East     0    +1
    3    South   +1     0
    4    West     0    −1
    ===  =====  ====  ====
    """
    r, c = prey_pos
    legal: List[int] = []
    for action_idx, (dr, dc) in enumerate(config.ACTIONS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            legal.append(action_idx)
    return legal


# ---------------------------------------------------------------------------
# Public policy function
# ---------------------------------------------------------------------------

def prey_action(
    prey_pos: Tuple[int, int],
    predator_positions: Sequence[Tuple[int, int]],
    legal_moves: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
    random_prob: float = config.PREY_RANDOM_PROB,
    pre_sampled_rand: Optional[float] = None,
    pre_sampled_choice: Optional[int] = None,
) -> int:
    """
    Choose the prey's next action using a greedy-flee policy with ε-random
    exploration.

    Decision logic
    --------------
    1. Compute ``legal_moves`` if not supplied.
    2. Determine whether to act randomly:

       * If ``pre_sampled_rand`` is given → random iff
         ``pre_sampled_rand < random_prob``.
       * Otherwise → draw ``u ~ Uniform[0,1)`` from ``rng`` (or a fresh
         ``default_rng()`` as a fallback) and random iff ``u < random_prob``.

    3. **Random branch** — return a uniformly-chosen action from
       ``legal_moves``:

       * If ``pre_sampled_choice`` is given, treat it as the index into
         ``legal_moves`` (modulo its length for safety).
       * Otherwise sample with ``rng``.

    4. **Greedy-flee branch** — for each legal move compute the resulting
       position and its minimum Manhattan distance to any predator.  Return
       the move with the highest such distance.  Ties broken uniformly at
       random using the same ``rng``.

    Parameters
    ----------
    prey_pos : (row, col)
        Current prey position.
    predator_positions : sequence of (row, col)
        Current positions of all predators (length 3 in the standard setup).
    legal_moves : sequence of int or None
        Pre-computed legal actions.  If ``None``, computed via
        ``get_legal_moves(prey_pos)``.
    rng : np.random.Generator or None
        NumPy random generator for live-mode draws.  If both ``rng`` and
        ``pre_sampled_rand`` are ``None``, a fresh ``default_rng()`` is
        created (non-deterministic).
    random_prob : float
        Probability of choosing a uniformly random legal action (default
        ``config.PREY_RANDOM_PROB = 0.1``).
    pre_sampled_rand : float or None
        Pre-drawn uniform value in ``[0, 1)`` for the random-trigger roll.
        When supplied, ``rng`` is *not* used for this roll.
    pre_sampled_choice : int or None
        Pre-drawn integer index used in the random branch to index into
        ``legal_moves``.  Ignored if the greedy branch is taken.

    Returns
    -------
    int
        Action integer in ``{0, 1, 2, 3, 4}``.

    Raises
    ------
    ValueError
        If ``predator_positions`` is empty.
    """
    if not predator_positions:
        raise ValueError("predator_positions must not be empty.")

    # --- Legal moves --------------------------------------------------------
    if legal_moves is None:
        legal_moves = get_legal_moves(prey_pos)
    legal_moves = list(legal_moves)  # defensive copy / ensure list

    # Fast-path: only one legal move (e.g., trapped in corner)
    if len(legal_moves) == 1:
        return legal_moves[0]

    # --- Fallback RNG (non-deterministic) -----------------------------------
    _rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    # --- Random-trigger roll ------------------------------------------------
    if pre_sampled_rand is not None:
        take_random = pre_sampled_rand < random_prob
    else:
        take_random = bool(_rng.random() < random_prob)

    # --- Random branch ------------------------------------------------------
    if take_random:
        if pre_sampled_choice is not None:
            idx = int(pre_sampled_choice) % len(legal_moves)
            return legal_moves[idx]
        return int(_rng.choice(legal_moves))

    # --- Greedy-flee branch -------------------------------------------------
    best_actions: List[int] = []
    best_dist: float = -1.0

    for action in legal_moves:
        dr, dc = config.ACTIONS[action]
        nr, nc = prey_pos[0] + dr, prey_pos[1] + dc

        # Minimum Manhattan distance to any predator from the candidate cell
        min_dist = min(
            abs(nr - pr) + abs(nc - pc)
            for pr, pc in predator_positions
        )

        if min_dist > best_dist:
            best_dist = min_dist
            best_actions = [action]
        elif min_dist == best_dist:
            best_actions.append(action)

    # Tie-break uniformly at random
    if len(best_actions) == 1:
        return best_actions[0]
    return int(_rng.choice(best_actions))

