"""
env/environment.py
==================
20×20 discrete grid environment for the 3-predator / 1-prey RL simulation.

Team Convergence (Team 25) — Predator-Prey Dynamics.
Owner: Group A (Environment & Visualisation) — Lohit & Ansh.

Dependencies: Python standard library, numpy, config.py (repo root).
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Ensure the repository root is on the import path so config is always found,
# regardless of the working directory from which tests are run.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402  (after path surgery above)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Cell = Tuple[int, int]          # (row, col)
ObsDict = Dict                  # observation dictionary returned by reset/step
InfoDict = Dict                 # diagnostic dictionary returned by step


class Environment:
    """
    Discrete 20×20 grid predator-prey environment.

    Three predators cooperate to capture one prey agent.  All agents move
    simultaneously each timestep.  The environment is fully observable at the
    raw state level (positions are exposed in ``obs``); observation *binning*
    for the Q-table is handled by Group C.

    Parameters
    ----------
    seed : int or None
        Seed for the internal NumPy ``Generator``.  Pass ``None`` for
        non-deterministic behaviour.
    max_steps : int
        Maximum number of steps before a timeout is declared.  Defaults to
        ``config.MAX_STEPS`` (300).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = config.MAX_STEPS,
    ) -> None:
        self.max_steps: int = max_steps
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Episode state — initialised properly by reset()
        self.predators: List[Cell] = []
        self.prey: Cell = (0, 0)
        self.prev_prey: Optional[Cell] = None
        self.step_count: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        positions: Optional[Union[Sequence[Cell], Dict]] = None,
        seed: Optional[int] = None,
    ) -> ObsDict:
        """
        Begin a new episode and return the initial observation.

        Parameters
        ----------
        positions : None, sequence of 4 (row, col) tuples, or dict
            * ``None`` — sample 4 distinct cells uniformly from the 400-cell grid.
            * Sequence ``[pred0, pred1, pred2, prey]`` — explicit placement.
            * Dict ``{'predators': [(r1,c1),(r2,c2),(r3,c3)], 'prey': (r,c)}``.
        seed : int or None
            If provided, re-seeds the internal RNG before sampling.

        Returns
        -------
        dict
            ``{'predators': [...], 'prey': (r,c), 'prev_prey': None, 'step': 0}``
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if positions is None:
            self.predators, self.prey = self._sample_positions()
        else:
            self.predators, self.prey = self._parse_positions(positions)

        self.prev_prey = None
        self.step_count = 0
        self.done = False

        return self._get_obs()

    def step(
        self,
        predator_actions: Sequence[int],
        prey_action: int,
    ) -> Tuple[ObsDict, bool, InfoDict]:
        """
        Advance the simulation by one timestep.

        All agents move **simultaneously**.  Out-of-bounds moves are silently
        clipped (the agent stays in place).  Predators may share cells.

        Parameters
        ----------
        predator_actions : sequence of 3 ints in {0,1,2,3,4}
            One action per predator.
        prey_action : int in {0,1,2,3,4}
            Action for the prey agent.

        Returns
        -------
        obs : dict
        done : bool
        info : dict
            ``{'capture': bool, 'capture_step': int|None, 'timeout': bool,
               'step_count': int}``

        Raises
        ------
        RuntimeError
            If called on a finished episode (``self.done is True``).
        ValueError
            If action integers are out of range or ``predator_actions`` has
            wrong length.
        """
        # --- Pre-condition checks ----------------------------------------
        if self.done:
            raise RuntimeError(
                "step() called on a finished episode.  Call reset() first."
            )
        self._validate_actions(predator_actions, prey_action)

        # --- Record pre-move state ----------------------------------------
        old_predators: List[Cell] = list(self.predators)
        old_prey: Cell = self.prey

        # --- Compute new positions ----------------------------------------
        new_predators: List[Cell] = [
            self._apply_action(old_predators[i], predator_actions[i])
            for i in range(config.NUM_PREDATORS)
        ]
        new_prey: Cell = self._apply_action(old_prey, prey_action)

        # --- Commit state -------------------------------------------------
        self.prev_prey = old_prey
        self.predators = new_predators
        self.prey = new_prey
        self.step_count += 1

        # --- Capture detection -------------------------------------------
        captured = self._check_capture(old_predators, old_prey, new_predators, new_prey)

        # --- Terminal conditions -----------------------------------------
        if captured:
            self.done = True
            info: InfoDict = {
                "capture":      True,
                "capture_step": self.step_count,
                "timeout":      False,
                "step_count":   self.step_count,
            }
        elif self.step_count >= self.max_steps:
            self.done = True
            info = {
                "capture":      False,
                "capture_step": None,
                "timeout":      True,
                "step_count":   self.step_count,
            }
        else:
            self.done = False
            info = {
                "capture":      False,
                "capture_step": None,
                "timeout":      False,
                "step_count":   self.step_count,
            }

        return self._get_obs(), self.done, info

    def render(self, ax=None):
        """
        Render the current grid state using Matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to draw on.  If ``None``, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the rendered frame.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        gs = config.GRID_SIZE
        ax.set_xlim(0, gs)
        ax.set_ylim(0, gs)
        ax.set_aspect("equal")
        ax.set_xticks(range(gs + 1))
        ax.set_yticks(range(gs + 1))
        ax.grid(True, linewidth=0.4, color="lightgrey")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Helper: grid (row, col) → matplotlib (x, y) with row 0 at top
        def _to_xy(r: int, c: int) -> Tuple[float, float]:
            return c + 0.5, (gs - 1 - r) + 0.5

        for i, (r, c) in enumerate(self.predators):
            x, y = _to_xy(r, c)
            ax.add_patch(plt.Circle((x, y), 0.35, color="crimson", zorder=3))
            ax.text(x, y, f"P{i}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=4)

        pr, pc = self.prey
        x, y = _to_xy(pr, pc)
        ax.add_patch(plt.Circle((x, y), 0.35, color="royalblue", zorder=3))
        ax.text(x, y, "Q", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=4)

        ax.set_title(
            f"Step {self.step_count} | Done={self.done}",
            fontsize=10,
        )
        return ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> ObsDict:
        """Return the current observation dictionary."""
        return {
            "predators": list(self.predators),
            "prey":      self.prey,
            "prev_prey": self.prev_prey,
            "step":      self.step_count,
        }

    def _sample_positions(self) -> Tuple[List[Cell], Cell]:
        """Sample 4 distinct cells uniformly without replacement."""
        indices = self.rng.choice(
            config.GRID_SIZE * config.GRID_SIZE, size=4, replace=False
        )
        cells = [(int(idx // config.GRID_SIZE), int(idx % config.GRID_SIZE))
                 for idx in indices]
        predators = cells[:3]
        prey = cells[3]
        return predators, prey

    @staticmethod
    def _parse_positions(
        positions: Union[Sequence[Cell], Dict],
    ) -> Tuple[List[Cell], Cell]:
        """
        Parse and validate explicit position input.

        Accepts either:
        * A sequence of exactly 4 ``(row, col)`` tuples:
          ``[pred0, pred1, pred2, prey]``
        * A dict: ``{'predators': [p0, p1, p2], 'prey': (r, c)}``

        Returns
        -------
        (predators, prey) where predators is a list of 3 Cell tuples.
        """
        if isinstance(positions, dict):
            pred_list = [tuple(p) for p in positions["predators"]]
            prey_cell = tuple(positions["prey"])
            all_cells = pred_list + [prey_cell]
        else:
            all_cells = [tuple(p) for p in positions]
            if len(all_cells) != 4:
                raise ValueError(
                    f"positions must contain exactly 4 cells; got {len(all_cells)}."
                )
            pred_list = all_cells[:3]
            prey_cell = all_cells[3]

        gs = config.GRID_SIZE
        for cell in all_cells:
            r, c = cell
            if not (0 <= r < gs and 0 <= c < gs):
                raise ValueError(
                    f"Cell {cell} is out of bounds for a {gs}×{gs} grid."
                )

        if len(set(all_cells)) != len(all_cells):
            raise ValueError(
                f"Duplicate cells detected in positions: {all_cells}."
            )

        return [tuple(p) for p in pred_list], tuple(prey_cell)

    @staticmethod
    def _apply_action(cell: Cell, action: int) -> Cell:
        """
        Apply an action to a cell, clipping to grid boundaries.

        Out-of-bounds candidate positions are silently clipped so the agent
        stays at its current cell.
        """
        dr, dc = config.ACTIONS[action]
        new_r = cell[0] + dr
        new_c = cell[1] + dc
        gs = config.GRID_SIZE
        if 0 <= new_r < gs and 0 <= new_c < gs:
            return (new_r, new_c)
        return cell  # clip: stay in place

    @staticmethod
    def _check_capture(
        old_predators: List[Cell],
        old_prey: Cell,
        new_predators: List[Cell],
        new_prey: Cell,
    ) -> bool:
        """
        Return ``True`` if any capture condition is satisfied.

        Two capture mechanisms:
        1. **Same-cell**: ``new_predators[i] == new_prey``
        2. **Edge-swap**: ``old_predators[i] == new_prey AND
                           new_predators[i] == old_prey``
        """
        for i in range(config.NUM_PREDATORS):
            # Same-cell capture
            if new_predators[i] == new_prey:
                return True
            # Edge-swap capture
            if old_predators[i] == new_prey and new_predators[i] == old_prey:
                return True
        return False

    @staticmethod
    def _validate_actions(
        predator_actions: Sequence[int],
        prey_action: int,
    ) -> None:
        """Raise ValueError for invalid action inputs."""
        if len(predator_actions) != config.NUM_PREDATORS:
            raise ValueError(
                f"predator_actions must have length {config.NUM_PREDATORS}; "
                f"got {len(predator_actions)}."
            )
        valid = set(range(config.NUM_ACTIONS))
        for i, a in enumerate(predator_actions):
            if a not in valid:
                raise ValueError(
                    f"predator_actions[{i}] = {a} is not a valid action "
                    f"(must be in {{0,1,2,3,4}})."
                )
        if prey_action not in valid:
            raise ValueError(
                f"prey_action = {prey_action} is not a valid action "
                f"(must be in {{0,1,2,3,4}})."
            )

