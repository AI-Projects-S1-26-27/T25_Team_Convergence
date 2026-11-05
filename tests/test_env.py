"""
tests/test_env.py
=================
Comprehensive pytest suite for env/environment.py.

Covers T-02 requirements:
  1.  reset() — 4 distinct in-bounds positions, deterministic with same seed
  2.  reset(positions=...) — exact placement (sequence & dict forms)
  3.  Out-of-bounds moves at all 4 walls clip to Stay (no error)
  4.  Predators can share cells simultaneously
  5.  Action encoding correctness (Stay, N, E, S, W)
  6.  Same-cell capture → done=True, capture=True, correct capture_step
  7.  Edge-swap capture → done=True, capture=True (all 3 predator indices)
  8.  Timeout at step 300 → done=True, capture=False, capture_step=None
"""

from __future__ import annotations

import sys
import os

import pytest

# ---------------------------------------------------------------------------
# Path setup — make both the repo root and tests/ importable from anywhere
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from env.environment import Environment  # noqa: E402
import config                            # noqa: E402


# ===========================================================================
# Helper
# ===========================================================================

def _make_env(**kwargs) -> Environment:
    """Return a fresh Environment with optional kwargs."""
    return Environment(**kwargs)


# ===========================================================================
# 1. reset() — distinct in-bounds positions, determinism
# ===========================================================================

class TestReset:
    def test_reset_returns_four_distinct_in_bounds_cells(self):
        env = _make_env(seed=0)
        obs = env.reset()

        all_cells = obs["predators"] + [obs["prey"]]
        assert len(all_cells) == 4, "Should have 3 predators + 1 prey"

        for cell in all_cells:
            r, c = cell
            assert 0 <= r < config.GRID_SIZE, f"Row {r} out of bounds"
            assert 0 <= c < config.GRID_SIZE, f"Col {c} out of bounds"

        assert len(set(all_cells)) == 4, "All 4 starting cells must be distinct"

    def test_reset_deterministic_same_seed(self):
        obs_a = _make_env(seed=42).reset()
        obs_b = _make_env(seed=42).reset()
        assert obs_a["predators"] == obs_b["predators"]
        assert obs_a["prey"] == obs_b["prey"]

    def test_reset_different_seeds_give_different_positions(self):
        obs_a = _make_env(seed=1).reset()
        obs_b = _make_env(seed=2).reset()
        # It is astronomically unlikely (but technically possible) for two
        # distinct seeds to give identical layouts; treat as a soft warning.
        same = (obs_a["predators"] == obs_b["predators"] and
                obs_a["prey"] == obs_b["prey"])
        assert not same, "Different seeds should produce different positions"

    def test_reset_step_is_zero(self):
        obs = _make_env(seed=5).reset()
        assert obs["step"] == 0

    def test_reset_prev_prey_is_none(self):
        obs = _make_env(seed=5).reset()
        assert obs["prev_prey"] is None

    def test_reset_seed_kwarg_re_seeds_rng(self):
        env = _make_env()
        obs_a = env.reset(seed=99)
        obs_b = env.reset(seed=99)
        assert obs_a["predators"] == obs_b["predators"]
        assert obs_a["prey"] == obs_b["prey"]

    def test_reset_clears_done_flag(self):
        env = _make_env(seed=0, max_steps=1)
        env.reset()
        # Force a timeout
        env.step([0, 0, 0], 0)
        assert env.done
        env.reset()
        assert not env.done

    def test_reset_resets_step_count(self):
        env = _make_env(seed=0)
        env.reset()
        env.step([0, 0, 0], 0)
        env.reset()
        assert env.step_count == 0


# ===========================================================================
# 2. reset(positions=...) — explicit placement
# ===========================================================================

class TestResetWithPositions:
    PRED_CELLS = [(0, 0), (1, 1), (2, 2)]
    PREY_CELL  = (5, 5)
    ALL_CELLS  = PRED_CELLS + [PREY_CELL]

    def test_sequence_placement(self):
        env = _make_env()
        obs = env.reset(positions=self.ALL_CELLS)
        assert obs["predators"] == self.PRED_CELLS
        assert obs["prey"] == self.PREY_CELL

    def test_dict_placement(self):
        env = _make_env()
        obs = env.reset(positions={"predators": self.PRED_CELLS,
                                   "prey": self.PREY_CELL})
        assert obs["predators"] == self.PRED_CELLS
        assert obs["prey"] == self.PREY_CELL

    def test_wrong_count_raises(self):
        env = _make_env()
        with pytest.raises(ValueError):
            env.reset(positions=[(0, 0), (1, 1)])  # only 2 cells

    def test_out_of_bounds_cell_raises(self):
        env = _make_env()
        with pytest.raises(ValueError):
            env.reset(positions=[(0, 0), (1, 1), (2, 2), (20, 0)])  # row 20 invalid

    def test_negative_cell_raises(self):
        env = _make_env()
        with pytest.raises(ValueError):
            env.reset(positions=[(-1, 0), (1, 1), (2, 2), (5, 5)])

    def test_duplicate_cells_raise(self):
        env = _make_env()
        with pytest.raises(ValueError):
            env.reset(positions=[(0, 0), (0, 0), (2, 2), (5, 5)])  # dup predators

    def test_corner_positions_accepted(self):
        """All 4 grid corners should be valid."""
        env = _make_env()
        obs = env.reset(positions=[(0, 0), (0, 19), (19, 0), (19, 19)])
        assert obs["predators"] == [(0, 0), (0, 19), (19, 0)]
        assert obs["prey"] == (19, 19)


# ===========================================================================
# 3. Out-of-bounds moves clip to Stay (all 4 walls & all 4 corners)
# ===========================================================================

class TestOutOfBoundsClipping:
    """
    Place one predator at a wall/corner cell and attempt to move out.
    Agent must remain in place; no exception should be raised.
    """

    def _step_single_predator(self, start_cell: tuple, action: int) -> tuple:
        """
        Place predator 0 at start_cell, two other predators safely away,
        prey safely away, then step predator 0 with the given action.
        Returns predator 0's new position.
        """
        env = _make_env()
        preds = [start_cell, (10, 10), (10, 11)]
        prey  = (15, 15)
        # Avoid conflicts if start_cell overlaps with safe cells
        env.reset(positions=preds + [prey])
        obs, _, _ = env.step([action, 0, 0], 0)
        return obs["predators"][0]

    def test_north_wall_move_north(self):
        new_pos = self._step_single_predator((0, 5), action=1)  # North from row 0
        assert new_pos == (0, 5), f"Expected (0,5), got {new_pos}"

    def test_south_wall_move_south(self):
        new_pos = self._step_single_predator((19, 5), action=3)  # South from row 19
        assert new_pos == (19, 5)

    def test_west_wall_move_west(self):
        new_pos = self._step_single_predator((5, 0), action=4)  # West from col 0
        assert new_pos == (5, 0)

    def test_east_wall_move_east(self):
        new_pos = self._step_single_predator((5, 19), action=2)  # East from col 19
        assert new_pos == (5, 19)

    def test_top_left_corner_move_north(self):
        new_pos = self._step_single_predator((0, 0), action=1)
        assert new_pos == (0, 0)

    def test_top_left_corner_move_west(self):
        new_pos = self._step_single_predator((0, 0), action=4)
        assert new_pos == (0, 0)

    def test_bottom_right_corner_move_south(self):
        new_pos = self._step_single_predator((19, 19), action=3)
        assert new_pos == (19, 19)

    def test_bottom_right_corner_move_east(self):
        new_pos = self._step_single_predator((19, 19), action=2)
        assert new_pos == (19, 19)

    def test_prey_clips_at_north_wall(self):
        env = _make_env()
        env.reset(positions=[(10, 10), (10, 11), (10, 12), (0, 5)])
        obs, _, _ = env.step([0, 0, 0], 1)  # prey moves North from (0,5)
        assert obs["prey"] == (0, 5)

    def test_no_exception_on_oob_move(self):
        """Calling step with an OOB move must never raise an exception."""
        env = _make_env()
        env.reset(positions=[(0, 0), (1, 0), (0, 1), (5, 5)])
        try:
            env.step([1, 0, 0], 0)   # predator 0 tries North from (0,0)
        except Exception as exc:
            pytest.fail(f"Unexpected exception on OOB move: {exc}")


# ===========================================================================
# 4. Predators can share cells simultaneously
# ===========================================================================

class TestPredatorsShareCells:
    def test_two_predators_move_to_same_cell(self):
        """
        Predator 0 at (5,5), Predator 1 at (5,7).
        P0 moves East, P1 moves West — both land on (5,6).
        Prey is far away so no capture.
        """
        env = _make_env()
        env.reset(positions=[(5, 5), (5, 7), (10, 10), (15, 15)])
        obs, done, info = env.step([2, 4, 0], 0)   # P0→E, P1→W, P2→Stay, prey→Stay
        preds = obs["predators"]
        assert preds[0] == (5, 6)
        assert preds[1] == (5, 6)
        assert not info["capture"], "Shared predator cell should not trigger capture"

    def test_all_three_predators_converge(self):
        """All 3 predators move to the same centre cell."""
        env = _make_env()
        # Place predators N/S/W of (10,10); prey far away
        env.reset(positions=[(9, 10), (11, 10), (10, 9), (0, 0)])
        obs, done, info = env.step([3, 1, 2, ], 0)  # S, N, E — all converge on (10,10)
        preds = obs["predators"]
        assert preds[0] == (10, 10)
        assert preds[1] == (10, 10)
        assert preds[2] == (10, 10)
        assert not info["capture"]


# ===========================================================================
# 5. Action encoding correctness
# ===========================================================================

class TestActionEncoding:
    """Verify each action integer moves agents to the exact expected cell."""

    def _move_prey(self, start: tuple, action: int) -> tuple:
        env = _make_env()
        env.reset(positions=[(10, 10), (10, 11), (10, 12), start])
        obs, _, _ = env.step([0, 0, 0], action)
        return obs["prey"]

    def test_stay(self):
        assert self._move_prey((5, 5), 0) == (5, 5)

    def test_north(self):
        assert self._move_prey((5, 5), 1) == (4, 5)   # row -1

    def test_east(self):
        assert self._move_prey((5, 5), 2) == (5, 6)   # col +1

    def test_south(self):
        assert self._move_prey((5, 5), 3) == (6, 5)   # row +1

    def test_west(self):
        assert self._move_prey((5, 5), 4) == (5, 4)   # col -1

    def test_predator_north(self):
        env = _make_env()
        env.reset(positions=[(5, 5), (10, 10), (10, 11), (15, 15)])
        obs, _, _ = env.step([1, 0, 0], 0)
        assert obs["predators"][0] == (4, 5)

    def test_predator_east(self):
        env = _make_env()
        env.reset(positions=[(5, 5), (10, 10), (10, 11), (15, 15)])
        obs, _, _ = env.step([2, 0, 0], 0)
        assert obs["predators"][0] == (5, 6)

    def test_predator_south(self):
        env = _make_env()
        env.reset(positions=[(5, 5), (10, 10), (10, 11), (15, 15)])
        obs, _, _ = env.step([3, 0, 0], 0)
        assert obs["predators"][0] == (6, 5)

    def test_predator_west(self):
        env = _make_env()
        env.reset(positions=[(5, 5), (10, 10), (10, 11), (15, 15)])
        obs, _, _ = env.step([4, 0, 0], 0)
        assert obs["predators"][0] == (5, 4)


# ===========================================================================
# 6. Same-cell capture
# ===========================================================================

class TestSameCellCapture:
    """
    Any predator moving onto the prey's new cell triggers capture.
    Tests all 3 predator indices individually.
    """

    @pytest.mark.parametrize("pred_idx", [0, 1, 2])
    def test_same_cell_capture_predator_moves_onto_prey(self, pred_idx):
        """
        Predator ``pred_idx`` is placed directly north of the prey.
        Predator moves South; prey Stays → same cell.
        """
        env = _make_env()
        prey_pos = (10, 10)
        pred_positions = [(9, 10), (1, 1), (1, 2)]  # default: P0 north of prey
        # Put the tested predator north of prey; others far away
        pred_positions[pred_idx] = (9, 10)
        for j in range(3):
            if j != pred_idx:
                pred_positions[j] = (1, j)

        env.reset(positions=pred_positions + [prey_pos])
        actions = [0, 0, 0]
        actions[pred_idx] = 3   # South → land on (10,10) == prey
        obs, done, info = env.step(actions, 0)   # prey Stays

        assert done, "done must be True after same-cell capture"
        assert info["capture"], "info['capture'] must be True"
        assert info["capture_step"] == 1, "Capture on step 1"
        assert not info["timeout"]

    def test_same_cell_capture_prey_moves_onto_predator(self):
        """
        Prey moves onto a stationary predator cell → still a same-cell capture
        (new_predator == new_prey).
        """
        env = _make_env()
        env.reset(positions=[(10, 10), (1, 1), (1, 2), (10, 11)])
        # Prey moves West from (10,11) → (10,10) where predator 0 stays
        obs, done, info = env.step([0, 0, 0], 4)
        assert done
        assert info["capture"]

    def test_capture_step_matches_step_count(self):
        env = _make_env()
        env.reset(positions=[(9, 10), (1, 1), (1, 2), (10, 10)])
        obs, done, info = env.step([3, 0, 0], 0)
        assert info["capture_step"] == info["step_count"] == 1

    def test_step_raises_after_capture(self):
        env = _make_env()
        env.reset(positions=[(9, 10), (1, 1), (1, 2), (10, 10)])
        env.step([3, 0, 0], 0)   # capture on step 1
        with pytest.raises(RuntimeError):
            env.step([0, 0, 0], 0)   # must raise — episode is done


# ===========================================================================
# 7. Edge-swap capture
# ===========================================================================

class TestEdgeSwapCapture:
    """
    Two agents swap positions in one step → capture.
    Tests all 3 predator indices.
    """

    @pytest.mark.parametrize("pred_idx", [0, 1, 2])
    def test_edge_swap_capture(self, pred_idx):
        """
        Predator at (10, 9), prey at (10, 10).
        Predator moves East (→10,10); prey moves West (→10,9).
        They cross the same edge → edge-swap capture.
        """
        pred_start = (10, 9)
        prey_start = (10, 10)

        pred_positions = [(1, 1), (1, 2), (1, 3)]
        pred_positions[pred_idx] = pred_start
        # Make sure other predators don't accidentally land on prey
        others = [(r, c) for r, c in [(1, 1), (1, 2), (1, 3)]
                  if (r, c) != pred_start][:2]
        for j in range(3):
            if j != pred_idx:
                pred_positions[j] = others.pop(0)

        env = _make_env()
        env.reset(positions=pred_positions + [prey_start])

        actions = [0, 0, 0]
        actions[pred_idx] = 2   # East: (10,9) → (10,10)
        obs, done, info = env.step(actions, 4)  # prey West: (10,10) → (10,9)

        assert done, f"Edge-swap capture failed for predator index {pred_idx}"
        assert info["capture"], "info['capture'] must be True on edge-swap"
        assert info["capture_step"] is not None
        assert not info["timeout"]

    def test_edge_swap_vertical(self):
        """
        Predator at (9, 10), prey at (10, 10).
        Predator moves South, prey moves North → vertical edge-swap.
        """
        env = _make_env()
        env.reset(positions=[(9, 10), (1, 1), (1, 2), (10, 10)])
        obs, done, info = env.step([3, 0, 0], 1)  # P0 South, prey North
        assert done
        assert info["capture"]

    def test_no_capture_without_swap(self):
        """
        Adjacent predator and prey both move in the same direction —
        no capture should occur (they maintain distance).
        """
        env = _make_env()
        # P0=(5,5), prey=(5,6) both move East — predator ends at (5,6), prey (5,7)
        env.reset(positions=[(5, 5), (1, 1), (1, 2), (5, 6)])
        obs, done, info = env.step([2, 0, 0], 2)   # both move East
        assert not done
        assert not info["capture"]


# ===========================================================================
# 8. Timeout at step 300
# ===========================================================================

class TestTimeout:
    def test_timeout_at_max_steps(self):
        env = _make_env(seed=0)
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (19, 19)])
        # Run 299 steps with no capture (all stay)
        for step_n in range(config.MAX_STEPS - 1):
            obs, done, info = env.step([0, 0, 0], 0)
            assert not done, f"Should not be done at step {step_n + 1}"

        # 300th step — should timeout
        obs, done, info = env.step([0, 0, 0], 0)
        assert done, "done must be True at step 300"
        assert not info["capture"], "capture must be False on timeout"
        assert info["capture_step"] is None, "capture_step must be None on timeout"
        assert info["timeout"], "timeout must be True"
        assert info["step_count"] == config.MAX_STEPS

    def test_timeout_step_count_in_obs(self):
        env = _make_env(seed=0)
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (19, 19)])
        for _ in range(config.MAX_STEPS):
            obs, done, info = env.step([0, 0, 0], 0)
        assert obs["step"] == config.MAX_STEPS

    def test_custom_max_steps(self):
        """Environment respects a custom max_steps != 300."""
        env = _make_env(max_steps=5)
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (19, 19)])
        for _ in range(4):
            _, done, _ = env.step([0, 0, 0], 0)
            assert not done
        _, done, info = env.step([0, 0, 0], 0)
        assert done
        assert info["timeout"]
        assert info["step_count"] == 5

    def test_step_raises_after_timeout(self):
        env = _make_env(max_steps=1)
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (19, 19)])
        env.step([0, 0, 0], 0)  # timeout on step 1
        with pytest.raises(RuntimeError):
            env.step([0, 0, 0], 0)


# ===========================================================================
# 9. Miscellaneous edge cases & contract checks
# ===========================================================================

class TestMiscellaneous:
    def test_prev_prey_updated_after_step(self):
        env = _make_env()
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (5, 5)])
        obs1, _, _ = env.step([0, 0, 0], 2)   # prey East → (5,6)
        assert obs1["prev_prey"] == (5, 5)
        obs2, _, _ = env.step([0, 0, 0], 2)   # prey East → (5,7)
        assert obs2["prev_prey"] == (5, 6)

    def test_invalid_predator_action_raises(self):
        env = _make_env()
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (5, 5)])
        with pytest.raises(ValueError):
            env.step([5, 0, 0], 0)   # action 5 is invalid

    def test_invalid_prey_action_raises(self):
        env = _make_env()
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (5, 5)])
        with pytest.raises(ValueError):
            env.step([0, 0, 0], -1)   # action -1 is invalid

    def test_wrong_number_of_predator_actions_raises(self):
        env = _make_env()
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (5, 5)])
        with pytest.raises(ValueError):
            env.step([0, 0], 0)   # only 2 actions supplied

    def test_num_observations_constant(self):
        """Verify the 16,200 state count is computed correctly."""
        expected = (
            config.OBS_BEARING_BINS
            * config.OBS_RANGE_BINS
            * config.OBS_TEAMMATE_QUAD_BINS
            * config.OBS_TEAMMATE_RANGE_BINS
            * config.OBS_PREV_PREY_BINS
            * config.OBS_WALL_BINS
        )
        assert config.NUM_OBSERVATIONS == 16_200
        assert config.NUM_OBSERVATIONS == expected

    def test_action_names_dict_coverage(self):
        """ACTION_NAMES must be a dict covering all 5 action indices."""
        assert isinstance(config.ACTION_NAMES, dict)
        assert set(config.ACTION_NAMES.keys()) == {0, 1, 2, 3, 4}

    def test_obs_keys_present(self):
        env = _make_env(seed=7)
        obs = env.reset()
        for key in ("predators", "prey", "prev_prey", "step"):
            assert key in obs, f"Key '{key}' missing from obs"

    def test_step_count_increments(self):
        env = _make_env(seed=0)
        env.reset(positions=[(0, 0), (0, 1), (0, 2), (19, 19)])
        for expected_step in range(1, 6):
            obs, _, info = env.step([0, 0, 0], 0)
            assert obs["step"] == expected_step
            assert info["step_count"] == expected_step

