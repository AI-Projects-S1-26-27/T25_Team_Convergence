"""
tests/test_prey.py
==================
Unit and integration tests for agents/prey.py (Task T-03).

Covers:
  1. test_prey_moves_toward_open_corner: prey flees towards open corner
  2. test_prey_boundary_legal_moves: corners and edges legal moves
  3. test_prey_trapped_edge_case: returns 0 on legal_moves=[0] or []
  4. test_reproducibility_with_seed: deterministic RNG reproducibility
  5. manhattan_dist tests
  6. get_legal_moves comprehensive tests (interior, corners, edges)
  7. prey_action stochastic & greedy flee mechanics
  8. Integration with Environment
"""

from __future__ import annotations

import sys
import os
from collections import Counter

import pytest
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents.prey import get_legal_moves, prey_action, manhattan_dist
from env.environment import Environment
import config


# ===========================================================================
# Explicitly Required Tests (Task T-03 Specification)
# ===========================================================================

def test_prey_moves_toward_open_corner():
    """
    Place prey at center (10, 10).
    Place 3 predators in corners: (0, 0) [Top-Left], (0, 19) [Top-Right], and (19, 0) [Bottom-Left].
    The only safe open corner is Bottom-Right (19, 19).
    Use a fixed seed numpy RNG to account for the 10% random branch across 100 trials,
    and assert that the prey predominantly selects South (3) or East (2).
    """
    prey_pos = (10, 10)
    pred_positions = [(0, 0), (0, 19), (19, 0)]
    rng = np.random.default_rng(42)

    actions = [
        prey_action(prey_pos, pred_positions, rng=rng)
        for _ in range(100)
    ]

    # Count selections of South (3) or East (2)
    se_count = sum(1 for a in actions if a in (2, 3))
    # In greedy branch (90% of trials), South and East maximize distance towards the open corner
    assert se_count >= 80, (
        f"Prey did not predominantly select South (3) or East (2). Got {se_count}/100"
    )


def test_prey_boundary_legal_moves():
    """
    Test corners (0, 0), (0, 19), (19, 0), (19, 19) and edges.
    Verify that get_legal_moves never suggests an action that walks off the 20x20 grid.
    """
    corners = [(0, 0), (0, 19), (19, 0), (19, 19)]
    edges = [(0, 10), (19, 10), (10, 0), (10, 19)]
    gs = config.GRID_SIZE

    for pos in corners + edges:
        moves = get_legal_moves(pos, grid_size=gs)
        assert 0 in moves, f"Stay (0) must always be legal at {pos}"
        for action in moves:
            dr, dc = config.ACTIONS[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            assert 0 <= nr < gs, f"Action {action} from {pos} gives out of bounds row {nr}"
            assert 0 <= nc < gs, f"Action {action} from {pos} gives out of bounds col {nc}"


def test_prey_trapped_edge_case():
    """
    Call prey_action with legal_moves=[0] or empty list, assert it safely returns 0 (Stay).
    """
    preds = [(1, 1), (2, 2), (3, 3)]

    # Call with legal_moves=[0]
    res1 = prey_action((0, 0), preds, legal_moves=[0])
    assert res1 == 0, f"Expected 0 for legal_moves=[0], got {res1}"

    # Call with empty list []
    res2 = prey_action((0, 0), preds, legal_moves=[])
    assert res2 == 0, f"Expected 0 for legal_moves=[], got {res2}"


def test_reproducibility_with_seed():
    """
    Ensure that passing the same seed via np.random.default_rng(seed) produces
    the exact same action sequence.
    """
    prey_pos = (10, 10)
    pred_positions = [(5, 5), (15, 5), (10, 15)]

    rng1 = np.random.default_rng(12345)
    seq1 = [prey_action(prey_pos, pred_positions, rng=rng1) for _ in range(50)]

    rng2 = np.random.default_rng(12345)
    seq2 = [prey_action(prey_pos, pred_positions, rng=rng2) for _ in range(50)]

    assert seq1 == seq2, f"Action sequences differed with identical seed: {seq1} vs {seq2}"


# ===========================================================================
# Manhattan Distance Tests
# ===========================================================================

def test_manhattan_dist_basic():
    assert manhattan_dist((0, 0), (0, 0)) == 0
    assert manhattan_dist((0, 0), (3, 4)) == 7
    assert manhattan_dist((10, 5), (5, 10)) == 10
    assert manhattan_dist((19, 19), (0, 0)) == 38


# ===========================================================================
# Comprehensive Test Suite: get_legal_moves
# ===========================================================================

class TestGetLegalMoves:

    def test_interior_all_five_legal(self):
        legal = get_legal_moves((10, 10))
        assert sorted(legal) == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize("corner,expected", [
        ((0,  0),  sorted([0, 2, 3])),   # Stay, East, South
        ((0,  19), sorted([0, 3, 4])),   # Stay, South, West (N and E blocked)
        ((19, 0),  sorted([0, 1, 2])),   # Stay, North, East
        ((19, 19), sorted([0, 1, 4])),   # Stay, North, West
    ])
    def test_corners(self, corner, expected):
        legal = get_legal_moves(corner)
        assert sorted(legal) == expected

    @pytest.mark.parametrize("pos,forbidden_action", [
        ((0,   5), 1),   # North wall -> action 1 (N) blocked
        ((19,  5), 3),   # South wall -> action 3 (S) blocked
        ((5,   0), 4),   # West wall  -> action 4 (W) blocked
        ((5,  19), 2),   # East wall  -> action 2 (E) blocked
    ])
    def test_wall_edges_missing_one_action(self, pos, forbidden_action):
        legal = get_legal_moves(pos)
        assert len(legal) == 4
        assert forbidden_action not in legal

    @pytest.mark.parametrize("pos", [
        (0, 0), (0, 19), (19, 0), (19, 19), (10, 10), (0, 10), (10, 0)
    ])
    def test_stay_always_legal(self, pos):
        assert 0 in get_legal_moves(pos)


# ===========================================================================
# Random-Trigger Roll Tests
# ===========================================================================

class TestRandomTrigger:

    def test_pre_sampled_rand_below_prob_takes_random(self):
        legal = get_legal_moves((10, 10))
        action = prey_action(
            (10, 10), [(5, 5), (5, 6), (5, 7)],
            legal_moves=legal,
            random_prob=0.1,
            pre_sampled_rand=0.0,       # < 0.1 -> random
            pre_sampled_choice=0,       # -> legal[0] = 0
        )
        assert action == legal[0]

    def test_pre_sampled_rand_above_prob_takes_greedy(self):
        action = prey_action(
            (10, 10), [(11, 10), (11, 11), (11, 12)],
            rng=np.random.default_rng(42),
            random_prob=0.1,
            pre_sampled_rand=0.5,       # >= 0.1 -> greedy
        )
        assert action == 1  # North

    def test_pre_sampled_rand_exactly_at_threshold_is_greedy(self):
        legal = get_legal_moves((10, 10))
        action = prey_action(
            (10, 10), [(11, 10), (11, 11), (11, 12)],
            rng=np.random.default_rng(0),
            random_prob=0.1,
            pre_sampled_rand=0.1,       # == threshold -> greedy (strict <)
            pre_sampled_choice=0,
        )
        assert action in legal
        assert action != 3  # South would move toward predators


# ===========================================================================
# Pre-Sampled Choice Tests
# ===========================================================================

class TestPreSampledChoice:

    def test_choice_index_zero(self):
        legal = get_legal_moves((10, 10))
        action = prey_action(
            (10, 10), [(0, 0), (0, 1), (0, 2)],
            random_prob=1.0,
            pre_sampled_rand=0.0,
            pre_sampled_choice=0,
        )
        assert action == legal[0]

    def test_choice_index_last(self):
        legal = get_legal_moves((10, 10))
        action = prey_action(
            (10, 10), [(0, 0), (0, 1), (0, 2)],
            random_prob=1.0,
            pre_sampled_rand=0.0,
            pre_sampled_choice=4,
        )
        assert action == legal[4]

    def test_choice_index_modulo(self):
        legal = get_legal_moves((10, 10))
        action = prey_action(
            (10, 10), [(0, 0), (0, 1), (0, 2)],
            random_prob=1.0,
            pre_sampled_rand=0.0,
            pre_sampled_choice=7,
        )
        assert action == legal[2]


# ===========================================================================
# Greedy Cardinal Flee Tests
# ===========================================================================

class TestGreedyFlee:

    GREEDY = dict(random_prob=0.0, pre_sampled_rand=1.0)

    def _action(self, prey, preds):
        return prey_action(
            prey, preds,
            rng=np.random.default_rng(0),
            **self.GREEDY,
        )

    def test_flee_south_from_north_predator(self):
        action = self._action((10, 10), [(9, 9), (9, 10), (9, 11)])
        assert action == 3  # South

    def test_flee_north_from_south_predator(self):
        action = self._action((10, 10), [(12, 10), (11, 9), (11, 11)])
        assert action == 1  # North

    def test_flee_east_from_west_predator(self):
        action = self._action((10, 10), [(10, 9), (11, 9), (9, 9)])
        assert action == 2  # East

    def test_flee_west_from_east_predator(self):
        action = self._action((10, 10), [(10, 11), (9, 11), (11, 11)])
        assert action == 4  # West

    def test_flee_produces_legal_action(self):
        for seed in range(20):
            rng_local = np.random.default_rng(seed)
            prey_pos = (rng_local.integers(0, 20), rng_local.integers(0, 20))
            preds = [
                (rng_local.integers(0, 20), rng_local.integers(0, 20))
                for _ in range(3)
            ]
            legal = get_legal_moves(prey_pos)
            action = prey_action(
                prey_pos, preds,
                rng=np.random.default_rng(seed + 100),
                **self.GREEDY,
            )
            assert action in legal


# ===========================================================================
# Multiple Predators & Tie-Breaking Tests
# ===========================================================================

class TestGreedyMultiplePredators:

    def test_maximises_min_distance(self):
        action = prey_action(
            (10, 10),
            [(10, 11), (10, 12), (10, 13)],
            random_prob=0.0,
            pre_sampled_rand=1.0,
            rng=np.random.default_rng(0),
        )
        assert action == 4  # West

    def test_avoids_nearest_predator(self):
        action = prey_action(
            (5, 5),
            [(5, 6), (0, 0), (19, 19)],
            random_prob=0.0,
            pre_sampled_rand=1.0,
            rng=np.random.default_rng(0),
        )
        assert action != 2  # East would move toward nearest predator


class TestTieBreaking:

    def test_all_tied_actions_appear(self):
        seen = set()
        for seed in range(200):
            action = prey_action(
                (10, 10), [(0, 0), (0, 1), (0, 2)],
                rng=np.random.default_rng(seed),
                random_prob=0.0,
                pre_sampled_rand=1.0,
            )
            seen.add(action)
        assert len(seen) >= 2


class TestErrors:

    def test_empty_predators_raises(self):
        with pytest.raises(ValueError, match="predator_positions"):
            prey_action((10, 10), [])


class TestCornerTrap:

    @pytest.mark.parametrize("corner,expected_legal", [
        ((0,  0),  [0, 2, 3]),
        ((0,  19), [0, 3, 4]),
        ((19, 0),  [0, 1, 2]),
        ((19, 19), [0, 1, 4]),
    ])
    def test_corner_has_exactly_three_legal_moves(self, corner, expected_legal):
        legal = get_legal_moves(corner)
        assert sorted(legal) == sorted(expected_legal)

    def test_single_legal_move_returned_no_rng_needed(self):
        action = prey_action(
            (0, 0), [(1, 1), (2, 2), (3, 3)],
            legal_moves=[0],
            rng=None,
        )
        assert action == 0


class TestRandomProbExtremes:

    def test_random_prob_zero_always_greedy(self):
        preds = [(10, 11), (9, 11), (11, 11)]
        seen = set()
        for seed in range(30):
            action = prey_action(
                (10, 10), preds,
                rng=np.random.default_rng(seed),
                random_prob=0.0,
            )
            seen.add(action)
        assert seen == {4}  # West is the unique greedy optimum

    def test_random_prob_one_always_random(self):
        preds = [(10, 11), (10, 12), (10, 13)]
        seen = set()
        for seed in range(100):
            action = prey_action(
                (10, 10), preds,
                rng=np.random.default_rng(seed),
                random_prob=1.0,
            )
            seen.add(action)
        assert len(seen) > 1


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:

    def test_prey_action_in_env_step(self):
        env = Environment(seed=0)
        rng = np.random.default_rng(0)
        obs = env.reset()

        for _ in range(50):
            action = prey_action(obs["prey"], obs["predators"], rng=rng)
            obs, done, info = env.step([0, 0, 0], action)
            assert action in range(5)
            if done:
                break

    def test_greedy_prey_survives_longer_than_stationary(self):
        positions = [(0, 0), (0, 1), (1, 0), (19, 19)]

        def run_episode(use_greedy: bool, seed: int) -> int:
            env = Environment(seed=seed)
            rng_local = np.random.default_rng(seed)
            obs = env.reset(positions=positions)
            for step in range(config.MAX_STEPS):
                if use_greedy:
                    pa = prey_action(
                        obs["prey"], obs["predators"],
                        rng=rng_local,
                        random_prob=0.0,
                    )
                else:
                    pa = 0
                obs, done, info = env.step([3, 2, 3], pa)
                if done:
                    return step + 1
            return config.MAX_STEPS

        greedy_steps = [run_episode(True, s) for s in range(10)]
        stay_steps = [run_episode(False, s) for s in range(10)]

        avg_greedy = sum(greedy_steps) / len(greedy_steps)
        avg_stay = sum(stay_steps) / len(stay_steps)

        assert avg_greedy >= avg_stay

    def test_pre_sampled_mode_reproducible_across_runs(self):
        rng_build = np.random.default_rng(55)
        n_steps = 20
        pre_rands = rng_build.random(n_steps)
        pre_choices = rng_build.integers(0, 5, size=n_steps)

        def replay():
            env = Environment(seed=0)
            obs = env.reset(positions=[(0, 0), (0, 1), (1, 0), (19, 19)])
            acts = []
            for t in range(n_steps):
                pa = prey_action(
                    obs["prey"], obs["predators"],
                    pre_sampled_rand=float(pre_rands[t]),
                    pre_sampled_choice=int(pre_choices[t]),
                )
                acts.append(pa)
                obs, done, _ = env.step([0, 0, 0], pa)
                if done:
                    break
            return acts

        assert replay() == replay()


# ===========================================================================
# Direct runner for standalone execution
# ===========================================================================

if __name__ == "__main__":
    test_prey_moves_toward_open_corner()
    test_prey_boundary_legal_moves()
    test_prey_trapped_edge_case()
    test_reproducibility_with_seed()
    test_manhattan_dist_basic()
    print("All top-level tests passed successfully!")
