"""
tests/test_prey.py
==================
Unit and integration tests for agents/prey.py (Task T-03).

Coverage:
  1.  get_legal_moves — interior cell: all 5 actions legal
  2.  get_legal_moves — all 4 corners: exactly {Stay + 2 moves}
  3.  get_legal_moves — all 4 wall edges (non-corner): 4 actions
  4.  get_legal_moves — always includes Stay (action 0)
  5.  get_legal_moves — resulting cells never out of bounds
  6.  prey_action — takes random action when pre_sampled_rand < random_prob
  7.  prey_action — takes greedy action when pre_sampled_rand >= random_prob
  8.  prey_action — pre_sampled_choice selects correct index in random branch
  9.  prey_action — greedy flees away from single predator (cardinal directions)
  10. prey_action — greedy maximises min-distance when multiple predators present
  11. prey_action — tie-breaking is uniform (all tied actions appear over trials)
  12. prey_action — deterministic given fixed rng seed
  13. prey_action — live mode uses rng (no pre_sampled args)
  14. prey_action — legal_moves=None computes internally (same result)
  15. prey_action — raises ValueError when predator_positions is empty
  16. prey_action — single legal move returned immediately (corner trap)
  17. prey_action — custom random_prob = 0 → always greedy
  18. prey_action — custom random_prob = 1 → always random
  19. Integration — prey_action plugs into env.step() without error
  20. Integration — greedy prey survives longer than a stationary prey
"""

from __future__ import annotations

import sys
import os

import pytest
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agents.prey import get_legal_moves, prey_action  # noqa: E402
from env.environment import Environment               # noqa: E402
import config                                         # noqa: E402


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def interior_pos():
    return (10, 10)


# ===========================================================================
# 1–5: get_legal_moves
# ===========================================================================

class TestGetLegalMoves:

    # 1. Interior cell — all 5 actions legal
    def test_interior_all_five_legal(self, interior_pos):
        legal = get_legal_moves(interior_pos)
        assert sorted(legal) == [0, 1, 2, 3, 4]

    # 2. All 4 corners — exactly 3 actions (Stay + 2 cardinal)
    @pytest.mark.parametrize("corner,expected", [
        ((0,  0),  sorted([0, 2, 3])),   # Stay, East, South
        ((0,  19), sorted([0, 3, 4])),   # Stay, South, West (N and E blocked)
        ((19, 0),  sorted([0, 1, 2])),   # Stay, North, East
        ((19, 19), sorted([0, 1, 4])),   # Stay, North, West
    ])
    def test_corners(self, corner, expected):
        legal = get_legal_moves(corner)
        assert sorted(legal) == expected

    # 3. Wall edges (non-corner) — 4 actions
    @pytest.mark.parametrize("pos,forbidden_action", [
        ((0,   5), 1),   # North wall → action 1 (N) blocked
        ((19,  5), 3),   # South wall → action 3 (S) blocked
        ((5,   0), 4),   # West wall  → action 4 (W) blocked
        ((5,  19), 2),   # East wall  → action 2 (E) blocked
    ])
    def test_wall_edges_missing_one_action(self, pos, forbidden_action):
        legal = get_legal_moves(pos)
        assert len(legal) == 4
        assert forbidden_action not in legal

    # 4. Stay is always legal
    @pytest.mark.parametrize("pos", [
        (0, 0), (0, 19), (19, 0), (19, 19), (10, 10), (0, 10), (10, 0)
    ])
    def test_stay_always_legal(self, pos):
        assert 0 in get_legal_moves(pos)

    # 5. Resulting cells never exceed grid bounds
    @pytest.mark.parametrize("pos", [
        (0, 0), (0, 19), (19, 0), (19, 19), (0, 10), (10, 19), (10, 10)
    ])
    def test_resulting_cells_in_bounds(self, pos):
        gs = config.GRID_SIZE
        for action in get_legal_moves(pos):
            dr, dc = config.ACTIONS[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            assert 0 <= nr < gs, f"Row {nr} OOB for action {action} at {pos}"
            assert 0 <= nc < gs, f"Col {nc} OOB for action {action} at {pos}"


# ===========================================================================
# 6–7: Random-trigger roll via pre_sampled_rand
# ===========================================================================

class TestRandomTrigger:

    def test_pre_sampled_rand_below_prob_takes_random(self):
        """
        pre_sampled_rand = 0.0 < random_prob=0.1 → random branch.
        We supply pre_sampled_choice=0 so the choice is deterministic.
        """
        legal = get_legal_moves((10, 10))   # [0,1,2,3,4]
        action = prey_action(
            (10, 10), [(5, 5), (5, 6), (5, 7)],
            legal_moves=legal,
            random_prob=0.1,
            pre_sampled_rand=0.0,       # < 0.1 → random
            pre_sampled_choice=0,       # → legal[0] = 0
        )
        assert action == legal[0]

    def test_pre_sampled_rand_above_prob_takes_greedy(self):
        """
        Prey at (10,10), single predator directly south at (11,10).
        Greedy move is North (action 1) to maximise distance.
        pre_sampled_rand=0.5 >= 0.1 → greedy branch.
        """
        action = prey_action(
            (10, 10), [(11, 10), (11, 11), (11, 12)],
            rng=np.random.default_rng(42),
            random_prob=0.1,
            pre_sampled_rand=0.5,       # >= 0.1 → greedy
        )
        assert action == 1  # North

    def test_pre_sampled_rand_exactly_at_threshold_is_greedy(self):
        """
        random_prob=0.1, pre_sampled_rand=0.1 → NOT random (strict <).
        Greedy branch is taken. We don't assert *which* greedy action wins
        (ties are broken randomly) — just that it is legal and that the
        random branch was NOT taken (i.e. we can verify by checking against
        a known random-branch outcome forced with pre_sampled_choice=0).
        """
        legal = get_legal_moves((10, 10))   # [0,1,2,3,4]
        # Random-branch would return legal[0] = 0 if pre_sampled_choice=0
        random_branch_action = legal[0]  # == 0

        action = prey_action(
            (10, 10), [(11, 10), (11, 11), (11, 12)],
            rng=np.random.default_rng(0),
            random_prob=0.1,
            pre_sampled_rand=0.1,       # == threshold → greedy (strict <)
            pre_sampled_choice=0,       # only matters if random branch taken
        )
        # Action must be legal
        assert action in legal
        # The greedy policy with these predators must move AWAY (not toward them).
        # Predators are at row 11 (south) → greedy moves should be northward
        # or sideways, NOT south (action 3).
        assert action != 3  # South would move toward the predators


# ===========================================================================
# 8: pre_sampled_choice selects correct index
# ===========================================================================

class TestPreSampledChoice:

    def test_choice_index_zero(self):
        legal = get_legal_moves((10, 10))   # [0,1,2,3,4]
        action = prey_action(
            (10, 10), [(0, 0), (0, 1), (0, 2)],
            random_prob=1.0,            # always random
            pre_sampled_rand=0.0,
            pre_sampled_choice=0,
        )
        assert action == legal[0]

    def test_choice_index_last(self):
        legal = get_legal_moves((10, 10))   # [0,1,2,3,4]
        action = prey_action(
            (10, 10), [(0, 0), (0, 1), (0, 2)],
            random_prob=1.0,
            pre_sampled_rand=0.0,
            pre_sampled_choice=4,       # → legal[4] = 4
        )
        assert action == legal[4]

    def test_choice_index_modulo(self):
        """Out-of-range index is taken modulo len(legal_moves) for safety."""
        legal = get_legal_moves((10, 10))   # len = 5
        action = prey_action(
            (10, 10), [(0, 0), (0, 1), (0, 2)],
            random_prob=1.0,
            pre_sampled_rand=0.0,
            pre_sampled_choice=7,       # 7 % 5 = 2 → legal[2] = 2
        )
        assert action == legal[2]


# ===========================================================================
# 9: Greedy flees in correct cardinal direction
# ===========================================================================

class TestGreedyFlee:
    """
    Place a single predator in each cardinal direction close to the prey.
    The optimal flee move is away from the predator (or the most open axis).
    We set random_prob=0 and pre_sampled_rand=1.0 to guarantee greedy.
    """

    GREEDY = dict(random_prob=0.0, pre_sampled_rand=1.0)

    def _action(self, prey, preds):
        return prey_action(
            prey, preds,
            rng=np.random.default_rng(0),
            **self.GREEDY,
        )

    def test_flee_south_from_north_predator(self):
        """
        Prey at (10,10). Predators cluster north: (9,9),(9,10),(9,11).
        min-dist for each candidate move:
          Stay  (10,10) → 1  East  (10,11) → 1  West (10,9) → 1
          North  (9,10) → 0  South(11,10) → 2  ← unique best
        Unique optimal flee: South (action 3).
        """
        action = self._action((10, 10), [(9, 9), (9, 10), (9, 11)])
        assert action == 3  # South

    def test_flee_north_from_south_predator(self):
        """
        Prey at (10,10). Predators cluster south: (12,10),(11,9),(11,11).
        min-dist for each candidate move:
          Stay  (10,10) → 1  South(11,10) → 0  East (10,11) → 1
          West  (10, 9) → 1  North (9,10) → 3   ← unique best
        Unique optimal flee: North (action 1).
        """
        action = self._action((10, 10), [(12, 10), (11, 9), (11, 11)])
        assert action == 1  # North

    def test_flee_east_from_west_predator(self):
        """
        Prey at (10,10). All 3 predators directly west.
        Unique optimal flee: East (action 2).
        """
        action = self._action((10, 10), [(10, 9), (11, 9), (9, 9)])
        assert action == 2  # East

    def test_flee_west_from_east_predator(self):
        """Predator directly east → flee West (action 4)."""
        action = self._action((10, 10), [(10, 11), (9, 11), (11, 11)])
        assert action == 4  # West

    def test_flee_produces_legal_action(self):
        """Greedy action is always in the legal set."""
        for seed in range(20):
            rng_local = np.random.default_rng(seed)
            prey_pos  = (rng_local.integers(0, 20), rng_local.integers(0, 20))
            preds     = [
                (rng_local.integers(0, 20), rng_local.integers(0, 20))
                for _ in range(3)
            ]
            legal = get_legal_moves(prey_pos)
            action = prey_action(
                prey_pos, preds,
                rng=np.random.default_rng(seed + 100),
                **self.GREEDY,
            )
            assert action in legal, (
                f"Greedy action {action} not in legal {legal} at {prey_pos}"
            )


# ===========================================================================
# 10: Greedy maximises min distance with multiple predators
# ===========================================================================

class TestGreedyMultiplePredators:

    def test_maximises_min_distance(self):
        """
        Prey at (10,10). Predators at (10,11), (10,12), (10,13).
        All predators are to the east — best flee is West (action 4),
        which maximises distance to the nearest predator (10,11).
        """
        action = prey_action(
            (10, 10),
            [(10, 11), (10, 12), (10, 13)],
            random_prob=0.0,
            pre_sampled_rand=1.0,
            rng=np.random.default_rng(0),
        )
        assert action == 4  # West

    def test_avoids_nearest_predator(self):
        """
        Prey at (5,5). One predator very close (5,6), two far away.
        Greedy should not move East (toward nearest).
        """
        legal = get_legal_moves((5, 5))
        action = prey_action(
            (5, 5),
            [(5, 6), (0, 0), (19, 19)],
            random_prob=0.0,
            pre_sampled_rand=1.0,
            rng=np.random.default_rng(0),
        )
        # Moving East would land on (5,6) → distance 0 from predator 0 — worst move
        assert action != 2


# ===========================================================================
# 11: Tie-breaking is uniform
# ===========================================================================

class TestTieBreaking:

    def test_all_tied_actions_appear(self):
        """
        Prey at (10,10), single predator far at (0,0).
        Many moves have the same min-distance — all should appear
        across enough trials with different rng states.
        """
        seen = set()
        for seed in range(200):
            action = prey_action(
                (10, 10), [(0, 0), (0, 1), (0, 2)],
                rng=np.random.default_rng(seed),
                random_prob=0.0,
                pre_sampled_rand=1.0,
            )
            seen.add(action)
        # At (10,10) with predators at top-left corner, South and East
        # moves give equal max-min distance; we expect at least 2 distinct
        # actions to appear across 200 seeds (tie-breaking is random).
        assert len(seen) >= 2, f"Only one action seen across all seeds: {seen}"


# ===========================================================================
# 12–13: Determinism and live-mode RNG
# ===========================================================================

class TestDeterminism:

    def test_same_seed_same_result(self):
        args = dict(
            prey_pos=(10, 10),
            predator_positions=[(5, 5), (15, 5), (10, 15)],
        )
        a1 = prey_action(**args, rng=np.random.default_rng(42))
        a2 = prey_action(**args, rng=np.random.default_rng(42))
        assert a1 == a2

    def test_different_seeds_can_differ(self):
        """
        With random_prob=1.0 and multiple legal moves, different seeds
        should sometimes produce different actions.
        """
        results = {
            prey_action(
                (10, 10), [(5, 5), (5, 6), (5, 7)],
                rng=np.random.default_rng(s),
                random_prob=1.0,
            )
            for s in range(50)
        }
        assert len(results) > 1, "Expected variation across seeds"

    def test_live_mode_no_presampled(self):
        """Calling with only rng should work without error."""
        action = prey_action(
            (10, 10), [(5, 5), (5, 6), (5, 7)],
            rng=np.random.default_rng(7),
        )
        assert action in range(5)


# ===========================================================================
# 14: legal_moves=None computes internally
# ===========================================================================

class TestLegalMovesNone:

    def test_none_matches_explicit(self):
        pos = (3, 17)
        preds = [(0, 0), (1, 1), (2, 2)]
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)

        action_explicit = prey_action(
            pos, preds,
            legal_moves=get_legal_moves(pos),
            rng=rng_a,
            random_prob=0.0,
            pre_sampled_rand=1.0,
        )
        action_none = prey_action(
            pos, preds,
            legal_moves=None,
            rng=rng_b,
            random_prob=0.0,
            pre_sampled_rand=1.0,
        )
        assert action_explicit == action_none


# ===========================================================================
# 15: ValueError for empty predator_positions
# ===========================================================================

class TestErrors:

    def test_empty_predators_raises(self):
        with pytest.raises(ValueError, match="predator_positions"):
            prey_action((10, 10), [])


# ===========================================================================
# 16: Single legal move returned immediately (corner trap)
# ===========================================================================

class TestCornerTrap:

    @pytest.mark.parametrize("corner,expected_legal", [
        ((0,  0),  [0, 2, 3]),
        ((0,  19), [0, 3, 4]),
        ((19, 0),  [0, 1, 2]),
        ((19, 19), [0, 1, 4]),
    ])
    def test_corner_has_exactly_three_legal_moves(self, corner, expected_legal):
        """Corners have exactly 3 legal moves (Stay + 2 cardinal)."""
        legal = get_legal_moves(corner)
        assert sorted(legal) == sorted(expected_legal)

    def test_single_legal_move_returned_no_rng_needed(self):
        """
        Fabricate a situation where only Stay is supplied.
        Should return 0 immediately with no rng calls.
        """
        action = prey_action(
            (0, 0), [(1, 1), (2, 2), (3, 3)],
            legal_moves=[0],           # only Stay is legal
            rng=None,                  # no rng needed for 1-action case
        )
        assert action == 0


# ===========================================================================
# 17–18: random_prob extremes
# ===========================================================================

class TestRandomProbExtremes:

    def test_random_prob_zero_always_greedy(self):
        """
        With random_prob=0, random branch is never taken.
        All 3 predators are directly east — the unique greedy move is West.
        (North/South are not strictly better: they don't increase distance
        to the nearest predator at (10,11), they just trade row for col.)

        To make West the *unique* optimal move we need all predators in a
        column to the east so that West uniquely maximises min-distance.
        Predators: (10,11), (9,11), (11,11) — all in col 11 east of prey.
        West → (10,9): min-dist = min(2,2,2) = 2
        North→ (9,10): min-dist = min(1+1, 0+1, 2+1) = min(2,1,3) = 1  ← worse
        South→ (11,10): similar analysis → 1 ← worse
        East → (10,11): min-dist = 0 ← worst
        Stay → (10,10): min-dist = 1
        → West (action 4) is uniquely optimal.
        """
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
        """With random_prob=1, greedy branch is never taken."""
        preds = [(10, 11), (10, 12), (10, 13)]
        seen = set()
        for seed in range(100):
            action = prey_action(
                (10, 10), preds,
                rng=np.random.default_rng(seed),
                random_prob=1.0,
            )
            seen.add(action)
        # With random_prob=1 and 100 seeds, should see multiple actions
        assert len(seen) > 1


# ===========================================================================
# 19–20: Integration tests with Environment
# ===========================================================================

class TestIntegration:

    def test_prey_action_in_env_step(self):
        """
        prey_action output must be accepted by env.step() without error
        across 50 consecutive steps.
        """
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
        """
        Run 10 episodes with identical start positions.
        A greedy-fleeing prey should survive at least as long as a stationary
        prey (action 0 every step) on average.
        """
        positions = [(0, 0), (0, 1), (1, 0), (19, 19)]  # preds + prey

        def run_episode(use_greedy: bool, seed: int) -> int:
            env = Environment(seed=seed)
            rng_local = np.random.default_rng(seed)
            obs = env.reset(positions=positions)
            for step in range(config.MAX_STEPS):
                if use_greedy:
                    pa = prey_action(
                        obs["prey"], obs["predators"],
                        rng=rng_local,
                        random_prob=0.0,  # pure greedy for fair comparison
                    )
                else:
                    pa = 0  # always Stay
                obs, done, info = env.step([3, 2, 3], pa)  # preds march S/E/S
                if done:
                    return step + 1
            return config.MAX_STEPS

        greedy_steps  = [run_episode(True,  s) for s in range(10)]
        stay_steps    = [run_episode(False, s) for s in range(10)]

        avg_greedy = sum(greedy_steps) / len(greedy_steps)
        avg_stay   = sum(stay_steps)   / len(stay_steps)

        assert avg_greedy >= avg_stay, (
            f"Greedy prey (avg {avg_greedy:.1f}) survived fewer steps than "
            f"stationary prey (avg {avg_stay:.1f})"
        )

    def test_pre_sampled_mode_reproducible_across_runs(self):
        """
        Build pre_sampled arrays for one episode and replay twice — must get
        identical prey action sequences.
        """
        rng_build = np.random.default_rng(55)
        n_steps   = 20
        pre_rands   = rng_build.random(n_steps)
        pre_choices = rng_build.integers(0, 5, size=n_steps)

        def replay():
            env  = Environment(seed=0)
            obs  = env.reset(positions=[(0,0),(0,1),(1,0),(19,19)])
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
