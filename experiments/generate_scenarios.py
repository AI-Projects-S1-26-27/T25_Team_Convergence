"""
experiments/generate_scenarios.py  (Task T-11)
==============================================
Creates the fixed, matched scenarios every policy is evaluated on.

A scenario freezes everything random about an episode, so a learned team and a
scripted team start from the same cells and face the same prey "dice":

    {
      'init_positions': [(r,c) x3 predators, (r,c) prey],   # 4 distinct cells
      'seed':           int,                                # unique per scenario
      'index':          int,                                # position in its set
      'prey_randoms':   [float x 300],  # uniform [0,1): prey acts randomly at step t if < prey_random_prob
      'prey_choices':   [int x 300],    # pre-drawn index for the prey's random move / tie-break
    }

`prey_choices` are drawn from [0, 60). 60 = lcm(3,4,5), so `choice % n_legal`
is exactly uniform for any n_legal in {3, 4, 5} (corner / edge / interior).

Seed split (all disjoint, as the proposal requires):
    training    seeds 0-9      (live randomness inside train.py; no file)
    validation  seeds 100-109  -> experiments/val_scenarios.pkl  (5 per seed = 50)
    test        seed 200       -> experiments/test_scenarios.pkl (200)

Run once:    python experiments/generate_scenarios.py
Do NOT regenerate or edit the test file afterwards; every group relies on it.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402

EXPERIMENTS_DIR = os.path.join(_REPO_ROOT, "experiments")
TEST_PATH = os.path.join(EXPERIMENTS_DIR, "test_scenarios.pkl")
VAL_PATH = os.path.join(EXPERIMENTS_DIR, "val_scenarios.pkl")

VAL_SCENARIOS_PER_SEED = 5
CHOICE_RANGE = 60  # lcm(3, 4, 5)


def make_scenario(scenario_seed: int, index: int) -> Dict:
    """Build one scenario deterministically from its own seed."""
    rng = np.random.default_rng(scenario_seed)
    n_cells = config.GRID_SIZE * config.GRID_SIZE
    flat = rng.choice(n_cells, size=config.NUM_PREDATORS + config.NUM_PREY, replace=False)
    cells = [(int(f // config.GRID_SIZE), int(f % config.GRID_SIZE)) for f in flat]
    return {
        "init_positions": cells,  # [pred0, pred1, pred2, prey]
        "seed": int(scenario_seed),
        "index": int(index),
        "prey_randoms": [float(x) for x in rng.random(config.MAX_STEPS)],
        "prey_choices": [int(x) for x in rng.integers(0, CHOICE_RANGE, size=config.MAX_STEPS)],
    }


def make_test_scenarios(n: int = config.NUM_TEST_SCENARIOS, base_seed: int = config.TEST_SEED) -> List[Dict]:
    """The held-out test set: n scenarios from the single test seed."""
    return [make_scenario(base_seed * 10_000 + i, i) for i in range(n)]


def make_val_scenarios(
    seeds=config.VAL_SEEDS, per_seed: int = VAL_SCENARIOS_PER_SEED
) -> List[Dict]:
    """The validation set: `per_seed` scenarios from each validation seed."""
    out: List[Dict] = []
    for s in seeds:
        for j in range(per_seed):
            out.append(make_scenario(s * 10_000 + j, len(out)))
    return out


def save_scenarios(scenarios: List[Dict], path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(scenarios, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_scenarios(path: str = TEST_PATH) -> List[Dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    for path, scenarios in ((TEST_PATH, make_test_scenarios()), (VAL_PATH, make_val_scenarios())):
        if os.path.exists(path) and not args.force:
            print(f"SKIP  {os.path.relpath(path, _REPO_ROOT)} already exists (use --force to overwrite)")
            continue
        save_scenarios(scenarios, path)
        print(f"WROTE {os.path.relpath(path, _REPO_ROOT)}  ({len(scenarios)} scenarios)")


if __name__ == "__main__":
    main()
