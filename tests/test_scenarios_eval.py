"""tests/test_scenarios_eval.py -- experiments/generate_scenarios.py, evaluate.py, train.py (T-10, T-11)"""

from __future__ import annotations

import csv
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from experiments.evaluate import RandomPolicy, evaluate, run_scenario, summarize
from experiments.generate_scenarios import (
    TEST_PATH,
    VAL_PATH,
    load_scenarios,
    make_scenario,
    make_test_scenarios,
    make_val_scenarios,
)
from train import train


def test_scenario_structure_and_distinct_cells():
    sc = make_scenario(123, 0)
    assert set(sc) == {"init_positions", "seed", "index", "prey_randoms", "prey_choices"}
    assert len(sc["init_positions"]) == 4 and len(set(sc["init_positions"])) == 4
    assert all(0 <= r < 20 and 0 <= c < 20 for r, c in sc["init_positions"])
    assert len(sc["prey_randoms"]) == config.MAX_STEPS
    assert all(0.0 <= x < 1.0 for x in sc["prey_randoms"])
    assert all(0 <= x < 60 for x in sc["prey_choices"])


def test_generation_is_deterministic():
    assert make_test_scenarios(10) == make_test_scenarios(10)
    assert make_scenario(5, 0) != make_scenario(6, 0)


def test_test_and_val_sets_are_disjoint_and_correct_size():
    test, val = make_test_scenarios(), make_val_scenarios()
    assert len(test) == config.NUM_TEST_SCENARIOS == 200
    assert len(val) == 50
    assert not {s["seed"] for s in test} & {s["seed"] for s in val}


def test_saved_files_match_generator():
    # Guards against anyone editing/regenerating the shared test file by accident.
    assert load_scenarios(TEST_PATH) == make_test_scenarios()
    assert load_scenarios(VAL_PATH) == make_val_scenarios()


def test_evaluation_is_reproducible_and_summary_is_consistent():
    scs = make_test_scenarios(10)
    r1 = evaluate(RandomPolicy(0), scs)
    r2 = evaluate(RandomPolicy(0), scs)
    assert r1 == r2
    s = summarize(r1)
    assert s["n"] == 10 and 0.0 <= s["capture_rate"] <= 1.0
    assert all(r["steps_to_capture"] <= config.MAX_STEPS for r in r1)
    assert all((r["steps_to_capture"] == 300) for r in r1 if not r["captured"])


def test_trajectory_recording_starts_at_initial_positions():
    sc = make_scenario(77, 0)
    res = run_scenario(RandomPolicy(0), sc, record=True)
    preds, prey = res["trajectory"][0]
    assert preds + [prey] == sc["init_positions"]
    assert len(res["trajectory"]) == res["steps_to_capture"] + 1


def test_prey_random_prob_changes_outcomes_not_crashes():
    scs = make_test_scenarios(5)
    for p in (0.0, 0.1, 0.2):
        assert len(evaluate(RandomPolicy(0), scs, prey_random_prob=p)) == 5


def test_summarize_handles_no_captures():
    s = summarize([{"captured": False, "steps_to_capture": 300}] * 3)
    assert s["capture_rate"] == 0.0 and s["mean_time_penalized"] == 300


def test_short_training_run_writes_everything(tmp_path):
    out = train(seed=0, episodes=40, checkpoint_interval=20, run_name="t", results_dir=str(tmp_path), verbose=False)
    root = tmp_path / "t"
    assert (root / "checkpoints" / "seed_0" / "ep_20" / "predator_0.npy").exists()
    assert (root / "checkpoints" / "seed_0" / "ep_40" / "predator_2.npy").exists()
    assert (root / "best" / "seed_0" / "predator_1.npy").exists()
    info = json.loads((root / "best" / "seed_0" / "info.json").read_text())
    assert info["best_episode"] in (20, 40) and out["best_episode"] == info["best_episode"]
    rows = list(csv.DictReader((root / "logs" / "seed_0.csv").open()))
    assert [r["episode"] for r in rows] == ["20", "40"]
    assert {"episode", "epsilon", "val_success_rate"} <= set(rows[0])


def test_training_is_reproducible_for_a_seed(tmp_path):
    a = train(seed=3, episodes=30, checkpoint_interval=30, run_name="a", results_dir=str(tmp_path), verbose=False)
    b = train(seed=3, episodes=30, checkpoint_interval=30, run_name="b", results_dir=str(tmp_path), verbose=False)
    import numpy as np
    for i in range(3):
        qa = np.load(tmp_path / "a" / "best" / "seed_3" / f"predator_{i}.npy")
        qb = np.load(tmp_path / "b" / "best" / "seed_3" / f"predator_{i}.npy")
        assert np.array_equal(qa, qb)
    assert a["best_val_rate"] == b["best_val_rate"]
