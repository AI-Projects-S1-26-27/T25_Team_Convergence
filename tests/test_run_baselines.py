"""tests/test_run_baselines.py -- experiments/run_baselines.py (EVAL-B)"""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.generate_scenarios import make_val_scenarios
from experiments.run_baselines import run_baselines, save_results

NAMES = {"direct_pursuit", "predictive_interception", "coordinated_pursuit"}


def test_results_have_expected_structure(tmp_path):
    scenarios = make_val_scenarios()[:3]
    results = run_baselines(scenarios)
    assert set(results) == NAMES
    for rs in results.values():
        assert len(rs) == 3
        for r, sc in zip(rs, scenarios):
            assert r["scenario"] == sc["index"]
            assert set(r) == {"scenario", "captured", "steps_to_capture", "capture_step", "trajectory"}
            preds0, prey0 = r["trajectory"][0]
            assert [tuple(p) for p in preds0] + [tuple(prey0)] == list(sc["init_positions"])
            assert len(r["trajectory"]) == r["steps_to_capture"] + 1

    out = tmp_path / "baseline_eval.json"
    summary_path = save_results(results, str(out), 0.1, str(tmp_path / "x.pkl"))
    assert json.loads(out.read_text()) == results
    summary = json.loads(open(summary_path).read())
    assert set(summary["summary"]) == NAMES
    assert summary["settings"]["prey_random_prob"] == 0.1
    assert os.path.basename(summary_path) == "baseline_summary.json"


def test_runs_are_reproducible_and_trajectories_optional():
    scenarios = make_val_scenarios()[:3]
    a = run_baselines(scenarios, record=False)
    b = run_baselines(scenarios, record=False)
    assert a == b
    assert all("trajectory" not in r for rs in a.values() for r in rs)
