"""
experiments/run_baselines.py  (Task EVAL-B — baseline evaluation on the test set)
=================================================================================
Runs the three scripted baselines on the 200 held-out test scenarios using the
team's shared evaluator (experiments/evaluate.py), so they face exactly the same
start cells and prey randomness as the learned predators.

    python experiments/run_baselines.py                          # prey_random_prob = 0.1
    python experiments/run_baselines.py --prey-random-prob 0.2 \
        --out results/robustness/baselines_p0.2.json              # robustness variant

Writes two files (default names shown):

  results/baseline_eval.json
      {"direct_pursuit": [200 result dicts], "predictive_interception": [...],
       "coordinated_pursuit": [...]}
      Each result dict (from evaluate.run_scenario):
        scenario          index 0..199 into experiments/test_scenarios.pkl
        captured          bool
        steps_to_capture  int  (300 if not captured)
        capture_step      int or null
        trajectory        [[ [[r,c]x3 predators], [r,c] prey ], ...]  one entry per
                          time step, starting with the initial positions
                          (omit with --no-trajectories)

  results/baseline_summary.json
      Settings used + capture rate, timeout-penalised mean time and conditional
      median time per baseline (also printed as a table).

The test-scenario file is only read, never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from baselines.coordinated_pursuit import CoordinatedPursuit  # noqa: E402
from baselines.direct_pursuit import DirectPursuit  # noqa: E402
from baselines.predictive_interception import PredictiveInterception  # noqa: E402
from experiments.evaluate import evaluate, summarize  # noqa: E402
from experiments.generate_scenarios import TEST_PATH, load_scenarios  # noqa: E402

BASELINES = (DirectPursuit, PredictiveInterception, CoordinatedPursuit)
DEFAULT_OUT = os.path.join(_REPO_ROOT, "results", "baseline_eval.json")
POLICY_SEED = 0  # base seed for tie-breaking; combined with each scenario's seed


def _to_jsonable(result: Dict) -> Dict:
    out = dict(result)
    if "trajectory" in out:
        out["trajectory"] = [
            [[list(p) for p in preds], list(prey)] for preds, prey in out["trajectory"]
        ]
    return out


def run_baselines(
    scenarios: Sequence[Dict],
    prey_random_prob: float = config.PREY_RANDOM_PROB,
    record: bool = True,
) -> Dict[str, List[Dict]]:
    """Evaluate every baseline on `scenarios`. Returns {name: [result dicts]}."""
    return {
        cls.name: [
            _to_jsonable(r)
            for r in evaluate(cls(seed=POLICY_SEED), scenarios, prey_random_prob, record)
        ]
        for cls in BASELINES
    }


def save_results(
    results: Dict[str, List[Dict]],
    out_path: str,
    prey_random_prob: float,
    scenario_path: str,
) -> str:
    """Write the results file and a summary file next to it. Returns summary path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)

    root, ext = os.path.splitext(out_path)
    summary_path = root.replace("baseline_eval", "baseline_summary") if "baseline_eval" in root \
        else root + "_summary"
    summary_path += ext or ".json"
    summary = {
        "settings": {
            "scenario_file": os.path.relpath(scenario_path, _REPO_ROOT).replace(os.sep, "/"),
            "n_scenarios": len(next(iter(results.values()))),
            "prey_random_prob": prey_random_prob,
            "max_steps": config.MAX_STEPS,
            "policy_seed": POLICY_SEED,
        },
        "summary": {name: summarize(rs) for name, rs in results.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def print_table(results: Dict[str, List[Dict]]) -> None:
    print(f"{'baseline':<26}{'capture rate':>13}{'mean time*':>12}{'median time':>13}")
    for name, rs in results.items():
        s = summarize(rs)
        print(f"{name:<26}{s['capture_rate']:>12.1%}{s['mean_time_penalized']:>12.1f}"
              f"{s['median_time_captured']:>13.1f}")
    print("* timeout-penalised: failed episodes count as 300 steps")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Group B scripted baselines.")
    parser.add_argument("--scenarios", default=TEST_PATH, help="scenario .pkl (default: test set)")
    parser.add_argument("--prey-random-prob", type=float, default=config.PREY_RANDOM_PROB)
    parser.add_argument("--out", default=DEFAULT_OUT, help="results JSON path")
    parser.add_argument("--no-trajectories", action="store_true", help="skip trajectory recording")
    args = parser.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    t0 = time.time()
    results = run_baselines(scenarios, args.prey_random_prob, record=not args.no_trajectories)
    summary_path = save_results(results, args.out, args.prey_random_prob, args.scenarios)

    print(f"{len(scenarios)} scenarios, prey_random_prob={args.prey_random_prob}, "
          f"{time.time() - t0:.1f}s")
    print_table(results)
    print(f"saved {os.path.relpath(args.out, _REPO_ROOT)} and "
          f"{os.path.relpath(summary_path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
