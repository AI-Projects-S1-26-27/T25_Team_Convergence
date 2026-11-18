"""
experiments/evaluate_all_seeds.py
=================================
Evaluate the validation-selected best checkpoint of EVERY trained seed on the
200 held-out test scenarios (frozen Q-tables, epsilon = 0, no learning).

    python experiments/evaluate_all_seeds.py

Writes results/test_eval/<run-name>_prey_random_<p>.json containing, for each
seed, the per-scenario outcomes plus its summary, and an across-seed summary
(mean, std, and a 95% bootstrap CI that resamples whole training runs).

Conditions evaluated at prey randomness 0.1: full, ablation_no_shaping,
ablation_no_teammate. The full model is also evaluated at 0.0 and 0.2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from experiments.evaluate import evaluate, summarize  # noqa: E402
from experiments.generate_scenarios import TEST_PATH, load_scenarios  # noqa: E402
from rl.agent import TeamPolicy  # noqa: E402
from train import load_obs_module  # noqa: E402

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "test_eval")

CONDITIONS = {  # run-name -> observation module
    "full": "full",
    "ablation_no_shaping": "full",
    "ablation_no_teammate": "no_teammate",
}
N_BOOT = 10_000


def bootstrap_ci(values: List[float], n_boot: int = N_BOOT, seed: int = 0) -> List[float]:
    """95% CI of the mean, resampling whole training runs (seeds) with replacement."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def aggregate(per_seed: Dict[int, Dict]) -> Dict:
    out = {}
    for key in ("capture_rate", "mean_time_penalized", "median_time_captured"):
        vals = [s[key] for s in per_seed.values()]
        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "ci95": bootstrap_ci(vals),
        }
    return out


def evaluate_condition(run_name: str, prey_random_prob: float, scenarios, suffix: str = "") -> Dict:
    obs_mod = load_obs_module(CONDITIONS[run_name])
    run_name = run_name + suffix
    per_seed, outcomes = {}, {}
    for seed in config.TRAIN_SEEDS:
        directory = os.path.join(RESULTS_DIR, run_name, "best", f"seed_{seed}")
        team = TeamPolicy.from_dir(directory, obs_mod.encode_team_observations, obs_mod.NUM_OBSERVATIONS)
        results = evaluate(team, scenarios, prey_random_prob=prey_random_prob)
        per_seed[seed] = summarize(results)
        outcomes[seed] = [[int(r["captured"]), r["steps_to_capture"]] for r in results]
        print(f"  {run_name} seed {seed} @ p={prey_random_prob}: "
              f"capture={per_seed[seed]['capture_rate']:.3f} "
              f"time={per_seed[seed]['mean_time_penalized']:.1f}", flush=True)
    return {
        "run_name": run_name,
        "prey_random_prob": prey_random_prob,
        "scenario_file": os.path.relpath(TEST_PATH, _REPO_ROOT).replace("\\", "/"),
        "scenario_count": len(scenarios),
        "outcome_format": "per seed: list over scenarios of [captured(0/1), steps_to_capture (300 if failed)]",
        "across_seeds": aggregate(per_seed),
        "per_seed_summary": {str(s): v for s, v in per_seed.items()},
        "per_seed_outcomes": {str(s): v for s, v in outcomes.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all seeds on the held-out test scenarios.")
    parser.add_argument("--suffix", default="", help="run-name suffix, e.g. _30k for results/full_30k")
    args = parser.parse_args()
    scenarios = load_scenarios(TEST_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [(name, 0.1) for name in CONDITIONS] + [("full", 0.0), ("full", 0.2)]
    for name, p in jobs:
        payload = evaluate_condition(name, p, scenarios, args.suffix)
        path = os.path.join(OUT_DIR, f"{name}{args.suffix}_prey_random_{p}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.write("\n")
        a = payload["across_seeds"]
        print(f"=> {name} p={p}: capture {a['capture_rate']['mean']:.3f} "
              f"CI{[round(x, 3) for x in a['capture_rate']['ci95']]}, "
              f"time {a['mean_time_penalized']['mean']:.1f} "
              f"CI{[round(x, 1) for x in a['mean_time_penalized']['ci95']]}", flush=True)


if __name__ == "__main__":
    main()
