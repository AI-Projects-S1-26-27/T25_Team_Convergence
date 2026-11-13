"""Run random-agent and robustness evaluations on the shared test scenarios.

Examples:
    python experiments/run_remaining_evaluations.py --mode random
    python experiments/run_remaining_evaluations.py --mode robustness
    python experiments/run_remaining_evaluations.py --mode all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from baselines.coordinated_pursuit import CoordinatedPursuitPolicy  # noqa: E402
from experiments.evaluate import RandomPolicy, evaluate, summarize  # noqa: E402
from experiments.generate_scenarios import TEST_PATH, load_scenarios  # noqa: E402
from rl.agent import TeamPolicy  # noqa: E402
from rl.observation import encode_team_observations  # noqa: E402

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")


def _write_results(path: str, policy_name: str, probability: float, scenarios: Sequence[dict], results: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "policy": policy_name,
        "prey_random_prob": probability,
        "scenario_file": os.path.relpath(TEST_PATH, _REPO_ROOT).replace(os.sep, "/"),
        "scenario_count": len(scenarios),
        "summary": summarize(results),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"{path}: capture={payload['summary']['capture_rate']:.3f}, "
          f"penalized_mean_time={payload['summary']['mean_time_penalized']:.2f}")


def _best_full_checkpoint() -> tuple[int, str]:
    candidates = []
    for seed in config.TRAIN_SEEDS:
        directory = os.path.join(RESULTS_DIR, "full", "best", f"seed_{seed}")
        info_path = os.path.join(directory, "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        candidates.append((
            float(info["best_val_success_rate"]),
            -float(info["best_val_mean_time"]),
            -seed,
            seed,
            directory,
        ))
    if not candidates:
        raise FileNotFoundError("No full-model best checkpoints found under results/full/best/")
    winner = max(candidates)
    return winner[3], winner[4]


def run_random(scenarios: Sequence[dict]) -> None:
    results = evaluate(RandomPolicy(seed=0), scenarios)
    _write_results(
        os.path.join(RESULTS_DIR, "ablation_random", "random_eval.json"),
        "random_agent", config.PREY_RANDOM_PROB, scenarios, results,
    )


def run_robustness(scenarios: Sequence[dict]) -> None:
    seed, checkpoint = _best_full_checkpoint()
    learned = TeamPolicy.from_dir(checkpoint, encode_team_observations, config.NUM_OBSERVATIONS)
    print(f"Using full-model seed {seed} from {os.path.relpath(checkpoint, _REPO_ROOT)}")
    for probability in (0.0, 0.1, 0.2):
        for name, policy in (
            ("learned", learned),
            ("coordinated_pursuit", CoordinatedPursuitPolicy(seed=0)),
        ):
            results = evaluate(policy, scenarios, prey_random_prob=probability)
            path = os.path.join(
                RESULTS_DIR, "robustness", f"{name}_prey_random_{probability:.1f}.json"
            )
            _write_results(path, name, probability, scenarios, results)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("random", "robustness", "all"), default="all")
    args = parser.parse_args(argv)
    scenarios = load_scenarios(TEST_PATH)
    if len(scenarios) != config.NUM_TEST_SCENARIOS:
        raise ValueError(f"expected {config.NUM_TEST_SCENARIOS} test scenarios, got {len(scenarios)}")
    if args.mode in ("random", "all"):
        run_random(scenarios)
    if args.mode in ("robustness", "all"):
        run_robustness(scenarios)


if __name__ == "__main__":
    main()
