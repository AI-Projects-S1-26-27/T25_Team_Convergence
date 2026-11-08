"""
experiments/evaluate.py
=======================
Shared evaluator: run any predator team on matched scenarios.

A *policy* is any callable `policy(env_obs) -> [a0, a1, a2]`, where `env_obs` is
the dict returned by Environment.reset()/step() (keys 'predators', 'prey',
'prev_prey', 'step'). Optionally it may have `reset(seed)`, called at the start
of every scenario so stochastic policies are reproducible.

    results = evaluate(policy, scenarios)                       # prey_random_prob = 0.1
    results = evaluate(policy, scenarios, prey_random_prob=0.2) # robustness test
    summarize(results)  -> {'capture_rate', 'mean_time_penalized', 'median_time_captured', ...}

Each result dict: {'scenario', 'captured', 'steps_to_capture', 'capture_step'}
where steps_to_capture is MAX_STEPS (300) for a timeout.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from agents.prey import prey_action  # noqa: E402
from env.environment import Environment  # noqa: E402

Policy = Callable[[dict], Sequence[int]]


class RandomPolicy:
    """Uniform random actions every step (ablation C / sanity baseline)."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng([self._seed, 0 if seed is None else seed])

    def __call__(self, env_obs: dict) -> List[int]:
        return [int(a) for a in self.rng.integers(0, config.NUM_ACTIONS, size=config.NUM_PREDATORS)]


def run_scenario(
    policy: Policy,
    scenario: Dict,
    prey_random_prob: float = config.PREY_RANDOM_PROB,
    env: Optional[Environment] = None,
    record: bool = False,
) -> Dict:
    """Play one scenario to the end. Set record=True to also return the trajectory."""
    if env is None:
        env = Environment(seed=scenario["seed"])
    if hasattr(policy, "reset"):
        policy.reset(scenario["seed"])

    obs = env.reset(positions=scenario["init_positions"])
    trajectory = [(list(obs["predators"]), obs["prey"])] if record else None
    info = {"capture": False, "capture_step": None}
    done = False

    while not done:
        t = obs["step"]
        pred_actions = policy(obs)
        prey_a = prey_action(
            obs["prey"],
            obs["predators"],
            random_prob=prey_random_prob,
            pre_sampled_rand=scenario["prey_randoms"][t],
            pre_sampled_choice=scenario["prey_choices"][t],
        )
        obs, done, info = env.step(pred_actions, prey_a)
        if record:
            trajectory.append((list(obs["predators"]), obs["prey"]))

    captured = bool(info["capture"])
    result = {
        "scenario": scenario.get("index"),
        "captured": captured,
        "steps_to_capture": int(info["capture_step"]) if captured else config.MAX_STEPS,
        "capture_step": info["capture_step"],
    }
    if record:
        result["trajectory"] = trajectory
    return result


def evaluate(
    policy: Policy,
    scenarios: Sequence[Dict],
    prey_random_prob: float = config.PREY_RANDOM_PROB,
    record: bool = False,
) -> List[Dict]:
    """Run `policy` on every scenario (same Environment object reused)."""
    env = Environment(seed=0)
    return [run_scenario(policy, sc, prey_random_prob, env, record) for sc in scenarios]


def summarize(results: Sequence[Dict]) -> Dict:
    """Primary outcome measures from the proposal."""
    n = len(results)
    captured = [r for r in results if r["captured"]]
    times = [r["steps_to_capture"] for r in results]
    return {
        "n": n,
        "capture_rate": len(captured) / n if n else 0.0,
        "mean_time_penalized": float(np.mean(times)) if n else float("nan"),
        "median_time_captured": (
            float(np.median([r["steps_to_capture"] for r in captured])) if captured else float("nan")
        ),
    }
