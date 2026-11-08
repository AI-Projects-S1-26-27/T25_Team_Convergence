"""
train.py  (Task T-10)
=====================
Train three independent tabular Q-learning predators.

    python train.py --seed 0                    # full model (shaped reward, teammate features)
    python train.py --seed 0 --no-shaping       # ablation B
    python train.py --seed 0 --obs no_teammate --run-name ablation_no_teammate   # ablation A

Output (under results/<run-name>/):
    logs/seed_<s>.csv                        episode, epsilon, val_success_rate (+ extra columns)
    checkpoints/seed_<s>/ep_<n>/predator_<i>.npy   every --checkpoint-interval episodes
    best/seed_<s>/predator_<i>.npy           checkpoint with the best validation success rate
    best/seed_<s>/info.json                  which episode was best and its validation numbers

Validation uses experiments/val_scenarios.pkl (disjoint from the test set) with
epsilon = 0. Checkpoints are chosen on validation only.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import time
from typing import Dict, Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from agents.prey import prey_action  # noqa: E402
from env.environment import Environment  # noqa: E402
from experiments.evaluate import evaluate, summarize  # noqa: E402
from experiments.generate_scenarios import VAL_PATH, load_scenarios  # noqa: E402
from rl.agent import PredatorAgent, TeamPolicy, get_epsilon, save_team  # noqa: E402
from rl.rewards import base_reward, shaped_reward  # noqa: E402

OBS_MODULES = {
    "full": "rl.observation",
    "no_teammate": "rl.observation_no_teammate",
}


def load_obs_module(name: str):
    """Observation module exposing NUM_OBSERVATIONS and encode_team_observations(env_obs)."""
    mod = importlib.import_module(OBS_MODULES[name])
    if not hasattr(mod, "NUM_OBSERVATIONS"):
        mod.NUM_OBSERVATIONS = config.NUM_OBSERVATIONS
    return mod


def train(
    seed: int,
    shaped: bool = True,
    obs_name: str = "full",
    run_name: str = "full",
    episodes: int = config.TOTAL_EPISODES,
    checkpoint_interval: int = config.CHECKPOINT_INTERVAL,
    alpha: float = config.ALPHA,
    gamma: float = config.GAMMA,
    results_dir: Optional[str] = None,
    val_path: str = VAL_PATH,
    verbose: bool = True,
) -> Dict:
    """Train one seed end to end. Returns {'best_episode', 'best_val_rate', 'seconds'}."""
    results_dir = results_dir or os.path.join(_REPO_ROOT, "results")
    root = os.path.join(results_dir, run_name)
    log_path = os.path.join(root, "logs", f"seed_{seed}.csv")
    ckpt_root = os.path.join(root, "checkpoints", f"seed_{seed}")
    best_dir = os.path.join(root, "best", f"seed_{seed}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    obs_mod = load_obs_module(obs_name)
    encode_team = obs_mod.encode_team_observations
    n_states = obs_mod.NUM_OBSERVATIONS

    # Independent, reproducible random streams: env, prey, and one per predator.
    env_ss, prey_ss, *agent_ss = np.random.SeedSequence(seed).spawn(2 + config.NUM_PREDATORS)
    env = Environment(seed=int(env_ss.generate_state(1)[0]))
    prey_rng = np.random.default_rng(prey_ss)
    agents = [
        PredatorAgent(n_states=n_states, seed=int(ss.generate_state(1)[0])) for ss in agent_ss
    ]

    val_scenarios = load_scenarios(val_path)
    team = TeamPolicy(agents, encode_team)

    best = {"rate": -1.0, "time": float("inf"), "episode": 0}
    window_caps = 0
    window_n = 0
    t0 = time.time()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "epsilon", "val_success_rate", "val_mean_time", "train_capture_rate", "visited_states"]
        )

        for ep in range(episodes):
            epsilon = get_epsilon(ep, episodes)
            obs = env.reset()
            states = encode_team(obs)
            done = False
            info: Dict = {}

            while not done:
                actions = [agents[i].act(states[i], epsilon) for i in range(config.NUM_PREDATORS)]
                prey_a = prey_action(obs["prey"], obs["predators"], rng=prey_rng)
                next_obs, done, info = env.step(actions, prey_a)
                capture = info["capture"]

                if shaped:
                    reward = shaped_reward(
                        obs["predators"], obs["prey"], capture,
                        next_obs["predators"], next_obs["prey"], gamma=gamma, done=done,
                    )
                else:
                    reward = base_reward(capture, next_obs["step"])

                next_states = encode_team(next_obs)
                for i in range(config.NUM_PREDATORS):
                    agents[i].update(states[i], actions[i], reward, next_states[i], done, alpha, gamma)

                obs, states = next_obs, next_states

            window_caps += int(info["capture"])
            window_n += 1

            done_eps = ep + 1
            if done_eps % checkpoint_interval == 0 or done_eps == episodes:
                eps_now = get_epsilon(done_eps, episodes)
                val = summarize(evaluate(team, val_scenarios))
                visited = int(np.mean([a.visited_states() for a in agents]))
                writer.writerow(
                    [done_eps, f"{eps_now:.4f}", f"{val['capture_rate']:.4f}",
                     f"{val['mean_time_penalized']:.2f}", f"{window_caps / window_n:.4f}", visited]
                )
                f.flush()
                save_team(agents, os.path.join(ckpt_root, f"ep_{done_eps}"))

                if (val["capture_rate"], -val["mean_time_penalized"]) > (best["rate"], -best["time"]):
                    best = {"rate": val["capture_rate"], "time": val["mean_time_penalized"], "episode": done_eps}

                if verbose:
                    print(
                        f"[seed {seed}] Ep {done_eps}/{episodes} | eps={eps_now:.2f} | "
                        f"val_rate={val['capture_rate']:.2f} | train_rate={window_caps / window_n:.2f} | "
                        f"{time.time() - t0:.0f}s",
                        flush=True,
                    )
                window_caps = 0
                window_n = 0

    # Best checkpoint (validation only).
    if os.path.isdir(best_dir):
        shutil.rmtree(best_dir)
    shutil.copytree(os.path.join(ckpt_root, f"ep_{best['episode']}"), best_dir)
    seconds = time.time() - t0
    with open(os.path.join(best_dir, "info.json"), "w") as f:
        json.dump(
            {
                "seed": seed, "run_name": run_name, "obs": obs_name, "shaped": shaped,
                "episodes": episodes, "best_episode": best["episode"],
                "best_val_success_rate": best["rate"], "best_val_mean_time": best["time"],
                "alpha": alpha, "gamma": gamma, "train_seconds": round(seconds, 1),
            },
            f, indent=2,
        )
    if verbose:
        print(f"[seed {seed}] best checkpoint: ep {best['episode']} (val_rate={best['rate']:.2f})", flush=True)
    return {"best_episode": best["episode"], "best_val_rate": best["rate"], "seconds": seconds}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train independent Q-learning predators.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shaped", action="store_true", help="potential-based shaping (default; accepted for clarity)")
    p.add_argument("--no-shaping", action="store_true", help="ablation B: base reward only")
    p.add_argument("--obs", choices=sorted(OBS_MODULES), default="full",
                   help="'no_teammate' = ablation A observation")
    p.add_argument("--run-name", default=None,
                   help="results/<run-name>/ (default: full, ablation_no_shaping or ablation_no_teammate)")
    p.add_argument("--episodes", type=int, default=config.TOTAL_EPISODES)
    p.add_argument("--checkpoint-interval", type=int, default=config.CHECKPOINT_INTERVAL)
    p.add_argument("--alpha", type=float, default=config.ALPHA)
    p.add_argument("--gamma", type=float, default=config.GAMMA)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def default_run_name(obs_name: str, shaped: bool) -> str:
    if obs_name == "no_teammate":
        return "ablation_no_teammate"
    return "full" if shaped else "ablation_no_shaping"


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.shaped and args.no_shaping:
        raise SystemExit("choose either --shaped or --no-shaping, not both")
    shaped = not args.no_shaping
    train(
        seed=args.seed, shaped=shaped, obs_name=args.obs,
        run_name=args.run_name or default_run_name(args.obs, shaped),
        episodes=args.episodes, checkpoint_interval=args.checkpoint_interval,
        alpha=args.alpha, gamma=args.gamma, results_dir=args.results_dir, verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
