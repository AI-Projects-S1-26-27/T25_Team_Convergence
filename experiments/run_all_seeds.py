"""
experiments/run_all_seeds.py  (Task T-11, runner part)
======================================================
Train seeds 0-9 for one experimental condition, in parallel.

    python experiments/run_all_seeds.py                       # full model
    python experiments/run_all_seeds.py --no-shaping          # ablation B
    python experiments/run_all_seeds.py --obs no_teammate     # ablation A
    python experiments/run_all_seeds.py --workers 4 --seeds 0 1 2   # subset

Each seed calls train.train(...) in its own process; results land in
results/<run-name>/ exactly as with `python train.py --seed N`.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from train import OBS_MODULES, default_run_name, train  # noqa: E402


def _run(kwargs):
    return kwargs["seed"], train(**kwargs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seeds", type=int, nargs="+", default=config.TRAIN_SEEDS)
    p.add_argument("--no-shaping", action="store_true")
    p.add_argument("--obs", choices=sorted(OBS_MODULES), default="full")
    p.add_argument("--run-name", default=None)
    p.add_argument("--episodes", type=int, default=config.TOTAL_EPISODES)
    p.add_argument("--checkpoint-interval", type=int, default=config.CHECKPOINT_INTERVAL)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--results-dir", default=None)
    args = p.parse_args(argv)

    shaped = not args.no_shaping
    run_name = args.run_name or default_run_name(args.obs, shaped)
    jobs = [
        dict(seed=s, shaped=shaped, obs_name=args.obs, run_name=run_name, episodes=args.episodes,
             checkpoint_interval=args.checkpoint_interval, results_dir=args.results_dir, verbose=False)
        for s in args.seeds
    ]
    print(f"Training {len(jobs)} seeds for '{run_name}' with {args.workers} workers...", flush=True)

    summary = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, j) for j in jobs]
        for fut in as_completed(futures):
            seed, res = fut.result()
            summary[seed] = res
            print(f"  seed {seed} done: best ep {res['best_episode']}, "
                  f"val_rate={res['best_val_rate']:.2f}, {res['seconds']:.0f}s", flush=True)

    rates = [summary[s]["best_val_rate"] for s in sorted(summary)]
    print(f"Mean best val success over {len(rates)} seeds: {sum(rates) / len(rates):.3f}")


if __name__ == "__main__":
    main()
