"""
experiments/tune_budget.py
==========================
Pick the training budget / exploration schedule on VALIDATION scenarios only.

Trains a small grid of settings on a few seeds (full model: shaped reward,
teammate features) and ranks them by validation success. The test set is never
touched. Output: results/tuning/<setting>/ and results/tuning/summary.json.

    python experiments/tune_budget.py
    python experiments/tune_budget.py --seeds 0 1 2 --workers 9
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402
from train import train  # noqa: E402

TUNING_DIR = os.path.join(_REPO_ROOT, "results", "tuning")

# name -> (episodes, eps_decay_frac). 20 checkpoints per run in every case.
GRID = {
    "baseline_10k_decay0.8": (10_000, 0.8),
    "longer_30k_decay0.8": (30_000, 0.8),
    "faster_10k_decay0.4": (10_000, 0.4),
    "both_30k_decay0.4": (30_000, 0.4),
}


def _run(job):
    name, seed, episodes, frac = job
    train(
        seed=seed, run_name=name, episodes=episodes, checkpoint_interval=episodes // 20,
        eps_decay_frac=frac, results_dir=TUNING_DIR, verbose=False,
    )
    return name, seed


def score(name: str, seeds) -> dict:
    """Validation-only scores. 'last3' averages the final 3 checkpoints (not best-of-20, so less optimistic)."""
    last3, best, last3_time = [], [], []
    for s in seeds:
        rows = list(csv.DictReader(open(os.path.join(TUNING_DIR, name, "logs", f"seed_{s}.csv"))))
        rate = [float(r["val_success_rate"]) for r in rows]
        time_ = [float(r["val_mean_time"]) for r in rows]
        last3.append(np.mean(rate[-3:]))
        last3_time.append(np.mean(time_[-3:]))
        best.append(max(rate))
    return {
        "val_success_last3": float(np.mean(last3)),
        "val_success_best": float(np.mean(best)),
        "val_mean_time_last3": float(np.mean(last3_time)),
        "per_seed_last3": [float(x) for x in last3],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()

    jobs = [(n, s, ep, fr) for n, (ep, fr) in GRID.items() for s in args.seeds]
    print(f"{len(jobs)} runs ({len(GRID)} settings x {len(args.seeds)} seeds), {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_run, j) for j in jobs]):
            n, s = fut.result()
            print(f"  done {n} seed {s}", flush=True)

    summary = {n: {"episodes": GRID[n][0], "eps_decay_frac": GRID[n][1], **score(n, args.seeds)} for n in GRID}
    with open(os.path.join(TUNING_DIR, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "selection_metric": "val_success_last3 (validation scenarios only)",
                   "settings": summary}, f, indent=2)
    print("\nsetting                  episodes decay  val_success(last3)  val_success(best)  val_time(last3)")
    for n, r in sorted(summary.items(), key=lambda kv: -kv[1]["val_success_last3"]):
        print(f"{n:24s} {r['episodes']:7d} {r['eps_decay_frac']:5.1f}  {r['val_success_last3']:17.3f}  "
              f"{r['val_success_best']:17.3f}  {r['val_mean_time_last3']:14.1f}")


if __name__ == "__main__":
    main()
