"""
viz/generate_demo.py
====================
Auto-selects representative evaluation scenarios from the 200 held-out test
set and generates high-clarity side-by-side comparison GIFs under results/gifs/.

Task T-17b (Team 25 Group A).
Runnable via: python -m viz.generate_demo
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from rl.agent import TeamPolicy
from rl.observation import encode_team_observations
from baselines.direct_pursuit import DirectPursuit
from baselines.predictive_interception import PredictiveInterception
from baselines.coordinated_pursuit import CoordinatedPursuit
from experiments.generate_scenarios import load_scenarios, TEST_PATH
from experiments.evaluate import run_scenario
from viz.renderer import side_by_side_replay, save_replay_gif

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
GIFS_DIR = os.path.join(RESULTS_DIR, "gifs")


def find_best_checkpoint() -> Tuple[int, str]:
    """Find the best full-model checkpoint based on validation success and mean time."""
    best_dir = os.path.join(RESULTS_DIR, "full", "best")
    candidates = []
    for seed in config.TRAIN_SEEDS:
        ckpt_dir = os.path.join(best_dir, f"seed_{seed}")
        info_path = os.path.join(ckpt_dir, "info.json")
        if os.path.isfile(info_path):
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            candidates.append((
                float(info.get("best_val_success_rate", 0.0)),
                -float(info.get("best_val_mean_time", 999.0)),
                seed,
                ckpt_dir,
            ))
    if not candidates:
        fallback = os.path.join(best_dir, "seed_4")
        return 4, fallback
    winner = max(candidates)
    return winner[2], winner[3]


def load_evaluation_data() -> Tuple[Dict, Dict]:
    """Load baseline and learned evaluation logs across the 200 test scenarios."""
    baseline_eval_path = os.path.join(RESULTS_DIR, "baseline_eval.json")
    learned_eval_path = os.path.join(RESULTS_DIR, "robustness", "learned_prey_random_0.1.json")

    with open(baseline_eval_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    with open(learned_eval_path, "r", encoding="utf-8") as f:
        learned_data = json.load(f)

    return baseline_data, learned_data


def auto_select_scenarios(learned_results: List[Dict], baseline_data: Dict) -> Dict[str, int]:
    """
    Select representative scenario indices dynamically:
    1. Fastest capture (best)
    2. Median capture (typical)
    3. Slowest successful capture (worst)
    4. Timeout failure (failure case, or longest chase if 100% captured)
    5. Coordinated flank / interception win (clear advantage over CoordinatedPursuit)
    6. Direct pursuit comparison win
    """
    captured = [r for r in learned_results if r["captured"]]
    timeouts = [r for r in learned_results if not r["captured"]]

    # 1. Fastest capture
    best_r = min(captured, key=lambda x: x["steps_to_capture"])
    best_sc = best_r["scenario"]

    # 2. Slowest capture
    worst_r = max(captured, key=lambda x: x["steps_to_capture"])
    worst_sc = worst_r["scenario"]

    # 3. Median capture
    sorted_caps = sorted(captured, key=lambda x: x["steps_to_capture"])
    typical_r = sorted_caps[len(sorted_caps) // 2]
    typical_sc = typical_r["scenario"]

    # 4. Timeout failure (or fallback to hardest/slowest chase)
    timeout_sc = timeouts[0]["scenario"] if timeouts else worst_sc

    # 5. Clear flanking win against Coordinated Pursuit (maximum advantage)
    cp_results = {r["scenario"]: r["steps_to_capture"] for r in baseline_data["coordinated_pursuit"]}
    flank_candidates = [
        r for r in captured
        if r["scenario"] in cp_results and cp_results[r["scenario"]] > r["steps_to_capture"]
    ]
    if flank_candidates:
        flank_r = max(flank_candidates, key=lambda r: cp_results[r["scenario"]] - r["steps_to_capture"])
        flank_sc = flank_r["scenario"]
    else:
        flank_sc = best_sc

    # 6. Direct pursuit comparison win (maximum advantage)
    dp_results = {r["scenario"]: r["steps_to_capture"] for r in baseline_data["direct_pursuit"]}
    dp_candidates = [
        r for r in captured
        if r["scenario"] in dp_results and dp_results[r["scenario"]] > r["steps_to_capture"]
    ]
    if dp_candidates:
        dp_r = max(dp_candidates, key=lambda r: dp_results[r["scenario"]] - r["steps_to_capture"])
        dp_sc = dp_r["scenario"]
    else:
        dp_sc = best_sc

    return {
        "best": best_sc,
        "typical": typical_sc,
        "worst": worst_sc,
        "timeout": timeout_sc,
        "flank": flank_sc,
        "direct_comp": dp_sc,
    }


def _stride_sync(traj_l: list, traj_r: list, max_len: int = 120, stride: int = 3) -> Tuple[list, list]:
    """Stride both trajectories together if episode is long to optimize GIF render time and file size."""
    if max(len(traj_l), len(traj_r)) > max_len:
        sl = traj_l[::stride]
        if traj_l[-1] != sl[-1]:
            sl.append(traj_l[-1])
        sr = traj_r[::stride]
        if traj_r[-1] != sr[-1]:
            sr.append(traj_r[-1])
        return sl, sr
    return traj_l, traj_r


def generate_all_demo_gifs() -> List[Tuple[str, int]]:
    """Generate the full suite of 5-6 demo GIFs."""
    os.makedirs(GIFS_DIR, exist_ok=True)
    t_start = time.time()

    print("=== Generating Demo Replay GIFs (Task T-17b) ===")

    # 1. Load checkpoints and scenarios
    seed, ckpt_dir = find_best_checkpoint()
    print(f"Loading best learned model: Seed {seed} from {os.path.relpath(ckpt_dir, _REPO_ROOT)}")
    learned_policy = TeamPolicy.from_dir(ckpt_dir, encode_team_observations, config.NUM_OBSERVATIONS)

    test_scenarios = load_scenarios(TEST_PATH)
    baseline_data, learned_data = load_evaluation_data()
    learned_results = learned_data["results"]

    # 2. Auto-select scenarios
    selection = auto_select_scenarios(learned_results, baseline_data)
    print("Auto-selected scenario indices:")
    for k, v in selection.items():
        print(f"  - {k}: Scenario {v}")

    # Helper to get baseline trajectory from baseline_eval.json
    def get_baseline_traj(baseline_name: str, scenario_idx: int):
        entry = [r for r in baseline_data[baseline_name] if r["scenario"] == scenario_idx][0]
        return entry["trajectory"]

    generated = []

    # 1. best.gif (Fastest capture)
    sc_idx = selection["best"]
    sc = test_scenarios[sc_idx]
    traj_learned = run_scenario(learned_policy, sc, record=True)["trajectory"]
    traj_baseline = get_baseline_traj("coordinated_pursuit", sc_idx)
    steps_l = len(traj_learned) - 1
    steps_r = len(traj_baseline) - 1
    out_path = os.path.join(GIFS_DIR, "best.gif")
    print(f"\n[1/6] Generating {os.path.basename(out_path)} (Scenario {sc_idx}: {steps_l} vs {steps_r} steps)...")
    side_by_side_replay(
        traj_learned, traj_baseline,
        label_l=f"Learned RL ({steps_l} steps)",
        label_r=f"Coordinated Pursuit ({steps_r} steps)",
        filename=out_path,
        fps=4,
    )
    generated.append((out_path, os.path.getsize(out_path)))

    # 2. typical.gif (Median capture)
    sc_idx = selection["typical"]
    sc = test_scenarios[sc_idx]
    traj_learned = run_scenario(learned_policy, sc, record=True)["trajectory"]
    traj_baseline = get_baseline_traj("coordinated_pursuit", sc_idx)
    steps_l = len(traj_learned) - 1
    steps_r = len(traj_baseline) - 1
    out_path = os.path.join(GIFS_DIR, "typical.gif")
    print(f"[2/6] Generating {os.path.basename(out_path)} (Scenario {sc_idx}: {steps_l} vs {steps_r} steps)...")
    side_by_side_replay(
        traj_learned, traj_baseline,
        label_l=f"Learned RL ({steps_l} steps)",
        label_r=f"Coordinated Pursuit ({steps_r} steps)",
        filename=out_path,
        fps=6,
    )
    generated.append((out_path, os.path.getsize(out_path)))

    # 3. worst.gif (Slowest successful capture)
    sc_idx = selection["worst"]
    sc = test_scenarios[sc_idx]
    traj_learned = run_scenario(learned_policy, sc, record=True)["trajectory"]
    traj_baseline = get_baseline_traj("coordinated_pursuit", sc_idx)
    steps_l = len(traj_learned) - 1
    steps_r = len(traj_baseline) - 1
    s_learned, s_baseline = _stride_sync(traj_learned, traj_baseline, max_len=120, stride=3)
    out_path = os.path.join(GIFS_DIR, "worst.gif")
    print(f"[3/6] Generating {os.path.basename(out_path)} (Scenario {sc_idx}: {steps_l} vs {steps_r} steps)...")
    side_by_side_replay(
        s_learned, s_baseline,
        label_l=f"Learned RL ({steps_l} steps)",
        label_r=f"Coordinated Pursuit ({steps_r} steps)",
        filename=out_path,
        fps=6,
    )
    generated.append((out_path, os.path.getsize(out_path)))

    # 4. timeout_failure.gif (300-step failure case or longest chase)
    sc_idx = selection["timeout"]
    sc = test_scenarios[sc_idx]
    traj_learned = run_scenario(learned_policy, sc, record=True)["trajectory"]
    traj_baseline = get_baseline_traj("coordinated_pursuit", sc_idx)
    steps_l = len(traj_learned) - 1
    steps_r = len(traj_baseline) - 1
    s_learned, s_baseline = _stride_sync(traj_learned, traj_baseline, max_len=120, stride=3)
    out_path = os.path.join(GIFS_DIR, "timeout_failure.gif")
    lbl_l = f"Learned RL (Timeout: {steps_l} steps)" if steps_l >= config.MAX_STEPS else f"Learned RL ({steps_l} steps)"
    print(f"[4/6] Generating {os.path.basename(out_path)} (Scenario {sc_idx}: {steps_l} vs {steps_r} steps)...")
    side_by_side_replay(
        s_learned, s_baseline,
        label_l=lbl_l,
        label_r=f"Coordinated Pursuit ({steps_r} steps)",
        filename=out_path,
        fps=6,
    )
    generated.append((out_path, os.path.getsize(out_path)))

    # 5. coordinated_flank_demo.gif (Flanking win)
    sc_idx = selection["flank"]
    sc = test_scenarios[sc_idx]
    traj_learned = run_scenario(learned_policy, sc, record=True)["trajectory"]
    traj_baseline = get_baseline_traj("coordinated_pursuit", sc_idx)
    steps_l = len(traj_learned) - 1
    steps_r = len(traj_baseline) - 1
    s_learned, s_baseline = _stride_sync(traj_learned, traj_baseline, max_len=120, stride=2)
    out_path = os.path.join(GIFS_DIR, "coordinated_flank_demo.gif")
    print(f"[5/6] Generating {os.path.basename(out_path)} (Scenario {sc_idx}: {steps_l} vs {steps_r} steps)...")
    side_by_side_replay(
        s_learned, s_baseline,
        label_l=f"Learned RL ({steps_l} steps)",
        label_r=f"Coordinated Pursuit ({steps_r} steps)",
        filename=out_path,
        fps=5,
    )
    generated.append((out_path, os.path.getsize(out_path)))

    # 6. learned_vs_direct_pursuit.gif (Direct pursuit comparison)
    sc_idx = selection["direct_comp"]
    sc = test_scenarios[sc_idx]
    traj_learned = run_scenario(learned_policy, sc, record=True)["trajectory"]
    traj_baseline = get_baseline_traj("direct_pursuit", sc_idx)
    steps_l = len(traj_learned) - 1
    steps_r = len(traj_baseline) - 1
    s_learned, s_baseline = _stride_sync(traj_learned, traj_baseline, max_len=120, stride=2)
    out_path = os.path.join(GIFS_DIR, "learned_vs_direct_pursuit.gif")
    print(f"[6/6] Generating {os.path.basename(out_path)} (Scenario {sc_idx}: {steps_l} vs {steps_r} steps)...")
    side_by_side_replay(
        s_learned, s_baseline,
        label_l=f"Learned RL ({steps_l} steps)",
        label_r=f"Direct Pursuit ({steps_r} steps)",
        filename=out_path,
        fps=5,
    )
    generated.append((out_path, os.path.getsize(out_path)))

    elapsed = time.time() - t_start
    print(f"\nSuccessfully generated {len(generated)} demo GIFs in {elapsed:.1f}s.")
    return generated


if __name__ == "__main__":
    results = generate_all_demo_gifs()
    for path, size in results:
        print(f"  {os.path.basename(path):<30} {size / 1024:>7.1f} KB  ({path})")
