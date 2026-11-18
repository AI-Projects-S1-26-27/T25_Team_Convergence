"""Held-out analysis for Team Convergence: evaluates every saved policy on the
200 test scenarios and computes outcome + coordination metrics (proposal Sec. 5).
Usage: python analyze.py <repo_root> <out_dir>
"""
import json, math, os, sys, pickle, csv
import numpy as np

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
os.makedirs(OUT, exist_ok=True)

import config
from experiments.evaluate import evaluate, summarize, RandomPolicy
from experiments.generate_scenarios import load_scenarios, TEST_PATH
from rl.agent import TeamPolicy
import rl.observation as obs_full
import rl.observation_no_teammate as obs_nt
from baselines.direct_pursuit import DirectPursuit
from baselines.predictive_interception import PredictiveInterception
from baselines.coordinated_pursuit import CoordinatedPursuit
from agents.prey import get_legal_moves

G = config.GRID_SIZE
scen = load_scenarios(TEST_PATH)
assert len(scen) == 200

RUNS = {
    "full": (obs_full.encode_team_observations, config.NUM_OBSERVATIONS),
    "ablation_no_teammate": (obs_nt.encode_team_observations, obs_nt.NUM_OBSERVATIONS),
    "ablation_no_shaping": (obs_full.encode_team_observations, config.NUM_OBSERVATIONS),
}

# ---------------------------------------------------------------- metrics

def manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def angle(u, v):
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu == 0 or nv == 0:
        return None
    c = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))

def flank_step(preds, prey):
    vs = [(p[0] - prey[0], p[1] - prey[1]) for p in preds if 0 < manh(p, prey) <= 6]
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a = angle(vs[i], vs[j])
            if a is not None and a >= 120.0:
                return True
    return False

def coverage(old_preds, new_preds, prey):
    """Fraction of legal prey moves from `prey` that the predators' actual next
    moves would capture (same cell or edge swap)."""
    legal = get_legal_moves(tuple(prey))
    blocked = 0
    for a in legal:
        dr, dc = config.ACTIONS[a]
        cand = (prey[0] + dr, prey[1] + dc)
        hit = False
        for o, n in zip(old_preds, new_preds):
            if tuple(n) == cand or (tuple(o) == cand and tuple(n) == tuple(prey)):
                hit = True
                break
        blocked += hit
    return blocked / len(legal)

def clip(v):
    return max(0, min(G - 1, v))

def coord_metrics(traj):
    """traj: list of (preds, prey) from t=0..T. Returns per-episode metrics."""
    T = len(traj) - 1
    if T <= 0:
        return dict(flank_frac=0.0, flank_events=0, coverage=0.0, intercept_rate=0.0,
                    chase_rate=0.0, steps=0)
    flank = [flank_step(traj[t][0], traj[t][1]) for t in range(T)]
    # sustained flanking: runs of >=3 consecutive steps
    sustained = [False] * T
    events, run = 0, 0
    for t in range(T):
        run = run + 1 if flank[t] else 0
        if run == 3:
            events += 1
            for k in range(t - 2, t + 1):
                sustained[k] = True
        elif run > 3:
            sustained[t] = True
    cov = [coverage(traj[t][0], traj[t + 1][0], traj[t][1]) for t in range(T)]
    # interception vs direct chase (per predator-step, t>=1 so a projection exists)
    inter = chase = n = 0
    for t in range(1, T):
        preds, prey = traj[t]
        prev = traj[t - 1][1]
        proj = (clip(prey[0] + 2 * (prey[0] - prev[0])), clip(prey[1] + 2 * (prey[1] - prev[1])))
        nxt = traj[t + 1][0]
        future = cov[t + 1:t + 6]
        for p, q in zip(preds, nxt):
            n += 1
            d_cur0, d_cur1 = manh(p, prey), manh(q, prey)
            d_pr0, d_pr1 = manh(p, proj), manh(q, proj)
            if d_cur1 < d_cur0:
                chase += 1
            if tuple(proj) != tuple(prey) and d_pr1 < d_pr0 and d_cur1 >= d_cur0:
                if future and max(future) > cov[t]:
                    inter += 1
    return dict(
        flank_frac=sum(sustained) / T,
        flank_events=events,
        coverage=float(np.mean(cov)),
        intercept_rate=inter / n if n else 0.0,
        chase_rate=chase / n if n else 0.0,
        steps=T,
    )

def near_wall(c, m=2):
    return c[0] < m or c[0] >= G - m or c[1] < m or c[1] >= G - m

wall_start = np.array([any(near_wall(tuple(c)) for c in s["init_positions"]) for s in scen])

def episode_rows(results):
    rows = []
    for r in results:
        traj = r["trajectory"]
        m = coord_metrics(traj)
        final_prey = tuple(traj[-1][1])
        rows.append(dict(
            scenario=r["scenario"], captured=r["captured"], time=r["steps_to_capture"],
            capture_near_wall=bool(r["captured"] and near_wall(final_prey)), **m))
    return rows

# ---------------------------------------------------------------- evaluate
data = {}  # policy -> list (per seed) of list of episode rows

for cls in (DirectPursuit, PredictiveInterception, CoordinatedPursuit):
    res = evaluate(cls(seed=0), scen, record=True)
    data[cls.name] = [episode_rows(res)]
    print(cls.name, summarize(res), flush=True)

res = evaluate(RandomPolicy(seed=0), scen, record=True)
data["random"] = [episode_rows(res)]
print("random", summarize(res), flush=True)

infos = {}
for run, (enc, ns) in RUNS.items():
    data[run] = []
    infos[run] = []
    for s in range(10):
        d = os.path.join(REPO, "results", run, "best", f"seed_{s}")
        pol = TeamPolicy.from_dir(d, enc, ns)
        res = evaluate(pol, scen, record=True)
        data[run].append(episode_rows(res))
        infos[run].append(json.load(open(os.path.join(d, "info.json"))))
        sm = summarize(res)
        print(run, s, sm, flush=True)

# robustness: all seeds of full + all baselines at p in {0, .1, .2}
robust = {}
for p in (0.0, 0.1, 0.2):
    robust[p] = {}
    for cls in (DirectPursuit, PredictiveInterception, CoordinatedPursuit):
        robust[p][cls.name] = [summarize(evaluate(cls(seed=0), scen, prey_random_prob=p))]
    for run, (enc, ns) in RUNS.items():
        robust[p][run] = []
        for s in range(10):
            d = os.path.join(REPO, "results", run, "best", f"seed_{s}")
            robust[p][run].append(summarize(evaluate(TeamPolicy.from_dir(d, enc, ns), scen, prey_random_prob=p)))
    print("robust", p, {k: np.mean([x["capture_rate"] for x in v]) for k, v in robust[p].items()}, flush=True)

with open(os.path.join(OUT, "episodes.pkl"), "wb") as f:
    pickle.dump(dict(data=data, infos=infos, robust=robust, wall_start=wall_start), f)

# ---------------------------------------------------------------- statistics
rng = np.random.default_rng(12345)
B = 2000

def arr(policy, key):
    return np.array([[row[key] for row in seed_rows] for seed_rows in data[policy]], dtype=float)  # seeds x 200

def boot_ci(M, stat=np.mean):
    """M: seeds x scenarios. Resample seeds and scenarios jointly (proposal Sec. 5)."""
    S, N = M.shape
    vals = np.empty(B)
    for b in range(B):
        si = rng.integers(S, size=S) if S > 1 else np.zeros(1, int)
        ni = rng.integers(N, size=N)
        vals[b] = stat(M[np.ix_(si, ni)])
    return float(stat(M)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def median_captured(M_time, M_cap):
    return float(np.median(M_time[M_cap > 0])) if (M_cap > 0).any() else float("nan")

summary = {}
for pol in data:
    cap, tm = arr(pol, "captured"), arr(pol, "time")
    S = cap.shape[0]
    row = {
        "n_seeds": S,
        "capture_rate": boot_ci(cap),
        "mean_time_penalized": boot_ci(tm),
        "median_time_captured": median_captured(tm, cap),
        "seed_capture_rates": cap.mean(1).tolist(),
        "seed_mean_times": tm.mean(1).tolist(),
        "flank_frac": boot_ci(arr(pol, "flank_frac")),
        "flank_events": boot_ci(arr(pol, "flank_events")),
        "coverage": boot_ci(arr(pol, "coverage")),
        "intercept_rate": boot_ci(arr(pol, "intercept_rate")),
        "chase_rate": boot_ci(arr(pol, "chase_rate")),
        "capture_near_wall_frac": float(arr(pol, "capture_near_wall").sum() / max(1, cap.sum())),
        "wall_start": {
            "capture_rate": float(cap[:, wall_start].mean()), "mean_time": float(tm[:, wall_start].mean())},
        "central_start": {
            "capture_rate": float(cap[:, ~wall_start].mean()), "mean_time": float(tm[:, ~wall_start].mean())},
    }
    summary[pol] = row

# paired differences in penalised time (learned - baseline), per scenario
paired = {}
for run in RUNS:
    L = arr(run, "time")
    for base in ("direct_pursuit", "predictive_interception", "coordinated_pursuit"):
        Bt = arr(base, "time")[0]
        D = L - Bt[None, :]
        paired[f"{run}-{base}"] = boot_ci(D)
    # paired vs full
    if run != "full":
        D = L - arr("full", "time")
        paired[f"{run}-full"] = boot_ci(D)

# learning curves
curves = {}
for run in RUNS:
    eps, vr, vt, cov = None, [], [], []
    for s in range(10):
        with open(os.path.join(REPO, "results", run, "logs", f"seed_{s}.csv")) as f:
            rows = list(csv.DictReader(f))
        eps = [int(r["episode"]) for r in rows]
        vr.append([float(r["val_success_rate"]) for r in rows])
        vt.append([float(r["val_mean_time"]) for r in rows])
        cov.append([int(r["visited_states"]) for r in rows])
    vr, vt, cov = np.array(vr), np.array(vt), np.array(cov)
    m = vr.mean(0)
    first50 = next((e for e, v in zip(eps, m) if v >= 0.5), None)
    first90 = next((e for e, v in zip(eps, m) if v >= 0.9), None)
    ns = RUNS[run][1]
    curves[run] = dict(
        episodes=eps, val_rate_mean=m.tolist(), val_rate_std=vr.std(0).tolist(),
        val_time_mean=vt.mean(0).tolist(), visited_mean=cov.mean(0).tolist(),
        first_ep_mean_val_ge_0_5=first50, first_ep_mean_val_ge_0_9=first90,
        final_coverage_frac=float(cov[:, -1].mean() / ns), n_states=ns,
        train_seconds_mean=float(np.mean([i["train_seconds"] for i in infos[run]])),
        best_val_rate_mean=float(np.mean([i["best_val_success_rate"] for i in infos[run]])),
        best_val_time_mean=float(np.mean([i["best_val_mean_time"] for i in infos[run]])),
        best_episode=[i["best_episode"] for i in infos[run]],
    )

rob = {str(p): {k: {"capture_rate_mean": float(np.mean([x["capture_rate"] for x in v])),
                    "capture_rate_sd": float(np.std([x["capture_rate"] for x in v])),
                    "mean_time_mean": float(np.mean([x["mean_time_penalized"] for x in v]))}
                for k, v in robust[p].items()} for p in robust}

out = dict(summary=summary, paired=paired, curves=curves, robustness=rob,
           n_wall_start=int(wall_start.sum()), n_central_start=int((~wall_start).sum()))
with open(os.path.join(OUT, "analysis.json"), "w") as f:
    json.dump(out, f, indent=2)
print("done")
