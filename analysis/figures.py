"""Figures for the report and deck. Usage: python figures.py <analysis_out_dir>"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
A = json.load(open(os.path.join(OUT, "analysis.json")))
R = json.load(open(os.path.join(OUT, "roles.json")))
S = A["summary"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#555555", "axes.labelcolor": "#333333",
    "xtick.color": "#444444", "ytick.color": "#444444", "legend.frameon": False,
})

ORDER = ["full", "ablation_no_teammate", "ablation_no_shaping",
         "direct_pursuit", "predictive_interception", "coordinated_pursuit", "random"]
LABEL = {"full": "Learned (full)", "ablation_no_teammate": "Learned, no teammate obs",
         "ablation_no_shaping": "Learned, no shaping", "direct_pursuit": "Direct pursuit",
         "predictive_interception": "Predictive interception",
         "coordinated_pursuit": "Coordinated pursuit", "random": "Random"}
COL = {"full": "#9c4f46", "ablation_no_teammate": "#5d8467", "ablation_no_shaping": "#c08d4f",
       "direct_pursuit": "#7d7d7d", "predictive_interception": "#a89a74",
       "coordinated_pursuit": "#5b7290", "random": "#bdbdbd"}

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

# 1. learning curves ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for run in ["full", "ablation_no_teammate", "ablation_no_shaping"]:
    c = A["curves"][run]
    x = np.array(c["episodes"]); m = np.array(c["val_rate_mean"]); s = np.array(c["val_rate_std"])
    ax.plot(x, m, color=COL[run], lw=2, label=LABEL[run])
    ax.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1), color=COL[run], alpha=0.15, lw=0)
ax.set_xlabel("Training episodes")
ax.set_ylabel("Validation capture rate (ε = 0)")
ax.set_ylim(0, 1.02)
n_ep = max(A["curves"]["full"]["episodes"])  # training budget, read from the logs
ax.set_xlim(A["curves"]["full"]["episodes"][0], n_ep)
ax.axvline(0.8 * n_ep, color="#999999", lw=0.8, ls=":")
ax.text(0.8 * n_ep + n_ep / 125, 0.04, "ε reaches 0.05", fontsize=9, color="#666666")
ax.legend(loc="upper left")
save(fig, "fig1_learning_curves")

# 2. outcome comparison (capture rate + penalised time) ---------------------
pols = [p for p in ORDER]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
y = np.arange(len(pols))[::-1]
for ax, key, xl in ((axes[0], "capture_rate", "Capture rate (95% CI)"),
                    (axes[1], "mean_time_penalized", "Penalised mean time, turns (95% CI)")):
    for yi, p in zip(y, pols):
        v, lo, hi = S[p][key]
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=COL[p], ms=7, capsize=3, lw=1.5)
    ax.set_xlabel(xl)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[p] for p in pols])
axes[0].set_xlim(-0.03, 1.05)
axes[1].set_xlim(0, 310)
fig.tight_layout()
save(fig, "fig2_outcomes")

# 2b. paired difference vs coordinated pursuit (deck-friendly) ---------------
# 3. coordination metrics ---------------------------------------------------
mets = [("coverage", "Escape-route coverage"), ("flank_frac", "Share of turns in sustained flanking"),
        ("chase_rate", "Predator moves that close on the prey")]
pols3 = [p for p in ORDER if p != "random"] + ["random"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
for ax, (k, xl) in zip(axes, mets):
    for yi, p in zip(y, pols3):
        v, lo, hi = S[p][k]
        ax.barh(yi, v, color=COL[p], height=0.65)
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="none", ecolor="#333333", capsize=2, lw=1)
    ax.set_xlabel(xl)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[p] for p in pols3])
fig.tight_layout()
save(fig, "fig3_coordination")

# 4. role concentration -----------------------------------------------------
pols4 = ["full", "ablation_no_teammate", "ablation_no_shaping",
         "direct_pursuit", "predictive_interception", "coordinated_pursuit"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
y4 = np.arange(len(pols4))[::-1]
for ax, k, xl in ((axes[0], "top_share", "Captures made by the busiest predator"),
                  (axes[1], "stay", "Predator turns spent standing still")):
    for yi, p in zip(y4, pols4):
        vals = [d[k] for d in R["per_seed"][p]]
        ax.barh(yi, np.mean(vals), color=COL[p], height=0.65, alpha=0.85)
        if len(vals) > 1:
            ax.scatter(vals, np.full(len(vals), yi) + np.linspace(-0.18, 0.18, len(vals)),
                       s=10, color="#333333", zorder=3, label="one training seed" if (p == "full" and k == "top_share") else None)
    ax.set_xlabel(xl)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
axes[0].axvline(1 / 3, color="#999999", ls=":", lw=1)
axes[0].text(0.345, y4[-1] - 0.55, "equal share", fontsize=9, color="#666666")
axes[0].set_yticks(y4); axes[0].set_yticklabels([LABEL[p] for p in pols4])
axes[0].legend(loc="lower right", fontsize=9)
fig.tight_layout()
save(fig, "fig4_roles")

# 5. robustness -------------------------------------------------------------
ps = ["0.0", "0.1", "0.2"]
fig, ax = plt.subplots(figsize=(7, 4))
for p in ["full", "ablation_no_teammate", "ablation_no_shaping", "direct_pursuit",
          "predictive_interception", "coordinated_pursuit"]:
    m = [A["robustness"][q][p]["capture_rate_mean"] for q in ps]
    sd = [A["robustness"][q][p]["capture_rate_sd"] for q in ps]
    ls = "-" if p.startswith(("full", "abl")) else "--"
    ax.errorbar([0, 0.1, 0.2], m, yerr=sd if p.startswith(("full", "abl")) else None, color=COL[p],
                ls=ls, marker="o", lw=2, capsize=3, label=LABEL[p])
ax.set_xticks([0, 0.1, 0.2]); ax.set_xticklabels(["0 (deterministic)", "0.1 (training)", "0.2"])
ax.set_xlabel("Prey random-move probability p")
ax.set_ylabel("Capture rate on 200 test scenarios")
ax.set_ylim(0.4, 1.03)
ax.legend(loc="lower right", fontsize=9)
save(fig, "fig5_robustness")

# 6. paired differences in penalised time -----------------------------------
fig, ax = plt.subplots(figsize=(7, 3.4))
rows = [("full", "coordinated_pursuit"), ("full", "direct_pursuit"),
        ("ablation_no_teammate", "coordinated_pursuit"), ("ablation_no_teammate", "direct_pursuit"),
        ("ablation_no_shaping", "coordinated_pursuit")]
yy = np.arange(len(rows))[::-1]
for yi, (a, b) in zip(yy, rows):
    v, lo, hi = A["paired"][f"{a}-{b}"]
    ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=COL[a], ms=7, capsize=3, lw=1.5)
ax.axvline(0, color="#555555", lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([f"{LABEL[a]} − {LABEL[b]}" for a, b in rows], fontsize=9)
ax.set_xlabel("Paired difference in penalised time, turns (learned slower → positive)")
ax.grid(axis="x", color="#e5e5e5", lw=0.8)
save(fig, "fig6_paired")
print(sorted(os.listdir(FIG)))
