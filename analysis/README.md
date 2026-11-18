# Analysis

Post-hoc analysis behind the numbers and figures in the final report (Group D).
The outputs in `out/` are for the 30,000-episode models.
These scripts only read the repository's code and stored results. They import the
environment, evaluator, baselines and learned policies, and they change no other files.

## Scripts

| Script | What it does | Writes |
|---|---|---|
| `analyze.py` | Evaluates the three scripted baselines, a random team and the 10 saved seeds of each learned model (full, no-teammate, no-shaping) on the 200 test scenarios; computes coordination measures (sustained flanking, escape-route coverage, interception), bootstrap 95% confidence intervals, matched-scenario differences, learning curves and robustness at p = 0, 0.1, 0.2 | `out/analysis.json`, `out/episodes.pkl` |
| `role_metrics.py` | Lazy-agent measures per seed: share of captures by the busiest predator (`top_share`), share of predator turns spent standing still (`stay`), mean distance of the farthest predator (`far`) | `out/roles.json` |
| `figures.py` | Draws the report figures from the two JSON files | `out/figures/fig1…fig6.png` |

## How to run

From inside this `analysis` folder, after installing the repository's requirements
(`pip install -r ../requirements.txt`):

```bash
python analyze.py .. out      # a few minutes
python role_metrics.py ..     # writes out/roles.json in the current folder
python figures.py out
```

Run them in this order: `figures.py` needs both JSON files. The scripts read the
30,000-episode models used in the report: `results/full_30k`,
`results/ablation_no_teammate_30k` and `results/ablation_no_shaping_30k`.

## Outputs kept in the repository

- `out/analysis.json`: all statistics used in the report. Under `summary`, each entry is
  [estimate, lower 95% CI, upper 95% CI]. Also has `paired`, `curves` and `robustness`.
- `out/roles.json`: per-seed role measures, and under `summary` [mean, min, max] across seeds.
- `out/figures/`: the report's figures. Figure 1 = `fig1_learning_curves.png`, Figure 2 = `fig6_paired.png`,
  Figure 3 = `fig3_coordination.png`, Figure 4 = `fig4_roles.png`, Figure 5 = `fig5_robustness.png`.
  `fig2_outcomes.png` is an extra and is not used in the report.

`out/episodes.pkl` (raw per-episode results) is not committed; `analyze.py` recreates it.

## Checks

- The baseline results match `results/baseline_summary.json`.
- Every learned seed's capture rate and mean time match Group C's own evaluation in `results/test_eval/`.
- Rerunning the scripts on another machine gave identical JSON output.
