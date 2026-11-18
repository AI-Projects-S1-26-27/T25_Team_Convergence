# Analysis

Post-hoc analysis behind the numbers and figures in the final report (Group D).
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

Run them in this order: `figures.py` needs both JSON files.

## Outputs kept in the repository

- `out/analysis.json`: all statistics used in the report. Under `summary`, each entry is
  [estimate, lower 95% CI, upper 95% CI]. Also has `paired`, `curves` and `robustness`.
- `out/roles.json`: per-seed role measures, and under `summary` [mean, min, max] across seeds.
- `out/figures/`: Figures 1–5 of the report (`fig6_paired.png` is an extra).

`out/episodes.pkl` (raw per-episode results) is not committed; `analyze.py` recreates it.

## Checks

- The baseline results match `results/baseline_summary.json`.
- The best learned seed (seed 4: 97.5% capture, 87.9 turns) matches the stored robustness results.
- Rerunning the scripts on another machine gave identical JSON output.
