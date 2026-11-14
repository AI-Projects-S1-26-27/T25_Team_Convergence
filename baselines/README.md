# Scripted baselines (Group B)

Three hand-coded predator teams that the learned Q-learning predators are compared
against. No learning is involved — each team follows a fixed rule from the project
proposal (Section 4). All three share the same movement and capture rules as the
learned agents, and are evaluated with the team's shared evaluator
(`experiments/evaluate.py`) on the same matched scenarios.

## The three baselines

Every baseline picks a **target cell** for each predator, then moves each predator one
greedy step toward its target (`step_toward` in `common.py`). Only the target differs.

| Baseline | File | Target for each predator |
|---|---|---|
| Direct Pursuit (T-04) | `direct_pursuit.py` | The prey's current cell |
| Predictive Interception (T-05) | `predictive_interception.py` | The prey's cell projected 2 turns ahead by repeating its last move, clipped to the grid |
| Coordinated Pursuit (T-06) | `coordinated_pursuit.py` | A *different* cell for each predator, chosen from the projected cell and its in-bounds N/E/S/W neighbours so that total travel distance is minimal (all ≤ 60 assignments checked) |

On step 0 there is no previous prey position, so the projection is the prey's current
cell (Predictive Interception behaves like Direct Pursuit for that step).

## Using a baseline

Each baseline is a class following the evaluator's policy format,
`policy(env_obs) -> [a0, a1, a2]` with an optional `policy.reset(seed)`:

```python
from baselines.coordinated_pursuit import CoordinatedPursuit
from experiments.evaluate import evaluate, summarize
from experiments.generate_scenarios import load_scenarios

results = evaluate(CoordinatedPursuit(), load_scenarios(), record=True)
print(summarize(results))
```

The plain functions (`direct_pursuit_actions`, `predictive_interception_actions`,
`coordinated_pursuit_actions`, `coordinated_targets`, `project_prey`) can also be called
directly on positions, e.g. by the coordination-metric code.

## Reproducing the results

```
python experiments/run_baselines.py
```

Runs all three baselines on the 200 test scenarios (`experiments/test_scenarios.pkl`,
read-only) and writes:

- `results/baseline_eval.json` — `{baseline_name: [200 result dicts]}`; each dict has
  `scenario`, `captured`, `steps_to_capture` (300 if not captured), `capture_step`, and
  `trajectory` (`[[[r,c] x3 predators], [r,c] prey]` per step, starting at the initial positions).
- `results/baseline_summary.json` — settings used and the summary numbers.

Robustness variant (different prey randomness):

```
python experiments/run_baselines.py --prey-random-prob 0.2 --out results/robustness/baselines_p0.2.json
```

## Results (200 test scenarios, prey randomness 0.1)

| Baseline | Capture rate | Mean time (timeout = 300) | Median time (captures) |
|---|---|---|---|
| Direct Pursuit | 100% | 40.9 | 31.5 |
| Predictive Interception | 100% | 44.8 | 36.0 |
| Coordinated Pursuit | 100% | 32.0 | 29.0 |

## Design notes

- **Randomness:** used only to break ties between equally good moves or assignments,
  via each policy's own NumPy generator, re-seeded from the scenario seed in `reset()`.
  It never touches the prey's randomness, so scenarios stay matched, and every run is
  exactly reproducible.
- **Faithful to the proposal:** no extra tactics were added (e.g. no "step onto the prey
  when adjacent" shortcut), so the baselines match the rules stated in the proposal.
- **Coordinated Pursuit version:** Group C wrote an earlier stand-in
  `coordinated_pursuit.py` for their robustness runs (targets = projected cell plus the
  "widest" adjacent pair, which in practice is always the cells above and to the right).
  The team chose the proposal's rule (this implementation) as the official baseline.
  The old class name `CoordinatedPursuitPolicy` still works as an alias, so
  `experiments/run_remaining_evaluations.py` runs unchanged; its Coordinated Pursuit
  robustness files should be regenerated with this version
  (`python experiments/run_remaining_evaluations.py --mode robustness`).
- **Coordinated Pursuit limitation:** distinct targets stop predators stacking, but the
  rule does not force them onto different *sides* of the prey; predators approaching
  from one side take that side's cells.

## Tests

```
python -m pytest tests/test_baselines_common.py tests/test_direct_pursuit.py tests/test_predictive_interception.py tests/test_coordinated_pursuit.py tests/test_run_baselines.py
```
