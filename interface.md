# Environment Interface Contract — `PRE-00`

**Team Convergence (Team 25) — Predator-Prey Dynamics**  
**Owner:** Group A (Environment & Visualisation) — Lohit & Ansh  
**Status:** Authoritative — all groups must code against this document.

---

## 1. Overview

The `Environment` class (defined in `env/environment.py`) implements a **20 × 20 discrete grid**
world with **3 predators** and **1 prey**, all moving simultaneously each step.

Groups B (Baselines), C (Reinforcement Learning), and D (Visualisation) must interact
**exclusively** through the API described below. Do **not** access internal attributes directly.

---

## 2. Import

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # repo root on path
from env.environment import Environment
```

---

## 3. Constructor

```python
env = Environment(seed: Optional[int] = None, max_steps: int = 300)
```

| Parameter   | Type           | Description |
|-------------|----------------|-------------|
| `seed`      | `int` or `None`| Seed for the internal NumPy random generator. `None` → non-deterministic. |
| `max_steps` | `int`          | Episode length limit before timeout. Default matches `config.MAX_STEPS = 300`. |

---

## 4. `reset()` — Begin a New Episode

```python
obs = env.reset(
    positions: Optional[Sequence[Tuple[int, int]] | dict] = None,
    seed:      Optional[int] = None,
) -> dict
```

### Parameters

| Parameter   | Type | Description |
|-------------|------|-------------|
| `positions` | `None` *or* a **sequence of 4 `(row, col)` tuples** `[pred0, pred1, pred2, prey]` *or* a **dict** `{'predators': [(r1,c1),(r2,c2),(r3,c3)], 'prey': (r,c)}` | Force exact initial placement. When `None`, 4 distinct cells are sampled uniformly. |
| `seed`      | `int` or `None` | If provided, re-seeds the internal RNG before sampling positions. |

### Validation (when `positions` is supplied)
- Exactly 4 distinct cells required (3 predator + 1 prey).
- Every `(row, col)` must satisfy `0 <= row < 20` and `0 <= col < 20`.
- Duplicate cells raise `ValueError`.

### Return — initial observation `dict`

```python
{
    'predators':  [(r1, c1), (r2, c2), (r3, c3)],  # list of 3 (row, col) tuples
    'prey':       (r, c),                            # single (row, col) tuple
    'prev_prey':  None,                              # always None at step 0
    'step':       0,
}
```

> **`prev_prey`** is `None` at the start of every episode and is set to the
> prey's position *before* the most recent `step()` call thereafter.
> Group C uses `prev_prey` to compute the **Previous Prey Move** observation bin.

---

## 5. `step()` — Advance One Timestep

```python
obs, done, info = env.step(
    predator_actions: Sequence[int],   # length-3 list/tuple, each in {0,1,2,3,4}
    prey_action:      int,             # single int in {0,1,2,3,4}
) -> Tuple[dict, bool, dict]
```

### Movement — Simultaneous & Clipped

All four agents move **simultaneously** using the shared action encoding:

| Integer | Name  | Δrow | Δcol |
|---------|-------|------|------|
| `0`     | Stay  |  0   |  0   |
| `1`     | North | −1   |  0   |
| `2`     | East  |  0   | +1   |
| `3`     | South | +1   |  0   |
| `4`     | West  |  0   | −1   |

**Out-of-bounds rule:** If a candidate move would place an agent outside `[0, 19]`
in either dimension, that agent **stays at its current cell** (no error raised).

Predators are allowed to share cells with each other.

### Capture Conditions (checked after all moves)

Two mechanisms trigger capture:

1. **Same-cell capture** — `new_predator[i] == new_prey` for any predator `i`.
2. **Edge-swap capture** — `old_predator[i] == new_prey AND new_predator[i] == old_prey`
   for any predator `i`. (Agents "passed through" each other on the same edge.)

### Return Values

**`obs`** — updated observation `dict` (same schema as `reset()`, `step > 0`):

```python
{
    'predators':  [(r1, c1), (r2, c2), (r3, c3)],
    'prey':       (r, c),
    'prev_prey':  (r_old, c_old),   # prey position before this step; never None after step 1
    'step':       int,              # 1-indexed current step count
}
```

**`done`** — `bool`; `True` on capture **or** timeout.

**`info`** — diagnostic `dict`:

```python
{
    'capture':      bool,           # True if prey was caught this step
    'capture_step': int | None,     # step count when captured; None if not captured
    'timeout':      bool,           # True if step_count >= max_steps without capture
    'step_count':   int,            # current step count (same as obs['step'])
}
```

### Pre-conditions / Errors

- Calling `step()` on a finished episode (`done=True`) raises `RuntimeError`.
- Each action integer must be in `{0, 1, 2, 3, 4}`; invalid values raise `ValueError`.
- `predator_actions` must have length 3; wrong length raises `ValueError`.

---

## 6. `render()` — Visualise Current State

```python
env.render(ax=None) -> matplotlib.axes.Axes
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `ax`      | `matplotlib.axes.Axes` or `None` | Axes to draw on. If `None`, a new figure and axes are created. |

Returns the `Axes` object so callers can embed the frame in a subplot grid or save it.

Colour convention (subject to Group D finalisation):

| Symbol | Colour | Meaning |
|--------|--------|---------|
| `P`    | Red    | Predator |
| `Q`    | Blue   | Prey |

---

## 7. Coordinate System

```
         col 0  col 1  …  col 19
row  0  [ (0,0)  (0,1)  …  (0,19) ]   ← North wall
row  1  [ (1,0)  …                 ]
  …
row 19  [ (19,0) …      … (19,19) ]   ← South wall
```

- **North** = decreasing row index.
- **South** = increasing row index.
- **East**  = increasing col index.
- **West**  = decreasing col index.

---

## 8. Full Usage Example

```python
from env.environment import Environment

env = Environment(seed=42)
obs = env.reset()

print(obs['predators'])   # e.g. [(3,7), (12,1), (18,15)]
print(obs['prey'])        # e.g. (5,10)
print(obs['prev_prey'])   # None

for _ in range(300):
    pred_actions = [1, 3, 0]   # predator 0 → North, 1 → South, 2 → Stay
    prey_action  = 4            # prey → West
    obs, done, info = env.step(pred_actions, prey_action)
    if done:
        print(info)
        break
```

---

---

## 9. Prey Agent API (`agents/prey.py` — Task T-03)

### Import

```python
from agents.prey import get_legal_moves, prey_action
```

---

### `get_legal_moves(prey_pos, grid_size=20) -> List[int]`

Returns every action integer in `{0,1,2,3,4}` whose resulting cell stays
within `[0, grid_size-1]`.  Action `0` (Stay) is always legal, so the
returned list is **never empty**.

```python
legal = get_legal_moves((0, 0))   # → [0, 2, 3]  (Stay, East, South)
legal = get_legal_moves((10, 10)) # → [0, 1, 2, 3, 4]
```

---

### `prey_action(...) -> int`

```python
action = prey_action(
    prey_pos:           Tuple[int, int],
    predator_positions: Sequence[Tuple[int, int]],
    legal_moves:        Optional[Sequence[int]] = None,
    rng:                Optional[np.random.Generator] = None,
    random_prob:        float = 0.1,          # config.PREY_RANDOM_PROB
    pre_sampled_rand:   Optional[float] = None,
    pre_sampled_choice: Optional[int]   = None,
) -> int
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prey_pos` | `(row, col)` | Current prey position. |
| `predator_positions` | sequence of `(row, col)` | All predator positions (length 3 normally). |
| `legal_moves` | `List[int]` or `None` | Pre-computed legal actions; computed internally if `None`. |
| `rng` | `np.random.Generator` or `None` | Generator for live-mode random draws. |
| `random_prob` | `float` | Probability of a random action. Default `0.1`. |
| `pre_sampled_rand` | `float` or `None` | Pre-drawn `[0,1)` value for the random-trigger roll. When supplied, `rng` is **not** called for this roll. |
| `pre_sampled_choice` | `int` or `None` | Pre-drawn index into `legal_moves` used in the random branch. Ignored if the greedy branch is taken. |

#### Decision Logic

```
if pre_sampled_rand is not None:
    take_random = (pre_sampled_rand < random_prob)
else:
    take_random = (rng.random() < random_prob)

if take_random:
    return legal_moves[pre_sampled_choice % len(legal_moves)]   # or rng.choice
else:
    # Greedy-flee: pick move maximising min Manhattan dist to any predator.
    # Ties broken uniformly at random with rng.
    return argmax_move
```

#### Two Calling Modes

**Live training** (Groups B & C):
```python
rng = np.random.default_rng(seed)
action = prey_action(obs['prey'], obs['predators'], rng=rng)
```

**Deterministic replay** (Tasks T-11 & T-16 — matched scenarios):
```python
# pre_sampled arrays built once per scenario; indexing is caller's responsibility
action = prey_action(
    obs['prey'], obs['predators'],
    pre_sampled_rand=scenario['prey_randoms'][step],
    pre_sampled_choice=scenario['prey_choices'][step],
)
```

---

## 10. Constants Reference (`config.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `GRID_SIZE` | 20 | Grid side length (cells 0–19) |
| `NUM_PREDATORS` | 3 | Number of predator agents |
| `NUM_PREY` | 1 | Number of prey agents |
| `MAX_STEPS` | 300 | Default episode length cap |
| `NUM_ACTIONS` | 5 | Discrete actions per agent |
| `NUM_OBSERVATIONS` | 16,200 | Total observation states for Q-table |
| `REWARD_CAPTURE` | 10.0 | Shared reward on capture |
| `REWARD_STEP` | −0.01 | Per-step living penalty |
