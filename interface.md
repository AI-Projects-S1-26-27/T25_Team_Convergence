# Environment Interface Specification (Group A)

## 1. Overview
This document specifies the exact interface for the 2D grid predator-prey simulation environment.
All controller teams (Group B: Baselines, Group C: Reinforcement Learning) must interact with the environment exclusively via this API.

## 2. Environment Class: `Environment(seed=None)`
- **Constructor**: `env = Environment(seed=None)`
  - `seed`: Optional integer for deterministic reproducibility.

- **Reset Method**: `obs = env.reset(positions=None)`
  - `positions`: Optional dictionary to force exact placement:
    `{'predators': [(r1, c1), (r2, c2), (r3, c3)], 'prey': (rp, cp)}`.
    If `None`, randomly places 4 agents in distinct cells.
  - `obs`: Returns the initial observation dictionary:
    `{'predators': [(r1, c1), (r2, c2), (r3, c3)], 'prey': (rp, cp), 'step': 0}`.

- **Step Method**: `obs, done, info = env.step(predator_actions, prey_action)`
  - `predator_actions`: A list or tuple of 3 integers (each 0 to 4), one per predator.
  - `prey_action`: An integer (0 to 4) chosen by the prey agent.
  - `obs`: Updated dictionary `{'predators': [...], 'prey': (...), 'step': int}`.
  - `done`: Boolean flag. `True` if:
    1. Any predator captures the prey (same-cell capture or edge-swap capture).
    2. Step count reaches `300` (timeout).
  - `info`: Diagnostic dictionary:
    `{'capture': bool, 'capture_step': int or None}`.

- **Render Method**: `env.render(mode="human")`
  - Renders the current state on a 20x20 Matplotlib grid.

## 3. Coordinate System & Actions
- The grid is a 20x20 matrix. 
- Row index `r` is in `[0, 19]` (0 is top row, 19 is bottom row).
- Column index `c` is in `[0, 19]` (0 is left col, 19 is right col).
- Action encoding:
  - `0`: Stay `(0, 0)`
  - `1`: North `(-1, 0)`
  - `2`: East `(0, 1)`
  - `3`: South `(1, 0)`
  - `4`: West `(0, -1)`
