# Multi Agent Predator-Prey Dynamics

### Emergent Coordination via Reinforcement Learning

**Course:** Artificial Intelligence  
**Track:** Demo  
**Topic:** Reinforcement Learning and Multi-Agent Systems  
**Institution:** BITS Pilani, Hyderabad Campus  
**Team:** Team Convergence (Team 25)

## Team Members

| Name | Student ID | Group Role |
|---|---|---|
| Aditya Maheshwari | 2024AAPS0261H | Group B (Baselines) |
| Ansh Dwivedi | 2024AAPS0326H | Group A (Environment & Visualization) |
| Aryaman Ray | 2024AAPS0227H | Group C (RL & Training) |
| Krishna Pandey | 2024A7PS0065H | Group B (Baselines) |
| Kartikey Gaur | 2024AAPS0225H | Group C (RL & Training) |
| Lohith Reddy Thumburu | 2024AAPS0226H | Group A (Environment & Visualization) |
| Vedaang Kharya | 2024A8PS0511H | Group C (RL & Training) |

---

## Project Overview

A multi-agent reinforcement learning project investigating whether coordinated hunting behavior can emerge through decentralized reinforcement learning without explicit inter-agent communication or hardcoded roles.

The environment is a discrete $20 \times 20$ grid containing:
- **3 Predator Agents**: Cooperating to capture the prey using independent Tabular Q-learning with a shared team reward signal.
- **1 Prey Agent**: An evader following a greedy-flee heuristic (maximizing minimum distance to nearest predator) with a 10% panic/exploration randomness factor.

Learned agents are evaluated against three benchmark heuristics across 200 held-out matched test scenarios:
1. **Direct Pursuit**: Greedy individual chase towards the prey's current position.
2. **Predictive Interception**: Interception aimed 2 steps ahead along the prey's movement vector.
3. **Coordinated Pursuit**: Minimum-cost target assignment distributing predators across projected adjacent cells.

---

## Repository Layout

```text
Predator_Prey_Dynamics/
├── config.py                 # Global simulation, action, reward, and observation constants
├── interface.md              # Authoritative API contract for Environment and Prey agent
├── requirements.txt          # Core dependencies (numpy, matplotlib, Pillow)
├── requirements-dev.txt      # Development dependencies (pytest)
├── train.py                  # Single-seed training entry-point (Task T-10)
├── env/
│   └── environment.py        # 20x20 grid simulation (Environment class, simultaneous movement)
├── agents/
│   └── prey.py               # Stochastic greedy-flee evader policy (prey_action)
├── baselines/
│   ├── common.py             # Shared baseline utilities and base class
│   ├── direct_pursuit.py     # Baseline 1: Greedy distance minimization
│   ├── predictive_interception.py  # Baseline 2: 2-step lookahead pursuit
│   └── coordinated_pursuit.py      # Baseline 3: Hungarian-style target assignment
├── rl/
│   ├── agent.py              # QLearningAgent, epsilon decay, TeamPolicy
│   ├── observation.py        # 16,200-state observation encoder (bearing, range, teammate, wall)
│   ├── observation_no_teammate.py  # 1,080-state ablation encoder
│   └── rewards.py            # Potential-based reward shaping functions
├── viz/
│   ├── renderer.py           # Matplotlib frame visualizer, episode recorder, GIF replay engine
│   └── generate_demo.py      # Automated 200-scenario selector & demo GIF generation pipeline
├── experiments/
│   ├── generate_scenarios.py # Deterministic generation of 200 test & 50 validation scenarios
│   ├── evaluate.py           # Shared scenario evaluation harness
│   ├── run_baselines.py      # Baseline evaluation pipeline (results/baseline_eval.json)
│   ├── run_all_seeds.py      # Multi-seed parallel training runner
│   ├── run_remaining_evaluations.py  # Robustness & ablation evaluator
│   ├── test_scenarios.pkl    # 200 pre-generated held-out test scenarios
│   └── val_scenarios.pkl     # 50 pre-generated disjoint validation scenarios
├── results/                  # Training logs, Q-tables, JSON summaries, and GIF demos
└── tests/                    # Comprehensive pytest test suite (227 tests)
```

---

## 1. Installation Instructions

The project requires **Python 3.10+** (tested and validated on Python 3.10, 3.11, 3.12, and 3.13).

### Windows PowerShell

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install core runtime dependencies (numpy, matplotlib, Pillow)
python -m pip install -r requirements.txt

# Install development & test dependencies (pytest)
python -m pip install -r requirements-dev.txt
```

### Linux / macOS

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core runtime dependencies (numpy, matplotlib, Pillow)
pip install -r requirements.txt

# Install development & test dependencies (pytest)
pip install -r requirements-dev.txt
```

---

## 2. How to Run Training

The training entry point trains three independent tabular Q-learning predator agents.

### A. Single-Seed Training (Full Model)

To train the full model (16,200-state observation with potential-based reward shaping) for 10,000 episodes:
```bash
python train.py --seed 0
```
- Trains for 10,000 episodes (default).
- Validates every 500 episodes against the 50 disjoint scenarios in `experiments/val_scenarios.pkl`.
- Automatically logs metrics to `results/full/logs/seed_0.csv`.
- Periodically saves checkpoints to `results/full/checkpoints/seed_0/ep_<N>/`.
- Saves the top-performing checkpoint (highest validation capture rate) to `results/full/best/seed_0/`.

### B. Multi-Seed Parallel Training

To train all 10 experimental seeds (`seed_0` through `seed_9`) in parallel across multiple CPU cores:
```bash
python experiments/run_all_seeds.py --workers 4
```

### C. Training Ablations

- **Ablation A (No Teammate Observation Features, 1,080 states):**
  ```bash
  python train.py --seed 0 --obs no_teammate --run-name ablation_no_teammate
  # Multi-seed:
  python experiments/run_all_seeds.py --obs no_teammate --workers 4
  ```
- **Ablation B (No Potential-Based Reward Shaping, base reward only):**
  ```bash
  python train.py --seed 0 --no-shaping --run-name ablation_no_shaping
  # Multi-seed:
  python experiments/run_all_seeds.py --no-shaping --workers 4
  ```

---

## 3. How to Run Evaluation

All policies are evaluated on the exact same 200 held-out matched scenarios in `experiments/test_scenarios.pkl`.

### A. Baseline Heuristics Evaluation

Evaluate Direct Pursuit, Predictive Interception, and Coordinated Pursuit:
```bash
python experiments/run_baselines.py
```
- Evaluates each heuristic across all 200 test scenarios.
- Writes full step-by-step trajectories to `results/baseline_eval.json`.
- Writes summary metrics (capture rate, penalized mean time, median time) to `results/baseline_summary.json`.

### B. Robustness & Learned Policy Evaluation

Evaluate the best full-model learned checkpoint and Coordinated Pursuit across varying prey exploration/panic probabilities ($p \in \{0.0, 0.1, 0.2\}$):
```bash
python experiments/run_remaining_evaluations.py --mode robustness
```
- Outputs `results/robustness/learned_prey_random_<0.0|0.1|0.2>.json`
- Outputs `results/robustness/coordinated_pursuit_prey_random_<0.0|0.1|0.2>.json`

### C. Sanity Baseline (Random Predators)

Evaluate uniform random predator actions (Ablation C):
```bash
python experiments/run_remaining_evaluations.py --mode random
```
- Outputs `results/ablation_random/random_eval.json`.

### D. Full Evaluation Suite

Run all evaluations (random + robustness) in a single command:
```bash
python experiments/run_remaining_evaluations.py --mode all
```

---

## 4. How to Reproduce a Specific Seed

Every run uses deterministic seeding with NumPy `SeedSequence` spawning separate streams for environment initial placement, prey decisions, and each predator's tie-breaking/exploration.

### Step-by-Step Seed Reproduction Walkthrough (Seed 0)

1. **Execute the Training Command:**
   ```bash
   python train.py --seed 0
   ```
2. **Deterministic Validation Monitoring:**
   - Every 500 episodes, greedy evaluation ($\epsilon = 0$) is executed on the 50 validation scenarios in `experiments/val_scenarios.pkl`.
   - Metrics are appended to `results/full/logs/seed_0.csv`.
3. **Inspect Output Files for Seed 0:**
   - **Validation Summary:** Open `results/full/best/seed_0/info.json`:
     ```json
     {
       "seed": 0,
       "run_name": "full",
       "obs": "full",
       "shaped": true,
       "episodes": 10000,
       "best_episode": 10000,
       "best_val_success_rate": 0.9,
       "best_val_mean_time": 117.34,
       "alpha": 0.2,
       "gamma": 0.99,
       "train_seconds": 208.4
     }
     ```
   - **Learned Q-Tables:** The final best tables are saved as `predator_0.npy`, `predator_1.npy`, and `predator_2.npy` in `results/full/best/seed_0/`.
4. **Reproducibility Guarantee:**
   Because `train.py` utilizes `np.random.SeedSequence(seed).spawn(...)`, running `python train.py --seed 0` on any standard machine produces identical Q-tables and a validation success rate matching 0.90 within strict numerical tolerance ($\le 0.02$).

---

## 5. Results Directory Guide

The `results/` folder stores all training artifacts, quantitative evaluations, and visual media:

```text
results/
├── baseline_eval.json             # Full 200-scenario step trajectories for all 3 baselines
├── baseline_summary.json          # Aggregate metrics (capture rate, mean time, median time)
├── full/                          # Full model (16,200 states + shaped reward)
│   ├── best/seed_<0-9>/           # Best Q-tables (predator_0/1/2.npy) & info.json per seed
│   ├── checkpoints/seed_<0-9>/    # Checkpoints saved every 500 episodes (ep_500/, ...)
│   └── logs/seed_<0-9>.csv        # Training logs (ep, eps, val_rate, val_time, train_rate, visited)
├── ablation_no_teammate/          # Ablation A (1,080 states, no teammate relative features)
│   ├── best/seed_<0-9>/           # Best Q-tables and info.json per seed
│   └── logs/seed_<0-9>.csv        # Training logs
├── ablation_no_shaping/           # Ablation B (Sparse goal reward only, no potential shaping)
│   ├── best/seed_<0-9>/           # Best Q-tables and info.json per seed
│   └── logs/seed_<0-9>.csv        # Training logs
├── ablation_random/
│   └── random_eval.json           # 200-scenario test evaluation of random predator actions
├── robustness/                    # Robustness evaluations on 200 held-out test scenarios
│   ├── learned_prey_random_0.0.json            # Learned policy against deterministic prey (p=0.0)
│   ├── learned_prey_random_0.1.json            # Learned policy against default prey (p=0.1)
│   ├── learned_prey_random_0.2.json            # Learned policy against high-panic prey (p=0.2)
│   ├── coordinated_pursuit_prey_random_0.0.json # Coordinated Pursuit baseline (p=0.0)
│   ├── coordinated_pursuit_prey_random_0.1.json # Coordinated Pursuit baseline (p=0.1)
│   └── coordinated_pursuit_prey_random_0.2.json # Coordinated Pursuit baseline (p=0.2)
└── gifs/                          # Generated animated comparison demo assets (Tasks T-17a & T-17b)
    ├── best.gif                   # Best learned capture episode (Scenario #21)
    ├── typical.gif                # Median duration capture episode (Scenario #26)
    ├── worst.gif                  # Slowest capture episode (Scenario #74)
    ├── timeout_failure.gif        # Evasion timeout episode (Scenario #27)
    ├── coordinated_flank_demo.gif # Flanking maneuver demo (Scenario #13)
    └── learned_vs_direct_pursuit.gif # Learned vs Direct Pursuit demo (Scenario #13)
```

---

## 6. Visualization & Demo Replays (Tasks T-17a & T-17b)

The visualization module ([viz/renderer.py](viz/renderer.py)) provides an animated grid viewer with:
- $20 \times 20$ discrete grid with coordinate axes and inverted y-axis (row 0 at top).
- **Predators**: Circles in Red (`P1`), Blue (`P2`), and Green (`P3`).
- **Prey**: Circle in Yellow/Orange (`Prey`).
- **Trajectory Trails**: Faded path lines ($\alpha = 0.3$) displaying historical movement.
- **Side-by-Side Synchronized Playback**: Side-by-side view with live step counters, freezing completed episodes upon capture while the other finishes.

### Generating All Demo GIFs

To automatically scan the 200 test scenarios and generate representative comparison GIFs:
```bash
python -m viz.generate_demo
```

### Demo Comparison Assets (`results/gifs/`)

| Filename | Scenario | Policies Compared | Outcome & Analysis |
|---|---|---|---|
| `best.gif` | #21 | Learned vs. Coordinated | Learned captures in 22 steps vs. 14 steps. Rapid corner containment. |
| `typical.gif` | #26 | Learned vs. Coordinated | Learned captures in 70 steps vs. 23 steps. Steady multi-angle pursuit. |
| `worst.gif` | #74 | Learned vs. Coordinated | Learned captures in 291 steps vs. 33 steps. Extended evasion near timeout. |
| `timeout_failure.gif` | #27 | Learned vs. Coordinated | Learned times out (300 steps) vs. 30 steps. Illustrates prey evasion in open space. |
| `coordinated_flank_demo.gif` | #13 | Learned vs. Coordinated | Learned captures in 43 steps vs. 105 steps. Demonstrates perimeter flanking. |
| `learned_vs_direct_pursuit.gif` | #13 | Learned vs. Direct | Learned captures in 43 steps vs. 107 steps. Avoids linear chase stagnation. |

---

## 7. Running Unit Tests

The test suite validates the environment, movement dynamics, capture mechanics, prey evaders, baseline controllers, observation encoders, reward shaping, scenario generation, and rendering:

```bash
# Run the complete test suite (227 tests)
python -m pytest -v

# Run individual test modules
python -m pytest tests/test_env.py -v         # Core environment dynamics & edge-swap captures
python -m pytest tests/test_prey.py -v        # Greedy flee heuristic and tie-breaking
python -m pytest tests/test_viz.py -v         # Renderer, GIF generation, and side-by-side replays
python -m pytest tests/test_baselines_common.py -v # Baseline utility functions
python -m pytest tests/test_direct_pursuit.py -v   # Direct Pursuit baseline
python -m pytest tests/test_predictive_interception.py -v # Predictive Interception baseline
python -m pytest tests/test_coordinated_pursuit.py -v    # Coordinated Pursuit baseline
python -m pytest tests/test_agent.py -v       # Tabular Q-learning agent logic
python -m pytest tests/test_rewards.py -v     # Potential-based reward shaping
python -m pytest tests/test_scenarios_eval.py -v # Scenarios and evaluation harness
```
