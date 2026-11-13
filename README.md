# Multi Agent Predator-Prey Dynamics

### Emergent Coordination via Reinforcement Learning

**Course:** Artificial Intelligence  
**Track:** Demo  
**Topic:** Reinforcement Learning and Multi-Agent Systems  
**Institution:** BITS Pilani, Hyderabad Campus  
**Team:** Team Convergence (Team 25)

## Team Members

| Name | Student ID |
|---|---|
| Aditya Maheshwari | 2024AAPS0261H |
| Ansh Dwivedi | 2024AAPS0326H |
| Aryaman Ray | 2024AAPS0227H |
| Krishna Pandey | 2024A7PS0065H |
| Kartikey Gaur | 2024AAPS0225H |
| Lohith Reddy Thumburu | 2024AAPS0226H |
| Vedaang Kharya | 2024A8PS0511H |


A multi-agent AI project exploring whether coordinated hunting behaviour can emerge through learning rather than explicit programming.

We propose a 2D predator-prey simulation with three independently learning predators and a non-learning prey. The predators will use tabular Q-learning with a shared capture objective, without being given predefined hunting roles or rewards for specific tactics such as flanking or interception.

The learned agents will be compared against hand-coded strategies including direct pursuit, predictive interception, and coordinated pursuit. Performance will be evaluated using capture success, capture time, and trajectory-based measures of coordination such as flanking and escape-route coverage.

The project aims to investigate when simple multi-agent learning can produce coordinated behaviour, when it fails, and how its behaviour compares with explicitly programmed strategies.

## Group C training and evaluation

### Python setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The runtime dependencies are in `requirements.txt`; `requirements-dev.txt` also installs pytest for the repository test suite.

### Train the remaining ablations

Run from the repository root. Each command trains seeds 0–9 for 10,000 episodes and saves outputs under `results/<condition>/`.

```powershell
.\.venv\Scripts\python.exe experiments/run_all_seeds.py --obs no_teammate --workers 4
.\.venv\Scripts\python.exe experiments/run_all_seeds.py --no-shaping --workers 4
```

Lower `--workers` if the computer cannot sustain four concurrent training jobs. For a single seed, use `train.py`, for example:

```powershell
.\.venv\Scripts\python.exe train.py --seed 0 --obs no_teammate
.\.venv\Scripts\python.exe train.py --seed 0 --no-shaping
```

Each condition writes checkpoint validation logs as CSV under `logs/`, best and periodic Q-tables as NumPy `.npy` files, and best-checkpoint metadata as `info.json`. Periodic checkpoint directories are ignored by Git; logs and best checkpoints are retained as experiment outputs.

### Random and robustness evaluations

```powershell
.\.venv\Scripts\python.exe experiments/run_remaining_evaluations.py --mode random
.\.venv\Scripts\python.exe experiments/run_remaining_evaluations.py --mode robustness
```

Use `--mode all` to run both. The random-agent JSON is written to `results/ablation_random/random_eval.json`. The robustness runner chooses the best full-model seed using validation success rate, breaking ties with validation mean time, and writes six JSON files to `results/robustness/` for the learned policy and Coordinated Pursuit at prey randomness 0.0, 0.1, and 0.2. Each JSON contains all 200 per-scenario outcomes and summary metrics.

All evaluations load the fixed `experiments/test_scenarios.pkl`; do not regenerate or edit that file. Scenario inputs are Python pickle files. Training logs use CSV; Q-tables use NumPy `.npy`; metadata and evaluation results use JSON.

### Observation-space note

The implemented full observation space has 16,200 states. The no-teammate observation implementation has 1,080 states (8 bearing × 3 prey range × 5 previous move × 9 wall contexts). The original work-distribution document lists 30,720 for that ablation, which does not match the feature bins implemented in this repository.

### Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
