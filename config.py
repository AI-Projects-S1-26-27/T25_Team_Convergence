"""
Global configuration constants for the 2D Predator-Prey RL simulation.

Project Proposal PDF Specification — 16,200 total observation states.
Team Convergence (Team 25) — Predator-Prey Dynamics.
"""

# ---------------------------------------------------------------------------
# Grid & Agent Setup
# ---------------------------------------------------------------------------
GRID_SIZE = 20          # 0-indexed: valid rows/cols are 0 to 19
NUM_PREDATORS = 3
NUM_PREY = 1
MAX_STEPS = 300

# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------
# 5 discrete actions; index corresponds to the integer passed to env.step()
ACTIONS = [
    (0,  0),   # 0: Stay
    (-1, 0),   # 1: North  (row decreases)
    (0,  1),   # 2: East   (col increases)
    (1,  0),   # 3: South  (row increases)
    (0, -1),   # 4: West   (col decreases)
]

ACTION_NAMES = {
    0: "Stay",
    1: "North",
    2: "East",
    3: "South",
    4: "West",
}

NUM_ACTIONS = 5

# ---------------------------------------------------------------------------
# Observation Space — PDF Spec yields 8 × 3 × 5 × 3 × 5 × 9 = 16,200 states
# ---------------------------------------------------------------------------
# Prey bearing:          8 bins  (N, NE, E, SE, S, SW, W, NW)
OBS_BEARING_BINS = 8

# Prey range:            3 bins  (Manhattan dist 0-2 | 3-6 | >=7)
OBS_RANGE_BINS = 3

# Nearest-teammate quadrant: 5 bins (4 cardinal quadrants + 1 overlap/same cell)
OBS_TEAMMATE_QUAD_BINS = 5

# Nearest-teammate range:    3 bins (Manhattan dist 0-2 | 3-6 | >=7)
OBS_TEAMMATE_RANGE_BINS = 3

# Previous prey move:    5 bins (Stay, N, E, S, W — mirrors action encoding)
OBS_PREV_PREY_BINS = 5

# Wall context:          9 bins (3 horizontal × 3 vertical regions)
#   Horizontal zones: rows 0-1 (near top), rows 2-17 (interior), rows 18-19 (near bottom)
#   Vertical   zones: cols 0-1 (near left), cols 2-17 (interior), cols 18-19 (near right)
OBS_WALL_BINS = 9

NUM_OBSERVATIONS = (
    OBS_BEARING_BINS
    * OBS_RANGE_BINS
    * OBS_TEAMMATE_QUAD_BINS
    * OBS_TEAMMATE_RANGE_BINS
    * OBS_PREV_PREY_BINS
    * OBS_WALL_BINS
)  # = 16,200

# ---------------------------------------------------------------------------
# RL Hyperparameters
# ---------------------------------------------------------------------------
ALPHA = 0.2             # Q-learning step size
GAMMA = 0.99            # Discount factor
EPS_START = 1.0         # Initial ε (fully random)
EPS_END = 0.05          # Final ε (almost greedy)
EPS_DECAY_FRAC = 0.8    # Fraction of TOTAL_EPISODES over which ε decays linearly

TOTAL_EPISODES = 10_000
CHECKPOINT_INTERVAL = 500

# ---------------------------------------------------------------------------
# Reward Signal
# ---------------------------------------------------------------------------
REWARD_CAPTURE = 10.0           # Shared reward on successful capture
REWARD_STEP = -0.01             # Per-step living penalty
MAX_MANHATTAN_DIST = 38.0       # Maximum possible Manhattan distance on 20×20 grid

# ---------------------------------------------------------------------------
# Prey Policy
# ---------------------------------------------------------------------------
PREY_RANDOM_PROB = 0.1          # Probability prey takes a random action

# ---------------------------------------------------------------------------
# Reproducibility — Seeds
# ---------------------------------------------------------------------------
TRAIN_SEEDS = list(range(0, 10))        # 10 training seeds
VAL_SEEDS = list(range(100, 110))       # 10 validation seeds
TEST_SEED = 200                         # Held-out test seed
NUM_TEST_SCENARIOS = 200
