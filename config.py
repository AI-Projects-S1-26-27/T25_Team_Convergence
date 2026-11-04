"""
Global configuration constants for 2D Predator-Prey simulation.
"""
GRID_SIZE = 20
MAX_STEPS = 300
NUM_PREDATORS = 3

# Action indices: 0: Stay, 1: North, 2: East, 3: South, 4: West
ACTIONS = [
    (0, 0),   # 0: Stay
    (-1, 0),  # 1: North (up in row index)
    (0, 1),   # 2: East  (right in col index)
    (1, 0),   # 3: South (down in row index)
    (0, -1)   # 4: West  (left in col index)
]
ACTION_NAMES = ["STAY", "NORTH", "EAST", "SOUTH", "WEST"]

# Prey policy parameters
PREY_RANDOM_PROB = 0.1
SEED_DEFAULT = 42
