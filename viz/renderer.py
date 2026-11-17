"""
viz/renderer.py
===============
Grid visualizer, episode recorder, and GIF animator for 2D Predator-Prey simulation.

Tasks T-17a & T-17b (Team 25 Group A).
Owner: Group A (Simulation & Visual Layer).
"""

from __future__ import annotations

import os
import sys
import tempfile
import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

# Ensure temporary writable directory for matplotlib cache before importing matplotlib
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_agy")

# Ensure matplotlib uses headless Agg backend
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# Windows / OneDrive path resolution fix for Python 3.13 / Matplotlib saving
_orig_resolve = pathlib.Path.resolve
def _safe_resolve(self, strict=False):
    try:
        return _orig_resolve(self, strict=strict)
    except PermissionError:
        return self.absolute()
pathlib.Path.resolve = _safe_resolve

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from env.environment import Environment
from agents.prey import prey_action

# Color specifications
PREDATOR_COLORS = ["#E63946", "#1D3557", "#2A9D8F"]  # P1: Red, P2: Blue, P3: Green
PREDATOR_LABELS = ["P1", "P2", "P3"]
PREY_COLOR = "#E9C46A"                                # Prey: Yellow / Orange
PREY_LABEL = "Prey"


def render_frame(
    predator_positions: Sequence[Tuple[int, int]],
    prey_position: Tuple[int, int],
    step: int,
    ax: Optional[plt.Axes] = None,
    paths: Optional[Union[Sequence[Tuple[Sequence[Tuple[int, int]], Tuple[int, int]]], Dict]] = None,
) -> plt.Axes:
    """
    Draw a single state frame on a clean 20x20 grid.

    Parameters
    ----------
    predator_positions : sequence of (row, col)
        Current positions of the 3 predators.
    prey_position : (row, col)
        Current position of the prey.
    step : int
        Current timestep count shown in title.
    ax : matplotlib.axes.Axes, optional
        Target axes to draw on. If None, creates a fresh figure and axes.
    paths : sequence of frames or dict, optional
        Historical coordinates up to the current step for overlaying faded trajectory lines.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    gs = config.GRID_SIZE

    # Grid limits [-0.5, 19.5], inverted y-axis so row 0 is at top
    ax.set_xlim(-0.5, gs - 0.5)
    ax.set_ylim(gs - 0.5, -0.5)
    ax.set_aspect("equal")

    # Light grid lines
    ax.set_xticks(range(gs))
    ax.set_yticks(range(gs))
    ax.grid(True, which="both", color="#D9D9D9", linestyle="-", linewidth=0.7, zorder=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Subplot title showing step counter
    ax.set_title(f"Step: {step}", fontsize=11, fontweight="bold", pad=8)

    # Overlay historical faded trajectory paths if provided
    if paths is not None:
        if isinstance(paths, dict):
            if "predators" in paths:
                for p_idx, p_coords in enumerate(paths["predators"]):
                    color = PREDATOR_COLORS[p_idx % len(PREDATOR_COLORS)]
                    if len(p_coords) > 1:
                        xs = [c for r, c in p_coords]
                        ys = [r for r, c in p_coords]
                        ax.plot(xs, ys, color=color, alpha=0.3, linewidth=2.0, zorder=2)
            if "prey" in paths:
                q_coords = paths["prey"]
                if len(q_coords) > 1:
                    xs = [c for r, c in q_coords]
                    ys = [r for r, c in q_coords]
                    ax.plot(xs, ys, color=PREY_COLOR, alpha=0.35, linewidth=2.0, zorder=2)
        elif isinstance(paths, (list, tuple)) and len(paths) > 0:
            for p_idx, color in enumerate(PREDATOR_COLORS):
                p_coords = [frame[0][p_idx] for frame in paths if len(frame[0]) > p_idx]
                if len(p_coords) > 1:
                    xs = [c for r, c in p_coords]
                    ys = [r for r, c in p_coords]
                    ax.plot(xs, ys, color=color, alpha=0.3, linewidth=2.0, zorder=2)

            q_coords = [frame[1] for frame in paths]
            if len(q_coords) > 1:
                xs = [c for r, c in q_coords]
                ys = [r for r, c in q_coords]
                ax.plot(xs, ys, color=PREY_COLOR, alpha=0.35, linewidth=2.0, zorder=2)

    # Draw predators
    for i, (r, c) in enumerate(predator_positions):
        color = PREDATOR_COLORS[i % len(PREDATOR_COLORS)]
        label = PREDATOR_LABELS[i % len(PREDATOR_LABELS)]
        circle = patches.Circle((c, r), radius=0.38, color=color, zorder=4)
        ax.add_patch(circle)
        ax.text(
            c, r, label,
            ha="center", va="center",
            color="white", fontsize=8, fontweight="bold",
            zorder=5,
        )

    # Draw prey
    pr, pc = prey_position
    prey_circle = patches.Circle((pc, pr), radius=0.38, color=PREY_COLOR, zorder=4)
    ax.add_patch(prey_circle)
    ax.text(
        pc, pr, PREY_LABEL,
        ha="center", va="center",
        color="#264653", fontsize=7, fontweight="bold",
        zorder=5,
    )

    return ax


def record_episode(
    env: Environment,
    predator_policy: Callable,
    prey_policy: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int]]]:
    """
    Run one episode and record (predator_positions, prey_position) for every step.

    Parameters
    ----------
    env : Environment
        The simulation environment.
    predator_policy : callable
        Policy returning 3 predator actions. Can be callable(obs) or callable(predators, prey).
    prey_policy : callable, optional
        Prey policy. If None, defaults to agents.prey.prey_action.
    seed : int, optional
        Optional seed to reset the environment.

    Returns
    -------
    list of (predator_positions, prey_position) tuples from t=0 to termination.
    """
    if seed is not None:
        obs = env.reset(seed=seed)
    else:
        obs = env.reset()

    if hasattr(predator_policy, "reset"):
        predator_policy.reset(seed)

    trajectory = [(list(obs["predators"]), tuple(obs["prey"]))]
    done = False

    while not done:
        # Determine predator actions
        try:
            pred_actions = predator_policy(obs)
        except TypeError:
            pred_actions = predator_policy(obs["predators"], obs["prey"])

        # Determine prey action
        if prey_policy is None:
            prey_act = prey_action(obs["prey"], obs["predators"])
        else:
            try:
                prey_act = prey_policy(obs)
            except TypeError:
                prey_act = prey_policy(obs["prey"], obs["predators"])

        obs, done, info = env.step(pred_actions, prey_act)
        trajectory.append((list(obs["predators"]), tuple(obs["prey"])))

    return trajectory


def save_replay_gif(
    trajectory: Sequence[Tuple[Sequence[Tuple[int, int]], Tuple[int, int]]],
    filename: str,
    fps: int = 4,
) -> None:
    """
    Save a recorded trajectory as an animated GIF using FuncAnimation and Pillow.

    Parameters
    ----------
    trajectory : sequence of (predator_positions, prey_position)
        Full episode trajectory.
    filename : str
        Target filepath for the output .gif.
    fps : int, default 4
        Frames per second.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx: int):
        ax.clear()
        preds, prey = trajectory[frame_idx]
        hist_paths = trajectory[: frame_idx + 1]
        render_frame(preds, prey, step=frame_idx, ax=ax, paths=hist_paths)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        repeat=True,
    )
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)


def side_by_side_replay(
    traj_learned: Sequence[Tuple[Sequence[Tuple[int, int]], Tuple[int, int]]],
    traj_baseline: Sequence[Tuple[Sequence[Tuple[int, int]], Tuple[int, int]]],
    label_l: str,
    label_r: str,
    filename: str,
    fps: int = 4,
) -> None:
    """
    Create a side-by-side synchronized comparison animation and save as GIF.

    If one policy captures earlier, it freezes on its final frame while the
    other continues animating until both complete.

    Parameters
    ----------
    traj_learned : sequence of (predators, prey)
        Trajectory for the left subplot.
    traj_baseline : sequence of (predators, prey)
        Trajectory for the right subplot.
    label_l : str
        Title prefix for left subplot.
    label_r : str
        Title prefix for right subplot.
    filename : str
        Destination .gif path.
    fps : int, default 4
        Frames per second.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    max_len = max(len(traj_learned), len(traj_baseline))

    def update(frame_idx: int):
        ax_l.clear()
        ax_r.clear()

        # Left: Learned policy
        idx_l = min(frame_idx, len(traj_learned) - 1)
        preds_l, prey_l = traj_learned[idx_l]
        paths_l = traj_learned[: idx_l + 1]
        render_frame(preds_l, prey_l, step=idx_l, ax=ax_l, paths=paths_l)

        is_cap_l = (len(traj_learned) <= config.MAX_STEPS) and (idx_l == len(traj_learned) - 1)
        status_l = f" [Captured @ t={idx_l}]" if is_cap_l else ""
        ax_l.set_title(f"{label_l} | Step: {idx_l}{status_l}", fontsize=11, fontweight="bold", pad=8)

        # Right: Baseline policy
        idx_r = min(frame_idx, len(traj_baseline) - 1)
        preds_r, prey_r = traj_baseline[idx_r]
        paths_r = traj_baseline[: idx_r + 1]
        render_frame(preds_r, prey_r, step=idx_r, ax=ax_r, paths=paths_r)

        is_cap_r = (len(traj_baseline) <= config.MAX_STEPS) and (idx_r == len(traj_baseline) - 1)
        status_r = f" [Captured @ t={idx_r}]" if is_cap_r else ""
        ax_r.set_title(f"{label_r} | Step: {idx_r}{status_r}", fontsize=11, fontweight="bold", pad=8)

        fig.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=max_len,
        repeat=True,
    )
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)
