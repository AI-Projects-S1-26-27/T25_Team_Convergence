"""
tests/test_viz.py
=================
Unit tests for viz/renderer.py (Task T-17a and T-17b).
"""

from __future__ import annotations

import os
import sys
import tempfile
import pytest
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from env.environment import Environment
from baselines.direct_pursuit import DirectPursuit
from viz.renderer import (
    render_frame,
    record_episode,
    save_replay_gif,
    side_by_side_replay,
    PREDATOR_COLORS,
    PREY_COLOR,
)


def test_render_frame_creates_valid_axes():
    preds = [(5, 5), (5, 7), (8, 6)]
    prey = (6, 6)
    fig, ax = plt.subplots(figsize=(6, 6))
    ret_ax = render_frame(preds, prey, step=3, ax=ax)
    assert ret_ax is ax

    # Check bounds
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim == (-0.5, config.GRID_SIZE - 0.5)
    # Inverted y-axis check
    assert ylim == (config.GRID_SIZE - 0.5, -0.5)

    # Title check
    title = ax.get_title()
    assert "Step: 3" in title

    plt.close(fig)


def test_render_frame_with_paths():
    frames = [
        ([(5, 5), (5, 7), (8, 6)], (6, 6)),
        ([(5, 6), (5, 6), (7, 6)], (6, 7)),
    ]
    fig, ax = plt.subplots()
    render_frame(frames[1][0], frames[1][1], step=1, ax=ax, paths=frames)
    # Check that patches and line artists were added
    assert len(ax.patches) >= 4  # 3 predators + 1 prey
    assert len(ax.lines) >= 1    # at least one trajectory line plotted
    plt.close(fig)


def test_record_episode():
    env = Environment(seed=42)
    policy = DirectPursuit(seed=42)
    traj = record_episode(env, policy, seed=42)

    assert isinstance(traj, list)
    assert len(traj) > 1, "Episode should run at least 1 step"
    assert len(traj) <= config.MAX_STEPS + 1

    for frame in traj:
        preds, prey = frame
        assert len(preds) == config.NUM_PREDATORS
        for r, c in preds:
            assert 0 <= r < config.GRID_SIZE
            assert 0 <= c < config.GRID_SIZE
        pr, pc = prey
        assert 0 <= pr < config.GRID_SIZE
        assert 0 <= pc < config.GRID_SIZE


def test_save_replay_gif():
    traj = [
        ([(5, 5), (5, 7), (8, 6)], (6, 6)),
        ([(5, 6), (5, 6), (7, 6)], (6, 7)),
        ([(5, 6), (5, 6), (6, 6)], (6, 6)),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_replay.gif")
        save_replay_gif(traj, out_path, fps=4)
        assert os.path.isfile(out_path), "GIF file was not created"
        assert os.path.getsize(out_path) > 0, "GIF file is empty"


def test_side_by_side_replay():
    traj_a = [
        ([(5, 5), (5, 7), (8, 6)], (6, 6)),
        ([(5, 6), (5, 6), (6, 6)], (6, 6)),
    ]
    traj_b = [
        ([(5, 5), (5, 7), (8, 6)], (6, 6)),
        ([(5, 6), (5, 7), (7, 6)], (6, 7)),
        ([(5, 7), (5, 7), (6, 6)], (6, 7)),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_side_by_side.gif")
        side_by_side_replay(
            traj_a, traj_b,
            label_l="Team A", label_r="Team B",
            filename=out_path,
            fps=4,
        )
        assert os.path.isfile(out_path), "Side-by-side GIF was not created"
        assert os.path.getsize(out_path) > 0, "Side-by-side GIF is empty"
