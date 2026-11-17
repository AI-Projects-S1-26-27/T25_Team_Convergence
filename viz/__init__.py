"""
viz package
===========
Matplotlib-based visualizer, replay recorder, and animated GIF generator.
"""

from viz.renderer import (
    render_frame,
    record_episode,
    save_replay_gif,
    side_by_side_replay,
    PREDATOR_COLORS,
    PREY_COLOR,
)

__all__ = [
    "render_frame",
    "record_episode",
    "save_replay_gif",
    "side_by_side_replay",
    "PREDATOR_COLORS",
    "PREY_COLOR",
]
