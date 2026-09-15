"""Publication-quality visualisations of conflict-event data.

All figures are produced at 300 DPI with colourblind-safe palettes,
suitable for direct inclusion in journal submissions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from surrogate_safety_abm.simulation.recorder import ConflictEvent

# Colourblind-safe palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]


def _finalise(fig: Figure, out: Path | None) -> Figure:
    """Apply common styling and optionally save to disk."""
    fig.tight_layout()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig


def plot_ttc_histogram(
    events: list[ConflictEvent],
    out: Path | None = None,
    bins: int = 20,
) -> Figure:
    """Plot a histogram of TTC values across all conflict events."""
    if not events:
        raise ValueError("events must be non-empty")
    ttc = [ev.ttc_s for ev in events]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(ttc, bins=bins, color=PALETTE[0], edgecolor="white")
    ax.set_xlabel("TTC (s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Time-to-Collision")
    return _finalise(fig, out)


def plot_drac_histogram(
    events: list[ConflictEvent],
    out: Path | None = None,
    bins: int = 20,
) -> Figure:
    """Plot a histogram of DRAC values across all conflict events."""
    if not events:
        raise ValueError("events must be non-empty")
    drac = [ev.drac_ms2 for ev in events]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(drac, bins=bins, color=PALETTE[1], edgecolor="white")
    ax.set_xlabel(r"DRAC (m/s$^2$)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Deceleration Rate to Avoid Crash")
    return _finalise(fig, out)


def plot_ttc_cdf(
    events: list[ConflictEvent],
    out: Path | None = None,
) -> Figure:
    """Plot the empirical cumulative distribution function of TTC."""
    if not events:
        raise ValueError("events must be non-empty")
    ttc = np.sort(np.array([ev.ttc_s for ev in events]))
    y = np.arange(1, len(ttc) + 1) / len(ttc)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.step(ttc, y, where="post", color=PALETTE[2], linewidth=1.5)
    ax.set_xlabel("TTC (s)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical CDF of Time-to-Collision")
    ax.grid(True, alpha=0.3)
    return _finalise(fig, out)


def plot_conflict_heatmap(
    events: list[ConflictEvent],
    out: Path | None = None,
    bins: int = 30,
) -> Figure:
    """Plot a 2D histogram of TTC vs DRAC to identify severity clusters."""
    if not events:
        raise ValueError("events must be non-empty")
    ttc = [ev.ttc_s for ev in events]
    drac = [ev.drac_ms2 for ev in events]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    h = ax.hist2d(ttc, drac, bins=bins, cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("TTC (s)")
    ax.set_ylabel(r"DRAC (m/s$^2$)")
    ax.set_title("Joint distribution of conflict indicators")
    return _finalise(fig, out)
