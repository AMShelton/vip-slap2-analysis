from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import seaborn as sns


from .model import MorphologyTree
from .smoothing import smooth_branch_segments

Projection = Literal["xy", "xz", "zy"]


def _apply_plot_style() -> None:
    sns.set()
    sns.set_style("white")
    plt.rcParams.update({
        "legend.fontsize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlesize": "x-large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "pdf.fonttype": 42,
    })


def _finalize_and_save_figure(
    fig: plt.Figure,
    save_path_stem: str | Path,
    formats: Sequence[str] = (".pdf", ".png"),
    dpi: int = 300,
    close: bool = True,
) -> None:
    save_path_stem = Path(save_path_stem)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(save_path_stem.with_suffix(fmt), dpi=dpi, transparent=False)
    if close:
        plt.close(fig)



def _projection_columns(projection: Projection) -> tuple[str, str]:
    mapping = {
        "xy": ("x_um", "y_um"),
        "xz": ("x_um", "z_um"),
        "zy": ("z_um", "y_um"),
    }
    return mapping[projection]


def _make_segments(tree: MorphologyTree, projection: Projection, smooth: bool = True) -> tuple[list[np.ndarray], list[float]]:
    c0, c1 = _projection_columns(projection)
    branch_order_values: list[float] = []
    segments: list[np.ndarray] = []

    if smooth:
        branch_segments = smooth_branch_segments(tree)
        for seg in branch_segments:
            xy = seg[[c0, c1]].to_numpy(dtype=float)
            if len(xy) >= 2:
                segments.append(xy)
                branch_order_values.append(float(seg["segment_branch_order"].iloc[0]))
    else:
        for seg in tree.branch_segments():
            xy = seg[[c0, c1]].to_numpy(dtype=float)
            if len(xy) >= 2:
                segments.append(xy)
                branch_order_values.append(float(seg["segment_branch_order"].iloc[0]))

    return segments, branch_order_values


def plot_morphology_projection(
    tree: MorphologyTree,
    projection: Projection = "xy",
    smooth: bool = True,
    color_by: Literal["branch_order", "single_color"] = "branch_order",
    line_width: float = 2.0,
    alpha: float = 0.95,
    show_root: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    _apply_plot_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    segments, order_values = _make_segments(tree, projection=projection, smooth=smooth)
    if not segments:
        return ax

    if color_by == "branch_order":
        cmap = plt.cm.viridis
        lc = LineCollection(segments, cmap=cmap, linewidths=line_width, alpha=alpha)
        lc.set_array(np.asarray(order_values, dtype=float))
    else:
        color = sns.color_palette("deep", n_colors=1)[0]
        lc = LineCollection(segments, colors=[color], linewidths=line_width, alpha=alpha)
    ax.add_collection(lc)

    c0, c1 = _projection_columns(projection)
    nodes = tree.nodes
    ax.set_xlim(nodes[c0].min() - 2, nodes[c0].max() + 2)
    ax.set_ylim(nodes[c1].min() - 2, nodes[c1].max() + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel(f"{c0[0].upper()} (µm)")
    ax.set_ylabel(f"{c1[0].upper()} (µm)")
    ax.set_title(f"Morphology projection: {projection.upper()}")

    if show_root:
        root = tree.get_row(tree.root_id)
        ax.scatter(root[c0], root[c1], s=40, zorder=5)

    return ax


def plot_morphology_triptych(
    tree: MorphologyTree,
    smooth: bool = True,
    save_path_stem: Optional[str | Path] = None,
    formats: Sequence[str] = (".pdf", ".svg", ".png"),
    dpi: int = 300,
) -> tuple[plt.Figure, list[plt.Axes]]:
    _apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, projection in zip(axes, ["xy", "xz", "zy"]):
        plot_morphology_projection(tree, projection=projection, smooth=smooth, ax=ax)
    fig.suptitle(tree.source_path.stem if tree.source_path is not None else "Morphology", y=1.02)

    if save_path_stem is not None:
        _finalize_and_save_figure(fig, save_path_stem=save_path_stem, formats=formats, dpi=dpi, close=False)
    return fig, list(axes)


def save_single_projection(
    tree: MorphologyTree,
    save_path: str | Path,
    projection: Projection = "xy",
    smooth: bool = True,
    dpi: int = 300,
) -> Path:
    save_path = Path(save_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_morphology_projection(tree, projection=projection, smooth=smooth, ax=ax)
    _finalize_and_save_figure(fig, save_path_stem=save_path.with_suffix(""), formats=[save_path.suffix], dpi=dpi)
    return save_path
