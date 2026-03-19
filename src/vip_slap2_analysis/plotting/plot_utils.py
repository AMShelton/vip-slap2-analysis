from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

from vip_slap2_analysis.utils.utils import save_figure


def apply_plot_style() -> None:
    """
    Apply a plotting style aligned with Andrew's notebook aesthetic.
    """
    sns.set()
    sns.set_style("white")
    params = {
        "legend.fontsize": "x-large",
        "axes.labelsize": "xx-large",
        "axes.titlesize": "xx-large",
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
    }
    plt.rcParams.update(params)


def get_pnw_colors(n_colors: int = 8, palette: str = "Sailboat") -> List:
    """
    Use PNW_cmap if available; otherwise fall back to seaborn colors.
    """
    try:
        import PNW_cmap  # type: ignore
        _, _, cmap = PNW_cmap.get_PNW_cmap(palette, n_colors=n_colors)
        return list(cmap)
    except Exception:
        return sns.color_palette("deep", n_colors=n_colors)


def get_dmd_colors(palette: str = "Sailboat") -> dict:
    """
    Return a stable DMD color mapping.
    """
    colors = get_pnw_colors(n_colors=8, palette=palette)
    return {
        1: colors[1] if len(colors) > 1 else "tab:blue",
        2: colors[5] if len(colors) > 5 else "tab:orange",
    }


def style_axis(ax: plt.Axes, spine_width: float = 2.5) -> None:
    """
    Apply standard axis styling.
    """
    ax.tick_params(axis="x", which="major", reset=True, top=False)
    ax.tick_params(axis="y", which="major", reset=True, right=False)
    sns.despine(ax=ax)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(spine_width)


def finalize_and_save_figure(
    fig: plt.Figure,
    save_path_stem: str | Path,
    formats: Optional[Sequence[str]] = None,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """
    Tighten layout, save figure, and optionally close it.
    """
    if formats is None:
        formats = [".pdf", ".png"]

    fig.tight_layout()
    save_figure(fig, str(save_path_stem), formats=list(formats), dpi=dpi)

    if close:
        plt.close(fig)