"""Single entry point for figure style: palettes, axis finishing, saving.

Two palette families are kept on purpose (Andrew, 2026-11-10):

* **Categorical** (``CATEGORICAL``): discrete, high-contrast hexes for coding
  categories such as conditions, DMDs or responses to different images.
* **Continuous** (``continuous_colors``): PNW_cmap palettes, which are mostly
  continuous and suit ordered or graded data (depth, time, rank).

Existing helpers in ``plot_utils`` and ``utils.utils.save_figure`` are re-exported
unchanged, so notebooks can import everything from here.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from vip_slap2_analysis.plotting.plot_utils import (  # noqa: F401  (re-exports)
    apply_plot_style,
    finalize_and_save_figure,
    get_dmd_colors,
    get_pnw_colors,
    style_axis,
)

def save_figure(*args, **kwargs):
    """Lazy re-export of ``utils.utils.save_figure`` (keeps this module import-light)."""
    from vip_slap2_analysis.utils.utils import save_figure as _save

    return _save(*args, **kwargs)


# Categorical palette used across notebooks (hard-coded in 40+ places before).
CATEGORICAL: Dict[str, str] = {
    "blue": "#7bbcd5",
    "orange": "#EBA287",
    "green": "#d1e2b0",
    "light_blue": "#9fd3f2",
}

# Seven pastel category colours (previously pasted into 23 notebooks).
PASTEL_7: Sequence[str] = (
    "#c5cae9", "#ffcdd2", "#c8e6c9", "#ffe0b2", "#e1bee7", "#d7ccc8", "#cfd8dc",
)


def categorical_colors(n: int) -> List[str]:
    """Return ``n`` categorical colours: the named set first, then the pastels, cycling."""
    if n < 0:
        raise ValueError("n must be non-negative")
    pool = list(CATEGORICAL.values()) + list(PASTEL_7)
    return [pool[i % len(pool)] for i in range(n)]


def continuous_colors(n: int, palette: str = "Sailboat") -> List:
    """Return ``n`` colours from a PNW palette (seaborn 'deep' fallback if unavailable)."""
    return list(get_pnw_colors(n_colors=n, palette=palette))
