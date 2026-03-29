from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .model import MorphologyTree


def _moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or arr.shape[0] < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(arr, dtype=float)
    for i in range(arr.shape[1]):
        out[:, i] = np.convolve(padded[:, i], kernel, mode="valid")
    return out


def smooth_polyline(
    xyz_um: np.ndarray,
    points_per_segment: int = 8,
    spline_smoothness: float = 1.0,
) -> np.ndarray:
    """Generate a visually smoother branch polyline.

    This uses piecewise linear upsampling followed by gentle moving-average smoothing.
    It is deliberately lightweight and robust for batch rendering of anisotropic z-sampled
    reconstructions.
    """
    del spline_smoothness
    xyz_um = np.asarray(xyz_um, dtype=float)
    if xyz_um.ndim != 2 or xyz_um.shape[1] != 3:
        raise ValueError("xyz_um must have shape (n_points, 3)")
    if xyz_um.shape[0] < 3:
        return xyz_um.copy()

    n_samples = max(int((xyz_um.shape[0] - 1) * points_per_segment + 1), xyz_um.shape[0])
    interp_idx = np.linspace(0, xyz_um.shape[0] - 1, n_samples)
    up = np.vstack([np.interp(interp_idx, np.arange(xyz_um.shape[0]), xyz_um[:, i]) for i in range(3)]).T
    smooth = _moving_average(up, window=5)
    smooth[0] = xyz_um[0]
    smooth[-1] = xyz_um[-1]
    return smooth


def smooth_branch_segments(
    tree: MorphologyTree,
    points_per_segment: int = 8,
    spline_smoothness: float = 1.0,
) -> List[pd.DataFrame]:
    out: List[pd.DataFrame] = []
    for seg in tree.branch_segments():
        xyz = seg[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
        smooth_xyz = smooth_polyline(
            xyz,
            points_per_segment=points_per_segment,
            spline_smoothness=spline_smoothness,
        )
        sdf = pd.DataFrame(smooth_xyz, columns=["x_um", "y_um", "z_um"])
        sdf["segment_start_id"] = int(seg["segment_start_id"].iloc[0])
        sdf["segment_end_id"] = int(seg["segment_end_id"].iloc[0])
        sdf["segment_branch_order"] = int(seg["segment_branch_order"].iloc[0])
        sdf["segment_strahler_order"] = int(seg["segment_strahler_order"].iloc[0])
        out.append(sdf)
    return out
