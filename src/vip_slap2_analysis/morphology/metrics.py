from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .model import MorphologyTree


@dataclass
class ShollResult:
    radii_um: np.ndarray
    intersections: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"radius_um": self.radii_um, "n_intersections": self.intersections})


def compute_basic_metrics(tree: MorphologyTree) -> pd.Series:
    tip_path_lengths = [tree.path_length_to_root_um(nid) for nid in tree.tip_ids]
    edges = tree.edge_table()
    bbox = tree.bounding_box_um()
    metrics = {
        "n_nodes": int(len(tree.nodes)),
        "n_edges": int(len(edges)),
        "n_roots": int(len(tree.roots)),
        "n_branch_points": int(len(tree.branch_point_ids)),
        "n_tips": int(len(tree.tip_ids)),
        "total_cable_length_um": float(tree.total_cable_length_um()),
        "mean_edge_length_um": float(edges["length_um"].mean()) if not edges.empty else 0.0,
        "max_edge_length_um": float(edges["length_um"].max()) if not edges.empty else 0.0,
        "mean_radius_um": float(tree.nodes["radius_um"].mean()),
        "max_path_length_to_tip_um": float(max(tip_path_lengths)) if tip_path_lengths else 0.0,
        "mean_path_length_to_tip_um": float(np.mean(tip_path_lengths)) if tip_path_lengths else 0.0,
        "max_branch_order": int(tree.branch_orders().max()),
        "max_strahler_order": int(tree.strahler_orders().max()),
        **bbox,
    }
    return pd.Series(metrics)


def compute_sholl_intersections(
    tree: MorphologyTree,
    center_xyz_um: Optional[Sequence[float]] = None,
    radius_step_um: float = 5.0,
    max_radius_um: Optional[float] = None,
) -> ShollResult:
    if radius_step_um <= 0:
        raise ValueError("radius_step_um must be positive")

    if center_xyz_um is None:
        center_xyz_um = tree.get_xyz(tree.root_id)
    center = np.asarray(center_xyz_um, dtype=float)

    edges = tree.edge_table()
    if edges.empty:
        radii = np.arange(0.0, radius_step_um, radius_step_um)
        return ShollResult(radii_um=radii, intersections=np.zeros_like(radii, dtype=int))

    p0 = edges[["x0_um", "y0_um", "z0_um"]].to_numpy(dtype=float)
    p1 = edges[["x1_um", "y1_um", "z1_um"]].to_numpy(dtype=float)
    d0 = np.linalg.norm(p0 - center[None, :], axis=1)
    d1 = np.linalg.norm(p1 - center[None, :], axis=1)

    if max_radius_um is None:
        max_radius_um = float(max(d0.max(), d1.max()))

    radii = np.arange(0.0, max_radius_um + radius_step_um, radius_step_um)
    intersections = np.zeros_like(radii, dtype=int)
    for i, r in enumerate(radii):
        intersections[i] = int(np.sum(((d0 <= r) & (d1 > r)) | ((d1 <= r) & (d0 > r))))
    return ShollResult(radii_um=radii, intersections=intersections)


def compare_with_snt_measurements(tree: MorphologyTree, measurements: pd.DataFrame) -> pd.Series:
    ours = compute_basic_metrics(tree)
    comparison: Dict[str, float] = {"total_cable_length_um": float(ours["total_cable_length_um"])}
    if measurements is None or measurements.empty:
        return pd.Series(comparison)

    row = measurements.iloc[0]
    for their_key, ours_key in [
        ("Cable length (µm) [Single value]", "total_cable_length_um"),
        ("No. of branch points [Single value]", "n_branch_points"),
        ("No. of tips [Single value]", "n_tips"),
    ]:
        if their_key in row.index:
            comparison[f"snt::{their_key}"] = float(row[their_key])
            comparison[f"delta::{ours_key}"] = float(ours[ours_key]) - float(row[their_key])
    return pd.Series(comparison)
