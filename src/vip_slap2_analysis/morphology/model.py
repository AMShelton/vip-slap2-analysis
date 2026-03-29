from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MorphologyNode:
    """Single SWC node."""

    node_id: int
    node_type: int
    x_um: float
    y_um: float
    z_um: float
    radius_um: float
    parent_id: int

    @property
    def xyz_um(self) -> np.ndarray:
        return np.array([self.x_um, self.y_um, self.z_um], dtype=float)


@dataclass
class MorphologyTree:
    """In-memory morphology tree parsed from SWC."""

    nodes: pd.DataFrame
    metadata: Dict[str, str] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def __post_init__(self) -> None:
        required = ["node_id", "node_type", "x_um", "y_um", "z_um", "radius_um", "parent_id"]
        missing = [c for c in required if c not in self.nodes.columns]
        if missing:
            raise ValueError(f"MorphologyTree missing required columns: {missing}")

        self.nodes = self.nodes.copy()
        self.nodes = self.nodes.sort_values("node_id").reset_index(drop=True)
        self.nodes["node_id"] = self.nodes["node_id"].astype(int)
        self.nodes["node_type"] = self.nodes["node_type"].astype(int)
        self.nodes["parent_id"] = self.nodes["parent_id"].astype(int)

        self._node_index = {int(row.node_id): idx for idx, row in self.nodes.iterrows()}
        self._children = self._build_children_map()

    def _build_children_map(self) -> Dict[int, List[int]]:
        children: Dict[int, List[int]] = {int(nid): [] for nid in self.nodes["node_id"]}
        for _, row in self.nodes.iterrows():
            parent = int(row.parent_id)
            if parent != -1 and parent in children:
                children[parent].append(int(row.node_id))
        return children

    @property
    def node_ids(self) -> np.ndarray:
        return self.nodes["node_id"].to_numpy(dtype=int)

    @property
    def roots(self) -> List[int]:
        roots = self.nodes.loc[self.nodes["parent_id"] < 0, "node_id"].astype(int).tolist()
        if roots:
            return roots
        parent_ids = set(self.nodes["parent_id"].astype(int).tolist())
        return [nid for nid in self.node_ids.tolist() if nid not in parent_ids]

    @property
    def root_id(self) -> int:
        roots = self.roots
        if not roots:
            raise ValueError("Morphology contains no root node.")
        return int(roots[0])

    @property
    def child_counts(self) -> pd.Series:
        return pd.Series({nid: len(children) for nid, children in self._children.items()}, name="n_children")

    @property
    def tip_ids(self) -> List[int]:
        return [nid for nid, children in self._children.items() if len(children) == 0]

    @property
    def branch_point_ids(self) -> List[int]:
        return [nid for nid, children in self._children.items() if len(children) > 1]

    def get_row(self, node_id: int) -> pd.Series:
        return self.nodes.iloc[self._node_index[int(node_id)]]

    def get_xyz(self, node_id: int) -> np.ndarray:
        row = self.get_row(node_id)
        return row[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)

    def get_parent_id(self, node_id: int) -> int:
        return int(self.get_row(node_id)["parent_id"])

    def get_children_ids(self, node_id: int) -> List[int]:
        return list(self._children[int(node_id)])

    def edge_table(self) -> pd.DataFrame:
        rows = []
        for _, row in self.nodes.iterrows():
            child_id = int(row.node_id)
            parent_id = int(row.parent_id)
            if parent_id < 0:
                continue
            pxyz = self.get_xyz(parent_id)
            cxyz = row[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
            rows.append(
                {
                    "parent_id": parent_id,
                    "child_id": child_id,
                    "length_um": float(np.linalg.norm(cxyz - pxyz)),
                    "x0_um": float(pxyz[0]),
                    "y0_um": float(pxyz[1]),
                    "z0_um": float(pxyz[2]),
                    "x1_um": float(cxyz[0]),
                    "y1_um": float(cxyz[1]),
                    "z1_um": float(cxyz[2]),
                }
            )
        return pd.DataFrame(rows)

    def total_cable_length_um(self) -> float:
        edges = self.edge_table()
        if edges.empty:
            return 0.0
        return float(edges["length_um"].sum())

    def path_to_root(self, node_id: int) -> List[int]:
        path = [int(node_id)]
        seen = {int(node_id)}
        parent = self.get_parent_id(node_id)
        while parent >= 0:
            if parent in seen:
                raise ValueError("Cycle detected in morphology tree.")
            path.append(parent)
            seen.add(parent)
            parent = self.get_parent_id(parent)
        return path

    def path_length_to_root_um(self, node_id: int) -> float:
        total = 0.0
        current = int(node_id)
        while True:
            parent = self.get_parent_id(current)
            if parent < 0:
                break
            total += float(np.linalg.norm(self.get_xyz(current) - self.get_xyz(parent)))
            current = parent
        return total

    def branch_orders(self) -> pd.Series:
        order = {self.root_id: 0}
        queue = [self.root_id]
        while queue:
            current = queue.pop(0)
            current_order = order[current]
            for child in self.get_children_ids(current):
                order[child] = current_order + (1 if current in self.branch_point_ids else 0)
                queue.append(child)
        return pd.Series(order, name="branch_order")

    def strahler_orders(self) -> pd.Series:
        order: Dict[int, int] = {}

        def _compute(node_id: int) -> int:
            children = self.get_children_ids(node_id)
            if not children:
                order[node_id] = 1
                return 1
            child_orders = [_compute(child) for child in children]
            mx = max(child_orders)
            nmax = sum(co == mx for co in child_orders)
            val = mx + 1 if nmax >= 2 else mx
            order[node_id] = val
            return val

        for root in self.roots:
            _compute(root)
        return pd.Series(order, name="strahler_order")

    def bounding_box_um(self) -> Dict[str, float]:
        mins = self.nodes[["x_um", "y_um", "z_um"]].min()
        maxs = self.nodes[["x_um", "y_um", "z_um"]].max()
        spans = maxs - mins
        return {
            "x_min_um": float(mins["x_um"]),
            "y_min_um": float(mins["y_um"]),
            "z_min_um": float(mins["z_um"]),
            "x_max_um": float(maxs["x_um"]),
            "y_max_um": float(maxs["y_um"]),
            "z_max_um": float(maxs["z_um"]),
            "x_span_um": float(spans["x_um"]),
            "y_span_um": float(spans["y_um"]),
            "z_span_um": float(spans["z_um"]),
        }

    def with_node_annotations(self) -> pd.DataFrame:
        df = self.nodes.copy()
        df = df.merge(self.child_counts.rename("n_children"), left_on="node_id", right_index=True, how="left")
        branch_orders = self.branch_orders().rename("branch_order")
        strahler = self.strahler_orders().rename("strahler_order")
        df = df.merge(branch_orders, left_on="node_id", right_index=True, how="left")
        df = df.merge(strahler, left_on="node_id", right_index=True, how="left")
        df["is_tip"] = df["node_id"].isin(self.tip_ids)
        df["is_branch_point"] = df["node_id"].isin(self.branch_point_ids)
        df["path_length_to_root_um"] = [self.path_length_to_root_um(nid) for nid in df["node_id"]]
        return df

    def branch_segments(self) -> List[pd.DataFrame]:
        """Return proximal-to-distal polylines between critical nodes.

        Critical nodes are roots, branch points, and tips. Each returned dataframe contains
        contiguous samples along a single branch segment.
        """
        critical = set(self.roots) | set(self.branch_point_ids) | set(self.tip_ids)
        ann = self.nodes.copy().set_index("node_id")
        ann = ann.join(self.branch_orders().rename("branch_order"))
        ann = ann.join(self.strahler_orders().rename("strahler_order"))
        segments: List[pd.DataFrame] = []

        for start in sorted(critical):
            for child in self.get_children_ids(start):
                node_ids = [start, child]
                current = child
                while True:
                    if current in critical:
                        break
                    children = self.get_children_ids(current)
                    if len(children) != 1:
                        break
                    current = children[0]
                    node_ids.append(current)
                    if current in critical:
                        break

                seg = ann.loc[node_ids].reset_index()
                seg["segment_start_id"] = start
                seg["segment_end_id"] = node_ids[-1]
                seg["segment_branch_order"] = int(seg["branch_order"].max())
                seg["segment_strahler_order"] = int(seg["strahler_order"].max())
                segments.append(seg)
        return segments


@dataclass
class MorphologyBundle:
    """Convenience container for a reconstruction and associated SNT tables."""

    tree: MorphologyTree
    quick_measurements: Optional[pd.DataFrame] = None
    full_measurements: Optional[pd.DataFrame] = None
    sholl_table: Optional[pd.DataFrame] = None
    traces_path: Optional[Path] = None
    image_path: Optional[Path] = None
    extras: Dict[str, Path] = field(default_factory=dict)
