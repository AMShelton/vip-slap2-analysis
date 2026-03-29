from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .model import MorphologyBundle, MorphologyTree


def read_swc(swc_path: str | Path) -> MorphologyTree:
    swc_path = Path(swc_path)
    metadata: Dict[str, str] = {}
    rows = []
    for line in swc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            body = stripped[1:].strip()
            if ":" in body:
                key, value = body.split(":", 1)
                metadata[key.strip()] = value.strip()
            continue
        parts = stripped.split()
        if len(parts) < 7:
            continue
        rows.append(
            {
                "node_id": int(parts[0]),
                "node_type": int(parts[1]),
                "x_um": float(parts[2]),
                "y_um": float(parts[3]),
                "z_um": float(parts[4]),
                "radius_um": float(parts[5]),
                "parent_id": int(parts[6]),
            }
        )
    if not rows:
        raise ValueError(f"No SWC nodes found in {swc_path}")
    return MorphologyTree(nodes=pd.DataFrame(rows), metadata=metadata, source_path=swc_path)


def _read_optional_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def load_snt_bundle(base_dir: str | Path, swc_name: Optional[str] = None) -> MorphologyBundle:
    """Load a morphology folder exported from Fiji/SNT.

    Expected contents include an SWC file and optionally SNT-generated CSV tables.
    """
    base_dir = Path(base_dir)
    if base_dir.is_file() and base_dir.suffix.lower() == ".swc":
        swc_path = base_dir
        base_dir = swc_path.parent
    else:
        if swc_name is not None:
            swc_path = base_dir / swc_name
        else:
            swcs = sorted(base_dir.glob("*.swc"))
            if not swcs:
                raise FileNotFoundError(f"No SWC files found in {base_dir}")
            swc_path = swcs[0]

    tree = read_swc(swc_path)
    quick = _read_optional_csv(base_dir / "QuickMeasurements.csv")
    full = _read_optional_csv(base_dir / "SNT_Measurements.csv")

    sholl_candidates = sorted(base_dir.glob("Sholl_Table*.csv"))
    sholl = _read_optional_csv(sholl_candidates[0]) if sholl_candidates else None

    traces_candidates = sorted(base_dir.glob("*.traces"))
    image_candidates = sorted(base_dir.glob("*.tif")) + sorted(base_dir.glob("*.tiff"))

    known = {swc_path.name, "QuickMeasurements.csv", "SNT_Measurements.csv"}
    known |= {p.name for p in sholl_candidates}
    known |= {p.name for p in traces_candidates}
    known |= {p.name for p in image_candidates}
    extras = {p.name: p for p in base_dir.iterdir() if p.name not in known}

    return MorphologyBundle(
        tree=tree,
        quick_measurements=quick,
        full_measurements=full,
        sholl_table=sholl,
        traces_path=traces_candidates[0] if traces_candidates else None,
        image_path=image_candidates[0] if image_candidates else None,
        extras=extras,
    )
