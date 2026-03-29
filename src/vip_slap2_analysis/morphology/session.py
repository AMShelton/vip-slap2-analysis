from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MorphologyAssets:
    session_id: str
    session_dir: Path
    morphology_dir: Optional[Path] = None
    swc_files: List[Path] = field(default_factory=list)
    tracing_files: List[Path] = field(default_factory=list)
    measurement_csvs: List[Path] = field(default_factory=list)
    prepared_tiffs: List[Path] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


MORPHOLOGY_DIR_NAMES = [
    "morphology",
    "reconstructions",
    "reconstruction",
    "morph",
]


def discover_morphology_assets(session_dir: str | Path, session_id: Optional[str] = None) -> MorphologyAssets:
    session_dir = Path(session_dir)
    if session_id is None:
        session_id = session_dir.name

    candidate_dirs = [p for p in session_dir.rglob("*") if p.is_dir() and p.name.lower() in MORPHOLOGY_DIR_NAMES]
    if not candidate_dirs:
        candidate_dirs = [p for p in session_dir.rglob("*") if p.is_dir() and any(q in p.name.lower() for q in MORPHOLOGY_DIR_NAMES)]

    morphology_dir = candidate_dirs[0] if candidate_dirs else None
    search_root = morphology_dir if morphology_dir is not None else session_dir

    swc_files = sorted(search_root.rglob("*.swc"))
    tracing_files = sorted(search_root.rglob("*.traces"))
    measurement_csvs = sorted([p for p in search_root.rglob("*.csv") if "measurement" in p.name.lower() or "sholl" in p.name.lower()])
    prepared_tiffs = sorted(search_root.rglob("*.tif")) + sorted(search_root.rglob("*.tiff"))

    return MorphologyAssets(
        session_id=session_id,
        session_dir=session_dir,
        morphology_dir=morphology_dir,
        swc_files=swc_files,
        tracing_files=tracing_files,
        measurement_csvs=measurement_csvs,
        prepared_tiffs=prepared_tiffs,
    )
