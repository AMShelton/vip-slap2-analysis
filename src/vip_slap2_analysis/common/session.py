from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class SessionAssets:
    session_id: str
    subject_id: int
    session_dir: Path

    summary_mat: Optional[Path] = None
    bonsai_event_log_csv: Optional[Path] = None
    harp_dir: Optional[Path] = None
    photodiode_pkl: Optional[Path] = None
    harp_df_csv: Optional[Path] = None

    qc_dir: Optional[Path] = None
    derived_dir: Optional[Path] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        if self.qc_dir is not None:
            self.qc_dir.mkdir(parents=True, exist_ok=True)
        if self.derived_dir is not None:
            self.derived_dir.mkdir(parents=True, exist_ok=True)