"""Session-level asset containers used across VIP SLAP2 workflows.

This module defines the lightweight data structure used to pass resolved
session paths and metadata between registry, behavior, glutamate, calcium, QC,
and extraction routines. It intentionally contains no IO beyond optional
creation of configured output directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class SessionAssets:
    """Resolved file-system assets and metadata for one imaging session.

    The registry constructs this object after locating the SummaryLoCo MAT
    file, Bonsai event log, HARP directory, extracted HARP files, and analysis
    output directories for a session. Downstream modules accept this single
    object rather than separately passing many paths.

    Attributes
    ----------
    session_id
        Session identifier from the registry.
    subject_id
        Numeric subject identifier.
    session_dir
        Root directory for the session.
    summary_mat
        Optional path to the SLAP2 SummaryLoCo MAT file.
    bonsai_event_log_csv
        Optional path to the Bonsai/BonVision event log.
    harp_dir
        Optional path to the HARP behavior directory.
    photodiode_pkl
        Optional path to the extracted photodiode pickle.
    harp_df_csv
        Optional path to the extracted HARP digital-input CSV.
    qc_dir
        Optional directory for QC outputs.
    derived_dir
        Optional directory for derived analysis outputs.
    metadata
        Additional session-level metadata copied from the registry.
    """

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
        """Create configured QC and derived-data directories if present.

        The method is intentionally no-op for missing directory fields so that
        partially resolved assets can still be passed through validation code.
        """
        if self.qc_dir is not None:
            self.qc_dir.mkdir(parents=True, exist_ok=True)
        if self.derived_dir is not None:
            self.derived_dir.mkdir(parents=True, exist_ok=True)