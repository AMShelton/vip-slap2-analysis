"""Session-level asset containers used across VIP SLAP2 workflows.

This module defines the lightweight data structure used to pass resolved
session paths and metadata between registry, behavior, glutamate, calcium,
voltage, QC, and extraction routines. It intentionally contains no IO beyond
optional creation of configured output directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Mapping


@dataclass
class SessionAssets:
    """Resolved file-system assets and metadata for one imaging session.

    The registry constructs this object after locating the canonical session
    paths used by downstream behavior, glutamate, calcium, voltage, QC, and
    extraction workflows. Downstream modules accept this single object rather
    than separately passing many paths.

    ``summary_mat`` is retained as the canonical SLAP2 ``SummaryLoCo`` MAT file
    used by the existing glutamate/calcium pipeline. Additional modality-specific
    inputs, such as voltage ``dendriticVoltageSummary`` MAT files and paired H5
    trace files, are stored in ``modality_assets`` to avoid growing this dataclass
    with many modality-specific fields.

    Attributes
    ----------
    session_id
        Session identifier from the registry.
    subject_id
        Numeric subject identifier.
    session_dir
        Root directory for the session.
    summary_mat
        Optional path to the canonical SLAP2 SummaryLoCo MAT file.
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
    modality_assets
        Nested mapping of modality name to resolved asset paths. For example,
        ``modality_assets["voltage"]["summary_mat"]`` and
        ``modality_assets["voltage"]["trace_h5"]``.
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

    modality_assets: Dict[str, Dict[str, Optional[Path]]] = field(default_factory=dict)
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

    def get_modality_assets(self, modality: str) -> Dict[str, Optional[Path]]:
        """Return the resolved asset dictionary for ``modality``.

        Parameters
        ----------
        modality
            Modality key such as ``"glutamate"``, ``"calcium"``, or
            ``"voltage"``. Matching is case-insensitive.

        Returns
        -------
        dict
            Asset-name to path mapping. Missing modalities return an empty dict.
        """
        return dict(self.modality_assets.get(modality.lower(), {}))

    def get_asset(
        self,
        modality: str,
        name: str,
        default: Optional[Path] = None,
    ) -> Optional[Path]:
        """Return one resolved modality asset path if present.

        Examples
        --------
        ``asset.get_asset("voltage", "summary_mat")``
            Path to ``dendriticVoltageSummary-*.mat``.
        ``asset.get_asset("voltage", "trace_h5")``
            Path to ``dendriticVoltageTraces-*.h5``.
        """
        return self.modality_assets.get(modality.lower(), {}).get(name, default)

    def require_asset(self, modality: str, name: str) -> Path:
        """Return a required modality asset or raise a helpful error."""
        path = self.get_asset(modality, name)
        if path is None:
            raise FileNotFoundError(
                f"Missing required {modality!r} asset {name!r} for session "
                f"{self.session_id!r}."
            )
        return Path(path)

    def set_asset(self, modality: str, name: str, path: Optional[Path]) -> None:
        """Set or update one modality asset path.

        This is useful for notebooks or preprocessing code that discover a file
        after the registry has constructed the initial :class:`SessionAssets`.
        """
        key = modality.lower()
        self.modality_assets.setdefault(key, {})[name] = (
            Path(path) if path is not None else None
        )

    def has_modality(
        self,
        modality: str,
        required: Optional[Mapping[str, object]] = None,
    ) -> bool:
        """Return whether a modality is represented by resolved assets.

        Parameters
        ----------
        modality
            Modality key to test.
        required
            Optional mapping whose keys are required asset names. Values are
            ignored; this allows callers to pass a small dict literal.

        Returns
        -------
        bool
            True when the modality exists and, if ``required`` is supplied, all
            required asset names resolve to non-None paths.
        """
        assets = self.modality_assets.get(modality.lower())
        if not assets:
            return False
        if required is None:
            return any(path is not None for path in assets.values())
        return all(assets.get(name) is not None for name in required.keys())

    def qc_subdir(self, name: str, *, create: bool = False) -> Path:
        """Return ``qc_dir / name`` for modality-specific QC outputs."""
        if self.qc_dir is None:
            raise ValueError("asset.qc_dir must be set")
        path = Path(self.qc_dir) / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def derived_subdir(self, name: str, *, create: bool = False) -> Path:
        """Return ``derived_dir / name`` for modality-specific derived outputs."""
        if self.derived_dir is None:
            raise ValueError("asset.derived_dir must be set")
        path = Path(self.derived_dir) / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path
