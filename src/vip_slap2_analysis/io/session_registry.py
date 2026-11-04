"""Session registry utilities for VIP SLAP2 analysis datasets.

This module reads the project-level session summary workbook and resolves
per-session file assets used by downstream behavior, glutamate, calcium,
voltage, and quality-control pipelines.

The registry intentionally keeps discovery lightweight: it normalizes the
summary tables, applies simple metadata filters, and locates the most recently
modified matching files under a session directory. Modality-specific processed
assets are exposed through ``SessionAssets.modality_assets`` rather than through
separate asset dataclasses.
"""

from __future__ import annotations

import os
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from vip_slap2_analysis.common.session import SessionAssets


VOLTAGE_SUMMARY_RE = re.compile(
    r"dendriticVoltageSummary[-_](?P<stamp>\d{6}-\d{6})\.mat$",
    re.IGNORECASE,
)


# Preferred path fragments are used only as tie-breakers. Recursive newest-file
# discovery is still the final fallback, which keeps the resolver tolerant of old
# session layouts and partially reorganized sessions.
PREFERRED_SUMMARY_PARTS = (
    ("source_extraction", "ExperimentSummary"),
    ("ExperimentSummary",),
)
PREFERRED_VOLTAGE_PARTS = (
    ("source_extraction", "dendriticVoltageExtraction"),
    ("dendriticVoltageExtraction",),
)

# Modern (HDF5) products. GIAnT writes ``source_extraction/<method>/experiment_summary.h5``
# with ``<method>`` in {summarize_LoCo, SILo}; the voltage extractor writes
# ``<processed>/Voltage/VoltageSummary.h5`` (pilot sessions used ``voltage_summary.h5``).
# ``summarize_LoCo`` is listed first because it is the like-for-like successor of the
# legacy ``SummaryLoCo-*.mat`` product; SILo output is still kept as a candidate.
PREFERRED_GIANT_PARTS = (
    ("source_extraction", "summarize_LoCo"),
    ("source_extraction", "SILo"),
    ("source_extraction",),
)
PREFERRED_VOLTEX_PARTS = (
    ("source_extraction", "Voltage"),
    ("Voltage",),
)

# Which summary family extraction should load when both exist.
#   "modern" : validated HDF5 product first, legacy MAT as fallback (default)
#   "legacy" : legacy MAT first (modern still discovered and recorded)
SUMMARY_PREFERENCE_DEFAULT = "modern"


def _coerce_path(x) -> Optional[Path]:
    """Convert a spreadsheet path-like value to ``Path`` or ``None``.

    Parameters
    ----------
    x:
        Value read from the summary spreadsheet. Missing values, including
        pandas ``NaN`` entries, are treated as absent paths.

    Returns
    -------
    pathlib.Path or None
        Normalized path when ``x`` is present; otherwise ``None``.
    """
    if pd.isna(x) or x is None:
        return None
    return Path(str(x))


def _find_matches(base: Path, pattern: str) -> list[Path]:
    """Return all recursive matches for ``pattern`` below ``base``."""
    if base is None or not Path(base).exists():
        return []
    return [Path(p) for p in sorted(glob.glob(str(Path(base) / "**" / pattern), recursive=True))]


def _path_contains_parts(path: Path, parts: Sequence[str]) -> bool:
    """Return True when ``path`` contains all ``parts`` in order."""
    lower_parts = [p.lower() for p in path.parts]
    start = 0
    for part in parts:
        target = part.lower()
        try:
            idx = lower_parts.index(target, start)
        except ValueError:
            return False
        start = idx + 1
    return True


def _rank_path(path: Path, preferred_parts: Sequence[Sequence[str]] = ()) -> tuple[int, float]:
    """Return a sortable rank favoring preferred layout fragments, then mtime."""
    preference = 0
    for idx, parts in enumerate(preferred_parts):
        if _path_contains_parts(path, parts):
            preference = len(preferred_parts) - idx
            break
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return preference, mtime


def _find_one(
    base: Path,
    pattern: str,
    *,
    preferred_parts: Sequence[Sequence[str]] = (),
) -> Optional[Path]:
    """Find the best file or directory matching ``pattern`` below ``base``.

    Parameters
    ----------
    base:
        Root directory for the recursive search.
    pattern:
        Glob pattern to search for under ``base``.
    preferred_parts:
        Optional ordered path-fragment tuples used to prefer canonical processed
        layouts before falling back to newest modified match.

    Returns
    -------
    pathlib.Path or None
        Best matching path, or ``None`` if no match exists.
    """
    matches = _find_matches(base, pattern)
    if matches:
        return max(matches, key=lambda p: _rank_path(p, preferred_parts))
    return None


def _voltage_stamp(path: Path) -> Optional[str]:
    """Extract the timestamp shared by paired voltage MAT/H5 files."""
    match = VOLTAGE_SUMMARY_RE.search(path.name)
    return match.group("stamp") if match else None


def _find_voltage_trace_h5(session_dir: Path, summary_mat: Optional[Path]) -> Optional[Path]:
    """Resolve the paired ``dendriticVoltageTraces`` H5 file for a summary MAT."""
    if summary_mat is not None:
        stamp = _voltage_stamp(summary_mat)
        if stamp is not None:
            same_dir = summary_mat.with_name(f"dendriticVoltageTraces-{stamp}.h5")
            if same_dir.exists():
                return same_dir
            paired = _find_one(
                session_dir,
                f"dendriticVoltageTraces-{stamp}.h5",
                preferred_parts=PREFERRED_VOLTAGE_PARTS,
            )
            if paired is not None:
                return paired

        # Same-directory fallback catches future filename variants where the
        # summary exists but the timestamp pattern changes slightly.
        same_dir_matches = sorted(summary_mat.parent.glob("dendriticVoltageTraces*.h5"))
        if same_dir_matches:
            return max(same_dir_matches, key=os.path.getmtime)

    return _find_one(
        session_dir,
        "dendriticVoltageTraces*.h5",
        preferred_parts=PREFERRED_VOLTAGE_PARTS,
    )


def _find_trial_table_for(summary_h5: Optional[Path]) -> Optional[Path]:
    """Locate ``trial_table.h5`` next to or one level above a modern summary file."""
    if summary_h5 is None:
        return None
    # GIAnT: <processed>/trial_table.h5 with the summary at
    # <processed>/source_extraction/<method>/experiment_summary.h5 (or one level less);
    # Voltex: <processed>/trial_table.h5 with <processed>/Voltage/VoltageSummary.h5.
    base = summary_h5.parent
    for _ in range(3):
        cand = base / "trial_table.h5"
        if cand.exists():
            return cand
        base = base.parent
    return None


def _find_dynamic_data_dir(session_dir: Path) -> Optional[Path]:
    """Locate the raw SLAP2 ``dynamic_data`` directory (holds ``.dat``/``.meta`` files).

    Only directories that actually contain a ``.meta`` or ``.dat`` file are
    accepted; the newest match wins when several exist (nested layouts can
    duplicate the session folder).
    """
    hits = []
    for p in _find_matches(session_dir, "dynamic_data"):
        if p.is_dir() and (any(p.glob("*.meta")) or any(p.glob("*.dat"))):
            hits.append(p)
    if not hits:
        return None
    return max(hits, key=lambda p: (len(list(p.glob("*.meta"))), os.path.getmtime(p)))


def _select_summary(
    *,
    modern: Optional[Path],
    legacy: Optional[Path],
    modern_format: str,
    legacy_format: str,
    prefer: str,
) -> dict[str, Optional[object]]:
    """Apply the modern -> legacy precedence and record the decision.

    Existence is all that is known at registry time; schema validation happens in
    the reader (``validate_*_schema``) and extraction falls back to the legacy
    product -- with the reason recorded in QC -- when the modern file is
    incompatible. Nothing is silently merged across formats.
    """
    prefer = str(prefer or SUMMARY_PREFERENCE_DEFAULT).lower()
    if prefer not in ("modern", "legacy"):
        raise ValueError("summary preference must be 'modern' or 'legacy'")
    order = [(modern, modern_format), (legacy, legacy_format)]
    if prefer == "legacy":
        order.reverse()
    selected, fmt = next(((p, f) for p, f in order if p is not None), (None, None))
    return {
        "summary_path": selected,
        "summary_format": fmt,
        "summary_preference": prefer,
        "summary_candidates": {
            modern_format: str(modern) if modern is not None else None,
            legacy_format: str(legacy) if legacy is not None else None,
        },
    }


def _build_modality_assets(
    *,
    summary_mat: Optional[Path],
    voltage_summary_mat: Optional[Path],
    voltage_trace_h5: Optional[Path],
    giant_summary_h5: Optional[Path] = None,
    giant_trial_table_h5: Optional[Path] = None,
    voltex_summary_h5: Optional[Path] = None,
    voltex_trial_table_h5: Optional[Path] = None,
    dynamic_data_dir: Optional[Path] = None,
    summary_preference: str = SUMMARY_PREFERENCE_DEFAULT,
) -> dict[str, dict[str, Optional[object]]]:
    """Build generic modality asset mappings for a session.

    Legacy keys (``summary_mat``, ``trace_h5``, ``extraction_dir``) are unchanged.
    New keys: ``summary_h5`` (modern product), ``trial_table_h5``,
    ``summary_path``/``summary_format`` (the product extraction should load under
    the current precedence), ``summary_candidates`` and ``summary_preference``.
    ``slap2`` additionally exposes ``dynamic_data_dir`` for raw-header access.
    """
    modality_assets: dict[str, dict[str, Optional[object]]] = {}

    if summary_mat is not None or giant_summary_h5 is not None:
        # Existing glutamate and soma-calcium code both use SummaryLoCo-derived
        # ExperimentSummary files. Store both keys explicitly so modality-aware
        # downstream code can be generic without changing legacy asset.summary_mat.
        selection = _select_summary(
            modern=giant_summary_h5, legacy=summary_mat,
            modern_format="giant_experiment_summary_h5", legacy_format="legacy_summary_loco_mat",
            prefer=summary_preference,
        )
        extraction_dir = (
            summary_mat.parent if summary_mat is not None else giant_summary_h5.parent  # type: ignore[union-attr]
        )
        for key in ("glutamate", "calcium", "slap2"):
            modality_assets[key] = {
                "summary_mat": summary_mat,
                "summary_h5": giant_summary_h5,
                "trial_table_h5": giant_trial_table_h5,
                "extraction_dir": extraction_dir,
                **selection,
            }

    if voltage_summary_mat is not None or voltage_trace_h5 is not None or voltex_summary_h5 is not None:
        extraction_dir = None
        if voltage_summary_mat is not None:
            extraction_dir = voltage_summary_mat.parent
        elif voltage_trace_h5 is not None:
            extraction_dir = voltage_trace_h5.parent
        elif voltex_summary_h5 is not None:
            extraction_dir = voltex_summary_h5.parent
        legacy_voltage = voltage_summary_mat if (voltage_summary_mat is not None and voltage_trace_h5 is not None) else None
        selection = _select_summary(
            modern=voltex_summary_h5, legacy=legacy_voltage,
            modern_format="voltex_voltage_summary_h5", legacy_format="legacy_dendritic_voltage_mat_h5",
            prefer=summary_preference,
        )
        modality_assets["voltage"] = {
            "summary_mat": voltage_summary_mat,
            "trace_h5": voltage_trace_h5,
            "summary_h5": voltex_summary_h5,
            "trial_table_h5": voltex_trial_table_h5,
            "extraction_dir": extraction_dir,
            **selection,
        }

    if dynamic_data_dir is not None:
        modality_assets.setdefault("slap2", {})
        modality_assets["slap2"]["dynamic_data_dir"] = dynamic_data_dir
        modality_assets["slap2"]["meta_files"] = sorted(dynamic_data_dir.glob("*.meta"))

    return modality_assets


@dataclass
class VIPSessionRegistry:
    """Registry of VIP SLAP2 subjects, sessions, and resolved file assets.

    The registry is backed by the project summary workbook. It provides a
    table-level API for filtering sessions and a resolver that converts a
    session row into a :class:`vip_slap2_analysis.common.session.SessionAssets`
    object for downstream processing.

    Attributes
    ----------
    summary_xlsx:
        Path to the summary workbook used to construct the registry.
    subjects_df:
        DataFrame containing subject-level metadata.
    sessions_df:
        DataFrame containing session-level metadata.
    """
    summary_xlsx: Path
    subjects_df: pd.DataFrame
    sessions_df: pd.DataFrame

    @classmethod
    def from_basepath(cls, basepath: str | Path) -> "VIPSessionRegistry":
        """Construct a registry from the first ``*summary.xlsx`` under a root.

        Parameters
        ----------
        basepath:
            Directory to search for the project summary workbook.

        Returns
        -------
        VIPSessionRegistry
            Registry loaded from the discovered workbook.

        Raises
        ------
        FileNotFoundError
            If no matching summary workbook exists under ``basepath``.
        """
        basepath = Path(basepath)
        matches = sorted(glob.glob(str(basepath / "**summary.xlsx")))
        if not matches:
            raise FileNotFoundError(f"No *summary.xlsx found under {basepath}")
        return cls.from_excel(matches[0])

    @classmethod
    def from_excel(cls, summary_xlsx: str | Path) -> "VIPSessionRegistry":
        """Load subject and session tables from a summary workbook.

        Parameters
        ----------
        summary_xlsx:
            Path to the Excel workbook containing ``sessions`` and ``subjects``
            sheets.

        Returns
        -------
        VIPSessionRegistry
            Registry with normalized path and date columns.
        """
        summary_xlsx = Path(summary_xlsx)

        # sessions sheet is tidy already
        sessions_df = pd.read_excel(summary_xlsx, sheet_name="sessions").copy()

        # subjects sheet has header lower down in your current file
        subjects_df = pd.read_excel(summary_xlsx, sheet_name="subjects", header=3).copy()

        # normalize some useful columns
        if "session_dir" in sessions_df.columns:
            sessions_df["session_dir"] = sessions_df["session_dir"].map(_coerce_path)

        if "session_date" in sessions_df.columns:
            sessions_df["session_date"] = pd.to_datetime(sessions_df["session_date"], errors="coerce")

        if "data_dir" in subjects_df.columns:
            subjects_df["data_dir"] = subjects_df["data_dir"].map(_coerce_path)

        return cls(
            summary_xlsx=summary_xlsx,
            subjects_df=subjects_df,
            sessions_df=sessions_df,
        )

    def sessions(
        self,
        subject_ids: Optional[Sequence[int]] = None,
        session_types: Optional[Sequence[str]] = None,
        paradigms: Optional[Sequence[str]] = None,
        indicators: Optional[Sequence[str]] = None,
        min_quality: Optional[str] = None,
        exclude_session_types: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return session rows matching optional metadata filters.

        Parameters
        ----------
        subject_ids:
            Subject IDs to include.
        session_types:
            Session types to include.
        paradigms:
            Behavioral or experimental paradigms to include.
        indicators:
            Values from the ``indicator1`` column to include.
        min_quality:
            Reserved for future quality filtering. The current implementation
            accepts the argument but does not apply it.
        exclude_session_types:
            Session types to remove after any inclusion filters.

        Returns
        -------
        pandas.DataFrame
            Filtered copy of the session table with a reset integer index.
        """
        df = self.sessions_df.copy()

        if subject_ids is not None:
            df = df[df["subject_id"].isin(subject_ids)]

        if session_types is not None:
            df = df[df["session_type"].isin(session_types)]

        if exclude_session_types is not None:
            df = df[~df["session_type"].isin(exclude_session_types)]

        if paradigms is not None:
            df = df[df["paradigm"].isin(paradigms)]

        if indicators is not None:
            df = df[df["indicator1"].isin(indicators)]

        return df.reset_index(drop=True)

    def get_session_row(self, session_id: str) -> pd.Series:
        """Return the unique session table row for ``session_id``.

        Parameters
        ----------
        session_id:
            Session identifier to retrieve.

        Returns
        -------
        pandas.Series
            Matching row from ``sessions_df``.

        Raises
        ------
        KeyError
            If the session ID is not present.
        ValueError
            If multiple rows share the same session ID.
        """
        df = self.sessions_df[self.sessions_df["session_id"] == session_id]
        if len(df) == 0:
            raise KeyError(f"Session not found: {session_id}")
        if len(df) > 1:
            raise ValueError(f"Multiple rows found for session_id={session_id}")
        return df.iloc[0]

    def resolve_assets(
        self,
        session: pd.Series | str,
        *,
        summary_preference: str = SUMMARY_PREFERENCE_DEFAULT,
    ) -> SessionAssets:
        """Resolve filesystem assets for a session row or session ID.

        Parameters
        ----------
        session:
            Either a session ID or a row from ``sessions_df``.
        summary_preference:
            ``"modern"`` (default) selects GIAnT ``experiment_summary.h5`` /
            Voltex ``VoltageSummary.h5`` as the product extraction should load,
            with the legacy MAT products as fallback; ``"legacy"`` reverses the
            order. Both families are always discovered and recorded.

        Returns
        -------
        SessionAssets
            Object containing canonical session paths, generic modality assets,
            and row metadata.
        """
        row = self.get_session_row(session) if isinstance(session, str) else session
        session_dir = Path(row["session_dir"])

        summary_mat = _find_one(
            session_dir,
            "SummaryLoCo*.mat",
            preferred_parts=PREFERRED_SUMMARY_PARTS,
        )
        voltage_summary_mat = _find_one(
            session_dir,
            "dendriticVoltageSummary*.mat",
            preferred_parts=PREFERRED_VOLTAGE_PARTS,
        )
        voltage_trace_h5 = _find_voltage_trace_h5(session_dir, voltage_summary_mat)

        giant_summary_h5 = _find_one(
            session_dir,
            "experiment_summary.h5",
            preferred_parts=PREFERRED_GIANT_PARTS,
        )
        voltex_summary_h5 = _find_one(session_dir, "VoltageSummary.h5", preferred_parts=PREFERRED_VOLTEX_PARTS)
        if voltex_summary_h5 is None:
            voltex_summary_h5 = _find_one(session_dir, "voltage_summary.h5", preferred_parts=PREFERRED_VOLTEX_PARTS)
        dynamic_data_dir = _find_dynamic_data_dir(session_dir)

        bonsai_csv = _find_one(session_dir, "bonsai_event_log*.csv")
        harp_dir = _find_one(session_dir, "*Behavior.harp")

        photodiode_pkl = None
        harp_df_csv = None

        if harp_dir is not None:
            photodiode_pkl = _find_one(harp_dir, "photodiode*.pkl")
            harp_df_csv = _find_one(harp_dir, "HARP_df*.csv")

        qc_dir = session_dir / "analysis" / "qc"
        derived_dir = session_dir / "analysis" / "derived"

        return SessionAssets(
            session_id=str(row["session_id"]),
            subject_id=int(row["subject_id"]),
            session_dir=session_dir,
            summary_mat=summary_mat,
            bonsai_event_log_csv=bonsai_csv,
            harp_dir=harp_dir,
            photodiode_pkl=photodiode_pkl,
            harp_df_csv=harp_df_csv,
            qc_dir=qc_dir,
            derived_dir=derived_dir,
            modality_assets=_build_modality_assets(
                summary_mat=summary_mat,
                voltage_summary_mat=voltage_summary_mat,
                voltage_trace_h5=voltage_trace_h5,
                giant_summary_h5=giant_summary_h5,
                giant_trial_table_h5=_find_trial_table_for(giant_summary_h5),
                voltex_summary_h5=voltex_summary_h5,
                voltex_trial_table_h5=_find_trial_table_for(voltex_summary_h5),
                dynamic_data_dir=dynamic_data_dir,
                summary_preference=summary_preference,
            ),
            metadata=row.to_dict(),
        )
