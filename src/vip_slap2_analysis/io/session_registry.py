"""Session registry utilities for VIP SLAP2 analysis datasets.

This module reads the project-level session summary workbook and resolves
per-session file assets used by downstream behavior, glutamate, calcium, and
quality-control pipelines.

The registry intentionally keeps discovery lightweight: it normalizes the
summary tables, applies simple metadata filters, and locates the most recently
modified matching files under a session directory.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from vip_slap2_analysis.common.session import SessionAssets


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


def _find_one(base: Path, pattern: str) -> Optional[Path]:
    """Find the newest file or directory matching ``pattern`` below ``base``.

    Parameters
    ----------
    base:
        Root directory for the recursive search.
    pattern:
        Glob pattern to search for under ``base``.

    Returns
    -------
    pathlib.Path or None
        Most recently modified match, or ``None`` if no match exists.
    """
    matches = sorted(glob.glob(str(base / "**" / pattern), recursive=True))
    if matches:
        return Path(max(matches, key=os.path.getmtime))
    else:
        return None


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

    def resolve_assets(self, session: pd.Series | str) -> SessionAssets:
        """Resolve filesystem assets for a session row or session ID.

        Parameters
        ----------
        session:
            Either a session ID or a row from ``sessions_df``.

        Returns
        -------
        SessionAssets
            Object containing canonical session paths and row metadata.
        """
        row = self.get_session_row(session) if isinstance(session, str) else session
        session_dir = Path(row["session_dir"])

        summary_mat = _find_one(session_dir, "SummaryLoCo*.mat")
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
            metadata=row.to_dict(),
        )