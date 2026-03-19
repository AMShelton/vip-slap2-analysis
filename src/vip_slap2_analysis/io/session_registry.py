from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from vip_slap2_analysis.common.session import SessionAssets


def _coerce_path(x) -> Optional[Path]:
    if pd.isna(x) or x is None:
        return None
    return Path(str(x))


def _find_one(base: Path, pattern: str) -> Optional[Path]:
    matches = sorted(glob.glob(str(base / "**" / pattern), recursive=True))
    if matches:
        return Path(max(matches, key=os.path.getmtime))
    else:
        None


@dataclass
class VIPSessionRegistry:
    summary_xlsx: Path
    subjects_df: pd.DataFrame
    sessions_df: pd.DataFrame

    @classmethod
    def from_basepath(cls, basepath: str | Path) -> "VIPSessionRegistry":
        basepath = Path(basepath)
        matches = sorted(glob.glob(str(basepath / "**summary.xlsx")))
        if not matches:
            raise FileNotFoundError(f"No *summary.xlsx found under {basepath}")
        return cls.from_excel(matches[0])

    @classmethod
    def from_excel(cls, summary_xlsx: str | Path) -> "VIPSessionRegistry":
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
        df = self.sessions_df[self.sessions_df["session_id"] == session_id]
        if len(df) == 0:
            raise KeyError(f"Session not found: {session_id}")
        if len(df) > 1:
            raise ValueError(f"Multiple rows found for session_id={session_id}")
        return df.iloc[0]

    def resolve_assets(self, session: pd.Series | str) -> SessionAssets:
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