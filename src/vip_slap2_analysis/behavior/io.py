from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from vip_slap2_analysis.behavior.read_harp import HarpReader
# from vip_slap2_analysis.behavior.preprocess import process_single_harp_session  # move this import if needed

PathLike = Union[str, Path]


@dataclass
class BehaviorPaths:
    session_dir: Path
    bonsai_csv: Path
    harp_dir: Path
    extracted_dir: Path
    photodiode_pkl: Path
    harp_df_csv: Path
    corrected_bonsai_csv: Path
    qc_dir: Path


def resolve_behavior_paths(asset) -> BehaviorPaths:
    session_dir = Path(asset.session_dir)

    bonsai_csv = getattr(asset, "bonsai_event_log", None)
    if bonsai_csv is None:
        matches = glob.glob(str(session_dir / "**" / "bonsai_event_log*.csv"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No bonsai_event_log*.csv found under {session_dir}")
        bonsai_csv = Path(matches[0])
    else:
        bonsai_csv = Path(bonsai_csv)

    harp_dir = getattr(asset, "harp_dir", None)
    if harp_dir is None:
        matches = glob.glob(str(session_dir / "**" / "*Behavior.harp"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No *Behavior.harp directory found under {session_dir}")
        harp_dir = Path(matches[0])
    else:
        harp_dir = Path(harp_dir)

    extracted_dir = harp_dir / "extracted_files"
    photodiode_pkl = extracted_dir / "photodiode.pkl"
    harp_df_csv = extracted_dir / "HARP_df.csv"
    corrected_bonsai_csv = bonsai_csv

    qc_dir = Path(asset.qc_dir) / "behavior"
    qc_dir.mkdir(parents=True, exist_ok=True)

    return BehaviorPaths(
        session_dir=session_dir,
        bonsai_csv=bonsai_csv,
        harp_dir=harp_dir,
        extracted_dir=extracted_dir,
        photodiode_pkl=photodiode_pkl,
        harp_df_csv=harp_df_csv,
        corrected_bonsai_csv=corrected_bonsai_csv,
        qc_dir=qc_dir,
    )


def ensure_harp_extracted(
    paths: BehaviorPaths,
    *,
    overwrite: bool = False,
    harp_extract_fn=None,
) -> Dict[str, bool]:
    if harp_extract_fn is None:
        harp_extract_fn = process_single_harp_session

    reused = True

    if (not paths.photodiode_pkl.exists()) or overwrite:
        harp_extract_fn(paths.harp_dir, save=True, overwrite=overwrite)
        reused = False

    if (not paths.harp_df_csv.exists()) or overwrite:
        harp_df, _ = load_harp_df(paths.harp_dir)
        paths.extracted_dir.mkdir(parents=True, exist_ok=True)
        harp_df.to_csv(paths.harp_df_csv, index=False)
        reused = False

    return {"reused_existing_extracted_files": reused}


def load_harp_df(harp_dir: PathLike) -> Tuple[pd.DataFrame, np.ndarray]:
    harp_handler = HarpReader(harp_dir)
    harp_df = harp_handler.reader.DigitalInputState.read().copy()
    harp_df["time"] = harp_df.index.astype(float)
    harp_df = harp_df.sort_values("time").reset_index(drop=True)
    acq_time = (harp_df["time"] - harp_df["time"].iloc[0]).to_numpy(dtype=float)
    return harp_df, acq_time


def load_photodiode_df(photodiode_pkl: PathLike) -> pd.DataFrame:
    return pd.read_pickle(photodiode_pkl).copy()


def load_bonsai_df(bonsai_csv: PathLike) -> pd.DataFrame:
    return pd.read_csv(bonsai_csv)


def save_epochs_csv(epoch_df: pd.DataFrame, out_csv: PathLike) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    epoch_df.to_csv(out_csv, index=False)