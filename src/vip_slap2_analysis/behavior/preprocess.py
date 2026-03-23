from __future__ import annotations

import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from read_harp import HarpReader
from dataclasses import dataclass, asdict
from scipy.signal import medfilt
from typing import Optional, Tuple, Dict, Any, List,Union

from .io import (
    BehaviorPaths,
    ensure_harp_extracted,
    load_bonsai_df,
    load_harp_df,
    load_photodiode_df,
    resolve_behavior_paths,
    save_epochs_csv,
)
from .validation import (
    audit_event_coverage,
    validate_bonsai_event_log,
    validate_harp_data,
)
from .epochs import (
    detect_epochs_adaptive,
    epochs_to_dataframe,
    shift_epochs_to_photodiode_time,
    summarize_epochs,
)
@dataclass
class BehaviorProcessingResult:
    paths: BehaviorPaths
    bonsai_validation: Dict[str, Any]
    harp_validation: Dict[str, Any]
    alignment_meta: Dict[str, Any]
    epochs_summary: Dict[str, Any]
    event_coverage: Dict[str, Any]
    ready_for_physiology_extraction: bool
    status: str = "success"
    failure_stage: Optional[str] = None
    failure_reasons: Optional[List[str]] = None

def _write_behavior_failure_metadata(
    paths: Path,
    *,
    status: str,
    failure_stage: str,
    failure_reasons: list[str],
    bonsai_validation: dict | None = None,
    harp_validation: dict | None = None,
    alignment_meta: dict | None = None,
    epochs_summary: dict | None = None,
    event_coverage: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    out = {
        "schema_version": "0.1.0",
        "status": status,
        "failure_stage": failure_stage,
        "failure_reasons": failure_reasons,
        "ready_for_physiology_extraction": False,
        "bonsai_validation": bonsai_validation or {},
        "harp_validation": harp_validation or {},
        "alignment": alignment_meta or {},
        "epochs": epochs_summary or {},
        "event_coverage": event_coverage or {},
        "prepost_sec": (metadata or {}).get("prepost_sec", None),
    }

    with open(Path(paths.qc_dir) / "behavior_validation.json", "w") as f:
        json.dump(out, f, indent=2)

    return out

def make_behavior_qc_plots(
    harp_df: pd.DataFrame,
    photodiode_df: pd.DataFrame,
    epoch_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(harp_df["time"], harp_df["DI3"], lw=0.5)
    for _, row in epoch_df.iterrows():
        ax.axvspan(row["start_time"], row["end_time"], alpha=0.2)
    ax.set_title("DI3 with detected imaging epochs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DI3")
    fig.tight_layout()
    fig.savefig(out_dir / "di3_epochs.png", dpi=200)
    plt.close(fig)

    pd_col = photodiode_df.columns[0]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(photodiode_df.index, photodiode_df[pd_col], lw=0.5)
    ax.set_title("Photodiode trace")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(pd_col)
    fig.tight_layout()
    fig.savefig(out_dir / "photodiode_trace.png", dpi=200)
    plt.close(fig)


def process_behavior_session(
    asset,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite_harp_extract: bool = False,
    overwrite_alignment: bool = False,
    save_corrected_in_place: bool = True,
    min_epoch_duration: float = 6.0,
    trial_gap_start: float = 0.02,
    expected_trial_epoch_min: Optional[int] = None,
    use_piecewise_warp: bool = True,
) -> BehaviorProcessingResult:
    metadata = metadata or {}
    paths = resolve_behavior_paths(asset)

    reuse_info = ensure_harp_extracted(
        paths,
        overwrite=overwrite_harp_extract,
        harp_extract_fn=process_single_harp_session,
    )

    stim_df = load_bonsai_df(paths.bonsai_csv)
    harp_df, acq_time = load_harp_df(paths.harp_dir)
    photodiode_df = load_photodiode_df(paths.photodiode_pkl)

    bonsai_validation = validate_bonsai_event_log(stim_df)
    harp_validation = validate_harp_data(harp_df, photodiode_df)

    if not bonsai_validation["passed"]:
        _write_behavior_failure_metadata(
            paths,
            status="failed_validation",
            failure_stage="bonsai_validation",
            failure_reasons=bonsai_validation["warnings"],
            bonsai_validation=bonsai_validation,
            harp_validation={},
            alignment_meta={},
            epochs_summary={},
            event_coverage={},
            metadata=metadata,
        )
        return BehaviorProcessingResult(
            paths=paths,
            bonsai_validation=bonsai_validation,
            harp_validation={},
            alignment_meta={},
            epochs_summary={},
            event_coverage={},
            ready_for_physiology_extraction=False,
            status="failed_validation",
            failure_stage="bonsai_validation",
            failure_reasons=bonsai_validation["warnings"],
        )

    if not harp_validation["passed"]:
        _write_behavior_failure_metadata(
            paths,
            status="failed_validation",
            failure_stage="harp_validation",
            failure_reasons=harp_validation["warnings"],
            bonsai_validation=bonsai_validation,
            harp_validation=harp_validation,
            alignment_meta={},
            epochs_summary={},
            event_coverage={},
            metadata=metadata,
        )
        return BehaviorProcessingResult(
            paths=paths,
            bonsai_validation=bonsai_validation,
            harp_validation=harp_validation,
            alignment_meta={},
            epochs_summary={},
            event_coverage={},
            ready_for_physiology_extraction=False,
            status="failed_validation",
            failure_stage="harp_validation",
            failure_reasons=harp_validation["warnings"],
        )

    already_corrected = "corrected_timestamps" in stim_df.columns
    if already_corrected and not overwrite_alignment:
        corrected_df = stim_df.copy()
        alignment_meta = {
            "alignment_method": "existing_corrected_timestamps",
            "reused_existing_corrected_event_log": True,
        }
    else:
        corrected_df, meta = correct_event_log(
            paths.bonsai_csv,
            paths.photodiode_pkl,
            savepath=paths.corrected_bonsai_csv if save_corrected_in_place else None,
            qc_dir=paths.qc_dir,
            use_piecewise_warp=use_piecewise_warp,
            insert_missing_first_stim_rows=True,
        )
        alignment_meta = asdict(meta)
        alignment_meta["reused_existing_corrected_event_log"] = False

        if not save_corrected_in_place:
            corrected_path = paths.qc_dir / "bonsai_event_log_corrected.csv"
            corrected_df.to_csv(corrected_path, index=False)
            paths.corrected_bonsai_csv = corrected_path

    acq_type = metadata.get("epochs_mode", "continuous")
    epochs, gap_used = detect_epochs_adaptive(
        harp_df,
        acq_time,
        acq_type=acq_type,
        min_duration=min_epoch_duration,
        gap_start=trial_gap_start,
        target_min=expected_trial_epoch_min,
    )
    epochs = shift_epochs_to_photodiode_time(epochs, harp_df, photodiode_df)
    epoch_df = epochs_to_dataframe(epochs)
    save_epochs_csv(epoch_df, paths.qc_dir / "imaging_epochs.csv")

    event_coverage = audit_event_coverage(corrected_df, epoch_df)
    epochs_summary = summarize_epochs(epoch_df, mode=acq_type, gap_threshold_used=gap_used)

    ready = (
        bonsai_validation["passed"]
        and harp_validation["passed"]
        and epochs_summary["passed"]
        and (event_coverage["image_identity_in_epochs"] > 0)
    )

    behavior_meta = {
        "schema_version": "0.1.0",
        "session_dir": str(paths.session_dir),
        "bonsai_event_log": str(paths.bonsai_csv),
        "corrected_bonsai_event_log": str(paths.corrected_bonsai_csv),
        "harp_dir": str(paths.harp_dir),
        "photodiode_pkl": str(paths.photodiode_pkl),
        "harp_df_csv": str(paths.harp_df_csv),
        **reuse_info,
        "bonsai_validation": bonsai_validation,
        "harp_validation": harp_validation,
        "alignment": alignment_meta,
        "epochs": epochs_summary,
        "event_coverage": event_coverage,
        "prepost_sec": metadata.get("prepost_sec", None),
        "ready_for_physiology_extraction": ready,
    }

    with open(paths.qc_dir / "behavior_validation.json", "w") as f:
        json.dump(behavior_meta, f, indent=2)

    with open(paths.qc_dir / "alignment_meta.json", "w") as f:
        json.dump(alignment_meta, f, indent=2)

    with open(paths.qc_dir / "imaging_epochs_summary.json", "w") as f:
        json.dump(epochs_summary, f, indent=2)

    make_behavior_qc_plots(harp_df, photodiode_df, epoch_df, paths.qc_dir)

    return BehaviorProcessingResult(
        paths=paths,
        bonsai_validation=bonsai_validation,
        harp_validation=harp_validation,
        alignment_meta=alignment_meta,
        epochs_summary=epochs_summary,
        event_coverage=event_coverage,
        ready_for_physiology_extraction=ready,
    )


def process_single_harp_session(session_path, save=True, overwrite=False):
    """
    Process a single HARP session folder.

    Parameters:
    -----------
    session_path : str or Path
        Path to the directory containing HARP binary files.
    save : bool
        If True, saves .pkl files for encoder, photodiode, licks, and rewards.
    overwrite : bool
        If True, existing extracted files will be overwritten.
    """
    session_path = Path(session_path)
    try:
        print(f"Processing {session_path}...")
        reader = HarpReader(session_path)
        extracted_dir = session_path / "extracted_files"
        
        if extracted_dir.exists() and not overwrite:
            print(f"→ Skipping {session_path.name}: already processed.")
            return

        if save:
            extracted_dir.mkdir(exist_ok=True)
            reader.get_encoder.to_pickle(extracted_dir / 'encoder.pkl')
            reader.get_photodiode.to_pickle(extracted_dir / 'photodiode.pkl')
            reader.get_licks.to_pickle(extracted_dir / 'licks.pkl')
            reader.get_rewards.to_pickle(extracted_dir / 'rewards.pkl')
            print(f"→ Saved data to {extracted_dir}")
        else:
            print(reader.get_encoder.head())
            print(reader.get_photodiode.head())
            print(reader.get_licks.head())
            print(reader.get_rewards.head())
    except Exception as e:
        print(f"❌ Error processing {session_path.name}: {e}")

PathLike = Union[str, Path]

@dataclass
class AlignmentMeta:
    alignment_method: str
    slope: float
    intercept: float
    display_rate_est_hz: float
    matched_edges: Optional[int] = None
    edge_rmse_intervals: Optional[float] = None
    bv_duration_s: Optional[float] = None
    harp_duration_s: Optional[float] = None
    bv_t0: Optional[float] = None
    phase: Optional[int] = None
    edge_start: Optional[int] = None
    used_piecewise_warp: bool = False


# =============================================================================
# Public entry point
# =============================================================================

def correct_event_log(
    bonsai_event_log_csv: PathLike,
    photodiode_pkl: PathLike,
    savepath: Optional[PathLike] = None,
    qc_dir: Optional[PathLike] = None,
    modulo_frames: int = 30,
    min_edge_separation_s: float = 0.005,
    insert_missing_first_stim_rows: bool = True,
    use_piecewise_warp: bool = True,
) -> Tuple[pd.DataFrame, AlignmentMeta]:
    """
    Align Bonsai/BonVision event log (BonVision time) to HARP time using photodiode.

    Output columns:
      - timestamp_bv_rel
      - corrected_timestamps  (HARP-relative seconds)
      - alignment_method
      - photodiode_event / photodiode_state

    Pathways:
      1) BV photodiode present:
         - Extract BV photodiode RISE times from photodiode_state at Frame rows (mode='auto')
         - Extract HARP photodiode RISE times from analog trace
         - Fit affine and optionally apply piecewise warp anchored at rises
         (UNCHANGED)

      2) BV photodiode absent:
         - Extract HARP photodiode EDGE times (rises + falls) from analog trace
         - Choose BV flip train as Frame % modulo_frames == phase
         - Find best affine match allowing small edge_start shift, but constrain edge_start
           to be consistent with intercept bound (prevents late-match pathologies).
         (CHANGED)
    """
    bonsai_event_log_csv = Path(bonsai_event_log_csv)
    photodiode_pkl = Path(photodiode_pkl)
    qc_dir = Path(qc_dir) if qc_dir is not None else None
    if qc_dir is not None:
        qc_dir.mkdir(parents=True, exist_ok=True)

    bv = _load_bonsai_event_log(bonsai_event_log_csv)
    harp_pd = _load_photodiode(photodiode_pkl)

    if insert_missing_first_stim_rows:
        bv = _insert_first_stim_rows(bv)

    # Dense PD columns (ffill) for BV—may be all-NaN if BV has no photodiode logging
    bv = _add_bv_photodiode_columns_dense(bv)

    # BV-relative timestamps
    t_bv0 = _choose_bv_t0(bv)
    bv = bv.copy()
    bv["timestamp_bv_rel"] = bv["Timestamp"].to_numpy(dtype=float) - float(t_bv0)

    # HARP photodiode edges in HARP-relative seconds
    # NOTE: modulo_frames is interpreted as frames per TRANSITION for display rate estimate
    display_rate_hz, harp_edges_s, harp_rise_s, edge_period_s, thr = _estimate_display_rate_and_edges_from_harp(
        harp_pd,
        modulo_frames=modulo_frames,
        min_edge_separation_s=min_edge_separation_s,
    )

    # Duration guardrail
    bv_duration = _estimate_bv_duration_seconds(bv, t_bv0=t_bv0)
    harp_duration = float(harp_pd.index.values[-1] - harp_pd.index.values[0])
    if harp_duration < 0.5 * bv_duration:
        raise RuntimeError(
            "Photodiode duration mismatch: "
            f"BV~{bv_duration:.1f}s, HARP~{harp_duration:.1f}s. "
            "Likely wrong/truncated photodiode.pkl."
        )

    # -------------------------------------------------------------------------
    # Pathway 1 (UNCHANGED): BV photodiode exists -> match RISES (BV vs HARP rises)
    # -------------------------------------------------------------------------
    pd_parse = _bv_photodiode_rises_from_state(
        bv,
        mode="auto",
        harp_rise_s=harp_rise_s,
        max_shift=10,
    )

    if pd_parse is not None and len(pd_parse) >= 3 and len(harp_rise_s) >= 3:
        bv_rise_rel = pd_parse  # BV-relative seconds
        shift, rmse_int = _best_start_offset_by_intervals(bv_rise_rel, harp_rise_s, max_shift=10)

        bv_r = bv_rise_rel.copy()
        hp_r = harp_rise_s.copy()
        if shift > 0:
            bv_r = bv_r[shift:]
        elif shift < 0:
            hp_r = hp_r[-shift:]

        slope, intercept, matched = _fit_affine(bv_r, hp_r)

        if use_piecewise_warp:
            corrected = _piecewise_warp(
                t_query=bv["timestamp_bv_rel"].to_numpy(dtype=float),
                t_src=bv_r[:matched],
                t_dst=hp_r[:matched],
            )
            method = "bv_photodiode_piecewise"
            used_pw = True
        else:
            corrected = slope * bv["timestamp_bv_rel"].to_numpy(dtype=float) + intercept
            method = "bv_photodiode_affine"
            used_pw = False

        meta = AlignmentMeta(
            alignment_method=method,
            slope=float(slope),
            intercept=float(intercept),
            display_rate_est_hz=float(display_rate_hz),
            matched_edges=int(matched),
            edge_rmse_intervals=float(rmse_int),
            bv_duration_s=float(bv_duration),
            harp_duration_s=float(harp_duration),
            bv_t0=float(t_bv0),
            used_piecewise_warp=used_pw,
        )

    else:
        # ---------------------------------------------------------------------
        # Pathway 2 (CHANGED): BV photodiode absent -> match BV modulo-class flips
        #                       to HARP EDGES (rises+falls)
        # ---------------------------------------------------------------------
        slope, intercept, phase, edge_start, matched, rmse = _fit_frame_modclass_to_harp_edges_anchored_edges(
            bv=bv,
            harp_edge_s=harp_edges_s,   # <-- rises+falls, sorted
            modulo_frames=30,           # frames per transition
            b_abs_max=3.0,

            # Dropout detection defaults are good; you can tune:
            dropout_gap_factor=1.5,
            dropout_min_gap_s=1.5,
            dropout_min_edges_per_segment=80,

            # 2025 priority: use a wide prior to break periodic ties
            use_b_prior=True,
            b_prior=0.30,
            b_prior_sigma=0.25,
            prior_strength_b=1.0,
        )
        corrected = slope * bv["timestamp_bv_rel"].to_numpy(dtype=float) + intercept

        meta = AlignmentMeta(
            alignment_method="frame_modclass_affine_anchored_edges",
            slope=float(slope),
            intercept=float(intercept),
            display_rate_est_hz=float(display_rate_hz),
            matched_edges=int(matched),
            edge_rmse_intervals=float(rmse),
            bv_duration_s=float(bv_duration),
            harp_duration_s=float(harp_duration),
            bv_t0=float(t_bv0),
            phase=int(phase),
            edge_start=int(edge_start),
            used_piecewise_warp=False,
        )

    out = bv.copy()
    out["corrected_timestamps"] = corrected
    out["alignment_method"] = meta.alignment_method

    if qc_dir is not None:
        (qc_dir / "alignment_meta.json").write_text(pd.Series(meta.__dict__).to_json(indent=2))

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(savepath, index=False)

    return out, meta


# =============================================================================
# Internals: IO and preprocessing
# =============================================================================

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed")]


def _load_bonsai_event_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_unnamed(df)
    for c in ("Frame", "Timestamp", "Value"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing required column '{c}'. Columns={list(df.columns)}")
    return df


def _load_photodiode(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if isinstance(df, pd.Series):
        df = df.to_frame("AnalogInput0")
    if "AnalogInput0" not in df.columns:
        df = df.rename(columns={df.columns[0]: "AnalogInput0"})
    df = df.copy()
    df.index = df.index.astype(float)
    df.index.name = "Time"
    return df


def _add_bv_photodiode_columns_dense(stim_df: pd.DataFrame) -> pd.DataFrame:
    """
    photodiode_event is 0/1 at Photodiode-* rows, NaN elsewhere.
    photodiode_state is forward-filled across all rows.
    """
    df = stim_df.copy()
    val = df["Value"].astype(str)
    pd_event = np.full(len(df), np.nan, dtype=float)
    pd_event[val.str.fullmatch(r"Photodiode-1", case=False)] = 1.0
    pd_event[val.str.fullmatch(r"Photodiode-0", case=False)] = 0.0
    df["photodiode_event"] = pd_event
    df["photodiode_state"] = pd.Series(pd_event).ffill().to_numpy()
    return df


def _choose_bv_t0(bv: pd.DataFrame) -> float:
    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if frame_mask.any():
        return float(bv.loc[frame_mask, "Timestamp"].iloc[0])
    return float(bv["Timestamp"].min())


def _insert_first_stim_rows(event_log: pd.DataFrame) -> pd.DataFrame:
    df = event_log.copy()
    tif_mask = df["Value"].astype(str).str.contains(r"\.tif{1,2}f?$", case=False, na=False)
    if not tif_mask.any():
        return df
    tif_value = df.loc[tif_mask, "Value"].iloc[0]
    prepend = pd.DataFrame(
        [{"Frame": -1, "Timestamp": 0.0, "Value": "Frame"},
         {"Frame": -1, "Timestamp": 0.0, "Value": tif_value}]
    )
    for c in df.columns:
        if c not in prepend.columns:
            prepend[c] = np.nan
    prepend = prepend[df.columns]
    return pd.concat([prepend, df], ignore_index=True)


def _estimate_bv_duration_seconds(bv: pd.DataFrame, *, t_bv0: float) -> float:
    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if frame_mask.any():
        return float(bv.loc[frame_mask, "Timestamp"].max() - t_bv0)
    return float(bv["Timestamp"].max() - t_bv0)


# =============================================================================
# Internals: HARP edge detection (CHANGED helper)
# =============================================================================

def _get_signal_edges(
    signal: np.ndarray,
    time: np.ndarray,
    *,
    min_edge_separation_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    sig = np.asarray(signal)
    t = np.asarray(time)

    y_s = medfilt(sig, kernel_size=9) if len(sig) >= 9 else sig.copy()
    thr = (np.percentile(y_s, 95) + np.percentile(y_s, 5)) / 2.0
    binary = (y_s > thr).astype(np.int8)

    db = np.diff(binary, prepend=binary[0])
    rise_idx = np.where(db == 1)[0]
    fall_idx = np.where(db == -1)[0]

    t_rise = t[rise_idx]
    t_fall = t[fall_idx]

    def _debounce(tt: np.ndarray) -> np.ndarray:
        if len(tt) <= 1:
            return tt
        keep = [0]
        for i in range(1, len(tt)):
            if (tt[i] - tt[keep[-1]]) > min_edge_separation_s:
                keep.append(i)
        return tt[keep]

    t_rise = _debounce(t_rise)
    t_fall = _debounce(t_fall)
    return rise_idx, fall_idx, t_rise, t_fall, float(thr)


def _estimate_display_rate_and_edges_from_harp(
    photodiode_df: pd.DataFrame,
    *,
    modulo_frames: int,
    min_edge_separation_s: float,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """
    CHANGED: returns both-edge train for robust fallback.

    Returns:
      display_rate_hz  (estimated as modulo_frames / edge_period)
      t_edges          (rises + falls, sorted)    [HARP-relative seconds]
      t_rise           (rises only)              [HARP-relative seconds]  (used by BV-PD pathway)
      edge_period      (median diff of t_edges)
      thr
    """
    t = photodiode_df.index.values - photodiode_df.index.values[0]
    y = photodiode_df["AnalogInput0"].values

    _, _, t_rise, t_fall, thr = _get_signal_edges(y, t, min_edge_separation_s=min_edge_separation_s)

    if len(t_rise) < 2 and len(t_fall) < 2:
        raise ValueError("Not enough HARP photodiode edges.")

    t_edges = np.sort(np.concatenate([t_rise, t_fall]))
    if len(t_edges) < 3:
        raise ValueError("Not enough HARP photodiode edges.")

    edge_period = float(np.median(np.diff(t_edges)))
    display_rate = float(modulo_frames / edge_period)

    return display_rate, t_edges, t_rise, edge_period, float(thr)


# =============================================================================
# Internals: BV photodiode rises (UNCHANGED logic)
# =============================================================================

def _bv_photodiode_rises_from_state(
    bv: pd.DataFrame,
    *,
    mode: str = "auto",   # "auto" | "current" | "midpoint" | "previous"
    harp_rise_s: Optional[np.ndarray] = None,
    max_shift: int = 10,
) -> Optional[np.ndarray]:
    """
    Extract BV photodiode rise times (BV-relative seconds) from photodiode_state,
    using only Frame rows.

    mode:
      - "current":  rise time = t[i]
      - "midpoint": rise time = 0.5*(t[i-1] + t[i])
      - "previous": rise time = t[i-1]
      - "auto":     pick best mode by residual RMSE after affine fitting to harp_rise_s
    """
    if "photodiode_state" not in bv.columns:
        return None

    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if not frame_mask.any():
        return None

    fr = bv.loc[frame_mask, ["timestamp_bv_rel", "photodiode_state"]].copy()
    state = fr["photodiode_state"].ffill().fillna(0.0).to_numpy(dtype=float)
    t = fr["timestamp_bv_rel"].to_numpy(dtype=float)

    ds = np.diff(state, prepend=state[0])
    idx = np.where(ds == 1)[0]
    if len(idx) == 0:
        return None

    def rises_with(m: str) -> np.ndarray:
        if m == "current":
            return t[idx]
        if m == "midpoint":
            idx2 = idx[idx > 0]
            return 0.5 * (t[idx2 - 1] + t[idx2])
        if m == "previous":
            idx2 = idx[idx > 0]
            return t[idx2 - 1]
        raise ValueError(f"Unknown mode: {m}")

    if mode != "auto":
        r = rises_with(mode)
        return np.asarray(r, dtype=float) if len(r) else None

    if harp_rise_s is None or len(harp_rise_s) < 3:
        r = rises_with("previous")
        return np.asarray(r, dtype=float) if len(r) else None

    modes = ["previous", "midpoint", "current"]
    best = None

    for m in modes:
        bv_r = rises_with(m)
        if len(bv_r) < 3:
            continue

        shift, _ = _best_start_offset_by_intervals(bv_r, harp_rise_s, max_shift=max_shift)
        bv_r2, hp_r2 = _apply_shift(bv_r, harp_rise_s, shift)
        if len(bv_r2) < 3:
            continue

        a, b, _ = _fit_affine(bv_r2, hp_r2)
        res = (a * bv_r2 + b) - hp_r2
        rmse = float(np.sqrt(np.mean(res ** 2)))

        cand = (rmse, m, bv_r)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        r = rises_with("previous")
        return np.asarray(r, dtype=float) if len(r) else None

    return np.asarray(best[2], dtype=float)


# =============================================================================
# Internals: affine + shift + piecewise warp
# =============================================================================

def _best_start_offset_by_intervals(
    bv_times: np.ndarray,
    harp_times: np.ndarray,
    *,
    max_shift: int = 5
) -> Tuple[int, float]:
    bv_times = np.asarray(bv_times, dtype=float)
    harp_times = np.asarray(harp_times, dtype=float)

    bv_dt = np.diff(bv_times)
    hp_dt = np.diff(harp_times)

    n = min(len(bv_dt), len(hp_dt), 20)
    if n < 5:
        return 0, float("inf")

    bv_dt = bv_dt[:n]
    hp_dt = hp_dt[:n]

    best_shift = 0
    best_rmse = float("inf")

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = bv_dt[shift:]
            b = hp_dt[: len(a)]
        else:
            b = hp_dt[-shift:]
            a = bv_dt[: len(b)]

        m = min(len(a), len(b))
        if m < 5:
            continue

        rmse = float(np.sqrt(np.mean((a[:m] - b[:m]) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift

    return best_shift, best_rmse


def _apply_shift(bv: np.ndarray, hp: np.ndarray, shift: int) -> Tuple[np.ndarray, np.ndarray]:
    bv = np.asarray(bv, dtype=float)
    hp = np.asarray(hp, dtype=float)
    if shift > 0:
        bv = bv[shift:]
    elif shift < 0:
        hp = hp[-shift:]
    n = min(len(bv), len(hp))
    return bv[:n], hp[:n]


def _fit_affine(bv_edge_times: np.ndarray, harp_edge_times: np.ndarray) -> Tuple[float, float, int]:
    bv = np.asarray(bv_edge_times, dtype=float)
    hp = np.asarray(harp_edge_times, dtype=float)
    n = min(len(bv), len(hp))
    if n < 3:
        raise ValueError("Need at least 3 edges to fit affine.")
    bv = bv[:n]
    hp = hp[:n]
    slope, intercept = np.polyfit(bv, hp, 1)
    return float(slope), float(intercept), int(n)


def _piecewise_warp(t_query: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    t_query = np.asarray(t_query, dtype=float)
    t_src = np.asarray(t_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)

    if len(t_src) < 2:
        raise ValueError("Need at least 2 anchors for piecewise warp.")

    order = np.argsort(t_src)
    t_src = t_src[order]
    t_dst = t_dst[order]

    out = np.interp(t_query, t_src, t_dst)

    left = t_query < t_src[0]
    right = t_query > t_src[-1]

    m0 = (t_dst[1] - t_dst[0]) / (t_src[1] - t_src[0])
    b0 = t_dst[0] - m0 * t_src[0]
    out[left] = m0 * t_query[left] + b0

    m1 = (t_dst[-1] - t_dst[-2]) / (t_src[-1] - t_src[-2])
    b1 = t_dst[-1] - m1 * t_src[-1]
    out[right] = m1 * t_query[right] + b1

    return out


# =============================================================================
# Internals: CHANGED fallback — modulo-class vs HARP EDGES (rises+falls)
# =============================================================================

def _split_edge_train_into_segments(
    harp_edge_s: np.ndarray,
    *,
    gap_factor: float = 1.5,
    min_gap_s: float = 1.5,
    min_edges_per_segment: int = 50,
    always_keep_first_segment: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Split a HARP edge train into contiguous segments separated by dropout-sized gaps.

    Robust behavior:
    - If always_keep_first_segment=True, the first segment is kept even if short.
      This prevents losing the true onset time when an early dropout occurs.

    Returns:
      segments: list of edge arrays
      meta: QC info
    """
    edges = np.asarray(harp_edge_s, dtype=float)
    edges = np.sort(edges)

    if edges.size < 5:
        return [edges], dict(
            n_edges=int(edges.size),
            median_dt=np.nan,
            n_gaps=0,
            max_gap_s=0.0,
            gap_indices=[],
            segment_lengths=[int(edges.size)],
            segment_starts=[float(edges[0]) if edges.size else np.nan],
        )

    dt = np.diff(edges)
    median_dt = float(np.median(dt))
    gap_mask = (dt > (gap_factor * median_dt)) & (dt > min_gap_s)
    gap_idx = np.where(gap_mask)[0]

    if gap_idx.size == 0:
        segs = [edges]
    else:
        segs = np.split(edges, (gap_idx + 1).tolist())

    # Filter small segments, but optionally keep the first no matter what
    kept = []
    for i, s in enumerate(segs):
        if s.size >= int(min_edges_per_segment):
            kept.append(s)
        elif always_keep_first_segment and i == 0:
            kept.append(s)

    meta = dict(
        n_edges=int(edges.size),
        median_dt=median_dt,
        n_gaps=int(gap_idx.size),
        max_gap_s=float(dt[gap_idx].max()) if gap_idx.size else 0.0,
        gap_indices=gap_idx.tolist(),
        segment_lengths=[int(s.size) for s in kept],
        segment_starts=[float(s[0]) for s in kept] if kept else [],
    )

    # If everything got filtered (should be rare), keep the largest segment.
    if not kept:
        segs.sort(key=lambda s: s.size, reverse=True)
        kept = [segs[0]]
        meta["segment_lengths"] = [int(kept[0].size)]
        meta["segment_starts"] = [float(kept[0][0])]

    return kept, meta

def _fit_frame_modclass_to_harp_edges_anchored_edges(
    *,
    bv: "pd.DataFrame",
    harp_edge_s: np.ndarray,
    modulo_frames: int,
    slope_bounds: Tuple[float, float] = (0.98, 1.02),
    min_pairs: int = 100,
    max_edge_start: int = 20000,
    max_pairs_per_fit: int = 5000,
    rmse_tol: float = 0.2,
    n_frac: float = 0.95,
    b_abs_max: float = 3.0,
    # --- Dropout handling ---
    dropout_gap_factor: float = 1.5,
    dropout_min_gap_s: float = 1.5,
    dropout_min_edges_per_segment: int = 80,
    # --- Robust selection / priors (recommended for 2025) ---
    use_b_prior: bool = True,
    b_prior: float = 0.30,         # default for 2025-ish sessions; override upstream if you have better
    b_prior_sigma: float = 0.25,   # wide, because 2025 appears multi-modal
    prior_strength_b: float = 1.0,
    # Optionally (usually weak)
    use_a_prior: bool = False,
    a_prior: float = 1.0,
    a_prior_sigma: float = 0.01,
    prior_strength_a: float = 0.1,
) -> Tuple[float, float, int, int, int, float]:
    """
    Dropout-aware fallback:

    - Uses HARP EDGES (rises+falls).
    - Treats modulo_frames as "frames per TRANSITION".
    - Detects edge-train dropouts and fits on contiguous edge segments.
    - Does NOT force edge_start=0; instead searches edge_start within constraints
      and selects using a robust score (RMSE + optional priors).

    Returns (a, b, phase, edge_start, n, rmse).

    Notes
    -----
    1) If a dropout exists, fitting across the whole edge train can "cycle slip" by ~N seconds.
       We prevent that by splitting into segments and picking the best segment.
    2) If your system is periodic/ambiguous, priors help choose the correct branch.
       For 2025, I recommend a WIDE prior (sigma ~0.2–0.3) unless you have day-by-day continuity.
    """

    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if not frame_mask.any():
        raise RuntimeError("No 'Frame' rows found in BV event log; cannot run fallback.")

    fr = bv.loc[frame_mask, ["Frame", "timestamp_bv_rel"]].copy()
    frames = fr["Frame"].to_numpy(dtype=int)
    t_frame = fr["timestamp_bv_rel"].to_numpy(dtype=float)

    # Segment the edge train to avoid dropout-induced slips
    harp_edge_s = np.asarray(harp_edge_s, dtype=float)
    harp_edge_s = np.sort(harp_edge_s)
    segments, seg_meta = _split_edge_train_into_segments(
        harp_edge_s,
        gap_factor=dropout_gap_factor,
        min_gap_s=dropout_min_gap_s,
        min_edges_per_segment=dropout_min_edges_per_segment,
    )

    # Candidate generation on a given segment
    def _candidates_for_segment(seg_edges: np.ndarray) -> List[Dict[str, Any]]:
        seg_edges = np.asarray(seg_edges, dtype=float)
        seg_edges = np.sort(seg_edges)

        # Constrain edge_start based on |b| bound, but allow segment start later if b_abs_max allows.
        # Since b ~ time of first matched edge, edge_start values beyond this are meaningless for small b_abs_max.
        edge_start_limit = int(np.searchsorted(seg_edges, b_abs_max + 0.25))
        max_es = int(min(max_edge_start, max(0, seg_edges.size - min_pairs), edge_start_limit))
        if max_es < 0:
            return []

        # Adapt min_pairs for shorter segments
        min_pairs_eff = int(min(min_pairs, max(20, seg_edges.size // 5)))

        cands: List[Dict[str, Any]] = []
        for phase in range(int(modulo_frames)):
            sel = (frames % modulo_frames) == phase
            if int(np.sum(sel)) < min_pairs_eff:
                continue

            bv_flip = np.sort(t_frame[sel])[:max_pairs_per_fit]

            for edge_start in range(max_es + 1):
                n = min(bv_flip.size, seg_edges.size - edge_start, max_pairs_per_fit)
                if n < min_pairs_eff:
                    continue

                t_bv = bv_flip[:n]
                t_hp = seg_edges[edge_start:edge_start + n]

                a, b, _ = _fit_affine(t_bv, t_hp)
                if not (slope_bounds[0] <= a <= slope_bounds[1]):
                    continue
                if abs(b) > b_abs_max:
                    continue

                err = (a * t_bv + b) - t_hp
                rmse = float(np.sqrt(np.mean(err ** 2)))

                # Score: prioritize small RMSE; optionally add priors to break periodic ties
                score = rmse * rmse
                if use_b_prior:
                    sigb = max(float(b_prior_sigma), 1e-6)
                    zb = (b - float(b_prior)) / sigb
                    score += float(prior_strength_b) * (zb * zb)
                if use_a_prior:
                    siga = max(float(a_prior_sigma), 1e-6)
                    za = (a - float(a_prior)) / siga
                    score += float(prior_strength_a) * (za * za)

                cands.append(
                    dict(
                        phase=int(phase),
                        edge_start=int(edge_start),
                        a=float(a),
                        b=float(b),
                        rmse=float(rmse),
                        n=int(n),
                        score=float(score),
                        seg_start=float(seg_edges[0]),
                        seg_len=int(seg_edges.size),
                    )
                )
        return cands

    # Evaluate segments. Strategy:
    # - prefer segments that start early and are long, BUT only if they yield valid candidates.
    # We'll compute candidates per segment and then select globally among "good" candidates.
    all_candidates: List[Dict[str, Any]] = []
    for seg in segments:
        all_candidates.extend(_candidates_for_segment(seg))

    if not all_candidates:
        raise RuntimeError(
            "No valid candidates in frame-modclass fallback (edges) after dropout segmentation.\n"
            f"Edge QC: n_edges={seg_meta.get('n_edges')}, median_dt={seg_meta.get('median_dt'):.3f}, "
            f"n_gaps={seg_meta.get('n_gaps')}, max_gap_s={seg_meta.get('max_gap_s'):.3f}\n"
            f"Segments kept: starts={seg_meta.get('segment_starts')}, lengths={seg_meta.get('segment_lengths')}\n"
            "Likely causes: wrong modulo_frames, poor BV flip definition (Frame%modulo), "
            "b_abs_max too small, or photodiode.pkl mismatched."
        )

    # Filter to "good" set using your existing rmse_tol and n_frac logic,
    # but apply those AFTER gathering across segments.
    best_rmse = min(c["rmse"] for c in all_candidates)
    max_n = max(c["n"] for c in all_candidates)

    good = [
        c for c in all_candidates
        if c["rmse"] <= best_rmse * (1.0 + rmse_tol)
        and c["n"] >= int(max_n * n_frac)
        and abs(c["b"]) <= b_abs_max
    ]

    if not good:
        # Keep the best few for diagnostics
        top = sorted(all_candidates, key=lambda c: c["rmse"])[:8]
        msg = (
            "No valid candidates after applying rmse/n filters.\n"
            f"Constraints: slope_bounds={slope_bounds}, b_abs_max={b_abs_max}, "
            f"rmse_tol={rmse_tol}, n_frac={n_frac}, min_pairs={min_pairs}\n"
            f"Edge QC: n_edges={seg_meta.get('n_edges')}, median_dt={seg_meta.get('median_dt'):.3f}, "
            f"n_gaps={seg_meta.get('n_gaps')}, max_gap_s={seg_meta.get('max_gap_s'):.3f}\n"
            "Top candidates by RMSE:\n"
            + "\n".join(
                f"  rmse={c['rmse']:.4f}s score={c['score']:.3f} a={c['a']:.6f} b={c['b']:.3f}s "
                f"phase={c['phase']} edge_start={c['edge_start']} n={c['n']} "
                f"(seg_start={c['seg_start']:.3f}, seg_len={c['seg_len']})"
                for c in top
            )
        )
        raise RuntimeError(msg)

    # Final selection:
    # - Primary: score (RMSE + priors)
    # - Secondary: edge_start (prefer earlier pairing *within the chosen segment*)
    # - Tertiary: segment start time (prefer earlier segments)
    # - Quaternary: rmse
    good.sort(key=lambda c: (c["seg_start"], c["score"], c["edge_start"], c["rmse"]))
    best = good[0]
    return best["a"], best["b"], best["phase"], best["edge_start"], best["n"], best["rmse"]


# =============================================================================
# Optional: tighten your HARP edge extraction by rejecting obvious non-1Hz segments
# (Not required, but pairs nicely with dropout handling.)
# =============================================================================
def _edge_train_qc_summary(harp_edge_s: np.ndarray) -> Dict[str, Any]:
    edges = np.asarray(harp_edge_s, float)
    edges = np.sort(edges)
    if edges.size < 5:
        return dict(n_edges=int(edges.size))
    dt = np.diff(edges)
    return dict(
        n_edges=int(edges.size),
        median_dt=float(np.median(dt)),
        p95_dt=float(np.percentile(dt, 95)),
        max_dt=float(np.max(dt)),
    )


# def process_harp_sessions(harp_root_dir, save=True, overwrite=False):
#     """
#     Process all subdirectories in `harp_root_dir` that contain HARP binary data.

#     Parameters:
#     -----------
#     harp_root_dir : str or Path
#         Path to the parent directory containing session folders.
#     save : bool
#         If True, saves .pkl files for encoder, photodiode, licks, and rewards.
#     overwrite : bool
#         If True, existing extracted files will be overwritten.
#     """
#     harp_root_dir = Path(harp_root_dir)
#     session_dirs = [d for d in harp_root_dir.iterdir() if d.is_dir()]

#     for session in session_dirs:
#         try:
#             print(f"Processing {session}...")
#             reader = HarpReader(session)
#             extracted_dir = session / "extracted_files"
#             if extracted_dir.exists() and not overwrite:
#                 print(f"→ Skipping {session.name}: already processed.")
#                 continue

#             if save:
#                 extracted_dir.mkdir(exist_ok=True)
#                 reader.get_encoder.to_pickle(extracted_dir / 'encoder.pkl')
#                 reader.get_photodiode.to_pickle(extracted_dir / 'photodiode.pkl')
#                 reader.get_licks.to_pickle(extracted_dir / 'licks.pkl')
#                 reader.get_rewards.to_pickle(extracted_dir / 'rewards.pkl')
#                 print(f"→ Saved data to {extracted_dir}")
#             else:
#                 print(reader.get_encoder.head())
#                 print(reader.get_photodiode.head())
#                 print(reader.get_licks.head())
#                 print(reader.get_rewards.head())
#         except Exception as e:
#             print(f"❌ Error processing {session.name}: {e}")

