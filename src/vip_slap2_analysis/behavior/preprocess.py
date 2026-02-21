import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from read_harp import HarpReader
from dataclasses import dataclass
from scipy.signal import medfilt
from typing import Optional, Tuple, Dict, Any, List,Union

def process_harp_sessions(harp_root_dir, save=True, overwrite=False):
    """
    Process all subdirectories in `harp_root_dir` that contain HARP binary data.

    Parameters:
    -----------
    harp_root_dir : str or Path
        Path to the parent directory containing session folders.
    save : bool
        If True, saves .pkl files for encoder, photodiode, licks, and rewards.
    overwrite : bool
        If True, existing extracted files will be overwritten.
    """
    harp_root_dir = Path(harp_root_dir)
    session_dirs = [d for d in harp_root_dir.iterdir() if d.is_dir()]

    for session in session_dirs:
        try:
            print(f"Processing {session}...")
            reader = HarpReader(session)
            extracted_dir = session / "extracted_files"
            if extracted_dir.exists() and not overwrite:
                print(f"→ Skipping {session.name}: already processed.")
                continue

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
            print(f"❌ Error processing {session.name}: {e}")

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

def get_signal_edges(signal,time,est_rate = 30):
    # Light smoothing against noise; adjust kernel if your PD is already clean.
    y_s = medfilt(signal, kernel_size=5) if len(signal)>=5 else signal.copy()

    # Auto-threshold at mid of bimodal distribution
    thr = (np.percentile(y_s, 95) + np.percentile(y_s, 5)) / 2.0
    binary = (y_s > thr).astype(int)

    # Rising edges: transitions 0->1
    db = np.diff(binary, prepend=binary[0])
    rise_idx = np.where(db==1)[0]
    t_rise = time[rise_idx]

    # Falling edges: transitions 1->0
    fall_idx = np.where(-db==1)[0]
    t_fall = time[fall_idx]

    # Optional: collapse spurious multiple edges within a frame (debounce)
    # Merge edges closer than, say, 5 ms (typical if photodiode bounces)
    if len(t_rise)>1:
        keep = [0]
        for i in range(1, len(t_rise)):
            if (t_rise[i] - t_rise[keep[-1]]) > 0.005:
                keep.append(i)
        t_rise = t_rise[keep]
        
    if len(t_fall)>1:
        keep = [0]
        for i in range(1, len(t_fall)):
            if (t_fall[i] - t_fall[keep[-1]]) > 0.005:
                keep.append(i)
        t_fall = t_fall[keep]
        
    estimated_rate = est_rate #Hz

    cycle_rates = []

    for start,stop in zip(t_rise,t_fall):
        dt = np.diff([start,stop])[0]
        n_frames = estimated_rate/dt
        cycle_rates.append(n_frames)
    avg_rate = np.median(cycle_rates)    

    print(f'Median frame rate of LCD screen: {avg_rate:.6} Hz')

    return (rise_idx,fall_idx,t_rise,t_fall,avg_rate)

# def get_time_offset(photodiode_df,modulo=60):

#     #Calculate the offset time between the first photodiode flip and the first image presentation
#     pd_time = photodiode_df.index - photodiode_df.index[0]
#     pd_signal = photodiode_df['AnalogInput0'].values

#     signal_metrics = get_signal_edges(pd_signal,pd_time,est_rate=30)

#     flip_time = signal_metrics[2][0]
#     avg_rate = signal_metrics[-1]

#     offset_time =  flip_time - (1/avg_rate)*modulo
#     print(f'Time offset of image presentation from photodiode signal: {offset_time:.4} seconds')

#     return offset_time

# def correct_event_log(stimulus_df,photodiode_df,savepath = None):

#     #Update stimulus_df to contain information about the first stimulus shown (for whatever reason these aren't logged in the trial log)
#     tif_row = stimulus_df[stimulus_df['Value'].str.contains('.tif', case=False, na=False)].iloc[0]
#     tif_value = tif_row['Value']
#     new_row = pd.DataFrame([{'Frame': -1, 'Timestamp': 0.0, 'Value': 'Frame'}])
#     new_row_1 = pd.DataFrame([{'Frame': -1, 'Timestamp': 0.0, 'Value': tif_value}])
#     stimulus_df.index = stimulus_df.index + 1
#     stimulus_df = pd.concat([new_row, new_row_1, stimulus_df]).reset_index(drop=True)  

#     offset_time = get_time_offset(photodiode_df)  

#     stimulus_df['corrected_timestamp'] = stimulus_df['Timestamp'] + offset_time

#     if savepath:
#         stimulus_df.to_csv(savepath)
#         print('Saved stimulus table to savepath')

    # return stimulus_df
    
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


def correct_event_log(
    bonsai_event_log_csv: PathLike,
    photodiode_pkl: PathLike,
    savepath: Optional[PathLike] = None,
    modulo_frames: int = 60,
    min_edge_separation_s: float = 0.005,
    insert_missing_first_stim_rows: bool = True,
    # robust fallback selection controls
    slope_bounds: Tuple[float, float] = (0.90, 1.10),
    min_pairs: int = 200,
    max_edge_start: int = 20000,
    max_pairs_per_fit: int = 5000,
    rmse_tol: float = 0.25,
    n_frac: float = 0.90,
    b_abs_max: float = 5.0,
) -> Tuple[pd.DataFrame, AlignmentMeta]:
    """
    Align a BonVision/Bonsai event log (BonVision time) to HARP time using a photodiode trace.

    Produces a modified event log containing:
      - corrected_timestamps: BonVision timestamps mapped into HARP timebase (seconds, HARP-relative)
      - alignment_method: which strategy was used
      - photodiode_event / photodiode_state: annotated for Photodiode-0/1 rows (if present),
        with photodiode_state forward-filled across ALL rows (easy plotting).

    Strategy:
      1) If BonVision photodiode events are present (Value in {"Photodiode-0","Photodiode-1"}):
         - create photodiode_event + photodiode_state (ffill) for all rows
         - collapse repeated states to obtain true edges
         - match rising-edge trains to HARP rising edges (small integer shift)
         - fit affine map on BV-relative times: t_harp_rel = slope * t_bv_rel + intercept
      2) Else (no BonVision photodiode):
         - estimate display rate from HARP PD
         - robustly align using Frame % modulo class + edge_start search
         - among “good” fits, choose smallest |intercept| (anchors to expected small correction)

    Notes on timebases:
      - HARP photodiode times are converted to relative seconds: idx - idx[0]
      - BonVision timestamps are converted to BV-relative: Timestamp - t_bv0
      - corrected_timestamps are HARP-relative seconds (add photodiode_df.index[0] externally if you need absolute HARP)

    Returns
    -------
    (out_df, meta)
    """
    bonsai_event_log_csv = Path(bonsai_event_log_csv)
    photodiode_pkl = Path(photodiode_pkl)

    bv = _load_bonsai_event_log(bonsai_event_log_csv)
    harp_pd = _load_photodiode(photodiode_pkl)

    if insert_missing_first_stim_rows:
        bv = _insert_first_stim_rows(bv)

    # Add photodiode_event + photodiode_state (dense ffill) for ALL rows
    bv = _add_bv_photodiode_columns(bv)

    # BV reference time (for BV-relative timestamps)
    t_bv0 = _choose_bv_t0(bv)
    bv = bv.copy()
    bv["timestamp_bv_rel"] = bv["Timestamp"].to_numpy(dtype=float) - float(t_bv0)

    # HARP PD in seconds relative to recording start
    display_rate_hz, harp_rise_s, rise_period_s = _estimate_display_rate_from_harp(
        harp_pd, modulo_frames=modulo_frames, min_edge_separation_s=min_edge_separation_s
    )

    # Duration guardrail (prevents nonsense if wrong photodiode file is loaded)
    bv_duration = _estimate_bv_duration_seconds(bv, t_bv0=t_bv0)
    harp_duration = float(harp_pd.index.values[-1] - harp_pd.index.values[0])
    if harp_duration < 0.5 * bv_duration:
        raise RuntimeError(
            "Photodiode duration mismatch:\n"
            f"  BV duration ~ {bv_duration:.1f} s\n"
            f"  HARP photodiode duration ~ {harp_duration:.1f} s\n"
            "Likely truncated/wrong photodiode.pkl for this session."
        )

    # Try BonVision photodiode alignment first
    pd_parse = _bv_photodiode_edge_times(bv)
    corrected_rel: np.ndarray

    if pd_parse is not None:
        _pd_rows, _edges, bv_rise_abs = pd_parse
        # convert BV rises to BV-relative
        bv_rise = np.asarray(bv_rise_abs, dtype=float) - float(t_bv0)

        if len(bv_rise) >= 3 and len(harp_rise_s) >= 3:
            shift, rmse_int = _best_start_offset_by_intervals(bv_rise, harp_rise_s, max_shift=5)

            bv_rise_adj = bv_rise.copy()
            harp_rise_adj = harp_rise_s.copy()

            if shift > 0:
                bv_rise_adj = bv_rise_adj[shift:]
            elif shift < 0:
                harp_rise_adj = harp_rise_adj[-shift:]

            slope, intercept, matched = _fit_affine(bv_rise_adj, harp_rise_adj)

            corrected_rel = slope * bv["timestamp_bv_rel"].to_numpy(dtype=float) + intercept

            meta = AlignmentMeta(
                alignment_method="bv_photodiode_affine",
                slope=float(slope),
                intercept=float(intercept),
                display_rate_est_hz=float(display_rate_hz),
                matched_edges=int(matched),
                edge_rmse_intervals=float(rmse_int),
                bv_duration_s=float(bv_duration),
                harp_duration_s=float(harp_duration),
                bv_t0=float(t_bv0),
            )
        else:
            # Photodiode present but not usable → fall back
            pd_parse = None

    # Fallback: robust frame%modulo alignment (no BV photodiode)
    if pd_parse is None:
        slope, intercept, phase, edge_start, matched, rmse = _fit_frame_modclass_to_harp_edges(
            bv=bv,
            harp_rise_s=harp_rise_s,
            modulo_frames=modulo_frames,
            slope_bounds=slope_bounds,
            min_pairs=min_pairs,
            max_edge_start=max_edge_start,
            max_pairs_per_fit=max_pairs_per_fit,
            rmse_tol=rmse_tol,
            n_frac=n_frac,
            b_abs_max=b_abs_max,
        )

        corrected_rel = slope * bv["timestamp_bv_rel"].to_numpy(dtype=float) + intercept

        meta = AlignmentMeta(
            alignment_method="frame_modclass_affine_anchored",
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
        )

    # Build output dataframe
    out = bv.copy()
    out["corrected_timestamps"] = corrected_rel  # HARP-relative seconds
    out["alignment_method"] = meta.alignment_method

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(savepath, index=False)

    return out, meta


# =============================================================================
# Internals
# =============================================================================

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed")]


def _load_bonsai_event_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_unnamed(df)
    required = ["Frame", "Timestamp", "Value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Columns={list(df.columns)}")
    return df


def _load_photodiode(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    if isinstance(df, pd.Series):
        df = df.to_frame("AnalogInput0")

    if "AnalogInput0" not in df.columns:
        df = df.rename(columns={df.columns[0]: "AnalogInput0"})

    df = df.copy()
    df.index = df.index.astype(float)  # HARP timebase
    df.index.name = "Time"
    return df


def _add_bv_photodiode_columns(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Your requested behavior:
      - photodiode_event: sparse 0/1 where Value is Photodiode-0/1, NaN elsewhere
      - photodiode_state: forward-filled across ALL rows (easy plotting)
    """
    df = event_log.copy()
    val = df["Value"].astype(str)

    pd_event = np.full(len(df), np.nan, dtype=float)
    pd_event[val.str.fullmatch(r"Photodiode-1", case=False)] = 1.0
    pd_event[val.str.fullmatch(r"Photodiode-0", case=False)] = 0.0

    df["photodiode_event"] = pd_event
    df["photodiode_state"] = pd.Series(pd_event).ffill().to_numpy()

    return df


def _choose_bv_t0(bv: pd.DataFrame) -> float:
    """
    Choose BV reference time:
      - first Timestamp among Value=='Frame' if present, else min Timestamp.
    """
    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if frame_mask.any():
        return float(bv.loc[frame_mask, "Timestamp"].iloc[0])
    return float(bv["Timestamp"].min())


def _insert_first_stim_rows(event_log: pd.DataFrame) -> pd.DataFrame:
    """
    Prepends:
      Frame=-1, Timestamp=0, Value='Frame'
      Frame=-1, Timestamp=0, Value='<first .tif or .tiff found>'
    """
    df = event_log.copy()
    tif_mask = df["Value"].astype(str).str.contains(r"\.tif{1,2}f?$", case=False, na=False)
    if not tif_mask.any():
        return df

    tif_value = df.loc[tif_mask, "Value"].iloc[0]

    prepend = pd.DataFrame(
        [
            {"Frame": -1, "Timestamp": 0.0, "Value": "Frame"},
            {"Frame": -1, "Timestamp": 0.0, "Value": tif_value},
        ]
    )

    for c in df.columns:
        if c not in prepend.columns:
            prepend[c] = np.nan
    prepend = prepend[df.columns]

    return pd.concat([prepend, df], ignore_index=True)


def _get_signal_edges(
    signal: np.ndarray,
    time: np.ndarray,
    *,
    min_edge_separation_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Threshold + debounce edge detection for the photodiode analog trace.
    Uses a small median filter to suppress ripple.
    """
    sig = np.asarray(signal)
    t = np.asarray(time)

    y_s = medfilt(sig, kernel_size=5) if len(sig) >= 5 else sig.copy()
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


def _estimate_display_rate_from_harp(
    photodiode_df: pd.DataFrame,
    *,
    modulo_frames: int,
    min_edge_separation_s: float,
) -> Tuple[float, np.ndarray, float]:
    """
    Assumes photodiode toggles once per `modulo_frames` display frames.

    period = median(diff(rising_edges))
    display_rate_hz = modulo_frames / period
    """
    t = photodiode_df.index.values - photodiode_df.index.values[0]  # HARP-relative seconds
    y = photodiode_df["AnalogInput0"].values

    _, _, t_rise, _, _ = _get_signal_edges(y, t, min_edge_separation_s=min_edge_separation_s)
    if len(t_rise) < 3:
        raise ValueError("Not enough HARP photodiode rising edges to estimate display rate.")

    period = float(np.median(np.diff(t_rise)))
    display_rate = float(modulo_frames / period)
    return display_rate, t_rise, period


def _bv_photodiode_edge_times(event_log: pd.DataFrame):
    """
    Extract true photodiode edges from BonVision event log rows.

    We use the dense photodiode_state already computed, but edges are derived by
    selecting rows where photodiode_event is not NaN and collapsing duplicates.
    """
    if "photodiode_event" not in event_log.columns:
        return None

    mask = ~pd.isna(event_log["photodiode_event"])
    if not mask.any():
        return None

    pd_rows = event_log.loc[mask, ["Frame", "Timestamp", "Value", "photodiode_event"]].copy()
    pd_rows = pd_rows.sort_values(["Timestamp", "Frame"]).reset_index(drop=True)
    pd_rows["state"] = pd_rows["photodiode_event"].astype(int)

    state = pd_rows["state"].values
    change = np.r_[True, state[1:] != state[:-1]]
    edges = pd_rows.loc[change, ["Timestamp", "state"]].reset_index(drop=True)

    edge_states = edges["state"].values
    prev = np.r_[edge_states[0], edge_states[:-1]]
    bv_rise = edges.loc[(prev == 0) & (edge_states == 1), "Timestamp"].values

    return pd_rows, edges, bv_rise


def _best_start_offset_by_intervals(
    bv_times: np.ndarray,
    harp_times: np.ndarray,
    *,
    max_shift: int = 5,
) -> Tuple[int, float]:
    """
    Pick an integer shift (within +/- max_shift) that best aligns the early inter-edge intervals.
    Returns (shift, rmse_of_intervals).
    """
    bv_times = np.asarray(bv_times, dtype=float)
    harp_times = np.asarray(harp_times, dtype=float)

    bv_dt = np.diff(bv_times)
    harp_dt = np.diff(harp_times)

    n = min(len(bv_dt), len(harp_dt), 20)
    if n <= 2:
        return 0, float("inf")

    bv_dt = bv_dt[:n]
    harp_dt = harp_dt[:n]

    best_shift = 0
    best_rmse = float("inf")

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            a = bv_dt[shift:]
            b = harp_dt[: len(a)]
        else:
            b = harp_dt[-shift:]
            a = bv_dt[: len(b)]

        m = min(len(a), len(b))
        if m < 5:
            continue

        rmse = float(np.sqrt(np.mean((a[:m] - b[:m]) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift

    return best_shift, best_rmse


def _fit_affine(
    bv_edge_times: np.ndarray,
    harp_edge_times: np.ndarray,
) -> Tuple[float, float, int]:
    """
    Fit t_harp = slope * t_bv + intercept using least squares (polyfit).
    """
    bv = np.asarray(bv_edge_times, dtype=float)
    hp = np.asarray(harp_edge_times, dtype=float)
    n = min(len(bv), len(hp))
    if n < 3:
        raise ValueError("Need at least 3 matched edges to fit affine map.")
    bv = bv[:n]
    hp = hp[:n]
    slope, intercept = np.polyfit(bv, hp, 1)
    return float(slope), float(intercept), int(n)


def _estimate_bv_duration_seconds(bv: pd.DataFrame, *, t_bv0: float) -> float:
    """
    Estimate BV duration (relative to t_bv0). Prefer Frame rows if present.
    """
    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if frame_mask.any():
        return float(bv.loc[frame_mask, "Timestamp"].max() - t_bv0)
    return float(bv["Timestamp"].max() - t_bv0)


def _fit_frame_modclass_to_harp_edges(
    *,
    bv: pd.DataFrame,
    harp_rise_s: np.ndarray,
    modulo_frames: int,
    slope_bounds: Tuple[float, float],
    min_pairs: int,
    max_edge_start: int,
    max_pairs_per_fit: int,
    rmse_tol: float,
    n_frac: float,
    b_abs_max: float,
) -> Tuple[float, float, int, int, int, float]:
    """
    Robust fallback when BV photodiode is absent.

    Use modulo-class frames:
      - take Frame rows
      - for each phase in [0..modulo_frames-1], candidate flip times are frames where Frame%modulo==phase
      - search edge_start shift into harp_rise_s
      - fit affine t_harp = a*t_bv + b on BV-relative times
      - select among near-best fits the one with smallest |b| (anchors to small correction)
    """
    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if not frame_mask.any():
        raise RuntimeError("No Frame rows found; cannot do frame-modclass alignment.")

    frame_rows = bv.loc[frame_mask, ["Frame", "timestamp_bv_rel"]].copy()
    frames = frame_rows["Frame"].to_numpy(dtype=int)
    t_frame = frame_rows["timestamp_bv_rel"].to_numpy(dtype=float)

    if len(harp_rise_s) < min_pairs:
        raise RuntimeError(f"Not enough HARP edges (have {len(harp_rise_s)}, need {min_pairs}).")

    max_edge_start = int(min(max_edge_start, max(0, len(harp_rise_s) - min_pairs)))

    candidates: List[Dict[str, Any]] = []
    for phase in range(int(modulo_frames)):
        sel = (frames % modulo_frames) == phase
        if int(np.sum(sel)) < min_pairs:
            continue

        bv_flip_t = t_frame[sel]
        order = np.argsort(bv_flip_t)
        bv_flip_t = bv_flip_t[order][:max_pairs_per_fit]

        for edge_start in range(max_edge_start + 1):
            n = min(len(bv_flip_t), len(harp_rise_s) - edge_start, max_pairs_per_fit)
            if n < min_pairs:
                continue

            t_bv = bv_flip_t[:n]
            t_hp = harp_rise_s[edge_start:edge_start + n]

            a, b, _ = _fit_affine(t_bv, t_hp)

            if not (np.isfinite(a) and np.isfinite(b) and a > 0 and slope_bounds[0] <= a <= slope_bounds[1]):
                continue

            err = (a * t_bv + b) - t_hp
            rmse = float(np.sqrt(np.mean(err**2)))
            candidates.append(dict(phase=int(phase), edge_start=int(edge_start), a=float(a), b=float(b), rmse=float(rmse), n=int(n)))

    if not candidates:
        raise RuntimeError("No valid (phase, edge_start) candidates found. Check modulo/edge detection/slope bounds.")

    best_rmse = min(c["rmse"] for c in candidates)
    max_n = max(c["n"] for c in candidates)

    good = [
        c for c in candidates
        if c["rmse"] <= best_rmse * (1.0 + rmse_tol)
        and c["n"] >= int(max_n * n_frac)
        and abs(c["b"]) <= b_abs_max
    ]
    if not good:
        # Relax |b| if nothing passes
        good = [
            c for c in candidates
            if c["rmse"] <= best_rmse * (1.0 + rmse_tol)
            and c["n"] >= int(max_n * n_frac)
        ]

    good.sort(key=lambda c: (abs(c["b"]), c["edge_start"], c["rmse"]))
    best = good[0]

    return best["a"], best["b"], best["phase"], best["edge_start"], best["n"], best["rmse"]
