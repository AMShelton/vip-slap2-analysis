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
    used_piecewise_warp: bool = False


def correct_event_log(
    bonsai_event_log_csv: PathLike,
    photodiode_pkl: PathLike,
    savepath: Optional[PathLike] = None,
    qc_dir: Optional[PathLike] = None,
    modulo_frames: int = 60,
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

    If BV photodiode exists, we build a piecewise-linear warp anchored at matched PD rises
    (removes per-pulse jitter that comes from BV event quantization).
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

    # --- YOUR REQUESTED photodiode_state generation (dense ffill across all rows)
    bv = _add_bv_photodiode_columns_dense(bv)

    # BV-relative timestamps (prevents huge intercepts)
    t_bv0 = _choose_bv_t0(bv)
    bv = bv.copy()
    bv["timestamp_bv_rel"] = bv["Timestamp"].to_numpy(dtype=float) - float(t_bv0)

    # HARP photodiode edges in HARP-relative seconds
    display_rate_hz, harp_rise_s, rise_period_s, thr = _estimate_display_rate_from_harp(
        harp_pd,
        modulo_frames=modulo_frames,
        min_edge_separation_s=min_edge_separation_s,
    )

    # Duration guardrail (avoids silently aligning wrong/truncated PD)
    bv_duration = _estimate_bv_duration_seconds(bv, t_bv0=t_bv0)
    harp_duration = float(harp_pd.index.values[-1] - harp_pd.index.values[0])
    if harp_duration < 0.5 * bv_duration:
        raise RuntimeError(
            f"Photodiode duration mismatch: BV~{bv_duration:.1f}s, HARP~{harp_duration:.1f}s. "
            "Likely wrong/truncated photodiode.pkl."
        )

    # --- Try BV photodiode path
    pd_parse = _bv_photodiode_rises_from_state(
    bv,
    mode="auto",
    harp_rise_s=harp_rise_s,
    max_shift=10,
)
    if pd_parse is not None and len(pd_parse) >= 3 and len(harp_rise_s) >= 3:
        bv_rise_rel = pd_parse  # already in BV-relative seconds
        shift, rmse_int = _best_start_offset_by_intervals(bv_rise_rel, harp_rise_s, max_shift=10)

        bv_r = bv_rise_rel.copy()
        hp_r = harp_rise_s.copy()
        if shift > 0:
            bv_r = bv_r[shift:]
        elif shift < 0:
            hp_r = hp_r[-shift:]

        # Fit affine (for meta + initial correspondence)
        slope, intercept, matched = _fit_affine(bv_r, hp_r)

        if use_piecewise_warp:
            # Piecewise linear warp anchored at matched rises (removes per-pulse jitter)
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
        # --- Fallback: no BV photodiode, use robust modulo-class fit anchored by min |b|
        slope, intercept, phase, edge_start, matched, rmse = _fit_frame_modclass_to_harp_edges_anchored(
            bv=bv,
            harp_rise_s=harp_rise_s,
            modulo_frames=modulo_frames,
        )
        corrected = slope * bv["timestamp_bv_rel"].to_numpy(dtype=float) + intercept
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


def analyze_jitter(
    corrected_event_log: pd.DataFrame,
    photodiode_pkl: PathLike,
    qc_dir: PathLike,
    min_edge_separation_s: float = 0.005,
) -> pd.DataFrame:
    """
    Quantify jitter between BV photodiode rises (mapped into HARP time) and HARP photodiode rises.
    Saves:
      - jitter_residuals.csv
      - jitter_residuals.png (residual vs time)
      - jitter_hist.png
    Returns residual dataframe.
    """
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    harp_pd = _load_photodiode(Path(photodiode_pkl))
    t_rel = harp_pd.index.values.astype(float) - harp_pd.index.values.astype(float)[0]
    y = harp_pd["AnalogInput0"].values.astype(float)

    _, _, harp_rise, _, _ = _get_signal_edges(y, t_rel, min_edge_separation_s=min_edge_separation_s)

    bv_rise_corr = _bv_photodiode_rises_from_corrected(corrected_event_log)
    if bv_rise_corr is None or len(bv_rise_corr) < 3:
        raise RuntimeError("No BV photodiode rises found in corrected_event_log to quantify jitter.")

    shift, _ = _best_start_offset_by_intervals(bv_rise_corr, harp_rise, max_shift=10)
    bv_r, hp_r = _apply_shift(bv_rise_corr, harp_rise, shift)

    res = bv_r - hp_r
    df = pd.DataFrame({"harp_rise_s": hp_r, "bv_rise_corrected_s": bv_r, "residual_s": res})
    df.to_csv(qc_dir / "jitter_residuals.csv", index=False)

    # Plots
    plt.figure(figsize=(10,4))
    plt.plot(hp_r, res * 1000.0, marker="o", linestyle="none")
    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("HARP rise time (s)")
    plt.ylabel("Residual (ms)")
    plt.title("Photodiode rise jitter (BV corrected - HARP)")
    plt.tight_layout()
    plt.savefig(qc_dir / "jitter_residuals.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(res * 1000.0, bins=40)
    plt.xlabel("Residual (ms)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(qc_dir / "jitter_hist.png", dpi=200)
    plt.close()

    return df


# =============================================================================
# Internals
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
    Exactly your requested implementation: photodiode_state forward-filled across all rows.
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


def _estimate_display_rate_from_harp(
    photodiode_df: pd.DataFrame,
    *,
    modulo_frames: int,
    min_edge_separation_s: float,
) -> Tuple[float, np.ndarray, float, float]:
    t = photodiode_df.index.values - photodiode_df.index.values[0]  # HARP-relative seconds
    y = photodiode_df["AnalogInput0"].values
    _, _, t_rise, _, thr = _get_signal_edges(y, t, min_edge_separation_s=min_edge_separation_s)
    if len(t_rise) < 3:
        raise ValueError("Not enough HARP photodiode rising edges.")
    period = float(np.median(np.diff(t_rise)))
    display_rate = float(modulo_frames / period)
    return display_rate, t_rise, period, thr


def _estimate_bv_duration_seconds(bv: pd.DataFrame, *, t_bv0: float) -> float:
    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    if frame_mask.any():
        return float(bv.loc[frame_mask, "Timestamp"].max() - t_bv0)
    return float(bv["Timestamp"].max() - t_bv0)


def _bv_photodiode_rises_from_state(
    bv: pd.DataFrame,
    *,
    mode: str = "auto",   # "auto" | "current" | "midpoint" | "previous"
    harp_rise_s: Optional[np.ndarray] = None,
    max_shift: int = 10,
) -> Optional[np.ndarray]:
    """
    Extract BV photodiode rise times (BV-relative seconds) from photodiode_state,
    using only Frame rows (reduces scheduling jitter from Photodiode-* events).

    The key: ds==1 is observed at index i, meaning the transition occurred between
    frames i-1 and i. Depending on how BV logs the state, using t[i] can be ~1 frame late.

    mode:
      - "current":  rise time = t[i]
      - "midpoint": rise time = 0.5*(t[i-1] + t[i])
      - "previous": rise time = t[i-1]
      - "auto":     tries the above and picks the one with smallest residual RMSE
                    after affine fitting to harp_rise_s (requires harp_rise_s).
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

    # auto-select best mode by fitting to harp and checking residuals
    if harp_rise_s is None or len(harp_rise_s) < 3:
        # fallback if user asked for auto but didn't provide harp_rise_s
        r = rises_with("previous")
        return np.asarray(r, dtype=float) if len(r) else None

    modes = ["previous", "midpoint", "current"]
    best = None

    for m in modes:
        bv_r = rises_with(m)
        if len(bv_r) < 3:
            continue

        # align start by interval pattern, then fit affine and score residuals
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

    # best[1] is selected mode; return the full rise vector in that mode
    return np.asarray(best[2], dtype=float)


def _bv_photodiode_rises_from_corrected(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Extract BV photodiode rises in corrected (HARP) time by using photodiode_event rows.
    """
    if "photodiode_event" not in df.columns or "corrected_timestamps" not in df.columns:
        return None
    mask = ~df["photodiode_event"].isna()
    if not mask.any():
        return None

    sub = df.loc[mask, ["corrected_timestamps", "photodiode_event"]].copy()
    sub = sub[sub["photodiode_event"].ne(sub["photodiode_event"].shift())]
    states = sub["photodiode_event"].to_numpy(dtype=float)
    times = sub["corrected_timestamps"].to_numpy(dtype=float)
    prev = np.r_[states[0], states[:-1]]
    rises = times[(prev == 0) & (states == 1)]
    return np.asarray(rises, dtype=float)


def _best_start_offset_by_intervals(bv_times: np.ndarray, harp_times: np.ndarray, *, max_shift: int = 5) -> Tuple[int, float]:
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
    """
    Piecewise-linear mapping that ensures t_src[k] -> t_dst[k] exactly.
    Uses linear extrapolation outside the anchor range.
    """
    t_query = np.asarray(t_query, dtype=float)
    t_src = np.asarray(t_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)

    if len(t_src) < 2:
        raise ValueError("Need at least 2 anchors for piecewise warp.")

    # Ensure sorted
    order = np.argsort(t_src)
    t_src = t_src[order]
    t_dst = t_dst[order]

    # Core interpolation
    out = np.interp(t_query, t_src, t_dst)

    # Linear extrapolation on left/right using first/last segment slope
    left = t_query < t_src[0]
    right = t_query > t_src[-1]

    m0 = (t_dst[1] - t_dst[0]) / (t_src[1] - t_src[0])
    b0 = t_dst[0] - m0 * t_src[0]
    out[left] = m0 * t_query[left] + b0

    m1 = (t_dst[-1] - t_dst[-2]) / (t_src[-1] - t_src[-2])
    b1 = t_dst[-1] - m1 * t_src[-1]
    out[right] = m1 * t_query[right] + b1

    return out


def _fit_frame_modclass_to_harp_edges_anchored(
    *,
    bv: pd.DataFrame,
    harp_rise_s: np.ndarray,
    modulo_frames: int,
    slope_bounds: Tuple[float, float] = (0.90, 1.10),
    min_pairs: int = 200,
    max_edge_start: int = 20000,
    max_pairs_per_fit: int = 5000,
    rmse_tol: float = 0.25,
    n_frac: float = 0.90,
    b_abs_max: float = 5.0,
) -> Tuple[float, float, int, int, int, float]:
    """
    Fallback: choose flip train as Frame%modulo==phase; match to harp_rise with edge_start shift.
    Then pick among near-best fits the one with smallest |b|.
    """
    frame_mask = bv["Value"].astype(str).str.lower().eq("frame")
    fr = bv.loc[frame_mask, ["Frame", "timestamp_bv_rel"]].copy()
    frames = fr["Frame"].to_numpy(dtype=int)
    t_frame = fr["timestamp_bv_rel"].to_numpy(dtype=float)

    max_edge_start = int(min(max_edge_start, max(0, len(harp_rise_s) - min_pairs)))

    candidates: List[Dict[str, Any]] = []
    for phase in range(int(modulo_frames)):
        sel = (frames % modulo_frames) == phase
        if int(np.sum(sel)) < min_pairs:
            continue
        bv_flip = np.sort(t_frame[sel])[:max_pairs_per_fit]

        for edge_start in range(max_edge_start + 1):
            n = min(len(bv_flip), len(harp_rise_s) - edge_start, max_pairs_per_fit)
            if n < min_pairs:
                continue
            t_bv = bv_flip[:n]
            t_hp = harp_rise_s[edge_start:edge_start + n]
            a, b, _ = _fit_affine(t_bv, t_hp)
            if not (slope_bounds[0] <= a <= slope_bounds[1]):
                continue
            err = (a * t_bv + b) - t_hp
            rmse = float(np.sqrt(np.mean(err**2)))
            candidates.append(dict(phase=phase, edge_start=edge_start, a=a, b=b, rmse=rmse, n=n))

    if not candidates:
        raise RuntimeError("No valid candidates in frame-modclass fallback.")

    best_rmse = min(c["rmse"] for c in candidates)
    max_n = max(c["n"] for c in candidates)

    good = [
        c for c in candidates
        if c["rmse"] <= best_rmse * (1.0 + rmse_tol)
        and c["n"] >= int(max_n * n_frac)
        and abs(c["b"]) <= b_abs_max
    ]
    if not good:
        good = [c for c in candidates if c["rmse"] <= best_rmse * (1.0 + rmse_tol) and c["n"] >= int(max_n * n_frac)]

    good.sort(key=lambda c: (abs(c["b"]), c["edge_start"], c["rmse"]))
    best = good[0]
    return best["a"], best["b"], best["phase"], best["edge_start"], best["n"], best["rmse"]

