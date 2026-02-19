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
from typing import Optional, Tuple, Dict, Any

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

# -----------------------------
# Helpers
# -----------------------------

@dataclass
class AffineMap:
    """Map BV time -> HARP time: t_harp = a * t_bv + b"""
    a: float
    b: float
    phase: int
    n_pairs: int
    rmse: float
    max_abs_err: float


def _ensure_dir(p: Optional[Path]) -> Optional[Path]:
    if p is None:
        return None
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_time_and_signal_from_harp_photodiode_df(photodiode_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    photodiode_df: indexed by HARP time (often large absolute counter-like values)
                  has column 'AnalogInput0' (per your example).
    Returns:
        t (seconds relative to start), y (signal)
    """
    if photodiode_df.index.name is None:
        # still works, but helpful to warn in QC metrics
        pass

    # Convert to "seconds since start" (your prior code did this)
    t = photodiode_df.index.to_numpy(dtype=float)
    t = t - t[0]

    # Choose a reasonable default signal column
    if "AnalogInput0" in photodiode_df.columns:
        y = photodiode_df["AnalogInput0"].to_numpy(dtype=float)
    else:
        # fallback: first numeric column
        num_cols = [c for c in photodiode_df.columns if np.issubdtype(photodiode_df[c].dtype, np.number)]
        if not num_cols:
            raise ValueError("photodiode_df has no numeric column to use as photodiode signal.")
        y = photodiode_df[num_cols[0]].to_numpy(dtype=float)

    return t, y


def get_signal_edges(signal: np.ndarray, time: np.ndarray, debounce_s: float = 0.005) -> Dict[str, Any]:
    """
    Robust-ish edge finder for a two-level photodiode signal.
    Returns dict containing rise_idx, fall_idx, t_rise, t_fall, threshold.
    """
    signal = np.asarray(signal)
    time = np.asarray(time)
    if signal.shape != time.shape:
        raise ValueError("signal and time must have the same shape")

    # Smooth a bit
    y_s = medfilt(signal, kernel_size=5) if len(signal) >= 5 else signal.copy()

    # Threshold at midpoint of robust high/low percentiles
    thr = (np.percentile(y_s, 95) + np.percentile(y_s, 5)) / 2.0
    binary = (y_s > thr).astype(np.int8)

    db = np.diff(binary, prepend=binary[0])
    rise_idx = np.where(db == 1)[0]
    fall_idx = np.where(db == -1)[0]

    t_rise = time[rise_idx]
    t_fall = time[fall_idx]

    # Debounce: remove edges that happen too close together
    def _debounce(t_edge: np.ndarray) -> np.ndarray:
        if len(t_edge) <= 1:
            return t_edge
        keep = [0]
        for i in range(1, len(t_edge)):
            if (t_edge[i] - t_edge[keep[-1]]) > debounce_s:
                keep.append(i)
        return t_edge[keep]

    t_rise = _debounce(t_rise)
    t_fall = _debounce(t_fall)

    return {
        "threshold": float(thr),
        "rise_idx": rise_idx,
        "fall_idx": fall_idx,
        "t_rise": t_rise,
        "t_fall": t_fall,
    }


def _extract_frame_times_from_stimulus_df(stimulus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns at least: 'Frame', 'Timestamp', 'Value'
    Returns a df of rows corresponding to frame timing, with unique Frame -> Timestamp.
    """
    required = {"Frame", "Timestamp", "Value"}
    missing = required - set(stimulus_df.columns)
    if missing:
        raise ValueError(f"stimulus_df missing required columns: {sorted(missing)}")

    frame_df = stimulus_df[stimulus_df["Value"].astype(str).str.lower() == "frame"][["Frame", "Timestamp"]].copy()

    # If multiple 'Frame' entries exist for a given frame index, keep the first.
    frame_df = frame_df.dropna(subset=["Frame", "Timestamp"])
    frame_df["Frame"] = frame_df["Frame"].astype(int)
    frame_df = frame_df.sort_values("Frame").drop_duplicates("Frame", keep="first").reset_index(drop=True)
    return frame_df


def _fit_affine(t_bv: np.ndarray, t_harp: np.ndarray) -> Tuple[float, float]:
    """
    Least-squares fit: t_harp = a * t_bv + b
    """
    t_bv = np.asarray(t_bv, dtype=float)
    t_harp = np.asarray(t_harp, dtype=float)
    if len(t_bv) < 2:
        raise ValueError("Need at least 2 matched points to fit affine mapping.")
    A = np.column_stack([t_bv, np.ones_like(t_bv)])
    a, b = np.linalg.lstsq(A, t_harp, rcond=None)[0]
    return float(a), float(b)


def _compute_fit_errors(t_bv: np.ndarray, t_harp: np.ndarray, a: float, b: float) -> Tuple[float, float, np.ndarray]:
    pred = a * np.asarray(t_bv) + b
    err = pred - np.asarray(t_harp)
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err))) if len(err) else float("nan")
    return rmse, max_abs, err


### Use affine transform to model jitter of photodiode signal for better time alignment

def fit_bv_to_harp_affine_from_photodiode(
    stimulus_df: pd.DataFrame,
    photodiode_df: pd.DataFrame,
    modulo: int = 60,
    edge: str = "rise",
    max_pairs: Optional[int] = None,
    qc_dir: Optional[Path] = None,
    qc_prefix: str = "time_alignment",
) -> AffineMap:
    """
    Fit an affine transform mapping BonVision timestamps -> HARP time using many photodiode flips.

    Strategy:
      - Extract HARP photodiode edge times (rise or fall).
      - Extract BV frame timestamps.
      - Assume the photodiode marker flips every `modulo` frames, but the phase (which frame index aligns
        to the first flip) may be unknown.
      - Try all phases in [0, modulo-1], build matched pairs:
            BV time at frames: phase + modulo*k   <->   HARP flip time #k
        Fit affine, score by RMSE, choose best.

    Returns:
      AffineMap(a,b,phase,n_pairs,rmse,max_abs_err)
    """
    qc_dir = _ensure_dir(qc_dir)

    frame_df = _extract_frame_times_from_stimulus_df(stimulus_df)
    t_bv_by_frame = dict(zip(frame_df["Frame"].to_numpy(), frame_df["Timestamp"].to_numpy(dtype=float)))

    # HARP photodiode edges
    t_harp, y_harp = _get_time_and_signal_from_harp_photodiode_df(photodiode_df)
    edges = get_signal_edges(y_harp, t_harp)
    t_edges = edges["t_rise"] if edge.lower() == "rise" else edges["t_fall"]
    if max_pairs is not None:
        t_edges = t_edges[:max_pairs]

    if len(t_edges) < 2:
        raise ValueError(f"Not enough photodiode {edge} edges to fit (found {len(t_edges)}).")

    best: Optional[AffineMap] = None
    best_err: Optional[np.ndarray] = None
    best_pairs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # (frames, t_bv, t_harp)

    # Try phases
    for phase in range(modulo):
        # Expected frames for each flip k
        frames = phase + modulo * np.arange(len(t_edges), dtype=int)

        # Keep only frames that exist in the BV log
        t_bv = []
        t_h = []
        f_used = []
        for f, th in zip(frames, t_edges):
            if f in t_bv_by_frame:
                t_bv.append(t_bv_by_frame[f])
                t_h.append(th)
                f_used.append(f)

        if len(t_bv) < 2:
            continue

        t_bv = np.asarray(t_bv, dtype=float)
        t_h = np.asarray(t_h, dtype=float)
        f_used = np.asarray(f_used, dtype=int)

        a, b = _fit_affine(t_bv, t_h)
        rmse, max_abs, err = _compute_fit_errors(t_bv, t_h, a, b)

        # Prefer lower RMSE; break ties with more pairs
        if (best is None) or (rmse < best.rmse - 1e-12) or (abs(rmse - best.rmse) <= 1e-12 and len(t_bv) > best.n_pairs):
            best = AffineMap(a=a, b=b, phase=phase, n_pairs=len(t_bv), rmse=rmse, max_abs_err=max_abs)
            best_err = err
            best_pairs = (f_used, t_bv, t_h)

    if best is None or best_pairs is None or best_err is None:
        raise RuntimeError(
            "Could not fit BV->HARP affine mapping. "
            "Check that stimulus_df contains Frame timing events and photodiode edges are present."
        )

    # QC artifacts
    if qc_dir is not None:
        f_used, t_bv, t_h = best_pairs

        metrics = {
            "method": "affine_fit_from_harp_photodiode_vs_bv_frames",
            "edge": edge,
            "modulo": modulo,
            "phase": int(best.phase),
            "a": float(best.a),
            "b": float(best.b),
            "n_pairs": int(best.n_pairs),
            "rmse_s": float(best.rmse),
            "max_abs_err_s": float(best.max_abs_err),
            "harp_pd_threshold": float(edges["threshold"]),
        }
        (qc_dir / f"{qc_prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))

        # 1) Residual vs time
        plt.figure()
        plt.plot(t_h, best_err, marker="o", linestyle="none")
        plt.axhline(0.0)
        plt.xlabel("HARP photodiode edge time (s, rel.)")
        plt.ylabel("Prediction error (a*t_BV + b - t_HARP) (s)")
        plt.title(f"BV→HARP affine fit residuals (phase={best.phase}, n={best.n_pairs})")
        plt.tight_layout()
        plt.savefig(qc_dir / f"{qc_prefix}_residuals_vs_time.png", dpi=200)
        plt.close()

        # 2) Residual histogram
        plt.figure()
        plt.hist(best_err, bins=60)
        plt.xlabel("Residual (s)")
        plt.ylabel("Count")
        plt.title("Residual distribution")
        plt.tight_layout()
        plt.savefig(qc_dir / f"{qc_prefix}_residual_hist.png", dpi=200)
        plt.close()

        # 3) Photodiode trace with detected edges (first ~10s)
        plt.figure()
        # plot first N seconds worth of samples
        tmax = min(10.0, float(t_harp[-1]))
        mask = t_harp <= tmax
        plt.plot(t_harp[mask], y_harp[mask])
        for te in t_edges[t_edges <= tmax]:
            plt.axvline(te, linestyle="--")
        plt.xlabel("HARP time (s, rel.)")
        plt.ylabel("Photodiode signal")
        plt.title("HARP photodiode trace with detected edges")
        plt.tight_layout()
        plt.savefig(qc_dir / f"{qc_prefix}_harp_photodiode_edges.png", dpi=200)
        plt.close()

        # 4) Matched points scatter (BV time vs HARP time)
        plt.figure()
        plt.plot(t_bv, t_h, marker="o", linestyle="none", label="matched")
        # fit line
        xline = np.linspace(np.min(t_bv), np.max(t_bv), 100)
        plt.plot(xline, best.a * xline + best.b, label="fit")
        plt.xlabel("BonVision time (s)")
        plt.ylabel("HARP time (s, rel.)")
        plt.title("Matched BV frame times to HARP photodiode edges")
        plt.legend()
        plt.tight_layout()
        plt.savefig(qc_dir / f"{qc_prefix}_matched_points.png", dpi=200)
        plt.close()

        # 5) Save matched pairs table (for debugging)
        pairs_df = pd.DataFrame(
            {
                "frame": f_used,
                "t_bv": t_bv,
                "t_harp_edge": t_h,
                "t_harp_pred": best.a * t_bv + best.b,
                "residual_s": best_err,
            }
        )
        pairs_df.to_csv(qc_dir / f"{qc_prefix}_matched_pairs.csv", index=False)

    return best

#### If photodiode is represented in BonVision time, use this pathway for aligning behavior events and HARP time

def fit_bv_to_harp_affine_from_dual_photodiodes(
    stimulus_df: pd.DataFrame,
    harp_photodiode_df: pd.DataFrame,
    bv_photodiode_col: Optional[str] = None,
    bv_time_col: str = "Timestamp",
    qc_dir: Optional[Path] = None,
    qc_prefix: str = "time_alignment_dual_pd",
) -> AffineMap:
    """
    Skeleton: align a photodiode recorded in BOTH BV and HARP time bases.

    Expected:
      - stimulus_df contains a BV photodiode trace column (bv_photodiode_col), sampled at BV frame times
      - harp_photodiode_df contains the HARP photodiode trace, sampled at HARP time

    Outline:
      1) Extract edges from BV photodiode vs BV time.
      2) Extract edges from HARP photodiode vs HARP time.
      3) Match edge sequences robustly (e.g., by:
           - taking first K edges and doing a linear fit,
           - or using cross-correlation on edge trains,
           - or dynamic time warping if drift is substantial).
      4) Fit affine: t_harp = a*t_bv + b using matched edges.
      5) Emit QC plots/metrics.

    For now, this implements (1)(2) and a *basic* matching approach:
      - use the first min(n_bv_edges, n_harp_edges) edges as correspondences.

    Upgrade later: robust matching (RANSAC) + outlier rejection + optional scaling drift segments.
    """
    qc_dir = _ensure_dir(qc_dir)

    # Pick photodiode column if not provided
    if bv_photodiode_col is None:
        candidates = [c for c in stimulus_df.columns if "photodiode" in str(c).lower() or "pd" == str(c).lower()]
        if len(candidates) == 1:
            bv_photodiode_col = candidates[0]
        elif len(candidates) > 1:
            # Prefer the most explicit
            bv_photodiode_col = sorted(candidates, key=lambda s: (("photodiode" not in s.lower()), len(s)))[0]
        else:
            raise ValueError(
                "stimulus_df does not appear to contain a BV photodiode column. "
                "Provide bv_photodiode_col explicitly or omit this pathway."
            )

    if bv_time_col not in stimulus_df.columns:
        raise ValueError(f"stimulus_df missing bv_time_col='{bv_time_col}'")

    # BV photodiode edges
    bv_time = stimulus_df[bv_time_col].to_numpy(dtype=float)
    bv_sig = stimulus_df[bv_photodiode_col].to_numpy(dtype=float)
    bv_edges = get_signal_edges(bv_sig, bv_time)
    t_bv_edges = bv_edges["t_rise"]  # choose rise by default

    # HARP photodiode edges
    t_harp, y_harp = _get_time_and_signal_from_harp_photodiode_df(harp_photodiode_df)
    harp_edges = get_signal_edges(y_harp, t_harp)
    t_harp_edges = harp_edges["t_rise"]

    n = min(len(t_bv_edges), len(t_harp_edges))
    if n < 2:
        raise ValueError(f"Not enough edges to fit dual-photodiode mapping (BV={len(t_bv_edges)}, HARP={len(t_harp_edges)}).")

    # Basic correspondence (placeholder)
    t_bv = t_bv_edges[:n]
    t_h = t_harp_edges[:n]

    a, b = _fit_affine(t_bv, t_h)
    rmse, max_abs, err = _compute_fit_errors(t_bv, t_h, a, b)

    amap = AffineMap(a=a, b=b, phase=-1, n_pairs=n, rmse=rmse, max_abs_err=max_abs)

    if qc_dir is not None:
        metrics = {
            "method": "affine_fit_from_dual_photodiodes_BASIC",
            "bv_photodiode_col": bv_photodiode_col,
            "bv_time_col": bv_time_col,
            "a": float(amap.a),
            "b": float(amap.b),
            "n_pairs": int(amap.n_pairs),
            "rmse_s": float(amap.rmse),
            "max_abs_err_s": float(amap.max_abs_err),
            "bv_pd_threshold": float(bv_edges["threshold"]),
            "harp_pd_threshold": float(harp_edges["threshold"]),
            "note": "This is a skeleton alignment. Replace naive edge pairing with robust matching + outlier rejection.",
        }
        (qc_dir / f"{qc_prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))

        plt.figure()
        plt.plot(t_h, err, marker="o", linestyle="none")
        plt.axhline(0.0)
        plt.xlabel("HARP PD edge time (s, rel.)")
        plt.ylabel("Prediction error (s)")
        plt.title("Dual-PD affine fit residuals (BASIC matching)")
        plt.tight_layout()
        plt.savefig(qc_dir / f"{qc_prefix}_residuals_vs_time.png", dpi=200)
        plt.close()

    return amap


# -----------------------------
# Updated API: get_time_offset + correct_event_log
# -----------------------------

def get_time_offset(
    photodiode_df: pd.DataFrame,
    stimulus_df: Optional[pd.DataFrame] = None,
    modulo: int = 60,
    qc_dir: Optional[Path] = None,
    qc_prefix: str = "time_alignment",
) -> Dict[str, Any]:
    """
    Backwards-compatible-ish replacement for get_time_offset.

    OLD behavior: returned a single offset b such that t_harp ~= t_bv + b
    NEW behavior: returns dict with affine map parameters (a,b) and metadata.

    If stimulus_df is provided, we fit affine using frames and HARP photodiode.
    If stimulus_df has BV photodiode, we will prefer the dual-photodiode pathway (skeleton).
    """
    if stimulus_df is None:
        raise ValueError("New get_time_offset requires stimulus_df to fit an affine mapping.")

    # If BV photodiode is present, prefer that pathway
    has_bv_pd = any("photodiode" in str(c).lower() for c in stimulus_df.columns)
    if has_bv_pd:
        amap = fit_bv_to_harp_affine_from_dual_photodiodes(
            stimulus_df=stimulus_df,
            harp_photodiode_df=photodiode_df,
            qc_dir=qc_dir,
            qc_prefix=f"{qc_prefix}_dual_pd",
        )
        return {"a": amap.a, "b": amap.b, "method": "dual_photodiode_basic", "phase": amap.phase}

    # Otherwise: infer via modulo frame flipping and fit affine using many flips
    amap = fit_bv_to_harp_affine_from_photodiode(
        stimulus_df=stimulus_df,
        photodiode_df=photodiode_df,
        modulo=modulo,
        edge="rise",
        qc_dir=qc_dir,
        qc_prefix=qc_prefix,
    )
    return {"a": amap.a, "b": amap.b, "method": "frame_vs_harp_photodiode_affine", "phase": amap.phase}


def correct_event_log(
    stimulus_df: pd.DataFrame,
    photodiode_df: pd.DataFrame,
    savepath: Optional[Path] = None,
    qc_dir: Optional[Path] = None,
    modulo: int = 60,
) -> pd.DataFrame:
    """
    Updated correct_event_log:
      - preserves your "insert missing first stimulus" behavior
      - computes BV->HARP affine mapping using many flips (or dual-PD pathway if available)
      - writes corrected_timestamp_harp = a * Timestamp + b
      - saves QC artifacts to qc_dir if provided
    """
    stimulus_df = stimulus_df.copy()

    # Preserve original "missing first tif" patch
    try:
        tif_row = stimulus_df[stimulus_df["Value"].astype(str).str.contains(r"\.tif|\.tiff", case=False, na=False)].iloc[0]
        tif_value = tif_row["Value"]
        new_row = pd.DataFrame([{"Frame": -1, "Timestamp": 0.0, "Value": "Frame"}])
        new_row_1 = pd.DataFrame([{"Frame": -1, "Timestamp": 0.0, "Value": tif_value}])
        stimulus_df.index = stimulus_df.index + 1
        stimulus_df = pd.concat([new_row, new_row_1, stimulus_df]).reset_index(drop=True)
    except Exception:
        # If no tif found, just proceed
        pass

    qc_dir = _ensure_dir(qc_dir)

    fit = get_time_offset(
        photodiode_df=photodiode_df,
        stimulus_df=stimulus_df,
        modulo=modulo,
        qc_dir=qc_dir,
        qc_prefix="time_alignment",
    )
    a = float(fit["a"])
    b = float(fit["b"])

    stimulus_df["corrected_timestamp_harp"] = a * stimulus_df["Timestamp"].to_numpy(dtype=float) + b

    # Save out
    if savepath is not None:
        savepath = Path(savepath)
        stimulus_df.to_csv(savepath, index=False)

    # Also store a tiny “header” row of fit params (optional, but handy)
    if qc_dir is not None:
        summary = {
            "a": a,
            "b": b,
            "method": fit.get("method", ""),
            "phase": int(fit.get("phase", -1)),
            "modulo": int(modulo),
            "output_column": "corrected_timestamp_harp",
        }
        (qc_dir / "time_alignment_summary.json").write_text(json.dumps(summary, indent=2))

    return stimulus_df

