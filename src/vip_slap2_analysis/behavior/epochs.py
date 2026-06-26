"""Imaging-epoch detection utilities for behavior-aligned physiology sessions.

The preferred detector for SLAP2 sessions is pulse-train based: DI3 is treated as
an acquisition line/cycle clock, and large gaps between rising edges define breaks
between imaging epochs.  A legacy level-based detector is retained for sessions
where DI3 is a sustained acquisition gate rather than a pulse train.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label


def _as_bool_signal(signal) -> np.ndarray:
    """Convert a digital input vector to boolean with robust numeric handling."""
    x = np.asarray(signal)
    if x.dtype == bool:
        return x.reshape(-1)
    xf = x.astype(float).reshape(-1)
    finite = np.isfinite(xf)
    if not finite.any():
        return np.zeros_like(xf, dtype=bool)
    vals = xf[finite]
    # Use a midpoint threshold when the digital line is stored as 0/1 or voltage.
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    thr = 0.5 * (lo + hi) if hi > lo else 0.5
    return xf > thr


def _rising_edges(signal_bool: np.ndarray) -> np.ndarray:
    """Return sample indices of low-to-high transitions including an active first sample."""
    s = np.asarray(signal_bool, dtype=bool).reshape(-1)
    if s.size == 0:
        return np.array([], dtype=int)
    return np.flatnonzero(np.diff(s.astype(np.int8), prepend=0) == 1)


def detect_pulse_train_epochs(
    signal,
    time,
    *,
    gap_factor: float = 10.0,
    min_gap_s: float = 0.5,
    min_duration: float = 5.0,
    min_pulses: int = 10,
) -> Tuple[List[List[float]], Dict[str, object]]:
    """Detect imaging epochs from gaps in a DI3 pulse train.

    Parameters
    ----------
    signal, time
        Digital acquisition line and matching timestamps.
    gap_factor
        A gap is called when inter-rising-edge interval exceeds
        ``gap_factor * median_period``.
    min_gap_s
        Absolute lower bound for a gap call, protecting against high-rate jitter.
    min_duration
        Minimum retained epoch duration in seconds.
    min_pulses
        Minimum number of rising edges in a retained epoch.

    Returns
    -------
    epochs, diagnostics
        Epoch rows are ``[start_idx, end_idx, start_time, end_time]``.
    """
    t = np.asarray(time, dtype=float).reshape(-1)
    s = _as_bool_signal(signal)
    if t.size != s.size:
        raise ValueError(f"signal and time length mismatch: {s.size} vs {t.size}")
    edges = _rising_edges(s)
    if edges.size < max(2, int(min_pulses)):
        return [], {
            "method": "pulse_train",
            "n_rising_edges": int(edges.size),
            "warnings": ["Too few DI3 rising edges for pulse-train epoch detection."],
        }
    edge_t = t[edges]
    intervals = np.diff(edge_t)
    finite_intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if finite_intervals.size == 0:
        return [], {
            "method": "pulse_train",
            "n_rising_edges": int(edges.size),
            "warnings": ["No finite positive rising-edge intervals."],
        }
    period = float(np.nanmedian(finite_intervals))
    gap_threshold = float(max(min_gap_s, gap_factor * period))
    gap_idx = np.flatnonzero(intervals > gap_threshold)
    starts = np.concatenate([[0], gap_idx + 1])
    stops = np.concatenate([gap_idx + 1, [edges.size]])
    epochs: List[List[float]] = []
    gap_rows = []
    for gi in gap_idx:
        gap_rows.append({
            "edge_before_index": int(gi),
            "edge_after_index": int(gi + 1),
            "gap_start_time": float(edge_t[gi]),
            "gap_end_time": float(edge_t[gi + 1]),
            "gap_duration_s": float(edge_t[gi + 1] - edge_t[gi]),
        })
    for a, b in zip(starts, stops):
        if b <= a:
            continue
        n_pulse = int(b - a)
        start_edge = int(edges[a])
        end_edge = int(edges[b - 1])
        start_time = float(t[start_edge])
        end_time = float(t[end_edge])
        duration = end_time - start_time
        if n_pulse >= int(min_pulses) and duration >= float(min_duration):
            epochs.append([start_edge, end_edge + 1, start_time, end_time])
    diagnostics: Dict[str, object] = {
        "method": "pulse_train",
        "n_rising_edges": int(edges.size),
        "median_period_s": period,
        "gap_threshold_s": gap_threshold,
        "gap_factor": float(gap_factor),
        "min_gap_s": float(min_gap_s),
        "n_gaps": int(gap_idx.size),
        "gaps": gap_rows,
        "n_epochs": int(len(epochs)),
        "min_duration_s": float(min_duration),
        "min_pulses": int(min_pulses),
        "warnings": [] if epochs else ["No pulse-train imaging epochs passed duration/pulse thresholds."],
    }
    return epochs, diagnostics


def detect_imaging_epochs(signal, time, gap_threshold=0.05, min_duration=5.0, mode="trial"):
    """Detect imaging intervals from a binary acquisition level signal.

    This is the legacy detector.  Use :func:`detect_pulse_train_epochs` for SLAP2
    DI3 line-clock data.
    """
    if mode not in ["trial", "continuous"]:
        raise ValueError("mode must be 'trial' or 'continuous'")
    signal = _as_bool_signal(signal)
    time = np.asarray(time, dtype=float)
    labeled, num_features = label(signal)
    slices = []
    for region in range(1, num_features + 1):
        idx = np.where(labeled == region)[0]
        if len(idx) == 0:
            continue
        slices.append((idx[0], idx[-1] + 1))
    if not slices:
        return []
    if mode == "continuous":
        start_idx = slices[0][0]
        end_idx = slices[-1][1]
        start_time = time[start_idx]
        end_time = time[end_idx - 1]
        if (end_time - start_time) >= min_duration:
            return [(start_idx, end_idx, start_time, end_time)]
        return []
    epochs = []
    current_start, current_end = slices[0]
    for i in range(1, len(slices)):
        next_start, next_end = slices[i]
        gap = time[next_start] - time[current_end - 1]
        if gap > gap_threshold:
            start_time = time[current_start]
            end_time = time[current_end - 1]
            if (end_time - start_time) >= min_duration:
                epochs.append((current_start, current_end, start_time, end_time))
            current_start, current_end = next_start, next_end
        else:
            current_end = next_end
    if (time[current_end - 1] - time[current_start]) >= min_duration:
        epochs.append((current_start, current_end, time[current_start], time[current_end - 1]))
    return epochs


def _get_di3_and_time(harp_df: pd.DataFrame, acq_time=None) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve DI3 and a zero-referenced HARP time vector.

    ``harp_df["time"]`` stores the absolute HARP clock in many extracted CSVs.
    Corrected Bonsai timestamps produced by ``correct_event_log`` are relative to
    the HARP photodiode recording start, so epoch detection must operate on the
    zero-referenced acquisition time supplied by ``load_harp_df`` whenever it is
    available.  Using the absolute HARP clock here silently puts epoch bounds in
    a different timebase than corrected events and yields zero event coverage.
    """
    if "DI3" not in harp_df.columns:
        raise ValueError("harp_df must contain a DI3 column")
    signal = harp_df["DI3"].to_numpy()

    if acq_time is not None:
        time = np.asarray(acq_time, dtype=float).reshape(-1)
    elif "time" in harp_df.columns:
        raw_time = harp_df["time"].to_numpy(dtype=float).reshape(-1)
        time = raw_time - float(raw_time[0]) if raw_time.size else raw_time
    else:
        raw_time = harp_df.index.to_numpy(dtype=float).reshape(-1)
        time = raw_time - float(raw_time[0]) if raw_time.size else raw_time

    if time.size != signal.size:
        raise ValueError(f"DI3/time length mismatch: {signal.size} vs {time.size}")
    return signal, time


def detect_epochs_adaptive(
    harp_df: pd.DataFrame,
    acq_time,
    *,
    acq_type: str,
    min_duration: float,
    gap_start: float,
    target_min: Optional[int] = None,
    epoch_detection_method: str = "pulse_train",
    pulse_gap_factor: float = 10.0,
    pulse_min_gap_s: float = 0.5,
    pulse_min_pulses: int = 10,
    pulse_min_duration: Optional[float] = None,
):
    """Detect imaging epochs using pulse-train or legacy level logic.

    Returns ``(epochs, threshold_or_gap_used)`` for backward compatibility.  The
    richer pulse-train diagnostics are available from
    :func:`detect_pulse_train_epochs` and are used by behavior preprocessing.
    """
    signal, time = _get_di3_and_time(harp_df, acq_time)
    if str(epoch_detection_method).lower() == "pulse_train":
        epochs, diag = detect_pulse_train_epochs(
            signal,
            time,
            gap_factor=pulse_gap_factor,
            min_gap_s=pulse_min_gap_s,
            min_duration=max(float(min_duration), float(pulse_min_duration if pulse_min_duration is not None else min_duration)),
            min_pulses=pulse_min_pulses,
        )
        return [list(e) for e in epochs], float(diag.get("gap_threshold_s", np.nan))

    gap = gap_start
    epochs = detect_imaging_epochs(signal, time, gap_threshold=gap, min_duration=min_duration, mode=acq_type)
    if acq_type == "trial" and target_min is not None:
        while len(epochs) < target_min:
            gap *= 1.5
            epochs = detect_imaging_epochs(signal, time, gap_threshold=gap, min_duration=min_duration, mode="trial")
            if gap > 5.0:
                break
    return [list(e) for e in epochs], float(gap)


def shift_epochs_to_photodiode_time(
    epochs: List[List[float]],
    harp_df: pd.DataFrame,
    photodiode_df: pd.DataFrame,
) -> List[List[float]]:
    """Shift detected HARP-relative epochs into photodiode-relative seconds.

    ``correct_event_log`` detects HARP photodiode edges from
    ``photodiode_df.index - photodiode_df.index[0]``.  The epoch CSV must use the
    same relative coordinate system.  Therefore, after epoch detection on
    zero-referenced HARP digital-input time, add the small offset between the
    HARP digital input start and the photodiode analog input start.
    """
    if len(epochs) == 0:
        return []

    if "time" in harp_df.columns:
        harp_start = float(harp_df["time"].iloc[0])
    else:
        harp_start = float(harp_df.index[0])
    pd_start = float(photodiode_df.index[0])
    offset = harp_start - pd_start

    out = [list(e) for e in epochs]
    for e in out:
        e[2] = float(e[2]) + offset
        e[3] = float(e[3]) + offset
    return out


def epochs_to_dataframe(epochs: List[List[float]]) -> pd.DataFrame:
    """Convert detected epoch tuples into a standard DataFrame."""
    epoch_df = pd.DataFrame(epochs, columns=["start_idx", "end_idx", "start_time", "end_time"])
    if len(epoch_df):
        epoch_df.insert(0, "epoch_index", np.arange(1, len(epoch_df) + 1, dtype=int))
        epoch_df["duration_s"] = epoch_df["end_time"] - epoch_df["start_time"]
    else:
        epoch_df["epoch_index"] = []
        epoch_df["duration_s"] = []
    return epoch_df


def summarize_epochs(epoch_df: pd.DataFrame, *, mode: str, gap_threshold_used: float, detection_method: str = "unknown") -> dict:
    """Build a compact QC summary for detected imaging epochs."""
    return {
        "mode": mode,
        "detection_method": str(detection_method),
        "gap_threshold_used": float(gap_threshold_used) if np.isfinite(gap_threshold_used) else None,
        "n_epochs": int(len(epoch_df)),
        "durations_s": epoch_df["duration_s"].round(6).tolist() if len(epoch_df) else [],
        "mean_duration_s": float(epoch_df["duration_s"].mean()) if len(epoch_df) else 0.0,
        "timebase": "photodiode_harp_seconds",
        "passed": len(epoch_df) > 0,
        "warnings": [] if len(epoch_df) > 0 else ["No imaging epochs detected."],
    }
