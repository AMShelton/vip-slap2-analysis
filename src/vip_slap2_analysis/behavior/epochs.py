"""Imaging-epoch detection utilities for behavior-aligned physiology sessions.

This module detects valid SLAP2 imaging intervals from the HARP DI3 acquisition
line.  The preferred detector treats DI3 as a pulse train and splits epochs at
large inter-pulse gaps.  This is robust to brief laser-on/realignment artifacts
that can make DI3 momentarily active without representing a real imaging epoch.

The public epoch representation is shared across glutamate, calcium, and voltage
workflows: ``(start_idx, end_idx, start_time, end_time)`` in the supplied HARP
sample/time coordinate system.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label


EpochList = List[Tuple[int, int, float, float]]


def _as_bool_1d(x: Sequence[object]) -> np.ndarray:
    """Return a one-dimensional boolean array."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(bool)


def _as_time_1d(time: Sequence[float]) -> np.ndarray:
    """Return a one-dimensional finite float time vector."""
    t = np.asarray(time, dtype=float)
    if t.ndim != 1:
        t = t.reshape(-1)
    if t.size == 0:
        raise ValueError("time must contain at least one sample")
    if np.any(~np.isfinite(t)):
        raise ValueError("time contains non-finite values")
    return t


def _rising_edges(signal: np.ndarray) -> np.ndarray:
    """Return indices where a boolean signal transitions from low to high."""
    if signal.size == 0:
        return np.array([], dtype=int)
    prev = np.concatenate([[False], signal[:-1]])
    return np.flatnonzero(signal & ~prev)


def detect_imaging_epochs(signal, time, gap_threshold=0.05, min_duration=5.0, mode="trial"):
    """Detect imaging intervals from a binary acquisition signal.

    This is the legacy level/gap detector.  It is retained for compatibility with
    earlier notebooks and with non-pulse digital lines.  For SLAP2 line/cycle
    clocks, prefer :func:`detect_pulse_train_epochs`, which splits continuous
    acquisition at large inter-pulse gaps and rejects isolated realignment pulses.
    """
    if mode not in ["trial", "continuous"]:
        raise ValueError("mode must be 'trial' or 'continuous'")

    signal = _as_bool_1d(signal)
    time = _as_time_1d(time)
    if signal.size != time.size:
        raise ValueError("signal and time must have the same length")

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
        gap = time[next_start] - time[current_end]
        if gap > gap_threshold:
            start_time = time[current_start]
            end_time = time[current_end - 1]
            if (end_time - start_time) >= min_duration:
                epochs.append((current_start, current_end, start_time, end_time))
            current_start, current_end = next_start, next_end
        else:
            current_end = next_end

    final_duration = time[current_end - 1] - time[current_start]
    if final_duration >= min_duration:
        epochs.append((current_start, current_end, time[current_start], time[current_end - 1]))

    return epochs


def detect_pulse_train_epochs(
    signal,
    time,
    *,
    min_duration: float = 6.0,
    min_pulses: int = 10,
    gap_factor: float = 10.0,
    min_gap_s: float = 0.5,
    max_gap_s: Optional[float] = None,
    nominal_period_q: float = 50.0,
    return_diagnostics: bool = False,
) -> Tuple[EpochList, Dict[str, object]] | EpochList:
    """Detect imaging epochs from gaps in the DI3 pulse train.

    Parameters
    ----------
    signal, time
        Boolean DI3 samples and matching time vector.  ``time`` may be absolute
        HARP seconds or zero-referenced acquisition seconds.
    min_duration
        Minimum accepted epoch duration in seconds.
    min_pulses
        Minimum number of rising edges required for an epoch.  This rejects short
        laser-on/realignment artifacts that do not represent real acquisition.
    gap_factor
        Multiplier on the nominal inter-pulse period used to define an epoch gap.
    min_gap_s
        Absolute lower bound on the gap threshold.  This prevents over-splitting
        when the pulse train has mild jitter.
    max_gap_s
        Optional absolute upper bound on the gap threshold.
    nominal_period_q
        Percentile of inter-pulse intervals used to estimate the nominal period.
        Values near 50 are robust when only a small number of large gaps exist.
    return_diagnostics
        If True, return ``(epochs, diagnostics)``.  Otherwise return epochs only.

    Notes
    -----
    Epoch boundaries are defined from the first to last rising edge in each
    accepted pulse-train segment, with the end extended by one nominal period.
    This makes the output suitable for event filtering without pretending that
    samples exist during acquisition pauses.
    """
    signal = _as_bool_1d(signal)
    time = _as_time_1d(time)
    if signal.size != time.size:
        raise ValueError("signal and time must have the same length")

    rise_idx = _rising_edges(signal)
    rise_t = time[rise_idx]

    diagnostics: Dict[str, object] = {
        "method": "pulse_train_gaps",
        "n_rising_edges": int(rise_idx.size),
        "min_duration_s": float(min_duration),
        "min_pulses": int(min_pulses),
        "gap_factor": float(gap_factor),
        "min_gap_s": float(min_gap_s),
        "max_gap_s": None if max_gap_s is None else float(max_gap_s),
        "nominal_period_s": np.nan,
        "gap_threshold_s": np.nan,
        "n_large_gaps": 0,
        "large_gaps_s": [],
        "large_gap_times_s": [],
        "rejected_segments": [],
    }

    if rise_idx.size == 0:
        return ([], diagnostics) if return_diagnostics else []

    if rise_idx.size == 1:
        return ([], diagnostics) if return_diagnostics else []

    dt = np.diff(rise_t)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if finite_dt.size == 0:
        return ([], diagnostics) if return_diagnostics else []

    nominal_period = float(np.percentile(finite_dt, nominal_period_q))
    gap_threshold = max(float(min_gap_s), float(gap_factor) * nominal_period)
    if max_gap_s is not None:
        gap_threshold = min(gap_threshold, float(max_gap_s))

    large_gap_mask = dt > gap_threshold
    break_after = np.flatnonzero(large_gap_mask)

    diagnostics["nominal_period_s"] = nominal_period
    diagnostics["gap_threshold_s"] = float(gap_threshold)
    diagnostics["n_large_gaps"] = int(large_gap_mask.sum())
    diagnostics["large_gaps_s"] = [float(x) for x in dt[large_gap_mask]]
    diagnostics["large_gap_times_s"] = [float(x) for x in rise_t[break_after]]

    # segments are inclusive ranges in rising-edge index space
    starts = np.concatenate([[0], break_after + 1])
    stops = np.concatenate([break_after, [rise_idx.size - 1]])

    epochs: EpochList = []
    rejected = []
    for seg_i, (s, e) in enumerate(zip(starts, stops), start=1):
        n_pulses = int(e - s + 1)
        start_idx = int(rise_idx[s])
        last_rise_idx = int(rise_idx[e])
        start_time = float(rise_t[s])
        end_time = float(rise_t[e] + nominal_period)
        duration = float(end_time - start_time)

        # end_idx is the first row strictly after the estimated epoch end.
        end_idx = int(np.searchsorted(time, end_time, side="right"))
        end_idx = max(end_idx, last_rise_idx + 1)
        end_idx = min(end_idx, time.size)

        if duration >= float(min_duration) and n_pulses >= int(min_pulses):
            epochs.append((start_idx, end_idx, start_time, end_time))
        else:
            rejected.append(
                {
                    "segment": int(seg_i),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_s": duration,
                    "n_pulses": n_pulses,
                    "reason": "too_short_or_too_few_pulses",
                }
            )

    diagnostics["rejected_segments"] = rejected
    diagnostics["n_epochs"] = int(len(epochs))
    diagnostics["epoch_durations_s"] = [float(e[3] - e[2]) for e in epochs]

    return (epochs, diagnostics) if return_diagnostics else epochs


def detect_epochs_adaptive(
    harp_df: pd.DataFrame,
    acq_time: np.ndarray,
    acq_type: str,
    min_duration: float = 6.0,
    gap_start: float = 0.02,
    target_min: Optional[int] = None,
    *,
    method: str = "pulse_train",
    di_col: str = "DI3",
    pulse_min_pulses: int = 10,
    pulse_gap_factor: float = 10.0,
    pulse_min_gap_s: float = 0.5,
    pulse_max_gap_s: Optional[float] = None,
    return_diagnostics: bool = False,
) -> Tuple[List[List[float]], float] | Tuple[List[List[float]], float, Dict[str, object]]:
    """Detect imaging epochs, preferring pulse-train gap detection for DI3.

    ``method='pulse_train'`` is recommended for SLAP2 QC across modalities.  It
    splits epochs at large gaps in the DI3 rising-edge train and works for both
    trial-wise and continuous acquisition modes.  ``method='level'`` preserves the
    legacy active-region detector.
    """
    if di_col not in harp_df.columns:
        raise KeyError(f"{di_col!r} not found in HARP dataframe")

    method = str(method).lower()
    diagnostics: Dict[str, object] = {"method": method}

    if method in ("pulse", "pulse_train", "pulse_train_gaps"):
        epochs, diagnostics = detect_pulse_train_epochs(
            harp_df[di_col].to_numpy(),
            acq_time,
            min_duration=min_duration,
            min_pulses=pulse_min_pulses,
            gap_factor=pulse_gap_factor,
            min_gap_s=pulse_min_gap_s,
            max_gap_s=pulse_max_gap_s,
            return_diagnostics=True,
        )
        gap_used = float(diagnostics.get("gap_threshold_s", np.nan))

    elif method in ("level", "legacy"):
        gap = gap_start
        epochs = detect_imaging_epochs(
            harp_df[di_col].to_numpy(),
            acq_time,
            gap_threshold=gap,
            min_duration=min_duration,
            mode=acq_type,
        )

        if acq_type == "trial" and target_min:
            while len(epochs) < target_min:
                gap += 0.002
                epochs = detect_imaging_epochs(
                    harp_df[di_col].to_numpy(),
                    acq_time,
                    gap_threshold=gap,
                    min_duration=min_duration,
                    mode=acq_type,
                )
        gap_used = float(gap)
        diagnostics = {
            "method": "legacy_level_gap",
            "gap_threshold_s": gap_used,
            "n_epochs": int(len(epochs)),
        }
    else:
        raise ValueError("method must be 'pulse_train' or 'level'")

    out_epochs = [list(e) for e in epochs]
    if return_diagnostics:
        return out_epochs, gap_used, diagnostics
    return out_epochs, gap_used


def shift_epochs_to_photodiode_time(
    epochs: List[List[float]],
    harp_df: pd.DataFrame,
    photodiode_df: pd.DataFrame,
) -> List[List[float]]:
    """Shift detected HARP epochs into the photodiode time coordinate system.

    The shift is computed from the offset between the HARP digital-input time
    column and the photodiode DataFrame index.
    """
    t_shift = float(harp_df["time"].iloc[0]) - float(photodiode_df.index[0])
    out = [e.copy() for e in epochs]
    for e in out:
        e[2] = float(e[2] + t_shift)
        e[3] = float(e[3] + t_shift)
    return out


def epochs_to_dataframe(epochs: List[List[float]]) -> pd.DataFrame:
    """Convert detected epoch tuples into a standard DataFrame."""
    epoch_df = pd.DataFrame(
        epochs,
        columns=["start_idx", "end_idx", "start_time", "end_time"],
    )
    if len(epoch_df):
        epoch_df["epoch_idx"] = np.arange(1, len(epoch_df) + 1, dtype=int)
        epoch_df["duration_s"] = epoch_df["end_time"] - epoch_df["start_time"]
    else:
        epoch_df["epoch_idx"] = []
        epoch_df["duration_s"] = []
    return epoch_df


def summarize_epochs(
    epoch_df: pd.DataFrame,
    *,
    mode: str,
    gap_threshold_used: float,
    detection_diagnostics: Optional[Dict[str, object]] = None,
) -> dict:
    """Build a compact QC summary for detected imaging epochs."""
    warnings: List[str] = []
    if len(epoch_df) == 0:
        warnings.append("No imaging epochs detected.")
    if len(epoch_df) > 1:
        warnings.append(
            "Multiple imaging epochs detected; downstream extraction/alignment should be epoch-aware."
        )

    summary = {
        "mode": mode,
        "gap_threshold_used": float(gap_threshold_used) if np.isfinite(gap_threshold_used) else None,
        "n_epochs": int(len(epoch_df)),
        "durations_s": epoch_df["duration_s"].round(6).tolist() if len(epoch_df) else [],
        "mean_duration_s": float(epoch_df["duration_s"].mean()) if len(epoch_df) else 0.0,
        "timebase": "photodiode_harp_seconds",
        "passed": len(epoch_df) > 0,
        "warnings": warnings,
    }
    if detection_diagnostics is not None:
        summary["detection"] = detection_diagnostics
    return summary
