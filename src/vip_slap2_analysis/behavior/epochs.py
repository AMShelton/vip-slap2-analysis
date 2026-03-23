from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label


def detect_imaging_epochs(signal, time, gap_threshold=0.05, min_duration=5.0, mode="trial"):
    if mode not in ["trial", "continuous"]:
        raise ValueError("mode must be 'trial' or 'continuous'")

    signal = np.asarray(signal).astype(bool)
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


def detect_epochs_adaptive(
    harp_df: pd.DataFrame,
    acq_time: np.ndarray,
    acq_type: str,
    min_duration: float = 6.0,
    gap_start: float = 0.02,
    target_min: Optional[int] = None,
) -> Tuple[List[List[float]], float]:
    gap = gap_start
    epochs = detect_imaging_epochs(
        harp_df["DI3"].to_numpy(),
        acq_time,
        gap_threshold=gap,
        min_duration=min_duration,
        mode=acq_type,
    )

    if acq_type == "trial" and target_min:
        while len(epochs) < target_min:
            gap += 0.002
            epochs = detect_imaging_epochs(
                harp_df["DI3"].to_numpy(),
                acq_time,
                gap_threshold=gap,
                min_duration=min_duration,
                mode=acq_type,
            )

    return [list(e) for e in epochs], float(gap)


def shift_epochs_to_photodiode_time(
    epochs: List[List[float]],
    harp_df: pd.DataFrame,
    photodiode_df: pd.DataFrame,
) -> List[List[float]]:
    t_shift = float(harp_df["time"].iloc[0]) - float(photodiode_df.index[0])
    out = [e.copy() for e in epochs]
    for e in out:
        e[2] = float(e[2] + t_shift)
        e[3] = float(e[3] + t_shift)
    return out


def epochs_to_dataframe(epochs: List[List[float]]) -> pd.DataFrame:
    epoch_df = pd.DataFrame(
        epochs,
        columns=["start_idx", "end_idx", "start_time", "end_time"],
    )
    if len(epoch_df):
        epoch_df["duration_s"] = epoch_df["end_time"] - epoch_df["start_time"]
    else:
        epoch_df["duration_s"] = []
    return epoch_df


def summarize_epochs(epoch_df: pd.DataFrame, *, mode: str, gap_threshold_used: float) -> dict:
    return {
        "mode": mode,
        "gap_threshold_used": float(gap_threshold_used),
        "n_epochs": int(len(epoch_df)),
        "durations_s": epoch_df["duration_s"].round(6).tolist() if len(epoch_df) else [],
        "mean_duration_s": float(epoch_df["duration_s"].mean()) if len(epoch_df) else 0.0,
        "timebase": "photodiode_harp_seconds",
        "passed": len(epoch_df) > 0,
        "warnings": [] if len(epoch_df) > 0 else ["No imaging epochs detected."],
    }