from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def validate_bonsai_event_log(stim_df: pd.DataFrame) -> Dict[str, Any]:
    required = {"Frame", "Timestamp", "Value"}
    cols = set(stim_df.columns)
    missing = sorted(required - cols)

    values = stim_df["Value"].astype(str).fillna("") if "Value" in stim_df.columns else pd.Series([], dtype=str)
    unique_stim_names = sorted({v for v in values.unique() if v.lower().endswith((".tif", ".tiff"))})

    has_corrected = "corrected_timestamps" in stim_df.columns
    has_bv_photodiode = values.str.contains("photodiode", case=False, na=False).any()

    warnings = []
    passed = True

    if missing:
        warnings.append(f"Missing required Bonsai columns: {missing}")
        passed = False

    if len(unique_stim_names) <= 1:
        warnings.append("Found <=1 unique .tif/.tiff stimulus names; possible overwritten or malformed Bonsai log.")
        passed = False

    return {
        "has_required_columns": len(missing) == 0,
        "missing_columns": missing,
        "n_rows": int(len(stim_df)),
        "n_unique_tiff_values": int(len(unique_stim_names)),
        "stimulus_names": unique_stim_names,
        "has_corrected_timestamps": has_corrected,
        "has_bv_photodiode_rows": bool(has_bv_photodiode),
        "passed": passed,
        "warnings": warnings,
    }


def validate_harp_data(harp_df: pd.DataFrame, photodiode_df: pd.DataFrame) -> Dict[str, Any]:
    warnings = []
    passed = True

    has_di3 = "DI3" in harp_df.columns
    time_monotonic = bool(np.all(np.diff(harp_df["time"].to_numpy(dtype=float)) >= 0))

    if not has_di3:
        warnings.append("HARP DigitalInputState does not contain DI3.")
        passed = False

    if not time_monotonic:
        warnings.append("HARP time column is not monotonic.")
        passed = False

    pd_col = photodiode_df.columns[0]
    pd_vals = photodiode_df[pd_col].to_numpy(dtype=float)
    dynamic_range = float(np.nanmax(pd_vals) - np.nanmin(pd_vals)) if len(pd_vals) else 0.0

    if dynamic_range <= 0:
        warnings.append("Photodiode trace has zero dynamic range.")
        passed = False

    return {
        "has_di3": has_di3,
        "n_samples": int(len(harp_df)),
        "time_monotonic": time_monotonic,
        "photodiode_column": str(pd_col),
        "photodiode_dynamic_range": dynamic_range,
        "passed": passed,
        "warnings": warnings,
    }


def get_event_time_column(stim_df: pd.DataFrame) -> str:
    if "corrected_timestamps" in stim_df.columns:
        return "corrected_timestamps"
    if "corrected_timestamp" in stim_df.columns:
        return "corrected_timestamp"
    return "Timestamp"


def extract_event_times(stim_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    tcol = get_event_time_column(stim_df)
    vals = stim_df["Value"].astype(str)

    image_mask = vals.str.lower().str.endswith((".tif", ".tiff"))
    change_mask = vals.str.contains("ChangeFlash", case=False, na=False)
    omission_mask = vals.str.contains("Omission", case=False, na=False)

    return {
        "image_identity": stim_df.loc[image_mask, tcol].to_numpy(dtype=float),
        "change": stim_df.loc[change_mask, tcol].to_numpy(dtype=float),
        "omission": stim_df.loc[omission_mask, tcol].to_numpy(dtype=float),
    }


def count_events_in_epochs(event_times: np.ndarray, epoch_df: pd.DataFrame) -> int:
    if len(event_times) == 0 or len(epoch_df) == 0:
        return 0

    count = 0
    for t in event_times:
        in_any = ((epoch_df["start_time"] <= t) & (t <= epoch_df["end_time"])).any()
        count += int(in_any)
    return int(count)


def audit_event_coverage(stim_df: pd.DataFrame, epoch_df: pd.DataFrame) -> Dict[str, Any]:
    ev = extract_event_times(stim_df)
    out = {}
    for k, times in ev.items():
        out[f"{k}_total"] = int(len(times))
        out[f"{k}_in_epochs"] = count_events_in_epochs(times, epoch_df)
    return out