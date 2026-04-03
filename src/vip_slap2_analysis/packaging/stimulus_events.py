from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


DEFAULT_EVENT_TIME_COLUMN = "corrected_timestamps"
DEFAULT_EVENT_VALUE_COLUMN = "Value"
DEFAULT_SPECIAL_EVENTS = ("Change", "Omission")


def load_bonsai_event_log(csv_path: str | Path) -> pd.DataFrame:
    """Load a Bonsai event log CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Bonsai event log not found: {csv_path}")
    return pd.read_csv(csv_path)


def _normalize_value_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _event_times(df: pd.DataFrame, value: str, *, time_col: str, value_col: str) -> list[float]:
    sub = df.loc[df[value_col] == value, time_col]
    out = pd.to_numeric(sub, errors="coerce").dropna().to_numpy(dtype=float)
    return out.tolist()


def extract_stimulus_events_from_bonsai(
    event_log: pd.DataFrame,
    *,
    time_col: str = DEFAULT_EVENT_TIME_COLUMN,
    value_col: str = DEFAULT_EVENT_VALUE_COLUMN,
    image_suffix: str = ".tiff",
    special_events: Iterable[str] = DEFAULT_SPECIAL_EVENTS,
    drop_duplicate_pairs: bool = False,
) -> Dict[str, Any]:
    """
    Extract image/change/omission event times from a Bonsai event log.

    Parameters
    ----------
    event_log:
        DataFrame loaded from ``bonsai_event_log.csv``.
    time_col:
        Column containing post-processed HARP-aligned event times.
        For your data this should be ``corrected_timestamps``.
    value_col:
        Column containing event labels.
    image_suffix:
        Suffix used to identify image presentation rows.
    special_events:
        Event labels to extract in addition to image rows.
    drop_duplicate_pairs:
        Optional deduplication on exact (value, time) pairs.

    Returns
    -------
    dict
        JSON-serializable event structure.
    """
    if time_col not in event_log.columns:
        raise KeyError(
            f"Required event-time column '{time_col}' was not found. "
            f"Available columns: {list(event_log.columns)}"
        )
    if value_col not in event_log.columns:
        raise KeyError(
            f"Required event-value column '{value_col}' was not found. "
            f"Available columns: {list(event_log.columns)}"
        )

    df = event_log.copy()
    df[value_col] = _normalize_value_series(df[value_col])
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.loc[df[time_col].notna()].copy()

    if drop_duplicate_pairs:
        df = df.drop_duplicates(subset=[value_col, time_col], keep="first")

    image_mask = df[value_col].str.endswith(image_suffix, na=False)
    image_df = df.loc[image_mask, [value_col, time_col]].copy()

    ordered_image_values = image_df[value_col].tolist()
    ordered_image_times = image_df[time_col].astype(float).tolist()
    unique_image_values = image_df[value_col].drop_duplicates().tolist()
    image_times_by_value = {
        value: _event_times(image_df, value, time_col=time_col, value_col=value_col)
        for value in unique_image_values
    }

    special_event_times = {
        event_name: _event_times(df, event_name, time_col=time_col, value_col=value_col)
        for event_name in special_events
    }

    return {
        "time_source_column": time_col,
        "value_source_column": value_col,
        "n_rows": int(len(df)),
        "n_image_events": int(len(image_df)),
        "ordered_image_values": ordered_image_values,
        "ordered_image_times_s": ordered_image_times,
        "unique_image_values": unique_image_values,
        "image_times_by_value_s": image_times_by_value,
        "change_times_s": special_event_times.get("Change", []),
        "omission_times_s": special_event_times.get("Omission", []),
        "special_event_times_s": special_event_times,
    }


def extract_stimulus_events(
    bonsai_event_log_csv: str | Path,
    *,
    time_col: str = DEFAULT_EVENT_TIME_COLUMN,
    value_col: str = DEFAULT_EVENT_VALUE_COLUMN,
    image_suffix: str = ".tiff",
    special_events: Iterable[str] = DEFAULT_SPECIAL_EVENTS,
    drop_duplicate_pairs: bool = False,
) -> Dict[str, Any]:
    df = load_bonsai_event_log(bonsai_event_log_csv)
    return extract_stimulus_events_from_bonsai(
        df,
        time_col=time_col,
        value_col=value_col,
        image_suffix=image_suffix,
        special_events=special_events,
        drop_duplicate_pairs=drop_duplicate_pairs,
    )


def write_stimulus_events_json(
    output_path: str | Path,
    events: Mapping[str, Any],
    *,
    indent: int = 2,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(events), f, indent=indent)
    return output_path


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if pd.isna(obj):
        return None
    return obj
