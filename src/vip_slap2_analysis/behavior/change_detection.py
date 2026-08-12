"""Canonical event tables for the passive Detection-of-Change task."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from vip_slap2_analysis.common.alignment import EventWindows, load_corrected_bonsai_csv


def _time_col(df: pd.DataFrame) -> str:
    for column in ("corrected_timestamp", "corrected_timestamps", "HARP_timestamps"):
        if column in df.columns:
            return column
    raise ValueError("No corrected/HARP timestamp column found in Bonsai event log")


def _is_image(value: str) -> bool:
    value = str(value).lower()
    return value.endswith((".tif", ".tiff")) and "photodiode" not in value


def _is_change(value: str) -> bool:
    value = str(value).lower()
    return value == "change" or "changeflash" in value


def _is_omission(value: str) -> bool:
    return "omission" in str(value).lower()


def _image_label(value: str) -> str:
    return Path(str(value).replace("\\", "/")).stem


def _load_epochs(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["epoch_index", "start_time", "end_time"])
    epochs = pd.read_csv(path).copy()
    required = {"start_time", "end_time"}
    if not required.issubset(epochs.columns):
        raise ValueError(f"{path} lacks required epoch columns {sorted(required)}")
    if "epoch_index" not in epochs:
        epochs["epoch_index"] = np.arange(1, len(epochs) + 1)
    return epochs.sort_values("start_time").reset_index(drop=True)


def _epoch_for_time(t: float, epochs: pd.DataFrame) -> float:
    if not np.isfinite(t) or epochs.empty:
        return np.nan
    hit = epochs[(epochs["start_time"] <= t) & (t <= epochs["end_time"])]
    return float(hit.iloc[0]["epoch_index"]) if len(hit) else np.nan


def _window_retained(t: float, epochs: pd.DataFrame, window: Tuple[float, float]) -> bool:
    if not np.isfinite(t) or epochs.empty:
        return False
    pre, post = map(float, window)
    return bool(
        ((epochs["start_time"] + pre <= t) & (t <= epochs["end_time"] - post)).any()
    )


def _marker_targets(
    source_rows: np.ndarray,
    image_source_rows: np.ndarray,
) -> dict:
    """Map each marker row to the first subsequent image row, matching extraction."""
    out = {}
    for marker_row in source_rows:
        pos = np.searchsorted(image_source_rows, int(marker_row), side="right")
        if pos < len(image_source_rows):
            out[int(image_source_rows[pos])] = int(marker_row)
    return out


def build_change_detection_event_table(
    event_csv,
    *,
    imaging_epochs_csv=None,
    windows: Optional[EventWindows] = None,
    omission_pair_tolerance_sec: float = 0.05,
) -> pd.DataFrame:
    """Build one row per expected image cycle from a corrected Bonsai log.

    Change and omission marker timestamps are preserved separately from the image
    row timestamp. This is important because a ChangeFlash marker can precede the
    changed-to image by one display frame, while omission rows encode the expected
    image identity for a cycle in which the image itself was not displayed.
    """
    event_csv = Path(event_csv)
    df = load_corrected_bonsai_csv(event_csv).copy()
    tcol = _time_col(df)
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df["Value"] = df["Value"].fillna("").astype(str).str.strip()
    df = df[df[tcol].notna()].reset_index(names="source_row")

    image_mask = df["Value"].map(_is_image)
    change_mask = df["Value"].map(_is_change)
    omission_mask = df["Value"].map(_is_omission)

    images = df.loc[image_mask, ["source_row", tcol, "Value"]].copy()
    images = images.rename(columns={tcol: "onset_sec", "Value": "image_name"})
    image_rows = images["source_row"].to_numpy(dtype=int)

    change_targets = _marker_targets(
        df.loc[change_mask, "source_row"].to_numpy(dtype=int), image_rows
    )
    omission_targets = _marker_targets(
        df.loc[omission_mask, "source_row"].to_numpy(dtype=int), image_rows
    )

    marker_time = df.set_index("source_row")[tcol].to_dict()
    rows = []
    for event_id, source in enumerate(images.itertuples(index=False)):
        source_row = int(source.source_row)
        onset = float(source.onset_sec)
        change_source_row = change_targets.get(source_row)
        omission_source_row = omission_targets.get(source_row)
        change_onset = (
            float(marker_time[change_source_row]) if change_source_row is not None else np.nan
        )
        omission_onset = (
            float(marker_time[omission_source_row]) if omission_source_row is not None else np.nan
        )
        is_omission = bool(
            omission_source_row is not None
            and abs(onset - omission_onset) <= float(omission_pair_tolerance_sec)
        )

        rows.append(
            {
                "event_id": int(event_id),
                "source_row": source_row,
                "onset_sec": onset,
                "image_name": str(source.image_name),
                "image_label": _image_label(source.image_name),
                "is_change": change_source_row is not None,
                "change_onset_sec": change_onset,
                "is_omission": is_omission,
                "omission_onset_sec": omission_onset if is_omission else np.nan,
                "presented": not is_omission,
            }
        )

    events = pd.DataFrame(rows)
    if events.empty:
        return events

    sequence_id = -1
    expected_position = -1
    presented_position = -1
    sequence_ids = []
    expected_positions = []
    presented_positions = []
    for row in events.itertuples(index=False):
        if row.is_change:
            sequence_id += 1
            expected_position = 0
            presented_position = 0 if row.presented else -1
        else:
            expected_position += 1
            if row.presented:
                presented_position += 1
        sequence_ids.append(sequence_id)
        expected_positions.append(expected_position)
        presented_positions.append(presented_position if row.presented else np.nan)

    events["sequence_id"] = sequence_ids
    events["sequence_position_expected"] = expected_positions
    events["sequence_position_presented"] = presented_positions
    events["previous_event_id"] = events["event_id"].shift(1)
    events["next_event_id"] = events["event_id"].shift(-1)
    events["previous_image"] = events["image_name"].shift(1)
    events["next_image"] = events["image_name"].shift(-1)

    presented_ids = events["event_id"].where(events["presented"])
    events["previous_presented_event_id"] = presented_ids.ffill().shift(1)
    events["next_presented_event_id"] = presented_ids.bfill().shift(-1)

    event_to_image = events.set_index("event_id")["image_name"]
    events["previous_presented_image"] = events["previous_presented_event_id"].map(event_to_image)
    events["next_presented_image"] = events["next_presented_event_id"].map(event_to_image)
    events["session_time_min"] = (events["onset_sec"] - events["onset_sec"].min()) / 60.0
    events["time_block"] = np.floor(events["session_time_min"]).astype(int)

    epochs = _load_epochs(Path(imaging_epochs_csv) if imaging_epochs_csv else None)
    w = windows or EventWindows()
    events["imaging_epoch"] = events["onset_sec"].map(lambda t: _epoch_for_time(t, epochs))
    events["retained_image_window"] = events["onset_sec"].map(
        lambda t: _window_retained(t, epochs, w.image)
    )
    events["retained_change_window"] = events["change_onset_sec"].map(
        lambda t: _window_retained(t, epochs, w.change)
    )
    events["retained_omission_window"] = events["omission_onset_sec"].map(
        lambda t: _window_retained(t, epochs, w.omission)
    )
    return events


def build_change_detection_events(sessions: pd.DataFrame) -> pd.DataFrame:
    """Build and concatenate canonical DoC event tables for a session registry."""
    tables = []
    for session in sessions.itertuples(index=False):
        table = build_change_detection_event_table(
            session.event_csv,
            imaging_epochs_csv=getattr(session, "imaging_epochs_csv", None),
        )
        table.insert(0, "session_id", str(session.session_id))
        table.insert(0, "subject_id", str(session.subject_id))
        table.insert(2, "session_label", session.session_label)
        table.insert(3, "session_order", int(session.session_order))
        table.insert(5, "event_uid", table["session_id"].astype(str) + ":" + table["event_id"].astype(str))
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
