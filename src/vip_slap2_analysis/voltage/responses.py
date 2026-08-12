"""Lightweight readers and indices for processed voltage response products."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence
import re

import h5py
import numpy as np
import pandas as pd


def load_response_package(path) -> dict:
    """Load a voltage mean/sequence NPZ package."""
    with np.load(Path(path), allow_pickle=True) as npz:
        return npz["data"][0]


def decode_roi_ids(values) -> np.ndarray:
    return np.asarray(
        [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in np.asarray(values).reshape(-1)]
    )


def source_roi_axis(roi_ids, dmd: int, source_roi: int) -> int:
    """Map a source/full-session ROI number to the QC-filtered response axis."""
    ids = decode_roi_ids(roi_ids)
    label = f"DMD{int(dmd)}_ROI{int(source_roi)}"
    hits = np.flatnonzero(ids == label)
    if not len(hits):
        parsed = []
        for value in ids:
            match = re.search(r"roi[_-]?(\d+)$", str(value), flags=re.IGNORECASE)
            parsed.append(int(match.group(1)) if match else -1)
        hits = np.flatnonzero(np.asarray(parsed) == int(source_roi))
    if not len(hits):
        raise KeyError(f"{label} is absent from the kept response ROI axis")
    return int(hits[0])


def _coerce_timebase(data: np.ndarray, timebase: np.ndarray):
    """Trim the known one-sample summary/timebase mismatch without altering data."""
    data = np.asarray(data)
    timebase = np.asarray(timebase, dtype=float).reshape(-1)
    n = min(data.shape[-1], len(timebase))
    return data[..., :n], timebase[:n]


def get_mean_response(
    package: dict,
    *,
    dmd: int,
    source_roi: int,
    event_type: str = "image",
    image_name: Optional[str] = None,
):
    """Return one mean response trace and its event-relative timebase."""
    dmd_pkg = package[f"DMD{int(dmd)}"]
    axis = source_roi_axis(dmd_pkg["roi_ids"], dmd, source_roi)
    if event_type == "image":
        if image_name is None:
            raise ValueError("image_name is required when event_type='image'")
        summary = dmd_pkg["image_identity"][image_name]
        time_key = "image"
    elif event_type in {"change", "omission"}:
        summary = dmd_pkg[event_type]
        time_key = event_type
    else:
        raise ValueError("event_type must be 'image', 'change', or 'omission'")
    trace = np.asarray(summary["mean"])[axis]
    trace, t = _coerce_timebase(trace, package["timebase_sec"][time_key])
    return t, trace


def get_sequence_response(
    package: dict,
    *,
    dmd: int,
    source_roi: int,
    image_name: str,
    phase: str = "repeated",
) -> dict:
    """Return sequence-position mean responses for one ROI/image."""
    dmd_pkg = package[f"DMD{int(dmd)}"]
    axis = source_roi_axis(dmd_pkg["roi_ids"], dmd, source_roi)
    summary = dmd_pkg["image_identity"][image_name][phase]
    mean = np.asarray(summary["mean"])
    if mean.ndim == 3:
        mean = mean[:, axis, :]
    elif mean.ndim == 2:
        mean = mean[axis][None, :]
    mean, t = _coerce_timebase(mean, package["timebase_sec"]["image"])
    positions = summary.get("positions", summary.get("position", np.arange(mean.shape[0])))
    return {
        "timebase_sec": t,
        "mean": mean,
        "positions": np.asarray(positions).reshape(-1),
        "counts": np.asarray(summary.get("counts", [summary.get("n_events", np.nan)])).reshape(-1),
        "n_sequences": int(np.asarray(summary.get("n_sequences", 0)).reshape(-1)[0]),
    }


def _image_groups(group: h5py.Group):
    for key in sorted(group.keys()):
        item = group[key]
        if not isinstance(item, h5py.Group):
            continue
        image_name = item.attrs.get("image_name", key)
        if isinstance(image_name, bytes):
            image_name = image_name.decode()
        yield key, str(image_name), item


def load_single_trial_traces(
    path,
    *,
    dmd: int,
    event_type: str,
    source_rois: Optional[Sequence[int]] = None,
    image_name: Optional[str] = None,
    trial_indices: Optional[Sequence[int]] = None,
):
    """Read selected event/ROI slices from a single-trial H5 without loading it all."""
    path = Path(path)
    dmd_key = f"DMD{int(dmd)}"
    with h5py.File(path, "r") as h5:
        group = h5[dmd_key]
        roi_ids = decode_roi_ids(group["roi_ids"][:])
        if source_rois is None:
            roi_axes = np.arange(len(roi_ids), dtype=int)
        else:
            roi_axes = np.asarray(
                [source_roi_axis(roi_ids, dmd, int(roi)) for roi in source_rois], dtype=int
            )

        if event_type == "image":
            if image_name is None:
                raise ValueError("image_name is required when event_type='image'")
            matches = [(k, item) for k, name, item in _image_groups(group["image_identity"]) if name == image_name]
            if not matches:
                raise KeyError(f"Image {image_name!r} not found in {dmd_key}/image_identity")
            event_group = matches[0][1]
            time_key = "image"
        elif event_type in {"change", "omission"}:
            event_group = group[event_type]
            time_key = event_type
        else:
            raise ValueError("event_type must be 'image', 'change', or 'omission'")

        n_trials = int(event_group["traces"].shape[0])
        trials = np.arange(n_trials, dtype=int) if trial_indices is None else np.asarray(trial_indices, dtype=int)
        # h5py advanced indexing is restrictive; read trial-by-trial but only selected ROIs.
        traces = (
            np.stack([np.asarray(event_group["traces"][int(i), :, :])[roi_axes, :] for i in trials], axis=0)
            if len(trials)
            else np.empty((0, len(roi_axes), event_group["traces"].shape[-1]))
        )
        onsets = np.asarray(event_group["onsets_sec"][:], dtype=float)[trials]
        t = np.asarray(h5[f"timebase_sec/{time_key}"][:], dtype=float)
        traces, t = _coerce_timebase(traces, t)

    return {
        "traces": traces,
        "timebase_sec": t,
        "onsets_sec": onsets,
        "roi_ids": roi_ids[roi_axes],
        "trial_indices": trials,
    }


def _match_occurrences(
    trial_rows: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    trial_time_col: str,
    candidate_time_col: str,
    tolerance_sec: float,
) -> pd.DataFrame:
    """One-to-one nearest-time matching that preserves repeated timestamps."""
    result = trial_rows.copy()
    result["event_id"] = np.nan
    result["match_error_sec"] = np.nan
    available = candidates.copy()
    for idx, row in result.sort_values(trial_time_col).iterrows():
        if available.empty:
            break
        delta = np.abs(available[candidate_time_col].to_numpy(float) - float(row[trial_time_col]))
        j = int(np.argmin(delta))
        if delta[j] <= tolerance_sec:
            candidate = available.iloc[j]
            result.loc[idx, "event_id"] = int(candidate["event_id"])
            result.loc[idx, "match_error_sec"] = float(delta[j])
            available = available.drop(available.index[j])
    return result


def build_single_trial_index_for_session(
    path,
    events: pd.DataFrame,
    *,
    session_id: Optional[str] = None,
    match_tolerance_sec: float = 1e-3,
) -> pd.DataFrame:
    """Index every trial stored in a single-trial H5 and map it onto canonical events."""
    path = Path(path)
    rows = []
    with h5py.File(path, "r") as h5:
        for dmd_key in sorted(k for k in h5 if str(k).startswith("DMD")):
            dmd = int(str(dmd_key).replace("DMD", ""))
            group = h5[dmd_key]
            for key, image_name, item in _image_groups(group["image_identity"]):
                for trial_index, onset in enumerate(np.asarray(item["onsets_sec"][:], dtype=float)):
                    rows.append(
                        {
                            "session_id": session_id,
                            "dmd": dmd,
                            "event_type": "image",
                            "image_name": image_name,
                            "trial_index": int(trial_index),
                            "onset_sec": float(onset),
                            "dataset_path": f"/{dmd_key}/image_identity/{key}/traces",
                        }
                    )
            for event_type in ("change", "omission"):
                item = group[event_type]
                for trial_index, onset in enumerate(np.asarray(item["onsets_sec"][:], dtype=float)):
                    rows.append(
                        {
                            "session_id": session_id,
                            "dmd": dmd,
                            "event_type": event_type,
                            "image_name": "",
                            "trial_index": int(trial_index),
                            "onset_sec": float(onset),
                            "dataset_path": f"/{dmd_key}/{event_type}/traces",
                        }
                    )

    index = pd.DataFrame(rows)
    matched = []
    for (dmd, event_type, image_name), trials in index.groupby(
        ["dmd", "event_type", "image_name"], sort=False, dropna=False
    ):
        if event_type == "image":
            candidates = events[events["image_name"].eq(image_name)].copy()
            time_col = "onset_sec"
        elif event_type == "change":
            candidates = events[events["is_change"]].copy()
            time_col = "change_onset_sec"
        else:
            candidates = events[events["is_omission"]].copy()
            time_col = "omission_onset_sec"
        candidates = candidates[np.isfinite(candidates[time_col])]
        matched.append(
            _match_occurrences(
                trials,
                candidates,
                trial_time_col="onset_sec",
                candidate_time_col=time_col,
                tolerance_sec=float(match_tolerance_sec),
            )
        )

    out = pd.concat(matched, ignore_index=True) if matched else index
    out["matched"] = out["event_id"].notna()
    if len(out) and not out["matched"].all():
        n_bad = int((~out["matched"]).sum())
        raise ValueError(f"{path}: {n_bad} single-trial rows could not be matched to canonical events")
    out["event_id"] = out["event_id"].astype(int) if len(out) else out.get("event_id")
    return out.sort_values(["dmd", "event_type", "image_name", "trial_index"]).reset_index(drop=True)


def build_single_trial_index(sessions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build a cross-session single-trial index without loading event traces."""
    tables = []
    for session in sessions.itertuples(index=False):
        event_table = events[events["session_id"].astype(str).eq(str(session.session_id))]
        table = build_single_trial_index_for_session(
            session.single_trial_h5,
            event_table,
            session_id=str(session.session_id),
        )
        meta_cols = ["event_id", "subject_id", "session_label", "session_order"]
        if "event_uid" in event_table:
            meta_cols.append("event_uid")
        meta = event_table[meta_cols].drop_duplicates("event_id")
        table = table.merge(meta, on="event_id", how="left")
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
