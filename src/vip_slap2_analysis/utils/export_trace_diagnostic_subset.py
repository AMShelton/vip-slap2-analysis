#!/usr/bin/env python3
"""Create a compact diagnostic extract from a derived ASAP8 session-trace HDF5.

Expected input layout (produced by vip-slap2-analysis):
    DMD1/timebase_sec
    DMD1/raw_f, DMD1/f0, DMD1/dff      # ROI x time
    DMD1/roi_ids, DMD1/valid_rois_mask
    DMD2/...                            # optional

The output contains:
  1. Full-rate event-aligned windows for a stratified subset of image/change/
     omission events.
  2. Full-rate continuous snippets from the start, middle, and end of session.
  3. Full-session block summaries at a low rate for bleaching/nonstationarity.
  4. Timing, ROI, and reconstruction metadata needed to audit alignment.

This avoids uploading the full ~1 GB session file while preserving the pieces
needed to test event jitter, photobleaching, and timebase alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

EVENT_WINDOWS = {
    "image": (0.25, 0.50),
    "change": (1.00, 0.75),
    "omission": (1.00, 1.50),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-h5", required=True, type=Path,
                   help="Derived voltage_session_traces_*.h5 file")
    p.add_argument("--event-log", required=True, type=Path,
                   help="Corrected bonsai_event_log.csv")
    p.add_argument("--output", required=True, type=Path,
                   help="Output compact .h5 file")
    p.add_argument("--rois-per-dmd", type=int, default=2,
                   help="Number of valid ROIs to export per DMD (default: 2)")
    p.add_argument("--image-events", type=int, default=128,
                   help="Stratified image presentations to export")
    p.add_argument("--change-events", type=int, default=64,
                   help="Stratified change presentations to export")
    p.add_argument("--omission-events", type=int, default=64,
                   help="Maximum omission presentations to export")
    p.add_argument("--continuous-seconds", type=float, default=15.0,
                   help="Length of start/middle/end continuous snippets")
    p.add_argument("--summary-rate-hz", type=float, default=20.0,
                   help="Target rate for full-session block summaries")
    p.add_argument("--gzip-level", type=int, default=4)
    return p.parse_args()


def decode_strings(values: np.ndarray) -> List[str]:
    out: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def corrected_time_column(df: pd.DataFrame) -> str:
    for key in ("corrected_timestamps", "corrected_timestamp"):
        if key in df.columns:
            return key
    raise ValueError("Event log lacks corrected_timestamps/corrected_timestamp")


def parse_events(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tcol = corrected_time_column(df)
    values = df["Value"].fillna("").astype(str)
    lower = values.str.lower()
    masks = {
        "image": lower.str.endswith((".tif", ".tiff")),
        "change": lower.eq("changeflash") | lower.eq("change"),
        "omission": lower.str.contains("omission", regex=False),
    }
    events: Dict[str, pd.DataFrame] = {}
    for name, mask in masks.items():
        cols = [tcol, "Value"]
        for optional in ("Frame", "Timestamp"):
            if optional in df.columns:
                cols.append(optional)
        sub = df.loc[mask, cols].copy()
        sub = sub.rename(columns={tcol: "onset_sec", "Value": "label"})
        sub["source_row"] = np.flatnonzero(mask.to_numpy())
        sub = sub.sort_values("onset_sec").reset_index(drop=True)
        events[name] = sub
    return events


def stratified_indices(n_total: int, n_keep: int) -> np.ndarray:
    if n_total <= 0 or n_keep <= 0:
        return np.empty(0, dtype=int)
    if n_keep >= n_total:
        return np.arange(n_total, dtype=int)
    return np.unique(np.rint(np.linspace(0, n_total - 1, n_keep)).astype(int))


def choose_rois(group: h5py.Group, n_rois: int) -> Tuple[np.ndarray, List[str]]:
    if "roi_ids" in group:
        roi_ids = decode_strings(group["roi_ids"][:])
    else:
        n = int(group["dff"].shape[0])
        roi_ids = [str(i) for i in range(n)]

    if "valid_rois_mask" in group:
        valid = np.asarray(group["valid_rois_mask"][:], dtype=bool).reshape(-1)
        if valid.size != len(roi_ids):
            valid = np.ones(len(roi_ids), dtype=bool)
    else:
        valid = np.ones(len(roi_ids), dtype=bool)

    idx = np.flatnonzero(valid)
    if idx.size == 0:
        idx = np.arange(len(roi_ids), dtype=int)
    idx = idx[: max(1, int(n_rois))]
    return idx, [roi_ids[i] for i in idx]


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        try:
            dst[key] = value
        except TypeError:
            dst[key] = str(value)


def create_dataset(group: h5py.Group, name: str, data: np.ndarray,
                   gzip_level: int, **kwargs) -> h5py.Dataset:
    arr = np.asarray(data)
    if arr.size == 0:
        return group.create_dataset(name, data=arr, **kwargs)
    return group.create_dataset(
        name,
        data=arr,
        compression="gzip",
        compression_opts=int(gzip_level),
        shuffle=True,
        **kwargs,
    )


def infer_rate(timebase: np.ndarray) -> float:
    if timebase.size < 2:
        raise ValueError("Timebase has fewer than two samples")
    diffs = np.diff(timebase[: min(timebase.size, 200_000)])
    dt = float(np.nanmedian(diffs[np.isfinite(diffs) & (diffs > 0)]))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not infer a positive sampling interval")
    return 1.0 / dt


def extract_event_windows(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    timebase: np.ndarray,
    roi_idx: np.ndarray,
    event_df: pd.DataFrame,
    event_name: str,
    n_keep: int,
    gzip_level: int,
) -> None:
    selected = stratified_indices(len(event_df), n_keep)
    selected_df = event_df.iloc[selected].reset_index(drop=True)
    pre, post = EVENT_WINDOWS[event_name]
    fs = infer_rate(timebase)
    n_samples = int(round((pre + post) * fs))
    rel_time = (np.arange(n_samples, dtype=np.float64) / fs) - pre

    dst_group.attrs["pre_sec"] = float(pre)
    dst_group.attrs["post_sec"] = float(post)
    dst_group.attrs["sample_rate_hz"] = float(fs)
    create_dataset(dst_group, "relative_time_sec", rel_time, gzip_level)
    create_dataset(dst_group, "onset_sec", selected_df["onset_sec"].to_numpy(float), gzip_level)
    create_dataset(dst_group, "source_row", selected_df["source_row"].to_numpy(np.int64), gzip_level)
    if "Frame" in selected_df:
        create_dataset(dst_group, "frame", selected_df["Frame"].to_numpy(np.int64), gzip_level)
    labels = np.asarray(selected_df["label"].astype(str).tolist(), dtype=h5py.string_dtype("utf-8"))
    dst_group.create_dataset("label", data=labels)

    starts = np.searchsorted(timebase, selected_df["onset_sec"].to_numpy(float) - pre, side="left")
    valid = (starts >= 0) & ((starts + n_samples) <= timebase.size)
    create_dataset(dst_group, "valid_window", valid.astype(bool), gzip_level)

    signals = [s for s in ("raw_f", "f0", "dff") if s in src_group]
    for signal in signals:
        out = np.full((len(selected_df), len(roi_idx), n_samples), np.nan, dtype=np.float32)
        ds = src_group[signal]
        for i, (start, ok) in enumerate(zip(starts, valid)):
            if ok:
                out[i] = np.asarray(ds[roi_idx, int(start): int(start) + n_samples], dtype=np.float32)
        create_dataset(dst_group, signal, out, gzip_level,
                       chunks=(1, len(roi_idx), min(n_samples, 8192)))


def extract_continuous_snippets(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    timebase: np.ndarray,
    roi_idx: np.ndarray,
    duration_sec: float,
    gzip_level: int,
) -> None:
    fs = infer_rate(timebase)
    n = max(1, int(round(duration_sec * fs)))
    starts = {
        "start": 0,
        "middle": max(0, (timebase.size - n) // 2),
        "end": max(0, timebase.size - n),
    }
    for name, start in starts.items():
        stop = min(timebase.size, start + n)
        grp = dst_group.create_group(name)
        create_dataset(grp, "timebase_sec", timebase[start:stop], gzip_level)
        for signal in ("raw_f", "f0", "dff"):
            if signal in src_group:
                data = np.asarray(src_group[signal][roi_idx, start:stop], dtype=np.float32)
                create_dataset(grp, signal, data, gzip_level,
                               chunks=(len(roi_idx), min(data.shape[1], 8192)))


def block_summary(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    timebase: np.ndarray,
    roi_idx: np.ndarray,
    target_rate_hz: float,
    gzip_level: int,
) -> None:
    fs = infer_rate(timebase)
    block = max(1, int(round(fs / float(target_rate_hz))))
    n_blocks = timebase.size // block
    if n_blocks == 0:
        return
    centers = timebase[: n_blocks * block].reshape(n_blocks, block)[:, block // 2]
    create_dataset(dst_group, "time_sec", centers.astype(np.float64), gzip_level)
    dst_group.attrs["source_sample_rate_hz"] = float(fs)
    dst_group.attrs["block_size_samples"] = int(block)
    dst_group.attrs["summary_rate_hz"] = float(fs / block)

    # Read one ROI at a time to bound memory use.
    for signal in ("raw_f", "f0", "dff"):
        if signal not in src_group:
            continue
        stats = {
            "mean": np.empty((len(roi_idx), n_blocks), dtype=np.float32),
            "std": np.empty((len(roi_idx), n_blocks), dtype=np.float32),
            "min": np.empty((len(roi_idx), n_blocks), dtype=np.float32),
            "max": np.empty((len(roi_idx), n_blocks), dtype=np.float32),
        }
        for j, roi in enumerate(roi_idx):
            x = np.asarray(src_group[signal][int(roi), : n_blocks * block], dtype=np.float32)
            x = x.reshape(n_blocks, block)
            stats["mean"][j] = np.nanmean(x, axis=1)
            stats["std"][j] = np.nanstd(x, axis=1)
            stats["min"][j] = np.nanmin(x, axis=1)
            stats["max"][j] = np.nanmax(x, axis=1)
        sig_grp = dst_group.create_group(signal)
        for stat, data in stats.items():
            create_dataset(sig_grp, stat, data, gzip_level,
                           chunks=(len(roi_idx), min(n_blocks, 8192)))


def main() -> None:
    args = parse_args()
    if not args.session_h5.exists():
        raise FileNotFoundError(args.session_h5)
    if not args.event_log.exists():
        raise FileNotFoundError(args.event_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    event_log = pd.read_csv(args.event_log)
    events = parse_events(event_log)

    with h5py.File(args.session_h5, "r") as src, h5py.File(args.output, "w") as dst:
        dst.attrs["source_session_h5"] = str(args.session_h5)
        dst.attrs["source_event_log"] = str(args.event_log)
        dst.attrs["export_parameters_json"] = json.dumps(vars(args), default=str)
        copy_attrs(src.attrs, dst.attrs)

        dmd_keys = sorted(k for k in src.keys() if k.upper().startswith("DMD"))
        if not dmd_keys:
            raise ValueError(
                "No DMD groups found. This script expects the derived "
                "voltage_session_traces_*.h5 layout, not the source MATLAB trace H5."
            )

        event_limits = {
            "image": args.image_events,
            "change": args.change_events,
            "omission": args.omission_events,
        }

        for dmd_key in dmd_keys:
            src_grp = src[dmd_key]
            if "timebase_sec" not in src_grp:
                raise KeyError(f"{dmd_key}/timebase_sec is missing")
            timebase = np.asarray(src_grp["timebase_sec"][:], dtype=np.float64)
            roi_idx, roi_ids = choose_rois(src_grp, args.rois_per_dmd)

            out_grp = dst.create_group(dmd_key)
            copy_attrs(src_grp.attrs, out_grp.attrs)
            create_dataset(out_grp, "selected_roi_indices", roi_idx.astype(np.int64), args.gzip_level)
            out_grp.create_dataset(
                "selected_roi_ids",
                data=np.asarray(roi_ids, dtype=h5py.string_dtype("utf-8")),
            )
            out_grp.attrs["session_start_sec"] = float(timebase[0])
            out_grp.attrs["session_end_sec"] = float(timebase[-1])
            out_grp.attrs["sample_rate_hz_inferred"] = float(infer_rate(timebase))
            out_grp.attrs["n_source_samples"] = int(timebase.size)

            event_root = out_grp.create_group("events")
            for event_name, event_df in events.items():
                event_grp = event_root.create_group(event_name)
                extract_event_windows(
                    src_group=src_grp,
                    dst_group=event_grp,
                    timebase=timebase,
                    roi_idx=roi_idx,
                    event_df=event_df,
                    event_name=event_name,
                    n_keep=event_limits[event_name],
                    gzip_level=args.gzip_level,
                )

            extract_continuous_snippets(
                src_group=src_grp,
                dst_group=out_grp.create_group("continuous_snippets"),
                timebase=timebase,
                roi_idx=roi_idx,
                duration_sec=args.continuous_seconds,
                gzip_level=args.gzip_level,
            )
            block_summary(
                src_group=src_grp,
                dst_group=out_grp.create_group("full_session_summary"),
                timebase=timebase,
                roi_idx=roi_idx,
                target_rate_hz=args.summary_rate_hz,
                gzip_level=args.gzip_level,
            )

    size_mb = args.output.stat().st_size / (1024 ** 2)
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
