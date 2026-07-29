"""Epoch-aware imaging timebase helpers shared across modalities.

Trace arrays remain concatenated in acquisition-sample order, while the returned
sample timebase jumps across periods when imaging was off. This keeps numerical
arrays compact without pretending that an acquisition restart was continuous.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]
ScaleMode = Union[bool, str]


@dataclass
class EpochAwareTimebase:
    """Per-sample and per-trial timing for an imaging session.

    ``trial_epoch`` and ``sample_epoch`` use one-indexed epoch identifiers. Zero
    denotes a zero-length/unassigned placeholder.
    """

    timebase_sec: np.ndarray
    trial_starts_sec: np.ndarray
    trial_epoch: np.ndarray
    sample_epoch: np.ndarray
    epoch_df: pd.DataFrame
    metadata: Dict[str, Any]


def normalize_epoch_dataframe(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, validated epoch table with canonical columns."""
    if epoch_df is None:
        raise ValueError("epoch_df cannot be None")
    df = epoch_df.copy()
    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"epoch_df is missing required columns: {sorted(missing)}")

    df["start_time"] = pd.to_numeric(df["start_time"], errors="raise").astype(float)
    df["end_time"] = pd.to_numeric(df["end_time"], errors="raise").astype(float)
    df = df.sort_values("start_time", kind="stable").reset_index(drop=True)
    df["duration_s"] = df["end_time"] - df["start_time"]

    for col in ("epoch", "epoch_index", "epoch_idx"):
        if col in df.columns:
            epoch_idx = pd.to_numeric(df[col], errors="raise").astype(int).to_numpy()
            break
    else:
        epoch_idx = np.arange(1, len(df) + 1, dtype=int)
    df["epoch_index"] = epoch_idx

    if len(df) == 0:
        raise ValueError("At least one imaging epoch is required.")
    if len(np.unique(epoch_idx)) != len(epoch_idx):
        raise ValueError("Imaging epoch identifiers must be unique.")
    duration = df["duration_s"].to_numpy(dtype=float)
    if np.any(~np.isfinite(duration)) or np.any(duration <= 0):
        raise ValueError("All imaging epochs must have positive finite durations.")
    starts = df["start_time"].to_numpy(dtype=float)
    ends = df["end_time"].to_numpy(dtype=float)
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)):
        raise ValueError("Imaging epoch boundaries must be finite.")
    if len(df) > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError("Imaging epochs overlap; expected non-overlapping HARP intervals.")

    gap_before = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        gap_before[1:] = starts[1:] - ends[:-1]
    df["gap_before_s"] = gap_before
    return df


def load_epoch_dataframe(path: Optional[PathLike]) -> Optional[pd.DataFrame]:
    """Load and normalize an imaging-epochs CSV if ``path`` exists."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return normalize_epoch_dataframe(pd.read_csv(p))


def epoch_csv_from_asset(asset: Any) -> Optional[Path]:
    """Resolve ``qc/behavior/imaging_epochs.csv`` from a SessionAssets-like object."""
    qdir = getattr(asset, "qc_dir", None)
    if qdir is None:
        return None
    return Path(qdir) / "behavior" / "imaging_epochs.csv"


def load_epoch_dataframe_from_asset(asset: Any) -> Optional[pd.DataFrame]:
    """Load behavior-QC imaging epochs from a SessionAssets-like object."""
    return load_epoch_dataframe(epoch_csv_from_asset(asset))


def epoch_sample_slices(
    trial_lengths_samples: Sequence[int],
    trial_epoch: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Return contiguous ``(epoch, start, stop)`` sample slices.

    Raises when the same epoch appears in disjoint trial blocks, because that would
    make downstream filtering/baseline estimation ambiguous.
    """
    lengths = np.asarray(trial_lengths_samples, dtype=int).reshape(-1)
    epochs = np.asarray(trial_epoch, dtype=int).reshape(-1)
    if lengths.size != epochs.size:
        raise ValueError("trial_lengths_samples and trial_epoch must have equal length")
    if np.any(lengths < 0):
        raise ValueError("trial_lengths_samples cannot contain negative values")

    out: List[Tuple[int, int, int]] = []
    pos = 0
    seen = set()
    current_epoch: Optional[int] = None
    current_start = 0
    for length, epoch in zip(lengths, epochs):
        length = int(length)
        epoch = int(epoch)
        if length <= 0 or epoch <= 0:
            pos += max(length, 0)
            continue
        if current_epoch is None:
            current_epoch = epoch
            current_start = pos
        elif epoch != current_epoch:
            out.append((current_epoch, current_start, pos))
            seen.add(current_epoch)
            if epoch in seen:
                raise ValueError(f"Epoch {epoch} appears in multiple disjoint trial blocks")
            current_epoch = epoch
            current_start = pos
        pos += length
    if current_epoch is not None:
        out.append((current_epoch, current_start, pos))
    return out


def assign_trials_to_epochs_by_duration(
    trial_lengths_samples: Sequence[int],
    *,
    sample_rate_hz: float,
    epoch_df: pd.DataFrame,
) -> np.ndarray:
    """Assign ordered trials to epochs by cumulative acquired duration.

    This is a fallback for modalities whose source summary lacks explicit
    trial-to-epoch labels. Extractor-provided labels are preferred whenever present.
    """
    lengths = np.asarray(trial_lengths_samples, dtype=int).reshape(-1)
    if lengths.size == 0:
        return np.zeros((0,), dtype=int)
    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        raise ValueError(f"sample_rate_hz must be positive and finite, got {sample_rate_hz}")
    epochs = normalize_epoch_dataframe(epoch_df)
    durations = epochs["duration_s"].to_numpy(dtype=float)
    cumulative_ends = np.cumsum(durations)
    cum_samples = np.concatenate([[0], np.cumsum(lengths)])
    centers_s = (cum_samples[:-1] + 0.5 * np.maximum(lengths, 1)) / float(sample_rate_hz)
    out = np.zeros(lengths.size, dtype=int)
    ids = epochs["epoch_index"].to_numpy(dtype=int)
    for i, (length, center) in enumerate(zip(lengths, centers_s)):
        if length <= 0:
            continue
        idx = int(np.searchsorted(cumulative_ends, center, side="right"))
        idx = min(max(idx, 0), len(ids) - 1)
        out[i] = int(ids[idx])
    return out


def _resolve_scale_mode(scale_each_epoch: ScaleMode) -> str:
    if isinstance(scale_each_epoch, (bool, np.bool_)):
        return "always" if bool(scale_each_epoch) else "never"
    mode = str(scale_each_epoch).strip().lower()
    aliases = {"true": "always", "false": "never", "yes": "always", "no": "never"}
    mode = aliases.get(mode, mode)
    if mode not in {"always", "never", "auto"}:
        raise ValueError("scale_each_epoch must be bool or one of {'always', 'never', 'auto'}")
    return mode


def build_epoch_aware_timebase(
    trial_lengths_samples: Sequence[int],
    *,
    sample_rate_hz: float,
    epoch_df: pd.DataFrame,
    trial_epoch: Optional[Sequence[int]] = None,
    scale_each_epoch: ScaleMode = "auto",
    scale_tolerance_sec: float = 0.050,
    strict_epoch_match: bool = True,
) -> EpochAwareTimebase:
    """Build a per-sample HARP timebase while preserving acquisition gaps.

    ``scale_each_epoch='auto'`` uses nominal sample spacing when the extracted
    duration agrees with behavior QC within ``scale_tolerance_sec``; otherwise it
    applies a small epoch-local linear correction. No interpolation or baseline
    operation spans an acquisition gap.
    """
    lengths = np.asarray(trial_lengths_samples, dtype=int).reshape(-1)
    if np.any(lengths < 0):
        raise ValueError("trial_lengths_samples cannot contain negative values")
    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        raise ValueError("sample_rate_hz must be positive and finite")
    epochs = normalize_epoch_dataframe(epoch_df)
    mode = _resolve_scale_mode(scale_each_epoch)

    if trial_epoch is None:
        trial_epoch_arr = assign_trials_to_epochs_by_duration(
            lengths, sample_rate_hz=sample_rate_hz, epoch_df=epochs
        )
        assignment_method = "duration_fallback"
    else:
        trial_epoch_arr = np.asarray(trial_epoch, dtype=int).reshape(-1)
        if trial_epoch_arr.size != lengths.size:
            raise ValueError(
                f"trial_epoch length {trial_epoch_arr.size} does not match trial_lengths {lengths.size}"
            )
        assignment_method = "provided"

    epoch_ids = epochs["epoch_index"].to_numpy(dtype=int)
    assigned_ids = np.unique(trial_epoch_arr[(trial_epoch_arr > 0) & (lengths > 0)])
    unknown = sorted(set(assigned_ids.tolist()) - set(epoch_ids.tolist()))
    if unknown:
        raise ValueError(f"Trials are assigned to unknown imaging epochs: {unknown}")
    missing = sorted(set(epoch_ids.tolist()) - set(assigned_ids.tolist()))
    if strict_epoch_match and missing:
        raise ValueError(
            "Behavior QC contains imaging epochs with no extracted samples: " + ", ".join(map(str, missing))
        )

    slices = epoch_sample_slices(lengths, trial_epoch_arr)
    slice_ids = [x[0] for x in slices]
    if strict_epoch_match and slice_ids != [int(x) for x in epoch_ids if x in assigned_ids]:
        raise ValueError(
            f"Extracted epoch order {slice_ids} does not match behavior epoch order {epoch_ids.tolist()}"
        )

    total_samples = int(np.sum(lengths))
    timebase = np.full((total_samples,), np.nan, dtype=float)
    sample_epoch = np.zeros((total_samples,), dtype=int)
    trial_starts = np.full((lengths.size,), np.nan, dtype=float)
    row_by_id = {int(row.epoch_index): row for row in epochs.itertuples(index=False)}

    samples_per_epoch: Dict[str, int] = {}
    effective_rates: Dict[str, float] = {}
    nominal_durations: Dict[str, float] = {}
    duration_errors: Dict[str, float] = {}
    scaled_by_epoch: Dict[str, bool] = {}

    trial_offsets = np.concatenate([[0], np.cumsum(lengths)])
    for epoch_id, start, stop in slices:
        row = row_by_id[int(epoch_id)]
        n_epoch = int(stop - start)
        nominal_duration = float(n_epoch / float(sample_rate_hz))
        duration_error = float(nominal_duration - float(row.duration_s))
        do_scale = mode == "always" or (mode == "auto" and abs(duration_error) > float(scale_tolerance_sec))
        dt = float(row.duration_s) / float(n_epoch) if do_scale else 1.0 / float(sample_rate_hz)
        timebase[start:stop] = float(row.start_time) + np.arange(n_epoch, dtype=float) * dt
        sample_epoch[start:stop] = int(epoch_id)
        samples_per_epoch[str(epoch_id)] = n_epoch
        effective_rates[str(epoch_id)] = float(1.0 / dt)
        nominal_durations[str(epoch_id)] = nominal_duration
        duration_errors[str(epoch_id)] = duration_error
        scaled_by_epoch[str(epoch_id)] = bool(do_scale)

    for i, (start, length, epoch_id) in enumerate(zip(trial_offsets[:-1], lengths, trial_epoch_arr)):
        if length > 0 and epoch_id > 0 and start < total_samples and np.isfinite(timebase[start]):
            trial_starts[i] = float(timebase[start])

    finite = np.isfinite(timebase)
    if total_samples and not np.all(finite):
        raise ValueError("Some positive-length samples were not assigned to an imaging epoch")
    if total_samples > 1 and np.any(np.diff(timebase) <= 0):
        raise ValueError("Constructed imaging timebase is not strictly increasing")

    acquired_duration = float(epochs["duration_s"].sum())
    session_span = float(epochs["end_time"].iloc[-1] - epochs["start_time"].iloc[0])
    gap_duration = float(session_span - acquired_duration)
    metadata: Dict[str, Any] = {
        "epoch_aware": True,
        "n_epochs": int(len(epochs)),
        "trial_epoch_assignment_method": assignment_method,
        "scale_each_epoch": mode,
        "scale_tolerance_sec": float(scale_tolerance_sec),
        "sample_rate_hz_nominal": float(sample_rate_hz),
        "samples_per_epoch": samples_per_epoch,
        "effective_sample_rate_hz_by_epoch": effective_rates,
        "nominal_acquired_duration_sec_by_epoch": nominal_durations,
        "duration_error_sec_by_epoch": duration_errors,
        "scaled_by_epoch": scaled_by_epoch,
        "epoch_start_time": epochs["start_time"].astype(float).tolist(),
        "epoch_end_time": epochs["end_time"].astype(float).tolist(),
        "epoch_gap_before_sec": epochs["gap_before_s"].astype(float).tolist(),
        "acquired_duration_sec": acquired_duration,
        "session_span_sec": session_span,
        "imaging_gap_duration_sec": gap_duration,
        "trial_epoch": trial_epoch_arr.astype(int).tolist(),
        "missing_behavior_epochs": missing,
        "strict_epoch_match": bool(strict_epoch_match),
    }
    return EpochAwareTimebase(
        timebase_sec=timebase,
        trial_starts_sec=trial_starts,
        trial_epoch=trial_epoch_arr.astype(int),
        sample_epoch=sample_epoch,
        epoch_df=epochs,
        metadata=metadata,
    )
