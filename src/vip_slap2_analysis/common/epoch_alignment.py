"""Epoch-aware imaging timebase helpers shared across modalities.

These utilities keep acquisition gaps explicit when reconstructing session-wide
SLAP2 traces from trial-wise outputs.  The core convention is that traces remain
concatenated in acquisition-sample order, but the returned ``timebase_sec`` jumps
across behavior-time gaps between imaging epochs.  Downstream event extraction can
then use the explicit timebase while normal epoch filtering prevents windows from
crossing gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


@dataclass
class EpochAwareTimebase:
    """Container returned by :func:`build_epoch_aware_timebase`.

    Attributes
    ----------
    timebase_sec
        Per-sample acquisition time in corrected behavior/HARP seconds.
    trial_starts_sec
        Corrected-time start of each trial in the reconstructed trace bundle.
    trial_epoch
        One-indexed epoch assignment for each trial, or zero for zero-length
        placeholder trials.
    epoch_df
        Normalized epoch table used to build the timebase.
    metadata
        JSON-friendly diagnostics describing the assignment and scaling.
    """
    timebase_sec: np.ndarray
    trial_starts_sec: np.ndarray
    trial_epoch: np.ndarray
    epoch_df: pd.DataFrame
    metadata: Dict[str, Any]


def normalize_epoch_dataframe(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean epoch table with canonical columns.

    Required input columns are ``start_time`` and ``end_time``.  If no epoch index
    column is present, epochs are assigned 1..N in row order.
    """
    if epoch_df is None:
        raise ValueError("epoch_df cannot be None")
    df = epoch_df.copy()
    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"epoch_df is missing required columns: {sorted(missing)}")
    df = df.sort_values("start_time").reset_index(drop=True)
    df["start_time"] = df["start_time"].astype(float)
    df["end_time"] = df["end_time"].astype(float)
    df["duration_s"] = df["end_time"] - df["start_time"]
    if "epoch" in df.columns:
        epoch_idx = df["epoch"].astype(int).to_numpy()
    elif "epoch_index" in df.columns:
        epoch_idx = df["epoch_index"].astype(int).to_numpy()
    elif "epoch_idx" in df.columns:
        epoch_idx = df["epoch_idx"].astype(int).to_numpy()
    else:
        epoch_idx = np.arange(1, len(df) + 1, dtype=int)
    df["epoch_index"] = epoch_idx
    if np.any(~np.isfinite(df["duration_s"].to_numpy())) or np.any(df["duration_s"].to_numpy() <= 0):
        raise ValueError("All imaging epochs must have positive finite durations.")
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
    """Load behavior QC imaging epochs from a SessionAssets-like object."""
    return load_epoch_dataframe(epoch_csv_from_asset(asset))


def assign_trials_to_epochs_by_duration(
    trial_lengths_samples: Sequence[int],
    *,
    sample_rate_hz: float,
    epoch_df: pd.DataFrame,
) -> np.ndarray:
    """Assign sequential trials to imaging epochs by cumulative acquired time.

    This is the safe fallback when extractor-level trial→epoch metadata is absent.
    It assumes trial traces are ordered by acquisition time and that gaps between
    epochs are not represented as samples.
    """
    lengths = np.asarray(trial_lengths_samples, dtype=int).reshape(-1)
    if lengths.size == 0:
        return np.zeros((0,), dtype=int)
    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        raise ValueError(f"sample_rate_hz must be positive and finite, got {sample_rate_hz}")
    epochs = normalize_epoch_dataframe(epoch_df)
    durations = epochs["duration_s"].to_numpy(dtype=float)
    starts_nominal = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    ends_nominal = np.cumsum(durations)
    cum_samples = np.concatenate([[0], np.cumsum(lengths)])
    centers_s = (cum_samples[:-1] + 0.5 * np.maximum(lengths, 1)) / float(sample_rate_hz)
    out = np.zeros(lengths.size, dtype=int)
    for i, (L, center) in enumerate(zip(lengths, centers_s)):
        if L <= 0:
            continue
        idx = int(np.searchsorted(ends_nominal, center, side="right"))
        idx = min(max(idx, 0), len(epochs) - 1)
        # If a center lands just beyond the total nominal duration due to rounding,
        # clamp to the final epoch.
        if center < starts_nominal[idx] and idx > 0:
            idx -= 1
        out[i] = int(epochs["epoch_index"].iloc[idx])
    return out


def build_epoch_aware_timebase(
    trial_lengths_samples: Sequence[int],
    *,
    sample_rate_hz: float,
    epoch_df: pd.DataFrame,
    trial_epoch: Optional[Sequence[int]] = None,
    scale_each_epoch: bool = True,
) -> EpochAwareTimebase:
    """Build a per-sample timebase that preserves gaps between imaging epochs.

    Trials are still concatenated in trace arrays, but samples assigned to later
    epochs receive later corrected timestamps.  If ``scale_each_epoch`` is true,
    samples within each epoch are linearly mapped onto the detected epoch duration;
    otherwise nominal ``1 / sample_rate_hz`` spacing is used from each epoch start.
    """
    lengths = np.asarray(trial_lengths_samples, dtype=int).reshape(-1)
    if np.any(lengths < 0):
        raise ValueError("trial_lengths_samples cannot contain negative values")
    epochs = normalize_epoch_dataframe(epoch_df)
    if trial_epoch is None:
        trial_epoch_arr = assign_trials_to_epochs_by_duration(
            lengths,
            sample_rate_hz=sample_rate_hz,
            epoch_df=epochs,
        )
        assignment_method = "duration_fallback"
    else:
        trial_epoch_arr = np.asarray(trial_epoch, dtype=int).reshape(-1)
        if trial_epoch_arr.size != lengths.size:
            raise ValueError(
                f"trial_epoch length {trial_epoch_arr.size} does not match trial_lengths {lengths.size}"
            )
        assignment_method = "provided"

    total_samples = int(np.sum(lengths))
    timebase = np.empty((total_samples,), dtype=float)
    trial_starts = np.full((lengths.size,), np.nan, dtype=float)

    pos = 0
    epoch_rows = {int(row.epoch_index): row for row in epochs.itertuples(index=False)}
    samples_per_epoch: Dict[int, int] = {}
    for epoch_idx in epoch_rows:
        samples_per_epoch[epoch_idx] = int(np.sum(lengths[trial_epoch_arr == epoch_idx]))

    epoch_offsets: Dict[int, int] = {k: 0 for k in epoch_rows}
    effective_rates: Dict[str, float] = {}
    duration_errors: Dict[str, float] = {}

    for trial_i, L in enumerate(lengths):
        epoch_idx = int(trial_epoch_arr[trial_i]) if trial_i < trial_epoch_arr.size else 0
        if L <= 0 or epoch_idx == 0:
            continue
        if epoch_idx not in epoch_rows:
            raise ValueError(f"Trial {trial_i + 1} assigned to unknown epoch {epoch_idx}")
        row = epoch_rows[epoch_idx]
        n_epoch = int(samples_per_epoch[epoch_idx])
        offset = int(epoch_offsets[epoch_idx])
        if n_epoch <= 0:
            continue
        if scale_each_epoch:
            # Use sample centers over [start, end), avoiding endpoint duplication.
            dt = float(row.duration_s) / float(n_epoch)
            t = float(row.start_time) + (offset + np.arange(L, dtype=float)) * dt
            effective_rate = float(n_epoch / float(row.duration_s))
            nominal_duration = float(n_epoch / float(sample_rate_hz))
            duration_error = float(nominal_duration - float(row.duration_s))
        else:
            t = float(row.start_time) + (offset + np.arange(L, dtype=float)) / float(sample_rate_hz)
            effective_rate = float(sample_rate_hz)
            duration_error = np.nan
        timebase[pos:pos + L] = t
        trial_starts[trial_i] = float(t[0])
        pos += L
        epoch_offsets[epoch_idx] = offset + L
        effective_rates[str(epoch_idx)] = effective_rate
        duration_errors[str(epoch_idx)] = float(duration_error)

    metadata = {
        "epoch_aware": True,
        "n_epochs": int(len(epochs)),
        "trial_epoch_assignment_method": assignment_method,
        "scale_each_epoch": bool(scale_each_epoch),
        "sample_rate_hz_nominal": float(sample_rate_hz),
        "samples_per_epoch": {str(k): int(v) for k, v in samples_per_epoch.items()},
        "effective_sample_rate_hz_by_epoch": effective_rates,
        "duration_error_sec_by_epoch": duration_errors,
        "epoch_start_time": epochs["start_time"].astype(float).tolist(),
        "epoch_end_time": epochs["end_time"].astype(float).tolist(),
        "trial_epoch": trial_epoch_arr.astype(int).tolist(),
    }
    return EpochAwareTimebase(
        timebase_sec=timebase,
        trial_starts_sec=trial_starts,
        trial_epoch=trial_epoch_arr.astype(int),
        epoch_df=epochs,
        metadata=metadata,
    )
