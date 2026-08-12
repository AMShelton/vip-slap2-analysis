"""Epoch-aware imaging timebase and duration-QC helpers shared across modalities.

Trace arrays remain concatenated in acquisition-sample order, while the returned
sample timebase jumps across periods when imaging was off.  Acquisition fragments
shorter than :data:`DEFAULT_MIN_EPOCH_DURATION_SEC` are retained in source files
but assigned analysis epoch ``0`` and excluded from downstream physiology.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]
ScaleMode = Union[bool, str]
DEFAULT_MIN_EPOCH_DURATION_SEC = 30.0


@dataclass
class EpochAwareTimebase:
    """Per-sample and per-trial timing for an imaging session.

    ``trial_epoch`` and ``sample_epoch`` use one-indexed *analysis* epoch
    identifiers. Zero denotes a rejected, zero-length, or unassigned trial.
    """

    timebase_sec: np.ndarray
    trial_starts_sec: np.ndarray
    trial_epoch: np.ndarray
    sample_epoch: np.ndarray
    epoch_df: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass
class EpochReconciliation:
    """Result of reconciling source-acquisition epochs with behavior epochs.

    Source epoch labels are preserved in ``source_trial_epoch``.  Accepted source
    epochs are mapped chronologically onto accepted behavior epochs and exposed as
    ``analysis_trial_epoch``.  Rejected source epochs receive analysis label 0.
    ``trial_lengths_samples`` contains the number of samples retained after
    duration filtering and clipping to the common behavior/physiology interval.
    """

    behavior_epoch_df: pd.DataFrame
    behavior_epoch_qc: pd.DataFrame
    source_epoch_qc: pd.DataFrame
    original_trial_lengths_samples: np.ndarray
    trial_lengths_samples: np.ndarray
    source_trial_epoch: np.ndarray
    analysis_trial_epoch: np.ndarray
    trial_keep_mask: np.ndarray
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


def classify_epochs_by_duration(
    epoch_df: pd.DataFrame,
    *,
    min_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
) -> pd.DataFrame:
    """Classify behavior/imaging epochs using the shared duration criterion.

    The returned table contains every source row and adds ``accepted``,
    ``discard_reason``, ``source_epoch_index`` and ``analysis_epoch_index``.
    Accepted epochs are relabeled 1..N in chronological order for analysis.
    """
    threshold = float(min_duration_sec)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("min_duration_sec must be finite and non-negative")
    df = normalize_epoch_dataframe(epoch_df)
    df["source_epoch_index"] = df["epoch_index"].astype(int)
    df["accepted"] = df["duration_s"].to_numpy(dtype=float) >= threshold
    df["discard_reason"] = np.where(
        df["accepted"],
        "",
        "duration_below_minimum",
    )
    analysis_ids = np.zeros(len(df), dtype=int)
    analysis_ids[df["accepted"].to_numpy(dtype=bool)] = np.arange(
        1, int(df["accepted"].sum()) + 1, dtype=int
    )
    df["analysis_epoch_index"] = analysis_ids
    df["min_duration_sec"] = threshold
    return df


def accepted_epoch_dataframe(
    epoch_df: pd.DataFrame,
    *,
    min_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
    require_any: bool = True,
) -> pd.DataFrame:
    """Return accepted epochs relabeled consecutively for downstream analysis."""
    qc = classify_epochs_by_duration(epoch_df, min_duration_sec=min_duration_sec)
    accepted = qc.loc[qc["accepted"]].copy().reset_index(drop=True)
    if require_any and len(accepted) == 0:
        raise ValueError(
            f"No imaging epochs met the minimum duration of {float(min_duration_sec):g} s."
        )
    if len(accepted):
        accepted["behavior_epoch_index"] = accepted["source_epoch_index"].astype(int)
        accepted["epoch_index"] = accepted["analysis_epoch_index"].astype(int)
        starts = accepted["start_time"].to_numpy(dtype=float)
        ends = accepted["end_time"].to_numpy(dtype=float)
        gap_before = np.zeros(len(accepted), dtype=float)
        if len(accepted) > 1:
            gap_before[1:] = starts[1:] - ends[:-1]
        accepted["gap_before_s"] = gap_before
    return accepted


def load_epoch_dataframe(
    path: Optional[PathLike],
    *,
    min_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
) -> Optional[pd.DataFrame]:
    """Load, duration-filter, and normalize an imaging-epochs CSV if it exists."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return accepted_epoch_dataframe(
        pd.read_csv(p), min_duration_sec=min_duration_sec, require_any=True
    )


def epoch_csv_from_asset(asset: Any) -> Optional[Path]:
    """Resolve ``qc/behavior/imaging_epochs.csv`` from a SessionAssets-like object."""
    qdir = getattr(asset, "qc_dir", None)
    if qdir is None:
        return None
    return Path(qdir) / "behavior" / "imaging_epochs.csv"


def load_epoch_dataframe_from_asset(
    asset: Any,
    *,
    min_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
) -> Optional[pd.DataFrame]:
    """Load accepted behavior-QC imaging epochs from a SessionAssets-like object."""
    return load_epoch_dataframe(
        epoch_csv_from_asset(asset), min_duration_sec=min_duration_sec
    )


def epoch_sample_slices(
    trial_lengths_samples: Sequence[int],
    trial_epoch: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Return contiguous ``(epoch, start, stop)`` sample slices.

    Raises when the same positive epoch appears in disjoint trial blocks. Rejected
    trials (epoch 0 or length 0) are skipped without occupying output samples.
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
    """Assign ordered trials to accepted epochs by cumulative acquired duration.

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


def _ordered_positive_epoch_ids(labels: np.ndarray, lengths: np.ndarray) -> List[int]:
    """Return positive epoch labels in first-occurrence order and validate blocks."""
    # ``epoch_sample_slices`` also verifies that an epoch does not reappear later.
    return [int(ep) for ep, _start, _stop in epoch_sample_slices(lengths, labels)]


def _max_samples_inside_epoch(duration_s: float, sample_rate_hz: float) -> int:
    """Maximum nominal-rate samples whose centers remain before an epoch end."""
    # Samples are placed at start + k/fs, so ceil(duration*fs) keeps the last sample
    # strictly before end for noninteger products and gives exactly N for integer N.
    return max(0, int(np.ceil(float(duration_s) * float(sample_rate_hz) - 1e-12)))


def reconcile_trial_epochs(
    trial_lengths_samples: Sequence[int],
    *,
    sample_rate_hz: float,
    behavior_epoch_df: pd.DataFrame,
    source_trial_epoch: Optional[Sequence[int]] = None,
    source_epoch_durations_sec: Optional[Mapping[int, float]] = None,
    min_epoch_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
    strict_epoch_match: bool = True,
    duration_warning_sec: float = 0.5,
) -> EpochReconciliation:
    """Apply the shared >=30 s QC rule and map source epochs to behavior epochs.

    All raw source epochs remain available upstream.  Source epochs shorter than
    ``min_epoch_duration_sec`` are excluded by setting their retained trial lengths
    and analysis labels to zero. Accepted source epochs are paired chronologically
    with accepted behavior epochs.  If a source epoch is longer than its paired
    behavior interval, the last retained trial is clipped and later samples are
    discarded; the nominal sampling interval is never rescaled.

    For modalities without extractor-provided source labels, trials are assigned to
    accepted behavior epochs by cumulative acquired duration, then clipped at each
    behavior endpoint. When ``source_epoch_durations_sec`` is supplied, those
    acquisition-level durations determine source-epoch acceptance; retained sample
    budgets still come from trial arrays so raw samples remain the source of truth.
    """
    lengths = np.asarray(trial_lengths_samples, dtype=int).reshape(-1)
    if np.any(lengths < 0):
        raise ValueError("trial_lengths_samples cannot contain negative values")
    fs = float(sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")

    behavior_qc = classify_epochs_by_duration(
        behavior_epoch_df, min_duration_sec=min_epoch_duration_sec
    )
    behavior = accepted_epoch_dataframe(
        behavior_epoch_df,
        min_duration_sec=min_epoch_duration_sec,
        require_any=True,
    )
    behavior_ids = behavior["epoch_index"].to_numpy(dtype=int)

    source_duration_overrides: Dict[int, float] = {}
    if source_epoch_durations_sec is not None:
        for key, value in source_epoch_durations_sec.items():
            epoch_id = int(key)
            duration = float(value)
            if epoch_id <= 0 or not np.isfinite(duration) or duration < 0:
                raise ValueError(
                    "source_epoch_durations_sec must map positive epoch IDs to "
                    "finite non-negative durations"
                )
            source_duration_overrides[epoch_id] = duration

    explicit_source_labels = source_trial_epoch is not None
    if source_trial_epoch is None:
        source_labels = assign_trials_to_epochs_by_duration(
            lengths, sample_rate_hz=fs, epoch_df=behavior
        )
        ordered_source = [int(x) for x in behavior_ids]
        accepted_source = list(ordered_source)
        source_assignment_method = "behavior_duration_fallback"
    else:
        source_labels = np.asarray(source_trial_epoch, dtype=int).reshape(-1)
        if source_labels.size != lengths.size:
            raise ValueError(
                f"source_trial_epoch length {source_labels.size} does not match "
                f"trial_lengths_samples length {lengths.size}"
            )
        if np.any(source_labels < 0):
            raise ValueError("source_trial_epoch cannot contain negative labels")
        ordered_source = _ordered_positive_epoch_ids(source_labels, lengths)
        source_assignment_method = "extractor_provided"
        accepted_source = []
        for source_id in ordered_source:
            trial_covered_duration = float(
                np.sum(lengths[source_labels == source_id]) / fs
            )
            duration = float(
                source_duration_overrides.get(int(source_id), trial_covered_duration)
            )
            if duration >= float(min_epoch_duration_sec):
                accepted_source.append(int(source_id))

    warnings: List[str] = []
    if strict_epoch_match and len(accepted_source) != len(behavior_ids):
        raise ValueError(
            "Accepted physiology acquisition epochs do not match accepted behavior "
            f"epochs after applying the {float(min_epoch_duration_sec):g} s threshold: "
            f"physiology={len(accepted_source)}, behavior={len(behavior_ids)}."
        )
    n_pairs = min(len(accepted_source), len(behavior_ids))
    source_to_analysis = {
        int(source_id): int(behavior_ids[i])
        for i, source_id in enumerate(accepted_source[:n_pairs])
    }
    if len(accepted_source) != len(behavior_ids):
        warnings.append(
            f"Mapped {n_pairs} accepted source epoch(s) to {len(behavior_ids)} behavior epoch(s); "
            "unpaired source epochs were discarded."
        )

    analysis_labels = np.asarray(
        [source_to_analysis.get(int(x), 0) if int(x) > 0 else 0 for x in source_labels],
        dtype=int,
    )
    retained_lengths = np.zeros_like(lengths)

    source_rows: List[Dict[str, Any]] = []
    behavior_by_analysis = {
        int(row.epoch_index): row for row in behavior.itertuples(index=False)
    }
    for source_id in ordered_source:
        idx = np.flatnonzero(source_labels == int(source_id))
        original_samples = int(np.sum(lengths[idx]))
        trial_covered_duration = float(original_samples / fs)
        source_duration = float(
            source_duration_overrides.get(int(source_id), trial_covered_duration)
        )
        accepted_by_duration = source_duration >= float(min_epoch_duration_sec)
        analysis_id = int(source_to_analysis.get(int(source_id), 0))
        keep_budget = 0
        behavior_duration = np.nan
        behavior_source_id = 0
        if analysis_id > 0:
            brow = behavior_by_analysis[analysis_id]
            behavior_duration = float(brow.duration_s)
            behavior_source_id = int(getattr(brow, "behavior_epoch_index", analysis_id))
            keep_budget = _max_samples_inside_epoch(behavior_duration, fs)

        remaining = int(keep_budget)
        for trial_idx in idx:
            if analysis_id <= 0 or remaining <= 0:
                retained_lengths[trial_idx] = 0
                continue
            keep = min(int(lengths[trial_idx]), remaining)
            retained_lengths[trial_idx] = keep
            remaining -= keep

        kept_samples = int(np.sum(retained_lengths[idx]))
        discarded_samples = int(original_samples - kept_samples)
        duration_error = (
            float(trial_covered_duration - behavior_duration)
            if np.isfinite(behavior_duration)
            else np.nan
        )
        if analysis_id > 0 and np.isfinite(duration_error) and abs(duration_error) > float(duration_warning_sec):
            direction = "longer" if duration_error > 0 else "shorter"
            warnings.append(
                f"Source epoch {source_id} is {abs(duration_error):.3f} s {direction} than "
                f"behavior epoch {behavior_source_id}; nominal-rate data were "
                + ("clipped to the common interval." if duration_error > 0 else "kept without rescaling.")
            )

        if not accepted_by_duration:
            discard_reason = "duration_below_minimum"
        elif analysis_id <= 0:
            discard_reason = "unpaired_accepted_source_epoch"
        elif discarded_samples > 0:
            discard_reason = "trimmed_to_behavior_epoch"
        else:
            discard_reason = ""

        source_rows.append(
            {
                "source_epoch_index": int(source_id),
                "source_duration_s": source_duration,
                "source_duration_basis": (
                    "acquisition_metadata"
                    if int(source_id) in source_duration_overrides
                    else "trial_covered_samples"
                ),
                "trial_covered_duration_s": trial_covered_duration,
                "source_samples": original_samples,
                "n_trials_total": int(idx.size),
                "accepted_by_duration": bool(accepted_by_duration),
                "analysis_epoch_index": analysis_id,
                "behavior_epoch_index": behavior_source_id,
                "behavior_duration_s": behavior_duration,
                "kept_samples": kept_samples,
                "kept_duration_s": float(kept_samples / fs),
                "discarded_samples": discarded_samples,
                "n_trials_kept": int(np.sum(retained_lengths[idx] > 0)),
                "discard_reason": discard_reason,
                "min_duration_sec": float(min_epoch_duration_sec),
            }
        )

    # Include any source label 0 trials as explicit rejected rows only in metadata.
    zero_trials = int(np.sum((source_labels <= 0) & (lengths > 0)))
    if zero_trials:
        warnings.append(f"{zero_trials} positive-length trial(s) had no source epoch label and were discarded.")

    # If source labels were inferred directly from behavior epochs, source durations
    # should not themselves trigger another duration rejection.  The behavior table
    # has already enforced the policy and endpoint clipping below remains valid.
    if not explicit_source_labels:
        for row in source_rows:
            row["accepted_by_duration"] = True
            if row["analysis_epoch_index"] > 0 and row["discard_reason"] == "duration_below_minimum":
                row["discard_reason"] = ""

    analysis_labels = np.where(retained_lengths > 0, analysis_labels, 0).astype(int)
    trial_keep_mask = retained_lengths > 0
    source_qc = pd.DataFrame(source_rows)

    metadata: Dict[str, Any] = {
        "policy": "retain_acquisition_epochs_at_or_above_minimum_duration",
        "min_epoch_duration_sec": float(min_epoch_duration_sec),
        "preserve_nominal_sample_spacing": True,
        "tail_policy": "clip_to_common_behavior_physiology_interval",
        "source_trial_epoch_assignment_method": source_assignment_method,
        "source_epoch_duration_basis": (
            "acquisition_metadata_when_available"
            if source_duration_overrides
            else "trial_covered_samples"
        ),
        "source_epoch_duration_overrides_sec": {
            str(k): float(v) for k, v in source_duration_overrides.items()
        },
        "source_epoch_count_raw": int(len(ordered_source)),
        "source_epoch_count_accepted": int(len(accepted_source)),
        "behavior_epoch_count_raw": int(len(behavior_qc)),
        "behavior_epoch_count_accepted": int(len(behavior)),
        "source_to_analysis_epoch": {str(k): int(v) for k, v in source_to_analysis.items()},
        "n_trials_total": int(lengths.size),
        "n_trials_retained": int(np.sum(trial_keep_mask)),
        "n_trials_rejected": int(np.sum(~trial_keep_mask)),
        "original_samples": int(np.sum(lengths)),
        "retained_samples": int(np.sum(retained_lengths)),
        "discarded_samples": int(np.sum(lengths) - np.sum(retained_lengths)),
        "warnings": warnings,
        "strict_epoch_match": bool(strict_epoch_match),
    }
    return EpochReconciliation(
        behavior_epoch_df=behavior,
        behavior_epoch_qc=behavior_qc,
        source_epoch_qc=source_qc,
        original_trial_lengths_samples=lengths.copy(),
        trial_lengths_samples=retained_lengths,
        source_trial_epoch=source_labels.astype(int),
        analysis_trial_epoch=analysis_labels.astype(int),
        trial_keep_mask=trial_keep_mask,
        metadata=metadata,
    )


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
    scale_each_epoch: ScaleMode = "never",
    scale_tolerance_sec: float = 0.050,
    strict_epoch_match: bool = True,
) -> EpochAwareTimebase:
    """Build a per-sample HARP timebase while preserving acquisition gaps.

    The current QC standard uses ``scale_each_epoch='never'``: nominal sample
    spacing is preserved, and :func:`reconcile_trial_epochs` clips excess samples
    to the common interval. ``'auto'`` and ``'always'`` remain available only for
    backward compatibility with older analyses.
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
    expected_order = [int(x) for x in epoch_ids if x in assigned_ids]
    if strict_epoch_match and slice_ids != expected_order:
        raise ValueError(
            f"Extracted epoch order {slice_ids} does not match behavior epoch order {expected_order}"
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
