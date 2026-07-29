"""Postprocess voltage-summary ROI traces for downstream analysis.

This module provides utilities for converting trial-wise SLAP2 voltage ROI traces
into session-wide arrays for QC and stimulus alignment.  The low-level voltage
reader, :class:`vip_slap2_analysis.voltage.summary.VoltageSummary`, exposes trace
blocks in a time-major convention: ``(n_samples, n_rois)``.  This module preserves
that convention for voltage I/O helpers and converts to the ROI-major convention
``(n_rois, n_samples)`` only when building a
:class:`vip_slap2_analysis.common.alignment.ReconstructedTraceBundle` for the
shared event-alignment machinery.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from vip_slap2_analysis.common.alignment import ReconstructedTraceBundle
from vip_slap2_analysis.common.epoch_alignment import build_epoch_aware_timebase, load_epoch_dataframe
from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.voltage.summary import VoltageSummary


TrialSlice = Tuple[int, slice]


def _valid_trials_for_dmd(vs, dmd: int) -> List[int]:
    """Return 1-indexed valid trials for ``dmd`` as plain Python integers."""
    dmd0 = int(dmd) - 1
    if dmd0 < 0 or dmd0 >= len(vs.valid_trials):
        raise IndexError(f"dmd must be in [1, {len(vs.valid_trials)}], got {dmd}")
    return [int(t) for t in vs.valid_trials[dmd0]]


def _expected_n_rois(vs, dmd: int) -> int:
    """Return the expected number of DMD-local voltage ROIs."""
    dmd0 = int(dmd) - 1
    if dmd0 < 0 or dmd0 >= len(vs.n_rois):
        raise IndexError(f"dmd must be in [1, {len(vs.n_rois)}], got {dmd}")
    return int(vs.n_rois[dmd0])


def _as_time_by_roi(
    x: np.ndarray,
    *,
    expected_n_rois: int,
    dmd: int,
    trial: int,
) -> np.ndarray:
    """Validate a voltage trace block as ``(n_samples, n_rois)``.

    ``VoltageSummary.get_roi_traces`` is expected to return time-major data.  If
    an ROI-major block is detected, raise immediately rather than silently
    corrupting trial slices or ROI traces.
    """
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D voltage traces for DMD {dmd}, trial {trial}; "
            f"got shape {arr.shape}."
        )

    if arr.shape[1] == expected_n_rois:
        return arr

    if arr.shape[0] == expected_n_rois and arr.shape[1] != expected_n_rois:
        raise ValueError(
            "Voltage trace orientation appears to be ROI x time, but the voltage "
            "pipeline requires time x ROI. Check VoltageSummary.get_roi_traces() "
            f"for DMD {dmd}, trial {trial}. Got shape {arr.shape}; expected "
            f"second dimension to equal n_rois={expected_n_rois}."
        )

    raise ValueError(
        f"Could not validate voltage trace shape for DMD {dmd}, trial {trial}. "
        f"Got shape {arr.shape}; expected (n_samples, {expected_n_rois})."
    )


def _discard_mask_for_trace(vs, *, dmd: int, trial: int, n_samples: int) -> np.ndarray:
    """Return an aligned boolean discard-frame mask for one trace block."""
    df = vs.get_discard_frames(dmd=dmd, trial=trial)
    df = np.asarray(df).astype(bool).squeeze()
    if df.ndim != 1:
        df = df.reshape(-1)
    if df.size == n_samples:
        return df
    if df.size > n_samples:
        return df[:n_samples]

    out = np.zeros(n_samples, dtype=bool)
    out[: df.size] = df
    return out


def concat_rois_across_trials(
    vs,
    dmd: int = 1,
    drop_discarded: bool = True,
    dtype=np.float32,
    *,
    return_array: bool = False,
    trace_mode: str = "trial",
):
    """Concatenate ROI traces across all valid trials for one DMD.

    Parameters
    ----------
    vs
        A :class:`vip_slap2_analysis.voltage.summary.VoltageSummary`-like object.
    dmd
        1-indexed DMD number.
    drop_discarded
        Remove samples marked by ``vs.get_discard_frames`` before concatenation.
        New split-H5 voltage outputs generally have all-False discard masks, but
        the option is retained for older formats.
    dtype
        Numpy dtype passed to ``vs.get_roi_traces``.
    return_array
        If False, return the historical list-of-ROIs representation.  If True,
        return a single time-major array shaped ``(n_total_samples, n_rois)``.
    trace_mode
        Trace mode passed to ``VoltageSummary.get_roi_traces``.  For current
        voltage outputs this should remain ``"trial"``.

    Returns
    -------
    roi_traces_or_array
        By default, a list with one 1D concatenated trace per ROI.  If
        ``return_array=True``, a 2D array shaped ``(n_total_samples, n_rois)``.
    trial_slices
        List of ``(trial_index_1based, slice_into_concatenated_time)`` entries.

    Notes
    -----
    The voltage I/O convention is time-major: ``(n_samples, n_rois)``.  This
    function preserves that convention when ``return_array=True``.
    """
    valid_trials = _valid_trials_for_dmd(vs, dmd)
    n_rois = _expected_n_rois(vs, dmd)

    chunks_by_roi: List[List[np.ndarray]] = [[] for _ in range(n_rois)]
    array_chunks: List[np.ndarray] = []
    trial_slices: List[TrialSlice] = []
    t_cursor = 0

    for trial in valid_trials:
        x = vs.get_roi_traces(
            dmd=dmd,
            trial=trial,
            drop_discarded=False,
            dtype=dtype,
            trace_mode=trace_mode,
        )
        x = _as_time_by_roi(x, expected_n_rois=n_rois, dmd=dmd, trial=trial)

        if drop_discarded:
            discard = _discard_mask_for_trace(
                vs, dmd=dmd, trial=trial, n_samples=x.shape[0]
            )
            x = x[~discard, :]

        seg_len = int(x.shape[0])
        trial_slices.append((int(trial), slice(t_cursor, t_cursor + seg_len)))
        t_cursor += seg_len

        if return_array:
            array_chunks.append(x.astype(dtype, copy=False))
        else:
            for roi_idx in range(n_rois):
                chunks_by_roi[roi_idx].append(x[:, roi_idx].astype(dtype, copy=False))

    if return_array:
        if array_chunks:
            return np.concatenate(array_chunks, axis=0), trial_slices
        return np.empty((0, n_rois), dtype=dtype), trial_slices

    roi_traces = [
        np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=dtype)
        for chunks in chunks_by_roi
    ]
    return roi_traces, trial_slices



def _normalize_trace_mode_for_lengths(vs, trace_mode: str) -> str:
    """Resolve ``auto`` to the concrete split-H5 trace mode without loading traces."""
    mode = str(trace_mode or "auto").lower()
    if mode != "auto":
        return mode
    try:
        modes = vs.available_trace_modes()
        if "trial" in modes:
            return "trial"
        if "continuous" in modes:
            return "continuous"
        return "epoch_continuous" if "epoch_continuous" in modes else mode
    except Exception:
        return mode


def _trace_time_len_for_trial(vs, *, dmd: int, trial: int, trace_mode: str) -> int:
    """Return trace length along time without materializing the trace when possible."""
    mode = _normalize_trace_mode_for_lengths(vs, trace_mode)
    if getattr(vs, "layout", None) == "split_h5" or getattr(vs, "_summary_layout", None) == "split_h5":
        ds = vs.get_trace_dataset(dmd=dmd, trial=trial, trace_mode=mode)
        n_expected = int(vs.n_total_rois) if mode == "trial" else int(vs.n_rois[int(dmd) - 1])
        return int(vs._infer_time_len_from_split_dataset(ds, n_expected))

    # Fallback for older summary layouts.  These are usually much smaller, and
    # this path preserves backward compatibility where no lazy dataset handle is
    # available.
    x = vs.get_roi_traces(dmd=dmd, trial=trial, dtype=None, trace_mode=trace_mode)
    x = _as_time_by_roi(x, expected_n_rois=_expected_n_rois(vs, dmd), dmd=dmd, trial=trial)
    return int(x.shape[0])


def _allocate_roi_time_array(
    *,
    shape: Tuple[int, int],
    dtype,
    fill_value: float = np.nan,
    memmap_threshold_bytes: int = 512 * 1024 ** 2,
    prefix: str = "vip_slap2_voltage_traces_",
) -> np.ndarray:
    """Allocate an ROI-by-time array, spilling large arrays to disk-backed memmap."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if nbytes >= int(memmap_threshold_bytes):
        path = Path(tempfile.gettempdir()) / f"{prefix}{uuid.uuid4().hex}.npy"
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))
        arr[:] = fill_value
        return arr
    return np.full(tuple(shape), fill_value, dtype=dtype)

def _load_valid_voltage_trials(
    vs,
    *,
    dmd: int,
    drop_discarded: bool,
    dtype,
    trace_mode: str,
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """Load valid trial traces as time-major arrays and return valid lengths."""
    valid_trials = set(_valid_trials_for_dmd(vs, dmd))
    n_rois = _expected_n_rois(vs, dmd)
    trial_data: Dict[int, np.ndarray] = {}
    valid_lengths: List[int] = []

    for trial in range(1, int(vs.n_trials) + 1):
        if trial not in valid_trials:
            continue

        x = vs.get_roi_traces(
            dmd=dmd,
            trial=trial,
            drop_discarded=False,
            dtype=dtype,
            trace_mode=trace_mode,
        )
        x = _as_time_by_roi(x, expected_n_rois=n_rois, dmd=dmd, trial=trial)

        if drop_discarded:
            discard = _discard_mask_for_trace(
                vs, dmd=dmd, trial=trial, n_samples=x.shape[0]
            )
            x = x[~discard, :]

        x = x.astype(dtype, copy=False)
        trial_data[trial] = x
        valid_lengths.append(int(x.shape[0]))

    return trial_data, valid_lengths


def _build_voltage_alignment_timebase(
    *,
    total_samples: int,
    sample_rate_hz: float,
    epoch_start_sec: float,
    epoch_end_sec: Optional[float],
    strategy: str = "auto",
    max_timebase_error_sec: float = 0.5,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Build the behavior-time vector used for voltage event alignment.

    ``sample_rate`` preserves the physical/metadata sample spacing.
    ``epoch_scaled`` maps the reconstructed sample stream exactly onto the
    behavior/imaging epoch.  ``auto`` uses ``epoch_scaled`` only when the nominal
    reconstructed duration and behavior epoch duration disagree by more than
    ``max_timebase_error_sec``.  This protects trial-based acquisitions from
    progressive stimulus-alignment drift while preserving the historical fixed-rate
    behavior for sessions whose metadata and behavior epoch already agree.
    """
    total_samples = int(total_samples)
    rate = float(sample_rate_hz)
    start = float(epoch_start_sec)
    if total_samples <= 0:
        return np.empty((0,), dtype=float), rate, {
            "timebase_strategy_requested": str(strategy),
            "timebase_strategy_used": "empty",
            "nominal_sample_rate_hz": rate,
            "alignment_sample_rate_hz": rate,
            "nominal_duration_sec": 0.0,
            "epoch_duration_sec": None,
            "duration_error_sec": None,
        }

    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"sample_rate_hz must be positive and finite, got {sample_rate_hz}.")

    requested = str(strategy or "auto").lower().replace("-", "_")
    aliases = {
        "nominal": "sample_rate",
        "fixed_rate": "sample_rate",
        "metadata": "sample_rate",
        "scale_to_epoch": "epoch_scaled",
        "scaled_epoch": "epoch_scaled",
        "behavior_epoch": "epoch_scaled",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "sample_rate", "epoch_scaled"}:
        raise ValueError(
            "timebase_strategy must be one of 'auto', 'sample_rate', or "
            f"'epoch_scaled', got {strategy!r}."
        )

    nominal_duration = float(total_samples / rate)
    epoch_duration: Optional[float] = None
    if epoch_end_sec is not None:
        epoch_duration = float(epoch_end_sec) - start
        if not np.isfinite(epoch_duration) or epoch_duration <= 0:
            epoch_duration = None

    used = requested
    if requested == "auto":
        used = "sample_rate"
        if epoch_duration is not None:
            if abs(nominal_duration - epoch_duration) > float(max_timebase_error_sec):
                used = "epoch_scaled"

    if used == "epoch_scaled":
        if epoch_duration is None:
            used = "sample_rate"
            alignment_rate = rate
            timebase = start + np.arange(total_samples, dtype=float) / alignment_rate
        else:
            alignment_rate = float(total_samples / epoch_duration)
            timebase = start + np.arange(total_samples, dtype=float) / alignment_rate
    else:
        alignment_rate = rate
        timebase = start + np.arange(total_samples, dtype=float) / alignment_rate

    info: Dict[str, Any] = {
        "timebase_strategy_requested": str(strategy),
        "timebase_strategy_used": used,
        "nominal_sample_rate_hz": rate,
        "alignment_sample_rate_hz": float(alignment_rate),
        "nominal_duration_sec": nominal_duration,
        "epoch_duration_sec": epoch_duration,
        "duration_error_sec": (
            float(nominal_duration - epoch_duration) if epoch_duration is not None else None
        ),
        "max_timebase_error_sec": float(max_timebase_error_sec),
    }
    return timebase, float(alignment_rate), info


def _expected_trial_lengths_from_summary(vs, dmd: int, n_trials: int) -> np.ndarray:
    """Use extractor line-range metadata for missing/empty trial placeholders."""
    out = np.zeros((int(n_trials),), dtype=int)
    try:
        ranges = vs.get_trial_line_ranges(dmd=dmd)
    except Exception:
        return out
    for key in ("nLines", "trialGlobalNLines"):
        if key not in ranges:
            continue
        arr = np.asarray(ranges[key], dtype=float).squeeze()
        if arr.ndim > 1:
            arr = np.ravel(arr)
        n = min(out.size, arr.size)
        vals = np.rint(arr[:n]).astype(int)
        vals[~np.isfinite(arr[:n])] = 0
        out[:n] = np.maximum(vals, 0)
        if np.any(out > 0):
            return out
    return out


def reconstruct_voltage_dmd_session_traces(
    vs,
    dmd: int,
    *,
    sample_rate_hz: float,
    epoch_start_sec: float,
    epoch_end_sec: Optional[float] = None,
    epoch_df: Optional[pd.DataFrame] = None,
    trial_epoch: Optional[np.ndarray] = None,
    drop_discarded: bool = True,
    dtype=np.float32,
    trace_mode: str = "trial",
    timebase_strategy: str = "auto",
    max_timebase_error_sec: float = 0.5,
    strict_epoch_match: bool = True,
) -> ReconstructedTraceBundle:
    """Reconstruct one DMD's voltage traces as a session-wide alignment bundle.

    Parameters
    ----------
    vs
        A :class:`vip_slap2_analysis.voltage.summary.VoltageSummary`-like object.
    dmd
        1-indexed DMD number.
    sample_rate_hz
        Voltage sample/line rate in hertz.  For current SLAP2 integration-mode
        voltage imaging this is expected to be approximately 10.8 kHz.
    epoch_start_sec
        Corrected behavior/HARP time corresponding to sample 0 of the first trial
        in the reconstructed voltage stream.  In the current pipeline this should
        be the first ``imaging_epochs.csv`` ``start_time``.
    epoch_end_sec
        Optional corrected behavior/HARP time corresponding to the end of the
        imaging epoch.  When provided, ``timebase_strategy='auto'`` can scale the
        voltage timebase to this duration if the nominal sample-rate duration
        would otherwise drift relative to behavior time.
    drop_discarded
        Remove samples marked by ``discardFrames`` before reconstruction.
    dtype
        Output dtype for loaded voltage traces.
    trace_mode
        Trace mode passed to ``VoltageSummary.get_roi_traces``.  Current voltage
        outputs are trial-based, so the default is ``"trial"``.
    timebase_strategy
        ``"sample_rate"`` preserves nominal sample spacing. ``"epoch_scaled"``
        maps the reconstructed trace onto ``epoch_start_sec`` → ``epoch_end_sec``.
        ``"auto"`` uses epoch scaling only when the nominal duration and behavior
        epoch differ by more than ``max_timebase_error_sec``.
    max_timebase_error_sec
        Duration mismatch threshold used by ``timebase_strategy='auto'``.

    Returns
    -------
    ReconstructedTraceBundle
        Bundle with ROI-major traces shaped ``(n_rois, n_total_samples)`` and an
        explicit per-sample timebase.  Invalid trials are represented by NaN
        blocks of the median valid trial length, mirroring the glutamate pipeline.

    Notes
    -----
    This function is the boundary between voltage I/O and shared event alignment:
    voltage blocks are read as ``(n_samples, n_rois)``, then transposed exactly
    once into ``(n_rois, n_total_samples)`` for downstream alignment helpers.
    """
    sample_rate_hz = float(sample_rate_hz)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(
            f"sample_rate_hz must be positive and finite, got {sample_rate_hz}."
        )

    n_trials = int(vs.n_trials)
    n_rois = _expected_n_rois(vs, dmd)
    valid_trials = set(_valid_trials_for_dmd(vs, dmd))
    valid_lengths_by_trial: Dict[int, int] = {}
    valid_lengths: List[int] = []

    for trial in sorted(valid_trials):
        n_raw = _trace_time_len_for_trial(vs, dmd=dmd, trial=trial, trace_mode=trace_mode)
        if drop_discarded:
            discard = _discard_mask_for_trace(vs, dmd=dmd, trial=trial, n_samples=n_raw)
            n_kept = int(np.sum(~discard))
        else:
            n_kept = int(n_raw)
        valid_lengths_by_trial[int(trial)] = n_kept
        valid_lengths.append(n_kept)

    if not valid_lengths:
        return ReconstructedTraceBundle(
            traces=np.empty((n_rois, 0), dtype=dtype),
            timebase_sec=np.empty((0,), dtype=float),
            trial_valid_mask=np.zeros((n_trials,), dtype=bool),
            trial_lengths_samples=np.zeros((n_trials,), dtype=int),
            trial_starts_sec=np.zeros((n_trials,), dtype=float),
            session_start_sec=float(epoch_start_sec),
            session_end_sec=float(epoch_start_sec),
            reconstructed_duration_sec=0.0,
            metadata={"timebase_strategy_used": "empty"},
        )

    default_len = int(round(float(np.median(valid_lengths))))
    trial_lengths = _expected_trial_lengths_from_summary(vs, dmd, n_trials)
    trial_lengths[trial_lengths <= 0] = default_len
    for trial, length in valid_lengths_by_trial.items():
        trial_lengths[trial - 1] = int(length)

    total_samples = int(np.sum(trial_lengths))
    traces = _allocate_roi_time_array(shape=(n_rois, total_samples), dtype=dtype)
    trial_valid_mask = np.zeros((n_trials,), dtype=bool)
    trial_starts_sec = np.zeros((n_trials,), dtype=float)

    if trial_epoch is None and hasattr(vs, "trial_epoch"):
        try:
            trial_epoch = np.asarray(vs.trial_epoch, dtype=int)
        except Exception:
            trial_epoch = None

    sample_epoch = None
    if epoch_df is not None and len(epoch_df) > 0:
        if len(epoch_df) > 1 and trial_epoch is None and strict_epoch_match:
            raise ValueError(
                "Multi-epoch voltage reconstruction requires extractor trialEpoch metadata. "
                "Re-run the patched extractDendrites.m before downstream processing."
            )
        strategy_norm = str(timebase_strategy or "auto").lower().replace("-", "_")
        scale_mode = {"sample_rate": "never", "nominal": "never", "epoch_scaled": "always"}.get(strategy_norm, "auto")
        epoch_tb = build_epoch_aware_timebase(
            trial_lengths,
            sample_rate_hz=sample_rate_hz,
            epoch_df=epoch_df,
            trial_epoch=trial_epoch,
            scale_each_epoch=scale_mode,
            scale_tolerance_sec=max_timebase_error_sec,
            strict_epoch_match=strict_epoch_match,
        )
        timebase_sec = epoch_tb.timebase_sec
        sample_epoch = epoch_tb.sample_epoch
        trial_epoch = epoch_tb.trial_epoch
        alignment_rate_hz = sample_rate_hz
        timebase_meta = dict(epoch_tb.metadata)
        timebase_meta["timebase_strategy_used"] = "epoch_aware" if len(epoch_df) > 1 else "epoch_scaled_single"
    else:
        timebase_sec, alignment_rate_hz, timebase_meta = _build_voltage_alignment_timebase(
            total_samples=total_samples,
            sample_rate_hz=sample_rate_hz,
            epoch_start_sec=epoch_start_sec,
            epoch_end_sec=epoch_end_sec,
            strategy=timebase_strategy,
            max_timebase_error_sec=max_timebase_error_sec,
        )

    pos = 0
    for trial in range(1, n_trials + 1):
        length = int(trial_lengths[trial - 1])
        if total_samples and pos < timebase_sec.size:
            trial_starts_sec[trial - 1] = float(timebase_sec[pos])
        else:
            trial_starts_sec[trial - 1] = float(epoch_start_sec + pos / alignment_rate_hz)

        if trial in valid_lengths_by_trial:
            n_raw = _trace_time_len_for_trial(vs, dmd=dmd, trial=trial, trace_mode=trace_mode)
            if drop_discarded:
                discard = _discard_mask_for_trace(vs, dmd=dmd, trial=trial, n_samples=n_raw)
            else:
                discard = None

            dst_cursor = 0
            chunk_samples = 262_144
            for src_start in range(0, int(n_raw), chunk_samples):
                src_stop = min(int(n_raw), src_start + chunk_samples)
                x = vs.get_roi_traces(
                    dmd=dmd,
                    trial=trial,
                    t_slice=slice(src_start, src_stop),
                    drop_discarded=False,
                    dtype=dtype,
                    trace_mode=trace_mode,
                )
                x = _as_time_by_roi(x, expected_n_rois=n_rois, dmd=dmd, trial=trial)
                if discard is not None:
                    keep = ~discard[src_start:src_stop]
                    if keep.size != x.shape[0]:
                        aligned_keep = np.zeros((x.shape[0],), dtype=bool)
                        n_copy = min(int(keep.size), int(x.shape[0]))
                        aligned_keep[:n_copy] = keep[:n_copy]
                        keep = aligned_keep
                    x = x[keep, :]
                if x.size == 0:
                    continue
                n_time = min(int(x.shape[0]), length - dst_cursor)
                if n_time <= 0:
                    break
                n_roi = min(n_rois, int(x.shape[1]))
                traces[:n_roi, pos + dst_cursor:pos + dst_cursor + n_time] = x[:n_time, :n_roi].T
                dst_cursor += n_time
            trial_valid_mask[trial - 1] = True

        pos += length

    session_end_sec = float(timebase_sec[-1]) if total_samples else float(epoch_start_sec)
    if epoch_df is not None and len(epoch_df) > 0:
        session_end_sec = float(epoch_df["end_time"].iloc[-1])
        reconstructed_duration_sec = float(session_end_sec - float(epoch_df["start_time"].iloc[0]))
    elif timebase_meta["timebase_strategy_used"] == "epoch_scaled" and epoch_end_sec is not None:
        session_end_sec = float(epoch_end_sec)
        reconstructed_duration_sec = float(float(epoch_end_sec) - float(epoch_start_sec))
    else:
        reconstructed_duration_sec = float(total_samples / alignment_rate_hz)

    metadata: Dict[str, Any] = dict(timebase_meta)
    metadata.update({
        "trace_mode": str(trace_mode),
        "drop_discarded": bool(drop_discarded),
        "n_trials_total": int(n_trials),
        "n_trials_valid": int(np.sum(trial_valid_mask)),
        "n_samples_total": int(total_samples),
        "trial_epoch": None if trial_epoch is None else np.asarray(trial_epoch, dtype=int).tolist(),
        "strict_epoch_match": bool(strict_epoch_match),
        "invalid_trial_length_source": "summary_line_ranges_then_median_fallback",
        "traces_storage": "memmap" if isinstance(traces, np.memmap) else "memory",
        "traces_memmap_path": str(getattr(traces, "filename", "")) if isinstance(traces, np.memmap) else None,
    })

    return ReconstructedTraceBundle(
        traces=traces,
        timebase_sec=timebase_sec,
        trial_valid_mask=trial_valid_mask,
        trial_lengths_samples=trial_lengths,
        trial_starts_sec=trial_starts_sec,
        session_start_sec=float(epoch_start_sec),
        session_end_sec=session_end_sec,
        reconstructed_duration_sec=reconstructed_duration_sec,
        metadata=metadata,
        sample_epoch=sample_epoch,
    )


def load_voltage_summary_from_asset(
    asset: SessionAssets,
    *,
    keep_open: bool = True,
    swap_xy_images: bool = True,
) -> VoltageSummary:
    """Load a :class:`VoltageSummary` from generic session assets.

    The voltage session asset model stores the split extraction outputs as
    ``asset.modality_assets["voltage"]`` rather than as voltage-specific
    dataclass fields.  This helper is the canonical bridge between the generic
    asset resolver and the voltage reader.

    Parameters
    ----------
    asset
        Session asset bundle returned by ``VIPSessionRegistry.resolve_assets``.
    keep_open
        Keep the underlying MAT/H5 file handles open between reader calls.
    swap_xy_images
        Passed through to :class:`VoltageSummary` for display-oriented image and
        mask access.

    Returns
    -------
    VoltageSummary
        Reader initialized with the required voltage summary MAT file and, when
        available, the paired trace H5 file resolved by the session registry.
    """
    summary_mat = asset.require_asset("voltage", "summary_mat")
    trace_h5 = asset.get_asset("voltage", "trace_h5")
    return VoltageSummary(
        summary_mat,
        trace_path=trace_h5,
        keep_open=keep_open,
        swap_xy_images=swap_xy_images,
    )


def resolve_voltage_sample_rate_hz(
    *,
    asset: Optional[SessionAssets] = None,
    vs: Optional[VoltageSummary] = None,
    sample_rate_hz: Optional[float] = None,
    default_hz: float = 10_800.0,
) -> float:
    """Resolve the voltage sample/line rate used for timebase construction.

    Current SLAP2 integration-mode voltage recordings use the line-scan rate as
    the effective voltage sample rate, approximately 10.8 kHz.  The explicit
    ``sample_rate_hz`` argument takes precedence so notebooks can override stale
    or ambiguous metadata.  If no explicit value is supplied, the function checks
    common metadata keys and then falls back to ``default_hz``.
    """
    if sample_rate_hz is not None:
        rate = float(sample_rate_hz)
        if np.isfinite(rate) and rate > 0:
            return rate
        raise ValueError(f"sample_rate_hz must be positive and finite, got {sample_rate_hz}.")

    candidate_keys = (
        "voltage_sample_rate_hz",
        "sample_rate_hz",
        "line_rate_hz",
        "lineRateHz",
        "slap2_line_rate_hz",
        "fs_hz",
        "fs_Hz",
    )

    # Prefer per-DMD SLAP2 metadata when it is available.  Older notebooks often
    # carried a 10.8 kHz default in asset metadata, but recent extraction outputs
    # store the actual line-scan rate under summary/dmd/metadata/lineRateHz.
    if vs is not None and hasattr(vs, "get_line_rate_hz"):
        try:
            rate = float(vs.get_line_rate_hz())
        except Exception:
            rate = float("nan")
        if np.isfinite(rate) and rate > 0:
            return rate

    if vs is not None:
        for mapping in (getattr(vs, "h5_attrs", {}), getattr(vs, "metadata", {})):
            if not isinstance(mapping, dict):
                continue
            for key in candidate_keys:
                if key in mapping and mapping[key] is not None:
                    try:
                        rate = float(mapping[key])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(rate) and rate > 0:
                        return rate

    if asset is not None and getattr(asset, "metadata", None):
        for key in candidate_keys:
            if key in asset.metadata and asset.metadata[key] is not None:
                try:
                    rate = float(asset.metadata[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(rate) and rate > 0:
                    return rate

    rate = float(default_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"default_hz must be positive and finite, got {default_hz}.")
    return rate


def _resolve_imaging_epoch_bounds(
    *,
    epoch_start_sec: Optional[float],
    epoch_end_sec: Optional[float],
    imaging_epochs_csv: Optional[Union[str, Path]],
    asset: Optional[SessionAssets],
) -> Tuple[float, Optional[float]]:
    """Resolve corrected-time imaging-epoch bounds for voltage sample mapping."""
    if epoch_start_sec is not None and epoch_end_sec is not None:
        return float(epoch_start_sec), float(epoch_end_sec)

    candidate: Optional[Path] = None
    if imaging_epochs_csv is not None:
        candidate = Path(imaging_epochs_csv)
    elif asset is not None and asset.qc_dir is not None:
        p = Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv"
        if p.exists():
            candidate = p

    if candidate is None:
        if epoch_start_sec is not None:
            return float(epoch_start_sec), (float(epoch_end_sec) if epoch_end_sec is not None else None)
        raise ValueError(
            "epoch_start_sec was not provided and imaging_epochs.csv could not be "
            "resolved from asset.qc_dir / 'behavior'."
        )

    import pandas as pd

    epochs = pd.read_csv(candidate)
    if "start_time" not in epochs.columns or len(epochs) == 0:
        raise ValueError(f"{candidate} must contain at least one start_time value.")
    start = float(epoch_start_sec) if epoch_start_sec is not None else float(epochs["start_time"].iloc[0])
    end: Optional[float]
    if epoch_end_sec is not None:
        end = float(epoch_end_sec)
    elif "end_time" in epochs.columns and len(epochs) > 0:
        end = float(epochs["end_time"].iloc[-1])
    else:
        end = None
    return start, end


def _resolve_epoch_start_sec(
    *,
    epoch_start_sec: Optional[float],
    imaging_epochs_csv: Optional[Union[str, Path]],
    asset: Optional[SessionAssets],
) -> float:
    """Backward-compatible helper returning only the first imaging-epoch start."""
    start, _ = _resolve_imaging_epoch_bounds(
        epoch_start_sec=epoch_start_sec,
        epoch_end_sec=None,
        imaging_epochs_csv=imaging_epochs_csv,
        asset=asset,
    )
    return start


def reconstruct_voltage_dmd_session_traces_from_asset(
    asset: SessionAssets,
    dmd: int,
    *,
    sample_rate_hz: Optional[float] = None,
    default_sample_rate_hz: float = 10_800.0,
    epoch_start_sec: Optional[float] = None,
    epoch_end_sec: Optional[float] = None,
    imaging_epochs_csv: Optional[Union[str, Path]] = None,
    drop_discarded: bool = True,
    dtype=np.float32,
    trace_mode: str = "trial",
    timebase_strategy: str = "auto",
    max_timebase_error_sec: float = 0.5,
    strict_epoch_match: bool = True,
) -> ReconstructedTraceBundle:
    """Load voltage assets and reconstruct one DMD as an alignment bundle.

    This is the asset-oriented wrapper that downstream QC/extraction notebooks can
    call after ``VIPSessionRegistry.resolve_assets``.  It keeps all path discovery
    in the common/io asset framework while delegating trace orientation and
    concatenation to :func:`reconstruct_voltage_dmd_session_traces`.
    """
    with load_voltage_summary_from_asset(asset, keep_open=True) as vs:
        rate = resolve_voltage_sample_rate_hz(
            asset=asset,
            vs=vs,
            sample_rate_hz=sample_rate_hz,
            default_hz=default_sample_rate_hz,
        )
        start, end = _resolve_imaging_epoch_bounds(
            epoch_start_sec=epoch_start_sec,
            epoch_end_sec=epoch_end_sec,
            imaging_epochs_csv=imaging_epochs_csv,
            asset=asset,
        )
        epoch_df = None
        candidate = Path(imaging_epochs_csv) if imaging_epochs_csv is not None else (Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv" if asset.qc_dir is not None else None)
        if candidate is not None and Path(candidate).exists():
            epoch_df = pd.read_csv(candidate)
        return reconstruct_voltage_dmd_session_traces(
            vs,
            dmd=dmd,
            sample_rate_hz=rate,
            epoch_start_sec=start,
            epoch_end_sec=end,
            epoch_df=epoch_df,
            drop_discarded=drop_discarded,
            dtype=dtype,
            trace_mode=trace_mode,
            timebase_strategy=timebase_strategy,
            max_timebase_error_sec=max_timebase_error_sec,
            strict_epoch_match=strict_epoch_match,
        )
