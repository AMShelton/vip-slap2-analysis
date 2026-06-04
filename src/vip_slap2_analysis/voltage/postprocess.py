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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from vip_slap2_analysis.common.alignment import ReconstructedTraceBundle
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


def reconstruct_voltage_dmd_session_traces(
    vs,
    dmd: int,
    *,
    sample_rate_hz: float,
    epoch_start_sec: float,
    drop_discarded: bool = True,
    dtype=np.float32,
    trace_mode: str = "trial",
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
    drop_discarded
        Remove samples marked by ``discardFrames`` before reconstruction.
    dtype
        Output dtype for loaded voltage traces.
    trace_mode
        Trace mode passed to ``VoltageSummary.get_roi_traces``.  Current voltage
        outputs are trial-based, so the default is ``"trial"``.

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
    trial_data, valid_lengths = _load_valid_voltage_trials(
        vs,
        dmd=dmd,
        drop_discarded=drop_discarded,
        dtype=dtype,
        trace_mode=trace_mode,
    )

    if not trial_data:
        return ReconstructedTraceBundle(
            traces=np.empty((n_rois, 0), dtype=dtype),
            timebase_sec=np.empty((0,), dtype=float),
            trial_valid_mask=np.zeros((n_trials,), dtype=bool),
            trial_lengths_samples=np.zeros((n_trials,), dtype=int),
            trial_starts_sec=np.zeros((n_trials,), dtype=float),
            session_start_sec=float(epoch_start_sec),
            session_end_sec=float(epoch_start_sec),
            reconstructed_duration_sec=0.0,
        )

    default_len = int(round(float(np.median(valid_lengths))))
    trial_lengths = np.full((n_trials,), default_len, dtype=int)
    for trial, x in trial_data.items():
        trial_lengths[trial - 1] = int(x.shape[0])

    total_samples = int(np.sum(trial_lengths))
    traces = np.full((n_rois, total_samples), np.nan, dtype=dtype)
    trial_valid_mask = np.zeros((n_trials,), dtype=bool)
    trial_starts_sec = np.zeros((n_trials,), dtype=float)

    pos = 0
    for trial in range(1, n_trials + 1):
        length = int(trial_lengths[trial - 1])
        trial_starts_sec[trial - 1] = float(epoch_start_sec + pos / sample_rate_hz)

        if trial in trial_data:
            x = trial_data[trial]
            n_time = min(length, int(x.shape[0]))
            n_roi = min(n_rois, int(x.shape[1]))
            traces[:n_roi, pos:pos + n_time] = x[:n_time, :n_roi].T
            trial_valid_mask[trial - 1] = True

        pos += length

    timebase_sec = (
        float(epoch_start_sec)
        + np.arange(total_samples, dtype=float) / sample_rate_hz
    )
    session_end_sec = float(timebase_sec[-1]) if total_samples else float(epoch_start_sec)

    return ReconstructedTraceBundle(
        traces=traces,
        timebase_sec=timebase_sec,
        trial_valid_mask=trial_valid_mask,
        trial_lengths_samples=trial_lengths,
        trial_starts_sec=trial_starts_sec,
        session_start_sec=float(epoch_start_sec),
        session_end_sec=session_end_sec,
        reconstructed_duration_sec=float(total_samples / sample_rate_hz),
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
        "slap2_line_rate_hz",
        "fs_hz",
        "fs_Hz",
    )

    if asset is not None and getattr(asset, "metadata", None):
        for key in candidate_keys:
            if key in asset.metadata and asset.metadata[key] is not None:
                try:
                    rate = float(asset.metadata[key])
                except (TypeError, ValueError):
                    continue
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

    rate = float(default_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"default_hz must be positive and finite, got {default_hz}.")
    return rate


def _resolve_epoch_start_sec(
    *,
    epoch_start_sec: Optional[float],
    imaging_epochs_csv: Optional[Union[str, Path]],
    asset: Optional[SessionAssets],
) -> float:
    """Resolve the corrected-time start of voltage sample 0."""
    if epoch_start_sec is not None:
        return float(epoch_start_sec)

    candidate: Optional[Path] = None
    if imaging_epochs_csv is not None:
        candidate = Path(imaging_epochs_csv)
    elif asset is not None and asset.qc_dir is not None:
        p = Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv"
        if p.exists():
            candidate = p

    if candidate is None:
        raise ValueError(
            "epoch_start_sec was not provided and imaging_epochs.csv could not be "
            "resolved from asset.qc_dir / 'behavior'."
        )

    import pandas as pd

    epochs = pd.read_csv(candidate)
    if "start_time" not in epochs.columns or len(epochs) == 0:
        raise ValueError(f"{candidate} must contain at least one start_time value.")
    return float(epochs["start_time"].iloc[0])


def reconstruct_voltage_dmd_session_traces_from_asset(
    asset: SessionAssets,
    dmd: int,
    *,
    sample_rate_hz: Optional[float] = 10_800.0,
    default_sample_rate_hz: float = 10_800.0,
    epoch_start_sec: Optional[float] = None,
    imaging_epochs_csv: Optional[Union[str, Path]] = None,
    drop_discarded: bool = True,
    dtype=np.float32,
    trace_mode: str = "trial",
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
        start = _resolve_epoch_start_sec(
            epoch_start_sec=epoch_start_sec,
            imaging_epochs_csv=imaging_epochs_csv,
            asset=asset,
        )
        return reconstruct_voltage_dmd_session_traces(
            vs,
            dmd=dmd,
            sample_rate_hz=rate,
            epoch_start_sec=start,
            drop_discarded=drop_discarded,
            dtype=dtype,
            trace_mode=trace_mode,
        )
