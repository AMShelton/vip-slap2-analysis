"""Event-aligned extraction for SLAP2 dendritic voltage traces.

This module mirrors the glutamate/calcium event-extraction pathway while keeping
voltage-specific constraints explicit:

* voltage trace blocks are loaded through the generic :class:`SessionAssets`
  modality asset dictionary;
* trial-wise voltage traces are reconstructed as ROI-major session bundles for
  the shared alignment/event code;
* large single-trial event tensors are written to chunked HDF5 rather than
  compressed NPZ; and
* the default extracted signal is raw fluorescence so baseline/F0 choices do not
  accidentally remove slow voltage dynamics.  A conservative static-F0 inverted
  dF/F transform is available, but rolling-baseline transforms are intentionally
  not baked into this extraction step.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.common.alignment import (
    EventWindows,
    ReconstructedTraceBundle,
    build_change_locked_sequences,
    extract_change_intervals,
    extract_image_intervals,
    extract_omission_intervals,
    extract_ordered_change_targets,
    filter_intervals_to_epochs,
    filter_ordered_images_to_epochs,
    load_corrected_bonsai_csv,
    load_imaging_epochs_csv,
)
from vip_slap2_analysis.voltage.postprocess import (
    load_voltage_summary_from_asset,
    reconstruct_voltage_dmd_session_traces,
    resolve_voltage_sample_rate_hz,
)


PathLike = Union[str, Path]
Interval = Tuple[float, float]
StimIntervalList = List[Interval]
StimIntervalDict = Dict[str, StimIntervalList]


# -----------------------------------------------------------------------------
# Small generic helpers
# -----------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy/path scalar metadata."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _write_json(path: PathLike, payload: Mapping[str, Any]) -> None:
    """Write JSON metadata with numpy/path-friendly scalar conversion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _metadata_json_attr(group: Union[h5py.File, h5py.Group], payload: Mapping[str, Any]) -> None:
    """Store a JSON-encoded metadata dictionary as an HDF5 attribute."""
    group.attrs["metadata_json"] = json.dumps(payload, default=_json_default)


def _as_float_tuple(x: Sequence[float]) -> Tuple[float, float]:
    """Validate and coerce a two-element window tuple."""
    if len(x) != 2:
        raise ValueError(f"Expected a two-element window tuple, got {x!r}")
    return float(x[0]), float(x[1])


def _time_vectors(windows: EventWindows, sample_rate_hz: float) -> Dict[str, np.ndarray]:
    """Construct event-relative time vectors for voltage event windows."""
    return {
        "image": np.arange(-windows.image[0], windows.image[1], 1.0 / sample_rate_hz),
        "change": np.arange(-windows.change[0], windows.change[1], 1.0 / sample_rate_hz),
        "omission": np.arange(-windows.omission[0], windows.omission[1], 1.0 / sample_rate_hz),
    }


def _trace_suffix(trace_signal: str, trace_mode: str) -> str:
    """Build a stable filename suffix for the voltage trace variant."""
    signal = str(trace_signal).lower().replace(" ", "_")
    mode = str(trace_mode).lower().replace(" ", "_")
    return f"{signal}_{mode}"


def _modality_dir(asset: SessionAssets, family: str, modality: str) -> Path:
    """Return a modality-specific qc/derived directory with backward fallback."""
    if family == "derived" and hasattr(asset, "derived_subdir"):
        return asset.derived_subdir(modality, create=True)  # type: ignore[attr-defined]
    if family == "qc" and hasattr(asset, "qc_subdir"):
        return asset.qc_subdir(modality, create=True)  # type: ignore[attr-defined]

    base = getattr(asset, f"{family}_dir", None)
    if base is None:
        raise ValueError(f"asset.{family}_dir must be set")
    out = Path(base) / modality
    out.mkdir(parents=True, exist_ok=True)
    return out


def _open_imaging_epochs(asset: SessionAssets):
    """Load behavior QC imaging epochs for the current session."""
    if asset.qc_dir is None:
        raise ValueError("asset.qc_dir is required to load behavior/imaging_epochs.csv")
    return load_imaging_epochs_csv(Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv")


def _load_voltage_roi_qc_mask(asset: SessionAssets, dmd: int) -> Optional[np.ndarray]:
    """Load optional first-pass voltage ROI keep mask from voltage QC outputs."""
    if asset.qc_dir is None:
        return None
    qdir = Path(asset.qc_dir) / "voltage"
    candidates = [
        qdir / f"dmd{dmd}_recommended_voltage_rois.npy",
        qdir / f"valid_voltage_rois_dmd{dmd}.npy",
        qdir / f"dmd{dmd}_valid_voltage_rois.npy",
    ]
    for p in candidates:
        if p.exists():
            return np.asarray(np.load(p), dtype=bool).reshape(-1)
    return None


def _roi_ids(dmd: int, n_rois_total: int, mask: np.ndarray) -> np.ndarray:
    """Return stable per-session ROI IDs after masking."""
    all_ids = np.array([f"DMD{dmd}_roi{i:04d}" for i in range(n_rois_total)], dtype=object)
    return all_ids[mask]


def _validate_or_fill_mask(mask: Optional[np.ndarray], n_rois: int) -> np.ndarray:
    """Return a boolean ROI mask of length ``n_rois``."""
    if mask is None:
        return np.ones((n_rois,), dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size != n_rois:
        raise ValueError(f"Voltage ROI QC mask length {mask.size} does not match n_rois={n_rois}.")
    return mask


# -----------------------------------------------------------------------------
# Voltage F0 / dF/F transforms
# -----------------------------------------------------------------------------


def _as_trace_2d(raw_f: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return raw fluorescence as ROI-by-time and whether the input was 1D."""
    arr = np.asarray(raw_f, dtype=np.float32)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected raw_f to be 1D or 2D ROI-by-time, got shape {arr.shape}")
    return arr, was_1d


def _restore_trace_dim(arr: np.ndarray, was_1d: bool) -> np.ndarray:
    """Restore a transformed trace to the caller's original dimensionality."""
    if was_1d:
        return np.asarray(arr[0], dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)


def _safe_f0_values(f0: np.ndarray, raw_f: np.ndarray) -> np.ndarray:
    """Guard against invalid or near-zero F0 values.

    Fluorescence denominators should be positive and finite.  When a bin/ROI has
    invalid F0, fall back to the ROI's robust absolute fluorescence scale rather
    than allowing division by zero or exploding dF/F values.
    """
    f0 = np.asarray(f0, dtype=np.float32)
    raw = np.asarray(raw_f, dtype=np.float32)
    eps = np.float32(np.finfo(np.float32).eps)

    if f0.ndim == 1:
        fallback = np.nanmedian(np.abs(raw), axis=1).astype(np.float32)
        fallback_bad = ~np.isfinite(fallback) | (fallback <= eps)
        fallback[fallback_bad] = np.float32(1.0)
        bad = ~np.isfinite(f0) | (np.abs(f0) <= eps)
        if np.any(bad):
            f0 = f0.copy()
            f0[bad] = fallback[bad]
        return f0

    if f0.ndim == 2:
        fallback = np.nanmedian(np.abs(raw), axis=1).astype(np.float32)
        fallback_bad = ~np.isfinite(fallback) | (fallback <= eps)
        fallback[fallback_bad] = np.float32(1.0)
        bad = ~np.isfinite(f0) | (np.abs(f0) <= eps)
        if np.any(bad):
            f0 = f0.copy()
            rows, _cols = np.where(bad)
            f0[bad] = fallback[rows]
        return f0

    raise ValueError(f"Unexpected F0 shape {f0.shape}")


def _compute_static_f0(
    raw_f: np.ndarray,
    *,
    percentile: float = 50.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a constant per-ROI F0 from the full session trace."""
    raw, was_1d = _as_trace_2d(raw_f)
    f0_roi = np.nanpercentile(raw, float(percentile), axis=1).astype(np.float32)
    f0_roi = _safe_f0_values(f0_roi, raw)
    f0 = np.repeat(f0_roi[:, None], raw.shape[1], axis=1).astype(np.float32)
    meta = {
        "f0_method": "static_percentile",
        "f0_percentile": float(percentile),
        "f0_is_time_varying": False,
        "f0_per_roi": f0_roi,
    }
    return _restore_trace_dim(f0, was_1d), meta


def _nanmoving_median_1d(x: np.ndarray, window_bins: int) -> np.ndarray:
    """Small, dependency-free centered moving median over already downsampled bins."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    w = max(1, int(window_bins))
    if w <= 1:
        return arr.astype(np.float32)
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        out[i] = np.nanmedian(arr[lo:hi]).astype(np.float32)
    return out


def _fill_nan_1d(x: np.ndarray, fallback: float) -> np.ndarray:
    """Linearly fill NaNs in a 1D vector before interpolation/smoothing."""
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr
    if not np.any(finite):
        return np.full_like(arr, np.float32(fallback), dtype=np.float32)
    idx = np.arange(arr.size, dtype=float)
    out = arr.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], arr[finite]).astype(np.float32)
    return out


def _compute_robust_f0(
    raw_f: np.ndarray,
    *,
    sample_rate_hz: float,
    percentile: float = 50.0,
    bin_sec: float = 5.0,
    smooth_sec: float = 180.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a slowly varying robust F0 model from the full session trace.

    The model is intentionally conservative for voltage analysis:

    1. Split each ROI's full session trace into coarse bins.
    2. Estimate a robust fluorescence level in each bin using a percentile.
    3. Smooth those bin-level estimates with a centered moving median over a
       long window.
    4. Interpolate the slow F0 model back to the native sample rate.

    This is designed to track bleaching/session-scale fluorescence drift while
    avoiding the short-window rolling-baseline behavior that could suppress slow
    voltage oscillations.
    """
    raw, was_1d = _as_trace_2d(raw_f)
    n_rois, n_samples = raw.shape
    rate = float(sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz!r}")

    bin_samples = max(1, int(round(float(bin_sec) * rate)))
    n_bins = int(np.ceil(n_samples / bin_samples))
    if n_bins <= 1:
        return _compute_static_f0(raw, percentile=percentile)

    bin_centers = np.empty(n_bins, dtype=np.float64)
    bin_f0 = np.full((n_rois, n_bins), np.nan, dtype=np.float32)
    for b in range(n_bins):
        start = b * bin_samples
        stop = min(n_samples, (b + 1) * bin_samples)
        if stop <= start:
            continue
        bin_centers[b] = (start + stop - 1) / 2.0
        with np.errstate(all="ignore"):
            bin_f0[:, b] = np.nanpercentile(raw[:, start:stop], float(percentile), axis=1).astype(np.float32)

    smooth_bins = max(1, int(round(float(smooth_sec) / float(bin_sec))))
    if smooth_bins % 2 == 0:
        smooth_bins += 1

    f0 = np.empty((n_rois, n_samples), dtype=np.float32)
    sample_idx = np.arange(n_samples, dtype=np.float64)
    f0_roi_median = np.nanmedian(np.abs(raw), axis=1).astype(np.float32)
    bad_fallback = ~np.isfinite(f0_roi_median) | (f0_roi_median <= np.finfo(np.float32).eps)
    f0_roi_median[bad_fallback] = np.float32(1.0)

    smoothed_bins = np.full_like(bin_f0, np.nan, dtype=np.float32)
    for r in range(n_rois):
        filled = _fill_nan_1d(bin_f0[r], fallback=float(f0_roi_median[r]))
        smoothed = _nanmoving_median_1d(filled, smooth_bins)
        smoothed = _fill_nan_1d(smoothed, fallback=float(f0_roi_median[r]))
        smoothed_bins[r] = smoothed
        f0[r] = np.interp(sample_idx, bin_centers, smoothed).astype(np.float32)

    f0 = _safe_f0_values(f0, raw)
    meta = {
        "f0_method": "robust_binned_percentile_moving_median",
        "f0_percentile": float(percentile),
        "f0_bin_sec": float(bin_sec),
        "f0_smooth_sec": float(smooth_sec),
        "f0_bin_samples": int(bin_samples),
        "f0_smooth_bins": int(smooth_bins),
        "f0_n_bins": int(n_bins),
        "f0_is_time_varying": True,
        "f0_bin_centers_sample": bin_centers.astype(np.float64),
        "f0_bin_values": smoothed_bins.astype(np.float32),
    }
    return _restore_trace_dim(f0, was_1d), meta


def compute_voltage_f0(
    raw_f: np.ndarray,
    *,
    sample_rate_hz: float,
    method: str = "robust",
    percentile: float = 50.0,
    robust_bin_sec: float = 5.0,
    robust_smooth_sec: float = 180.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute an F0 model for ASAP voltage fluorescence.

    Parameters
    ----------
    raw_f
        Raw fluorescence, either one ROI as ``(n_samples,)`` or many ROIs as
        ``(n_rois, n_samples)``.
    sample_rate_hz
        Voltage sample/line rate, typically 10.8 kHz for current SLAP2 voltage
        integration-mode recordings.
    method
        ``'static'`` for a constant per-ROI session F0 or ``'robust'`` for a
        slowly varying binned-percentile/moving-median F0 model.
    percentile
        Percentile used to estimate baseline fluorescence.  For inverse ASAP
        indicators, median or a modest upper percentile is usually safer than a
        calcium-style low percentile.
    robust_bin_sec, robust_smooth_sec
        Robust F0 parameters used only for ``method='robust'``.

    Returns
    -------
    f0, metadata
        F0 has the same shape as ``raw_f``.
    """
    m = str(method).lower().replace(" ", "_")
    if m in {"static", "static_f0", "static_percentile"}:
        f0, meta = _compute_static_f0(raw_f, percentile=percentile)
    elif m in {"robust", "robust_f0", "robust_baseline"}:
        f0, meta = _compute_robust_f0(
            raw_f,
            sample_rate_hz=sample_rate_hz,
            percentile=percentile,
            bin_sec=robust_bin_sec,
            smooth_sec=robust_smooth_sec,
        )
    else:
        raise ValueError("method must be 'static' or 'robust'")

    meta.update({
        "sample_rate_hz": float(sample_rate_hz),
        "input_shape": tuple(int(x) for x in np.asarray(raw_f).shape),
    })
    return f0, meta


def compute_voltage_dff(raw_f: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Compute ASAP-polarity-corrected dF/F.

    The returned signal is positive for fluorescence decreases:

    ``dff = (F0 - F) / F0``

    This is equivalent to inverted dF/F but is called ``dff`` throughout the
    voltage pipeline for convenience.
    """
    raw = np.asarray(raw_f, dtype=np.float32)
    f0_arr = np.asarray(f0, dtype=np.float32)
    f0_arr = _safe_f0_values(f0_arr, raw if raw.ndim == 2 else raw[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = (f0_arr - raw) / f0_arr
    return np.asarray(dff, dtype=np.float32)


def transform_voltage_signal(
    raw_f: np.ndarray,
    *,
    sample_rate_hz: float,
    method: str = "robust",
    percentile: float = 50.0,
    robust_bin_sec: float = 5.0,
    robust_smooth_sec: float = 180.0,
) -> Dict[str, Any]:
    """Return raw fluorescence, F0, and ASAP-polarity-corrected dF/F."""
    f0, f0_meta = compute_voltage_f0(
        raw_f,
        sample_rate_hz=sample_rate_hz,
        method=method,
        percentile=percentile,
        robust_bin_sec=robust_bin_sec,
        robust_smooth_sec=robust_smooth_sec,
    )
    dff = compute_voltage_dff(raw_f, f0)
    meta = {
        "trace_signal": f"dff_{method}_f0",
        "transform": "dff = (F0 - F) / F0",
        "polarity": "ASAP/inverse fluorescence corrected; positive dff corresponds to fluorescence decrease",
        **f0_meta,
    }
    return {
        "raw_f": np.asarray(raw_f, dtype=np.float32),
        "f0": np.asarray(f0, dtype=np.float32),
        "dff": np.asarray(dff, dtype=np.float32),
        "metadata": meta,
    }


def _parse_trace_signal(trace_signal: str) -> Tuple[str, Optional[str]]:
    """Normalize trace_signal names into extraction signal and F0 method."""
    signal = str(trace_signal).lower().replace(" ", "_")
    if signal in {"raw", "raw_f", "raw_fluorescence", "f"}:
        return "raw_f", None
    if signal in {"dff_static_f0", "inverted_dff_static_f0", "-dff_static_f0", "neg_dff_static_f0"}:
        return "dff", "static"
    if signal in {"dff_robust_f0", "inverted_dff_robust_f0", "-dff_robust_f0", "neg_dff_robust_f0"}:
        return "dff", "robust"
    raise ValueError(
        "Unsupported voltage trace_signal. Use 'raw', 'dff_static_f0', or 'dff_robust_f0'."
    )


def _transform_voltage_bundle(
    bundle: ReconstructedTraceBundle,
    *,
    trace_signal: str,
    sample_rate_hz: float,
    f0_percentile: float = 50.0,
    robust_f0_bin_sec: float = 5.0,
    robust_f0_smooth_sec: float = 180.0,
) -> Tuple[ReconstructedTraceBundle, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Apply an optional voltage transform to a reconstructed bundle.

    Returns
    -------
    transformed_bundle, transform_metadata, transform_payload
        ``transform_payload`` is ``None`` for raw extraction.  For dF/F methods,
        it contains ``raw_f``, ``f0``, and ``dff`` arrays for optional session-
        trace export.
    """
    output_signal, f0_method = _parse_trace_signal(trace_signal)
    if output_signal == "raw_f":
        return bundle, {
            "trace_signal": trace_signal,
            "output_signal": "raw_f",
            "transform": "none",
        }, None

    transformed = transform_voltage_signal(
        np.asarray(bundle.traces, dtype=np.float32),
        sample_rate_hz=sample_rate_hz,
        method=str(f0_method),
        percentile=f0_percentile,
        robust_bin_sec=robust_f0_bin_sec,
        robust_smooth_sec=robust_f0_smooth_sec,
    )
    out = replace(bundle, traces=np.asarray(transformed["dff"], dtype=np.float32))
    meta = dict(transformed["metadata"])
    meta.update({
        "trace_signal_requested": trace_signal,
        "output_signal": "dff",
    })
    return out, meta, transformed


def _write_session_transform_group(
    h5: h5py.File,
    *,
    dmd_key: str,
    bundle: ReconstructedTraceBundle,
    roi_ids: np.ndarray,
    roi_mask: np.ndarray,
    transform_meta: Mapping[str, Any],
    transform_payload: Optional[Mapping[str, Any]],
    dtype: np.dtype,
    compression: Optional[str],
    compression_opts: Optional[int],
) -> None:
    """Write full-session raw/F0/dFF traces for one DMD.

    Datasets are ROI-by-time, matching ``ReconstructedTraceBundle.traces``.
    For raw extraction, only ``raw_f`` is written.  For dF/F extraction,
    ``raw_f``, ``f0``, and ``dff`` are written.
    """
    grp = h5.create_group(dmd_key)
    grp.attrs["schema"] = "ROI-by-time full-session voltage traces"
    grp.attrs["signal_transform_json"] = json.dumps(transform_meta, default=_json_default)
    grp.create_dataset("timebase_sec", data=np.asarray(bundle.timebase_sec, dtype=np.float64))
    grp.create_dataset("roi_ids", data=np.asarray(roi_ids, dtype="S"))
    grp.create_dataset("valid_rois_mask", data=np.asarray(roi_mask, dtype=bool))
    grp.create_dataset("trial_valid_mask", data=np.asarray(bundle.trial_valid_mask, dtype=bool))
    grp.create_dataset("trial_lengths_samples", data=np.asarray(bundle.trial_lengths_samples, dtype=np.int64))

    if transform_payload is None:
        arrays = {"raw_f": np.asarray(bundle.traces, dtype=np.float32)}
    else:
        arrays = {
            "raw_f": np.asarray(transform_payload["raw_f"], dtype=np.float32),
            "f0": np.asarray(transform_payload["f0"], dtype=np.float32),
            "dff": np.asarray(transform_payload["dff"], dtype=np.float32),
        }

    for name, arr in arrays.items():
        if arr.ndim == 1:
            arr = arr[None, :]
        chunks = (1, max(1, min(arr.shape[1], 8192))) if arr.size else None
        grp.create_dataset(
            name,
            data=arr.astype(dtype, copy=False),
            dtype=np.dtype(dtype),
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=(compression is not None),
        )


def load_voltage_roi_transform_h5(
    session_traces_h5: PathLike,
    *,
    dmd: int,
    roi_index: int,
) -> Dict[str, Any]:
    """Load one ROI's saved full-session voltage transform from HDF5.

    Returns any of ``raw_f``, ``f0``, and ``dff`` that are present, plus the
    voltage timebase and transform metadata.  This is intended for lightweight
    inspection/plotting after ``process_voltage_extraction`` has run.
    """
    path = Path(session_traces_h5)
    dmd_key = f"DMD{int(dmd)}"
    out: Dict[str, Any] = {"session_traces_h5": str(path), "dmd": int(dmd), "roi_index": int(roi_index)}
    with h5py.File(path, "r") as h5:
        if dmd_key not in h5:
            raise KeyError(f"{dmd_key!r} not found in {path}")
        grp = h5[dmd_key]
        out["timebase_sec"] = grp["timebase_sec"][:]
        for key in ("raw_f", "f0", "dff"):
            if key in grp:
                out[key] = grp[key][int(roi_index), :]
        out["metadata"] = json.loads(grp.attrs.get("signal_transform_json", "{}"))
    return out


def compute_voltage_roi_transform_from_asset(
    asset: SessionAssets,
    *,
    dmd: int,
    roi_index: int,
    sample_rate_hz: Optional[float] = None,
    default_sample_rate_hz: float = 10_800.0,
    epoch_start_sec: Optional[float] = None,
    trace_mode: str = "trial",
    drop_discarded: bool = True,
    f0_method: str = "robust",
    f0_percentile: float = 50.0,
    robust_f0_bin_sec: float = 5.0,
    robust_f0_smooth_sec: float = 180.0,
) -> Dict[str, Any]:
    """Compute raw/F0/dFF for one ROI directly from a session asset.

    This helper is meant for interactive inspection.  It reconstructs the full
    DMD session trace using the same code path as batch extraction, selects one
    ROI, and computes static or robust F0 plus ASAP-polarity-corrected dF/F.
    """
    if epoch_start_sec is None:
        epoch_df = _open_imaging_epochs(asset)
        epoch_start_sec = float(epoch_df.iloc[0]["start_time"])

    with load_voltage_summary_from_asset(asset, keep_open=True) as vs:
        rate = resolve_voltage_sample_rate_hz(
            asset=asset,
            vs=vs,
            sample_rate_hz=sample_rate_hz,
            default_hz=default_sample_rate_hz,
        )
        bundle = reconstruct_voltage_dmd_session_traces(
            vs,
            dmd=int(dmd),
            sample_rate_hz=rate,
            epoch_start_sec=float(epoch_start_sec),
            drop_discarded=drop_discarded,
            dtype=np.float32,
            trace_mode=trace_mode,
        )

    if roi_index < 0 or roi_index >= bundle.traces.shape[0]:
        raise IndexError(f"roi_index={roi_index} is out of bounds for {bundle.traces.shape[0]} ROIs")

    raw_f = np.asarray(bundle.traces[int(roi_index), :], dtype=np.float32)
    transformed = transform_voltage_signal(
        raw_f,
        sample_rate_hz=rate,
        method=f0_method,
        percentile=f0_percentile,
        robust_bin_sec=robust_f0_bin_sec,
        robust_smooth_sec=robust_f0_smooth_sec,
    )
    return {
        "timebase_sec": np.asarray(bundle.timebase_sec, dtype=np.float64),
        "raw_f": transformed["raw_f"],
        "f0": transformed["f0"],
        "dff": transformed["dff"],
        "metadata": transformed["metadata"],
        "dmd": int(dmd),
        "roi_index": int(roi_index),
        "sample_rate_hz": float(rate),
    }


# -----------------------------------------------------------------------------
# Streaming event extraction and summaries
# -----------------------------------------------------------------------------


class _OnlineSummary:
    """Online NaN-aware mean/std accumulator for ROI-by-time event snippets."""

    def __init__(self, shape: Tuple[int, int]) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.sum = np.zeros(self.shape, dtype=np.float64)
        self.sumsq = np.zeros(self.shape, dtype=np.float64)
        self.count = np.zeros(self.shape, dtype=np.int64)
        self.n_events = 0

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape != self.shape:
            raise ValueError(f"Expected snippet shape {self.shape}, got {arr.shape}")
        finite = np.isfinite(arr)
        vals = np.where(finite, arr, 0.0)
        self.sum += vals
        self.sumsq += vals * vals
        self.count += finite.astype(np.int64)
        self.n_events += 1

    def as_dict(self) -> Dict[str, np.ndarray]:
        mean = np.full(self.shape, np.nan, dtype=np.float32)
        std = np.full(self.shape, np.nan, dtype=np.float32)
        valid = self.count > 0
        mean64 = np.zeros(self.shape, dtype=np.float64)
        mean64[valid] = self.sum[valid] / self.count[valid]
        var64 = np.zeros(self.shape, dtype=np.float64)
        var64[valid] = self.sumsq[valid] / self.count[valid] - mean64[valid] ** 2
        var64[var64 < 0] = 0.0
        mean[valid] = mean64[valid].astype(np.float32)
        std[valid] = np.sqrt(var64[valid]).astype(np.float32)
        return {
            "mean": mean,
            "std": std,
            "n_events": np.array(int(self.n_events), dtype=int),
            "n_finite": self.count.astype(np.int64),
        }


def _n_window_samples(pre_time: float, post_time: float, sample_rate_hz: float) -> Tuple[int, int, int]:
    """Return pre, post, and total sample counts for one extraction window."""
    n_pre = int(round(float(pre_time) * float(sample_rate_hz)))
    n_post = int(round(float(post_time) * float(sample_rate_hz)))
    return n_pre, n_post, n_pre + n_post


def _valid_timebase(bundle: ReconstructedTraceBundle) -> Optional[np.ndarray]:
    """Return the explicit bundle timebase if it is valid for nearest-neighbor lookup."""
    tb = np.asarray(bundle.timebase_sec, dtype=float).reshape(-1)
    if tb.size != bundle.traces.shape[1] or tb.size == 0:
        return None
    if not np.all(np.isfinite(tb)) or np.any(np.diff(tb) <= 0):
        return None
    return tb


def _nearest_sample_index(
    onset_sec: float,
    *,
    bundle: ReconstructedTraceBundle,
    sample_rate_hz: float,
    timebase_sec: Optional[np.ndarray],
) -> int:
    """Map one corrected event onset to the nearest voltage sample index."""
    n_samples = int(bundle.traces.shape[1])
    if n_samples <= 0:
        return 0

    if timebase_sec is not None:
        idx = int(np.searchsorted(timebase_sec, float(onset_sec), side="left"))
        if idx <= 0:
            return 0
        if idx >= n_samples:
            return n_samples - 1
        prev_idx = idx - 1
        if abs(float(timebase_sec[idx]) - float(onset_sec)) < abs(float(onset_sec) - float(timebase_sec[prev_idx])):
            return idx
        return prev_idx

    center = int(round((float(onset_sec) - float(bundle.session_start_sec)) * float(sample_rate_hz)))
    return max(0, min(center, n_samples - 1))


def _extract_one_snippet(
    bundle: ReconstructedTraceBundle,
    onset_sec: float,
    *,
    sample_rate_hz: float,
    pre_time: float,
    post_time: float,
    roi_mask: np.ndarray,
    timebase_sec: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Extract one ROI-by-time event snippet or return None if out of bounds."""
    n_pre, _n_post, n_win = _n_window_samples(pre_time, post_time, sample_rate_hz)
    center = _nearest_sample_index(
        onset_sec,
        bundle=bundle,
        sample_rate_hz=sample_rate_hz,
        timebase_sec=timebase_sec,
    )
    start = center - n_pre
    stop = start + n_win
    if start < 0 or stop > bundle.traces.shape[1]:
        return None
    return np.asarray(bundle.traces[roi_mask, start:stop], dtype=np.float32)


def _flatten_intervals(intervals: StimIntervalList) -> List[float]:
    """Return onset seconds from an interval list."""
    return [float(onset) for onset, _ in intervals]


def _write_onsets(group: h5py.Group, name: str, onsets: Sequence[float]) -> None:
    """Write retained event onsets as a float64 HDF5 dataset."""
    group.create_dataset(name, data=np.asarray(onsets, dtype=np.float64))


def _create_event_dataset(
    group: h5py.Group,
    name: str,
    *,
    shape: Tuple[int, int, int],
    dtype: np.dtype,
    compression: Optional[str],
    compression_opts: Optional[int],
) -> h5py.Dataset:
    """Create a chunked event dataset shaped events by ROIs by time."""
    n_events, n_rois, n_time = shape
    if n_events == 0:
        return group.create_dataset(name, shape=shape, dtype=dtype)

    chunks = (1, max(1, n_rois), max(1, min(n_time, 8192)))
    return group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=(compression is not None),
    )


def _extract_interval_list_streaming(
    bundle: ReconstructedTraceBundle,
    intervals: StimIntervalList,
    *,
    sample_rate_hz: float,
    pre_time: float,
    post_time: float,
    roi_mask: np.ndarray,
    h5_group: Optional[h5py.Group] = None,
    dataset_name: str = "traces",
    onsets_name: str = "onsets_sec",
    dtype: np.dtype = np.float32,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Extract, optionally write, and summarize a flat event interval list."""
    onsets_all = _flatten_intervals(intervals)
    n_pre, _n_post, n_win = _n_window_samples(pre_time, post_time, sample_rate_hz)
    _ = n_pre  # retained for readability and symmetry with image extraction
    n_rois = int(np.sum(roi_mask))
    timebase_sec = _valid_timebase(bundle)

    valid: List[Tuple[float, int, int]] = []
    for onset in onsets_all:
        center = _nearest_sample_index(
            onset,
            bundle=bundle,
            sample_rate_hz=sample_rate_hz,
            timebase_sec=timebase_sec,
        )
        start = center - int(round(pre_time * sample_rate_hz))
        stop = start + n_win
        if start < 0 or stop > bundle.traces.shape[1]:
            continue
        valid.append((onset, start, stop))

    writer: Optional[h5py.Dataset] = None
    if h5_group is not None:
        writer = _create_event_dataset(
            h5_group,
            dataset_name,
            shape=(len(valid), n_rois, n_win),
            dtype=np.dtype(dtype),
            compression=compression,
            compression_opts=compression_opts,
        )
        _write_onsets(h5_group, onsets_name, [v[0] for v in valid])

    summary = _OnlineSummary((n_rois, n_win))
    for i, (_onset, start, stop) in enumerate(valid):
        snip = np.asarray(bundle.traces[roi_mask, start:stop], dtype=dtype)
        if writer is not None:
            writer[i, :, :] = snip
        summary.update(snip)

    return summary.as_dict(), np.asarray([v[0] for v in valid], dtype=np.float64)


def _extract_image_identity_streaming(
    bundle: ReconstructedTraceBundle,
    image_times: StimIntervalDict,
    *,
    sample_rate_hz: float,
    pre_time: float,
    post_time: float,
    roi_mask: np.ndarray,
    h5_group: Optional[h5py.Group] = None,
    dtype: np.dtype = np.float32,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, str]]:
    """Extract/summarize image-identity event snippets one image at a time."""
    mean_by_image: Dict[str, Dict[str, np.ndarray]] = {}
    onsets_by_image: Dict[str, np.ndarray] = {}
    h5_name_map: Dict[str, str] = {}

    if h5_group is not None:
        h5_group.attrs["schema"] = "image groups are named image_XXXX; original image identity is in each group's image_name attr"

    for i, (image_name, intervals) in enumerate(image_times.items()):
        subgrp = None
        h5_key = f"image_{i:04d}"
        h5_name_map[h5_key] = str(image_name)
        if h5_group is not None:
            subgrp = h5_group.create_group(h5_key)
            subgrp.attrs["image_name"] = str(image_name)

        summary, onsets = _extract_interval_list_streaming(
            bundle,
            intervals,
            sample_rate_hz=sample_rate_hz,
            pre_time=pre_time,
            post_time=post_time,
            roi_mask=roi_mask,
            h5_group=subgrp,
            dataset_name="traces",
            onsets_name="onsets_sec",
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        mean_by_image[str(image_name)] = summary
        onsets_by_image[str(image_name)] = onsets

    return mean_by_image, onsets_by_image, h5_name_map


# -----------------------------------------------------------------------------
# Sequence summaries without retaining all single-trial image snippets
# -----------------------------------------------------------------------------


def _empty_position_summary(n_positions: int, n_rois: int, n_time: int) -> Dict[str, np.ndarray]:
    """Return a sequence-position summary with no events."""
    return {
        "mean": np.full((n_positions, n_rois, n_time), np.nan, dtype=np.float32),
        "std": np.full((n_positions, n_rois, n_time), np.nan, dtype=np.float32),
        "counts": np.zeros((n_positions,), dtype=int),
        "n_finite": np.zeros((n_positions, n_rois, n_time), dtype=np.int64),
    }


def _summaries_to_position_dict(summaries: Sequence[_OnlineSummary]) -> Dict[str, np.ndarray]:
    """Stack a list of online summaries along sequence position."""
    if len(summaries) == 0:
        return _empty_position_summary(0, 0, 0)

    dicts = [s.as_dict() for s in summaries]
    return {
        "mean": np.stack([d["mean"] for d in dicts], axis=0),
        "std": np.stack([d["std"] for d in dicts], axis=0),
        "counts": np.asarray([int(d["n_events"]) for d in dicts], dtype=int),
        "n_finite": np.stack([d["n_finite"] for d in dicts], axis=0),
    }


def _summarize_change_locked_sequences_streaming(
    seq_events: Dict[str, Dict[str, Any]],
    bundle: ReconstructedTraceBundle,
    *,
    sample_rate_hz: float,
    pre_time: float,
    post_time: float,
    roi_mask: np.ndarray,
) -> Dict[str, Any]:
    """Build change-locked sequence summaries by extracting snippets on demand."""
    n_rois = int(np.sum(roi_mask))
    _n_pre, _n_post, n_time = _n_window_samples(pre_time, post_time, sample_rate_hz)
    timebase_sec = _valid_timebase(bundle)
    out: Dict[str, Any] = {}

    for image_name, groups in seq_events.items():
        pre_summaries = [_OnlineSummary((n_rois, n_time)) for _ in range(2)]
        repeated_summaries: List[_OnlineSummary] = []
        terminal_summary = _OnlineSummary((n_rois, n_time))
        pre_n_sequences = 0
        repeated_n_sequences = 0
        terminal_n_sequences = 0
        sequence_lengths: List[int] = []

        for pre_evts, rep_evts, term_evt in zip(groups["prechange"], groups["repeated"], groups["terminal"]):
            pre_snips = [
                _extract_one_snippet(
                    bundle,
                    float(evt.onset),
                    sample_rate_hz=sample_rate_hz,
                    pre_time=pre_time,
                    post_time=post_time,
                    roi_mask=roi_mask,
                    timebase_sec=timebase_sec,
                )
                for evt in pre_evts
            ]
            if len(pre_snips) == 2 and all(s is not None for s in pre_snips):
                for pos, snip in enumerate(pre_snips):
                    pre_summaries[pos].update(snip)  # type: ignore[arg-type]
                pre_n_sequences += 1

            rep_count_this_sequence = 0
            for pos, evt in enumerate(rep_evts):
                while len(repeated_summaries) <= pos:
                    repeated_summaries.append(_OnlineSummary((n_rois, n_time)))
                snip = _extract_one_snippet(
                    bundle,
                    float(evt.onset),
                    sample_rate_hz=sample_rate_hz,
                    pre_time=pre_time,
                    post_time=post_time,
                    roi_mask=roi_mask,
                    timebase_sec=timebase_sec,
                )
                if snip is not None:
                    repeated_summaries[pos].update(snip)
                    rep_count_this_sequence += 1
            if rep_count_this_sequence > 0:
                repeated_n_sequences += 1
                sequence_lengths.append(int(rep_count_this_sequence))

            term_snip = _extract_one_snippet(
                bundle,
                float(term_evt.onset),
                sample_rate_hz=sample_rate_hz,
                pre_time=pre_time,
                post_time=post_time,
                roi_mask=roi_mask,
                timebase_sec=timebase_sec,
            )
            if term_snip is not None:
                terminal_summary.update(term_snip)
                terminal_n_sequences += 1

        pre = _summaries_to_position_dict(pre_summaries)
        rep = _summaries_to_position_dict(repeated_summaries)
        term = terminal_summary.as_dict()
        out[str(image_name)] = {
            "prechange": {
                **pre,
                "n_sequences": np.array(pre_n_sequences, dtype=int),
                "positions": np.array([-2, -1], dtype=int),
            },
            "repeated": {
                **rep,
                "n_sequences": np.array(repeated_n_sequences, dtype=int),
                "sequence_lengths": np.asarray(sequence_lengths, dtype=int),
                "positions": np.arange(rep["mean"].shape[0], dtype=int),
            },
            "terminal": {
                **term,
                "n_sequences": np.array(terminal_n_sequences, dtype=int),
                "position": np.array([999], dtype=int),
            },
        }

    return out


# -----------------------------------------------------------------------------
# Main public pipeline
# -----------------------------------------------------------------------------


def process_voltage_extraction(
    asset: SessionAssets,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    use_roi_qc: bool = True,
    overwrite: bool = False,
    sample_rate_hz: Optional[float] = None,
    default_sample_rate_hz: float = 10_800.0,
    trace_signal: str = "raw",
    f0_percentile: float = 50.0,
    robust_f0_bin_sec: float = 5.0,
    robust_f0_smooth_sec: float = 180.0,
    trace_mode: str = "trial",
    drop_discarded: bool = True,
    dtype: np.dtype = np.float32,
    write_single_trials: bool = True,
    write_sequence: bool = True,
    write_session_traces: bool = True,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
) -> Dict[str, Any]:
    """Extract voltage image/change/omission packages for one session asset.

    Parameters
    ----------
    asset
        Session asset bundle returned by ``VIPSessionRegistry.resolve_assets``.
        It must contain ``asset.modality_assets['voltage']['summary_mat']`` and,
        for split-H5 outputs, ``trace_h5``.
    metadata
        Optional caller metadata.  ``metadata['prepost_sec']`` can override event
        windows with keys ``image``/``image_identity``, ``change``, and
        ``omission``.
    use_roi_qc
        Apply the first-pass voltage ROI keep masks written by
        ``run_voltage_qc`` when available.
    overwrite
        Recompute outputs even if all expected outputs already exist.
    sample_rate_hz
        Explicit voltage sample/line rate.  If omitted, metadata and session
        assets are checked before falling back to ``default_sample_rate_hz``.
    default_sample_rate_hz
        Current SLAP2 integration-mode default is 10.8 kHz.
    trace_signal
        ``'raw'`` keeps raw fluorescence. ``'dff_static_f0'`` and
        ``'dff_robust_f0'`` compute ASAP-polarity-corrected dF/F on the
        reconstructed full-session ROI trace before event snippets are extracted.
    f0_percentile
        Percentile used for static and robust F0 estimation.
    robust_f0_bin_sec, robust_f0_smooth_sec
        Parameters for the robust full-session F0 model.  The defaults estimate a
        binned robust fluorescence level every 5 s and smooth those estimates
        over 180 s, which is intentionally conservative for slow voltage dynamics.
    trace_mode
        Passed to ``VoltageSummary.get_roi_traces``.  Current voltage outputs are
        trial-based, so this should usually be ``'trial'``.
    drop_discarded
        Remove samples marked by ``discardFrames`` before event extraction.
    write_single_trials
        Write large event tensors to HDF5.  Set False for metadata/summary-only
        dry runs.
    write_sequence
        Write change-locked sequence summaries to NPZ.
    write_session_traces
        Write full-session ROI-by-time traces to HDF5. For dF/F transforms this
        file contains ``raw_f``, ``f0``, and ``dff`` for every DMD/ROI, enabling
        later inspection and plotting without recomputing the transform.
    compression, compression_opts
        HDF5 compression settings for single-trial event datasets.

    Returns
    -------
    dict
        Paths and status metadata for the extraction run.
    """
    metadata = metadata or {}
    pp = metadata.get("prepost_sec", {})
    windows = EventWindows(
        image=_as_float_tuple(pp.get("image", pp.get("image_identity", (0.25, 0.50)))),
        change=_as_float_tuple(pp.get("change", (1.00, 0.75))),
        omission=_as_float_tuple(pp.get("omission", (1.00, 1.50))),
    )

    if asset.bonsai_event_log_csv is None:
        raise FileNotFoundError("asset.bonsai_event_log_csv is missing")
    if asset.qc_dir is None or asset.derived_dir is None:
        raise ValueError("asset.qc_dir and asset.derived_dir must be set")

    voltage_dir = _modality_dir(asset, "derived", "voltage")
    voltage_qc_dir = _modality_dir(asset, "qc", "voltage")

    # Resolve output names early so skipped runs are cheap.
    suffix = _trace_suffix(trace_signal, trace_mode)
    mean_npz = voltage_dir / f"voltage_mean_{suffix}.npz"
    single_h5 = voltage_dir / f"voltage_single_trial_{suffix}.h5"
    session_traces_h5 = voltage_dir / f"voltage_session_traces_{suffix}.h5"
    seq_npz = voltage_dir / f"voltage_sequence_{suffix}.npz"
    qc_json = voltage_qc_dir / f"voltage_extraction_qc_{suffix}.json"

    expected_outputs = [mean_npz, qc_json]
    if write_single_trials:
        expected_outputs.append(single_h5)
    if write_sequence:
        expected_outputs.append(seq_npz)
    if write_session_traces:
        expected_outputs.append(session_traces_h5)
    if all(p.exists() for p in expected_outputs) and not overwrite:
        return {
            "status": "exists",
            "mean_npz": str(mean_npz),
            "single_h5": str(single_h5) if write_single_trials else None,
            "session_traces_h5": str(session_traces_h5) if write_session_traces else None,
            "seq_npz": str(seq_npz) if write_sequence else None,
            "qc_json": str(qc_json),
        }

    stim_df = load_corrected_bonsai_csv(asset.bonsai_event_log_csv)
    epoch_df = _open_imaging_epochs(asset)
    epoch_start_sec = float(epoch_df.iloc[0]["start_time"])
    epoch_end_sec = float(epoch_df.iloc[-1]["end_time"])
    epoch_duration_sec = float(epoch_end_sec - epoch_start_sec)

    image_times, ordered_images = extract_image_intervals(stim_df)
    change_times = extract_change_intervals(stim_df)
    omission_times = extract_omission_intervals(stim_df)
    extract_ordered_change_targets(stim_df, ordered_images)

    image_times_f = filter_intervals_to_epochs(
        image_times,
        epoch_df,
        pre_time=windows.image[0],
        post_time=windows.image[1],
    )
    change_times_f = filter_intervals_to_epochs(
        change_times,
        epoch_df,
        pre_time=windows.change[0],
        post_time=windows.change[1],
    )
    omission_times_f = filter_intervals_to_epochs(
        omission_times,
        epoch_df,
        pre_time=windows.omission[0],
        post_time=windows.omission[1],
    )
    ordered_images_f = filter_ordered_images_to_epochs(
        ordered_images,
        epoch_df,
        pre_time=windows.image[0],
        post_time=windows.image[1],
    )
    seq_events = build_change_locked_sequences(ordered_images_f)

    with load_voltage_summary_from_asset(asset, keep_open=True) as vs:
        rate = resolve_voltage_sample_rate_hz(
            asset=asset,
            vs=vs,
            sample_rate_hz=sample_rate_hz or metadata.get("voltage_sample_rate_hz"),
            default_hz=default_sample_rate_hz,
        )
        tvecs = _time_vectors(windows, rate)

        summary_mat = asset.get_asset("voltage", "summary_mat") if hasattr(asset, "get_asset") else None
        trace_h5 = asset.get_asset("voltage", "trace_h5") if hasattr(asset, "get_asset") else None
        base_meta: Dict[str, Any] = {
            "schema_version": "0.1.0",
            "session_id": asset.session_id,
            "subject_id": int(asset.subject_id),
            "modality": "voltage",
            "summary_mat": str(summary_mat) if summary_mat is not None else None,
            "trace_h5": str(trace_h5) if trace_h5 is not None else None,
            "bonsai_event_log_csv": str(asset.bonsai_event_log_csv),
            "sample_rate_hz": float(rate),
            "default_sample_rate_hz": float(default_sample_rate_hz),
            "windows_sec": {
                "image": tuple(float(x) for x in windows.image),
                "change": tuple(float(x) for x in windows.change),
                "omission": tuple(float(x) for x in windows.omission),
            },
            "trace_signal": str(trace_signal),
            "trace_mode": str(trace_mode),
            "trace_suffix": suffix,
            "drop_discarded": bool(drop_discarded),
            "use_roi_qc": bool(use_roi_qc),
            "write_single_trials": bool(write_single_trials),
            "write_sequence": bool(write_sequence),
            "write_session_traces": bool(write_session_traces),
            "f0_percentile": float(f0_percentile),
            "robust_f0_bin_sec": float(robust_f0_bin_sec),
            "robust_f0_smooth_sec": float(robust_f0_smooth_sec),
            "epoch_start_sec": float(epoch_start_sec),
            "epoch_end_sec": float(epoch_end_sec),
            "voltage_summary_layout": getattr(vs, "layout", None),
            "voltage_summary_metadata": getattr(vs, "metadata", {}),
            "voltage_h5_attrs": getattr(vs, "h5_attrs", {}),
        }

        mean_pkg: Dict[str, Any] = {"metadata": base_meta, "timebase_sec": tvecs, "DMD1": {}, "DMD2": {}}
        seq_pkg: Dict[str, Any] = {"metadata": base_meta, "timebase_sec": {"image": tvecs["image"]}, "DMD1": {}, "DMD2": {}}
        qc: Dict[str, Any] = {
            "schema_version": "0.1.0",
            "session_id": asset.session_id,
            "summary_mat": base_meta["summary_mat"],
            "trace_h5": base_meta["trace_h5"],
            "bonsai_event_log_csv": str(asset.bonsai_event_log_csv),
            "sample_rate_hz": float(rate),
            "trace_signal": str(trace_signal),
            "trace_mode": str(trace_mode),
            "trace_suffix": suffix,
            "windows_sec": base_meta["windows_sec"],
            "event_counts": {
                "image_total": int(sum(len(v) for v in image_times.values())),
                "image_after_epoch_filter": int(sum(len(v) for v in image_times_f.values())),
                "change_total": int(len(change_times)),
                "change_after_epoch_filter": int(len(change_times_f)),
                "omission_total": int(len(omission_times)),
                "omission_after_epoch_filter": int(len(omission_times_f)),
                "n_unique_image_ids_total": int(len(image_times)),
                "n_unique_image_ids_after_epoch_filter": int(len(image_times_f)),
            },
            "epoch_duration_sec": float(epoch_duration_sec),
            "per_dmd": {},
        }

        h5: Optional[h5py.File] = None
        session_h5: Optional[h5py.File] = None
        if write_single_trials:
            single_h5.parent.mkdir(parents=True, exist_ok=True)
            h5 = h5py.File(single_h5, "w")
            _metadata_json_attr(h5, base_meta)
            h5.create_dataset("timebase_sec/image", data=tvecs["image"].astype(np.float64))
            h5.create_dataset("timebase_sec/change", data=tvecs["change"].astype(np.float64))
            h5.create_dataset("timebase_sec/omission", data=tvecs["omission"].astype(np.float64))
        if write_session_traces:
            session_traces_h5.parent.mkdir(parents=True, exist_ok=True)
            session_h5 = h5py.File(session_traces_h5, "w")
            _metadata_json_attr(session_h5, base_meta)

        try:
            for dmd in range(1, int(vs.n_dmds) + 1):
                bundle = reconstruct_voltage_dmd_session_traces(
                    vs,
                    dmd=dmd,
                    sample_rate_hz=rate,
                    epoch_start_sec=epoch_start_sec,
                    drop_discarded=drop_discarded,
                    dtype=dtype,
                    trace_mode=trace_mode,
                )
                if bundle.traces.size == 0:
                    qc["per_dmd"][f"DMD{dmd}"] = {"skipped": True, "reason": "no valid traces"}
                    continue

                bundle, transform_meta, transform_payload = _transform_voltage_bundle(
                    bundle,
                    trace_signal=trace_signal,
                    sample_rate_hz=rate,
                    f0_percentile=f0_percentile,
                    robust_f0_bin_sec=robust_f0_bin_sec,
                    robust_f0_smooth_sec=robust_f0_smooth_sec,
                )

                n_rois_total = int(bundle.traces.shape[0])
                roi_mask = _load_voltage_roi_qc_mask(asset, dmd) if use_roi_qc else None
                roi_mask = _validate_or_fill_mask(roi_mask, n_rois_total)
                n_rois_kept = int(np.sum(roi_mask))
                ids = _roi_ids(dmd, n_rois_total, roi_mask)

                dmd_key = f"DMD{dmd}"
                if session_h5 is not None:
                    _write_session_transform_group(
                        session_h5,
                        dmd_key=dmd_key,
                        bundle=bundle,
                        roi_ids=_roi_ids(dmd, n_rois_total, np.ones(n_rois_total, dtype=bool)),
                        roi_mask=roi_mask,
                        transform_meta=transform_meta,
                        transform_payload=transform_payload,
                        dtype=np.dtype(dtype),
                        compression=compression,
                        compression_opts=compression_opts,
                    )

                dmd_h5: Optional[h5py.Group] = None
                if h5 is not None:
                    dmd_h5 = h5.create_group(dmd_key)
                    dmd_h5.attrs["n_rois_total"] = n_rois_total
                    dmd_h5.attrs["n_rois_kept"] = n_rois_kept
                    dmd_h5.attrs["signal_transform_json"] = json.dumps(transform_meta, default=_json_default)
                    dmd_h5.create_dataset("roi_ids", data=np.asarray(ids, dtype="S"))
                    dmd_h5.create_dataset("valid_rois_mask", data=roi_mask.astype(bool))

                image_grp = dmd_h5.create_group("image_identity") if dmd_h5 is not None else None
                change_grp = dmd_h5.create_group("change") if dmd_h5 is not None else None
                omission_grp = dmd_h5.create_group("omission") if dmd_h5 is not None else None

                image_summary, image_onsets_used, image_h5_name_map = _extract_image_identity_streaming(
                    bundle,
                    image_times_f,
                    sample_rate_hz=rate,
                    pre_time=windows.image[0],
                    post_time=windows.image[1],
                    roi_mask=roi_mask,
                    h5_group=image_grp,
                    dtype=np.dtype(dtype),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                change_summary, change_onsets_used = _extract_interval_list_streaming(
                    bundle,
                    change_times_f,
                    sample_rate_hz=rate,
                    pre_time=windows.change[0],
                    post_time=windows.change[1],
                    roi_mask=roi_mask,
                    h5_group=change_grp,
                    dataset_name="traces",
                    onsets_name="onsets_sec",
                    dtype=np.dtype(dtype),
                    compression=compression,
                    compression_opts=compression_opts,
                )
                omission_summary, omission_onsets_used = _extract_interval_list_streaming(
                    bundle,
                    omission_times_f,
                    sample_rate_hz=rate,
                    pre_time=windows.omission[0],
                    post_time=windows.omission[1],
                    roi_mask=roi_mask,
                    h5_group=omission_grp,
                    dataset_name="traces",
                    onsets_name="onsets_sec",
                    dtype=np.dtype(dtype),
                    compression=compression,
                    compression_opts=compression_opts,
                )

                mean_pkg[dmd_key]["image_identity"] = image_summary
                mean_pkg[dmd_key]["change"] = change_summary
                mean_pkg[dmd_key]["omission"] = omission_summary
                mean_pkg[dmd_key]["roi_ids"] = ids
                mean_pkg[dmd_key]["valid_rois_mask"] = roi_mask
                mean_pkg[dmd_key]["signal_transform"] = transform_meta

                if write_sequence:
                    seq_pkg[dmd_key]["image_identity"] = _summarize_change_locked_sequences_streaming(
                        seq_events,
                        bundle,
                        sample_rate_hz=rate,
                        pre_time=windows.image[0],
                        post_time=windows.image[1],
                        roi_mask=roi_mask,
                    )
                    seq_pkg[dmd_key]["roi_ids"] = ids
                    seq_pkg[dmd_key]["valid_rois_mask"] = roi_mask
                    seq_pkg[dmd_key]["signal_transform"] = transform_meta

                image_count_by_id = {img: int(arr["n_events"]) for img, arr in image_summary.items()}
                zero_count_ids = [img for img, cnt in image_count_by_id.items() if cnt == 0]
                qc["per_dmd"][dmd_key] = {
                    "skipped": False,
                    "n_rois_total": int(n_rois_total),
                    "n_rois_kept": int(n_rois_kept),
                    "n_trials_total": int(vs.n_trials),
                    "n_trials_valid": int(np.sum(bundle.trial_valid_mask)),
                    "n_trials_invalid": int(np.sum(~bundle.trial_valid_mask)),
                    "trial_lengths_samples": bundle.trial_lengths_samples.tolist(),
                    "reconstructed_duration_sec": float(bundle.reconstructed_duration_sec),
                    "duration_vs_epoch_error_sec": float(bundle.reconstructed_duration_sec - epoch_duration_sec),
                    "n_image_ids_extracted": int(len(image_count_by_id)),
                    "image_count_by_id": image_count_by_id,
                    "zero_count_image_ids": zero_count_ids,
                    "n_change_events_extracted": int(change_summary["n_events"]),
                    "n_omission_events_extracted": int(omission_summary["n_events"]),
                    "image_h5_name_map": image_h5_name_map,
                    "stimulus_onsets_used_for_extraction": {
                        "image_identity": {k: [float(x) for x in v.tolist()] for k, v in image_onsets_used.items()},
                        "change": [float(x) for x in change_onsets_used.tolist()],
                        "omission": [float(x) for x in omission_onsets_used.tolist()],
                    },
                    "signal_transform": transform_meta,
                }

        finally:
            if h5 is not None:
                h5.close()
            if session_h5 is not None:
                session_h5.close()

    mean_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mean_npz, data=np.array([mean_pkg], dtype=object))
    if write_sequence:
        np.savez_compressed(seq_npz, data=np.array([seq_pkg], dtype=object))
    _write_json(qc_json, qc)

    return {
        "status": "ok",
        "mean_npz": str(mean_npz),
        "single_h5": str(single_h5) if write_single_trials else None,
        "session_traces_h5": str(session_traces_h5) if write_session_traces else None,
        "seq_npz": str(seq_npz) if write_sequence else None,
        "qc_json": str(qc_json),
    }
