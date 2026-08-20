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
  accidentally remove slow voltage dynamics.  Conservative static-F0 and robust
  F0 dF/F transforms are available, with sensor-aware polarity handling for
  quenched indicators such as ASAP7 and brightening indicators such as ASAP8.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import uuid
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.common.epoch_alignment import DEFAULT_MIN_EPOCH_DURATION_SEC
from vip_slap2_analysis.common.clock_qc import compare_slap2_harp_clock
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


def _open_imaging_epochs(
    asset: SessionAssets,
    *,
    min_epoch_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
):
    """Load behavior epochs accepted by the shared duration-QC policy."""
    if asset.qc_dir is None:
        raise ValueError("asset.qc_dir is required to load behavior/imaging_epochs.csv")
    return load_imaging_epochs_csv(
        Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv",
        min_duration_sec=min_epoch_duration_sec,
    )


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



def _array_nbytes(shape: Sequence[int], dtype: np.dtype = np.float32) -> int:
    """Return byte size for an array shape/dtype without allocating it."""
    return int(np.prod(tuple(int(x) for x in shape), dtype=np.int64)) * np.dtype(dtype).itemsize


def _allocate_transform_array(
    shape: Sequence[int],
    *,
    dtype: np.dtype = np.float32,
    memmap_threshold_bytes: int = 512 * 1024 ** 2,
    prefix: str = "vip_slap2_voltage_transform_",
) -> np.ndarray:
    """Allocate a transform array, using a disk-backed memmap for large sessions."""
    shape = tuple(int(x) for x in shape)
    dtype = np.dtype(dtype)
    if _array_nbytes(shape, dtype) >= int(memmap_threshold_bytes):
        path = Path(tempfile.gettempdir()) / f"{prefix}{uuid.uuid4().hex}.npy"
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return np.empty(shape, dtype=dtype)


def _fit_static_f0_model(raw: np.ndarray, *, percentile: float) -> Dict[str, Any]:
    """Fit a per-ROI constant F0 model without expanding it over time."""
    raw2, _was_1d = _as_trace_2d(raw)
    n_rois = int(raw2.shape[0])
    f0_roi = np.empty((n_rois,), dtype=np.float32)
    eps = np.float32(np.finfo(np.float32).eps)
    for r in range(n_rois):
        x = np.asarray(raw2[r, :], dtype=np.float32)
        with np.errstate(all="ignore"):
            f0_val = np.nanpercentile(x, float(percentile)).astype(np.float32)
            fallback = np.nanmedian(np.abs(x)).astype(np.float32)
        if not np.isfinite(fallback) or fallback <= eps:
            fallback = np.float32(1.0)
        if not np.isfinite(f0_val) or abs(float(f0_val)) <= float(eps):
            f0_val = fallback
        f0_roi[r] = np.float32(f0_val)
    return {
        "f0_method": "static_percentile",
        "f0_percentile": float(percentile),
        "f0_is_time_varying": False,
        "f0_per_roi": f0_roi,
    }


def _fit_robust_f0_model(
    raw: np.ndarray,
    *,
    sample_rate_hz: float,
    percentile: float,
    bin_sec: float,
    smooth_sec: float,
) -> Dict[str, Any]:
    """Fit the robust binned F0 model without materializing full-resolution F0."""
    raw2, _was_1d = _as_trace_2d(raw)
    n_rois, n_samples = map(int, raw2.shape)
    rate = float(sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz!r}")

    bin_samples = max(1, int(round(float(bin_sec) * rate)))
    n_bins = int(np.ceil(n_samples / bin_samples))
    if n_bins <= 1:
        return _fit_static_f0_model(raw2, percentile=percentile)

    bin_centers = np.empty(n_bins, dtype=np.float64)
    bin_f0 = np.full((n_rois, n_bins), np.nan, dtype=np.float32)
    for b in range(n_bins):
        start = b * bin_samples
        stop = min(n_samples, (b + 1) * bin_samples)
        if stop <= start:
            continue
        bin_centers[b] = (start + stop - 1) / 2.0
        x = np.asarray(raw2[:, start:stop], dtype=np.float32)
        with np.errstate(all="ignore"):
            bin_f0[:, b] = np.nanpercentile(x, float(percentile), axis=1).astype(np.float32)

    smooth_bins = max(1, int(round(float(smooth_sec) / float(bin_sec))))
    if smooth_bins % 2 == 0:
        smooth_bins += 1

    f0_roi_median = np.nanmedian(np.abs(bin_f0), axis=1).astype(np.float32)
    bad_fallback = ~np.isfinite(f0_roi_median) | (f0_roi_median <= np.finfo(np.float32).eps)
    f0_roi_median[bad_fallback] = np.float32(1.0)

    smoothed_bins = np.full_like(bin_f0, np.nan, dtype=np.float32)
    for r in range(n_rois):
        filled = _fill_nan_1d(bin_f0[r], fallback=float(f0_roi_median[r]))
        smoothed = _nanmoving_median_1d(filled, smooth_bins)
        smoothed_bins[r] = _fill_nan_1d(smoothed, fallback=float(f0_roi_median[r]))

    return {
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


def _fit_epochwise_f0_model(
    raw: np.ndarray,
    *,
    sample_epoch: Sequence[int],
    sample_rate_hz: float,
    method: str,
    percentile: float,
    robust_bin_sec: float,
    robust_smooth_sec: float,
) -> Dict[str, Any]:
    """Fit independent compact F0 models within contiguous acquisition epochs."""
    raw2, _ = _as_trace_2d(raw)
    labels = np.asarray(sample_epoch, dtype=int).reshape(-1)
    if labels.size != raw2.shape[1]:
        raise ValueError("sample_epoch length must match the voltage trace time axis")
    if labels.size == 0 or np.any(labels <= 0):
        raise ValueError("Every voltage sample must have a positive epoch label")

    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate([[0], changes])
    stops = np.concatenate([changes, [labels.size]])
    epoch_models: List[Dict[str, Any]] = []
    epoch_slices: List[Tuple[int, int, int]] = []
    seen = set()
    for start, stop in zip(starts, stops):
        epoch_id = int(labels[start])
        if epoch_id in seen:
            raise ValueError(f"Epoch {epoch_id} appears in disjoint sample blocks")
        seen.add(epoch_id)
        model = _fit_f0_model_chunked(
            raw2[:, int(start):int(stop)],
            sample_rate_hz=sample_rate_hz,
            method=method,
            percentile=percentile,
            robust_bin_sec=robust_bin_sec,
            robust_smooth_sec=robust_smooth_sec,
        )
        model["epoch_id"] = epoch_id
        epoch_models.append(model)
        epoch_slices.append((epoch_id, int(start), int(stop)))

    return {
        "f0_method": "epochwise",
        "base_f0_method": str(method),
        "f0_scope": "epoch",
        "f0_is_time_varying": any(bool(m.get("f0_is_time_varying", False)) for m in epoch_models),
        "epoch_slices": np.asarray(epoch_slices, dtype=np.int64),
        "epoch_models": epoch_models,
        "sample_rate_hz": float(sample_rate_hz),
        "input_shape": tuple(int(x) for x in raw2.shape),
        "f0_storage": "compact_epoch_models",
    }


def _f0_model_n_rois(model: Mapping[str, Any]) -> int:
    method = str(model.get("f0_method", ""))
    if method == "epochwise":
        models = list(model.get("epoch_models", []))
        if not models:
            return 0
        return _f0_model_n_rois(models[0])
    if "f0_per_roi" in model:
        return int(np.asarray(model["f0_per_roi"]).reshape(-1).size)
    if "f0_bin_values" in model:
        return int(np.asarray(model["f0_bin_values"]).shape[0])
    raise ValueError("Cannot infer ROI count from F0 model")


def _fit_f0_model_chunked(
    raw: np.ndarray,
    *,
    sample_rate_hz: float,
    method: str,
    percentile: float,
    robust_bin_sec: float,
    robust_smooth_sec: float,
) -> Dict[str, Any]:
    """Fit a compact F0 model for static or robust voltage dF/F."""
    m = str(method).lower().replace(" ", "_")
    if m in {"static", "static_f0", "static_percentile"}:
        meta = _fit_static_f0_model(raw, percentile=percentile)
    elif m in {"robust", "robust_f0", "robust_baseline"}:
        meta = _fit_robust_f0_model(
            raw,
            sample_rate_hz=sample_rate_hz,
            percentile=percentile,
            bin_sec=robust_bin_sec,
            smooth_sec=robust_smooth_sec,
        )
    else:
        raise ValueError("method must be 'static' or 'robust'")
    meta.update({
        "sample_rate_hz": float(sample_rate_hz),
        "input_shape": tuple(int(x) for x in np.asarray(raw).shape),
        "f0_storage": "compact_model",
    })
    return meta


def _f0_chunk_from_model(model: Mapping[str, Any], *, start: int, stop: int, n_samples: int) -> np.ndarray:
    """Materialize one ROI-by-time F0 chunk from a compact model."""
    method = str(model.get("f0_method", ""))
    start = int(start)
    stop = int(stop)
    if method == "epochwise":
        out = np.full((_f0_model_n_rois(model), max(0, stop - start)), np.nan, dtype=np.float32)
        slices = np.asarray(model.get("epoch_slices", []), dtype=np.int64).reshape(-1, 3)
        models = list(model.get("epoch_models", []))
        if len(models) != len(slices):
            raise ValueError("Malformed epochwise F0 model")
        for (_epoch_id, epoch_start, epoch_stop), submodel in zip(slices, models):
            overlap_start = max(start, int(epoch_start))
            overlap_stop = min(stop, int(epoch_stop))
            if overlap_stop <= overlap_start:
                continue
            local_start = overlap_start - int(epoch_start)
            local_stop = overlap_stop - int(epoch_start)
            out[:, overlap_start - start:overlap_stop - start] = _f0_chunk_from_model(
                submodel,
                start=local_start,
                stop=local_stop,
                n_samples=int(epoch_stop - epoch_start),
            )
        if np.any(~np.isfinite(out)):
            raise ValueError("Requested F0 chunk contains samples not covered by an epoch model")
        return out
    if method == "static_percentile":
        f0_roi = np.asarray(model["f0_per_roi"], dtype=np.float32).reshape(-1)
        return np.repeat(f0_roi[:, None], max(0, stop - start), axis=1).astype(np.float32, copy=False)

    if method == "robust_binned_percentile_moving_median":
        centers = np.asarray(model["f0_bin_centers_sample"], dtype=np.float64).reshape(-1)
        values = np.asarray(model["f0_bin_values"], dtype=np.float32)
        sample_idx = np.arange(start, stop, dtype=np.float64)
        out = np.empty((values.shape[0], sample_idx.size), dtype=np.float32)
        for r in range(values.shape[0]):
            out[r, :] = np.interp(sample_idx, centers, values[r]).astype(np.float32)
        return out

    raise ValueError(f"Unsupported compact F0 model method: {method!r}")


def _compute_dff_from_f0_model_chunked(
    raw: np.ndarray,
    f0_model: Mapping[str, Any],
    *,
    dff_sign: int = -1,
    chunk_samples: int = 262_144,
) -> np.ndarray:
    """Compute voltage-indicator-polarity-corrected dF/F in chunks."""
    sign = _validate_dff_sign(dff_sign)
    raw2, was_1d = _as_trace_2d(raw)
    n_rois, n_samples = map(int, raw2.shape)
    out = _allocate_transform_array((n_rois, n_samples), dtype=np.float32)
    for start in range(0, n_samples, int(chunk_samples)):
        stop = min(n_samples, start + int(chunk_samples))
        raw_chunk = np.asarray(raw2[:, start:stop], dtype=np.float32)
        f0_chunk = _f0_chunk_from_model(f0_model, start=start, stop=stop, n_samples=n_samples)
        f0_chunk = _safe_f0_values(f0_chunk, raw_chunk)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, start:stop] = sign * (raw_chunk - f0_chunk) / f0_chunk
    return _restore_trace_dim(out, was_1d)


def _metadata_without_large_arrays(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Return JSON-friendly compact-model metadata without large numeric arrays."""
    out = dict(meta)
    if str(out.get("f0_method", "")) == "epochwise":
        out["epoch_models"] = [_metadata_without_large_arrays(m) for m in out.get("epoch_models", [])]
    for key in ("f0_per_roi", "f0_bin_centers_sample", "f0_bin_values", "epoch_slices"):
        if key in out:
            arr = np.asarray(out[key])
            out[f"{key}_shape"] = tuple(int(x) for x in arr.shape)
            if key == "epoch_slices":
                out[key] = arr.astype(int).tolist()
            else:
                out.pop(key, None)
    return out


def _write_f0_model_group(group: h5py.Group, model: Mapping[str, Any]) -> None:
    """Persist a compact static, robust, or epochwise F0 model recursively."""
    group.attrs["metadata_json"] = json.dumps(_metadata_without_large_arrays(model), default=_json_default)
    for key in ("f0_per_roi", "f0_bin_centers_sample", "f0_bin_values", "epoch_slices"):
        if key in model:
            group.create_dataset(key, data=np.asarray(model[key]))
    if str(model.get("f0_method", "")) == "epochwise":
        epochs_group = group.create_group("epochs")
        slices = np.asarray(model.get("epoch_slices", []), dtype=np.int64).reshape(-1, 3)
        for row, submodel in zip(slices, model.get("epoch_models", [])):
            epoch_id = int(row[0])
            _write_f0_model_group(epochs_group.create_group(f"epoch_{epoch_id:04d}"), submodel)


def _normalize_indicator_text(indicator: Optional[Any]) -> str:
    """Return a lowercase indicator string suitable for simple matching."""
    if indicator is None:
        return ""
    try:
        if isinstance(indicator, (np.floating, float)) and not np.isfinite(indicator):
            return ""
    except TypeError:
        pass
    text = str(indicator).strip().lower()
    if text in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def _validate_dff_sign(dff_sign: int) -> int:
    """Validate the sign used to map fluorescence dF/F onto voltage dF/F."""
    sign = int(dff_sign)
    if sign not in {-1, 1}:
        raise ValueError(f"dff_sign must be -1 or 1, got {dff_sign!r}")
    return sign


def resolve_voltage_dff_polarity(
    indicator: Optional[Any] = None,
    *,
    dff_polarity: str = "auto",
) -> Dict[str, Any]:
    """Resolve voltage-indicator polarity for dF/F calculation.

    The voltage pipeline stores ``dff`` so that positive values correspond to
    positive membrane-voltage deflections.  Some indicators, including ASAP7,
    are quenched by depolarization and therefore need inverted fluorescence
    dF/F.  Others, including ASAP8, brighten with depolarization and therefore
    use standard fluorescence dF/F.

    Parameters
    ----------
    indicator
        Sensor name such as ``"ASAP7y"`` or ``"ASAP8"``.  Only used when
        ``dff_polarity='auto'``.
    dff_polarity
        ``'auto'`` infers polarity from ``indicator``.  Explicit aliases are
        accepted for reproducibility: ``'depolarization_decreases_fluorescence'``
        or ``'inverted'`` use ``(F0 - F) / F0``; ``'depolarization_increases_fluorescence'``
        or ``'standard'`` use ``(F - F0) / F0``.

    Returns
    -------
    dict
        JSON-friendly polarity metadata, including ``dff_sign`` where ``+1``
        means standard fluorescence dF/F and ``-1`` means inverted fluorescence
        dF/F.  Unknown indicators retain the historical ASAP7-compatible
        inverted default.
    """
    indicator_text = _normalize_indicator_text(indicator)
    requested = str(dff_polarity or "auto").strip().lower().replace(" ", "_").replace("-", "_")

    increases_aliases = {
        "increase",
        "increases",
        "brightens",
        "standard",
        "positive",
        "non_inverted",
        "not_inverted",
        "depolarization_increases_fluorescence",
        "depolarization_brightens",
        "f_minus_f0_over_f0",
    }
    decreases_aliases = {
        "decrease",
        "decreases",
        "quenched",
        "inverted",
        "inverse",
        "negative",
        "depolarization_decreases_fluorescence",
        "depolarization_quenches",
        "f0_minus_f_over_f0",
    }

    source = "explicit"
    if requested in increases_aliases:
        sign = 1
        response = "depolarization_increases_fluorescence"
        formula = "dff = (F - F0) / F0"
    elif requested in decreases_aliases:
        sign = -1
        response = "depolarization_decreases_fluorescence"
        formula = "dff = (F0 - F) / F0"
    elif requested == "auto":
        source = "indicator"
        if "asap8" in indicator_text:
            sign = 1
            response = "depolarization_increases_fluorescence"
            formula = "dff = (F - F0) / F0"
        elif "asap7" in indicator_text:
            sign = -1
            response = "depolarization_decreases_fluorescence"
            formula = "dff = (F0 - F) / F0"
        else:
            # Preserve backward compatibility for old extractions where the
            # indicator was implicit and the voltage dFF convention was ASAP7.
            source = "legacy_default"
            sign = -1
            response = "depolarization_decreases_fluorescence"
            formula = "dff = (F0 - F) / F0"
    else:
        raise ValueError(
            "dff_polarity must be 'auto', an explicit increase/standard alias, "
            "or an explicit decrease/inverted alias."
        )

    return {
        "indicator": indicator_text or None,
        "dff_polarity_requested": requested,
        "dff_polarity_source": source,
        "fluorescence_response_to_depolarization": response,
        "dff_sign": int(sign),
        "transform": formula,
        "polarity": (
            "Voltage-indicator-polarity corrected; positive dff corresponds to "
            "positive membrane-voltage deflection."
        ),
    }


def _metadata_indicator_for_dmd(asset: SessionAssets, dmd: int) -> Optional[str]:
    """Return the best available indicator metadata for one DMD."""
    meta = getattr(asset, "metadata", {}) or {}
    d = int(dmd)
    candidate_keys = (
        f"dmd{d}_indicator",
        f"dmd_{d}_indicator",
        f"indicator_dmd{d}",
        f"indicator_dmd_{d}",
        f"indicator{d}",
        f"indicator_{d}",
        "voltage_indicator",
        "indicator",
        "indicator1",
        "indicator2",
    )
    for key in candidate_keys:
        if key in meta:
            text = _normalize_indicator_text(meta.get(key))
            if text:
                return str(meta.get(key))
    return None


def _resolve_voltage_dff_polarity_for_dmd(
    asset: SessionAssets,
    dmd: int,
    *,
    voltage_indicator: Optional[str] = None,
    dff_polarity: str = "auto",
) -> Dict[str, Any]:
    """Resolve dF/F polarity for a DMD using explicit then asset metadata."""
    indicator = voltage_indicator if _normalize_indicator_text(voltage_indicator) else _metadata_indicator_for_dmd(asset, dmd)
    out = resolve_voltage_dff_polarity(indicator, dff_polarity=dff_polarity)
    out["dmd"] = int(dmd)
    return out


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
        if raw.ndim == 2 and raw.shape[0] == 1 and f0.size == raw.shape[1]:
            fallback_scalar = np.nanmedian(np.abs(raw[0])).astype(np.float32)
            if not np.isfinite(fallback_scalar) or fallback_scalar <= eps:
                fallback_scalar = np.float32(1.0)
            bad = ~np.isfinite(f0) | (np.abs(f0) <= eps)
            if np.any(bad):
                f0 = f0.copy()
                f0[bad] = fallback_scalar
            return f0

        fallback = np.nanmedian(np.abs(raw), axis=1).astype(np.float32)
        fallback_bad = ~np.isfinite(fallback) | (fallback <= eps)
        fallback[fallback_bad] = np.float32(1.0)
        bad = ~np.isfinite(f0) | (np.abs(f0) <= eps)
        if np.any(bad):
            f0 = f0.copy()
            if fallback.size == f0.size:
                f0[bad] = fallback[bad]
            else:
                f0[bad] = np.nanmedian(fallback).astype(np.float32)
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


def compute_voltage_dff(
    raw_f: np.ndarray,
    f0: np.ndarray,
    *,
    indicator: Optional[Any] = None,
    dff_polarity: str = "auto",
) -> np.ndarray:
    """Compute voltage-indicator-polarity-corrected dF/F.

    The returned signal is positive for positive membrane-voltage deflections,
    not necessarily for positive fluorescence changes.  ASAP7-like quenched
    sensors use inverted dF/F, ``(F0 - F) / F0``.  ASAP8-like brightening sensors
    use standard dF/F, ``(F - F0) / F0``.
    """
    polarity = resolve_voltage_dff_polarity(indicator, dff_polarity=dff_polarity)
    sign = int(polarity["dff_sign"])
    raw = np.asarray(raw_f, dtype=np.float32)
    f0_arr = np.asarray(f0, dtype=np.float32)
    f0_arr = _safe_f0_values(f0_arr, raw if raw.ndim == 2 else raw[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = sign * (raw - f0_arr) / f0_arr
    return np.asarray(dff, dtype=np.float32)


def transform_voltage_signal(
    raw_f: np.ndarray,
    *,
    sample_rate_hz: float,
    method: str = "robust",
    percentile: float = 50.0,
    robust_bin_sec: float = 5.0,
    robust_smooth_sec: float = 180.0,
    indicator: Optional[Any] = None,
    dff_polarity: str = "auto",
) -> Dict[str, Any]:
    """Return raw fluorescence, F0, and polarity-corrected voltage dF/F."""
    f0, f0_meta = compute_voltage_f0(
        raw_f,
        sample_rate_hz=sample_rate_hz,
        method=method,
        percentile=percentile,
        robust_bin_sec=robust_bin_sec,
        robust_smooth_sec=robust_smooth_sec,
    )
    polarity = resolve_voltage_dff_polarity(indicator, dff_polarity=dff_polarity)
    dff = compute_voltage_dff(raw_f, f0, indicator=indicator, dff_polarity=dff_polarity)
    meta = {
        "trace_signal": f"dff_{method}_f0",
        **polarity,
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
    indicator: Optional[Any] = None,
    dff_polarity: str = "auto",
    f0_scope: str = "epoch",
) -> Tuple[ReconstructedTraceBundle, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Apply an optional voltage transform to a reconstructed bundle.

    dF/F transforms are computed from a compact F0 model and materialized in
    chunks.  This avoids keeping full-session ``raw_f``, ``f0``, and ``dff``
    arrays in RAM simultaneously for long voltage recordings.
    """
    output_signal, f0_method = _parse_trace_signal(trace_signal)
    if output_signal == "raw_f":
        return bundle, {
            "trace_signal": trace_signal,
            "output_signal": "raw_f",
            "transform": "none",
            "indicator": _normalize_indicator_text(indicator) or None,
        }, None

    polarity = resolve_voltage_dff_polarity(indicator, dff_polarity=dff_polarity)

    scope = str(f0_scope).strip().lower()
    if scope not in {"epoch", "session"}:
        raise ValueError("f0_scope must be 'epoch' or 'session'")
    sample_epoch = getattr(bundle, "sample_epoch", None)
    if scope == "epoch":
        if sample_epoch is None:
            # Backward-compatible single-epoch bundles used in direct transforms/tests.
            sample_epoch = np.ones((bundle.traces.shape[1],), dtype=int)
        labels = np.asarray(sample_epoch, dtype=int).reshape(-1)
        unique_epochs = np.unique(labels[labels > 0])
        if unique_epochs.size <= 1:
            f0_model = _fit_f0_model_chunked(
                bundle.traces,
                sample_rate_hz=sample_rate_hz,
                method=str(f0_method),
                percentile=f0_percentile,
                robust_bin_sec=robust_f0_bin_sec,
                robust_smooth_sec=robust_f0_smooth_sec,
            )
            f0_model["f0_scope"] = "epoch"
            f0_model["epoch_id"] = int(unique_epochs[0]) if unique_epochs.size else 1
        else:
            f0_model = _fit_epochwise_f0_model(
                bundle.traces,
                sample_epoch=labels,
                sample_rate_hz=sample_rate_hz,
                method=str(f0_method),
                percentile=f0_percentile,
                robust_bin_sec=robust_f0_bin_sec,
                robust_smooth_sec=robust_f0_smooth_sec,
            )
    else:
        f0_model = _fit_f0_model_chunked(
            bundle.traces,
            sample_rate_hz=sample_rate_hz,
            method=str(f0_method),
            percentile=f0_percentile,
            robust_bin_sec=robust_f0_bin_sec,
            robust_smooth_sec=robust_f0_smooth_sec,
        )
        f0_model["f0_scope"] = "session"
    dff = _compute_dff_from_f0_model_chunked(
        bundle.traces,
        f0_model,
        dff_sign=int(polarity["dff_sign"]),
    )
    out = replace(bundle, traces=np.asarray(dff, dtype=np.float32))
    meta = {
        "trace_signal": f"dff_{f0_method}_f0",
        "trace_signal_requested": trace_signal,
        "output_signal": "dff",
        "chunked_transform": True,
        "f0_scope": scope,
        **polarity,
        **_metadata_without_large_arrays(f0_model),
    }
    payload = {
        "raw_f": bundle.traces,
        "dff": out.traces,
        "f0_model": f0_model,
    }
    return out, meta, payload


def _write_h5_2d_chunked(
    group: h5py.Group,
    name: str,
    arr: np.ndarray,
    *,
    dtype: np.dtype,
    compression: Optional[str],
    compression_opts: Optional[int],
    chunk_samples: int = 8192,
) -> h5py.Dataset:
    """Write an ROI-by-time array to HDF5 without materializing a second copy."""
    arr2, _was_1d = _as_trace_2d(arr)
    chunks = (1, max(1, min(int(arr2.shape[1]), int(chunk_samples)))) if arr2.size else None
    ds = group.create_dataset(
        name,
        shape=arr2.shape,
        dtype=np.dtype(dtype),
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=(compression is not None),
    )
    for start in range(0, int(arr2.shape[1]), int(chunk_samples)):
        stop = min(int(arr2.shape[1]), start + int(chunk_samples))
        ds[:, start:stop] = np.asarray(arr2[:, start:stop], dtype=dtype)
    return ds


def _write_h5_f0_from_model(
    group: h5py.Group,
    name: str,
    model: Mapping[str, Any],
    *,
    n_samples: int,
    dtype: np.dtype,
    compression: Optional[str],
    compression_opts: Optional[int],
    chunk_samples: int = 8192,
) -> h5py.Dataset:
    """Write full-resolution F0 to HDF5 from a compact model in chunks."""
    n_rois = _f0_model_n_rois(model)
    chunks = (1, max(1, min(int(n_samples), int(chunk_samples)))) if n_samples else None
    ds = group.create_dataset(
        name,
        shape=(n_rois, int(n_samples)),
        dtype=np.dtype(dtype),
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=(compression is not None),
    )
    for start in range(0, int(n_samples), int(chunk_samples)):
        stop = min(int(n_samples), start + int(chunk_samples))
        ds[:, start:stop] = _f0_chunk_from_model(model, start=start, stop=stop, n_samples=n_samples).astype(dtype, copy=False)
    return ds


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

    Datasets are written chunk-by-chunk so long voltage sessions do not require a
    second full-size in-memory array during HDF5 export.
    """
    grp = h5.create_group(dmd_key)
    grp.attrs["schema"] = "ROI-by-time full-session voltage traces"
    grp.attrs["signal_transform_json"] = json.dumps(transform_meta, default=_json_default)
    if transform_payload is not None and "f0_model" in transform_payload:
        grp.attrs["f0_model_json"] = json.dumps(_metadata_without_large_arrays(transform_payload["f0_model"]), default=_json_default)
        model_grp = grp.create_group("f0_model")
        model = transform_payload["f0_model"]
        _write_f0_model_group(model_grp, model)
    if getattr(bundle, "metadata", None):
        grp.attrs["reconstruction_metadata_json"] = json.dumps(bundle.metadata, default=_json_default)
    grp.attrs["session_start_sec"] = float(bundle.session_start_sec)
    grp.attrs["session_end_sec"] = float(bundle.session_end_sec)
    grp.attrs["reconstructed_duration_sec"] = float(bundle.reconstructed_duration_sec)
    grp.create_dataset("timebase_sec", data=np.asarray(bundle.timebase_sec, dtype=np.float64))
    grp.create_dataset("roi_ids", data=np.asarray(roi_ids, dtype="S"))
    grp.create_dataset("valid_rois_mask", data=np.asarray(roi_mask, dtype=bool))
    grp.create_dataset("trial_valid_mask", data=np.asarray(bundle.trial_valid_mask, dtype=bool))
    grp.create_dataset("trial_lengths_samples", data=np.asarray(bundle.trial_lengths_samples, dtype=np.int64))
    grp.create_dataset("trial_starts_sec", data=np.asarray(bundle.trial_starts_sec, dtype=np.float64))
    if getattr(bundle, "sample_epoch", None) is not None:
        grp.create_dataset("sample_epoch", data=np.asarray(bundle.sample_epoch, dtype=np.int16))
    trial_epoch = (getattr(bundle, "metadata", {}) or {}).get("trial_epoch", None)
    if trial_epoch is not None:
        grp.create_dataset("trial_epoch", data=np.asarray(trial_epoch, dtype=np.int16))

    if transform_payload is None:
        _write_h5_2d_chunked(
            grp,
            "raw_f",
            bundle.traces,
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        return

    _write_h5_2d_chunked(
        grp,
        "raw_f",
        np.asarray(transform_payload["raw_f"]),
        dtype=dtype,
        compression=compression,
        compression_opts=compression_opts,
    )
    if "f0_model" in transform_payload:
        _write_h5_f0_from_model(
            grp,
            "f0",
            transform_payload["f0_model"],
            n_samples=int(np.asarray(transform_payload["raw_f"]).shape[1]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
    elif "f0" in transform_payload:
        _write_h5_2d_chunked(
            grp,
            "f0",
            np.asarray(transform_payload["f0"]),
            dtype=dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
    _write_h5_2d_chunked(
        grp,
        "dff",
        np.asarray(transform_payload["dff"]),
        dtype=dtype,
        compression=compression,
        compression_opts=compression_opts,
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
        if "sample_epoch" in grp:
            out["sample_epoch"] = grp["sample_epoch"][:]
        for key in ("raw_f", "f0", "dff"):
            if key in grp:
                out[key] = grp[key][int(roi_index), :]
        out["metadata"] = json.loads(grp.attrs.get("signal_transform_json", "{}"))
        out["reconstruction_metadata"] = json.loads(
            grp.attrs.get("reconstruction_metadata_json", "{}")
        )
    return out


def compute_voltage_roi_transform_from_asset(
    asset: SessionAssets,
    *,
    dmd: int,
    roi_index: int,
    sample_rate_hz: Optional[float] = None,
    default_sample_rate_hz: float = 10_800.0,
    epoch_start_sec: Optional[float] = None,
    epoch_end_sec: Optional[float] = None,
    trace_mode: str = "trial",
    drop_discarded: bool = True,
    timebase_strategy: str = "sample_rate",
    max_timebase_error_sec: float = 0.5,
    min_epoch_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
    f0_method: str = "robust",
    f0_percentile: float = 50.0,
    robust_f0_bin_sec: float = 5.0,
    robust_f0_smooth_sec: float = 180.0,
    f0_scope: str = "epoch",
    voltage_indicator: Optional[str] = None,
    dff_polarity: str = "auto",
    strict_epoch_match: bool = True,
) -> Dict[str, Any]:
    """Compute raw/F0/dFF for one ROI directly from a session asset.

    This helper is meant for interactive inspection.  It reconstructs the full
    DMD session trace using the same code path as batch extraction, selects one
    ROI, and computes static or robust F0 plus indicator-polarity-corrected dF/F.
    """
    # Always load all behavior-derived imaging epochs. Even when callers supply
    # outer session bounds, epoch-scoped F0 and strict alignment require the
    # internal acquisition boundaries and imaging-off gaps.
    epoch_df = _open_imaging_epochs(
        asset,
        min_epoch_duration_sec=min_epoch_duration_sec,
    )
    if epoch_start_sec is None:
        epoch_start_sec = float(epoch_df.iloc[0]["start_time"])
    if epoch_end_sec is None and "end_time" in epoch_df.columns:
        epoch_end_sec = float(epoch_df.iloc[-1]["end_time"])

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
            epoch_end_sec=(float(epoch_end_sec) if epoch_end_sec is not None else None),
            epoch_df=epoch_df,
            drop_discarded=drop_discarded,
            dtype=np.float32,
            trace_mode=trace_mode,
            timebase_strategy=timebase_strategy,
            max_timebase_error_sec=max_timebase_error_sec,
            strict_epoch_match=strict_epoch_match,
            min_epoch_duration_sec=min_epoch_duration_sec,
        )

    if roi_index < 0 or roi_index >= bundle.traces.shape[0]:
        raise IndexError(f"roi_index={roi_index} is out of bounds for {bundle.traces.shape[0]} ROIs")

    polarity = _resolve_voltage_dff_polarity_for_dmd(
        asset,
        int(dmd),
        voltage_indicator=voltage_indicator,
        dff_polarity=dff_polarity,
    )
    transformed_bundle, transform_meta, payload = _transform_voltage_bundle(
        bundle,
        trace_signal=f"dff_{str(f0_method).lower()}_f0",
        sample_rate_hz=rate,
        f0_percentile=f0_percentile,
        robust_f0_bin_sec=robust_f0_bin_sec,
        robust_f0_smooth_sec=robust_f0_smooth_sec,
        indicator=polarity.get("indicator"),
        dff_polarity=dff_polarity,
        f0_scope=f0_scope,
    )
    assert payload is not None
    f0_roi = _f0_chunk_from_model(
        payload["f0_model"], start=0, stop=bundle.traces.shape[1], n_samples=bundle.traces.shape[1]
    )[int(roi_index)]
    return {
        "timebase_sec": np.asarray(bundle.timebase_sec, dtype=np.float64),
        "sample_epoch": (
            np.asarray(bundle.sample_epoch, dtype=np.int16)
            if getattr(bundle, "sample_epoch", None) is not None
            else None
        ),
        "raw_f": np.asarray(bundle.traces[int(roi_index)], dtype=np.float32),
        "f0": np.asarray(f0_roi, dtype=np.float32),
        "dff": np.asarray(transformed_bundle.traces[int(roi_index)], dtype=np.float32),
        "metadata": transform_meta,
        "dmd": int(dmd),
        "roi_index": int(roi_index),
        "sample_rate_hz": float(rate),
        "reconstruction_metadata": dict(getattr(bundle, "metadata", {}) or {}),
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
    if getattr(bundle, "sample_epoch", None) is not None:
        labels = np.asarray(bundle.sample_epoch[start:stop], dtype=int)
        if labels.size == 0 or np.unique(labels[labels > 0]).size != 1:
            return None
    if timebase_sec is not None and stop - start > 1:
        dt = np.diff(timebase_sec[start:stop])
        if np.any(dt <= 0) or np.any(dt > 5.0 / float(sample_rate_hz)):
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
    f0_scope: str = "epoch",
    voltage_indicator: Optional[str] = None,
    dff_polarity: str = "auto",
    trace_mode: str = "trial",
    drop_discarded: bool = True,
    timebase_strategy: str = "sample_rate",
    max_timebase_error_sec: float = 0.5,
    strict_epoch_match: bool = True,
    min_epoch_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
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
        ``'dff_robust_f0'`` compute indicator-polarity-corrected dF/F on the
        reconstructed full-session ROI trace before event snippets are extracted.
    f0_percentile
        Percentile used for static and robust F0 estimation.
    robust_f0_bin_sec, robust_f0_smooth_sec
        Parameters for the robust full-session F0 model.  The defaults estimate a
        binned robust fluorescence level every 5 s and smooth those estimates
        over 180 s, which is intentionally conservative for slow voltage dynamics.
    voltage_indicator
        Optional explicit sensor name, such as ``'ASAP7y'`` or ``'ASAP8'``.  If
        omitted, the session metadata indicator fields are used per DMD.
    dff_polarity
        ``'auto'`` infers whether depolarization increases or decreases
        fluorescence from the indicator name.  ASAP7-like indicators are treated
        as quenched and use ``(F0 - F) / F0``; ASAP8-like indicators are treated
        as brightening and use ``(F - F0) / F0``.  Explicit aliases such as
        ``'inverted'`` or ``'standard'`` can be used to override inference.
    trace_mode
        Passed to ``VoltageSummary.get_roi_traces``.  Current voltage outputs are
        trial-based, so this should usually be ``'trial'``.
    drop_discarded
        Remove samples marked by ``discardFrames`` before event extraction.
    timebase_strategy
        Backward-compatible timebase option. When HARP imaging epochs are present
        (the standard processing path), reconstruction always preserves the nominal
        SLAP2 sample interval, reconciles each source epoch independently, and clips
        unsupported tails rather than stretching physiology to HARP. The option is
        only consulted by the legacy no-epoch fallback.
    max_timebase_error_sec
        Duration-mismatch threshold used by ``timebase_strategy='auto'``.
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
    epoch_df = _open_imaging_epochs(
        asset, min_epoch_duration_sec=min_epoch_duration_sec
    )
    epoch_start_sec = float(epoch_df.iloc[0]["start_time"])
    epoch_end_sec = float(epoch_df.iloc[-1]["end_time"])
    epoch_session_span_sec = float(epoch_end_sec - epoch_start_sec)
    epoch_acquired_duration_sec = float(
        np.sum(epoch_df["end_time"].to_numpy(dtype=float) - epoch_df["start_time"].to_numpy(dtype=float))
    )
    epoch_gap_duration_sec = float(epoch_session_span_sec - epoch_acquired_duration_sec)

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
        clock_qc = compare_slap2_harp_clock(
            vs,
            behavior_qc_dir=Path(asset.qc_dir) / "behavior",
            harp_df_csv=asset.harp_df_csv,
        )

        summary_mat = asset.get_asset("voltage", "summary_mat") if hasattr(asset, "get_asset") else None
        trace_h5 = asset.get_asset("voltage", "trace_h5") if hasattr(asset, "get_asset") else None
        base_meta: Dict[str, Any] = {
            "schema_version": "0.1.1",
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
            "timebase_strategy": str(timebase_strategy),
            "max_timebase_error_sec": float(max_timebase_error_sec),
            "use_roi_qc": bool(use_roi_qc),
            "write_single_trials": bool(write_single_trials),
            "write_sequence": bool(write_sequence),
            "write_session_traces": bool(write_session_traces),
            "f0_percentile": float(f0_percentile),
            "robust_f0_bin_sec": float(robust_f0_bin_sec),
            "robust_f0_smooth_sec": float(robust_f0_smooth_sec),
            "f0_scope": str(f0_scope),
            "strict_epoch_match": bool(strict_epoch_match),
            "min_epoch_duration_sec": float(min_epoch_duration_sec),
            "epoch_duration_qc_policy": "accept_duration_greater_than_or_equal_to_threshold",
            "n_imaging_epochs": int(len(epoch_df)),
            "epoch_starts_sec": epoch_df["start_time"].astype(float).tolist(),
            "epoch_ends_sec": epoch_df["end_time"].astype(float).tolist(),
            "epoch_acquired_duration_sec": epoch_acquired_duration_sec,
            "epoch_session_span_sec": epoch_session_span_sec,
            "epoch_gap_duration_sec": epoch_gap_duration_sec,
            "voltage_indicator_override": _normalize_indicator_text(voltage_indicator) or None,
            "dff_polarity_requested": str(dff_polarity),
            "epoch_start_sec": float(epoch_start_sec),
            "epoch_end_sec": float(epoch_end_sec),
            "voltage_summary_layout": getattr(vs, "layout", None),
            "voltage_summary_metadata": getattr(vs, "metadata", {}),
            "voltage_h5_attrs": getattr(vs, "h5_attrs", {}),
            "source_acquisition_metadata_available": bool(
                any(vs.get_dmd_epoch_metadata(dmd) for dmd in range(1, int(vs.n_dmds) + 1))
            ),
            "slap2_harp_clock_qc": clock_qc,
        }

        mean_pkg: Dict[str, Any] = {"metadata": base_meta, "timebase_sec": tvecs, "DMD1": {}, "DMD2": {}}
        seq_pkg: Dict[str, Any] = {"metadata": base_meta, "timebase_sec": {"image": tvecs["image"]}, "DMD1": {}, "DMD2": {}}
        qc: Dict[str, Any] = {
            "schema_version": "0.1.1",
            "session_id": asset.session_id,
            "summary_mat": base_meta["summary_mat"],
            "trace_h5": base_meta["trace_h5"],
            "bonsai_event_log_csv": str(asset.bonsai_event_log_csv),
            "sample_rate_hz": float(rate),
            "trace_signal": str(trace_signal),
            "trace_mode": str(trace_mode),
            "trace_suffix": suffix,
            "voltage_indicator_override": _normalize_indicator_text(voltage_indicator) or None,
            "dff_polarity_requested": str(dff_polarity),
            "f0_scope": str(f0_scope),
            "strict_epoch_match": bool(strict_epoch_match),
            "min_epoch_duration_sec": float(min_epoch_duration_sec),
            "epoch_duration_qc_policy": "accept_duration_greater_than_or_equal_to_threshold",
            "n_imaging_epochs": int(len(epoch_df)),
            "epoch_acquired_duration_sec": epoch_acquired_duration_sec,
            "epoch_session_span_sec": epoch_session_span_sec,
            "epoch_gap_duration_sec": epoch_gap_duration_sec,
            "source_acquisition_metadata_available": base_meta["source_acquisition_metadata_available"],
            "slap2_harp_clock_qc": clock_qc,
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
            # Backward-compatible alias: historically this represented the outer
            # first-start to last-end span, not acquired imaging duration.
            "epoch_duration_sec": float(epoch_session_span_sec),
            "per_dmd": {},
        }

        h5: Optional[h5py.File] = None
        session_h5: Optional[h5py.File] = None
        if write_single_trials:
            single_h5.parent.mkdir(parents=True, exist_ok=True)
            h5 = h5py.File(single_h5, "w")
            _metadata_json_attr(h5, base_meta)
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
                    epoch_end_sec=epoch_end_sec,
                    epoch_df=epoch_df,
                    drop_discarded=drop_discarded,
                    dtype=dtype,
                    trace_mode=trace_mode,
                    timebase_strategy=timebase_strategy,
                    max_timebase_error_sec=max_timebase_error_sec,
                    strict_epoch_match=strict_epoch_match,
                    min_epoch_duration_sec=min_epoch_duration_sec,
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
                    indicator=_resolve_voltage_dff_polarity_for_dmd(
                        asset,
                        dmd,
                        voltage_indicator=voltage_indicator,
                        dff_polarity=dff_polarity,
                    ).get("indicator"),
                    dff_polarity=dff_polarity,
                    f0_scope=f0_scope,
                )

                event_rate = float((getattr(bundle, "metadata", {}) or {}).get("alignment_sample_rate_hz", rate))
                event_tvecs = _time_vectors(windows, event_rate)
                mean_pkg["timebase_sec"] = event_tvecs
                seq_pkg["timebase_sec"] = {"image": event_tvecs["image"]}
                if h5 is not None and "timebase_sec" not in h5:
                    h5.create_dataset("timebase_sec/image", data=event_tvecs["image"].astype(np.float64))
                    h5.create_dataset("timebase_sec/change", data=event_tvecs["change"].astype(np.float64))
                    h5.create_dataset("timebase_sec/omission", data=event_tvecs["omission"].astype(np.float64))

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
                    if getattr(bundle, "metadata", None):
                        dmd_h5.attrs["reconstruction_metadata_json"] = json.dumps(bundle.metadata, default=_json_default)
                    dmd_h5.create_dataset("roi_ids", data=np.asarray(ids, dtype="S"))
                    dmd_h5.create_dataset("valid_rois_mask", data=roi_mask.astype(bool))

                image_grp = dmd_h5.create_group("image_identity") if dmd_h5 is not None else None
                change_grp = dmd_h5.create_group("change") if dmd_h5 is not None else None
                omission_grp = dmd_h5.create_group("omission") if dmd_h5 is not None else None

                image_summary, image_onsets_used, image_h5_name_map = _extract_image_identity_streaming(
                    bundle,
                    image_times_f,
                    sample_rate_hz=event_rate,
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
                    sample_rate_hz=event_rate,
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
                    sample_rate_hz=event_rate,
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
                        sample_rate_hz=event_rate,
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
                    "duration_vs_session_span_error_sec": float(
                        bundle.reconstructed_duration_sec - epoch_session_span_sec
                    ),
                    "reconstruction_metadata": dict(getattr(bundle, "metadata", {}) or {}),
                    "source_acquisition_epochs": vs.get_dmd_epoch_metadata(dmd),
                    "alignment_sample_rate_hz": float(event_rate),
                    "timebase_strategy_used": str((getattr(bundle, "metadata", {}) or {}).get("timebase_strategy_used", "unknown")),
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
