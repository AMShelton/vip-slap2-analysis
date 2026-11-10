"""Small, pure trace helpers extracted from the ASAP7 analysis notebooks.

Each function reproduces, line for line, a definition that was copy-pasted
across several notebooks. Notebooks import them under their old private names,
so call sites are unchanged. No behaviour was altered when moving them.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def maybe_get(obj: Any, name: str, default: Any = None) -> Any:
    """Safely get an attribute or dict key from an asset-like object."""
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def interp_nans_1d_f32(y) -> np.ndarray:
    """Linearly interpolate NaN/inf samples of a 1D trace (float32).

    Returns the input unchanged (as float32) if all samples are finite, and zeros if
    fewer than two samples are finite.
    """
    y = np.asarray(y, dtype=np.float32)
    finite = np.isfinite(y)

    if finite.all():
        return y
    if finite.sum() < 2:
        return np.zeros_like(y, dtype=np.float32)

    x = np.arange(y.size)
    y = y.copy()
    y[~finite] = np.interp(x[~finite], x[finite], y[finite]).astype(np.float32)
    return y


def interp_nans_or_nan(x) -> np.ndarray:
    """Linearly interpolate NaN/inf samples (float64); all-NaN if fewer than two are finite."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.all():
        return x
    if finite.sum() < 2:
        return np.full_like(x, np.nan)
    idx = np.arange(x.size)
    y = x.copy()
    y[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return y


def robust_std(x) -> float:
    """MAD-based standard deviation estimate (1.4826 * MAD) over finite samples."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad


def standardize_rows(X, eps: float = 1e-8) -> np.ndarray:
    """Mean/std standardize rows for correlation/coherence work (non-finite -> 0)."""
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > eps), sd, 1.0)
    Z = (X - mu) / sd
    Z[~np.isfinite(Z)] = 0
    return Z.astype(np.float32)


def block_average_trace(x, t, target_rate_hz):
    """Block-average a trace and its timebase down to approximately ``target_rate_hz``."""
    if target_rate_hz is None:
        return np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return x, t
    fs = 1.0 / np.nanmedian(np.diff(t))
    factor = int(max(1, round(fs / float(target_rate_hz))))
    if factor <= 1:
        return x, t
    n = (len(x) // factor) * factor
    if n < factor:
        return x, t
    xb = np.nanmean(x[:n].reshape(-1, factor), axis=1)
    tb = np.nanmean(t[:n].reshape(-1, factor), axis=1)
    return xb, tb


def find_timebase_segments(
    t, fs_hz: float, gap_factor: float = 5.0, min_segment_sec: float = 2.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Find contiguous sample segments from a timebase.

    Boundaries are detected when dt is much larger than the nominal sampling interval.
    Returns ``(segments, audit)`` with ``segments`` an ``(n, 2)`` int64 array of
    ``[start, stop)`` sample indices.
    """
    t = np.asarray(t, dtype=np.float64)
    dt = np.diff(t)

    nominal_dt = 1.0 / float(fs_hz)
    gap_threshold = gap_factor * nominal_dt
    gap_locs = np.where(dt > gap_threshold)[0]

    starts = np.r_[0, gap_locs + 1]
    stops = np.r_[gap_locs + 1, len(t)]

    min_samples = max(2, int(np.ceil(min_segment_sec * fs_hz)))
    keep = (stops - starts) >= min_samples
    starts = starts[keep]
    stops = stops[keep]

    segments = np.column_stack([starts, stops]).astype(np.int64)
    audit = {
        "nominal_dt_sec": nominal_dt,
        "gap_threshold_sec": gap_threshold,
        "n_detected_gaps": int(len(gap_locs)),
        "n_kept_segments": int(len(segments)),
        "min_segment_sec": float(min_segment_sec),
        "max_dt_sec": float(np.nanmax(dt)) if dt.size else np.nan,
        "median_dt_sec": float(np.nanmedian(dt)) if dt.size else np.nan,
    }
    return segments, audit
