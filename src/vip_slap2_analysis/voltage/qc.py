"""Quality-control utilities for SLAP2 ASAP voltage extraction outputs.

This module evaluates the split ``dendriticVoltageSummary*.mat`` +
``dendriticVoltageTraces*.h5`` outputs produced by ``extractDendrites_new.m``.
It mirrors the glutamate/calcium QC style: callers pass a generic
``SessionAssets`` object, the function resolves modality assets through
``asset.modality_assets["voltage"]``, computes structural and trace-quality
metrics, and writes JSON/CSV/NPY artifacts under ``qc/voltage``.

The first-pass QC intentionally works on raw extracted fluorescence traces. It
checks whether the files are readable, whether trace blocks obey the voltage
pipeline shape convention ``(n_samples, n_rois)``, and whether ROIs have adequate
finite coverage and robust dynamic range.  It does not compute the canonical
inverted dF/F0 signal; baseline selection is analysis-critical for voltage data
and should be handled by a later preprocessing step.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.voltage.postprocess import (
    _as_time_by_roi,
    _discard_mask_for_trace,
    _expected_n_rois,
    _valid_trials_for_dmd,
    load_voltage_summary_from_asset,
    resolve_voltage_sample_rate_hz,
)
from vip_slap2_analysis.voltage.summary import VoltageSummary


DEFAULT_VOLTAGE_SAMPLE_RATE_HZ = 10_800.0


@dataclass
class VoltageQcThresholds:
    """Conservative thresholds for first-pass voltage ROI inclusion.

    These thresholds are deliberately permissive: the goal is to exclude missing,
    empty, saturated, or numerically pathological ROIs while preserving real
    voltage dynamics for downstream baseline, dF/F0, oscillation, and event-aligned
    analyses.
    """

    min_valid_trial_fraction: float = 0.50
    min_finite_fraction: float = 0.90
    min_trace_abs_p99: float = 1e-6
    min_trace_range_robust: float = 1e-6
    max_abs_median: float = 1e12
    max_nan_fraction: float = 0.10
    min_quality_percentile_within_dmd: float = 0.25
    min_quality_score: float = 0.0


@dataclass
class VoltageQcResult:
    """Container returned by :func:`run_voltage_qc`."""

    qc_table: pd.DataFrame
    trial_table: pd.DataFrame
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    output_dir: str
    keep_masks: Dict[str, Optional[str]]


# -----------------------------------------------------------------------------
# Robust scalar helpers
# -----------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy/pandas scalar values and paths."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _finite_values(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_percentile(x: np.ndarray, q: float) -> float:
    vals = _finite_values(x)
    if vals.size == 0:
        return np.nan
    return float(np.percentile(vals, q))


def _mad(x: np.ndarray) -> float:
    vals = _finite_values(x)
    if vals.size == 0:
        return np.nan
    med = np.median(vals)
    return float(np.median(np.abs(vals - med)))


def _robust_sigma(x: np.ndarray) -> float:
    m = _mad(x)
    return float(1.4826 * m) if np.isfinite(m) else np.nan


def _finite_fraction(x: np.ndarray) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        return np.nan
    return float(np.isfinite(arr).sum() / arr.size)


def _interp_internal_nans_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).copy()
    if arr.size == 0:
        return arr
    good = np.isfinite(arr)
    if good.all() or not good.any():
        return arr
    idx = np.arange(arr.size)
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    return arr


def _sg_window_len(fs_hz: float, win_s: float, poly: int, n: int) -> Optional[int]:
    """Return a valid odd Savitzky-Golay window length for a segment."""
    if not np.isfinite(fs_hz) or fs_hz <= 0 or n <= poly + 2:
        return None
    win = int(round(float(win_s) * float(fs_hz)))
    if win % 2 == 0:
        win += 1
    min_valid = poly + 2
    if min_valid % 2 == 0:
        min_valid += 1
    win = max(win, min_valid)
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win <= poly + 1:
        return None
    if win % 2 == 0:
        win -= 1
    return int(win) if win > poly + 1 else None


def _residual_snr_segments(
    segments: Sequence[np.ndarray],
    *,
    fs_hz: float,
    sg_win_s: float,
    sg_poly: int,
    max_points_per_segment: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate slow-structure-to-fast-residual ratio from valid-trial segments."""
    smooth_all: List[np.ndarray] = []
    resid_all: List[np.ndarray] = []

    for seg in segments:
        x = np.asarray(seg, dtype=float).reshape(-1)
        if max_points_per_segment is not None and x.size > max_points_per_segment:
            x = x[: int(max_points_per_segment)]
        x = _interp_internal_nans_1d(x)
        good = np.isfinite(x)
        if good.sum() < max(sg_poly + 3, 5):
            continue
        if not np.isfinite(x).all():
            continue
        win = _sg_window_len(fs_hz, sg_win_s, sg_poly, x.size)
        if win is None:
            continue
        x = x - np.median(x)
        smooth = savgol_filter(x, window_length=win, polyorder=sg_poly, mode="interp")
        resid = x - smooth
        smooth_all.append(smooth)
        resid_all.append(resid)

    if not smooth_all or not resid_all:
        return {
            "smooth_sigma_robust": np.nan,
            "residual_sigma_robust": np.nan,
            "residual_snr_linear": np.nan,
            "residual_snr_db": np.nan,
        }

    smooth_cat = np.concatenate(smooth_all)
    resid_cat = np.concatenate(resid_all)
    smooth_sigma = _robust_sigma(smooth_cat)
    resid_sigma = _robust_sigma(resid_cat)
    if not np.isfinite(resid_sigma) or resid_sigma <= 0:
        snr = np.nan
        snr_db = np.nan
    else:
        snr = float(smooth_sigma / resid_sigma)
        snr_db = float(20 * np.log10(snr)) if snr > 0 else np.nan
    return {
        "smooth_sigma_robust": smooth_sigma,
        "residual_sigma_robust": resid_sigma,
        "residual_snr_linear": snr,
        "residual_snr_db": snr_db,
    }


# -----------------------------------------------------------------------------
# Trace collection and metrics
# -----------------------------------------------------------------------------


def collect_dmd_voltage_segments(
    vs: VoltageSummary,
    dmd: int,
    *,
    drop_discarded: bool = True,
    dtype=np.float32,
    trace_mode: str = "trial",
    max_trials: Optional[int] = None,
) -> Tuple[List[List[np.ndarray]], Dict[str, Any], pd.DataFrame]:
    """Collect valid-trial voltage trace segments for one DMD.

    Returns
    -------
    roi_segments
        List of length ``n_rois``. Each entry is a list of one-dimensional
        valid-trial segments for that ROI.
    dmd_info
        DMD-level structural metadata.
    trial_table
        Per-trial shape and finite-data QC rows.
    """
    valid_trials = _valid_trials_for_dmd(vs, dmd)
    if max_trials is not None:
        valid_trials = valid_trials[: int(max_trials)]
    n_trials_total = int(vs.n_trials)
    n_valid_trials_total = len(_valid_trials_for_dmd(vs, dmd))
    n_valid_trials_loaded = len(valid_trials)
    valid_trial_fraction = (
        n_valid_trials_total / n_trials_total if n_trials_total > 0 else np.nan
    )
    n_rois = _expected_n_rois(vs, dmd)

    roi_segments: List[List[np.ndarray]] = [[] for _ in range(n_rois)]
    trial_rows: List[Dict[str, Any]] = []
    segment_lengths: List[int] = []

    for trial in valid_trials:
        x = vs.get_roi_traces(
            dmd=dmd,
            trial=trial,
            drop_discarded=False,
            dtype=dtype,
            trace_mode=trace_mode,
        )
        x = _as_time_by_roi(x, expected_n_rois=n_rois, dmd=dmd, trial=trial)
        n_samples_raw = int(x.shape[0])
        discard_fraction = 0.0
        n_discarded = 0

        if drop_discarded:
            discard = _discard_mask_for_trace(vs, dmd=dmd, trial=trial, n_samples=n_samples_raw)
            n_discarded = int(np.sum(discard))
            discard_fraction = float(n_discarded / n_samples_raw) if n_samples_raw else np.nan
            x = x[~discard, :]

        n_samples = int(x.shape[0])
        segment_lengths.append(n_samples)

        trial_rows.append(
            {
                "dmd": int(dmd),
                "trial": int(trial),
                "valid_trial": True,
                "n_samples_raw": n_samples_raw,
                "n_samples": n_samples,
                "n_rois": int(x.shape[1]),
                "n_discarded_samples": n_discarded,
                "discard_fraction": discard_fraction,
                "finite_fraction": _finite_fraction(x),
                "nan_fraction": 1.0 - _finite_fraction(x) if x.size else np.nan,
                "trace_p01": _safe_percentile(x, 1),
                "trace_p50": _safe_percentile(x, 50),
                "trace_p99": _safe_percentile(x, 99),
                "trace_abs_p99": _safe_percentile(np.abs(x), 99),
            }
        )

        for roi_idx in range(n_rois):
            roi_segments[roi_idx].append(x[:, roi_idx].astype(dtype, copy=False))

    dmd_info: Dict[str, Any] = {
        "dmd": int(dmd),
        "n_trials_total": n_trials_total,
        "n_valid_trials": int(n_valid_trials_total),
        "n_valid_trials_loaded": int(n_valid_trials_loaded),
        "valid_trial_fraction": float(valid_trial_fraction) if np.isfinite(valid_trial_fraction) else np.nan,
        "n_rois": int(n_rois),
        "mean_valid_trial_length": float(np.mean(segment_lengths)) if segment_lengths else np.nan,
        "median_valid_trial_length": float(np.median(segment_lengths)) if segment_lengths else np.nan,
        "min_valid_trial_length": int(np.min(segment_lengths)) if segment_lengths else 0,
        "max_valid_trial_length": int(np.max(segment_lengths)) if segment_lengths else 0,
        "total_valid_samples_per_roi": int(np.sum(segment_lengths)) if segment_lengths else 0,
    }
    return roi_segments, dmd_info, pd.DataFrame(trial_rows)


def _compute_roi_metrics(
    *,
    segments: Sequence[np.ndarray],
    fs_hz: float,
    sg_win_s: float,
    sg_poly: int,
    max_points_per_segment_for_snr: Optional[int],
) -> Dict[str, float]:
    """Compute robust trace metrics for one ROI from valid-trial segments."""
    if not segments:
        return {
            "n_segments": 0,
            "concat_n_samples": 0,
            "finite_fraction": 0.0,
            "nan_fraction": 1.0,
            "trace_median": np.nan,
            "trace_mad": np.nan,
            "trace_sigma_robust": np.nan,
            "trace_p01": np.nan,
            "trace_p05": np.nan,
            "trace_p95": np.nan,
            "trace_p99": np.nan,
            "trace_abs_p95": np.nan,
            "trace_abs_p99": np.nan,
            "trace_iqr": np.nan,
            "trace_range_robust": np.nan,
            "trace_drift_median_diff": np.nan,
            "trace_drift_frac_of_range": np.nan,
            "smooth_sigma_robust": np.nan,
            "residual_sigma_robust": np.nan,
            "residual_snr_linear": np.nan,
            "residual_snr_db": np.nan,
        }

    concat = np.concatenate([np.asarray(seg).reshape(-1) for seg in segments])
    finite_frac = _finite_fraction(concat)
    p01 = _safe_percentile(concat, 1)
    p05 = _safe_percentile(concat, 5)
    p25 = _safe_percentile(concat, 25)
    p50 = _safe_percentile(concat, 50)
    p75 = _safe_percentile(concat, 75)
    p95 = _safe_percentile(concat, 95)
    p99 = _safe_percentile(concat, 99)
    trace_range = float(p99 - p01) if np.isfinite(p99) and np.isfinite(p01) else np.nan
    trace_iqr = float(p75 - p25) if np.isfinite(p75) and np.isfinite(p25) else np.nan

    thirds = np.array_split(concat[np.isfinite(concat)], 3)
    if len(thirds) >= 3 and thirds[0].size and thirds[-1].size:
        drift = float(np.nanmedian(thirds[-1]) - np.nanmedian(thirds[0]))
    else:
        drift = np.nan
    drift_frac = float(abs(drift) / trace_range) if np.isfinite(drift) and trace_range > 0 else np.nan

    snr = _residual_snr_segments(
        segments,
        fs_hz=fs_hz,
        sg_win_s=sg_win_s,
        sg_poly=sg_poly,
        max_points_per_segment=max_points_per_segment_for_snr,
    )

    return {
        "n_segments": int(len(segments)),
        "concat_n_samples": int(concat.size),
        "finite_fraction": finite_frac,
        "nan_fraction": float(1.0 - finite_frac) if np.isfinite(finite_frac) else np.nan,
        "trace_median": p50,
        "trace_mad": _mad(concat),
        "trace_sigma_robust": _robust_sigma(concat),
        "trace_p01": p01,
        "trace_p05": p05,
        "trace_p95": p95,
        "trace_p99": p99,
        "trace_abs_p95": _safe_percentile(np.abs(concat), 95),
        "trace_abs_p99": _safe_percentile(np.abs(concat), 99),
        "trace_iqr": trace_iqr,
        "trace_range_robust": trace_range,
        "trace_drift_median_diff": drift,
        "trace_drift_frac_of_range": drift_frac,
        **snr,
    }


# -----------------------------------------------------------------------------
# Quality scoring
# -----------------------------------------------------------------------------


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.clip(x, 0.0, 1.0))


def _bounded_exp_score(x: float, k: float) -> float:
    if not np.isfinite(x) or x < 0:
        return np.nan
    return float(1.0 - np.exp(-k * x))


def _compute_quality_subscores(
    *,
    valid_trial_fraction: float,
    finite_fraction: float,
    trace_abs_p99: float,
    residual_snr_linear: float,
    support_exp: float,
    range_k: float,
    resid_k: float,
) -> Dict[str, float]:
    support_score = _clip01(valid_trial_fraction ** support_exp) if np.isfinite(valid_trial_fraction) else np.nan
    finite_score = _clip01(finite_fraction)
    range_score = _bounded_exp_score(trace_abs_p99, range_k)
    residual_snr_score = _bounded_exp_score(residual_snr_linear, resid_k)
    return {
        "support_score": support_score,
        "finite_score": finite_score,
        "range_score": range_score,
        "residual_snr_score": residual_snr_score,
    }


def _weighted_quality_score(subscores: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    denom = 0.0
    for key, weight in weights.items():
        value = subscores.get(key, np.nan)
        if np.isfinite(value):
            total += float(weight) * float(value)
            denom += float(weight)
    return float(total / denom) if denom > 0 else np.nan


def _evaluate_voltage_roi(
    row: Dict[str, Any],
    thresholds: VoltageQcThresholds,
) -> Tuple[bool, Dict[str, bool], List[str]]:
    checks = {
        "pass_valid_trial_fraction": bool(row["valid_trial_fraction"] >= thresholds.min_valid_trial_fraction),
        "pass_finite_fraction": bool(row["finite_fraction"] >= thresholds.min_finite_fraction),
        "pass_nan_fraction": bool(row["nan_fraction"] <= thresholds.max_nan_fraction),
        "pass_trace_abs_p99": bool(np.isfinite(row["trace_abs_p99"]) and row["trace_abs_p99"] >= thresholds.min_trace_abs_p99),
        "pass_trace_range_robust": bool(np.isfinite(row["trace_range_robust"]) and row["trace_range_robust"] >= thresholds.min_trace_range_robust),
        "pass_abs_median": bool(np.isfinite(row["trace_median"]) and abs(row["trace_median"]) <= thresholds.max_abs_median),
    }
    fail_reasons = [name.replace("pass_", "") for name, ok in checks.items() if not ok]
    return all(checks.values()), checks, fail_reasons


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def _save_roi_quality_plot(qc_df: pd.DataFrame, output_dir: Path) -> None:
    if qc_df.empty:
        return
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        for dmd in sorted(qc_df["dmd"].dropna().unique()):
            sub = qc_df.loc[qc_df["dmd"] == dmd].sort_values("roi_index")
            ax.plot(sub["roi_index"], sub["quality_score"], marker="o", linestyle="-", label=f"DMD{int(dmd)}")
        ax.set_xlabel("ROI index")
        ax.set_ylabel("Voltage QC quality score")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "voltage_roi_quality_scores.png", dpi=150)
        fig.savefig(output_dir / "voltage_roi_quality_scores.pdf")
        plt.close(fig)
    except Exception:
        pass


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    med = np.nanmedian(arr, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=1, keepdims=True)
    scale = 1.4826 * mad
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return (arr - med) / scale


def _save_dmd_trace_examples(
    vs: VoltageSummary,
    *,
    dmd: int,
    output_dir: Path,
    sample_rate_hz: float,
    trace_mode: str,
    dtype,
    max_rois: int = 6,
    max_points: int = 50_000,
) -> None:
    try:
        import matplotlib.pyplot as plt

        valid_trials = _valid_trials_for_dmd(vs, dmd)
        if not valid_trials:
            return
        trial = valid_trials[0]
        n_rois = _expected_n_rois(vs, dmd)
        x = vs.get_roi_traces(dmd=dmd, trial=trial, dtype=dtype, trace_mode=trace_mode)
        x = _as_time_by_roi(x, expected_n_rois=n_rois, dmd=dmd, trial=trial)
        x = x[: min(max_points, x.shape[0]), : min(max_rois, x.shape[1])]
        if x.size == 0:
            return
        t = np.arange(x.shape[0], dtype=float) / float(sample_rate_hz)
        z = _zscore_rows(x.T)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(z.shape[0]):
            ax.plot(t, z[i] + 4.0 * i, linewidth=0.8)
        ax.set_xlabel("Time in first valid trial (s)")
        ax.set_ylabel("ROI z-score + offset")
        ax.set_title(f"DMD{dmd} voltage trace examples, trial {trial}")
        fig.tight_layout()
        fig.savefig(output_dir / f"dmd{dmd}_voltage_trace_examples.png", dpi=150)
        fig.savefig(output_dir / f"dmd{dmd}_voltage_trace_examples.pdf")
        plt.close(fig)
    except Exception:
        pass


def _save_dmd_trial_heatmap(
    vs: VoltageSummary,
    *,
    dmd: int,
    output_dir: Path,
    sample_rate_hz: float,
    trace_mode: str,
    dtype,
    max_trials: int = 30,
    max_points: int = 20_000,
) -> None:
    try:
        import matplotlib.pyplot as plt

        valid_trials = _valid_trials_for_dmd(vs, dmd)[:max_trials]
        if not valid_trials:
            return
        n_rois = _expected_n_rois(vs, dmd)
        rows: List[np.ndarray] = []
        for trial in valid_trials:
            x = vs.get_roi_traces(dmd=dmd, trial=trial, dtype=dtype, trace_mode=trace_mode)
            x = _as_time_by_roi(x, expected_n_rois=n_rois, dmd=dmd, trial=trial)
            y = np.nanmean(x[:, : min(3, n_rois)], axis=1)
            rows.append(y[: min(max_points, y.size)])
        if not rows:
            return
        min_len = min(r.size for r in rows)
        if min_len == 0:
            return
        mat = np.vstack([r[:min_len] for r in rows])
        mat = _zscore_rows(mat)
        extent = [0, min_len / float(sample_rate_hz), len(rows), 0]
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", extent=extent, vmin=-5, vmax=5)
        ax.set_xlabel("Time in trial (s)")
        ax.set_ylabel("Valid trial")
        ax.set_title(f"DMD{dmd} mean voltage trace heatmap")
        fig.colorbar(im, ax=ax, label="Robust z-score")
        fig.tight_layout()
        fig.savefig(output_dir / f"dmd{dmd}_voltage_trial_heatmap.png", dpi=150)
        fig.savefig(output_dir / f"dmd{dmd}_voltage_trial_heatmap.pdf")
        plt.close(fig)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Metadata builders
# -----------------------------------------------------------------------------


def _build_metadata(
    *,
    asset: SessionAssets,
    vs: VoltageSummary,
    output_dir: Path,
    sample_rate_hz: float,
    thresholds: VoltageQcThresholds,
    sg_win_s: float,
    sg_poly: int,
    score_weights: Dict[str, float],
    score_params: Dict[str, float],
    dmd_summary: Dict[str, Any],
    drop_discarded: bool,
    trace_mode: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "session_metadata": {
            "session_id": asset.session_id,
            "subject_id": int(asset.subject_id),
            "session_dir": str(asset.session_dir),
            "voltage_summary_mat": str(asset.require_asset("voltage", "summary_mat")),
            "voltage_trace_h5": str(asset.get_asset("voltage", "trace_h5")) if asset.get_asset("voltage", "trace_h5") is not None else None,
            "sample_rate_hz": float(sample_rate_hz),
            "layout": vs.layout,
            "available_trace_modes": vs.available_trace_modes(),
            "trace_mode": trace_mode,
            "drop_discarded": bool(drop_discarded),
            "voltage_metadata": vs.metadata,
            "h5_attrs": vs.h5_attrs,
        },
        "parameters": {
            "thresholds": asdict(thresholds),
            "residual_snr": {
                "sg_window_seconds": sg_win_s,
                "sg_polyorder": sg_poly,
                "interpretation": "slow Savitzky-Golay component divided by fast residual; QC proxy only, not final oscillation analysis",
            },
            "quality_score": {
                "weights": score_weights,
                "score_params": score_params,
                "recommended_for_analysis": {
                    "rule": "passes conservative absolute checks and quality percentile within DMD >= threshold",
                    "min_quality_percentile_within_dmd": thresholds.min_quality_percentile_within_dmd,
                    "min_quality_score": thresholds.min_quality_score,
                },
            },
        },
        "dmd_summary": dmd_summary,
        "outputs": {
            "output_dir": str(output_dir),
            "roi_qc_csv": str(output_dir / "voltage_roi_qc.csv"),
            "trial_qc_csv": str(output_dir / "voltage_trial_qc.csv"),
        },
        "metric_descriptions": {
            "finite_fraction": "Fraction of finite samples in concatenated valid-trial raw fluorescence trace.",
            "trace_abs_p99": "99th percentile of absolute raw fluorescence values; first-pass dynamic range proxy.",
            "trace_range_robust": "q99 - q1 of concatenated valid-trial raw fluorescence trace.",
            "residual_snr_linear": "Robust scale of slow smoothed component divided by robust scale of residual fast component.",
            "quality_score": "Weighted composite of support, finite completeness, dynamic range, and residual SNR proxies.",
            "recommended_for_analysis": "Conservative keep flag for downstream event extraction, not a final biological inclusion criterion.",
        },
    }


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def run_voltage_qc(
    asset: SessionAssets,
    *,
    output_dir: Optional[Path] = None,
    sample_rate_hz: Optional[float] = DEFAULT_VOLTAGE_SAMPLE_RATE_HZ,
    default_sample_rate_hz: float = DEFAULT_VOLTAGE_SAMPLE_RATE_HZ,
    thresholds: Optional[VoltageQcThresholds] = None,
    trace_mode: str = "trial",
    drop_discarded: bool = True,
    dtype=np.float32,
    sg_win_s: float = 1.0,
    sg_poly: int = 3,
    score_weights: Optional[Dict[str, float]] = None,
    score_params: Optional[Dict[str, float]] = None,
    max_trials_for_metrics: Optional[int] = None,
    max_points_per_segment_for_snr: Optional[int] = 60_000,
    overwrite: bool = False,
    make_plots: bool = True,
) -> VoltageQcResult:
    """Run first-pass QC for one voltage session.

    Parameters
    ----------
    asset
        Generic session assets from ``VIPSessionRegistry.resolve_assets``.
    output_dir
        Optional output directory. Defaults to ``asset.qc_subdir("voltage")``.
    sample_rate_hz
        Effective voltage sample rate. The default is 10.8 kHz for current SLAP2
        integration-mode voltage imaging. Pass ``None`` to resolve from metadata
        and fall back to ``default_sample_rate_hz``.
    thresholds
        Optional threshold dataclass controlling first-pass ROI keep masks.
    trace_mode
        Trace mode passed to ``VoltageSummary.get_roi_traces``. Current voltage
        outputs are trial-based, so the default is ``"trial"``.
    drop_discarded
        Remove samples marked by ``discardFrames`` before computing metrics.
    max_trials_for_metrics
        Optional development/debug limit. Leave ``None`` for full-session QC.
    max_points_per_segment_for_snr
        Optional cap for the residual-SNR calculation only. Other metrics use all
        loaded samples. Set to ``None`` to use complete segments.
    overwrite
        Recompute QC even if JSON/CSV/mask outputs already exist.
    make_plots
        Save lightweight diagnostic plots under the QC directory.

    Returns
    -------
    VoltageQcResult
        In-memory QC tables plus metadata/summary dictionaries and output paths.
    """
    thresholds = thresholds or VoltageQcThresholds()
    if score_weights is None:
        score_weights = {
            "support_score": 0.15,
            "finite_score": 0.20,
            "range_score": 0.25,
            "residual_snr_score": 0.40,
        }
    if score_params is None:
        score_params = {
            "support_exp": 0.5,
            "range_k": 1.0,
            "resid_k": 0.8,
        }

    if not asset.has_modality("voltage", {"summary_mat": None}):
        raise FileNotFoundError(
            f"Session {asset.session_id!r} does not have a resolved voltage summary asset."
        )

    if output_dir is None:
        output_dir = asset.qc_subdir("voltage", create=True)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    qc_json_path = output_dir / "voltage_qc_metadata.json"
    summary_json_path = output_dir / "voltage_session_qc_summary.json"
    roi_csv_path = output_dir / "voltage_roi_qc.csv"
    trial_csv_path = output_dir / "voltage_trial_qc.csv"

    # Return cached outputs when all core artifacts exist and overwrite is False.
    if not overwrite and qc_json_path.exists() and summary_json_path.exists() and roi_csv_path.exists() and trial_csv_path.exists():
        qc_table = pd.read_csv(roi_csv_path)
        trial_table = pd.read_csv(trial_csv_path)
        with open(qc_json_path, "r") as f:
            metadata = json.load(f)
        with open(summary_json_path, "r") as f:
            summary = json.load(f)
        keep_masks = {
            f"DMD{int(dmd)}": str(output_dir / f"dmd{int(dmd)}_recommended_voltage_rois.npy")
            for dmd in sorted(qc_table["dmd"].dropna().unique())
        }
        return VoltageQcResult(qc_table, trial_table, metadata, summary, str(output_dir), keep_masks)

    rows: List[Dict[str, Any]] = []
    trial_tables: List[pd.DataFrame] = []
    dmd_summary: Dict[str, Any] = {}
    keep_masks: Dict[str, Optional[str]] = {}

    with load_voltage_summary_from_asset(asset, keep_open=True) as vs:
        fs_hz = resolve_voltage_sample_rate_hz(
            asset=asset,
            vs=vs,
            sample_rate_hz=sample_rate_hz,
            default_hz=default_sample_rate_hz,
        )

        for dmd in range(1, int(vs.n_dmds) + 1):
            label = f"DMD{dmd}"
            roi_segments, dmd_info, trial_df = collect_dmd_voltage_segments(
                vs,
                dmd=dmd,
                drop_discarded=drop_discarded,
                dtype=dtype,
                trace_mode=trace_mode,
                max_trials=max_trials_for_metrics,
            )
            dmd_summary[label] = dmd_info
            if not trial_df.empty:
                trial_df.insert(0, "session_id", asset.session_id)
                trial_df.insert(1, "subject_id", int(asset.subject_id))
                trial_tables.append(trial_df)

            for roi_index, segments in enumerate(roi_segments):
                metrics = _compute_roi_metrics(
                    segments=segments,
                    fs_hz=fs_hz,
                    sg_win_s=sg_win_s,
                    sg_poly=sg_poly,
                    max_points_per_segment_for_snr=max_points_per_segment_for_snr,
                )
                subscores = _compute_quality_subscores(
                    valid_trial_fraction=dmd_info["valid_trial_fraction"],
                    finite_fraction=metrics["finite_fraction"],
                    trace_abs_p99=metrics["trace_abs_p99"],
                    residual_snr_linear=metrics["residual_snr_linear"],
                    support_exp=score_params["support_exp"],
                    range_k=score_params["range_k"],
                    resid_k=score_params["resid_k"],
                )
                row: Dict[str, Any] = {
                    "session_id": asset.session_id,
                    "subject_id": int(asset.subject_id),
                    "dmd": int(dmd),
                    "roi_index": int(roi_index),
                    "roi_id": f"DMD{dmd}_roi{roi_index:04d}",
                    "sample_rate_hz": float(fs_hz),
                    "trace_mode": trace_mode,
                    "drop_discarded": bool(drop_discarded),
                    "n_trials_total": int(dmd_info["n_trials_total"]),
                    "n_valid_trials": int(dmd_info["n_valid_trials"]),
                    "n_valid_trials_loaded": int(dmd_info["n_valid_trials_loaded"]),
                    "valid_trial_fraction": float(dmd_info["valid_trial_fraction"]),
                    **metrics,
                    **subscores,
                }
                row["quality_score"] = _weighted_quality_score(subscores, score_weights)
                preliminary_keep, checks, fail_reasons = _evaluate_voltage_roi(row, thresholds)
                row.update(checks)
                row["preliminary_keep"] = bool(preliminary_keep)
                row["n_failed_checks"] = int(len(fail_reasons))
                row["fail_reasons"] = ";".join(fail_reasons)
                rows.append(row)

            if make_plots:
                _save_dmd_trace_examples(
                    vs,
                    dmd=dmd,
                    output_dir=output_dir,
                    sample_rate_hz=fs_hz,
                    trace_mode=trace_mode,
                    dtype=dtype,
                )
                _save_dmd_trial_heatmap(
                    vs,
                    dmd=dmd,
                    output_dir=output_dir,
                    sample_rate_hz=fs_hz,
                    trace_mode=trace_mode,
                    dtype=dtype,
                )

        qc_table = pd.DataFrame(rows)
        if qc_table.empty:
            qc_table = pd.DataFrame(
                columns=[
                    "session_id",
                    "subject_id",
                    "dmd",
                    "roi_index",
                    "roi_id",
                    "quality_score",
                    "recommended_for_analysis",
                ]
            )

        if not qc_table.empty and "quality_score" in qc_table:
            qc_table["quality_rank_within_dmd"] = (
                qc_table.groupby("dmd")["quality_score"]
                .rank(method="min", ascending=False)
                .astype(float)
            )
            qc_table["quality_percentile_within_dmd"] = (
                qc_table.groupby("dmd")["quality_score"]
                .rank(method="average", pct=True)
                .astype(float)
            )
            qc_table["recommended_for_analysis"] = (
                qc_table["preliminary_keep"].astype(bool)
                & (qc_table["quality_score"] >= thresholds.min_quality_score)
                & (qc_table["quality_percentile_within_dmd"] >= thresholds.min_quality_percentile_within_dmd)
            )
        else:
            qc_table["recommended_for_analysis"] = False

        trial_table = pd.concat(trial_tables, ignore_index=True) if trial_tables else pd.DataFrame()

        # Save keep masks per DMD in a naming style parallel to slap2/glutamate QC.
        for dmd in range(1, int(vs.n_dmds) + 1):
            sub = qc_table.loc[qc_table["dmd"] == dmd].sort_values("roi_index")
            keep = sub["recommended_for_analysis"].to_numpy(dtype=bool) if not sub.empty else np.zeros((0,), dtype=bool)
            keep_path = output_dir / f"dmd{dmd}_recommended_voltage_rois.npy"
            np.save(keep_path, keep)
            keep_masks[f"DMD{dmd}"] = str(keep_path)
            if f"DMD{dmd}" in dmd_summary:
                dmd_summary[f"DMD{dmd}"]["n_rois_recommended"] = int(np.sum(keep))
                dmd_summary[f"DMD{dmd}"]["recommended_voltage_rois_path"] = str(keep_path)

        metadata = _build_metadata(
            asset=asset,
            vs=vs,
            output_dir=output_dir,
            sample_rate_hz=fs_hz,
            thresholds=thresholds,
            sg_win_s=sg_win_s,
            sg_poly=sg_poly,
            score_weights=score_weights,
            score_params=score_params,
            dmd_summary=dmd_summary,
            drop_discarded=drop_discarded,
            trace_mode=trace_mode,
        )

    summary = {
        "schema_version": "0.1.0",
        "session_id": asset.session_id,
        "subject_id": int(asset.subject_id),
        "voltage_summary_mat": str(asset.require_asset("voltage", "summary_mat")),
        "voltage_trace_h5": str(asset.get_asset("voltage", "trace_h5")) if asset.get_asset("voltage", "trace_h5") is not None else None,
        "sample_rate_hz": float(metadata["session_metadata"]["sample_rate_hz"]),
        "n_voltage_rois_total": int(len(qc_table)),
        "n_voltage_rois_recommended": int(qc_table["recommended_for_analysis"].sum()) if "recommended_for_analysis" in qc_table else 0,
        "n_dmds": int(len(dmd_summary)),
        "dmd_summary": dmd_summary,
        "quality_score_summary": (
            qc_table.groupby("dmd")["quality_score"].agg(["count", "mean", "median", "std", "min", "max"]).to_dict()
            if len(qc_table) > 0 and "quality_score" in qc_table else {}
        ),
    }

    qc_table.to_csv(roi_csv_path, index=False)
    try:
        qc_table.to_parquet(output_dir / "voltage_roi_qc.parquet", index=False)
    except Exception:
        pass
    trial_table.to_csv(trial_csv_path, index=False)

    with open(qc_json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    if make_plots:
        _save_roi_quality_plot(qc_table, output_dir)

    return VoltageQcResult(
        qc_table=qc_table,
        trial_table=trial_table,
        metadata=metadata,
        summary=summary,
        output_dir=str(output_dir),
        keep_masks=keep_masks,
    )
