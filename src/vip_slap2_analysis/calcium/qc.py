from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.glutamate.summary import GlutamateSummary


DEFAULT_CAMP_REGEX = r"(?i)^[A-Za-z]*CaMP\d+[A-Za-z0-9._-]*$"


@dataclass
class CalciumQcThresholds:
    min_valid_trial_fraction: float = 0.5
    min_finite_fraction: float = 0.75
    min_dynamic_range: float = 0.10
    min_abs_peak_p99: float = 0.12
    min_snr_like: float = 2.0
    max_abs_drift: float = 0.05
    max_drift_frac: float = 0.30


@dataclass
class CalciumQcResult:
    should_process_calcium: bool
    indicator2: Optional[str]
    indicator_regex: str
    qc_json: Optional[str]
    qc_table_csv: Optional[str]
    keep_masks: Dict[str, Optional[str]]
    per_dmd: Dict[str, Any]


def _resolve_indicator2(asset: SessionAssets, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    metadata = metadata or {}
    if metadata.get("indicator2") is not None:
        return str(metadata.get("indicator2"))
    if getattr(asset, "metadata", None) and asset.metadata.get("indicator2") is not None:
        return str(asset.metadata.get("indicator2"))
    return None


def should_process_calcium_indicator(indicator2: Optional[str], pattern: str = DEFAULT_CAMP_REGEX) -> bool:
    if indicator2 is None:
        return False
    return re.match(pattern, str(indicator2)) is not None


def _resolve_fs_hz(asset: SessionAssets, metadata: Optional[Dict[str, Any]], exp: GlutamateSummary) -> float:
    metadata = metadata or {}
    for key in ["im_rate_hz", "im_rate_Hz", "fs_hz", "fs_Hz"]:
        if key in metadata:
            return float(metadata[key])
    if getattr(asset, "metadata", None):
        for key in ["im_rate_hz", "im_rate_Hz", "fs_hz", "fs_Hz"]:
            if key in asset.metadata:
                return float(asset.metadata[key])
    if hasattr(exp, "metadata") and isinstance(exp.metadata, dict) and "analyzeHz" in exp.metadata:
        return float(exp.metadata["analyzeHz"])
    raise ValueError("Could not resolve fs_hz / imaging rate for calcium QC.")


def _concat_trials_with_nans(dff: np.ndarray) -> np.ndarray:
    """dff: (n_trials, n_rois, T) -> (n_rois, n_trials*T)"""
    if dff.ndim != 3:
        raise ValueError(f"Expected dff shape (n_trials, n_rois, T), got {dff.shape}")
    n_trials, n_rois, T = dff.shape
    return np.transpose(dff, (1, 0, 2)).reshape(n_rois, n_trials * T)


def _safe_nanpercentile(x: np.ndarray, q: float) -> float:
    if not np.isfinite(x).any():
        return np.nan
    return float(np.nanpercentile(x, q))


def _roi_metrics(y: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(y)
    yf = y[finite]
    n_total = int(y.size)
    n_finite = int(finite.sum())
    finite_fraction = float(n_finite / n_total) if n_total else 0.0

    if n_finite == 0:
        return {
            "finite_fraction": 0.0,
            "n_finite": 0,
            "mad_sigma": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "dynamic_range_p95_p05": np.nan,
            "abs_peak_p99": np.nan,
            "snr_like": np.nan,
            "drift_median_diff": np.nan,
            "drift_frac": np.nan,
        }

    med = float(np.nanmedian(yf))
    mad_sigma = float(1.4826 * np.nanmedian(np.abs(yf - med)))
    p01 = _safe_nanpercentile(yf, 1)
    p05 = _safe_nanpercentile(yf, 5)
    p50 = _safe_nanpercentile(yf, 50)
    p95 = _safe_nanpercentile(yf, 95)
    p99 = _safe_nanpercentile(yf, 99)
    dynamic_range = float(p95 - p05)
    abs_peak_p99 = float(max(abs(p01), abs(p99)))
    snr_like = float(abs_peak_p99 / mad_sigma) if np.isfinite(mad_sigma) and mad_sigma > 0 else np.nan

    # robust drift: compare medians of early and late thirds of finite data
    thirds = np.array_split(yf, 3)
    if len(thirds) >= 3 and len(thirds[0]) > 0 and len(thirds[-1]) > 0:
        drift = float(np.nanmedian(thirds[-1]) - np.nanmedian(thirds[0]))
    else:
        drift = np.nan
    drift_frac = float(abs(drift) / dynamic_range) if np.isfinite(drift) and dynamic_range > 0 else np.nan

    return {
        "finite_fraction": finite_fraction,
        "n_finite": n_finite,
        "mad_sigma": mad_sigma,
        "p01": p01,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "dynamic_range_p95_p05": dynamic_range,
        "abs_peak_p99": abs_peak_p99,
        "snr_like": snr_like,
        "drift_median_diff": drift,
        "drift_frac": drift_frac,
    }


def _evaluate_roi(metrics: Dict[str, float], thr: CalciumQcThresholds) -> Tuple[bool, Dict[str, bool], List[str]]:
    checks = {
        "pass_finite_fraction": bool(metrics["finite_fraction"] >= thr.min_finite_fraction),
        "pass_dynamic_range": bool(np.isfinite(metrics["dynamic_range_p95_p05"]) and metrics["dynamic_range_p95_p05"] >= thr.min_dynamic_range),
        "pass_abs_peak_p99": bool(np.isfinite(metrics["abs_peak_p99"]) and metrics["abs_peak_p99"] >= thr.min_abs_peak_p99),
        "pass_snr_like": bool(np.isfinite(metrics["snr_like"]) and metrics["snr_like"] >= thr.min_snr_like),
        "pass_abs_drift": bool(np.isfinite(metrics["drift_median_diff"]) and abs(metrics["drift_median_diff"]) <= thr.max_abs_drift),
        "pass_drift_frac": bool(np.isfinite(metrics["drift_frac"]) and metrics["drift_frac"] <= thr.max_drift_frac),
    }
    fail_reasons = [k.replace("pass_", "") for k, v in checks.items() if not v]
    keep = all(checks.values())
    return keep, checks, fail_reasons


def run_calcium_qc(
    asset: SessionAssets,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    indicator_regex: str = DEFAULT_CAMP_REGEX,
    trace_type: str = "Fsvd",
    motion_correct: bool = False,
    max_session_minutes = None,
    thresholds: Optional[CalciumQcThresholds] = None,
    overwrite: bool = False,
    process_kwargs: Optional[Dict[str, Any]] = None,
) -> CalciumQcResult:
    thresholds = thresholds or CalciumQcThresholds()
    process_kwargs = process_kwargs or {}
    metadata = metadata or {}

    if asset.summary_mat is None:
        raise FileNotFoundError("asset.summary_mat is missing")
    if asset.qc_dir is None:
        raise ValueError("asset.qc_dir must be set for calcium QC")

    indicator2 = _resolve_indicator2(asset, metadata)
    should_process = should_process_calcium_indicator(indicator2, indicator_regex)

    qc_dir = Path(asset.qc_dir) / "calcium"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_json = qc_dir / "calcium_qc.json"
    qc_csv = qc_dir / "calcium_qc_table.csv"
    keep_paths = {
        "DMD1": qc_dir / "valid_soma_rois_dmd1.npy",
        "DMD2": qc_dir / "valid_soma_rois_dmd2.npy",
    }

    if all(p.exists() for p in [qc_json, qc_csv, keep_paths["DMD1"], keep_paths["DMD2"]]) and not overwrite:
        with open(qc_json, "r") as f:
            obj = json.load(f)
        return CalciumQcResult(
            should_process_calcium=obj.get("should_process_calcium", False),
            indicator2=obj.get("indicator2"),
            indicator_regex=obj.get("indicator_regex", indicator_regex),
            qc_json=str(qc_json),
            qc_table_csv=str(qc_csv),
            keep_masks={k: str(v) for k, v in keep_paths.items()},
            per_dmd=obj.get("per_dmd", {}),
        )

    if not should_process:
        obj = {
            "schema_version": "0.2.0",
            "session_id": asset.session_id,
            "subject_id": int(asset.subject_id),
            "indicator2": indicator2,
            "indicator_regex": indicator_regex,
            "should_process_calcium": False,
            "reason": "indicator2 did not match configured CaMP regex",
            "thresholds": asdict(thresholds),
            "per_dmd": {},
        }
        with open(qc_json, "w") as f:
            json.dump(obj, f, indent=2)
        return CalciumQcResult(False, indicator2, indicator_regex, str(qc_json), None, {"DMD1": None, "DMD2": None}, {})

    exp = GlutamateSummary(asset.summary_mat)
    fs_hz = _resolve_fs_hz(asset, metadata, exp)

    rows: List[Dict[str, Any]] = []
    per_dmd: Dict[str, Any] = {}

    for dmd in (1, 2):
        label = f"DMD{dmd}"
        keep_trials = np.asarray(exp.keep_trials[dmd - 1], dtype=bool)
        valid_trial_fraction = float(np.mean(keep_trials)) if keep_trials.size else 0.0

        proc = exp.get_processed_soma_ca_all_trials(
            dmd=dmd,
            trace_type=trace_type,
            fs_hz=fs_hz,
            include_invalid=True,
            motion_correct=motion_correct,
            max_session_minutes = max_session_minutes,
            **process_kwargs,
        )
        dff = np.asarray(proc["dff"], dtype=float)
        if dff.ndim != 3:
            raise ValueError(f"Expected processed calcium dff shape (n_trials, n_rois, T), got {dff.shape}")
        n_trials, n_rois, trial_len = dff.shape
        concat = _concat_trials_with_nans(dff)

        keep_mask = np.zeros((n_rois,), dtype=bool)
        roi_fail_reason_counts: Dict[str, int] = {}
        keep_due_to_session = bool(valid_trial_fraction >= thresholds.min_valid_trial_fraction)

        for roi_index in range(n_rois):
            roi_id = f"{label}_roi{roi_index:04d}"
            metrics = _roi_metrics(concat[roi_index])
            keep, checks, fail_reasons = _evaluate_roi(metrics, thresholds)
            if not keep_due_to_session:
                keep = False
                fail_reasons = ["valid_trial_fraction"] + fail_reasons
                checks["pass_valid_trial_fraction"] = False
            else:
                checks["pass_valid_trial_fraction"] = True

            for fr in fail_reasons:
                roi_fail_reason_counts[fr] = roi_fail_reason_counts.get(fr, 0) + 1

            keep_mask[roi_index] = keep
            rows.append({
                "session_id": asset.session_id,
                "subject_id": int(asset.subject_id),
                "dmd": label,
                "roi_index": int(roi_index),
                "roi_id": roi_id,
                "indicator2": indicator2,
                "trace_type": trace_type,
                "motion_correct": bool(motion_correct),
                "valid_trial_fraction": valid_trial_fraction,
                "n_trials": int(n_trials),
                "trial_len_samples": int(trial_len),
                "keep": bool(keep),
                "n_failed_checks": int(len(fail_reasons)),
                "fail_reasons": ";".join(fail_reasons),
                **metrics,
                **checks,
            })

        np.save(keep_paths[label], keep_mask)
        per_dmd[label] = {
            "processed": True,
            "n_trials": int(n_trials),
            "n_rois_total": int(n_rois),
            "n_rois_kept": int(np.sum(keep_mask)),
            "valid_trial_fraction": valid_trial_fraction,
            "session_valid_trial_fraction_pass": keep_due_to_session,
            "keep_mask_path": str(keep_paths[label]),
            "thresholds": asdict(thresholds),
            "roi_fail_reason_counts": roi_fail_reason_counts,
        }

    table = pd.DataFrame(rows)
    table.to_csv(qc_csv, index=False)

    obj = {
        "schema_version": "0.2.0",
        "session_id": asset.session_id,
        "subject_id": int(asset.subject_id),
        "indicator2": indicator2,
        "indicator_regex": indicator_regex,
        "should_process_calcium": True,
        "trace_type": trace_type,
        "motion_correct": bool(motion_correct),
        "fs_hz": float(fs_hz),
        "thresholds": asdict(thresholds),
        "per_dmd": per_dmd,
        "qc_table_csv": str(qc_csv),
    }
    with open(qc_json, "w") as f:
        json.dump(obj, f, indent=2)

    return CalciumQcResult(
        should_process_calcium=True,
        indicator2=indicator2,
        indicator_regex=indicator_regex,
        qc_json=str(qc_json),
        qc_table_csv=str(qc_csv),
        keep_masks={k: str(v) for k, v in keep_paths.items()},
        per_dmd=per_dmd,
    )
