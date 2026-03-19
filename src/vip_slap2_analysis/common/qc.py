from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from vip_slap2_analysis.glutamate.summary import GlutamateSummary
from vip_slap2_analysis.plotting.qc_plots import make_all_synapse_qc_plots


# =============================================================================
# Robust stats helpers
# =============================================================================

def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))


def robust_sigma(x: np.ndarray) -> float:
    """
    Robust analog of standard deviation using 1.4826 * MAD.
    """
    m = mad(x)
    if not np.isfinite(m):
        return np.nan
    return 1.4826 * m


def safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, q))


def robust_range(x: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0) -> float:
    lo = safe_percentile(x, q_lo)
    hi = safe_percentile(x, q_hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    return float(hi - lo)


def finite_fraction(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return np.nan
    return float(np.isfinite(x).sum() / x.size)


def _interp_internal_nans_1d(x: np.ndarray) -> np.ndarray:
    """
    Interpolate NaNs within a single valid-trial segment only.
    """
    x = np.asarray(x, dtype=float).copy()
    if x.size == 0:
        return x

    good = np.isfinite(x)
    if good.all():
        return x
    if not good.any():
        return x

    idx = np.arange(x.size)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


# =============================================================================
# Residual SNR
# =============================================================================

def _sg_window_len(fs_hz: float, win_s: float, poly: int) -> int:
    """
    Return a valid odd Savitzky-Golay window length.
    """
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and > 0.")

    win = int(round(win_s * fs_hz))
    if win % 2 == 0:
        win += 1

    min_valid = poly + 2
    if min_valid % 2 == 0:
        min_valid += 1

    return max(win, min_valid)


def residual_snr_segments(
    segments: Sequence[np.ndarray],
    fs_hz: float,
    win_s: float = 0.5,
    poly: int = 3,
    interpolate_nans: bool = True,
) -> Dict[str, float]:
    """
    Compute residual SNR from a list of valid-trial segments.
    """
    smooth_all = []
    resid_all = []

    win = _sg_window_len(fs_hz=fs_hz, win_s=win_s, poly=poly)

    for seg in segments:
        x = np.asarray(seg, dtype=float)
        if x.size < max(win, poly + 2):
            continue

        if interpolate_nans:
            x = _interp_internal_nans_1d(x)

        good = np.isfinite(x)
        if good.sum() < max(win, poly + 2):
            continue
        if not np.isfinite(x).all():
            continue

        x = x - np.median(x)
        smooth = savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
        resid = x - smooth

        smooth_all.append(smooth)
        resid_all.append(resid)

    if len(smooth_all) == 0 or len(resid_all) == 0:
        return {
            "smooth_sigma_robust": np.nan,
            "residual_sigma_robust": np.nan,
            "residual_snr_linear": np.nan,
            "residual_snr_db": np.nan,
        }

    smooth_all = np.concatenate(smooth_all)
    resid_all = np.concatenate(resid_all)

    smooth_sigma = robust_sigma(smooth_all)
    resid_sigma = robust_sigma(resid_all)

    if not np.isfinite(resid_sigma) or resid_sigma <= 0:
        snr_lin = np.nan
        snr_db = np.nan
    else:
        snr_lin = float(smooth_sigma / resid_sigma)
        snr_db = float(20 * np.log10(snr_lin)) if snr_lin > 0 else np.nan

    return {
        "smooth_sigma_robust": float(smooth_sigma) if np.isfinite(smooth_sigma) else np.nan,
        "residual_sigma_robust": float(resid_sigma) if np.isfinite(resid_sigma) else np.nan,
        "residual_snr_linear": snr_lin,
        "residual_snr_db": snr_db,
    }


# =============================================================================
# Trace extraction helpers
# =============================================================================

def _normalize_trial_traces_to_rois_by_time(x: np.ndarray) -> np.ndarray:
    """
    Normalize a trace array to shape (n_rois, n_time).
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 3:
        x = x[:, :, 0]
    if x.ndim != 2:
        raise ValueError(f"Unexpected trace shape {x.shape}; expected 2D or 3D.")

    return x.T


def _get_valid_trial_trace_matrix(
    exp: GlutamateSummary,
    dmd: int,
    trial: int,
    signal: str,
    mode: str,
) -> np.ndarray:
    x = exp.get_traces(
        dmd=dmd,
        trial=trial,
        signal=signal,
        mode=mode,
        squeeze_channels=True,
    )
    return _normalize_trial_traces_to_rois_by_time(x)


def collect_dmd_synapse_segments(
    exp: GlutamateSummary,
    dmd: int,
    signal: str = "dF",
    mode: str = "ls",
) -> Tuple[List[List[np.ndarray]], Dict[str, float]]:
    """
    Collect valid-trial segments per synapse for one DMD.

    Returns
    -------
    synapse_segments
        List of length n_synapses. Each entry is a list of 1D arrays,
        one per valid trial.
    dmd_info
        DMD-level metadata.
    """
    dmd0 = dmd - 1
    n_trials_total = int(exp.n_trials)
    valid_trials = list(exp.valid_trials[dmd0])
    n_valid_trials = len(valid_trials)
    valid_trial_fraction = n_valid_trials / n_trials_total if n_trials_total > 0 else np.nan
    n_synapses = int(exp.n_synapses[dmd0])

    synapse_segments: List[List[np.ndarray]] = [[] for _ in range(n_synapses)]
    segment_lengths = []

    for tr in valid_trials:
        x = _get_valid_trial_trace_matrix(exp, dmd=dmd, trial=tr, signal=signal, mode=mode)
        if x.shape[0] != n_synapses:
            raise ValueError(
                f"DMD{dmd} trial {tr}: expected {n_synapses} synapses, got {x.shape[0]}"
            )

        segment_lengths.append(x.shape[1])

        for syn_idx in range(n_synapses):
            synapse_segments[syn_idx].append(x[syn_idx].copy())

    dmd_info = {
        "dmd": dmd,
        "n_trials_total": n_trials_total,
        "n_valid_trials": n_valid_trials,
        "valid_trial_fraction": float(valid_trial_fraction) if np.isfinite(valid_trial_fraction) else np.nan,
        "n_synapses": n_synapses,
        "mean_valid_trial_length": float(np.mean(segment_lengths)) if len(segment_lengths) else np.nan,
        "total_valid_samples_per_synapse": int(np.sum(segment_lengths)) if len(segment_lengths) else 0,
    }
    return synapse_segments, dmd_info


# =============================================================================
# Quality scoring
# =============================================================================

def _bounded_exp_score(x: float, k: float = 1.0) -> float:
    if not np.isfinite(x) or x < 0:
        return np.nan
    return float(1.0 - np.exp(-k * x))


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return float(np.clip(x, 0.0, 1.0))


def compute_quality_subscores(
    valid_trial_fraction: float,
    finite_fraction_value: float,
    trace_abs_p99: float,
    residual_snr_linear: float,
    support_exp: float = 0.5,
    range_k: float = 0.75,
    resid_k: float = 0.8,
) -> Dict[str, float]:
    """
    Compute continuous sub-scores used in the composite quality score.
    """
    support_score = np.nan
    finite_score = np.nan
    range_score = np.nan
    residual_snr_score = np.nan

    if np.isfinite(valid_trial_fraction):
        support_score = _clip01(valid_trial_fraction ** support_exp)

    if np.isfinite(finite_fraction_value):
        finite_score = _clip01(finite_fraction_value)

    if np.isfinite(trace_abs_p99):
        range_score = _bounded_exp_score(max(trace_abs_p99, 0.0), k=range_k)

    if np.isfinite(residual_snr_linear):
        residual_snr_score = _bounded_exp_score(max(residual_snr_linear, 0.0), k=resid_k)

    return {
        "support_score": support_score,
        "finite_score": finite_score,
        "range_score": range_score,
        "residual_snr_score": residual_snr_score,
    }


def combine_quality_score(
    support_score: float,
    finite_score: float,
    range_score: float,
    residual_snr_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Weighted average of available sub-scores.
    """
    if weights is None:
        weights = {
            "support_score": 0.20,
            "finite_score": 0.20,
            "range_score": 0.25,
            "residual_snr_score": 0.35,
        }

    vals = {
        "support_score": support_score,
        "finite_score": finite_score,
        "range_score": range_score,
        "residual_snr_score": residual_snr_score,
    }

    num = 0.0
    den = 0.0
    for k, w in weights.items():
        v = vals.get(k, np.nan)
        if np.isfinite(v):
            num += w * v
            den += w

    if den == 0:
        return np.nan
    return float(num / den)


# =============================================================================
# Per-synapse metrics
# =============================================================================

def compute_synapse_quality_metrics(
    synapse_segments: Sequence[np.ndarray],
    session_id: str,
    subject_id: Optional[Union[int, str]],
    dmd: int,
    synapse_idx: int,
    n_trials_total: int,
    n_valid_trials: int,
    valid_trial_fraction: float,
    fs_hz: float,
    sg_win_s: float = 0.5,
    sg_poly: int = 3,
    score_weights: Optional[Dict[str, float]] = None,
    score_params: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Compute one row of QC metrics for a single synapse.
    """
    if score_params is None:
        score_params = {
            "support_exp": 0.5,
            "range_k": 0.75,
            "resid_k": 0.8,
        }

    concat = np.concatenate(synapse_segments) if len(synapse_segments) else np.array([], dtype=float)

    row: Dict[str, object] = {
        "session_id": session_id,
        "subject_id": subject_id,
        "dmd": int(dmd),
        "synapse_idx": int(synapse_idx),
        "synapse_id": f"DMD{dmd}_syn{synapse_idx:04d}",
        "n_trials_total": int(n_trials_total),
        "n_valid_trials": int(n_valid_trials),
        "valid_trial_fraction": float(valid_trial_fraction) if np.isfinite(valid_trial_fraction) else np.nan,
        "n_segments": int(len(synapse_segments)),
        "concat_n_samples": int(concat.size),
        "finite_fraction": finite_fraction(concat),
        "trace_median": float(np.nanmedian(concat)) if concat.size else np.nan,
        "trace_mad": mad(concat),
        "trace_sigma_robust": robust_sigma(concat),
        "trace_abs_p95": safe_percentile(np.abs(concat), 95),
        "trace_abs_p99": safe_percentile(np.abs(concat), 99),
        "trace_iqr": (
            safe_percentile(concat, 75) - safe_percentile(concat, 25)
            if concat.size else np.nan
        ),
        "trace_range_robust": robust_range(concat, q_lo=1.0, q_hi=99.0),
        "fs_hz": float(fs_hz),
    }

    row.update(
        residual_snr_segments(
            segments=synapse_segments,
            fs_hz=fs_hz,
            win_s=sg_win_s,
            poly=sg_poly,
            interpolate_nans=True,
        )
    )

    row.update(
        compute_quality_subscores(
            valid_trial_fraction=row["valid_trial_fraction"],
            finite_fraction_value=row["finite_fraction"],
            trace_abs_p99=row["trace_abs_p99"],
            residual_snr_linear=row["residual_snr_linear"],
            support_exp=score_params["support_exp"],
            range_k=score_params["range_k"],
            resid_k=score_params["resid_k"],
        )
    )

    row["quality_score"] = combine_quality_score(
        support_score=row["support_score"],
        finite_score=row["finite_score"],
        range_score=row["range_score"],
        residual_snr_score=row["residual_snr_score"],
        weights=score_weights,
    )

    return row


# =============================================================================
# Metadata
# =============================================================================

def build_qc_metadata(
    summary_mat_path: Union[str, Path],
    session_id: str,
    subject_id: Optional[Union[int, str]],
    trace_signal: str,
    trace_mode: str,
    fs_hz: float,
    sg_win_s: float,
    sg_poly: int,
    score_weights: Dict[str, float],
    score_params: Dict[str, float],
    dmd_summary: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    return {
        "schema_version": "0.1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_metadata": {
            "session_id": session_id,
            "subject_id": subject_id,
            "summary_mat_path": str(summary_mat_path),
            "trace_signal": trace_signal,
            "trace_mode": trace_mode,
            "fs_hz": fs_hz,
        },
        "parameters": {
            "residual_snr": {
                "sg_window_seconds": sg_win_s,
                "sg_polyorder": sg_poly,
                "nan_policy": "NaNs interpolated within valid-trial segments only; invalid-trial gaps not bridged.",
            },
            "quality_score": {
                "weights": score_weights,
                "transforms": {
                    "support_score": "clip(valid_trial_fraction ** support_exp, 0, 1)",
                    "finite_score": "clip(finite_fraction, 0, 1)",
                    "range_score": "1 - exp(-range_k * trace_abs_p99)",
                    "residual_snr_score": "1 - exp(-resid_k * residual_snr_linear)",
                },
                "score_params": score_params,
            },
        },
        "dmd_summary": dmd_summary,
        "metric_descriptions": {
            "session_id": "Session identifier propagated into the synapse QC table.",
            "subject_id": "Subject identifier propagated into the synapse QC table.",
            "dmd": "DMD index (1-indexed) on which the synapse was acquired.",
            "synapse_idx": "Zero-based synapse index within a DMD.",
            "synapse_id": "Session-local synapse identifier combining DMD and synapse index.",
            "n_trials_total": "Total number of trials reported by SummaryLoCo for this session.",
            "n_valid_trials": "Number of valid trials for this DMD.",
            "valid_trial_fraction": "n_valid_trials / n_trials_total for this DMD.",
            "n_segments": "Number of valid-trial trace segments contributing to this synapse.",
            "concat_n_samples": "Total number of samples across all valid-trial segments for this synapse.",
            "finite_fraction": "Fraction of finite values in the concatenated valid-trial trace.",
            "trace_median": "Median value of the concatenated valid-trial trace.",
            "trace_mad": "Median absolute deviation of the concatenated valid-trial trace.",
            "trace_sigma_robust": "Robust scale estimate of the concatenated valid-trial trace, computed as 1.4826 * MAD.",
            "trace_abs_p95": "95th percentile of absolute trace magnitude for the concatenated valid-trial trace.",
            "trace_abs_p99": "99th percentile of absolute trace magnitude for the concatenated valid-trial trace.",
            "trace_iqr": "Interquartile range of the concatenated valid-trial trace.",
            "trace_range_robust": "Robust trace range computed as q99 - q1 of the concatenated valid-trial trace.",
            "fs_hz": "Sampling rate in Hz used for residual SNR estimation.",
            "smooth_sigma_robust": "Robust scale of the Savitzky-Golay smoothed trace pooled across valid-trial segments.",
            "residual_sigma_robust": "Robust scale of the residual (trace - smoothed trace) pooled across valid-trial segments.",
            "residual_snr_linear": "Ratio of smooth_sigma_robust to residual_sigma_robust.",
            "residual_snr_db": "20 * log10(residual_snr_linear).",
            "support_score": "Continuous score in [0,1] summarizing valid trial support.",
            "finite_score": "Continuous score in [0,1] summarizing finite sample completeness.",
            "range_score": "Continuous score in [0,1] summarizing dynamic range proxy using trace_abs_p99.",
            "residual_snr_score": "Continuous score in [0,1] summarizing temporal structure relative to residual noise.",
            "quality_score": "Weighted composite quality score in [0,1] combining support, completeness, dynamic range, and residual SNR.",
            "quality_rank_within_dmd": "Rank of the synapse within its DMD, sorted by descending quality_score (1 = best).",
            "quality_percentile_within_dmd": "Percentile rank of the synapse within its DMD based on quality_score.",
            "recommended_for_analysis": "Convenience boolean flag based on within-DMD percentile and finite data completeness; not intended as a hard truth label.",
        },
    }


# =============================================================================
# Public API
# =============================================================================

@dataclass
class SessionQCResult:
    qc_df: pd.DataFrame
    metadata: Dict[str, object]
    summary: Dict[str, object]


def run_session_synapse_qc(
    summary_mat_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    session_id: Optional[str] = None,
    subject_id: Optional[Union[int, str]] = None,
    trace_signal: str = "dF",
    trace_mode: str = "ls",
    fs_hz: Optional[float] = None,
    sg_win_s: float = 0.5,
    sg_poly: int = 3,
    score_weights: Optional[Dict[str, float]] = None,
    score_params: Optional[Dict[str, float]] = None,
    save: bool = True,
    make_plots: bool = True,
) -> SessionQCResult:
    """
    Compute behavior-independent synapse QC from SummaryLoCo*.mat alone.
    """
    summary_mat_path = Path(summary_mat_path)

    if score_weights is None:
        score_weights = {
            "support_score": 0.20,
            "finite_score": 0.20,
            "range_score": 0.25,
            "residual_snr_score": 0.35,
        }

    if score_params is None:
        score_params = {
            "support_exp": 0.5,
            "range_k": 0.75,
            "resid_k": 0.8,
        }

    if output_dir is None:
        output_dir = summary_mat_path.parent / "analysis" / "qc"
    output_dir = Path(output_dir)

    exp = GlutamateSummary(summary_mat_path)

    if fs_hz is None:
        fs_hz = float(exp.metadata.get("analyzeHz", np.nan))
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError(
            "fs_hz was not provided and could not be inferred from summary metadata['analyzeHz']."
        )

    if session_id is None:
        session_id = summary_mat_path.parent.name

    rows: List[Dict[str, object]] = []
    dmd_summary: Dict[str, Dict[str, float]] = {}

    for dmd in range(1, exp.n_dmds + 1):
        synapse_segments, dmd_info = collect_dmd_synapse_segments(
            exp=exp,
            dmd=dmd,
            signal=trace_signal,
            mode=trace_mode,
        )
        dmd_summary[f"DMD{dmd}"] = dmd_info

        for syn_idx, segments in enumerate(synapse_segments):
            row = compute_synapse_quality_metrics(
                synapse_segments=segments,
                session_id=session_id,
                subject_id=subject_id,
                dmd=dmd,
                synapse_idx=syn_idx,
                n_trials_total=dmd_info["n_trials_total"],
                n_valid_trials=dmd_info["n_valid_trials"],
                valid_trial_fraction=dmd_info["valid_trial_fraction"],
                fs_hz=fs_hz,
                sg_win_s=sg_win_s,
                sg_poly=sg_poly,
                score_weights=score_weights,
                score_params=score_params,
            )
            rows.append(row)

    try:
        exp.close()
    except Exception:
        pass

    qc_df = pd.DataFrame(rows)

    qc_df["quality_rank_within_dmd"] = (
        qc_df.groupby("dmd")["quality_score"]
        .rank(method="min", ascending=False)
        .astype(float)
    )
    qc_df["quality_percentile_within_dmd"] = (
        qc_df.groupby("dmd")["quality_score"]
        .rank(method="average", pct=True)
        .astype(float)
    )

    qc_df["recommended_for_analysis"] = (
        (qc_df["quality_percentile_within_dmd"] >= 0.25) &
        (qc_df["finite_fraction"] >= 0.90)
    )

    metadata = build_qc_metadata(
        summary_mat_path=summary_mat_path,
        session_id=session_id,
        subject_id=subject_id,
        trace_signal=trace_signal,
        trace_mode=trace_mode,
        fs_hz=fs_hz,
        sg_win_s=sg_win_s,
        sg_poly=sg_poly,
        score_weights=score_weights,
        score_params=score_params,
        dmd_summary=dmd_summary,
    )

    summary = {
        "session_id": session_id,
        "subject_id": subject_id,
        "summary_mat_path": str(summary_mat_path),
        "n_synapses_total": int(len(qc_df)),
        "n_dmds": int(exp.n_dmds),
        "dmd_summary": dmd_summary,
        "quality_score_summary": (
            qc_df.groupby("dmd")["quality_score"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .to_dict()
            if len(qc_df) > 0 else {}
        ),
    }

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

        qc_df.to_csv(output_dir / "synapse_qc.csv", index=False)

        try:
            qc_df.to_parquet(output_dir / "synapse_qc.parquet", index=False)
        except Exception:
            pass

        with open(output_dir / "qc_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        with open(output_dir / "session_qc_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        for dmd in sorted(qc_df["dmd"].dropna().unique()):
            dft = qc_df.loc[qc_df["dmd"] == dmd].sort_values("synapse_idx")
            np.save(
                output_dir / f"dmd{int(dmd)}_recommended_synapses.npy",
                dft["recommended_for_analysis"].values.astype(bool),
            )

        if make_plots and len(qc_df) > 0:
            make_all_synapse_qc_plots(qc_df=qc_df, save_dir=output_dir)

    return SessionQCResult(qc_df=qc_df, metadata=metadata, summary=summary)