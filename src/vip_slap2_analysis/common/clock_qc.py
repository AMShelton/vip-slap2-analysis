"""Direct SLAP2-to-HARP acquisition-clock quality control.

The time-alignment pipeline uses HARP as the experiment clock and raw SLAP2
metadata as the acquisition-duration authority.  This module provides an
independent hardware-level audit: compare the HARP DI3 rising-edge cadence and
counts with the cycle/line metadata saved by the SLAP2 extractor.

Modern VIP SLAP2 sessions generally expose the DMD1 cycle clock on DI3. Older
sessions can expose a different integer-line cadence, so the relationship is
inferred from the observed HARP period rather than hard-coded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd


class AcquisitionSummary(Protocol):
    """Subset of summary-reader methods required for clock QC."""

    n_dmds: int

    def get_dmd_metadata(self, dmd: int) -> Dict[str, Any]: ...
    def get_dmd_epoch_metadata(self, dmd: int) -> List[Dict[str, Any]]: ...


def _float_scalar(value: Any) -> float:
    try:
        out = float(np.asarray(value).squeeze())
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _rising_edge_times_from_harp_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=["DI3", "time"])
    state = df["DI3"].astype(bool).to_numpy()
    times = df["time"].to_numpy(dtype=float)
    if state.size == 0:
        return np.empty((0,), dtype=float)
    rising = state & np.r_[True, ~state[:-1]]
    return times[np.flatnonzero(rising)]


def _robust_cadence_period_from_edges(edge_t: np.ndarray) -> float:
    """Estimate pulse cadence while excluding acquisition gaps/short fragments.

    HARP timestamps are quantized, so the median individual interval can be a few
    microseconds away from the true SLAP2 cadence.  A duration-over-interval-count
    estimate across long contiguous pulse trains averages that quantization error
    down and is more useful for line-period comparison.
    """
    edge_t = np.asarray(edge_t, dtype=float).reshape(-1)
    if edge_t.size < 2:
        return float("nan")
    dt = np.diff(edge_t)
    good = np.isfinite(dt) & (dt > 0)
    if not good.any():
        return float("nan")
    median_period = float(np.median(dt[good]))
    gap_threshold = max(10.0 * median_period, 0.5)
    split_after = np.flatnonzero(dt > gap_threshold)
    starts = np.r_[0, split_after + 1]
    stops = np.r_[split_after + 1, edge_t.size]

    total_duration = 0.0
    total_intervals = 0
    for start, stop in zip(starts, stops):
        n_edges = int(stop - start)
        if n_edges < 10:
            continue
        duration = float(edge_t[stop - 1] - edge_t[start])
        # Match the physiology QC convention: short acquisition fragments are not
        # used to estimate the stable acquisition clock.
        if duration < 30.0:
            continue
        total_duration += duration
        total_intervals += n_edges - 1
    if total_intervals > 0:
        return float(total_duration / total_intervals)
    return median_period


def _load_harp_pulse_summary(
    *,
    behavior_qc_dir: Optional[Path] = None,
    harp_df_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load modern DI3 diagnostics, falling back to direct HARP-CSV counting."""
    out: Dict[str, Any] = {
        "source": "unavailable",
        "n_rising_edges": None,
        "median_period_s": None,
        "cadence_period_s": None,
        "accepted_epochs": [],
    }
    if behavior_qc_dir is not None:
        behavior_qc_dir = Path(behavior_qc_dir)
        detection_json = behavior_qc_dir / "di3_pulse_train_detection.json"
        if detection_json.exists():
            try:
                data = json.loads(detection_json.read_text())
                accepted = [
                    row for row in data.get("candidate_epochs", [])
                    if bool(row.get("accepted", False))
                ]
                total_duration = 0.0
                total_intervals = 0
                for row in accepted:
                    try:
                        n_pulses = int(row.get("n_pulses", 0))
                        duration_s = float(row.get("duration_s"))
                    except Exception:
                        continue
                    if n_pulses >= 2 and np.isfinite(duration_s) and duration_s > 0:
                        total_duration += duration_s
                        total_intervals += n_pulses - 1
                cadence = (
                    float(total_duration / total_intervals)
                    if total_intervals > 0
                    else float(data.get("median_period_s"))
                )
                out.update({
                    "source": "di3_pulse_train_detection.json",
                    "n_rising_edges": int(data.get("n_rising_edges")),
                    "median_period_s": float(data.get("median_period_s")),
                    "cadence_period_s": cadence,
                    "accepted_epochs": accepted,
                })
                return out
            except Exception:
                pass

        imaging_csv = behavior_qc_dir / "imaging_epochs.csv"
        if imaging_csv.exists():
            try:
                epoch_df = pd.read_csv(imaging_csv)
                if "n_pulses" in epoch_df.columns:
                    out["accepted_epochs"] = [
                        {
                            "source_epoch_index": int(row.get("pulse_source_epoch_index", i + 1)),
                            "duration_s": float(row["duration_s"]),
                            "n_pulses": int(row["n_pulses"]),
                            "accepted": True,
                        }
                        for i, row in epoch_df.iterrows()
                    ]
            except Exception:
                pass

    if harp_df_csv is not None and Path(harp_df_csv).exists():
        try:
            edge_t = _rising_edge_times_from_harp_csv(Path(harp_df_csv))
            if edge_t.size:
                intervals = np.diff(edge_t)
                intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
                out.update({
                    "source": "harp_df_csv",
                    "n_rising_edges": int(edge_t.size),
                    "median_period_s": float(np.median(intervals)) if intervals.size else None,
                    "cadence_period_s": _robust_cadence_period_from_edges(edge_t),
                })
        except Exception:
            pass
    return out


def _epoch_line_rate(row: Dict[str, Any], fallback: Dict[str, Any]) -> float:
    md = row.get("metadata", {})
    if isinstance(md, dict):
        value = _float_scalar(md.get("lineRateHz"))
        if np.isfinite(value) and value > 0:
            return value
    return _float_scalar(fallback.get("lineRateHz"))


def compare_slap2_harp_clock(
    summary: AcquisitionSummary,
    *,
    behavior_qc_dir: Optional[Path] = None,
    harp_df_csv: Optional[Path] = None,
    direct_cycle_tolerance_lines: float = 0.15,
    integer_line_tolerance_lines: float = 0.15,
) -> Dict[str, Any]:
    """Compare HARP DI3 with raw SLAP2 acquisition metadata.

    Returns a compact JSON-serializable report. The closest DMD cycle cadence is
    selected automatically. A direct cycle-count comparison is only declared valid
    when the observed HARP period is within ``direct_cycle_tolerance_lines`` of that
    DMD's ``linesPerCycle``. Otherwise an integer-line cadence is reported without
    incorrectly treating the difference as dropped SLAP2 cycles.
    """
    harp = _load_harp_pulse_summary(
        behavior_qc_dir=behavior_qc_dir,
        harp_df_csv=harp_df_csv,
    )
    median_period = _float_scalar(harp.get("median_period_s"))
    cadence_period = _float_scalar(harp.get("cadence_period_s"))
    period = cadence_period if np.isfinite(cadence_period) and cadence_period > 0 else median_period
    n_edges = harp.get("n_rising_edges")

    dmd_rows: List[Dict[str, Any]] = []
    n_dmds = int(getattr(summary, "n_dmds", 0) or 0)
    for dmd in range(1, n_dmds + 1):
        md = summary.get_dmd_metadata(dmd)
        epochs = summary.get_dmd_epoch_metadata(dmd)
        line_rate = _float_scalar(md.get("lineRateHz"))
        lines_per_cycle = _float_scalar(md.get("linesPerCycle"))
        if (not np.isfinite(line_rate) or line_rate <= 0) and epochs:
            line_rate = _epoch_line_rate(epochs[0], md)
        if (not np.isfinite(lines_per_cycle) or lines_per_cycle <= 0) and epochs:
            lines_per_cycle = _float_scalar(epochs[0].get("linesPerCycle"))

        expected_period = (
            float(lines_per_cycle / line_rate)
            if np.isfinite(line_rate) and line_rate > 0 and np.isfinite(lines_per_cycle) and lines_per_cycle > 0
            else float("nan")
        )
        effective_lines = (
            float(period * line_rate)
            if np.isfinite(period) and period > 0 and np.isfinite(line_rate) and line_rate > 0
            else float("nan")
        )
        source_total_cycles = int(sum(
            int(row.get("nCycles", 0) or 0)
            for row in epochs if bool(row.get("available", True))
        )) if epochs else None
        dmd_rows.append({
            "dmd": int(dmd),
            "line_rate_hz": line_rate if np.isfinite(line_rate) else None,
            "lines_per_cycle": int(round(lines_per_cycle)) if np.isfinite(lines_per_cycle) else None,
            "expected_cycle_period_s": expected_period if np.isfinite(expected_period) else None,
            "expected_cycle_rate_hz": (1.0 / expected_period) if np.isfinite(expected_period) and expected_period > 0 else None,
            "effective_lines_per_harp_pulse": effective_lines if np.isfinite(effective_lines) else None,
            "source_total_cycles": source_total_cycles,
            "n_source_epochs": int(len(epochs)),
        })

    candidates = [
        row for row in dmd_rows
        if row["expected_cycle_period_s"] is not None and np.isfinite(period) and period > 0
    ]
    reference: Optional[Dict[str, Any]] = None
    if candidates:
        reference = min(
            candidates,
            key=lambda row: abs(period / float(row["expected_cycle_period_s"]) - 1.0),
        )

    relationship = "unresolved"
    direct_count_valid = False
    integer_line_cadence = None
    line_offset_from_cycle = None
    period_error_ppm = None
    if reference is not None:
        eff = float(reference["effective_lines_per_harp_pulse"])
        lpc = float(reference["lines_per_cycle"])
        nearest_line = int(round(eff))
        integer_line_cadence = nearest_line
        line_offset_from_cycle = float(eff - lpc)
        period_error_ppm = float(
            (period / float(reference["expected_cycle_period_s"]) - 1.0) * 1e6
        )
        if abs(eff - lpc) <= float(direct_cycle_tolerance_lines):
            relationship = "direct_dmd_cycle_clock"
            direct_count_valid = True
        elif abs(eff - nearest_line) <= float(integer_line_tolerance_lines):
            relationship = "integer_line_cadence_not_dmd_cycle"
        else:
            relationship = "noninteger_or_unresolved_cadence"

    result: Dict[str, Any] = {
        "available": bool(np.isfinite(period) and reference is not None),
        "harp_source": harp.get("source"),
        "harp_n_rising_edges": int(n_edges) if n_edges is not None else None,
        "harp_median_period_s": median_period if np.isfinite(median_period) else None,
        "harp_cadence_period_s": period if np.isfinite(period) else None,
        "harp_cadence_rate_hz": (1.0 / period) if np.isfinite(period) and period > 0 else None,
        "relationship": relationship,
        "reference_dmd": int(reference["dmd"]) if reference is not None else None,
        "effective_lines_per_harp_pulse": (
            float(reference["effective_lines_per_harp_pulse"]) if reference is not None else None
        ),
        "nearest_integer_line_cadence": integer_line_cadence,
        "line_offset_from_reference_cycle": line_offset_from_cycle,
        "period_error_ppm_vs_reference_cycle": period_error_ppm,
        "direct_cycle_count_comparison_valid": bool(direct_count_valid),
        "dmds": dmd_rows,
    }

    if direct_count_valid and reference is not None and n_edges is not None:
        source_total = reference.get("source_total_cycles")
        if source_total is not None and int(source_total) > 0:
            diff = int(n_edges) - int(source_total)
            result["session_cycle_count"] = {
                "harp_pulses": int(n_edges),
                "slap2_cycles": int(source_total),
                "difference_cycles": int(diff),
                "fractional_error": float(diff / int(source_total)),
            }

        source_epochs = summary.get_dmd_epoch_metadata(int(reference["dmd"]))
        accepted_harp = list(harp.get("accepted_epochs", []))
        if len(source_epochs) == len(accepted_harp) and source_epochs:
            per_epoch = []
            for source_row, harp_row in zip(source_epochs, accepted_harp):
                source_cycles = int(source_row.get("nCycles", 0) or 0)
                harp_pulses = int(harp_row.get("n_pulses", 0) or 0)
                per_epoch.append({
                    "source_epoch": int(source_row.get("epochIdx", len(per_epoch) + 1)),
                    "slap2_cycles": source_cycles,
                    "harp_accepted_pulses": harp_pulses,
                    "difference_cycles": int(harp_pulses - source_cycles),
                })
            result["accepted_epoch_cycle_counts"] = per_epoch

    return result
