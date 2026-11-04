"""Register HARP DI3 pulses to absolute SLAP2 acquisition-cycle indices.

Problem
-------
The processing pipeline places the first sample of each imaging epoch at the
first observed DI3 rising edge of the matching HARP pulse-train epoch
(``common.epoch_alignment.build_epoch_aware_timebase``). That is only correct
when the HARP recording covers the *start* of the SLAP2 acquisition. In
SLAP2-first recordings (roughly 2026-07-23 to 2026-10-27) the leading pulses
were never recorded, so "first observed pulse" is some unknown cycle ``k > 0``.

Identifiability (established on the 2026-10-27 reference session, see
``docs/RETROSPECTIVE_SYNC.md``)
------------------------------------------------------------------------------
With DI3 verified as the DMD cycle clock (one rise per cycle, rise at the cycle
start line), and the total number of source cycles ``N`` known from the raw
``.dat`` headers (or the extractor's line counts), the integer offset ``k`` of
the first observed pulse is fixed by **either** anchor:

* head anchor: the HARP recording contains a DI3-low period before the first
  pulse (HARP started first) -> ``k = 0``;
* tail anchor: the HARP recording continues past the last pulse (SLAP2 stopped
  first) -> ``k = N - n_observed`` (after accounting for internal dropped pulses).

When both anchors exist the two estimates must agree (consistency check). When
neither exists -- HARP started after SLAP2 *and* stopped before it -- the offset
is **not identifiable**; only the range ``[0, N - n_observed]`` can be stated and
the session must be classified as unresolved. Wall-clock evidence (launcher logs,
``.meta`` FPGA time reference, HARP folder stamps) agrees only to seconds and
cannot break the tie; it is reported, never used to pick an offset.

Nothing in this module modifies physiology or behavior assets. The result is a
JSON provenance record consumed (optionally) by the extraction code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

SCHEMA_VERSION = "vip_slap2_cycle_registration_v1"

STATUS_IDENTIFIED = "identified"
STATUS_IDENTIFIED_ASSUMED = "identified_with_assumptions"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_INCONSISTENT = "inconsistent"
STATUS_UNAVAILABLE = "unavailable"

RELATIONSHIP_DIRECT = "direct_dmd_cycle_clock"
RELATIONSHIP_INTEGER_LINE = "integer_line_cadence_not_dmd_cycle"
RELATIONSHIP_UNRESOLVED = "noninteger_or_unresolved_cadence"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class SourceEpochCycles:
    """Cycle bookkeeping for one source acquisition epoch of one DMD.

    ``cycle_start_s`` (FPGA seconds relative to the acquisition reference) is
    optional; when present (raw ``.dat`` headers available) it is used to verify
    that the source itself has no dropped cycles and to measure the HARP/FPGA
    clock ratio.
    """

    epoch_index: int
    n_cycles: int
    lines_per_cycle: int
    line_rate_hz: float
    cycle_start_s: Optional[np.ndarray] = None
    n_source_gaps: int = 0
    source: str = "summary_metadata"

    @property
    def nominal_cycle_period_s(self) -> float:
        return float(self.lines_per_cycle) / float(self.line_rate_hz)

    @property
    def nominal_duration_s(self) -> float:
        return float(self.n_cycles) * self.nominal_cycle_period_s


@dataclass
class Di3PulseEpoch:
    """One contiguous DI3 pulse train bounded by acquisition-off gaps or recording limits."""

    epoch_index: int
    rise_times_s: np.ndarray
    lead_s: float            # time from previous coverage boundary (gap or recording start) to first rise
    tail_s: float            # time from last rise to next coverage boundary (gap or recording end)
    bounded_before_by: str   # 'recording_start' | 'gap'
    bounded_after_by: str    # 'recording_end' | 'gap'

    @property
    def n_pulses(self) -> int:
        return int(self.rise_times_s.size)

    @property
    def duration_s(self) -> float:
        return float(self.rise_times_s[-1] - self.rise_times_s[0]) if self.n_pulses > 1 else 0.0


def split_di3_pulse_epochs(
    rise_times_s: Sequence[float],
    *,
    record_start_s: float,
    record_end_s: float,
    gap_factor: float = 10.0,
    min_gap_s: float = 0.5,
    min_pulses: int = 10,
) -> Tuple[List[Di3PulseEpoch], Dict[str, Any]]:
    """Split DI3 rises into pulse-train epochs with the behavior pipeline's gap rule.

    The rule matches :func:`vip_slap2_analysis.behavior.epochs.detect_pulse_train_epochs`
    (``gap > max(min_gap_s, gap_factor * median_period)``). Fragments with fewer
    than ``min_pulses`` rises are reported but not returned as epochs.
    """
    t = np.sort(np.asarray(rise_times_s, dtype=float).reshape(-1))
    diag: Dict[str, Any] = {"n_rising_edges": int(t.size), "fragments_dropped": []}
    if t.size < 2:
        return [], diag
    d = np.diff(t)
    period = float(np.median(d[d > 0])) if np.any(d > 0) else float("nan")
    thr = max(float(min_gap_s), float(gap_factor) * period) if np.isfinite(period) else float(min_gap_s)
    gap_idx = np.flatnonzero(d > thr)
    starts = np.r_[0, gap_idx + 1]
    stops = np.r_[gap_idx + 1, t.size]
    epochs: List[Di3PulseEpoch] = []
    for a, b in zip(starts, stops):
        seg = t[a:b]
        if seg.size < int(min_pulses):
            diag["fragments_dropped"].append({"start_s": float(seg[0]), "n_pulses": int(seg.size)})
            continue
        if a == 0:
            lead = float(seg[0] - record_start_s); before = "recording_start"
        else:
            lead = float(seg[0] - t[a - 1]); before = "gap"
        if b == t.size:
            tail = float(record_end_s - seg[-1]); after = "recording_end"
        else:
            tail = float(t[b] - seg[-1]); after = "gap"
        epochs.append(Di3PulseEpoch(len(epochs) + 1, seg, lead, tail, before, after))
    diag.update({"median_period_s": period, "gap_threshold_s": thr, "n_epochs": len(epochs)})
    return epochs, diag


# ---------------------------------------------------------------------------
# Registration result
# ---------------------------------------------------------------------------

@dataclass
class EpochCycleRegistration:
    """Registration of one source epoch to one DI3 pulse-train epoch."""

    source_epoch_index: int
    di3_epoch_index: int
    n_source_cycles: int
    n_observed_pulses: int
    n_internal_missing_pulses: int
    internal_gaps: List[Dict[str, float]]
    head_covered: bool
    tail_covered: bool
    lead_s: float
    tail_s: float
    anchor: str                                 # 'head' | 'tail' | 'both' | 'none'
    status: str
    cycle_index_offset: Optional[int]           # source cycle index of the first observed pulse
    candidate_offset_range: Tuple[int, int]
    missing_initial_cycles: Optional[int]
    missing_terminal_cycles: Optional[int]
    nominal_cycle_period_s: float
    observed_cycle_period_s: float
    rate_ratio_harp_over_source: float          # observed DI3 cadence / nominal source cadence
    period_error_ppm: float
    residual_rms_s: float                       # pulse times vs. linear model (HARP quantization ~32 us)
    residual_max_s: float
    first_pulse_harp_s: float
    last_pulse_harp_s: float
    notes: List[str] = field(default_factory=list)
    # not serialized by default (large)
    pulse_times_harp_s: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("pulse_times_harp_s", None)
        d["candidate_offset_range"] = [int(x) for x in self.candidate_offset_range]
        return d

    # -- timing map ---------------------------------------------------------
    def cycle_times_harp_s(self) -> Tuple[np.ndarray, np.ndarray]:
        """HARP time of every source cycle start and a coverage label per cycle.

        Observed cycles take their measured pulse time. Cycles outside the
        observed span are extrapolated at the observed cadence and labelled
        ``1`` (head) / ``2`` (tail); interior cycles whose pulse was dropped are
        linearly interpolated and labelled ``3``. Label ``0`` = measured.
        Requires ``status`` identified.
        """
        if self.cycle_index_offset is None or self.pulse_times_harp_s is None:
            raise ValueError("cycle times require an identified registration with pulse times")
        N = int(self.n_source_cycles)
        k0 = int(self.cycle_index_offset)
        pulses = np.asarray(self.pulse_times_harp_s, dtype=float)
        period = float(self.observed_cycle_period_s)
        # assign observed pulses to cycle indices, skipping dropped pulses
        idx = np.empty(pulses.size, dtype=int)
        idx[0] = k0
        if pulses.size > 1:
            steps = np.rint(np.diff(pulses) / period).astype(int)
            steps[steps < 1] = 1
            idx[1:] = k0 + np.cumsum(steps)
        times = np.full((N,), np.nan)
        label = np.full((N,), 3, dtype=np.int8)
        valid = (idx >= 0) & (idx < N)
        times[idx[valid]] = pulses[valid]
        label[idx[valid]] = 0
        all_idx = np.arange(N)
        inside = (all_idx >= idx[valid][0]) & (all_idx <= idx[valid][-1]) if valid.any() else np.zeros(N, bool)
        missing_inside = inside & ~np.isfinite(times)
        if missing_inside.any():
            times[missing_inside] = np.interp(all_idx[missing_inside], idx[valid], pulses[valid])
        head = all_idx < (idx[valid][0] if valid.any() else 0)
        tail = all_idx > (idx[valid][-1] if valid.any() else -1)
        if head.any():
            times[head] = pulses[valid][0] - (idx[valid][0] - all_idx[head]) * period
            label[head] = 1
        if tail.any():
            times[tail] = pulses[valid][-1] + (all_idx[tail] - idx[valid][-1]) * period
            label[tail] = 2
        return times, label


@dataclass
class CycleRegistration:
    """Session/DMD-level registration record (the QC/provenance object)."""

    schema_version: str
    session_id: Optional[str]
    dmd: int
    relationship: str
    acquisition_start_order: str
    start_order_evidence: str
    status: str
    safe_for_physiology: bool
    lines_per_cycle: Optional[int]
    line_rate_hz: Optional[float]
    nominal_cycle_period_s: Optional[float]
    observed_cycle_period_s: Optional[float]
    rate_ratio_harp_over_source: Optional[float]
    period_error_ppm: Optional[float]
    effective_lines_per_pulse: Optional[float]
    harp_record_start_s: Optional[float]
    harp_record_end_s: Optional[float]
    harp_n_rising_edges: int
    n_source_epochs: int
    n_di3_epochs: int
    epochs: List[EpochCycleRegistration]
    limitations: List[str]
    inputs: Dict[str, Any]
    wall_clock_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["epochs"] = [e.to_dict() for e in self.epochs]
        return d

    def epoch_by_source_index(self, source_epoch_index: int) -> Optional[EpochCycleRegistration]:
        for e in self.epochs:
            if int(e.source_epoch_index) == int(source_epoch_index):
                return e
        return None

    @property
    def identified(self) -> bool:
        return self.status in (STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED)

    @property
    def behavior_time_origin_harp_s(self) -> Optional[float]:
        """Absolute HARP time of behavior-time zero (first photodiode/AnalogData sample)."""
        v = self.inputs.get("behavior_time_origin_harp_s") if self.inputs else None
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _linear_fit_residual(pulses: np.ndarray, cycle_idx: np.ndarray) -> Tuple[float, float, float]:
    """Fit pulse_time = a*cycle + b; return (a, rms, max_abs) of residuals."""
    if pulses.size < 3:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(cycle_idx.astype(float), pulses, 1)
    r = pulses - (a * cycle_idx + b)
    return float(a), float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


def _register_epoch(
    src: SourceEpochCycles,
    di3: Di3PulseEpoch,
    *,
    relationship: str,
    nominal_period_s: float,
    min_anchor_cycles: float,
    min_anchor_s: float,
    internal_gap_factor: float,
    assume_first_pulse_is_acquisition_start: bool,
) -> EpochCycleRegistration:
    pulses = np.asarray(di3.rise_times_s, dtype=float)
    n_obs = int(pulses.size)
    notes: List[str] = []

    # observed cadence from this train, robust to dropped pulses
    d = np.diff(pulses)
    med = float(np.median(d)) if d.size else float("nan")
    steps = np.rint(d / med).astype(int) if np.isfinite(med) and med > 0 else np.ones_like(d, dtype=int)
    steps[steps < 1] = 1
    gaps = []
    missing_internal = 0
    for i in np.flatnonzero(d > float(internal_gap_factor) * med):
        m = int(steps[i] - 1)
        missing_internal += m
        gaps.append({"after_pulse_index": int(i), "gap_start_s": float(pulses[i]), "gap_s": float(d[i]), "missing_pulses": m})
    cycle_idx_rel = np.r_[0, np.cumsum(steps)] if n_obs > 1 else np.zeros((n_obs,), dtype=int)
    slope, rms, rmax = _linear_fit_residual(pulses, cycle_idx_rel)
    observed_period = slope if np.isfinite(slope) else med
    ratio = float(observed_period / nominal_period_s) if np.isfinite(observed_period) and nominal_period_s > 0 else float("nan")
    ppm = float((ratio - 1.0) * 1e6) if np.isfinite(ratio) else float("nan")

    anchor_thr = max(float(min_anchor_s), float(min_anchor_cycles) * nominal_period_s)
    head = bool(di3.lead_s >= anchor_thr)
    tail = bool(di3.tail_s >= anchor_thr)
    anchor = "both" if head and tail else "head" if head else "tail" if tail else "none"

    N = int(src.n_cycles)
    n_eff = n_obs + missing_internal
    offset: Optional[int] = None
    rng: Tuple[int, int]
    status = STATUS_AMBIGUOUS
    missing_initial: Optional[int] = None
    missing_terminal: Optional[int] = None

    if relationship != RELATIONSHIP_DIRECT:
        notes.append(
            f"DI3 cadence is '{relationship}', not one pulse per DMD cycle; integer cycle registration by "
            "pulse counting is not defined. Only coverage anchors are reported."
        )
        rng = (0, max(0, N))
        if head and assume_first_pulse_is_acquisition_start:
            offset = 0
            status = STATUS_IDENTIFIED_ASSUMED
            notes.append("Head coverage present; first pulse assumed to mark acquisition start (unverified phase).")
        elif head:
            rng = (0, 0)
    elif n_eff > N:
        status = STATUS_INCONSISTENT
        rng = (0, 0)
        notes.append(f"More DI3 pulses ({n_eff}, incl. {missing_internal} inferred dropped) than source cycles ({N}).")
    elif n_eff == N:
        if head or tail:
            offset = 0; status = STATUS_IDENTIFIED; rng = (0, 0)
            missing_initial = 0; missing_terminal = 0
            if not (head and tail):
                notes.append("Pulse count equals source cycle count; single anchor used, other end at recording limit.")
        else:
            # count equality with no margin at either end is still consistent only with k=0
            offset = 0; status = STATUS_IDENTIFIED_ASSUMED; rng = (0, 0)
            missing_initial = 0; missing_terminal = 0
            notes.append("Pulse count equals source cycle count but neither recording boundary has margin; "
                         "k=0 is the only count-consistent offset, assumed.")
    else:  # n_eff < N : some cycles have no recorded pulse
        deficit = N - n_eff
        if head and tail:
            status = STATUS_INCONSISTENT; rng = (0, deficit)
            notes.append(f"Both anchors covered but {deficit} source cycles lack pulses and no internal gaps explain them.")
        elif head:
            offset = 0; status = STATUS_IDENTIFIED; rng = (0, 0)
            missing_initial = 0; missing_terminal = deficit
        elif tail:
            offset = deficit; status = STATUS_IDENTIFIED; rng = (deficit, deficit)
            missing_initial = deficit; missing_terminal = 0
            notes.append("Tail anchor: HARP recorded past the last pulse; first observed pulse is cycle N - n_observed.")
        else:
            rng = (0, deficit)
            notes.append(f"Neither recording boundary has margin: offset in [0, {deficit}] is not identifiable from DI3 alone.")

    return EpochCycleRegistration(
        source_epoch_index=int(src.epoch_index),
        di3_epoch_index=int(di3.epoch_index),
        n_source_cycles=N,
        n_observed_pulses=n_obs,
        n_internal_missing_pulses=int(missing_internal),
        internal_gaps=gaps,
        head_covered=head,
        tail_covered=tail,
        lead_s=float(di3.lead_s),
        tail_s=float(di3.tail_s),
        anchor=anchor,
        status=status,
        cycle_index_offset=offset,
        candidate_offset_range=rng,
        missing_initial_cycles=missing_initial,
        missing_terminal_cycles=missing_terminal,
        nominal_cycle_period_s=float(nominal_period_s),
        observed_cycle_period_s=float(observed_period),
        rate_ratio_harp_over_source=ratio,
        period_error_ppm=ppm,
        residual_rms_s=rms,
        residual_max_s=rmax,
        first_pulse_harp_s=float(pulses[0]),
        last_pulse_harp_s=float(pulses[-1]),
        notes=notes,
        pulse_times_harp_s=pulses,
    )


def infer_clock_relationship(
    observed_period_s: float,
    *,
    lines_per_cycle: float,
    line_rate_hz: float,
    direct_tolerance_lines: float = 0.15,
    integer_tolerance_lines: float = 0.15,
) -> Tuple[str, float]:
    """Classify DI3 cadence vs the DMD cycle (same rule as ``common.clock_qc``)."""
    if not (np.isfinite(observed_period_s) and observed_period_s > 0 and line_rate_hz > 0):
        return RELATIONSHIP_UNRESOLVED, float("nan")
    eff = float(observed_period_s * line_rate_hz)
    if abs(eff - float(lines_per_cycle)) <= direct_tolerance_lines:
        return RELATIONSHIP_DIRECT, eff
    if abs(eff - round(eff)) <= integer_tolerance_lines:
        return RELATIONSHIP_INTEGER_LINE, eff
    return RELATIONSHIP_UNRESOLVED, eff


def register_di3_to_source_cycles(
    *,
    rise_times_s: Sequence[float],
    record_start_s: float,
    record_end_s: float,
    source_epochs: Sequence[SourceEpochCycles],
    dmd: int,
    session_id: Optional[str] = None,
    relationship: Optional[str] = None,
    min_anchor_cycles: float = 10.0,
    min_anchor_s: float = 0.5,
    epoch_gap_factor: float = 10.0,
    epoch_min_gap_s: float = 0.5,
    epoch_min_pulses: int = 10,
    internal_gap_factor: float = 1.5,
    assume_first_pulse_is_acquisition_start: bool = False,
    inputs: Optional[Dict[str, Any]] = None,
    wall_clock_evidence: Optional[Dict[str, Any]] = None,
) -> CycleRegistration:
    """Register DI3 pulse-train epochs to source acquisition epochs for one DMD.

    Source epochs and DI3 epochs are paired chronologically when their counts
    agree. Otherwise every epoch is reported ``inconsistent`` (no guessing).
    """
    src_list = list(source_epochs)
    rises = np.asarray(rise_times_s, dtype=float)
    di3_epochs, split_diag = split_di3_pulse_epochs(
        rises, record_start_s=record_start_s, record_end_s=record_end_s,
        gap_factor=epoch_gap_factor, min_gap_s=epoch_min_gap_s, min_pulses=epoch_min_pulses,
    )
    limitations: List[str] = []
    lpc = int(src_list[0].lines_per_cycle) if src_list else None
    lr = float(src_list[0].line_rate_hz) if src_list else None
    nominal = float(lpc / lr) if (lpc and lr) else float("nan")
    med_period = float(split_diag.get("median_period_s", float("nan")))
    # cadence over long trains only (quantization averages out)
    long_trains = [e for e in di3_epochs if e.n_pulses >= 10 and e.duration_s >= 30.0] or di3_epochs
    if long_trains:
        tot_d = sum(e.duration_s for e in long_trains); tot_n = sum(e.n_pulses - 1 for e in long_trains)
        observed = tot_d / tot_n if tot_n > 0 else med_period
    else:
        observed = med_period
    if relationship is None:
        if lpc and lr:
            relationship, eff = infer_clock_relationship(observed, lines_per_cycle=lpc, line_rate_hz=lr)
        else:
            relationship, eff = RELATIONSHIP_UNRESOLVED, float("nan")
    else:
        eff = float(observed * lr) if (lr and np.isfinite(observed)) else float("nan")

    # start order from the first DI3 epoch
    if di3_epochs:
        first = di3_epochs[0]
        thr = max(min_anchor_s, min_anchor_cycles * nominal) if np.isfinite(nominal) else min_anchor_s
        if first.lead_s >= thr:
            order, evidence = "harp_first", f"DI3 low for {first.lead_s:.3f} s before first pulse (>= {thr:.3f} s)"
        else:
            order, evidence = "slap2_first_or_simultaneous", f"first pulse {first.lead_s:.4f} s after HARP recording start (< {thr:.3f} s): leading pulses may be unrecorded"
    else:
        order, evidence = "unknown", "no DI3 pulse train"

    epochs: List[EpochCycleRegistration] = []
    if not src_list or not di3_epochs:
        status = STATUS_UNAVAILABLE
        limitations.append("Missing source epochs or DI3 pulse trains.")
    elif len(src_list) != len(di3_epochs):
        status = STATUS_INCONSISTENT
        limitations.append(
            f"{len(src_list)} source epoch(s) vs {len(di3_epochs)} DI3 pulse-train epoch(s); chronological pairing refused."
        )
    else:
        for s, e in zip(src_list, di3_epochs):
            reg = _register_epoch(
                s, e, relationship=relationship, nominal_period_s=s.nominal_cycle_period_s,
                min_anchor_cycles=min_anchor_cycles, min_anchor_s=min_anchor_s,
                internal_gap_factor=internal_gap_factor,
                assume_first_pulse_is_acquisition_start=assume_first_pulse_is_acquisition_start,
            )
            if s.n_source_gaps:
                reg.notes.append(f"Source .dat headers show {s.n_source_gaps} cycle-timing gap(s) inside this epoch.")
            epochs.append(reg)
        statuses = {e.status for e in epochs}
        if statuses <= {STATUS_IDENTIFIED}:
            status = STATUS_IDENTIFIED
        elif statuses <= {STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED}:
            status = STATUS_IDENTIFIED_ASSUMED
        elif STATUS_INCONSISTENT in statuses:
            status = STATUS_INCONSISTENT
        else:
            status = STATUS_AMBIGUOUS
    if relationship != RELATIONSHIP_DIRECT:
        limitations.append(f"DI3 cadence relationship '{relationship}': pulse counting cannot give integer cycle offsets.")
    if any(not e.head_covered for e in epochs):
        limitations.append("At least one epoch lacks head coverage (HARP started after the pulse train began).")
    if any(not e.tail_covered for e in epochs):
        limitations.append("At least one epoch lacks tail coverage (HARP stopped before the pulse train ended).")
    if np.isfinite(observed) and np.isfinite(nominal) and nominal > 0:
        ratio = observed / nominal; ppm = (ratio - 1) * 1e6
    else:
        ratio = ppm = float("nan")
    safe = status in (STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED)
    return CycleRegistration(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        dmd=int(dmd),
        relationship=relationship,
        acquisition_start_order=order,
        start_order_evidence=evidence,
        status=status,
        safe_for_physiology=bool(safe),
        lines_per_cycle=lpc,
        line_rate_hz=lr,
        nominal_cycle_period_s=nominal if np.isfinite(nominal) else None,
        observed_cycle_period_s=observed if np.isfinite(observed) else None,
        rate_ratio_harp_over_source=float(ratio) if np.isfinite(ratio) else None,
        period_error_ppm=float(ppm) if np.isfinite(ppm) else None,
        effective_lines_per_pulse=float(eff) if np.isfinite(eff) else None,
        harp_record_start_s=float(record_start_s),
        harp_record_end_s=float(record_end_s),
        harp_n_rising_edges=int(rises.size),
        n_source_epochs=len(src_list),
        n_di3_epochs=len(di3_epochs),
        epochs=epochs,
        limitations=limitations,
        inputs=dict(inputs or {}),
        wall_clock_evidence=dict(wall_clock_evidence or {}),
    )


# ---------------------------------------------------------------------------
# Source-line -> HARP-time mapping
# ---------------------------------------------------------------------------

def source_lines_to_harp_time(
    lines_1based: np.ndarray,
    epoch_reg: EpochCycleRegistration,
    *,
    lines_per_cycle: int,
    line_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map 1-based source line indices of one epoch to HARP seconds.

    Cycle starts take measured DI3 times (see
    :meth:`EpochCycleRegistration.cycle_times_harp_s`); within a cycle the
    nominal line period is scaled by the measured HARP/source rate ratio.
    Returns ``(time_s, coverage_label)`` with the per-cycle label of each sample.
    """
    lines = np.asarray(lines_1based, dtype=np.int64).reshape(-1)
    cyc_t, cyc_label = epoch_reg.cycle_times_harp_s()
    lpc = int(lines_per_cycle)
    cyc = (lines - 1) // lpc
    within = (lines - 1) % lpc
    if np.any(cyc < 0) or np.any(cyc >= cyc_t.size):
        raise ValueError("source line indices fall outside the registered cycle range")
    ratio = epoch_reg.rate_ratio_harp_over_source if np.isfinite(epoch_reg.rate_ratio_harp_over_source) else 1.0
    line_period_harp = ratio / float(line_rate_hz)
    return cyc_t[cyc] + within * line_period_harp, cyc_label[cyc]


@dataclass
class RegisteredSamplePlan:
    """Per-trial source-line bookkeeping derived from a registration.

    Produced by :func:`plan_registered_samples` and consumed by the modality
    ``reconstruct_*`` functions: how many leading samples of each *source* epoch
    precede HARP coverage, and a callable giving behavior-relative HARP times for
    any retained (source epoch, 1-based line) pairs.
    """

    source_epoch_leading_samples: Dict[int, int]
    applied_source_epochs: List[int]
    skipped_source_epochs: Dict[int, str]
    behavior_time_origin_harp_s: float
    method: str
    notes: List[str] = field(default_factory=list)

    def lines_to_behavior_time(
        self, reg: "CycleRegistration", source_epoch_index: int, lines_1based: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ereg = reg.epoch_by_source_index(source_epoch_index)
        if ereg is None or ereg.cycle_index_offset is None:
            raise ValueError(f"source epoch {source_epoch_index} is not identified in the registration")
        t, lab = source_lines_to_harp_time(
            lines_1based, ereg, lines_per_cycle=int(reg.lines_per_cycle), line_rate_hz=float(reg.line_rate_hz)
        )
        return t - float(self.behavior_time_origin_harp_s), lab


def plan_registered_samples(
    reg: CycleRegistration,
    *,
    trial_source_lines: Dict[int, np.ndarray],
    trial_source_epoch: Sequence[int],
    behavior_time_origin_harp_s: float,
) -> RegisteredSamplePlan:
    """Translate an identified registration into leading-sample skips per source epoch.

    ``trial_source_lines`` maps 1-based trial number -> 1-based source line of
    every sample of that trial (after any discard mask), ``trial_source_epoch``
    gives the source epoch label per trial (1-based list aligned with trial
    numbers 1..n). Leading samples are those whose cycle index is below the
    epoch's ``cycle_index_offset``. Epochs that are not identified are skipped
    (reported in ``skipped_source_epochs``) and keep the historical placement.
    """
    if reg.lines_per_cycle is None:
        raise ValueError("registration lacks lines_per_cycle")
    lpc = int(reg.lines_per_cycle)
    labels = np.asarray(trial_source_epoch, dtype=int).reshape(-1)
    leading: Dict[int, int] = {}
    applied: List[int] = []
    skipped: Dict[int, str] = {}
    notes: List[str] = []
    for ereg in reg.epochs:
        sid = int(ereg.source_epoch_index)
        if ereg.cycle_index_offset is None:
            skipped[sid] = f"status={ereg.status}; offset range {list(ereg.candidate_offset_range)}"
            continue
        k0 = int(ereg.cycle_index_offset)
        trials = [t for t in range(1, labels.size + 1) if labels[t - 1] == sid]
        n_lead = 0
        for t in trials:
            lines = trial_source_lines.get(t)
            if lines is None:
                continue
            cyc = (np.asarray(lines, dtype=np.int64) - 1) // lpc
            n_lead += int(np.sum(cyc < k0))
        leading[sid] = n_lead
        applied.append(sid)
        if k0 > 0:
            notes.append(f"Source epoch {sid}: skipping {n_lead} leading samples (first {k0} cycles have no HARP coverage).")
    return RegisteredSamplePlan(
        source_epoch_leading_samples=leading,
        applied_source_epochs=applied,
        skipped_source_epochs=skipped,
        behavior_time_origin_harp_s=float(behavior_time_origin_harp_s),
        method="di3_pulse_registered",
        notes=notes,
    )


def registered_epoch_sample_times(
    reg: CycleRegistration,
    plan: RegisteredSamplePlan,
    *,
    trial_source_lines: Dict[int, np.ndarray],
    source_trial_epoch: Sequence[int],
    analysis_trial_epoch: Sequence[int],
    trial_lengths_samples: Sequence[int],
    trial_source_start_samples: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Any]]:
    """Measured behavior-time of every retained sample, grouped by analysis epoch.

    Returns ``(epoch_sample_times, epoch_sample_labels, info)`` suitable for
    :func:`vip_slap2_analysis.common.epoch_alignment.build_epoch_aware_timebase`.
    Analysis epochs whose source epoch is not identified, or whose trials lack
    source lines, are omitted (they keep the nominal-rate placement) and listed
    in ``info['epochs_without_measured_times']``.
    """
    src = np.asarray(source_trial_epoch, dtype=int).reshape(-1)
    ana = np.asarray(analysis_trial_epoch, dtype=int).reshape(-1)
    lengths = np.asarray(trial_lengths_samples, dtype=int).reshape(-1)
    starts = np.asarray(trial_source_start_samples, dtype=int).reshape(-1)
    if starts.size != lengths.size:
        starts = np.zeros_like(lengths)
    times: Dict[int, List[np.ndarray]] = {}
    labels: Dict[int, List[np.ndarray]] = {}
    unusable: Dict[int, str] = {}
    for t0, (s_id, a_id, n, st) in enumerate(zip(src, ana, lengths, starts)):
        if n <= 0 or a_id <= 0:
            continue
        if int(s_id) not in plan.applied_source_epochs:
            unusable[int(a_id)] = f"source epoch {int(s_id)} not identified"
            continue
        lines = trial_source_lines.get(t0 + 1)
        if lines is None or lines.size < st + n:
            unusable[int(a_id)] = f"trial {t0 + 1} lacks source line indices"
            continue
        tt, lab = plan.lines_to_behavior_time(reg, int(s_id), np.asarray(lines)[st:st + n])
        times.setdefault(int(a_id), []).append(tt)
        labels.setdefault(int(a_id), []).append(lab)
    out_t = {k: np.concatenate(v) for k, v in times.items() if k not in unusable}
    out_l = {k: np.concatenate(v) for k, v in labels.items() if k not in unusable}
    info = {
        "method": plan.method,
        "epochs_with_measured_times": sorted(out_t.keys()),
        "epochs_without_measured_times": {str(k): v for k, v in unusable.items()},
        "behavior_time_origin_harp_s": plan.behavior_time_origin_harp_s,
    }
    return out_t, out_l, info


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def registration_paths(qc_dir: PathLike, dmd: int) -> Tuple[Path, Path]:
    d = Path(qc_dir) / "sync"
    return d / f"cycle_registration_DMD{int(dmd)}.json", d / f"cycle_registration_DMD{int(dmd)}_pulses.npz"


def save_cycle_registration(reg: CycleRegistration, qc_dir: PathLike) -> Path:
    """Write ``qc/sync/cycle_registration_DMD<n>.json`` (+ pulse times ``.npz``)."""
    json_path, npz_path = registration_paths(qc_dir, reg.dmd)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(reg.to_dict(), indent=2, default=_json_default))
    arrays = {f"epoch{e.source_epoch_index}_pulse_times_harp_s": np.asarray(e.pulse_times_harp_s, dtype=float)
              for e in reg.epochs if e.pulse_times_harp_s is not None}
    if arrays:
        np.savez_compressed(npz_path, **arrays)
    return json_path


def load_cycle_registration(qc_dir: PathLike, dmd: int) -> Optional[CycleRegistration]:
    """Load a saved registration; returns None when absent."""
    json_path, npz_path = registration_paths(qc_dir, dmd)
    if not json_path.exists():
        return None
    d = json.loads(json_path.read_text())
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{json_path}: unsupported schema {d.get('schema_version')!r}")
    pulses: Dict[str, np.ndarray] = {}
    if npz_path.exists():
        with np.load(npz_path) as z:
            pulses = {k: z[k] for k in z.files}
    epochs = []
    for e in d.get("epochs", []):
        e = dict(e)
        e["candidate_offset_range"] = tuple(int(x) for x in e.get("candidate_offset_range", (0, 0)))
        e["pulse_times_harp_s"] = pulses.get(f"epoch{e['source_epoch_index']}_pulse_times_harp_s")
        epochs.append(EpochCycleRegistration(**e))
    d["epochs"] = epochs
    return CycleRegistration(**d)


def resolve_cycle_registration(
    asset: Any,
    dmd: int,
    cycle_registration: Any = "auto",
) -> Tuple[Optional[CycleRegistration], Dict[str, Any]]:
    """Resolve the DI3 cycle registration to use for one DMD.

    ``cycle_registration`` may be ``"auto"`` (load
    ``qc/sync/cycle_registration_DMD<n>.json`` if present, else none), ``None``
    (never apply), a :class:`CycleRegistration`, or a mapping
    ``{dmd: CycleRegistration}``. Returns ``(registration_or_None, info)``.
    """
    info: Dict[str, Any] = {"requested": "auto" if isinstance(cycle_registration, str) else type(cycle_registration).__name__}
    reg: Optional[CycleRegistration] = None
    if cycle_registration is None:
        info["resolved"] = None
        return None, info
    if isinstance(cycle_registration, str):
        if cycle_registration.lower() != "auto":
            raise ValueError("cycle_registration must be 'auto', None, a CycleRegistration or a {dmd: CycleRegistration} mapping")
        qc_dir = getattr(asset, "qc_dir", None)
        if qc_dir is not None:
            reg = load_cycle_registration(qc_dir, int(dmd))
            info["resolved"] = None if reg is None else str(registration_paths(qc_dir, int(dmd))[0])
    elif isinstance(cycle_registration, CycleRegistration):
        reg = cycle_registration if int(cycle_registration.dmd) == int(dmd) else None
        info["resolved"] = "provided" if reg is not None else f"provided registration is for DMD {cycle_registration.dmd}"
    else:
        reg = cycle_registration.get(int(dmd))
        info["resolved"] = "provided_mapping" if reg is not None else "no entry for this DMD"
    if reg is not None:
        info.update({
            "status": reg.status,
            "acquisition_start_order": reg.acquisition_start_order,
            "safe_for_physiology": bool(reg.safe_for_physiology),
            "relationship": reg.relationship,
            "period_error_ppm": reg.period_error_ppm,
            "offsets_by_source_epoch": {str(e.source_epoch_index): e.cycle_index_offset for e in reg.epochs},
            "candidate_offset_ranges": {str(e.source_epoch_index): list(e.candidate_offset_range) for e in reg.epochs},
            "limitations": list(reg.limitations),
        })
    return reg, info


def require_identified_registration(
    reg: Optional[CycleRegistration], info: Dict[str, Any], *, dmd: int, session_id: str
) -> None:
    """Raise when physiology extraction demands an identified cycle registration and none exists."""
    if reg is None:
        raise RuntimeError(
            f"{session_id} DMD{dmd}: require_identified_cycle_alignment=True but no cycle registration was found "
            f"({info.get('resolved')}). Run common.cycle_registration.run_cycle_registration first."
        )
    if not reg.identified:
        ranges = {e.source_epoch_index: list(e.candidate_offset_range) for e in reg.epochs}
        raise RuntimeError(
            f"{session_id} DMD{dmd}: cycle registration status is {reg.status!r} "
            f"(start order {reg.acquisition_start_order}, candidate offsets {ranges}); physiology extraction refused. "
            f"Limitations: {reg.limitations}"
        )


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------

def source_epochs_from_summary(summary: Any, dmd: int) -> List[SourceEpochCycles]:
    """Build :class:`SourceEpochCycles` from a summary reader's per-epoch metadata.

    Uses ``get_dmd_epoch_metadata`` rows (``epochIdx``, ``nCycles``,
    ``linesPerCycle``, ``metadata.lineRateHz``) falling back to
    ``get_dmd_metadata`` for the line rate / lines per cycle.
    """
    md = summary.get_dmd_metadata(dmd) if hasattr(summary, "get_dmd_metadata") else {}
    rows = summary.get_dmd_epoch_metadata(dmd) if hasattr(summary, "get_dmd_epoch_metadata") else []

    def _f(x, default=float("nan")):
        try:
            v = float(np.asarray(x).squeeze())
            return v if np.isfinite(v) else default
        except Exception:
            return default

    lr0 = _f(md.get("lineRateHz"))
    lpc0 = _f(md.get("linesPerCycle"))
    out: List[SourceEpochCycles] = []
    for i, row in enumerate(rows, start=1):
        if not bool(row.get("available", True)):
            continue
        meta = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        lr = _f(meta.get("lineRateHz"), lr0)
        lpc = _f(row.get("linesPerCycle"), lpc0)
        n = _f(row.get("nCycles"))
        if not np.isfinite(n):
            tot = _f(row.get("totalNumLines"))
            n = tot / lpc if np.isfinite(tot) and lpc > 0 else float("nan")
        if not (np.isfinite(n) and np.isfinite(lpc) and np.isfinite(lr)):
            continue
        out.append(SourceEpochCycles(
            epoch_index=int(_f(row.get("epochIdx"), i)), n_cycles=int(round(n)),
            lines_per_cycle=int(round(lpc)), line_rate_hz=float(lr), source="summary_metadata",
        ))
    return out


def source_epochs_from_raw_dat(dynamic_data_dir: PathLike, dmd: int, *, line_rate_hz: Optional[float] = None) -> List[SourceEpochCycles]:
    """Build source epochs from raw ``.dat`` headers: one epoch per acquisition prefix (chronological)."""
    from vip_slap2_analysis.io.slap2_dat import (
        list_acquisition_prefixes, read_dmd_acquisition_timing, find_meta_file, read_meta_summary,
    )
    out: List[SourceEpochCycles] = []
    for i, prefix in enumerate(list_acquisition_prefixes(dynamic_data_dir), start=1):
        try:
            timing = read_dmd_acquisition_timing(dynamic_data_dir, dmd, prefix=prefix)
        except FileNotFoundError:
            continue
        lr = line_rate_hz
        if lr is None:
            meta = find_meta_file(dynamic_data_dir, dmd, prefix=prefix)
            if meta is not None:
                lr = read_meta_summary(meta).get("lineRateHz")
        if lr is None:
            # fall back to the FPGA-measured cadence
            lr = float(timing.lines_per_cycle / timing.median_cycle_period_s)
        out.append(SourceEpochCycles(
            epoch_index=i, n_cycles=int(timing.n_cycles), lines_per_cycle=int(timing.lines_per_cycle),
            line_rate_hz=float(lr), cycle_start_s=timing.cycle_start_s, n_source_gaps=int(timing.gap_indices.size),
            source=f"raw_dat:{prefix}",
        ))
    return out


def run_cycle_registration(
    asset: Any,
    *,
    dmd: int,
    summary: Any = None,
    dynamic_data_dir: Optional[PathLike] = None,
    save: bool = True,
    **kwargs: Any,
) -> CycleRegistration:
    """Compute (and optionally save) the DI3 -> source-cycle registration for one DMD.

    DI3 is read from the raw HARP ``Behavior_32.bin`` (recording bounds tightened
    by ``Behavior_44.bin``) when ``asset.harp_dir`` exists, else from the cached
    ``HARP_df.csv``. Source epochs come from raw ``.dat`` headers when
    ``dynamic_data_dir`` is given (authoritative cycle counts and source-side
    gap detection), otherwise from the summary reader's acquisition metadata.
    """
    from vip_slap2_analysis.io.harp_bin import (
        find_register_file, read_digital_input_edges, read_digital_input_edges_from_csv,
    )

    inputs: Dict[str, Any] = {}
    rec = None
    origin: Optional[float] = None
    harp_dir = getattr(asset, "harp_dir", None)
    if harp_dir is not None and Path(harp_dir).exists():
        reg32 = find_register_file(harp_dir, 32)
        reg44 = find_register_file(harp_dir, 44)
        if reg32 is not None:
            rec = read_digital_input_edges(reg32, record_end_from=reg44)
            inputs["di3_source"] = str(reg32)
            inputs["record_end_source"] = str(reg44) if reg44 else None
        if reg44 is not None:
            from vip_slap2_analysis.io.harp_bin import read_harp_register
            t44, _, _ = read_harp_register(reg44)
            finite = t44[np.isfinite(t44)]
            if finite.size:
                origin = float(finite.min())
                inputs["behavior_time_origin_source"] = f"{reg44} (first AnalogData message)"
    if rec is None:
        csv = getattr(asset, "harp_df_csv", None)
        if csv is None or not Path(csv).exists():
            raise FileNotFoundError("No HARP Behavior_32.bin or HARP_df.csv available for DI3 registration")
        rec = read_digital_input_edges_from_csv(csv)
        inputs["di3_source"] = str(csv)
    if origin is None:
        pkl = getattr(asset, "photodiode_pkl", None)
        if pkl is not None and Path(pkl).exists():
            import pandas as pd
            df = pd.read_pickle(pkl)
            origin = float(np.asarray(df.index, dtype=float)[0])
            inputs["behavior_time_origin_source"] = f"{pkl} (first photodiode sample)"
    inputs["behavior_time_origin_harp_s"] = origin

    if dynamic_data_dir is not None and Path(dynamic_data_dir).exists():
        src = source_epochs_from_raw_dat(dynamic_data_dir, dmd)
        inputs["source_epochs_source"] = f"raw_dat:{dynamic_data_dir}"
    elif summary is not None:
        src = source_epochs_from_summary(summary, dmd)
        inputs["source_epochs_source"] = "summary_metadata"
    else:
        raise ValueError("Either dynamic_data_dir or a summary reader is required for source cycle counts")

    reg = register_di3_to_source_cycles(
        rise_times_s=rec.rise_times_s, record_start_s=rec.record_start_s, record_end_s=rec.record_end_s,
        source_epochs=src, dmd=dmd, session_id=getattr(asset, "session_id", None),
        inputs={**inputs, "harp_recording": rec.to_dict()}, **kwargs,
    )
    if save and getattr(asset, "qc_dir", None) is not None:
        save_cycle_registration(reg, asset.qc_dir)
    return reg
