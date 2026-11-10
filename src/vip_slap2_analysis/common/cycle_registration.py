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
RELATIONSHIP_TRANSFERRED = "transferred_from_other_dmd_shared_fpga"


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
    has_partial_trailing_cycle: bool = False
    # absolute FPGA time (s) of the acquisition reference (referenceTimestamp / clock). All acquisitions of a
    # SLAP2 session share the counter, so cycle_start_s + fpga_reference_s is comparable across epochs.
    fpga_reference_s: Optional[float] = None

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
    lines_per_cycle: Optional[int] = None
    line_rate_hz: Optional[float] = None
    # how the HARP time of each cycle was obtained: 'di3_pulses' (measured rises) or
    # 'transferred_from_dmd<k>_shared_fpga' (FPGA timestamps mapped through another DMD's pulses)
    timing_source: str = "di3_pulses"
    # tolerance for the origin-consistency anchor (None -> caller default, 2 ms). Transferred
    # registrations need one cycle period because this DMD's cycle boundaries are not the DI3 edges.
    anchor_tolerance_s: Optional[float] = None
    # discontinuities in the HARP-vs-FPGA relation inside this epoch (one clock stepped); each entry
    # {"after_pulse_index", "harp_s", "step_s"}. Measured where FPGA cycle timestamps are available.
    relation_steps: List[Dict[str, float]] = field(default_factory=list)
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
        if not valid.any():
            raise ValueError("no observed pulse maps inside the source cycle range")
        if not valid.all():
            # pulses beyond the saved cycles (e.g. a partial trailing cycle) are ignored, never silently remapped
            pass
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


RELATION_STEP_THRESHOLD_S = 2e-4


def detect_relation_steps(x_fpga: np.ndarray, y_harp: np.ndarray, *, threshold_s: float = RELATION_STEP_THRESHOLD_S,
                          max_step_fraction: float = 1e-3) -> List[Dict[str, float]]:
    """Discontinuities of HARP-vs-FPGA: jumps of the affine-fit residual between consecutive pulses.

    Seen on 826031_2026-02-02 / 02-11 as a single -2.3/-2.5 ms step in a session;
    a dropped pulse is *not* a step (cycle indices already account for it). A
    train whose residual jumps more often than ``max_step_fraction`` of its
    intervals is jittery, not stepped: no steps are returned so that callers
    judge it on the global fit (804730_2025-07-30, 0.6 ms RMS jitter).
    """
    if x_fpga.size < 3:
        return []
    a, b = np.polyfit(x_fpga, y_harp, 1)
    r = y_harp - (a * x_fpga + b)
    d = np.diff(r)
    idx = np.flatnonzero(np.abs(d) > float(threshold_s))
    if idx.size > max(5, int(max_step_fraction * d.size)):
        return []
    return [{"after_pulse_index": int(i), "harp_s": float(y_harp[i + 1]), "step_s": float(d[i])} for i in idx]


def destep_harp_times(y_harp: np.ndarray, steps: Sequence[Dict[str, float]]) -> np.ndarray:
    """Remove detected steps so that HARP ~ FPGA is affine again (for fitting only; never applied to data)."""
    y = np.asarray(y_harp, dtype=float).copy()
    for st in steps:
        y[int(st["after_pulse_index"]) + 1:] -= float(st["step_s"])
    return y


def cumulative_step_before(t_harp: float, steps: Sequence[Dict[str, float]]) -> float:
    """Sum of relation steps that occurred before HARP time ``t_harp`` (to re-apply to de-stepped predictions)."""
    return float(sum(float(st["step_s"]) for st in steps if float(st["harp_s"]) <= float(t_harp)))


def piecewise_fpga_to_harp(ts_fpga: np.ndarray, x_fpga: np.ndarray, y_harp: np.ndarray, steps: Sequence[Dict[str, float]]) -> np.ndarray:
    """Map FPGA times to HARP by linear interpolation between measured pulses, affine extrapolation outside.

    Interpolation reproduces the measured HARP axis exactly at every pulse, so a
    clock step (``steps``) is carried into the mapped times instead of being
    smeared over the epoch by a single affine fit. Outside the pulse range the
    affine fit of the nearest step-free segment is used.
    """
    ts = np.asarray(ts_fpga, dtype=float)
    out = np.interp(ts, x_fpga, y_harp)
    # extrapolate with the *local* slope at each end (last/first `n_local` pulses of the end segment), so a
    # slowly slewing relation is continued from where it actually is, not from a whole-epoch affine fit
    n_local = 50
    bounds = [0] + [int(st["after_pulse_index"]) + 1 for st in steps] + [int(x_fpga.size)]
    first = (bounds[0], min(bounds[1], bounds[0] + n_local)); last = (max(bounds[-2], bounds[-1] - n_local), bounds[-1])
    for (lo, hi), side in ((first, "left"), (last, "right")):
        xs, ys = x_fpga[lo:hi], y_harp[lo:hi]
        if xs.size >= 3:
            a, b = np.polyfit(xs, ys, 1)
        else:
            a, b = np.polyfit(x_fpga, y_harp, 1)
        m = ts < x_fpga[0] if side == "left" else ts > x_fpga[-1]
        out[m] = a * ts[m] + b
    return out


def local_relation_roughness(x_fpga: np.ndarray, y_harp: np.ndarray) -> Tuple[float, float]:
    """(rms, max) deviation of each pulse from the straight line through its two neighbours, in seconds.

    This is what interpolation between measured pulses actually relies on: local
    linearity over two pulse intervals. A slow drift or ramp of the HARP-vs-FPGA
    relation leaves it small; pulse jitter, dropped-pulse mis-assignment or a
    clock step inside the pair raise it.
    """
    if x_fpga.size < 3:
        return 0.0, 0.0
    x0, x1, x2 = x_fpga[:-2], x_fpga[1:-1], x_fpga[2:]
    y0, y1, y2 = y_harp[:-2], y_harp[1:-1], y_harp[2:]
    with np.errstate(divide="ignore", invalid="ignore"):
        pred = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0)
    d = y1 - pred
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))


def _segment_fit_residuals(x_fpga: np.ndarray, y_harp: np.ndarray, steps: Sequence[Dict[str, float]]) -> Tuple[float, float]:
    """Max over step-free segments of (rms, max |resid|) of each segment's own affine fit."""
    bounds = [0] + [int(st["after_pulse_index"]) + 1 for st in steps] + [int(x_fpga.size)]
    rms_all, max_all = 0.0, 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        xs, ys = x_fpga[lo:hi], y_harp[lo:hi]
        if xs.size < 3:
            continue
        a, b = np.polyfit(xs, ys, 1)
        r = ys - (a * xs + b)
        rms_all = max(rms_all, float(np.sqrt(np.mean(r * r)))); max_all = max(max_all, float(np.max(np.abs(r))))
    return rms_all, max_all


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
    trailing_pulse_assumed = False
    if relationship == RELATIONSHIP_DIRECT and ((tail and src.has_partial_trailing_cycle) or (n_eff == N + 1 and (tail or head))):
        # The acquisition stops inside a cycle: that cycle emits its DI3 pulse but is not written to
        # the .dat (the reader floors the cycle count). Seen on real data as +1 pulse per interior
        # epoch (863774_2026-08-07: 23750 pulses vs 23749 saved cycles). With tail coverage, or one
        # pulse too many under a head anchor, the last observed pulse is the unsaved cycle's pulse
        # and is excluded from the count.
        n_eff -= 1
        if src.has_partial_trailing_cycle:
            notes.append("Last observed pulse attributed to the unsaved partial final cycle (file-size remainder present).")
        else:
            trailing_pulse_assumed = True
            notes.append("One more pulse than saved cycles with tail coverage: last pulse attributed to the cycle during "
                         "which the acquisition stopped (no file-size remainder to confirm; assumed).")
    offset: Optional[int] = None
    rng: Tuple[int, int]
    status = STATUS_AMBIGUOUS
    missing_initial: Optional[int] = None
    missing_terminal: Optional[int] = None

    if int(src.n_source_gaps) > 0 and relationship == RELATIONSHIP_DIRECT:
        # Source-side cycle timing gaps mean cycles without pulses *and* DI3 gaps that are not
        # dropped pulses; count-based registration is not defined for a discontinuous acquisition.
        status = STATUS_INCONSISTENT
        rng = (0, max(0, N - n_eff))
        notes.append(f"Raw .dat headers show {int(src.n_source_gaps)} source-side cycle gap(s); "
                     "count-based registration refused (discontinuous acquisition).")
    elif relationship != RELATIONSHIP_DIRECT:
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
            offset = 0; status = STATUS_IDENTIFIED_ASSUMED if trailing_pulse_assumed else STATUS_IDENTIFIED; rng = (0, 0)
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
            offset = deficit; rng = (deficit, deficit)
            status = STATUS_IDENTIFIED_ASSUMED if trailing_pulse_assumed else STATUS_IDENTIFIED
            missing_initial = deficit; missing_terminal = 0
            notes.append("Tail anchor: HARP recorded past the last pulse; first observed pulse is cycle N - n_observed"
                         + (" (after excluding the unsaved-cycle pulse)." if src.has_partial_trailing_cycle else
                            "; if the acquisition stopped inside an unsaved cycle this is low by exactly one cycle "
                            "(no file-size remainder found)."))
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
        lines_per_cycle=int(src.lines_per_cycle),
        line_rate_hz=float(src.line_rate_hz),
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


ANCHOR_FPGA_CROSS_EPOCH = "fpga_cross_epoch"
TIMING_INTEGER_LINE_FPGA = "di3_integer_line_cadence_via_fpga_lines"
TIMING_INTEGER_LINE_NOMINAL = "di3_integer_line_cadence_via_nominal_lines"
TIMING_INTEGER_LINE_AFFINE = "di3_integer_line_cadence_affine_fit_jittery_pulses"


def _resolve_integer_line_epochs(
    epochs: List[EpochCycleRegistration],
    src_list: Sequence[SourceEpochCycles],
    *,
    lines_per_pulse: Optional[int],
    max_rate_error: float = 2e-4,
    max_segment_rms_s: float = 2e-4,
    jitter_fallback_max_rms_s: float = 2e-3,
) -> List[str]:
    """Register epochs whose DI3 runs at an integer number of lines per pulse (not the DMD cycle).

    2025 glutamate sessions (and 826032_2026-02-02) expose a DI3 train every
    ``L`` lines of the FPGA line clock rather than one pulse per DMD cycle. The
    train is still a *line-locked* clock, so with a head anchor (HARP already
    recording when the train began) pulse ``j`` marks line ``j*L`` of the
    acquisition and the first pulse marks line 0 = cycle 0. That phase (pulse at
    line 0 rather than line L) is an assumption, hence ``identified_with_assumptions``.

    Jittery trains (804730_2025-07-30: 0.6 ms RMS pulse jitter on a 9.6 ms
    cadence, counts otherwise consistent) cannot be interpolated pulse by pulse;
    when the slope is right and the jitter is below ``jitter_fallback_max_rms_s``
    the affine fit over all pulses is used instead (its origin is precise to
    jitter/sqrt(n) if the jitter is zero-mean, biased by the mean delay if not),
    and the note says so.

    With raw ``.dat`` timestamps the FPGA time of every line ``j*L`` is known, so
    the pulse train is fitted against them: the slope must be 1 within
    ``max_rate_error`` (a wrong ``L`` is off by ~1/L) and the residual within
    step-free segments below ``max_segment_rms_s``. Cycle starts are then mapped
    FPGA -> HARP by interpolation between the measured pulses (exact at pulses),
    and stored per cycle like a transferred registration. Without raw
    timestamps the nominal line rate is used for the line times (legacy-grade
    within-interval accuracy, correct origin).
    """
    notes: List[str] = []
    if not lines_per_pulse or lines_per_pulse <= 0:
        notes.append("Integer-line cadence but the lines-per-pulse count could not be determined; epochs left unresolved.")
        return notes
    L = int(lines_per_pulse)
    by_idx = {int(s.epoch_index): s for s in src_list}

    def demote(ep: EpochCycleRegistration) -> None:
        # _register_epoch may have set offset 0 / identified_with_assumptions from the head anchor alone;
        # without a per-cycle timebase that would be misread downstream (pulses are not cycles here).
        ep.status = STATUS_AMBIGUOUS
        ep.cycle_index_offset = None
        ep.candidate_offset_range = (0, int(ep.n_source_cycles))

    for e in epochs:
        src = by_idx.get(int(e.source_epoch_index))
        if src is None or e.pulse_times_harp_s is None or len(e.pulse_times_harp_s) < 3:
            if e.status == STATUS_IDENTIFIED_ASSUMED:
                demote(e)
            continue
        if not e.head_covered:
            e.notes.append("Integer-line cadence without head coverage: the first observed pulse's line index is unknown; left unresolved.")
            demote(e)
            continue
        pulses = np.asarray(e.pulse_times_harp_s, dtype=float)
        lpc = int(src.lines_per_cycle); lr = float(src.line_rate_hz); N = int(src.n_cycles)
        pulse_period = L / lr
        steps = np.rint(np.diff(pulses) / pulse_period).astype(int) if pulses.size > 1 else np.zeros(0, dtype=int)
        steps[steps < 1] = 1
        line_idx = np.r_[0, np.cumsum(steps)] * L                      # first pulse = line 0 (assumption)
        total_lines = N * lpc
        inside = line_idx < total_lines
        if inside.sum() < 3:
            e.notes.append("Integer-line cadence: fewer than 3 pulses fall inside the saved acquisition; left unresolved.")
            demote(e)
            continue
        li = line_idx[inside]; y = pulses[inside]
        c_of = li // lpc; l_of = li % lpc
        if src.cycle_start_s is not None:
            cyc_fpga = np.asarray(src.cycle_start_s, dtype=float)
            x = cyc_fpga[c_of] + l_of / lr
            cycle_fpga = cyc_fpga
            timing_source = TIMING_INTEGER_LINE_FPGA
        else:
            x = li / lr
            cycle_fpga = np.arange(N) * lpc / lr
            timing_source = TIMING_INTEGER_LINE_NOMINAL
        a, b = np.polyfit(x, y, 1)
        if abs(a - 1.0) > max_rate_error:
            e.notes.append(f"Integer-line cadence: HARP/FPGA slope {a:.6f} with L={L} lines per pulse is not 1 within {max_rate_error*1e6:.0f} ppm "
                           f"(L wrong or cadence not line-locked); left unresolved.")
            demote(e)
            continue
        rel_steps = detect_relation_steps(x, y)
        rms, rmax = _segment_fit_residuals(x, y, rel_steps)
        jitter_note = None
        n_fit = int(inside.sum())
        if rms > max_segment_rms_s:
            r_all = y - (a * x + b)
            rms_all = float(np.sqrt(np.mean(r_all ** 2)))
            p5, p50, p95 = (float(np.percentile(r_all, q)) for q in (5, 50, 95))
            counts_ok = abs(int(inside.sum()) * L - total_lines) <= 2 * L
            # Robust path first: 804730_2025-07-30 has most pulses exactly on the grid and a minority delayed
            # by up to ~1.8 ms (p5/p50/p95 = -357/-311/+1506 us). The on-grid majority is identified from the
            # residual's median/MAD, the delayed pulses are dropped, and the rest is treated like a clean train.
            med = float(np.median(r_all)); mad = float(np.median(np.abs(r_all - med))) * 1.4826
            keep = np.abs(r_all - med) <= max(3.0 * mad, 1e-4)
            robust_done = False
            if keep.sum() >= max(3, int(0.5 * r_all.size)):
                xk, yk = x[keep], y[keep]
                ak, bk = np.polyfit(xk, yk, 1)
                steps_k = detect_relation_steps(xk, yk)
                rms_k, rmax_k = _segment_fit_residuals(xk, yk, steps_k)
                if rms_k <= max_segment_rms_s and abs(ak - 1.0) <= max_rate_error:
                    n_drop = int((~keep).sum())
                    x, y, a, b, rel_steps, rms, rmax = xk, yk, ak, bk, steps_k, rms_k, rmax_k
                    n_fit = int(keep.sum())
                    jitter_note = (f"{n_drop} of {r_all.size} DI3 pulses ({100.0*n_drop/r_all.size:.1f}%) are displaced from the FPGA line grid "
                                   f"(residual p5/p50/p95 {p5*1e6:+.0f}/{p50*1e6:+.0f}/{p95*1e6:+.0f} us about the all-pulse fit); they were excluded and the "
                                   f"remaining {n_fit} sit on the grid at {rms*1e6:.1f} us RMS. Cycle times interpolate between the retained pulses only.")
                    robust_done = True
            if robust_done:
                pass
            elif rms_all <= jitter_fallback_max_rms_s and counts_ok and e.tail_covered:
                # jittery but complete train: the affine fit over all pulses is the best available map
                rel_steps = []
                timing_source = TIMING_INTEGER_LINE_AFFINE
                jitter_note = (f"DI3 pulses jitter {rms_all*1e6:.0f} us RMS about the FPGA line grid (p5/p50/p95 {p5*1e6:+.0f}/{p50*1e6:+.0f}/{p95*1e6:+.0f} us), "
                               f"too much to interpolate pulse by pulse and with no on-grid majority; cycle times come from the affine fit over {int(inside.sum())} pulses "
                               f"(origin precision ~{rms_all/np.sqrt(inside.sum())*1e6:.0f} us if the jitter is zero-mean; a one-sided delay would bias it by its mean). "
                               f"Pulse count x L matches the saved lines ({int(inside.sum())} x {L} vs {total_lines}).")
                rms, rmax = rms_all, float(np.max(np.abs(r_all)))
            else:
                e.notes.append(f"Integer-line cadence: pulses do not sit on the FPGA line grid with L={L} (residual {rms*1e6:.0f} us RMS within "
                               f"{len(rel_steps) + 1} step-free segment(s), max {rmax*1e6:.0f} us; slope {(a-1)*1e6:+.1f} ppm; "
                               f"global residual {rms_all*1e6:.0f} us RMS, p5/p95 {p5*1e6:+.0f}/{p95*1e6:+.0f} us; "
                               f"counts {'consistent' if counts_ok else 'inconsistent'}, tail {'covered' if e.tail_covered else 'not covered'}); left unresolved.")
                demote(e)
                continue
        if timing_source == TIMING_INTEGER_LINE_AFFINE:
            mapped = a * np.asarray(cycle_fpga, dtype=float) + b
        else:
            mapped = piecewise_fpga_to_harp(cycle_fpga, x, y, rel_steps)
        n_cov = int(np.sum(mapped <= float(y[-1]) + pulse_period))     # cycles up to one pulse interval past the last pulse
        n_cov = max(1, min(N, n_cov))
        cyc_period = float(np.median(np.diff(mapped))) if mapped.size > 1 else float(lpc / lr)
        e.status = STATUS_IDENTIFIED_ASSUMED
        e.anchor = "head"
        e.cycle_index_offset = 0
        e.candidate_offset_range = (0, 0)
        e.missing_initial_cycles = 0
        e.missing_terminal_cycles = int(N - n_cov)
        e.n_observed_pulses = n_cov
        e.observed_cycle_period_s = cyc_period
        e.nominal_cycle_period_s = float(lpc / lr)
        e.rate_ratio_harp_over_source = float(a)
        e.period_error_ppm = float((a - 1.0) * 1e6)
        e.residual_rms_s = float(rms); e.residual_max_s = float(rmax)
        e.first_pulse_harp_s = float(mapped[0]); e.last_pulse_harp_s = float(mapped[n_cov - 1])
        e.lines_per_cycle = lpc; e.line_rate_hz = lr
        e.timing_source = timing_source
        e.anchor_tolerance_s = cyc_period + 0.002 + (3.0 * rms if timing_source == TIMING_INTEGER_LINE_AFFINE else 0.0)
        e.relation_steps = rel_steps
        e.pulse_times_harp_s = mapped[:n_cov]
        if jitter_note:
            e.notes.append(jitter_note)
        # the first-pass notes ("only coverage anchors are reported", "unverified phase") are superseded here
        e.notes = [n for n in e.notes if "Only coverage anchors are reported" not in n and "first pulse assumed to mark acquisition start" not in n]
        e.notes.append(f"Integer-line DI3 cadence: {L} lines per pulse; pulse j taken as line j*{L} with the first pulse at line 0 "
                       f"(head anchor, {e.lead_s:.3f} s of DI3-low lead; the pulse phase within the first {L} lines is assumed). "
                       f"{n_fit} pulses fitted to the FPGA line grid: slope {a:.9f} ({(a-1)*1e6:+.1f} ppm), residual {rms*1e6:.1f} us RMS. "
                       f"Cycle starts mapped per cycle by interpolation between pulses ({timing_source}); cycles 0..{n_cov-1} covered.")
        notes.append(f"Epoch {e.source_epoch_index}: integer-line cadence ({L} lines/pulse) registered from the head anchor; status identified_with_assumptions.")
    return notes


def _resolve_epochs_via_shared_fpga(
    epochs: List[EpochCycleRegistration],
    src_list: Sequence[SourceEpochCycles],
    *,
    max_fit_rms_s: float = 5e-4,
    integer_tolerance_cycles: float = 0.25,
) -> List[str]:
    """Second pass: place unresolved epochs on the HARP axis through the FPGA counter shared by all acquisitions.

    Pulses of the epochs identified by counting fix the affine map HARP = a*FPGA + b
    (a few ppm from 1). Every other epoch's cycle starts are then predicted in HARP
    time. An epoch is resolved only when its first pulse falls within
    ``integer_tolerance_cycles`` of a predicted cycle start *and* the number of
    predicted cycles inside its pulse train equals the pulses (plus inferred
    dropped ones); cycles predicted after the last pulse are cycles the DI3
    recording did not cover. Anything else is left as it was. With two or more
    identified epochs the single fit must hold across them (``max_fit_rms_s``),
    otherwise the counter is not continuous and nothing is resolved.
    Leave-one-out on 12 multi-acquisition sessions (2026-11-06) reproduced every
    counted offset to within 0.04 cycles; the tolerance here is six times that.
    """
    notes: List[str] = []
    by_idx = {int(s.epoch_index): s for s in src_list}

    def _abs(src: SourceEpochCycles) -> Optional[np.ndarray]:
        if src.cycle_start_s is None or src.fpga_reference_s is None:
            return None
        return float(src.fpga_reference_s) + np.asarray(src.cycle_start_s, dtype=float)

    xs, ys, fit_ids = [], [], []
    all_steps: List[Dict[str, float]] = []
    for e in epochs:
        if e.status != STATUS_IDENTIFIED or e.cycle_index_offset is None or e.pulse_times_harp_s is None or e.anchor == ANCHOR_FPGA_CROSS_EPOCH:
            continue
        src = by_idx.get(int(e.source_epoch_index)); ts = _abs(src) if src is not None else None
        if ts is None or not np.isfinite(e.observed_cycle_period_s):
            continue
        pulses = np.asarray(e.pulse_times_harp_s, dtype=float)
        idx = np.empty(pulses.size, dtype=int); idx[0] = int(e.cycle_index_offset)
        if pulses.size > 1:
            steps = np.rint(np.diff(pulses) / float(e.observed_cycle_period_s)).astype(int); steps[steps < 1] = 1
            idx[1:] = idx[0] + np.cumsum(steps)
        ok = idx < ts.size
        if ok.sum() >= 3:
            x = ts[idx[ok]]; y = pulses[ok]
            steps = detect_relation_steps(x, y)
            if steps and not e.relation_steps:
                e.relation_steps = steps
                e.notes.append(f"{len(steps)} step(s) in the HARP-vs-FPGA relation inside this epoch: "
                               + ", ".join(f"{st['step_s']*1e3:+.2f} ms at HARP {st['harp_s']:.3f} s" for st in steps)
                               + ". Measured pulse times are used as they are; the steps only matter for cross-epoch fits.")
            all_steps.extend(steps)
            xs.append(x); ys.append(destep_harp_times(y, steps)); fit_ids.append(int(e.source_epoch_index))
    targets = [e for e in epochs if e.status in (STATUS_INCONSISTENT, STATUS_AMBIGUOUS) and e.pulse_times_harp_s is not None
               and len(e.pulse_times_harp_s) > 0 and int(e.source_epoch_index) in by_idx and _abs(by_idx[int(e.source_epoch_index)]) is not None]
    if not targets or not fit_ids:
        return notes
    X = np.concatenate(xs); Y = np.concatenate(ys)
    a, b = np.polyfit(X, Y, 1)
    rms = float(np.sqrt(np.mean((Y - (a * X + b)) ** 2)))
    if len(fit_ids) >= 2 and rms > max_fit_rms_s:
        notes.append(f"Shared-FPGA map does not fit identified epochs {fit_ids} ({rms*1e6:.0f} us RMS); unresolved epochs left as they are.")
        return notes
    for e in targets:
        src = by_idx[int(e.source_epoch_index)]
        pulses = np.asarray(e.pulse_times_harp_s, dtype=float)
        t_first, t_last = float(pulses[0]), float(pulses[-1])
        # the fit lives on the de-stepped axis; measured pulses of this epoch carry the steps that preceded them
        pred = a * _abs(src) + b + cumulative_step_before(t_first, all_steps)
        period = float(np.median(np.diff(pred))) if pred.size > 1 else float(e.nominal_cycle_period_s)
        est = (t_first - pred[0]) / period
        k = int(np.rint(est))
        tol = max(2e-4, 0.1 * period)
        before = int(np.sum(pred < t_first - tol)); after = int(np.sum(pred > t_last + tol))
        inside = int(pred.size - before - after)
        n_eff = int(e.n_observed_pulses) + int(e.n_internal_missing_pulses)
        basis = (f"shared-FPGA prediction from epoch(s) {fit_ids} (fit {rms*1e6:.0f} us RMS"
                 + (", continuity verified across epochs" if len(fit_ids) >= 2 else ", single donor epoch: continuity assumed") + ")")
        if inside == 0:
            e.status = STATUS_UNAVAILABLE
            e.cycle_index_offset = None
            e.notes.append(f"The paired DI3 train lies entirely outside this acquisition ({basis}: first pulse at cycle {est:.1f}); "
                           "the acquisition has no DI3 coverage.")
            continue
        if abs(est - k) > integer_tolerance_cycles or k < 0 or k >= pred.size or abs(inside - n_eff) > 1:
            e.notes.append(f"{basis} puts the first pulse at cycle {est:.3f} with {inside} cycles inside the train vs {n_eff} pulses; "
                           "not integer-consistent, left unresolved.")
            continue
        e.cycle_index_offset = k
        e.candidate_offset_range = (k, k)
        e.status = STATUS_IDENTIFIED
        e.anchor = ANCHOR_FPGA_CROSS_EPOCH
        e.missing_initial_cycles = k
        e.missing_terminal_cycles = int(pred.size - k - n_eff)
        e.notes.append(f"Resolved by {basis}: first pulse at cycle {est:+.3f} -> k={k}; {before} cycles before the first pulse, "
                       f"{after} after the last (DI3 not recorded for those).")
        notes.append(f"Epoch {e.source_epoch_index} resolved through the shared FPGA counter (k={k}).")
    return notes


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
    unmatched_di3: List[Dict[str, Any]] = []
    unmatched_source: List[int] = []
    pairs: List[Tuple[SourceEpochCycles, Di3PulseEpoch]] = []
    if not src_list or not di3_epochs:
        status = STATUS_UNAVAILABLE
        limitations.append("Missing source epochs or DI3 pulse trains.")
    else:
        pairs, unmatched_di3, unmatched_source = pair_source_and_di3_epochs(
            src_list, di3_epochs, relationship=relationship,
        )
        if unmatched_di3:
            limitations.append(
                f"{len(unmatched_di3)} DI3 pulse train(s) without a saved acquisition (aborted/preview acquisitions?): "
                + ", ".join(f"{u['n_pulses']} pulses / {u['duration_s']:.1f} s (period {u.get('median_period_s', float('nan'))*1e3:.1f} ms) at +{u['start_s']:.1f} s" for u in unmatched_di3)
            )
        if unmatched_source:
            limitations.append(
                f"Source epoch(s) {unmatched_source} have no matching DI3 pulse train (acquired outside the HARP recording?)."
            )
        if not pairs:
            status = STATUS_INCONSISTENT
            limitations.append(
                f"{len(src_list)} source epoch(s) vs {len(di3_epochs)} DI3 pulse-train epoch(s); no count-consistent pairing found."
            )
    if pairs:
        for s, e in pairs:
            reg = _register_epoch(
                s, e, relationship=relationship, nominal_period_s=s.nominal_cycle_period_s,
                min_anchor_cycles=min_anchor_cycles, min_anchor_s=min_anchor_s,
                internal_gap_factor=internal_gap_factor,
                assume_first_pulse_is_acquisition_start=assume_first_pulse_is_acquisition_start,
            )
            if s.n_source_gaps:
                reg.notes.append(f"Source .dat headers show {s.n_source_gaps} cycle-timing gap(s) inside this epoch.")
            epochs.append(reg)
        for sid in unmatched_source:
            # keep an explicit record so downstream code sees the epoch as unregistered, not missing
            src_u = next(x for x in src_list if int(x.epoch_index) == int(sid))
            epochs.append(EpochCycleRegistration(
                source_epoch_index=int(sid), di3_epoch_index=0, n_source_cycles=int(src_u.n_cycles), n_observed_pulses=0,
                n_internal_missing_pulses=0, internal_gaps=[], head_covered=False, tail_covered=False, lead_s=float("nan"),
                tail_s=float("nan"), anchor="none", status=STATUS_UNAVAILABLE, cycle_index_offset=None,
                candidate_offset_range=(0, int(src_u.n_cycles)), missing_initial_cycles=None, missing_terminal_cycles=None,
                nominal_cycle_period_s=float(src_u.nominal_cycle_period_s), observed_cycle_period_s=float("nan"),
                rate_ratio_harp_over_source=float("nan"), period_error_ppm=float("nan"), residual_rms_s=float("nan"),
                residual_max_s=float("nan"), first_pulse_harp_s=float("nan"), last_pulse_harp_s=float("nan"),
                notes=["No DI3 pulse train matched this acquisition; no HARP coverage."],
                lines_per_cycle=int(src_u.lines_per_cycle), line_rate_hz=float(src_u.line_rate_hz),
            ))
        epochs.sort(key=lambda e: e.source_epoch_index)
        if relationship == RELATIONSHIP_DIRECT:
            fpga_notes = _resolve_epochs_via_shared_fpga(epochs, src_list)
            limitations.extend(fpga_notes)
        elif relationship == RELATIONSHIP_INTEGER_LINE and assume_first_pulse_is_acquisition_start:
            limitations.extend(_resolve_integer_line_epochs(epochs, src_list, lines_per_pulse=int(round(eff)) if np.isfinite(eff) else None))
        statuses = {e.status for e in epochs}
        if statuses <= {STATUS_IDENTIFIED}:
            status = STATUS_IDENTIFIED
        elif statuses <= {STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED}:
            status = STATUS_IDENTIFIED_ASSUMED
        elif STATUS_INCONSISTENT in statuses:
            status = STATUS_INCONSISTENT
        elif statuses <= {STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED, STATUS_UNAVAILABLE}:
            # identified where HARP covered anything; epochs without coverage carry no stimuli anyway
            status = STATUS_IDENTIFIED_ASSUMED if STATUS_IDENTIFIED_ASSUMED in statuses else STATUS_IDENTIFIED
        else:
            status = STATUS_AMBIGUOUS
    distinct_lpc = {int(s.lines_per_cycle) for s in src_list}
    if len(distinct_lpc) > 1:
        limitations.append(f"Source epochs have different linesPerCycle {sorted(distinct_lpc)}; per-epoch values are used.")
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
        inputs={**dict(inputs or {}), "unmatched_di3_trains": unmatched_di3, "unmatched_source_epochs": list(unmatched_source)},
        wall_clock_evidence=dict(wall_clock_evidence or {}),
    )


def pair_source_and_di3_epochs(
    source_epochs: Sequence[SourceEpochCycles],
    di3_epochs: Sequence[Di3PulseEpoch],
    *,
    relationship: str,
    count_tolerance: float = 0.5,
    period_tolerance: float = 0.02,
) -> Tuple[List[Tuple[SourceEpochCycles, Di3PulseEpoch]], List[Dict[str, Any]], List[int]]:
    """Pair saved acquisitions with DI3 pulse trains chronologically, by pulse/cycle count.

    Both lists are chronological. For each source epoch the next DI3 train whose
    pulse count is at most ``N + 1`` (one trailing pulse for the unsaved final
    cycle) and at least ``count_tolerance * N`` is taken; trains skipped on the way
    are DI3 activity without a saved ``.dat`` (aborted or preview acquisitions).
    Source epochs left over have no HARP coverage. For a non-direct cadence the
    comparison uses durations instead of counts. Nothing is matched by position
    alone, so a 7-vs-11 mismatch pairs the 7 real acquisitions and reports 4
    unmatched trains instead of refusing (863774_2026-08-07).

    A train whose median pulse period differs from the acquisition's nominal
    cycle period by more than ``period_tolerance`` is never paired under the
    direct relationship, whatever its count: preview/structural acquisitions
    emit DI3 at a different cadence (75 ms trains of 193-229 pulses seen next
    to 22 ms voltage acquisitions, 852835_2026-08-01 / 852834_2026-07-28).
    """
    pairs: List[Tuple[SourceEpochCycles, Di3PulseEpoch]] = []
    unmatched: List[Dict[str, Any]] = []
    leftover_source: List[int] = []
    j = 0
    trains = list(di3_epochs)
    for src in source_epochs:
        matched = None
        while j < len(trains):
            tr = trains[j]
            if relationship == RELATIONSHIP_DIRECT:
                n = tr.n_pulses
                ok = (n <= src.n_cycles + 1) and (n >= count_tolerance * src.n_cycles)
                if ok and n >= 3:
                    med = float(np.median(np.diff(tr.rise_times_s)))
                    ok = abs(med / float(src.nominal_cycle_period_s) - 1.0) <= float(period_tolerance)
            else:
                dur = tr.duration_s
                nominal = src.nominal_duration_s
                ok = (dur <= nominal * 1.02 + 1.0) and (dur >= count_tolerance * nominal)
            j += 1
            if ok:
                matched = tr
                break
            unmatched.append({"di3_epoch_index": int(tr.epoch_index), "n_pulses": int(tr.n_pulses),
                              "duration_s": float(tr.duration_s), "start_s": float(tr.rise_times_s[0]),
                              "median_period_s": float(np.median(np.diff(tr.rise_times_s))) if tr.n_pulses >= 2 else float("nan")})
        if matched is None:
            leftover_source.append(int(src.epoch_index))
        else:
            pairs.append((src, matched))
    for tr in trains[j:]:
        unmatched.append({"di3_epoch_index": int(tr.epoch_index), "n_pulses": int(tr.n_pulses),
                          "duration_s": float(tr.duration_s), "start_s": float(tr.rise_times_s[0]),
                          "median_period_s": float(np.median(np.diff(tr.rise_times_s))) if tr.n_pulses >= 2 else float("nan")})
    return pairs, unmatched, leftover_source


def transfer_registration_via_fpga(
    reg_src: CycleRegistration,
    *,
    src_cycle_start_s: Dict[int, np.ndarray],
    dst_cycle_start_s: Dict[int, np.ndarray],
    dst_lines_per_cycle: Dict[int, int],
    dst_line_rate_hz: Dict[int, float],
    dst_dmd: int,
    max_fit_residual_s: float = 2e-4,
) -> CycleRegistration:
    """Register a DMD that has no DI3 clock through the FPGA timestamps shared with ``reg_src``.

    Both DMDs' ``.dat`` line headers carry timestamps on the same 200 MHz FPGA
    clock (same ``referenceTimestamp``; verified on the 2026-10-27 reference
    where DMD1 and DMD2 start on the same tick). For each identified source
    epoch the affine map HARP = a * FPGA + b is fitted on the observed DI3 pulses
    of the source DMD against its FPGA cycle starts (residual ~ HARP tick), then
    applied to every cycle start of the destination DMD. The destination
    registration holds synthesized per-cycle HARP times for the cycles inside
    the HARP-covered window; its ``cycle_index_offset`` is the first destination
    cycle starting at or after the first source pulse (earlier cycles have no
    HARP coverage), and its anchor tolerance is one destination cycle period.

    ``*_cycle_start_s`` map source-epoch index -> FPGA cycle start times (s,
    relative to the acquisition reference) for each DMD.
    """
    epochs: List[EpochCycleRegistration] = []
    limitations = list(reg_src.limitations)
    notes_all: List[str] = []
    for e in reg_src.epochs:
        sid = int(e.source_epoch_index)
        ts_src = src_cycle_start_s.get(sid)
        ts_dst = dst_cycle_start_s.get(sid)
        lpc = int(dst_lines_per_cycle.get(sid, 0))
        lr = float(dst_line_rate_hz.get(sid, float("nan")))
        base = dict(
            source_epoch_index=sid, di3_epoch_index=int(e.di3_epoch_index), n_internal_missing_pulses=0, internal_gaps=[],
            head_covered=e.head_covered, tail_covered=e.tail_covered, lead_s=e.lead_s, tail_s=e.tail_s, anchor=e.anchor,
            nominal_cycle_period_s=(lpc / lr if lpc and np.isfinite(lr) and lr > 0 else float("nan")),
            lines_per_cycle=lpc or None, line_rate_hz=lr if np.isfinite(lr) else None,
            timing_source=f"transferred_from_dmd{reg_src.dmd}_shared_fpga",
        )
        if e.cycle_index_offset is None or e.pulse_times_harp_s is None or ts_src is None or ts_dst is None or not lpc:
            epochs.append(EpochCycleRegistration(
                **base, n_source_cycles=int(ts_dst.size) if ts_dst is not None else 0, n_observed_pulses=0,
                status=STATUS_UNAVAILABLE if e.cycle_index_offset is not None else e.status,
                cycle_index_offset=None, candidate_offset_range=(0, int(ts_dst.size) if ts_dst is not None else 0),
                missing_initial_cycles=None, missing_terminal_cycles=None, observed_cycle_period_s=float("nan"),
                rate_ratio_harp_over_source=float("nan"), period_error_ppm=float("nan"), residual_rms_s=float("nan"),
                residual_max_s=float("nan"), first_pulse_harp_s=float("nan"), last_pulse_harp_s=float("nan"),
                notes=[f"Source DMD{reg_src.dmd} epoch is {e.status}; nothing to transfer."] if e.cycle_index_offset is None
                else ["FPGA cycle timestamps for one of the DMDs are unavailable."],
            ))
            continue
        # observed source pulses -> source cycle indices (same rule as cycle_times_harp_s)
        pulses = np.asarray(e.pulse_times_harp_s, dtype=float)
        k0 = int(e.cycle_index_offset)
        idx = np.empty(pulses.size, dtype=int); idx[0] = k0
        if pulses.size > 1:
            steps = np.rint(np.diff(pulses) / float(e.observed_cycle_period_s)).astype(int); steps[steps < 1] = 1
            idx[1:] = k0 + np.cumsum(steps)
        ok = idx < ts_src.size
        x = ts_src[idx[ok]]; y = pulses[ok]
        a, b = np.polyfit(x, y, 1)
        resid = y - (a * x + b)
        rms_global = float(np.sqrt(np.mean(resid ** 2)))
        # a step in the HARP-vs-FPGA relation (one clock jumped) must not be smeared over the epoch:
        # map by interpolation between the measured pulses and judge the fit per step-free segment
        steps = detect_relation_steps(x, y)
        rms, rmax = _segment_fit_residuals(x, y, steps)
        # The map is interpolation between measured pulses, so the gate is *local* smoothness, not global
        # affinity: a HARP clock that slews over seconds (826031_02-02/02-11, 803121_10-29, 826033_02-21,
        # 838410_03-02: 300-530 us RMS about any affine fit, yet smooth) is mapped exactly at every pulse.
        local_rms, local_max = local_relation_roughness(x, y)
        # a step that is not instantaneous shows up as an affine residual far above the local roughness
        nonaffine = rms > max_fit_residual_s and local_rms <= max_fit_residual_s
        dst_t = piecewise_fpga_to_harp(np.asarray(ts_dst, dtype=float), x, y, steps)
        base["relation_steps"] = steps
        t_first = float(pulses[0]); t_last = float(pulses[-1])
        # coverage window with a tolerance for HARP quantization / fit noise (a destination cycle that
        # starts on the same FPGA tick as the first source pulse must count as covered)
        tol = max(2e-4, 5.0 * local_rms)
        inside = np.flatnonzero((dst_t >= t_first - tol) & (dst_t <= t_last + float(e.observed_cycle_period_s)))
        if local_rms > max_fit_residual_s or inside.size == 0:
            epochs.append(EpochCycleRegistration(
                **base, n_source_cycles=int(ts_dst.size), n_observed_pulses=0, status=STATUS_INCONSISTENT,
                cycle_index_offset=None, candidate_offset_range=(0, int(ts_dst.size)), missing_initial_cycles=None,
                missing_terminal_cycles=None, observed_cycle_period_s=float("nan"), rate_ratio_harp_over_source=float(a),
                period_error_ppm=float((a - 1) * 1e6), residual_rms_s=rms, residual_max_s=rmax,
                first_pulse_harp_s=float("nan"), last_pulse_harp_s=float("nan"),
                notes=[f"HARP-vs-FPGA relation is locally rough ({local_rms*1e6:.0f} us RMS, max {local_max*1e6:.0f} us pulse-to-neighbour deviation; "
                       f"affine residual {rms*1e6:.0f} us within {len(steps) + 1} segment(s), global {rms_global*1e6:.0f} us): interpolation between pulses "
                       f"is not reliable, or no destination cycle inside coverage."],
            ))
            continue
        k0d, k1d = int(inside[0]), int(inside[-1])
        dst_period = float(np.median(np.diff(ts_dst))) * a if ts_dst.size > 1 else base["nominal_cycle_period_s"]
        epochs.append(EpochCycleRegistration(
            **base, n_source_cycles=int(ts_dst.size), n_observed_pulses=int(k1d - k0d + 1), status=e.status,
            cycle_index_offset=k0d, candidate_offset_range=(k0d, k0d), missing_initial_cycles=k0d,
            missing_terminal_cycles=int(ts_dst.size - 1 - k1d), observed_cycle_period_s=float(dst_period),
            rate_ratio_harp_over_source=float(a), period_error_ppm=float((a - 1) * 1e6), residual_rms_s=rms,
            residual_max_s=rmax, first_pulse_harp_s=float(dst_t[k0d]), last_pulse_harp_s=float(dst_t[k1d]),
            notes=[f"Transferred from DMD{reg_src.dmd}: FPGA->HARP by interpolation between {int(ok.sum())} measured pulses "
                   f"(local roughness {local_rms*1e6:.1f} us RMS; affine a = {a:.9f}, residual {rms*1e6:.1f} us RMS); destination cycles {k0d}..{k1d} lie inside HARP coverage."]
                  + ([f"The HARP-vs-FPGA relation is not affine over this epoch (affine residual {rms*1e6:.0f} us vs local roughness {local_rms*1e6:.0f} us): "
                      "one clock drifts/slews rather than jumping; interpolation follows it exactly at every pulse."] if nonaffine else [])
                  + ([f"{len(steps)} step(s) in the HARP-vs-FPGA relation inside this epoch: "
                      + ", ".join(f"{st['step_s']*1e3:+.2f} ms at HARP {st['harp_s']:.3f} s" for st in steps)
                      + "; carried into the mapped times, not smeared."] if steps else []),
            anchor_tolerance_s=float(dst_period) + 0.002,
            pulse_times_harp_s=dst_t[k0d:k1d + 1],
        ))
        notes_all.append(f"epoch {sid}: fit residual {rms*1e6:.1f} us")
    statuses = {x.status for x in epochs}
    if statuses <= {STATUS_IDENTIFIED}:
        status = STATUS_IDENTIFIED
    elif statuses <= {STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED, STATUS_UNAVAILABLE} and (statuses & {STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED}):
        status = STATUS_IDENTIFIED_ASSUMED if STATUS_IDENTIFIED_ASSUMED in statuses else STATUS_IDENTIFIED
    elif STATUS_INCONSISTENT in statuses:
        status = STATUS_INCONSISTENT
    else:
        status = STATUS_AMBIGUOUS if STATUS_AMBIGUOUS in statuses else STATUS_UNAVAILABLE
    lpc0 = next((x.lines_per_cycle for x in epochs if x.lines_per_cycle), None)
    lr0 = next((x.line_rate_hz for x in epochs if x.line_rate_hz), None)
    ratios = [x.rate_ratio_harp_over_source for x in epochs if np.isfinite(x.rate_ratio_harp_over_source)]
    ratio = float(np.mean(ratios)) if ratios else None
    return CycleRegistration(
        schema_version=SCHEMA_VERSION, session_id=reg_src.session_id, dmd=int(dst_dmd),
        relationship=RELATIONSHIP_TRANSFERRED, acquisition_start_order=reg_src.acquisition_start_order,
        start_order_evidence=reg_src.start_order_evidence + f" [DI3 is DMD{reg_src.dmd}'s clock; transferred]",
        status=status, safe_for_physiology=status in (STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED),
        lines_per_cycle=lpc0, line_rate_hz=lr0,
        nominal_cycle_period_s=(lpc0 / lr0 if lpc0 and lr0 else None),
        observed_cycle_period_s=(float(np.mean([x.observed_cycle_period_s for x in epochs if np.isfinite(x.observed_cycle_period_s)])) if any(np.isfinite(x.observed_cycle_period_s) for x in epochs) else None),
        rate_ratio_harp_over_source=ratio, period_error_ppm=((ratio - 1) * 1e6 if ratio else None),
        effective_lines_per_pulse=None, harp_record_start_s=reg_src.harp_record_start_s, harp_record_end_s=reg_src.harp_record_end_s,
        harp_n_rising_edges=reg_src.harp_n_rising_edges, n_source_epochs=len(epochs), n_di3_epochs=reg_src.n_di3_epochs,
        epochs=epochs, limitations=limitations + [f"Timing transferred from DMD{reg_src.dmd} through shared FPGA timestamps; " + "; ".join(notes_all)],
        inputs={**reg_src.inputs, "transferred_from_dmd": int(reg_src.dmd)}, wall_clock_evidence=dict(reg_src.wall_clock_evidence),
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
    N = int(cyc_t.size)
    if np.any(cyc < 0) or np.any(cyc > N):
        raise ValueError("source line indices fall outside the registered cycle range")
    if np.any(cyc == N):
        # Legacy half-open trial tables end the last trial at nLinesTot + 1, one line past the
        # acquisition. Extrapolate that single cycle at the observed cadence and label it as tail
        # extrapolation (2) so it stays visible in sample_time_label.
        cyc_t = np.r_[cyc_t, cyc_t[-1] + float(epoch_reg.observed_cycle_period_s)]
        cyc_label = np.r_[cyc_label, np.int8(2)]
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
    # per-trial (1-based trial number) leading samples to skip; preferred over the aggregate
    trial_leading_samples: Dict[int, int] = field(default_factory=dict)
    # trials (1-based) of applied epochs that have no source lines (invalid/placeholder trials):
    # they carry no data and must be dropped (length 0) when the registration is applied, because
    # their nominal placeholder length cannot be placed on the measured timebase.
    trials_without_lines: Dict[int, List[int]] = field(default_factory=dict)

    @staticmethod
    def _epoch_lpc_rate(reg: "CycleRegistration", ereg: EpochCycleRegistration) -> Tuple[int, float]:
        lpc = ereg.lines_per_cycle if ereg.lines_per_cycle else reg.lines_per_cycle
        lr = ereg.line_rate_hz if ereg.line_rate_hz else reg.line_rate_hz
        if not lpc or not lr:
            raise ValueError("registration lacks lines_per_cycle / line_rate_hz")
        return int(lpc), float(lr)

    def lines_to_behavior_time(
        self, reg: "CycleRegistration", source_epoch_index: int, lines_1based: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ereg = reg.epoch_by_source_index(source_epoch_index)
        if ereg is None or ereg.cycle_index_offset is None:
            raise ValueError(f"source epoch {source_epoch_index} is not identified in the registration")
        lpc, lr = self._epoch_lpc_rate(reg, ereg)
        t, lab = source_lines_to_harp_time(lines_1based, ereg, lines_per_cycle=lpc, line_rate_hz=lr)
        return t - float(self.behavior_time_origin_harp_s), lab

    def first_retained_cycle_start_behavior_s(
        self, reg: "CycleRegistration", source_epoch_index: int, first_retained_line_1based: int
    ) -> float:
        """Behavior-relative HARP time of the start of the cycle holding ``first_retained_line_1based``.

        Used as the origin-consistency anchor: it must coincide (within HARP
        quantization) with the behavior epoch start when the first retained
        sample lies in the first HARP-recorded cycle.
        """
        ereg = reg.epoch_by_source_index(source_epoch_index)
        if ereg is None or ereg.cycle_index_offset is None:
            raise ValueError(f"source epoch {source_epoch_index} is not identified in the registration")
        lpc, _lr = self._epoch_lpc_rate(reg, ereg)
        cyc = (int(first_retained_line_1based) - 1) // lpc
        times, _labels = ereg.cycle_times_harp_s()
        return float(times[cyc] - float(self.behavior_time_origin_harp_s))


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
    labels = np.asarray(trial_source_epoch, dtype=int).reshape(-1)
    leading: Dict[int, int] = {}
    per_trial: Dict[int, int] = {}
    without_lines: Dict[int, List[int]] = {}
    applied: List[int] = []
    skipped: Dict[int, str] = {}
    notes: List[str] = []
    for ereg in reg.epochs:
        sid = int(ereg.source_epoch_index)
        if ereg.cycle_index_offset is None:
            skipped[sid] = f"status={ereg.status}; offset range {list(ereg.candidate_offset_range)}"
            continue
        if ereg.pulse_times_harp_s is None:
            skipped[sid] = "registration record has no pulse times (missing _pulses.npz); re-run registration"
            continue
        lpc = int(ereg.lines_per_cycle or reg.lines_per_cycle or 0)
        if lpc <= 0:
            skipped[sid] = "registration lacks lines_per_cycle"
            continue
        k0 = int(ereg.cycle_index_offset)
        trials = [t for t in range(1, labels.size + 1) if labels[t - 1] == sid]
        missing_lines = [t for t in trials if trial_source_lines.get(t) is None]
        with_lines = [t for t in trials if t not in missing_lines]
        if not with_lines:
            skipped[sid] = "no trial of this epoch has source line indices"
            continue
        if missing_lines:
            # Trials without source lines carry no data (invalid/placeholder trials). Every retained
            # sample is timed by its own source line, so these placeholders are dropped rather than
            # placed by a nominal length that cannot be located on the registered cycles.
            without_lines[sid] = missing_lines
            notes.append(f"Source epoch {sid}: trials {missing_lines} have no source lines and are dropped (no data).")
        n_lead = 0
        prev_last_line: Optional[int] = None
        n_dup_total = 0
        for t in with_lines:
            lines_t = np.asarray(trial_source_lines[t], dtype=np.int64)
            if lines_t.size > 1 and np.any(np.diff(lines_t) <= 0):
                skipped[sid] = f"trial {t}: source line indices are not strictly increasing within the trial"
                break
            cyc = (lines_t - 1) // lpc
            n_t = int(np.sum(cyc < k0))
            # Legacy (half-open) trial tables repeat the previous trial's last line as the first line of
            # the next trial. Those samples are duplicates of data already placed; skip them so every
            # source line appears once (this is the "duplicated boundary sample" defect, fixed at the
            # source in SLAP2-Voltage-Extraction 2026-09-03 but present in older products).
            if prev_last_line is not None:
                n_dup = int(np.sum(lines_t <= prev_last_line))
                if n_dup:
                    n_t = max(n_t, n_dup)
                    n_dup_total += n_dup
            per_trial[t] = n_t
            n_lead += n_t
            prev_last_line = int(lines_t[-1])
        else:
            if n_dup_total:
                notes.append(f"Source epoch {sid}: {n_dup_total} duplicated trial-boundary sample(s) skipped "
                             "(legacy half-open trial line ranges).")
            leading[sid] = n_lead
            applied.append(sid)
            if k0 > 0:
                notes.append(f"Source epoch {sid}: skipping {n_lead} leading samples (first {k0} cycles have no HARP coverage).")
            continue
        # for/else: loop broke on a bad trial -> epoch skipped (already recorded)
        continue
    return RegisteredSamplePlan(
        source_epoch_leading_samples=leading,
        applied_source_epochs=applied,
        skipped_source_epochs=skipped,
        behavior_time_origin_harp_s=float(behavior_time_origin_harp_s),
        method="di3_pulse_registered",
        notes=notes,
        trial_leading_samples=per_trial,
        trials_without_lines=without_lines,
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
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[str, Any], Dict[int, float]]:
    """Measured behavior-time of every retained sample, grouped by analysis epoch.

    Returns ``(epoch_sample_times, epoch_sample_labels, info, epoch_anchor_sec)``;
    ``epoch_anchor_sec`` is the behavior-relative start time of the cycle holding
    the first retained sample of each epoch (the origin-consistency anchor), all suitable for
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
    anchors: Dict[int, float] = {}
    tolerances: Dict[int, float] = {}
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
        try:
            tt, lab = plan.lines_to_behavior_time(reg, int(s_id), np.asarray(lines)[st:st + n])
            if int(a_id) not in anchors:
                anchors[int(a_id)] = plan.first_retained_cycle_start_behavior_s(reg, int(s_id), int(np.asarray(lines)[st]))
                ereg = reg.epoch_by_source_index(int(s_id))
                if ereg is not None and ereg.anchor_tolerance_s:
                    tolerances[int(a_id)] = float(ereg.anchor_tolerance_s)
        except ValueError as exc:
            unusable[int(a_id)] = f"trial {t0 + 1}: {exc}"
            continue
        times.setdefault(int(a_id), []).append(tt)
        labels.setdefault(int(a_id), []).append(lab)
    out_t = {k: np.concatenate(v) for k, v in times.items() if k not in unusable}
    out_l = {k: np.concatenate(v) for k, v in labels.items() if k not in unusable}
    out_a = {k: v for k, v in anchors.items() if k in out_t}
    info = {
        "method": plan.method,
        "epochs_with_measured_times": sorted(out_t.keys()),
        "epochs_without_measured_times": {str(k): v for k, v in unusable.items()},
        "behavior_time_origin_harp_s": plan.behavior_time_origin_harp_s,
        "epoch_anchor_cycle_start_sec": {str(k): float(v) for k, v in out_a.items()},
        "epoch_anchor_tolerance_sec": {str(k): float(v) for k, v in tolerances.items() if k in out_t},
        "timing_source_by_source_epoch": {str(e.source_epoch_index): e.timing_source for e in reg.epochs},
    }
    return out_t, out_l, info, out_a


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
    json_path.write_text(json.dumps(reg.to_dict(), indent=2, default=_json_default), encoding="utf-8")
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
    d = json.loads(json_path.read_text(encoding="utf-8"))
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
        mapping = {int(k): v for k, v in dict(cycle_registration).items()}
        reg = mapping.get(int(dmd))
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
# Wall-clock evidence (bounds only; never selects an offset)
# ---------------------------------------------------------------------------

def wall_clock_evidence_for_asset(asset: Any, dynamic_data_dir: Optional[PathLike]) -> Dict[str, Any]:
    """Collect PC wall-clock stamps of the SLAP2 acquisition start and the HARP recording start.

    SLAP2: ``.meta`` ``fpgaTimeReference`` (SLAP2 PC clock at FPGA tick 0) plus
    ``referenceTimestamp / fpgaSystemClock_Hz`` from the first ``.dat`` header.
    HARP: the Bonsai launcher log (``behavior/launcher_metadata/launcher.log``,
    'Selected harp folder ... <YYYYMMDDTHHMMSS>') on the behavior PC. The two PCs
    are not synchronised to better than seconds, so the difference is reported
    with an explicit +/- ``assumed_pc_clock_offset_s`` and only *bounds* the
    candidate offset range; it never picks the offset.
    """
    import re
    from datetime import datetime

    out: Dict[str, Any] = {"available": False}
    try:
        if dynamic_data_dir is None or not Path(dynamic_data_dir).exists():
            return out
        from vip_slap2_analysis.io.slap2_dat import (
            find_meta_file, list_acquisition_prefixes, read_dat_file_header, find_dmd_dat_files, read_meta_summary,
        )
        prefixes = list_acquisition_prefixes(dynamic_data_dir)
        if not prefixes:
            return out
        starts = []
        for prefix in prefixes:
            meta = find_meta_file(dynamic_data_dir, 1, prefix=prefix)
            files = find_dmd_dat_files(dynamic_data_dir, 1, prefix=prefix)
            if meta is None or not files:
                continue
            m = read_meta_summary(meta)
            h = read_dat_file_header(files[0])
            if m.get("fpgaTimeReference_epoch_ms") is None:
                continue
            # fpgaTimeReference is a naive local-time datetime; keep it as local seconds since epoch
            t0 = float(m["fpgaTimeReference_epoch_ms"]) / 1000.0 + h.referenceTimestamp / float(h.fpgaSystemClock_Hz)
            starts.append({"prefix": prefix, "slap2_acq_start_local_epoch_s": t0,
                           "slap2_acq_start_iso_local": datetime.utcfromtimestamp(t0).isoformat()})
        out["slap2_acquisitions"] = starts
        harp_dir = getattr(asset, "harp_dir", None)
        log = None
        if harp_dir is not None:
            cand = Path(harp_dir).parent / "launcher_metadata" / "launcher.log"
            if cand.exists():
                log = cand
        if log is not None:
            txt = log.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Selected harp folder:.*?(\d{8}T\d{6})", txt)
            if m:
                dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
                out["harp_folder_stamp_iso_local"] = dt.isoformat()
                out["harp_folder_local_epoch_s"] = (dt - datetime(1970, 1, 1)).total_seconds()
            m2 = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Starting Bonsai workflow attempt", txt, re.M)
            if m2:
                out["bonsai_workflow_start_iso_local"] = m2.group(1)
        out["assumed_pc_clock_offset_s"] = 5.0
        out["available"] = bool(starts and out.get("harp_folder_local_epoch_s") is not None)
    except Exception as exc:  # evidence only; never fail registration on it
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _apply_wall_clock_bound(reg: CycleRegistration) -> None:
    """Narrow ambiguous candidate ranges with wall-clock evidence (status unchanged)."""
    w = reg.wall_clock_evidence or {}
    if not w.get("available"):
        return
    acqs = w.get("slap2_acquisitions") or []
    harp_t = w.get("harp_folder_local_epoch_s")
    tol = float(w.get("assumed_pc_clock_offset_s", 5.0))
    for e in reg.epochs:
        if e.pulse_times_harp_s is None or not acqs or not np.isfinite(e.observed_cycle_period_s):
            continue
        i = int(e.source_epoch_index) - 1
        if i >= len(acqs):
            continue
        period = float(e.observed_cycle_period_s)
        # HARP folder stamp ~ HARP recording start; this epoch's first pulse follows it by
        # (first_pulse - record_start), for every epoch (lead_s is only that for the first train)
        if not np.isfinite(e.first_pulse_harp_s) or reg.harp_record_start_s is None:
            continue
        est_k = (harp_t + (float(e.first_pulse_harp_s) - float(reg.harp_record_start_s)) - float(acqs[i]["slap2_acq_start_local_epoch_s"])) / period
        if e.cycle_index_offset is not None:
            # calibration: how far the two PCs' wall clocks are from the registered truth (seconds, HARP PC minus SLAP2 PC)
            reg.wall_clock_evidence.setdefault("pc_clock_discrepancy_s", {})[str(e.source_epoch_index)] = float((est_k - e.cycle_index_offset) * period)
            continue
        if e.status != STATUS_AMBIGUOUS:
            continue
        lo, hi = e.candidate_offset_range
        lo2 = max(lo, int(np.floor(est_k - tol / period)))
        hi2 = min(hi, int(np.ceil(est_k + tol / period)))
        e.notes.append(
            f"Wall-clock evidence (SLAP2 PC .meta vs behavior PC HARP folder stamp, +/-{tol:.0f} s assumed clock offset) "
            f"suggests offset ~{est_k:.0f} cycles, i.e. [{lo2}, {hi2}] of the count-consistent range [{lo}, {hi}]. "
            "Evidence only; status remains ambiguous."
        )
        reg.wall_clock_evidence.setdefault("epoch_offset_estimates", {})[str(e.source_epoch_index)] = {
            "estimated_offset_cycles": float(est_k), "bounded_range": [int(lo2), int(hi2)]}


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


def default_timing_cache_dir(asset: Any) -> Optional[Path]:
    """``<qc_dir>/sync/dat_timing_cache`` when the asset has a qc_dir, else None."""
    qc = getattr(asset, "qc_dir", None)
    return Path(qc) / "sync" / "dat_timing_cache" if qc is not None else None


def source_epochs_from_raw_dat(dynamic_data_dir: PathLike, dmd: int, *, line_rate_hz: Optional[float] = None,
                               cache_dir: Optional[PathLike] = None) -> List[SourceEpochCycles]:
    """Build source epochs from raw ``.dat`` headers: one epoch per acquisition prefix (chronological)."""
    from vip_slap2_analysis.io.slap2_dat import (
        list_acquisition_prefixes, read_dmd_acquisition_timing, find_meta_file, read_meta_summary,
    )
    out: List[SourceEpochCycles] = []
    for i, prefix in enumerate(list_acquisition_prefixes(dynamic_data_dir), start=1):
        try:
            timing = read_dmd_acquisition_timing(dynamic_data_dir, dmd, prefix=prefix, cache_dir=cache_dir)
        except FileNotFoundError:
            continue
        except ValueError as exc:
            raise ValueError(f"{dynamic_data_dir} prefix {prefix!r} DMD{dmd}: {exc}") from exc
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
            source=f"raw_dat:{prefix}", has_partial_trailing_cycle=bool(getattr(timing, "has_partial_trailing_cycle", False)),
            fpga_reference_s=float(timing.reference_timestamp) / float(timing.fpga_clock_hz),
        ))
    return out


def run_cycle_registration(
    asset: Any,
    *,
    dmd: int,
    summary: Any = None,
    dynamic_data_dir: Optional[PathLike] = None,
    save: bool = True,
    transfer_from: Optional[CycleRegistration] = None,
    **kwargs: Any,
) -> CycleRegistration:
    """Compute (and optionally save) the DI3 -> source-cycle registration for one DMD.

    ``transfer_from`` (an identified registration of another DMD of the same
    session) switches to :func:`transfer_registration_via_fpga`, which needs the
    raw ``.dat`` headers of both DMDs (``dynamic_data_dir``). It is used for the
    DMD that does not drive DI3 (normally DMD2).

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
    record_start = float(rec.record_start_s)
    if origin is not None and origin < record_start:
        # The analog stream started earlier than the first digital-input message: the recording
        # (and therefore any DI3-low lead) began at the analog start.
        record_start = float(origin)
        inputs["record_start_source"] = "analog_stream_start"

    extra_limitations: List[str] = []
    if dynamic_data_dir is not None and Path(dynamic_data_dir).exists():
        cache_dir = default_timing_cache_dir(asset)
        src = source_epochs_from_raw_dat(dynamic_data_dir, dmd, cache_dir=cache_dir)
        inputs["source_epochs_source"] = f"raw_dat:{dynamic_data_dir}"
        if summary is not None:
            # Cross-check the summary's epoch structure against the raw acquisitions (R4).
            try:
                src_summary = source_epochs_from_summary(summary, dmd)
            except Exception:
                src_summary = []
            if src_summary:
                inputs["summary_epochs"] = [{"epochIdx": e.epoch_index, "nCycles": e.n_cycles, "linesPerCycle": e.lines_per_cycle} for e in src_summary]
                if len(src_summary) != len(src):
                    extra_limitations.append(
                        f"Summary reports {len(src_summary)} source epoch(s) but raw dynamic_data holds {len(src)} acquisition(s); "
                        "epoch pairing between summary and raw files is not established."
                    )
                else:
                    for a, b in zip(src, src_summary):
                        if abs(int(a.n_cycles) - int(b.n_cycles)) > 1:
                            extra_limitations.append(
                                f"Epoch {a.epoch_index}: raw .dat has {a.n_cycles} cycles but the summary reports {b.n_cycles}."
                            )
    elif summary is not None:
        src = source_epochs_from_summary(summary, dmd)
        inputs["source_epochs_source"] = "summary_metadata"
    else:
        raise ValueError("Either dynamic_data_dir or a summary reader is required for source cycle counts")

    wall = wall_clock_evidence_for_asset(asset, dynamic_data_dir)
    if transfer_from is not None:
        if dynamic_data_dir is None or not Path(dynamic_data_dir).exists():
            raise ValueError("transfer_from requires dynamic_data_dir (raw .dat headers of both DMDs)")
        src_ts = {e.epoch_index: e.cycle_start_s for e in source_epochs_from_raw_dat(dynamic_data_dir, int(transfer_from.dmd), cache_dir=default_timing_cache_dir(asset))}
        reg = transfer_registration_via_fpga(
            transfer_from, src_cycle_start_s=src_ts, dst_cycle_start_s={e.epoch_index: e.cycle_start_s for e in src},
            dst_lines_per_cycle={e.epoch_index: e.lines_per_cycle for e in src},
            dst_line_rate_hz={e.epoch_index: e.line_rate_hz for e in src}, dst_dmd=dmd,
        )
        reg.inputs.update(inputs)
        if save and getattr(asset, "qc_dir", None) is not None:
            save_cycle_registration(reg, asset.qc_dir)
        return reg
    reg = register_di3_to_source_cycles(
        rise_times_s=rec.rise_times_s, record_start_s=record_start, record_end_s=rec.record_end_s,
        source_epochs=src, dmd=dmd, session_id=getattr(asset, "session_id", None),
        inputs={**inputs, "harp_recording": rec.to_dict()}, wall_clock_evidence=wall, **kwargs,
    )
    _apply_wall_clock_bound(reg)
    if extra_limitations:
        reg.limitations.extend(extra_limitations)
        reg.status = STATUS_INCONSISTENT
        reg.safe_for_physiology = False
    if save and getattr(asset, "qc_dir", None) is not None:
        save_cycle_registration(reg, asset.qc_dir)
    return reg
