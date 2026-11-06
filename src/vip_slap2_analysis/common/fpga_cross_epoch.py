"""Cross-acquisition FPGA consistency check for DI3 cycle registrations.

Every ``.dat`` line header carries a timestamp on the SLAP2 FPGA system clock
(200 MHz), and each acquisition's file header stores ``referenceTimestamp``, the
tick at which that acquisition started. When the counter runs continuously for
the whole SLAP2 session, the *absolute* FPGA time of every cycle of every
acquisition is known (``referenceTimestamp / clock + cycle_start_s``), and the
acquisitions of one session share a single affine relation to HARP time,
``HARP = a * FPGA + b`` (``a - 1`` is the HARP/FPGA rate difference, a few ppm).

That gives timing evidence independent of DI3 pulse *counting*:

* Leave-one-epoch-out validation. For an epoch whose offset ``k`` was found by
  counting (tail anchor, see :mod:`cycle_registration`), fit the affine map on
  the pulses of the *other* identified epochs only, predict the HARP time of
  that epoch's cycle ``k``, and compare it with its first observed pulse. The
  two agree to within a fraction of a cycle only if the count was right (an
  error of one cycle shows up as one full period, ~9-22 ms against a fit
  residual of tens of microseconds).
* Diagnosis of ``inconsistent``/``ambiguous`` epochs in the same session: the
  fit predicts where each of that acquisition's cycles falls in HARP time, so
  the cycles without pulses are located at the head, the tail, or inside the
  pulse train.

Nothing here changes a registration. The result is a report; the caller
decides. The check is only meaningful when the counter is continuous across
acquisitions, which the fit itself establishes (a reset counter gives
residuals of seconds, not microseconds) and which is reported explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from vip_slap2_analysis.common.cycle_registration import (
    ANCHOR_FPGA_CROSS_EPOCH, CycleRegistration, EpochCycleRegistration, RELATIONSHIP_DIRECT, STATUS_IDENTIFIED,
    STATUS_IDENTIFIED_ASSUMED,
)

PathLike = Union[str, Path]
SCHEMA_VERSION = "vip_slap2_fpga_cross_epoch_v1"


@dataclass
class EpochFpgaCheck:
    source_epoch_index: int
    registration_status: str
    counted_offset: Optional[int]
    n_source_cycles: int
    n_observed_pulses: int
    used_in_fit: bool
    # leave-one-out (identified epochs) or full-fit (other epochs) prediction
    fit_epochs: List[int]
    predicted_first_pulse_cycle_s: Optional[float]      # predicted HARP time of cycle `counted_offset` (identified) ...
    first_pulse_harp_s: Optional[float]
    offset_residual_cycles: Optional[float]             # (first pulse - predicted cycle-k time) / period, identified epochs
    agrees_within_half_cycle: Optional[bool]
    fpga_offset_estimate_cycles: Optional[float]        # (first pulse - predicted cycle-0 time) / period, any epoch
    fpga_offset_estimate_rounded: Optional[int]
    cycles_predicted_before_first_pulse: Optional[int]
    cycles_predicted_after_last_pulse: Optional[int]
    cycles_predicted_inside_train: Optional[int]
    notes: List[str] = field(default_factory=list)


@dataclass
class FpgaCrossEpochCheck:
    schema_version: str
    session_id: Optional[str]
    dmd: int
    n_source_epochs: int
    n_identified_epochs: int
    fpga_counter_continuous: Optional[bool]     # None = cannot tell (fewer than 2 identified epochs)
    global_fit_n_pulses: int
    global_rate_ratio_harp_over_fpga: Optional[float]
    global_residual_rms_s: Optional[float]
    global_residual_max_s: Optional[float]
    all_identified_offsets_confirmed: Optional[bool]
    epochs: List[EpochFpgaCheck]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> List[str]:
        out = [f"{self.session_id} DMD{self.dmd}: FPGA counter continuous across acquisitions = {self.fpga_counter_continuous}; "
               f"global fit on {self.global_fit_n_pulses} pulses, residual "
               f"{(self.global_residual_rms_s or float('nan'))*1e6:.1f} us RMS / {(self.global_residual_max_s or float('nan'))*1e6:.1f} us max; "
               f"all counted offsets confirmed = {self.all_identified_offsets_confirmed}"]
        for e in self.epochs:
            if e.offset_residual_cycles is not None:
                out.append(f"  epoch {e.source_epoch_index} ({e.registration_status}, k={e.counted_offset}): leave-one-out from epochs "
                           f"{e.fit_epochs} puts the first pulse {e.offset_residual_cycles:+.3f} cycles from cycle k -> "
                           f"{'CONFIRMED' if e.agrees_within_half_cycle else 'DISAGREES'}")
            elif e.fpga_offset_estimate_cycles is not None:
                out.append(f"  epoch {e.source_epoch_index} ({e.registration_status}): FPGA-predicted first-pulse cycle = {e.fpga_offset_estimate_cycles:.3f} "
                           f"(cycles before first pulse {e.cycles_predicted_before_first_pulse}, inside train {e.cycles_predicted_inside_train} vs {e.n_observed_pulses} pulses, "
                           f"after last pulse {e.cycles_predicted_after_last_pulse})")
            for n in e.notes:
                out.append(f"    {n}")
        for l in self.limitations:
            out.append(f"  ! {l}")
        return out


def _pulse_cycle_indices(e: EpochCycleRegistration) -> Tuple[np.ndarray, np.ndarray]:
    """Observed pulses of an identified epoch -> (source cycle indices, pulse times); same rule as the registration."""
    pulses = np.asarray(e.pulse_times_harp_s, dtype=float)
    k0 = int(e.cycle_index_offset)
    idx = np.empty(pulses.size, dtype=int)
    idx[0] = k0
    if pulses.size > 1:
        steps = np.rint(np.diff(pulses) / float(e.observed_cycle_period_s)).astype(int)
        steps[steps < 1] = 1
        idx[1:] = k0 + np.cumsum(steps)
    return idx, pulses


def _fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    a, b = np.polyfit(x, y, 1)
    r = y - (a * x + b)
    return float(a), float(b), float(np.sqrt(np.mean(r * r))), float(np.max(np.abs(r)))


def cross_epoch_fpga_check(
    reg: CycleRegistration,
    absolute_cycle_start_s: Dict[int, np.ndarray],
    *,
    continuity_max_rms_s: float = 5e-4,
    agreement_tolerance_cycles: float = 0.5,
) -> FpgaCrossEpochCheck:
    """Check a DI3 registration against the FPGA timestamps shared by all acquisitions.

    ``absolute_cycle_start_s`` maps source epoch index -> cycle start times in
    *absolute* FPGA seconds (``referenceTimestamp / clock + cycle_start_s``),
    see :func:`absolute_fpga_cycle_starts`.
    """
    limitations: List[str] = []
    if reg.relationship != RELATIONSHIP_DIRECT:
        limitations.append(f"Registration relationship is '{reg.relationship}', not the direct DMD cycle clock; the check needs DI3 pulses of this DMD.")
    # epochs that were themselves resolved through the FPGA map must not feed the fit (circular)
    ident = [e for e in reg.epochs
             if e.status in (STATUS_IDENTIFIED, STATUS_IDENTIFIED_ASSUMED) and e.cycle_index_offset is not None
             and e.anchor != ANCHOR_FPGA_CROSS_EPOCH
             and e.pulse_times_harp_s is not None and int(e.source_epoch_index) in absolute_cycle_start_s
             and np.isfinite(e.observed_cycle_period_s)]
    per_epoch_xy: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for e in ident:
        idx, pulses = _pulse_cycle_indices(e)
        ts = absolute_cycle_start_s[int(e.source_epoch_index)]
        ok = idx < ts.size
        if ok.sum() < 3:
            continue
        per_epoch_xy[int(e.source_epoch_index)] = (ts[idx[ok]], pulses[ok])
    fit_ids = sorted(per_epoch_xy)

    glob_a = glob_b = glob_rms = glob_max = None
    n_fit = 0
    continuous: Optional[bool] = None
    if fit_ids:
        X = np.concatenate([per_epoch_xy[i][0] for i in fit_ids]); Y = np.concatenate([per_epoch_xy[i][1] for i in fit_ids])
        n_fit = int(X.size)
        glob_a, glob_b, glob_rms, glob_max = _fit(X, Y)
        if len(fit_ids) >= 2:
            continuous = bool(glob_rms <= continuity_max_rms_s)
            if not continuous:
                limitations.append(
                    f"A single affine HARP~FPGA map does not fit the {len(fit_ids)} identified epochs ({glob_rms*1e6:.0f} us RMS): "
                    "the FPGA counter is not continuous across these acquisitions (or a counted offset is wrong); "
                    "cross-epoch predictions below are not trustworthy.")
        else:
            limitations.append("Only one identified epoch: the fit cannot be validated across acquisitions; predictions for "
                               "other epochs assume a continuous FPGA counter (unverified here).")
    else:
        limitations.append("No identified epoch with pulses and FPGA cycle times; nothing to fit.")

    epochs_out: List[EpochFpgaCheck] = []
    confirmed_flags: List[bool] = []
    for e in reg.epochs:
        sid = int(e.source_epoch_index)
        ts = absolute_cycle_start_s.get(sid)
        row = EpochFpgaCheck(
            source_epoch_index=sid, registration_status=e.status, counted_offset=e.cycle_index_offset,
            n_source_cycles=int(e.n_source_cycles), n_observed_pulses=int(e.n_observed_pulses), used_in_fit=sid in per_epoch_xy,
            fit_epochs=[], predicted_first_pulse_cycle_s=None, first_pulse_harp_s=None, offset_residual_cycles=None,
            agrees_within_half_cycle=None, fpga_offset_estimate_cycles=None, fpga_offset_estimate_rounded=None,
            cycles_predicted_before_first_pulse=None, cycles_predicted_after_last_pulse=None, cycles_predicted_inside_train=None,
        )
        if ts is None or ts.size == 0:
            row.notes.append("No raw .dat cycle timestamps for this epoch.")
            epochs_out.append(row); continue
        if e.pulse_times_harp_s is None or len(e.pulse_times_harp_s) == 0:
            row.notes.append("No pulses recorded for this epoch.")
            epochs_out.append(row); continue
        if e.anchor == ANCHOR_FPGA_CROSS_EPOCH:
            row.notes.append("Offset was itself obtained from the shared-FPGA map; excluded from the fit (not an independent confirmation).")
        if sid in per_epoch_xy:
            # identified epoch: predict it from the *other* identified epochs only (leave-one-out)
            others = [i for i in fit_ids if i != sid]
            if not others:
                row.notes.append("Only identified epoch in this session; its offset cannot be cross-validated from other acquisitions.")
                epochs_out.append(row); continue
            Xo = np.concatenate([per_epoch_xy[i][0] for i in others]); Yo = np.concatenate([per_epoch_xy[i][1] for i in others])
            a, b, _, _ = _fit(Xo, Yo)
            row.fit_epochs = others
        else:
            if not fit_ids:
                row.notes.append("No identified epoch to predict from.")
                epochs_out.append(row); continue
            a, b = glob_a, glob_b
            row.fit_epochs = list(fit_ids)
        pred = a * np.asarray(ts, dtype=float) + b
        pulses = np.asarray(e.pulse_times_harp_s, dtype=float)
        period = float(np.median(np.diff(pred))) if pred.size > 1 else float(e.nominal_cycle_period_s)
        t_first, t_last = float(pulses[0]), float(pulses[-1])
        row.first_pulse_harp_s = t_first
        # where does the first pulse fall on the predicted cycle grid?
        est = (t_first - pred[0]) / period
        row.fpga_offset_estimate_cycles = float(est)
        row.fpga_offset_estimate_rounded = int(np.rint(est))
        tol = max(2e-4, 0.1 * period)
        row.cycles_predicted_before_first_pulse = int(np.sum(pred < t_first - tol))
        row.cycles_predicted_after_last_pulse = int(np.sum(pred > t_last + tol))
        row.cycles_predicted_inside_train = int(pred.size - row.cycles_predicted_before_first_pulse - row.cycles_predicted_after_last_pulse)
        if e.cycle_index_offset is not None and sid in per_epoch_xy:
            k = int(e.cycle_index_offset)
            if k < pred.size:
                row.predicted_first_pulse_cycle_s = float(pred[k])
                resid = (t_first - pred[k]) / period
                row.offset_residual_cycles = float(resid)
                row.agrees_within_half_cycle = bool(abs(resid) <= agreement_tolerance_cycles)
                confirmed_flags.append(row.agrees_within_half_cycle)
                if not row.agrees_within_half_cycle:
                    row.notes.append(f"Counted offset {k} is contradicted by the FPGA prediction (first pulse sits {resid:+.2f} cycles away; "
                                     f"FPGA says cycle {est:.2f}).")
        else:
            if row.cycles_predicted_before_first_pulse and row.cycles_predicted_after_last_pulse:
                row.notes.append("Cycles without pulses predicted at both ends of the pulse train.")
            elif row.cycles_predicted_before_first_pulse:
                row.notes.append(f"FPGA places {row.cycles_predicted_before_first_pulse} cycles before the first pulse (head); "
                                 "the pulse train covers the acquisition's tail.")
            elif row.cycles_predicted_after_last_pulse:
                row.notes.append(f"FPGA places {row.cycles_predicted_after_last_pulse} cycles after the last pulse (tail): "
                                 "the acquisition continued after DI3 pulses stopped being recorded.")
            if row.cycles_predicted_inside_train != int(e.n_observed_pulses) + int(e.n_internal_missing_pulses):
                row.notes.append(f"{row.cycles_predicted_inside_train} cycles predicted inside the train vs {e.n_observed_pulses} pulses "
                                 f"(+{e.n_internal_missing_pulses} inferred dropped).")
        epochs_out.append(row)

    all_ok: Optional[bool] = None
    if confirmed_flags and continuous:
        all_ok = all(confirmed_flags)
    elif confirmed_flags and continuous is False:
        all_ok = False
    return FpgaCrossEpochCheck(
        schema_version=SCHEMA_VERSION, session_id=reg.session_id, dmd=int(reg.dmd), n_source_epochs=len(reg.epochs),
        n_identified_epochs=len(fit_ids), fpga_counter_continuous=continuous, global_fit_n_pulses=n_fit,
        global_rate_ratio_harp_over_fpga=glob_a, global_residual_rms_s=glob_rms, global_residual_max_s=glob_max,
        all_identified_offsets_confirmed=all_ok, epochs=epochs_out, limitations=limitations,
    )


def absolute_fpga_cycle_starts(dynamic_data_dir: PathLike, dmd: int) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
    """Absolute FPGA cycle start times (s) per acquisition prefix, chronological, 1-based epoch index.

    Returns ``(times_by_epoch, header_info_by_epoch)``; the header info records
    ``referenceTimestamp`` and the clock so a counter reset is visible.
    """
    from vip_slap2_analysis.io.slap2_dat import list_acquisition_prefixes, read_dmd_acquisition_timing
    out: Dict[int, np.ndarray] = {}
    info: Dict[int, Dict[str, Any]] = {}
    for i, prefix in enumerate(list_acquisition_prefixes(dynamic_data_dir), start=1):
        try:
            t = read_dmd_acquisition_timing(dynamic_data_dir, dmd, prefix=prefix)
        except FileNotFoundError:
            continue
        out[i] = float(t.reference_timestamp) / float(t.fpga_clock_hz) + np.asarray(t.cycle_start_s, dtype=float)
        info[i] = {"prefix": prefix, "reference_timestamp": int(t.reference_timestamp), "fpga_clock_hz": int(t.fpga_clock_hz),
                   "n_cycles": int(t.n_cycles), "n_source_gaps": int(t.gap_indices.size)}
    return out, info


def check_paths(qc_dir: PathLike, dmd: int) -> Path:
    return Path(qc_dir) / "sync" / f"fpga_cross_epoch_DMD{int(dmd)}.json"


def run_fpga_cross_epoch_check(asset: Any, *, dmd: int, dynamic_data_dir: PathLike, reg: Optional[CycleRegistration] = None,
                               save: bool = True) -> FpgaCrossEpochCheck:
    """Load the saved registration of ``dmd`` (unless given) and check it against the raw ``.dat`` FPGA timestamps."""
    from vip_slap2_analysis.common.cycle_registration import load_cycle_registration
    if reg is None:
        qc_dir = getattr(asset, "qc_dir", None)
        reg = load_cycle_registration(qc_dir, dmd) if qc_dir is not None else None
        if reg is None:
            raise FileNotFoundError(f"No saved cycle registration for DMD{dmd} under {qc_dir}; run scripts/run_cycle_registration.py first")
    abs_ts, info = absolute_fpga_cycle_starts(dynamic_data_dir, dmd)
    chk = cross_epoch_fpga_check(reg, abs_ts)
    refs = [v["reference_timestamp"] for _, v in sorted(info.items())]
    if any(b <= a for a, b in zip(refs, refs[1:])):
        chk.limitations.append(f"referenceTimestamp is not increasing across acquisitions ({refs}): the FPGA counter was reset; "
                               "cross-acquisition predictions are void.")
        chk.fpga_counter_continuous = False
        chk.all_identified_offsets_confirmed = None
    d = chk.to_dict()
    d["acquisitions"] = info
    if save and getattr(asset, "qc_dir", None) is not None:
        p = check_paths(asset.qc_dir, dmd)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=1, default=_json_default))
    return chk


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
