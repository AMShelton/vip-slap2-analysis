"""Produce the retrospective-synchronization QC figures and the truncation benchmark.

    python scripts/make_sync_qc_figures.py --reference-dir <test_subject_2026-10-27_09-00-55> --out-dir <dir>

Outputs (all read-only with respect to the reference session):
  reference_registration_coverage.png   DI3 coverage vs DMD1 source cycles (positive control)
  reference_pulse_residuals.png         DI3 rise residuals vs FPGA cycle-start timestamps
  reference_truncation_benchmark.{png,csv}  identifiability under head/tail/both truncation
  synthetic_before_after_alignment.png  stimulus-locked average, historical vs registered timebase
  reference_registration_DMD1.json      the registration record for the reference
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vip_slap2_analysis.common.cycle_registration import (  # noqa: E402
    register_di3_to_source_cycles, source_epochs_from_raw_dat,
)
from vip_slap2_analysis.io.harp_bin import read_digital_input_edges  # noqa: E402
from vip_slap2_analysis.io.slap2_dat import read_dmd_acquisition_timing  # noqa: E402
from vip_slap2_analysis.plotting.sync_qc_plots import (  # noqa: E402
    plot_before_after_event_alignment, plot_pulse_residuals, plot_registration_coverage, plot_truncation_benchmark,
)


def reference_figures(ref: Path, out: Path) -> None:
    dd = ref / "slap2" / "dynamic_data"
    harp = ref / "behavior" / "VCO1_Behavior.harp"
    rec = read_digital_input_edges(harp / "Behavior_32.bin", record_end_from=harp / "Behavior_44.bin")
    src = source_epochs_from_raw_dat(dd, 1)
    timing = read_dmd_acquisition_timing(dd, 1)
    reg = register_di3_to_source_cycles(rise_times_s=rec.rise_times_s, record_start_s=rec.record_start_s,
                                        record_end_s=rec.record_end_s, source_epochs=src, dmd=1,
                                        session_id=ref.name, inputs={"harp_recording": rec.to_dict()})
    (out / "reference_registration_DMD1.json").write_text(json.dumps(reg.to_dict(), indent=2, default=str))
    plot_registration_coverage(reg).savefig(out / "reference_registration_coverage.png", dpi=150); plt.close("all")
    plot_pulse_residuals(reg, source_cycle_start_s=timing.cycle_start_s).savefig(out / "reference_pulse_residuals.png", dpi=150); plt.close("all")

    r = rec.rise_times_s
    scenarios = [
        ("complete (HARP first, stops after)", r, rec.record_start_s, rec.record_end_s),
        ("head missing 150 (SLAP2 first), tail covered", r[150:], r[150] - 0.001, rec.record_end_s),
        ("tail missing 160, head covered", r[:-160], rec.record_start_s, r[-161] + 0.001),
        ("head 150 + tail 160 missing", r[150:-160], r[150] - 0.001, r[-161] + 0.001),
        ("complete, 3 internal pulses dropped", np.delete(r, [3000, 3001, 5000]), rec.record_start_s, rec.record_end_s),
        ("head missing 150 + 2 dropped, tail covered", np.delete(r[150:], [3000, 3001]), r[150] - 0.001, rec.record_end_s),
        ("count equal, no margins either end", r, r[0] - 0.001, r[-1] + 0.001),
    ]
    rows = []
    for name, rises, start, end in scenarios:
        rg = register_di3_to_source_cycles(rise_times_s=rises, record_start_s=start, record_end_s=end, source_epochs=src, dmd=1)
        e = rg.epochs[0]
        lo, hi = e.candidate_offset_range
        rows.append({"scenario": name, "status": e.status, "anchor": e.anchor, "offset": e.cycle_index_offset,
                     "candidate_range": f"{lo}..{hi}", "candidate_count": hi - lo + 1, "n_source_cycles": e.n_source_cycles,
                     "n_observed": e.n_observed_pulses, "internal_missing": e.n_internal_missing_pulses,
                     "start_order": rg.acquisition_start_order, "period_error_ppm": e.period_error_ppm})
    pd.DataFrame(rows).to_csv(out / "reference_truncation_benchmark.csv", index=False)
    plot_truncation_benchmark(rows).savefig(out / "reference_truncation_benchmark.png", dpi=150); plt.close("all")


def synthetic_before_after(out: Path) -> None:
    """Simulate a SLAP2-first session: stimulus-locked response under both timebases."""
    from vip_slap2_analysis.common.cycle_registration import SourceEpochCycles, register_di3_to_source_cycles
    from vip_slap2_analysis.common.epoch_alignment import build_epoch_aware_timebase, reconcile_trial_epochs
    from vip_slap2_analysis.common.cycle_registration import plan_registered_samples, registered_epoch_sample_times

    LR, LPC, N, K, PPM, FS = 10686.6669921875, 227, 4000, 120, 6.0, 200.0
    P = LPC / LR
    cyc_t = 1000.0 + np.arange(N) * P * (1 + PPM * 1e-6)
    origin = cyc_t[K] - 0.004
    rises = cyc_t[K:]
    src = [SourceEpochCycles(1, N, LPC, LR)]
    reg = register_di3_to_source_cycles(rise_times_s=rises, record_start_s=origin, record_end_s=cyc_t[-1] + 30, source_epochs=src, dmd=1,
                                        inputs={"behavior_time_origin_harp_s": origin})
    dt = LR / FS
    lines = np.ceil(np.arange(1, N * LPC, dt)).astype(int)
    truth_t = (cyc_t[(lines - 1) // LPC] + ((lines - 1) % LPC) / LR * (1 + PPM * 1e-6)) - origin
    rng = np.random.default_rng(0)
    stim_t = np.sort(rng.uniform(truth_t[200], truth_t[-200], 60))
    # synthetic response: 0.3 s half-gaussian starting at each onset
    sig = np.zeros(lines.size)
    for s in stim_t:
        m = (truth_t >= s) & (truth_t < s + 0.5)
        sig[m] += np.exp(-((truth_t[m] - s - 0.1) ** 2) / (2 * 0.05**2))
    epoch_df = pd.DataFrame([{"epoch_index": 1, "start_time": rises[0] - origin, "end_time": rises[-1] - origin}])
    lengths = np.array([lines.size])
    rec = reconcile_trial_epochs(lengths, sample_rate_hz=FS, behavior_epoch_df=epoch_df, source_trial_epoch=[1])
    tb_legacy = build_epoch_aware_timebase(rec.trial_lengths_samples, sample_rate_hz=FS, epoch_df=rec.behavior_epoch_df, trial_epoch=rec.analysis_trial_epoch)
    plan = plan_registered_samples(reg, trial_source_lines={1: lines}, trial_source_epoch=[1], behavior_time_origin_harp_s=origin)
    rec2 = reconcile_trial_epochs(lengths, sample_rate_hz=FS, behavior_epoch_df=epoch_df, source_trial_epoch=[1],
                                  source_epoch_leading_samples=plan.source_epoch_leading_samples)
    times, labels, _ = registered_epoch_sample_times(reg, plan, trial_source_lines={1: lines}, source_trial_epoch=[1], analysis_trial_epoch=[1],
                                                    trial_lengths_samples=rec2.trial_lengths_samples, trial_source_start_samples=rec2.trial_source_start_samples)
    tb_reg = build_epoch_aware_timebase(rec2.trial_lengths_samples, sample_rate_hz=FS, epoch_df=rec2.behavior_epoch_df, trial_epoch=rec2.analysis_trial_epoch,
                                        epoch_sample_times_sec=times, epoch_sample_time_labels=labels)
    n_pre, n_post = int(0.25 * FS), int(0.75 * FS)
    t_rel = (np.arange(n_pre + n_post) - n_pre) / FS

    def locked(tb, data):
        snips = []
        for s in stim_t:
            i = int(np.argmin(np.abs(tb - s)))
            if i - n_pre >= 0 and i + n_post <= data.size:
                snips.append(data[i - n_pre:i + n_post])
        return np.mean(snips, axis=0)

    leg = locked(tb_legacy.timebase_sec, sig[: rec.trial_lengths_samples[0]])
    start = int(rec2.trial_source_start_samples[0])
    regd = locked(tb_reg.timebase_sec, sig[start:start + rec2.trial_lengths_samples[0]])
    fig = plot_before_after_event_alignment(t_rel=t_rel, legacy_mean=leg, registered_mean=regd,
                                            title=f"Synthetic SLAP2-first session: {K} unrecorded leading cycles ({K * P * 1e3:.0f} ms); "
                                                  f"HARP/source clock {PPM:+.0f} ppm", ylabel="simulated dF (a.u.)")
    fig.savefig(out / "synthetic_before_after_alignment.png", dpi=150); plt.close("all")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    reference_figures(Path(a.reference_dir), out)
    synthetic_before_after(out)
    print("wrote figures to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
