"""Cross-acquisition FPGA check: counted offsets are confirmed (or contradicted) by the shared FPGA counter."""
import numpy as np

from vip_slap2_analysis.common.cycle_registration import (
    STATUS_IDENTIFIED, STATUS_INCONSISTENT, SourceEpochCycles, register_di3_to_source_cycles,
)
from vip_slap2_analysis.common.fpga_cross_epoch import cross_epoch_fpga_check

LINE_RATE = 10686.6669921875
LPC = 227
PERIOD = LPC / LINE_RATE
HARP_PPM = 6.0
FPGA_T0 = 171.4        # absolute FPGA time (s) of the first acquisition's cycle 0


def _session(n_a=2000, n_b=3000, gap_s=40.0, k_missing=193, n_c=None, c_tail_without_pulses=0):
    """Three acquisitions on one continuous FPGA counter; HARP starts k_missing cycles into A."""
    fpga_a = FPGA_T0 + np.arange(n_a) * PERIOD
    fpga_b = fpga_a[-1] + gap_s + np.arange(n_b) * PERIOD
    harp = lambda t: 5000.0 + (t - FPGA_T0) * (1 + HARP_PPM * 1e-6)     # HARP = a*FPGA + b
    t_a, t_b = harp(fpga_a), harp(fpga_b)
    srcs = [SourceEpochCycles(epoch_index=1, n_cycles=n_a, lines_per_cycle=LPC, line_rate_hz=LINE_RATE, cycle_start_s=fpga_a - fpga_a[0]),
            SourceEpochCycles(epoch_index=2, n_cycles=n_b, lines_per_cycle=LPC, line_rate_hz=LINE_RATE, cycle_start_s=fpga_b - fpga_b[0])]
    abs_ts = {1: fpga_a, 2: fpga_b}
    rises = [t_a[k_missing:], t_b]
    end = t_b[-1] + 20.0
    if n_c:
        fpga_c = fpga_b[-1] + gap_s + np.arange(n_c) * PERIOD
        t_c = harp(fpga_c)
        srcs.append(SourceEpochCycles(epoch_index=3, n_cycles=n_c, lines_per_cycle=LPC, line_rate_hz=LINE_RATE, cycle_start_s=fpga_c - fpga_c[0]))
        abs_ts[3] = fpga_c
        rises.append(t_c[: n_c - c_tail_without_pulses])      # pulses stop early, acquisition continues
        end = t_c[-1] + 20.0
    rises = np.concatenate(rises)
    reg = register_di3_to_source_cycles(rise_times_s=rises, record_start_s=t_a[k_missing] - 0.003, record_end_s=end,
                                        source_epochs=srcs, dmd=1)
    return reg, abs_ts


def test_leave_one_out_confirms_tail_anchor_offset():
    reg, abs_ts = _session()
    assert reg.status == STATUS_IDENTIFIED and reg.epochs[0].cycle_index_offset == 193
    chk = cross_epoch_fpga_check(reg, abs_ts)
    assert chk.fpga_counter_continuous is True
    assert chk.global_residual_rms_s < 5e-5
    e1, e2 = chk.epochs
    assert e1.fit_epochs == [2] and e2.fit_epochs == [1]
    assert abs(e1.offset_residual_cycles) < 0.05 and e1.agrees_within_half_cycle
    assert abs(e2.offset_residual_cycles) < 0.05 and e2.agrees_within_half_cycle
    assert chk.all_identified_offsets_confirmed is True


def test_wrong_offset_is_contradicted_by_fpga():
    reg, abs_ts = _session()
    reg.epochs[0].cycle_index_offset = 192          # pretend the count was off by one cycle
    chk = cross_epoch_fpga_check(reg, abs_ts)
    e1 = chk.epochs[0]
    assert e1.agrees_within_half_cycle is False and abs(e1.offset_residual_cycles - 1.0) < 0.05
    assert chk.all_identified_offsets_confirmed is False
    assert any("contradicted" in n for n in e1.notes)


def test_reset_counter_is_detected_not_trusted():
    reg, abs_ts = _session()
    abs_ts[2] = abs_ts[2] - abs_ts[2][0] + 3.0       # counter restarted for acquisition B
    chk = cross_epoch_fpga_check(reg, abs_ts)
    assert chk.fpga_counter_continuous is False
    assert chk.all_identified_offsets_confirmed is False or chk.all_identified_offsets_confirmed is None
    assert any("not continuous" in l for l in chk.limitations)


def test_inconsistent_epoch_is_located_by_fpga_prediction():
    # third acquisition: 326 cycles saved, pulses stop after 193 (acquisition kept running)
    reg, abs_ts = _session(n_c=326, c_tail_without_pulses=133)
    e3 = reg.epochs[2]
    assert e3.status == STATUS_INCONSISTENT and e3.candidate_offset_range == (0, 133)
    chk = cross_epoch_fpga_check(reg, abs_ts)
    assert chk.fpga_counter_continuous is True
    r3 = chk.epochs[2]
    assert r3.used_in_fit is False and r3.fit_epochs == [1, 2]
    assert r3.fpga_offset_estimate_rounded == 0
    assert r3.cycles_predicted_before_first_pulse == 0
    assert r3.cycles_predicted_after_last_pulse == 133
    assert r3.cycles_predicted_inside_train == 193
    assert any("after the last pulse" in n for n in r3.notes)


def test_single_identified_epoch_cannot_be_cross_validated():
    reg, abs_ts = _session(n_a=2000, n_b=10, gap_s=40.0)
    # make epoch 2 unusable in the fit by dropping its pulses from the registration
    reg.epochs[1].pulse_times_harp_s = None
    chk = cross_epoch_fpga_check(reg, abs_ts)
    assert chk.n_identified_epochs == 1 and chk.fpga_counter_continuous is None
    assert chk.all_identified_offsets_confirmed is None
    assert chk.epochs[0].offset_residual_cycles is None


def test_absolute_fpga_cycle_starts_from_raw_dat(tmp_path):
    from synthetic_slap2 import FPGA_CLOCK_HZ, write_dat_chunks
    from vip_slap2_analysis.common.fpga_cross_epoch import absolute_fpga_cycle_starts
    dd = tmp_path / "dynamic_data"
    ref_a = 34_284_510_021
    ref_b = ref_a + int(round(60.0 * FPGA_CLOCK_HZ))          # second acquisition starts 60 s later on the same counter
    write_dat_chunks(dd, prefix="acquisition_090424", dmd=1, n_cycles=50, lines_per_cycle=LPC, reference_timestamp=ref_a, cycles_per_chunk=20)
    write_dat_chunks(dd, prefix="acquisition_090530", dmd=1, n_cycles=30, lines_per_cycle=LPC, reference_timestamp=ref_b, cycles_per_chunk=20)
    ts, info = absolute_fpga_cycle_starts(dd, 1)
    assert sorted(ts) == [1, 2] and info[1]["prefix"] == "acquisition_090424" and info[2]["reference_timestamp"] == ref_b
    assert abs(ts[1][0] - ref_a / FPGA_CLOCK_HZ) < 1e-9
    assert abs((ts[2][0] - ts[1][0]) - 60.0) < 1e-6
    assert ts[1].size == 50 and ts[2].size == 30
