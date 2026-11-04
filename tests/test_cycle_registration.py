import numpy as np
import pytest

from vip_slap2_analysis.common.cycle_registration import (
    RELATIONSHIP_DIRECT, RELATIONSHIP_INTEGER_LINE, STATUS_AMBIGUOUS, STATUS_IDENTIFIED,
    STATUS_IDENTIFIED_ASSUMED, STATUS_INCONSISTENT, SourceEpochCycles, load_cycle_registration,
    plan_registered_samples, register_di3_to_source_cycles, registered_epoch_sample_times,
    save_cycle_registration, source_lines_to_harp_time, split_di3_pulse_epochs,
)

LINE_RATE = 10686.6669921875
LPC = 227                      # ASAP voltage-style DMD1 cycle (20.7 ms)
PERIOD = LPC / LINE_RATE
N = 5000                       # source cycles
HARP_PPM = 6.0                 # HARP clock runs 6 ppm fast relative to the FPGA


def _truth(n=N, t0=1000.0):
    """HARP time of every source cycle start."""
    return t0 + np.arange(n) * PERIOD * (1 + HARP_PPM * 1e-6)


def _src(n=N, epoch=1):
    return SourceEpochCycles(epoch_index=epoch, n_cycles=n, lines_per_cycle=LPC, line_rate_hz=LINE_RATE)


def _reg(rises, start, end, srcs=None, **kw):
    return register_di3_to_source_cycles(
        rise_times_s=rises, record_start_s=start, record_end_s=end, source_epochs=srcs or [_src()], dmd=1, **kw
    )


def test_complete_coverage_identified_offset_zero():
    truth = _truth()
    reg = _reg(truth, truth[0] - 12.0, truth[-1] + 30.0)
    assert reg.relationship == RELATIONSHIP_DIRECT
    assert reg.status == STATUS_IDENTIFIED and reg.safe_for_physiology
    assert reg.acquisition_start_order == "harp_first"
    e = reg.epochs[0]
    assert e.anchor == "both" and e.cycle_index_offset == 0 and e.candidate_offset_range == (0, 0)
    assert abs(e.period_error_ppm - HARP_PPM) < 0.5
    times, label = e.cycle_times_harp_s()
    np.testing.assert_allclose(times, truth, atol=1e-9)
    assert np.all(label == 0)


def test_missing_head_with_tail_anchor_recovers_offset():
    truth = _truth()
    k = 150
    reg = _reg(truth[k:], truth[k] - 0.003, truth[-1] + 30.0)
    e = reg.epochs[0]
    assert reg.status == STATUS_IDENTIFIED
    assert reg.acquisition_start_order == "slap2_first_or_simultaneous"
    assert e.anchor == "tail" and e.cycle_index_offset == k
    assert e.missing_initial_cycles == k and e.missing_terminal_cycles == 0
    times, label = e.cycle_times_harp_s()
    np.testing.assert_allclose(times[k:], truth[k:], atol=1e-9)
    # extrapolated head cycles use the observed cadence
    np.testing.assert_allclose(times[:k], truth[:k], atol=1e-6)
    assert np.all(label[:k] == 1) and np.all(label[k:] == 0)


def test_missing_tail_with_head_anchor():
    truth = _truth()
    reg = _reg(truth[:-200], truth[0] - 5.0, truth[-201] + 0.002)
    e = reg.epochs[0]
    assert reg.status == STATUS_IDENTIFIED and e.anchor == "head"
    assert e.cycle_index_offset == 0 and e.missing_terminal_cycles == 200
    _, label = e.cycle_times_harp_s()
    assert np.sum(label == 2) == 200


def test_missing_both_ends_is_ambiguous_with_explicit_range():
    truth = _truth()
    reg = _reg(truth[150:-200], truth[150] - 0.002, truth[-201] + 0.002)
    e = reg.epochs[0]
    assert reg.status == STATUS_AMBIGUOUS and not reg.safe_for_physiology
    assert e.cycle_index_offset is None and e.candidate_offset_range == (0, 350)
    with pytest.raises(ValueError):
        e.cycle_times_harp_s()


def test_internal_dropped_pulses_are_counted_not_mistaken_for_offset():
    truth = _truth()
    rises = np.delete(truth, [1000, 1001, 3000])
    reg = _reg(rises, truth[0] - 5.0, truth[-1] + 5.0)
    e = reg.epochs[0]
    assert reg.status == STATUS_IDENTIFIED and e.cycle_index_offset == 0
    assert e.n_internal_missing_pulses == 3 and len(e.internal_gaps) == 2
    times, label = e.cycle_times_harp_s()
    assert np.sum(label == 3) == 3
    np.testing.assert_allclose(times, truth, atol=1e-6)


def test_more_pulses_than_cycles_is_inconsistent():
    truth = _truth(n=N + 10)
    reg = _reg(truth, truth[0] - 5.0, truth[-1] + 5.0)
    assert reg.status == STATUS_INCONSISTENT


def test_count_equal_without_margins_is_identified_with_assumptions():
    truth = _truth()
    reg = _reg(truth, truth[0] - 0.001, truth[-1] + 0.001)
    assert reg.status == STATUS_IDENTIFIED_ASSUMED and reg.epochs[0].cycle_index_offset == 0


def test_multi_epoch_pairing_and_interior_anchors():
    t_a = _truth(n=2000, t0=1000.0)
    t_b = _truth(n=3000, t0=t_a[-1] + 40.0)  # 40 s imaging-off gap
    srcs = [_src(n=2000, epoch=1), _src(n=3000, epoch=2)]
    # HARP starts late (misses 20 of epoch A) and stops early (misses 30 of epoch B)
    rises = np.r_[t_a[20:], t_b[:-30]]
    reg = _reg(rises, t_a[20] - 0.002, t_b[-31] + 0.002, srcs)
    assert reg.n_di3_epochs == 2 and reg.status == STATUS_IDENTIFIED
    a, b = reg.epochs
    assert a.anchor == "tail" and a.cycle_index_offset == 20
    assert b.anchor == "head" and b.cycle_index_offset == 0 and b.missing_terminal_cycles == 30


def test_epoch_count_mismatch_refuses_pairing():
    t_a = _truth(n=2000)
    reg = _reg(t_a, t_a[0] - 5, t_a[-1] + 5, [_src(n=1000, epoch=1), _src(n=1000, epoch=2)])
    assert reg.status == STATUS_INCONSISTENT and reg.epochs == []


def test_integer_line_cadence_blocks_count_registration():
    # DI3 every 228 lines while the DMD cycle is 227 lines (historical glutamate-like)
    per = 228 / LINE_RATE
    rises = 1000.0 + np.arange(4000) * per
    reg = _reg(rises, rises[0] - 5, rises[-1] + 5, [_src(n=N)])
    assert reg.relationship == RELATIONSHIP_INTEGER_LINE
    assert reg.status == STATUS_AMBIGUOUS
    reg2 = _reg(rises, rises[0] - 5, rises[-1] + 5, [_src(n=N)], assume_first_pulse_is_acquisition_start=True)
    assert reg2.status == STATUS_IDENTIFIED_ASSUMED and reg2.epochs[0].cycle_index_offset == 0


def test_source_lines_to_harp_time_and_rate_correction():
    truth = _truth()
    reg = _reg(truth, truth[0] - 5, truth[-1] + 5)
    e = reg.epochs[0]
    lines = np.array([1, LPC, LPC + 1, 10 * LPC + 5])
    t, lab = source_lines_to_harp_time(lines, e, lines_per_cycle=LPC, line_rate_hz=LINE_RATE)
    assert t[0] == pytest.approx(truth[0], abs=1e-9)
    assert t[2] == pytest.approx(truth[1], abs=1e-9)
    # within-cycle spacing uses the HARP-corrected line period
    assert (t[1] - t[0]) == pytest.approx((LPC - 1) / LINE_RATE * (1 + HARP_PPM * 1e-6), rel=1e-6)
    # line 10*LPC+5 is the 5th line (0-based offset 4) of cycle index 10
    assert t[3] == pytest.approx(truth[10] + 4 / LINE_RATE * (1 + HARP_PPM * 1e-6), rel=1e-9)


def test_save_and_load_roundtrip(tmp_path):
    truth = _truth()
    reg = _reg(truth[100:], truth[100] - 0.002, truth[-1] + 5, inputs={"behavior_time_origin_harp_s": 990.0})
    p = save_cycle_registration(reg, tmp_path / "qc")
    assert p.exists() and p.name == "cycle_registration_DMD1.json"
    back = load_cycle_registration(tmp_path / "qc", 1)
    assert back.status == reg.status and back.epochs[0].cycle_index_offset == 100
    assert back.behavior_time_origin_harp_s == 990.0
    np.testing.assert_allclose(back.epochs[0].pulse_times_harp_s, truth[100:])
    assert load_cycle_registration(tmp_path / "qc", 2) is None


def test_plan_and_sample_times_skip_leading_frames():
    truth = _truth()
    k = 40
    reg = _reg(truth[k:], truth[k] - 0.002, truth[-1] + 5, inputs={"behavior_time_origin_harp_s": 900.0})
    # frames every 53 lines (~200 Hz), 3 trials
    lines = np.arange(1, N * LPC, 53)
    thirds = np.array_split(lines, 3)
    trial_lines = {i + 1: arr for i, arr in enumerate(thirds)}
    plan = plan_registered_samples(reg, trial_source_lines=trial_lines, trial_source_epoch=[1, 1, 1], behavior_time_origin_harp_s=900.0)
    expected_lead = int(np.sum((lines - 1) // LPC < k))
    assert plan.source_epoch_leading_samples == {1: expected_lead}
    lengths = [len(t) for t in thirds]
    starts = [expected_lead, 0, 0]
    lengths[0] -= expected_lead
    times, labels, info = registered_epoch_sample_times(
        reg, plan, trial_source_lines=trial_lines, source_trial_epoch=[1, 1, 1], analysis_trial_epoch=[1, 1, 1],
        trial_lengths_samples=lengths, trial_source_start_samples=starts,
    )
    t = times[1]
    assert t.size == sum(lengths) and np.all(np.diff(t) > 0)
    first_line = thirds[0][expected_lead]
    expected_first = truth[(first_line - 1) // LPC] + ((first_line - 1) % LPC) / LINE_RATE * (1 + HARP_PPM * 1e-6) - 900.0
    assert t[0] == pytest.approx(expected_first, abs=1e-9)
    assert np.all(labels[1] == 0)
