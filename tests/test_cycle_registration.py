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
    times, labels, info, anchors = registered_epoch_sample_times(
        reg, plan, trial_source_lines=trial_lines, source_trial_epoch=[1, 1, 1], analysis_trial_epoch=[1, 1, 1],
        trial_lengths_samples=lengths, trial_source_start_samples=starts,
    )
    assert plan.trial_leading_samples == {1: expected_lead, 2: 0, 3: 0}
    # the anchor is the start of the first HARP-recorded cycle (= behavior epoch start), independent of frame phase
    assert anchors[1] == pytest.approx(truth[k] - 900.0, abs=1e-9)
    t = times[1]
    assert t.size == sum(lengths) and np.all(np.diff(t) > 0)
    first_line = thirds[0][expected_lead]
    expected_first = truth[(first_line - 1) // LPC] + ((first_line - 1) % LPC) / LINE_RATE * (1 + HARP_PPM * 1e-6) - 900.0
    assert t[0] == pytest.approx(expected_first, abs=1e-9)
    assert np.all(labels[1] == 0)


def test_anchor_guard_independent_of_frame_phase():
    """B1 regression: frames (5 ms) rarely start at a cycle boundary; the origin guard must use the cycle start."""
    from vip_slap2_analysis.common.epoch_alignment import build_epoch_aware_timebase, reconcile_trial_epochs
    import pandas as pd

    truth = _truth()
    for k in (100, 121, 123, 130, 150):
        reg = _reg(truth[k:], truth[k] - 0.002, truth[-1] + 5, inputs={"behavior_time_origin_harp_s": 900.0})
        dt = LINE_RATE / 200.0
        lines = np.ceil(np.arange(1, N * LPC, dt)).astype(int)
        plan = plan_registered_samples(reg, trial_source_lines={1: lines}, trial_source_epoch=[1], behavior_time_origin_harp_s=900.0)
        epoch_df = pd.DataFrame([{"epoch_index": 1, "start_time": truth[k] - 900.0, "end_time": truth[-1] - 900.0}])
        rec = reconcile_trial_epochs([lines.size], sample_rate_hz=200.0, behavior_epoch_df=epoch_df, source_trial_epoch=[1],
                                     trial_leading_samples=plan.trial_leading_samples)
        times, labels, info, anchors = registered_epoch_sample_times(
            reg, plan, trial_source_lines={1: lines}, source_trial_epoch=[1], analysis_trial_epoch=rec.analysis_trial_epoch,
            trial_lengths_samples=rec.trial_lengths_samples, trial_source_start_samples=rec.trial_source_start_samples)
        tb = build_epoch_aware_timebase(rec.trial_lengths_samples, sample_rate_hz=200.0, epoch_df=rec.behavior_epoch_df,
                                        trial_epoch=rec.analysis_trial_epoch, epoch_sample_times_sec=times,
                                        epoch_sample_time_labels=labels, epoch_anchor_sec=anchors)
        assert tb.metadata["timebase_method_by_epoch"] == {"1": "di3_pulse_registered"}
        first_line = lines[int(rec.trial_source_start_samples[0])]
        assert (first_line - 1) // LPC == k                      # first retained frame is in the first recorded cycle
        assert tb.timebase_sec[0] >= truth[k] - 900.0 - 1e-9     # and never before the first recorded pulse


def test_placeholder_trials_without_lines_are_dropped_not_guessed():
    truth = _truth()
    reg = _reg(truth[50:], truth[50] - 0.002, truth[-1] + 5, inputs={"behavior_time_origin_harp_s": 900.0})
    lines = np.arange(1, N * LPC, 53)
    a, b, c = np.array_split(lines, 3)
    plan = plan_registered_samples(reg, trial_source_lines={2: b, 3: c}, trial_source_epoch=[1, 1, 1], behavior_time_origin_harp_s=900.0)
    assert plan.applied_source_epochs == [1]
    assert plan.trials_without_lines == {1: [1]}
    assert plan.trial_leading_samples == {2: 0, 3: 0}   # trial 2 starts after the unrecorded cycles


def test_source_side_gaps_refuse_count_registration():
    truth = _truth()
    src = SourceEpochCycles(epoch_index=1, n_cycles=N, lines_per_cycle=LPC, line_rate_hz=LINE_RATE, n_source_gaps=2)
    reg = _reg(truth, truth[0] - 5, truth[-1] + 5, [src])
    assert reg.status == STATUS_INCONSISTENT
    assert any("discontinuous" in n for n in reg.epochs[0].notes)


def test_count_based_pairing_with_unmatched_di3_fragments_and_trailing_pulse():
    """863774_2026-08-07 pattern: 7 saved acquisitions, 11 DI3 trains, +1 pulse per interior acquisition."""
    from vip_slap2_analysis.common.cycle_registration import STATUS_UNAVAILABLE
    t = 1000.0
    srcs, trains = [], []
    gap = 40.0
    for i, n in enumerate([3000, 2000, 2500], start=1):
        srcs.append(SourceEpochCycles(epoch_index=i, n_cycles=n, lines_per_cycle=LPC, line_rate_hz=LINE_RATE,
                                      has_partial_trailing_cycle=True))
        # a short unmatched preview train (40 pulses) before each real acquisition
        trains.append(t + np.arange(40) * PERIOD); t += 40 * PERIOD + gap
        cyc = t + np.arange(n + 1) * PERIOD * (1 + HARP_PPM * 1e-6)  # +1 pulse: acquisition stopped inside cycle n
        trains.append(cyc); t = cyc[-1] + gap
    rises = np.concatenate(trains)
    # HARP started late in acquisition 1 (missing 25 cycles) and stopped right after acquisition 3's last pulse
    first_real = trains[1]
    rises = rises[(rises >= first_real[25] - 1e-9)]
    reg = _reg(rises, first_real[25] - 0.002, rises[-1] + 0.002, srcs)
    assert reg.n_di3_epochs == 5  # the first preview train fell before the recording start
    assert len(reg.inputs["unmatched_di3_trains"]) == 2
    a, b, c = reg.epochs
    assert a.status == STATUS_IDENTIFIED and a.anchor == "tail" and a.cycle_index_offset == 25
    assert b.status == STATUS_IDENTIFIED and b.anchor == "both" and b.cycle_index_offset == 0
    assert any("unsaved partial final cycle" in n for n in b.notes)
    # last acquisition: head covered by the gap, tail not (HARP stopped at the last pulse) -> head anchor
    assert c.status == STATUS_IDENTIFIED and c.anchor == "head" and c.cycle_index_offset == 0
    assert reg.status == STATUS_IDENTIFIED


def test_source_epoch_without_di3_train_is_reported_unavailable():
    from vip_slap2_analysis.common.cycle_registration import STATUS_UNAVAILABLE
    truth = _truth(n=2000)
    srcs = [_src(n=2000, epoch=1), _src(n=1500, epoch=2)]   # second acquisition after HARP stopped
    reg = _reg(truth, truth[0] - 5, truth[-1] + 5, srcs)
    assert [e.status for e in reg.epochs] == [STATUS_IDENTIFIED, STATUS_UNAVAILABLE]
    assert reg.inputs["unmatched_source_epochs"] == [2]
    assert reg.status == STATUS_IDENTIFIED  # identified wherever HARP recorded; epoch 2 carries no stimuli


def test_transfer_registration_to_second_dmd_via_shared_fpga():
    from vip_slap2_analysis.common.cycle_registration import RELATIONSHIP_TRANSFERRED, transfer_registration_via_fpga
    # DMD1: 227 lines/cycle, DMD2: 135 lines/cycle, same FPGA clock, same start tick
    LR = LINE_RATE
    N1, N2 = 4000, int(4000 * 227 / 135)
    fpga1 = np.arange(N1) * 227 / LR
    fpga2 = np.arange(N2) * 135 / LR
    harp_of = lambda fp: 1000.0 + fp * (1 + HARP_PPM * 1e-6)
    k = 80
    pulses = harp_of(fpga1)[k:]
    reg1 = _reg(pulses, pulses[0] - 0.002, pulses[-1] + 5, [_src(n=N1)])
    assert reg1.epochs[0].cycle_index_offset == k
    reg2 = transfer_registration_via_fpga(
        reg1, src_cycle_start_s={1: fpga1}, dst_cycle_start_s={1: fpga2},
        dst_lines_per_cycle={1: 135}, dst_line_rate_hz={1: LR}, dst_dmd=2,
    )
    assert reg2.relationship == RELATIONSHIP_TRANSFERRED and reg2.status == STATUS_IDENTIFIED and reg2.dmd == 2
    e = reg2.epochs[0]
    truth2 = harp_of(fpga2)
    k2 = int(np.flatnonzero(truth2 >= pulses[0] - 1e-6)[0])
    assert e.cycle_index_offset == k2 and e.anchor_tolerance_s > 135 / LR
    times, labels = e.cycle_times_harp_s()
    np.testing.assert_allclose(times, truth2, atol=5e-6)
    assert np.all(labels[k2:] == 0) and np.all(labels[:k2] == 1)
    assert e.timing_source == "transferred_from_dmd1_shared_fpga"


def test_legacy_half_open_trial_boundaries_duplicate_sample_is_skipped():
    """extractDendrites (pre 2026-09-03) trial tables repeat the previous trial's last line."""
    truth = _truth()
    reg = _reg(truth, truth[0] - 5, truth[-1] + 5, inputs={"behavior_time_origin_harp_s": 900.0})
    L = N * LPC
    edges = np.linspace(1, L + 1, 4).astype(int)            # half-open: last(i) == first(i+1)
    lines = {1: np.arange(edges[0], edges[1] + 1), 2: np.arange(edges[1], edges[2] + 1), 3: np.arange(edges[2], min(edges[3], L) + 1)}
    plan = plan_registered_samples(reg, trial_source_lines=lines, trial_source_epoch=[1, 1, 1], behavior_time_origin_harp_s=900.0)
    assert plan.trial_leading_samples == {1: 0, 2: 1, 3: 1}
    assert any("duplicated trial-boundary" in n for n in plan.notes)
    lengths = [lines[t].size - plan.trial_leading_samples[t] for t in (1, 2, 3)]
    times, labels, info, anchors = registered_epoch_sample_times(
        reg, plan, trial_source_lines=lines, source_trial_epoch=[1, 1, 1], analysis_trial_epoch=[1, 1, 1],
        trial_lengths_samples=lengths, trial_source_start_samples=[0, 1, 1])
    assert np.all(np.diff(times[1]) > 0) and times[1].size == min(L, edges[3]) - 0
