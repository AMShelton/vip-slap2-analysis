import numpy as np
import pandas as pd
import pytest

from synthetic_slap2 import (
    LINE_RATE_HZ, write_dat_chunks, write_giant_experiment_summary, write_giant_trial_table, write_meta_file,
    write_voltex_summary, write_voltex_trial_table,
)
from vip_slap2_analysis.io.giant_summary import GiantExperimentSummary, validate_giant_summary_schema
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
from vip_slap2_analysis.io.summary_loader import load_glutamate_summary, load_voltage_summary
from vip_slap2_analysis.voltage.voltex_summary import VoltexSummary, validate_voltex_summary_schema


def test_giant_schema_validation_and_reader(tmp_path):
    p = tmp_path / "experiment_summary.h5"
    trial_frames = [100, 0, 80]
    lines = [np.arange(1, 5301, 53), np.zeros(0, dtype=int), np.arange(5301, 5301 + 80 * 53, 53)]
    write_giant_experiment_summary(p, trial_frames=trial_frames, n_sources=5, n_channels=2, activity_channel=2,
                                   frame_lines_by_trial=lines)
    rep = validate_giant_summary_schema(p)
    assert rep["compatible"], rep["problems"]
    assert rep["paths"]["Path1"]["n_frames"] == 180 and rep["paths"]["Path1"]["n_sources"] == 5
    write_giant_trial_table(tmp_path / "trial_table.h5", epoch_by_trial=[1, 1, 2], first_line=[1, 5301, 5301], last_line=[5300, 5300, 9540])
    exp = GiantExperimentSummary(p)
    assert exp.n_dmds == 1 and exp.n_trials == 3 and exp.valid_trials[0] == [1, 3]
    assert exp.n_synapses == [5]
    assert exp.trial_epoch.tolist() == [1, 1, 2] and exp.trial_epoch_source.startswith("trial_table.h5")
    x = exp.get_traces(1, 1, signal="dF", mode="ls", squeeze_channels=False)
    assert x.shape == (100, 5, 2)
    # activity channel (MATLAB channel 2) is reordered to index 0 and holds the written values
    assert np.isfinite(x[:, :, 0]).all() and np.isnan(x[:, :, 1]).all()
    assert x[0, 1, 0] == pytest.approx(100.0) and x[1, 0, 0] == pytest.approx(1.0)
    dff = exp.get_traces(1, 3, signal="dFF", mode="ls", channels=0)
    assert dff.shape == (80, 5) and dff[0, 0] == pytest.approx(100.0 / 10.0)
    fl = exp.get_trial_source_lines(1, 3)
    assert fl[0] == 5301 and fl.size == 80
    with pytest.raises(ValueError):
        exp.get_traces(1, 2)
    assert exp.get_source_coords(1).shape == (5, 3)
    assert exp.get_dmd_metadata(1) == {} and exp.get_dmd_epoch_metadata(1) == []  # no raw headers
    exp.close()


def test_giant_reader_rejects_incompatible_file(tmp_path):
    import h5py
    p = tmp_path / "experiment_summary.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("row_major", data=np.array([[0]], dtype=np.uint8))
        f.create_group("Path1/frame_info")
    rep = validate_giant_summary_schema(p)
    assert not rep["compatible"] and any("missing" in s for s in rep["problems"])
    with pytest.raises(ValueError):
        GiantExperimentSummary(p)


def test_giant_acquisition_metadata_from_raw_headers(tmp_path):
    dd = tmp_path / "slap2" / "dynamic_data"
    write_dat_chunks(dd, prefix="acq_1", dmd=1, n_cycles=300, lines_per_cycle=67, reference_timestamp=5000)
    write_meta_file(dd, prefix="acq_1", dmd=1, n_cycles_by_dmd=[300, 400], lines_per_cycle=67)
    p = tmp_path / "experiment_summary.h5"
    write_giant_experiment_summary(p, trial_frames=[50], n_sources=2, frame_lines_by_trial=[np.arange(1, 50 * 53, 53)])
    exp = GiantExperimentSummary(p, dynamic_data_dir=dd)
    md = exp.get_dmd_metadata(1)
    assert md["linesPerCycle"] == 67 and md["lineRateHz"] == pytest.approx(LINE_RATE_HZ)
    ep = exp.get_dmd_epoch_metadata(1)
    assert len(ep) == 1 and ep[0]["nCycles"] == 300 and ep[0]["totalNumLines"] == 300 * 67
    assert exp.get_dmd_epoch_durations_sec(1)[1] == pytest.approx(300 * 67 / LINE_RATE_HZ)


def test_voltex_reader_api(tmp_path):
    p = tmp_path / "Voltage" / "VoltageSummary.h5"; p.parent.mkdir()
    censor = np.zeros(900, dtype=np.uint8); censor[10:20] = 1
    write_voltex_summary(p, trial_lines=[400, 0, 500], trial_epoch=[1, 1, 1], n_rois=3, motion_censor=censor)
    write_voltex_trial_table(tmp_path / "trial_table.h5", epoch_by_trial=[1, 1, 1], first_line=[1, 401, 401], last_line=[400, 400, 900])
    rep = validate_voltex_summary_schema(p)
    assert rep["compatible"] and rep["format_version"] == "0.2.3"
    vs = VoltexSummary(p)
    assert vs.n_dmds == 1 and vs.n_trials == 3 and vs.n_rois == [3] and vs.n_total_rois == 3
    assert vs.valid_trials[0] == [1, 3] and vs.available_trace_modes() == ["trial"]
    x = vs.get_roi_traces(dmd=1, trial=3, t_slice=slice(0, 10))
    assert x.shape == (10, 3) and x[0, 0] == pytest.approx(400.0) and x[0, 2] == pytest.approx(2400.0)
    assert vs.get_trial_source_lines(1, 3)[0] == 401
    assert vs.get_discard_frames(1, 1).sum() == 0
    vs2 = VoltexSummary(p, apply_motion_censor=True)
    assert vs2.get_discard_frames(1, 1).sum() == 10
    assert vs2.get_roi_traces(1, 1, drop_discarded=True).shape == (390, 3)
    d = vs.get_roi_traces(1, 1, trace_kind="dff")
    assert d[0, 0] == pytest.approx((0.0 - 100.0) / 100.0)
    assert vs.get_line_rate_hz() == pytest.approx(LINE_RATE_HZ)
    ep = vs.get_dmd_epoch_metadata(1)
    assert len(ep) == 1 and ep[0]["totalNumLines"] == 900
    assert vs.get_roi_masks(1).shape == (3, 4, 3)
    assert vs.trial_table["line_range_convention"] == "inclusive"
    vs.close()


def _registry_for(session_dir):
    df = pd.DataFrame([{"session_id": session_dir.name, "subject_id": 1, "session_dir": session_dir, "session_type": "A0",
                        "paradigm": "change_detection_passive", "indicator1": "iGluSnFR4f"}])
    return VIPSessionRegistry(summary_xlsx=session_dir / "none.xlsx", subjects_df=pd.DataFrame(), sessions_df=df)


def test_registry_discovers_modern_and_legacy_with_precedence(tmp_path):
    sd = tmp_path / "900001_2026-08-04_08-12-47"
    proc = sd / "processed" / "source_extraction"
    (proc / "summarize_LoCo").mkdir(parents=True)
    (proc / "SILo").mkdir(parents=True)
    write_giant_experiment_summary(proc / "summarize_LoCo" / "experiment_summary.h5", trial_frames=[10], n_sources=1)
    write_giant_experiment_summary(proc / "SILo" / "experiment_summary.h5", trial_frames=[10], n_sources=1)
    write_giant_trial_table(sd / "processed" / "trial_table.h5", epoch_by_trial=[1], first_line=[1], last_line=[10])
    (proc / "ExperimentSummary").mkdir()
    (proc / "ExperimentSummary" / "SummaryLoCo-260804-100000.mat").write_bytes(b"not a real mat")
    (proc / "Voltage").mkdir()
    write_voltex_summary(proc / "Voltage" / "VoltageSummary.h5", trial_lines=[100], trial_epoch=[1], n_rois=2)
    dd = sd / "slap2" / "dynamic_data"
    write_meta_file(dd if dd.mkdir(parents=True) is None else dd, prefix="acq", dmd=1, n_cycles_by_dmd=[1, 1], lines_per_cycle=67)
    reg = _registry_for(sd)
    a = reg.resolve_assets(sd.name)
    g = a.get_modality_assets("glutamate")
    assert g["summary_format"] == "giant_experiment_summary_h5"
    assert "summarize_LoCo" in str(g["summary_path"])  # summarize_LoCo preferred over SILo
    assert g["summary_mat"] is not None and g["summary_candidates"]["legacy_summary_loco_mat"] is not None
    assert g["trial_table_h5"] is not None
    v = a.get_modality_assets("voltage")
    assert v["summary_format"] == "voltex_voltage_summary_h5" and v["summary_mat"] is None
    assert a.get_asset("slap2", "dynamic_data_dir") == dd
    # legacy preference flips the selected product but keeps the modern one recorded
    b = reg.resolve_assets(sd.name, summary_preference="legacy")
    gb = b.get_modality_assets("glutamate")
    assert gb["summary_format"] == "legacy_summary_loco_mat" and gb["summary_h5"] is not None
    # backward-compatible top-level field still points at the legacy MAT
    assert a.summary_mat == g["summary_mat"]


def test_summary_loader_falls_back_when_modern_is_incompatible(tmp_path):
    import h5py
    sd = tmp_path / "900002_2026-08-04_08-12-47"
    proc = sd / "processed" / "source_extraction" / "summarize_LoCo"; proc.mkdir(parents=True)
    with h5py.File(proc / "experiment_summary.h5", "w") as f:  # incomplete modern file
        f.create_dataset("row_major", data=np.array([[0]], dtype=np.uint8))
    (sd / "behavior").mkdir()
    reg = _registry_for(sd)
    a = reg.resolve_assets(sd.name)
    # an HDF5 file that is neither GIAnT nor Voltex is reported, not assigned to a modality
    assert a.get_asset("glutamate", "summary_format") is None
    assert a.get_asset("slap2", "unclassified_summary_h5") and "experiment_summary.h5" in a.get_asset("slap2", "unclassified_summary_h5")[0]
    with pytest.raises(FileNotFoundError):
        load_glutamate_summary(a)  # nothing loadable -> explicit failure
    # a Voltex-format file *named* experiment_summary.h5 under Voltage/ is classified by content
    (sd / "processed" / "Voltage").mkdir()
    write_voltex_summary(sd / "processed" / "Voltage" / "experiment_summary.h5", trial_lines=[50], trial_epoch=[1], n_rois=1)
    a = reg.resolve_assets(sd.name)
    assert a.get_asset("glutamate", "summary_h5") is None
    assert str(a.get_asset("voltage", "summary_h5")).endswith("Voltage/experiment_summary.h5") or str(a.get_asset("voltage", "summary_h5")).endswith("Voltage\\experiment_summary.h5")
    vs, prov = load_voltage_summary(a)
    assert prov["selected_format"] == "voltex_voltage_summary_h5" and prov["attempts"][0]["schema_compatible"]
    assert vs.summary_format == "voltex_voltage_summary_h5"
    vs.close()
    # a schema-incompatible GIAnT-like file falls through to the legacy MAT with the reason recorded
    import h5py
    bad = sd / "processed" / "source_extraction" / "SILo"; bad.mkdir()
    with h5py.File(bad / "experiment_summary.h5", "w") as f:
        f.create_dataset("row_major", data=np.array([[0]], dtype=np.uint8))
        f.create_group("params")
        f.create_dataset("Path1/sources/temporal/dF_ls", data=np.zeros((3, 1, 2)))  # missing frame_info -> incompatible
    a = reg.resolve_assets(sd.name)
    assert a.get_asset("glutamate", "summary_format") == "giant_experiment_summary_h5"
    with pytest.raises(FileNotFoundError) as exc:
        load_glutamate_summary(a)  # incompatible modern, no legacy
    assert "attempts" in str(exc.value) and "missing" in str(exc.value)
