"""Ground-truth end-to-end test: SLAP2-first session, DI3 registration, extraction.

A synthetic acquisition is built where every frame's value encodes its own source
frame index, so the frame that lands at a stimulus onset can be read back. HARP
starts after the SLAP2 acquisition (missing the first ``K`` DI3 pulses) and stops
after it (tail anchor present). Without registration the historical code places
frame 0 at the first *recorded* pulse and is late by ``K`` cycles; with the
registration the correct frame is selected and the timebase follows the pulses.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_slap2 import (
    LINE_RATE_HZ, write_dat_chunks, write_giant_experiment_summary, write_giant_trial_table, write_harp_analog,
    write_harp_di3, write_meta_file, write_voltex_summary, write_voltex_trial_table,
)
from vip_slap2_analysis.common.cycle_registration import (
    STATUS_AMBIGUOUS, STATUS_IDENTIFIED, load_cycle_registration, run_cycle_registration,
)
from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.glutamate.extraction import process_glutamate_extraction
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
from vip_slap2_analysis.voltage.extraction import process_voltage_extraction

LPC = 227
N_CYCLES = 3000
PERIOD = LPC / LINE_RATE_HZ
PPM = 6.0
K_MISSING = 120                   # SLAP2 cycles acquired before HARP started recording
HARP_T0 = 5000.0                  # HARP time of source cycle 0 (never observed)
FS_FRAMES = 200.0


def _harp_time_of_line(line_1based: np.ndarray) -> np.ndarray:
    cyc = (np.asarray(line_1based) - 1) // LPC
    within = (np.asarray(line_1based) - 1) % LPC
    return HARP_T0 + cyc * PERIOD * (1 + PPM * 1e-6) + within / LINE_RATE_HZ * (1 + PPM * 1e-6)


def _build_session(tmp_path: Path, *, with_tail: bool = True):
    sid = "900777_2026-08-04_08-12-47"
    sd = tmp_path / sid
    dd = sd / "slap2" / "dynamic_data"
    write_dat_chunks(dd, prefix="acquisition_081247", dmd=1, n_cycles=N_CYCLES, lines_per_cycle=LPC, reference_timestamp=777_000)
    write_meta_file(dd, prefix="acquisition_081247", dmd=1, n_cycles_by_dmd=[N_CYCLES, N_CYCLES], lines_per_cycle=LPC)

    cycle_t = _harp_time_of_line(np.arange(N_CYCLES) * LPC + 1)
    harp_dir = sd / "behavior" / "VCO1_Behavior.harp"; harp_dir.mkdir(parents=True)
    record_start = cycle_t[K_MISSING] - 0.004
    record_end = cycle_t[-1] + (30.0 if with_tail else 0.002)
    observed = cycle_t[K_MISSING:]
    write_harp_di3(harp_dir / "Behavior_32.bin", observed, record_start_s=record_start)
    write_harp_analog(harp_dir / "Behavior_44.bin", record_start_s=record_start, record_end_s=record_end, rate_hz=200.0)
    origin = record_start  # behavior time zero = first analog sample

    # behavior QC outputs the pipeline would have produced
    qc = sd / "analysis" / "qc"; (qc / "behavior").mkdir(parents=True)
    derived = sd / "analysis" / "derived"; derived.mkdir(parents=True)
    ep = pd.DataFrame([{"epoch_index": 1, "start_idx": 0, "end_idx": 1, "start_time": observed[0] - origin,
                        "end_time": observed[-1] - origin, "duration_s": observed[-1] - observed[0], "n_pulses": observed.size}])
    ep.to_csv(qc / "behavior" / "imaging_epochs.csv", index=False)

    # stimuli: image onsets at known source lines (HARP time -> behavior-relative)
    stim_lines = np.array([800 * LPC + 10, 1500 * LPC + 100, 2200 * LPC + 50])
    stim_t = _harp_time_of_line(stim_lines) - origin
    rows = [{"Frame": 0, "Timestamp": 0.0, "Value": "Frame", "corrected_timestamps": 0.0}]
    for i, t in enumerate(stim_t):
        rows.append({"Frame": i + 1, "Timestamp": t, "Value": "im_A.tif", "corrected_timestamps": t})
    pd.DataFrame(rows).to_csv(harp_dir / "bonsai_event_log.csv", index=False)

    # GIAnT summary: frames every dt lines, value == frame index
    dt = LINE_RATE_HZ / FS_FRAMES
    frame_lines = np.ceil(np.arange(1, N_CYCLES * LPC, dt)).astype(int)
    split = frame_lines.size // 2
    trials = [frame_lines[:split], frame_lines[split:]]
    values = [np.arange(0, split, dtype=float)[:, None] * np.ones((1, 2)), np.arange(split, frame_lines.size, dtype=float)[:, None] * np.ones((1, 2))]
    proc = sd / "processed"
    (proc / "source_extraction" / "summarize_LoCo").mkdir(parents=True)
    write_giant_experiment_summary(proc / "source_extraction" / "summarize_LoCo" / "experiment_summary.h5",
                                   trial_frames=[len(t) for t in trials], n_sources=2, frame_lines_by_trial=trials,
                                   dF_by_trial=values, analyze_hz=FS_FRAMES)
    write_giant_trial_table(proc / "trial_table.h5", epoch_by_trial=[1, 1], first_line=[1, trials[1][0]], last_line=[trials[1][0], N_CYCLES * LPC])

    # Voltex summary: samples are lines, value == 1-based line index
    vol_lines = [N_CYCLES * LPC // 2, N_CYCLES * LPC - N_CYCLES * LPC // 2]
    raw = [np.arange(1, vol_lines[0] + 1, dtype=np.float32)[:, None] * np.ones((1, 1), np.float32),
           np.arange(vol_lines[0] + 1, N_CYCLES * LPC + 1, dtype=np.float32)[:, None] * np.ones((1, 1), np.float32)]
    (proc / "Voltage").mkdir()
    write_voltex_summary(proc / "Voltage" / "VoltageSummary.h5", trial_lines=vol_lines, trial_epoch=[1, 1], n_rois=1, raw_by_trial=raw)

    df = pd.DataFrame([{"session_id": sid, "subject_id": 900777, "session_dir": sd, "session_type": "A0",
                        "paradigm": "change_detection_passive", "indicator1": "iGluSnFR4f", "indicator2": "ASAP7y"}])
    reg = VIPSessionRegistry(summary_xlsx=sd / "x.xlsx", subjects_df=pd.DataFrame(), sessions_df=df)
    asset = reg.resolve_assets(sid)
    asset.ensure_dirs()
    return asset, dict(origin=origin, stim_lines=stim_lines, stim_t=stim_t, frame_lines=frame_lines, cycle_t=cycle_t, dd=dd)


def _image_center(single_npz: Path, *, dmd: str, pre: float, fs: float) -> np.ndarray:
    pkg = np.load(single_npz, allow_pickle=True)["data"][0]
    arr = pkg[dmd]["image_identity"]["im_A.tif"]       # (events, synapses, time)
    n_pre = int(round(pre * fs))
    return arr[:, 0, n_pre]


def test_registration_identifies_offset_from_raw_headers(tmp_path):
    asset, truth = _build_session(tmp_path)
    reg = run_cycle_registration(asset, dmd=1, dynamic_data_dir=truth["dd"])
    assert reg.status == STATUS_IDENTIFIED and reg.acquisition_start_order == "slap2_first_or_simultaneous"
    e = reg.epochs[0]
    assert e.anchor == "tail" and e.cycle_index_offset == K_MISSING
    assert abs(e.period_error_ppm - PPM) < 0.5
    assert reg.behavior_time_origin_harp_s == pytest.approx(truth["origin"], abs=1e-3)
    assert load_cycle_registration(asset.qc_dir, 1).epochs[0].cycle_index_offset == K_MISSING


def test_glutamate_extraction_selects_true_frame_only_with_registration(tmp_path):
    asset, truth = _build_session(tmp_path)
    fs = FS_FRAMES
    pre = 0.25
    meta = {"im_rate_hz": fs}
    # historical path: no registration -> frame 0 placed at the first recorded pulse
    out0 = process_glutamate_extraction(asset, metadata=meta, use_synapse_qc=False, overwrite=True,
                                        trace_signal="dF", trace_mode="ls", cycle_registration=None)
    assert out0["status"] == "ok"
    legacy_centers = _image_center(Path(out0["single_npz"]), dmd="DMD1", pre=pre, fs=fs)
    # registered path
    run_cycle_registration(asset, dmd=1, dynamic_data_dir=truth["dd"])
    out1 = process_glutamate_extraction(asset, metadata=meta, use_synapse_qc=False, overwrite=True,
                                        trace_signal="dF", trace_mode="ls", cycle_registration="auto",
                                        require_identified_cycle_alignment=True)
    centers = _image_center(Path(out1["single_npz"]), dmd="DMD1", pre=pre, fs=fs)
    expected = np.array([int(np.argmin(np.abs(truth["frame_lines"] - L))) for L in truth["stim_lines"]], dtype=float)
    assert centers.size == 3
    np.testing.assert_array_equal(centers, expected)                      # exact frame recovered
    assert np.all(np.abs(legacy_centers - expected) >= K_MISSING * PERIOD * fs * 0.9)  # legacy off by ~K cycles
    import json
    qc = json.loads(Path(out1["qc_json"]).read_text())
    assert qc["summary_format"] == "giant_experiment_summary_h5"
    assert qc["cycle_registration"]["DMD1"]["status"] == "identified"
    assert qc["per_dmd"]["DMD1"]["cycle_registration"]["applied"] is True
    assert qc["per_dmd"]["DMD1"]["timebase_method_by_epoch"] == {"1": "di3_pulse_registered"}
    assert qc["per_dmd"]["DMD1"]["cycle_registration"]["leading_samples_by_source_epoch"]["1"] > 0


def test_glutamate_extraction_output_subdir_leaves_existing_outputs(tmp_path):
    asset, truth = _build_session(tmp_path)
    meta = {"im_rate_hz": FS_FRAMES}
    kw = dict(metadata=meta, use_synapse_qc=False, trace_signal="dF", trace_mode="ls", cycle_registration=None)
    base = process_glutamate_extraction(asset, overwrite=True, **kw)
    sub = process_glutamate_extraction(asset, overwrite=False, output_subdir="reextract_test", **kw)
    assert sub["status"] == "ok"
    assert Path(sub["single_npz"]).parent.name == "reextract_test"
    assert Path(sub["single_npz"]).exists() and Path(base["single_npz"]).exists()
    assert Path(sub["single_npz"]) != Path(base["single_npz"])


def test_voltage_extraction_selects_true_line_only_with_registration(tmp_path):
    asset, truth = _build_session(tmp_path)
    run_cycle_registration(asset, dmd=1, dynamic_data_dir=truth["dd"])
    out = process_voltage_extraction(asset, trace_signal="raw", overwrite=True, write_single_trials=True,
                                     write_sequence=False, write_session_traces=True, use_roi_qc=False,
                                     cycle_registration="auto", require_identified_cycle_alignment=True,
                                     compression=None, compression_opts=None)
    assert out["status"] == "ok"
    import h5py
    with h5py.File(out["single_h5"], "r") as f:
        grp = f["DMD1/image_identity"]
        arr = grp["image_0000/traces"][()]
        assert grp["image_0000"].attrs["image_name"] == "im_A.tif"
    pre_samples = int(round(0.25 * LINE_RATE_HZ))
    centers = arr[:, 0, pre_samples]
    # value == 1-based source line; nearest-sample selection must land on the stimulus line (+-1 line)
    assert np.all(np.abs(centers - truth["stim_lines"]) <= 1), (centers, truth["stim_lines"])
    with h5py.File(out["session_traces_h5"], "r") as f:
        assert "sample_time_label" in f["DMD1"]
        tb = f["DMD1/timebase_sec"][()]
        # measured timebase: first retained sample is the first recorded pulse (behavior-relative)
        assert tb[0] == pytest.approx(truth["cycle_t"][K_MISSING] - truth["origin"], abs=1e-6)


def test_require_identified_alignment_refuses_ambiguous_session(tmp_path):
    asset, truth = _build_session(tmp_path, with_tail=False)  # HARP starts late AND stops at the last pulse
    reg = run_cycle_registration(asset, dmd=1, dynamic_data_dir=truth["dd"])
    assert reg.status == STATUS_AMBIGUOUS and reg.epochs[0].candidate_offset_range == (0, K_MISSING)
    with pytest.raises(RuntimeError, match="ambiguous"):
        process_glutamate_extraction(asset, metadata={"im_rate_hz": FS_FRAMES}, use_synapse_qc=False, overwrite=True,
                                     cycle_registration="auto", require_identified_cycle_alignment=True)
    with pytest.raises(RuntimeError, match="ambiguous"):
        process_voltage_extraction(asset, trace_signal="raw", overwrite=True, write_sequence=False,
                                   cycle_registration="auto", require_identified_cycle_alignment=True)
    # without the requirement the historical behavior is kept and the refusal reason is recorded
    out = process_glutamate_extraction(asset, metadata={"im_rate_hz": FS_FRAMES}, use_synapse_qc=False, overwrite=True,
                                       cycle_registration="auto", require_identified_cycle_alignment=False)
    import json
    qc = json.loads(Path(out["qc_json"]).read_text())
    assert qc["per_dmd"]["DMD1"]["cycle_registration"]["applied"] is False
    assert "ambiguous" in qc["per_dmd"]["DMD1"]["cycle_registration"]["reason"]


def test_voltage_polarity_crosscheck_and_trace_kind_guard(tmp_path):
    """Pipeline recomputes dF/F from raw_f with the registry indicator; Voltex's stored polarity must agree."""
    import json
    from vip_slap2_analysis.voltage.extraction import crosscheck_voltex_polarity, voltex_dff_polarity
    from vip_slap2_analysis.voltage.postprocess import load_voltage_summary_with_provenance
    asset, truth = _build_session(tmp_path)
    run_cycle_registration(asset, dmd=1, dynamic_data_dir=truth["dd"])
    # the synthetic Voltex file declares indicator_direction='decreases' (ASAP7-like)
    assert voltex_dff_polarity({"indicator_direction": "decreases", "indicator_name": "ASAP7y"})["dff_sign"] == -1
    assert voltex_dff_polarity({"indicator_direction": "auto", "indicator_name": "ASAP8"})["dff_sign"] == 1
    assert voltex_dff_polarity({"indicator_direction": "auto", "indicator_name": ""})["dff_sign"] is None
    vs, _ = load_voltage_summary_with_provenance(asset, keep_open=True)
    with vs:
        chk = crosscheck_voltex_polarity(vs, asset, voltage_indicator="ASAP7y")
        assert chk["checked"] and chk["consistent"] and chk["by_dmd"]["1"]["pipeline"]["dff_sign"] == -1
        with pytest.raises(RuntimeError, match="polarity disagrees"):
            crosscheck_voltex_polarity(vs, asset, voltage_indicator="ASAP8")
    # extraction records the check and refuses to run on Voltex's stored dff
    out = process_voltage_extraction(asset, trace_signal="raw", overwrite=True, write_sequence=False, write_single_trials=False,
                                     use_roi_qc=False, cycle_registration="auto", voltage_indicator="ASAP7y",
                                     compression=None, compression_opts=None)
    qc = json.loads(Path(out["qc_json"]).read_text())
    assert qc["voltex_polarity_crosscheck"]["consistent"] is True
    with pytest.raises(NotImplementedError, match="raw_f"):
        process_voltage_extraction(asset, trace_signal="raw", overwrite=True, write_sequence=False, voltex_trace_kind="dff",
                                   voltage_indicator="ASAP7y")


def test_voltage_extraction_uses_voltex_product_dff_by_default(tmp_path):
    """trace_source='product': Voltex's stored dff/f0 are the extracted signal; nothing is re-baselined."""
    import json
    import h5py
    from vip_slap2_analysis.voltage.postprocess import load_voltage_summary_from_asset
    asset, truth = _build_session(tmp_path)
    run_cycle_registration(asset, dmd=1, dynamic_data_dir=truth["dd"])
    common = dict(trace_signal="dff_robust_f0", overwrite=True, write_single_trials=False, write_sequence=False,
                  write_session_traces=True, use_roi_qc=False, cycle_registration="auto", voltage_indicator="ASAP7y",
                  compression=None, compression_opts=None)
    out = process_voltage_extraction(asset, **common)                       # default trace_source="product"
    assert out["status"] == "ok"
    assert Path(out["session_traces_h5"]).name == "voltage_session_traces_voltex_dff_trial.h5"
    qc = json.loads(Path(out["qc_json"]).read_text())
    assert qc["trace_source_requested"] == "product" and qc["trace_source_effective"] == "product"
    assert qc["trace_signal"] == "voltex_dff" and qc["trace_signal_requested"] == "dff_robust_f0"
    assert qc["preprocessor_params"] is not None
    # the session dff equals Voltex's stored dff (synthetic: (raw-100)/100), f0 equals Voltex's f0 (100)
    with h5py.File(out["session_traces_h5"], "r") as f:
        raw = f["DMD1/raw_f"][()]; dff = f["DMD1/dff"][()]; f0 = f["DMD1/f0"][()]
        meta = json.loads(f["DMD1"].attrs["signal_transform_json"])
    assert meta["trace_source"] == "product" and meta["transform"] == "preprocessor" and meta["dff_sign"] == -1
    ok = np.isfinite(raw) & np.isfinite(dff)
    assert np.allclose(dff[ok], (raw[ok] - 100.0) / 100.0, atol=1e-5) and np.allclose(f0[np.isfinite(f0)], 100.0)

    # explicit recompute keeps the historical products and names
    out2 = process_voltage_extraction(asset, trace_source="recompute", **common)
    assert Path(out2["session_traces_h5"]).name == "voltage_session_traces_dff_robust_f0_trial.h5"
    qc2 = json.loads(Path(out2["qc_json"]).read_text())
    assert qc2["trace_source_effective"] == "recompute" and qc2["trace_signal"] == "dff_robust_f0"
    with h5py.File(out2["session_traces_h5"], "r") as f:
        meta2 = json.loads(f["DMD1"].attrs["signal_transform_json"])
    assert meta2["trace_source"] == "recompute" and meta2["dff_sign"] == -1

    # on-the-fly inspection can read the product too
    from vip_slap2_analysis.voltage.extraction import compute_voltage_roi_transform_from_asset
    roi = compute_voltage_roi_transform_from_asset(asset, dmd=1, roi_index=0, cycle_registration="auto",
                                                   voltage_indicator="ASAP7y", trace_source="product")
    assert roi["metadata"]["trace_source"] == "product" and np.allclose(roi["f0"][np.isfinite(roi["f0"])], 100.0)
    # raw output is unaffected by trace_source
    out3 = process_voltage_extraction(asset, trace_signal="raw", overwrite=True, write_sequence=False, write_single_trials=False,
                                      use_roi_qc=False, cycle_registration="auto", voltage_indicator="ASAP7y",
                                      compression=None, compression_opts=None)
    assert json.loads(Path(out3["qc_json"]).read_text())["trace_source_effective"] == "n/a"
