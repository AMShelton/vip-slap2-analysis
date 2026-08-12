import numpy as np
import pandas as pd
import pytest

from vip_slap2_analysis.behavior.epochs import detect_pulse_train_epochs
from vip_slap2_analysis.common.epoch_alignment import (
    classify_epochs_by_duration,
    reconcile_trial_epochs,
)


def test_duration_qc_rejects_short_source_fragments_and_clips_tail():
    behavior = pd.DataFrame(
        {
            "epoch_index": [1],
            "start_time": [10.0],
            "end_time": [45.0],
        }
    )
    # Source epoch 1 is 40 s; epochs 2 and 3 are aborted 5/7 s fragments.
    lengths = np.array([2000, 2000, 500, 700])
    labels = np.array([1, 1, 2, 3])
    rec = reconcile_trial_epochs(
        lengths,
        sample_rate_hz=100.0,
        behavior_epoch_df=behavior,
        source_trial_epoch=labels,
        min_epoch_duration_sec=30.0,
        strict_epoch_match=True,
    )

    assert rec.trial_lengths_samples.tolist() == [2000, 1500, 0, 0]
    assert rec.analysis_trial_epoch.tolist() == [1, 1, 0, 0]
    assert rec.trial_keep_mask.tolist() == [True, True, False, False]
    qc = rec.source_epoch_qc.set_index("source_epoch_index")
    assert bool(qc.loc[1, "accepted_by_duration"])
    assert qc.loc[1, "discard_reason"] == "trimmed_to_behavior_epoch"
    assert not bool(qc.loc[2, "accepted_by_duration"])
    assert not bool(qc.loc[3, "accepted_by_duration"])


def test_accepted_epoch_count_mismatch_remains_fatal():
    behavior = pd.DataFrame(
        {"epoch_index": [1], "start_time": [0.0], "end_time": [40.0]}
    )
    with pytest.raises(ValueError, match="Accepted physiology acquisition epochs"):
        reconcile_trial_epochs(
            [3500, 3500],
            sample_rate_hz=100.0,
            behavior_epoch_df=behavior,
            source_trial_epoch=[1, 2],
            min_epoch_duration_sec=30.0,
            strict_epoch_match=True,
        )


def test_behavior_epoch_classification_uses_inclusive_30_second_threshold():
    df = pd.DataFrame(
        {
            "epoch_index": [1, 2, 3],
            "start_time": [0.0, 40.0, 80.0],
            "end_time": [29.999, 70.0, 111.0],
        }
    )
    qc = classify_epochs_by_duration(df, min_duration_sec=30.0)
    assert qc["accepted"].tolist() == [False, True, True]
    assert qc["analysis_epoch_index"].tolist() == [0, 1, 2]


def test_pulse_train_diagnostics_preserve_rejected_candidates():
    # Three 10-Hz pulse blocks: 40 s, 5 s, and 31 s.
    blocks = [(0.0, 40.0), (45.0, 50.0), (60.0, 91.0)]
    edge_times = np.concatenate(
        [np.arange(start, stop, 0.1) for start, stop in blocks]
    )
    time = np.arange(0.0, 92.0, 0.01)
    signal = np.zeros(time.size, dtype=bool)
    for edge in edge_times:
        idx = int(np.searchsorted(time, edge))
        signal[idx : min(idx + 2, signal.size)] = True

    epochs, diag = detect_pulse_train_epochs(
        signal,
        time,
        gap_factor=5.0,
        min_gap_s=0.5,
        min_duration=30.0,
        min_pulses=10,
    )
    assert len(epochs) == 2
    assert diag["n_candidate_epochs"] == 3
    assert diag["n_epochs_rejected"] == 1
    candidates = diag["candidate_epochs"]
    assert [row["accepted"] for row in candidates] == [True, False, True]
    assert candidates[1]["discard_reason"] == "duration_below_minimum"


def test_source_acquisition_duration_metadata_controls_acceptance():
    behavior = pd.DataFrame(
        {"epoch_index": [1], "start_time": [0.0], "end_time": [35.0]}
    )
    # Only 20 s of trial-covered samples are present, but the raw acquisition
    # lasted 40 s. Acquisition metadata should therefore keep this source epoch.
    rec = reconcile_trial_epochs(
        [1000, 1000],
        sample_rate_hz=100.0,
        behavior_epoch_df=behavior,
        source_trial_epoch=[1, 1],
        source_epoch_durations_sec={1: 40.0},
        min_epoch_duration_sec=30.0,
        strict_epoch_match=True,
    )
    row = rec.source_epoch_qc.iloc[0]
    assert bool(row["accepted_by_duration"])
    assert row["source_duration_basis"] == "acquisition_metadata"
    assert np.isclose(row["source_duration_s"], 40.0)
    assert np.isclose(row["trial_covered_duration_s"], 20.0)


def test_short_acquisition_metadata_rejects_even_when_trial_coverage_is_long():
    behavior = pd.DataFrame(
        {"epoch_index": [1], "start_time": [0.0], "end_time": [40.0]}
    )
    # Two source epochs: source 1 is accepted; source 2 has a 5 s raw
    # acquisition duration despite an overestimated trial-length fallback.
    rec = reconcile_trial_epochs(
        [4000, 4000],
        sample_rate_hz=100.0,
        behavior_epoch_df=behavior,
        source_trial_epoch=[1, 2],
        source_epoch_durations_sec={1: 40.0, 2: 5.0},
        min_epoch_duration_sec=30.0,
        strict_epoch_match=True,
    )
    assert rec.analysis_trial_epoch.tolist() == [1, 0]
    assert rec.trial_lengths_samples.tolist() == [4000, 0]
