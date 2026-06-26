import numpy as np
import pandas as pd

from vip_slap2_analysis.behavior.epochs import (
    detect_epochs_adaptive,
    epochs_to_dataframe,
    shift_epochs_to_photodiode_time,
)
from vip_slap2_analysis.behavior.validation import audit_event_coverage
from vip_slap2_analysis.packaging.stimulus_events import extract_stimulus_events_from_bonsai


def _make_pulse_harp_df(abs_start=1000.0, start_s=6.0, stop_s=20.0, dt=0.5):
    rel_time = np.arange(0.0, 25.0 + dt, dt)
    # Make a simple pulse train during the imaging epoch. Gaps outside the pulse
    # train are large enough to delimit one epoch.
    di3 = (rel_time >= start_s) & (rel_time <= stop_s)
    di3 &= (np.arange(rel_time.size) % 2 == 0)
    return pd.DataFrame({"DI3": di3, "time": abs_start + rel_time}), rel_time


def test_pulse_train_epochs_are_photodiode_relative_when_harp_time_is_absolute():
    harp_df, acq_time = _make_pulse_harp_df(abs_start=192612.531488)
    photodiode_df = pd.DataFrame(
        {"AnalogInput0": np.zeros(10)},
        index=np.linspace(192612.530976, 192612.539976, 10),
    )

    epochs, _ = detect_epochs_adaptive(
        harp_df,
        acq_time,
        acq_type="continuous",
        min_duration=2.0,
        gap_start=0.02,
        epoch_detection_method="pulse_train",
        pulse_gap_factor=10.0,
        pulse_min_gap_s=1.5,
        pulse_min_pulses=3,
        pulse_min_duration=2.0,
    )
    epoch_df = epochs_to_dataframe(shift_epochs_to_photodiode_time(epochs, harp_df, photodiode_df))

    assert len(epoch_df) == 1
    # Regression guard: this must not be in the absolute 192k-second HARP clock.
    assert 5.0 < float(epoch_df.loc[0, "start_time"]) < 7.0
    assert 19.0 < float(epoch_df.loc[0, "end_time"]) < 21.0

    stim_df = pd.DataFrame(
        {
            "Frame": [1, 2, 3],
            "Timestamp": [6.5, 10.0, 12.0],
            "Value": ["stimuli/images_A/a.tiff", "ChangeFlash", "Omission"],
            "corrected_timestamps": [6.5, 10.0, 12.0],
        }
    )
    assert audit_event_coverage(stim_df, epoch_df) == {
        "image_identity_total": 1,
        "image_identity_in_epochs": 1,
        "change_total": 1,
        "change_in_epochs": 1,
        "omission_total": 1,
        "omission_in_epochs": 1,
    }


def test_stimulus_event_packaging_reports_changeflash_times():
    event_log = pd.DataFrame(
        {
            "Value": ["stimuli/images_A/a.tiff", "ChangeFlash", "Omission"],
            "corrected_timestamps": [1.0, 2.0, 3.0],
        }
    )
    events = extract_stimulus_events_from_bonsai(event_log)
    assert events["change_times_s"] == [2.0]
    assert events["special_event_times_s"]["ChangeFlash"] == [2.0]
