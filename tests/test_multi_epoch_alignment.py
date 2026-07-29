import numpy as np
import pandas as pd

from vip_slap2_analysis.common.alignment import ReconstructedTraceBundle, extract_trace_snippet
from vip_slap2_analysis.common.epoch_alignment import build_epoch_aware_timebase
from vip_slap2_analysis.voltage.extraction import _transform_voltage_bundle
from vip_slap2_analysis.calcium.extraction import _reconstruct_ca_session_traces


def _epochs():
    return pd.DataFrame(
        {
            "epoch_index": [1, 2],
            "start_time": [0.0, 3.0],
            "end_time": [1.0, 4.0],
        }
    )


def test_epoch_timebase_preserves_gap_and_safe_snippets():
    tb = build_epoch_aware_timebase(
        [100, 100],
        sample_rate_hz=100.0,
        epoch_df=_epochs(),
        trial_epoch=[1, 2],
        scale_each_epoch="auto",
    )
    assert np.isclose(tb.timebase_sec[99], 0.99)
    assert np.isclose(tb.timebase_sec[100], 3.0)
    assert np.all(tb.sample_epoch[:100] == 1)
    assert np.all(tb.sample_epoch[100:] == 2)
    assert np.isclose(tb.metadata["imaging_gap_duration_sec"], 2.0)

    traces = np.vstack([np.arange(200, dtype=float)])
    bundle = ReconstructedTraceBundle(
        traces=traces,
        timebase_sec=tb.timebase_sec,
        trial_valid_mask=np.ones(2, dtype=bool),
        trial_lengths_samples=np.array([100, 100]),
        trial_starts_sec=tb.trial_starts_sec,
        session_start_sec=0.0,
        session_end_sec=4.0,
        reconstructed_duration_sec=4.0,
        metadata=tb.metadata,
        sample_epoch=tb.sample_epoch,
    )
    snippet = extract_trace_snippet(
        bundle, 3.5, sample_rate_hz=100.0, pre_time=0.1, post_time=0.1
    )
    assert snippet is not None
    assert snippet.shape == (1, 20)
    assert snippet[0, 10] == 150

    # Window would straddle the acquisition restart, so it must be rejected.
    assert extract_trace_snippet(
        bundle, 0.99, sample_rate_hz=100.0, pre_time=0.1, post_time=0.1
    ) is None


def test_voltage_f0_is_fit_independently_per_epoch():
    raw = np.concatenate(
        [np.full((1, 1000), 100.0), np.full((1, 1000), 200.0)], axis=1
    ).astype(np.float32)
    sample_epoch = np.concatenate([np.ones(1000), np.full(1000, 2)]).astype(int)
    bundle = ReconstructedTraceBundle(
        traces=raw,
        timebase_sec=np.concatenate([np.arange(1000) / 1000.0, 2.0 + np.arange(1000) / 1000.0]),
        trial_valid_mask=np.ones(2, dtype=bool),
        trial_lengths_samples=np.array([1000, 1000]),
        trial_starts_sec=np.array([0.0, 2.0]),
        session_start_sec=0.0,
        session_end_sec=3.0,
        reconstructed_duration_sec=3.0,
        metadata={"trial_epoch": [1, 2]},
        sample_epoch=sample_epoch,
    )
    epoch_out, epoch_meta, _ = _transform_voltage_bundle(
        bundle,
        trace_signal="dff_static_f0",
        sample_rate_hz=1000.0,
        indicator="ASAP8",
        f0_scope="epoch",
    )
    session_out, _, _ = _transform_voltage_bundle(
        bundle,
        trace_signal="dff_static_f0",
        sample_rate_hz=1000.0,
        indicator="ASAP8",
        f0_scope="session",
    )
    assert np.nanmax(np.abs(epoch_out.traces)) < 1e-6
    assert abs(float(np.nanmean(session_out.traces[:, :1000]))) > 0.1
    assert abs(float(np.nanmean(session_out.traces[:, 1000:]))) > 0.1
    assert epoch_meta["f0_scope"] == "epoch"


class _FakeCaSummary:
    def get_processed_soma_ca_all_trials(self, **kwargs):
        return {
            "ca_mc": [np.full((1, 100), 100.0), np.full((1, 100), 200.0)],
            "dff": [np.zeros((1, 100)), np.zeros((1, 100))],
        }

    def _estimate_ca_baseline(self, x, fs_hz, **kwargs):
        return np.full_like(x, np.nanmedian(x), dtype=float)


def test_calcium_epochwise_dff_does_not_bridge_restart_offset():
    bundle = _reconstruct_ca_session_traces(
        _FakeCaSummary(),
        dmd=1,
        im_rate_hz=100.0,
        epoch_start_sec=0.0,
        epoch_end_sec=4.0,
        epoch_df=_epochs(),
        dff_scope="epoch",
        motion_correct=False,
        strict_epoch_match=True,
    )
    assert bundle["traces"].shape == (1, 200)
    assert np.nanmax(np.abs(bundle["traces"])) < 1e-8
    assert bundle["trial_epoch"].tolist() == [1, 2]
    assert np.all(bundle["sample_epoch"][:100] == 1)
    assert np.all(bundle["sample_epoch"][100:] == 2)
