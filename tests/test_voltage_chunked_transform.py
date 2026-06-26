import numpy as np

from vip_slap2_analysis.common.alignment import ReconstructedTraceBundle
from vip_slap2_analysis.voltage.extraction import (
    _transform_voltage_bundle,
    transform_voltage_signal,
)


def _bundle(raw, sample_rate_hz=1000.0):
    n = raw.shape[1]
    return ReconstructedTraceBundle(
        traces=raw,
        timebase_sec=np.arange(n, dtype=float) / sample_rate_hz,
        trial_valid_mask=np.array([True]),
        trial_lengths_samples=np.array([n]),
        trial_starts_sec=np.array([0.0]),
        session_start_sec=0.0,
        session_end_sec=n / sample_rate_hz,
        reconstructed_duration_sec=n / sample_rate_hz,
        metadata={},
    )


def test_chunked_static_transform_matches_full_transform():
    rng = np.random.default_rng(1)
    raw = rng.normal(100.0, 2.0, size=(4, 5000)).astype(np.float32)
    bundle = _bundle(raw)

    out, meta, payload = _transform_voltage_bundle(
        bundle,
        trace_signal="dff_static_f0",
        sample_rate_hz=1000.0,
        f0_percentile=50.0,
    )
    expected = transform_voltage_signal(raw, sample_rate_hz=1000.0, method="static", percentile=50.0)

    assert meta["chunked_transform"] is True
    assert payload is not None and "f0_model" in payload
    np.testing.assert_allclose(out.traces, expected["dff"], rtol=1e-6, atol=1e-6)


def test_chunked_robust_transform_is_finite_and_shape_stable():
    rng = np.random.default_rng(2)
    raw = rng.normal(100.0, 2.0, size=(3, 6000)).astype(np.float32)
    bundle = _bundle(raw)

    out, meta, payload = _transform_voltage_bundle(
        bundle,
        trace_signal="dff_robust_f0",
        sample_rate_hz=1000.0,
        f0_percentile=50.0,
        robust_f0_bin_sec=1.0,
        robust_f0_smooth_sec=3.0,
    )

    assert out.traces.shape == raw.shape
    assert np.isfinite(out.traces).all()
    assert meta["f0_method"] == "robust_binned_percentile_moving_median"
    assert payload is not None and payload["f0_model"]["f0_bin_values"].shape[0] == raw.shape[0]
