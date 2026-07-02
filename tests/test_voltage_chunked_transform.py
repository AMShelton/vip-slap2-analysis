import numpy as np

from vip_slap2_analysis.common.alignment import ReconstructedTraceBundle
from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.voltage.extraction import (
    _resolve_voltage_dff_polarity_for_dmd,
    _transform_voltage_bundle,
    compute_voltage_dff,
    resolve_voltage_dff_polarity,
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


def test_voltage_dff_polarity_resolves_asap7_and_asap8():
    raw = np.asarray([9.0, 10.0, 11.0], dtype=np.float32)
    f0 = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)

    asap7 = compute_voltage_dff(raw, f0, indicator="ASAP7y")
    asap8 = compute_voltage_dff(raw, f0, indicator="ASAP8")

    np.testing.assert_allclose(asap7, np.asarray([0.1, 0.0, -0.1], dtype=np.float32))
    np.testing.assert_allclose(asap8, np.asarray([-0.1, 0.0, 0.1], dtype=np.float32))

    assert resolve_voltage_dff_polarity("ASAP7y")["dff_sign"] == -1
    assert resolve_voltage_dff_polarity("ASAP8")["dff_sign"] == 1


def test_chunked_static_transform_matches_full_transform_for_asap8():
    rng = np.random.default_rng(3)
    raw = rng.normal(100.0, 2.0, size=(4, 5000)).astype(np.float32)
    bundle = _bundle(raw)

    out, meta, payload = _transform_voltage_bundle(
        bundle,
        trace_signal="dff_static_f0",
        sample_rate_hz=1000.0,
        f0_percentile=50.0,
        indicator="ASAP8",
    )
    expected = transform_voltage_signal(
        raw,
        sample_rate_hz=1000.0,
        method="static",
        percentile=50.0,
        indicator="ASAP8",
    )

    assert meta["dff_sign"] == 1
    assert meta["fluorescence_response_to_depolarization"] == "depolarization_increases_fluorescence"
    assert payload is not None and "f0_model" in payload
    np.testing.assert_allclose(out.traces, expected["dff"], rtol=1e-6, atol=1e-6)


def test_voltage_polarity_uses_dmd_specific_metadata_with_legacy_fallback(tmp_path):
    asset = SessionAssets(
        session_id="session",
        subject_id=1,
        session_dir=tmp_path,
        metadata={"indicator1": "ASAP7y", "indicator2": "ASAP8"},
    )

    dmd1 = _resolve_voltage_dff_polarity_for_dmd(asset, 1)
    dmd2 = _resolve_voltage_dff_polarity_for_dmd(asset, 2)
    override = _resolve_voltage_dff_polarity_for_dmd(asset, 1, dff_polarity="standard")

    assert dmd1["dff_sign"] == -1
    assert dmd1["indicator"] == "asap7y"
    assert dmd2["dff_sign"] == 1
    assert dmd2["indicator"] == "asap8"
    assert override["dff_sign"] == 1
