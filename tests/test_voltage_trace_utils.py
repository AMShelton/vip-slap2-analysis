import numpy as np

from vip_slap2_analysis.voltage import trace_utils as tu


def test_maybe_get():
    class A:
        x = 3

    assert tu.maybe_get(A(), "x") == 3
    assert tu.maybe_get({"y": 4}, "y") == 4
    assert tu.maybe_get({}, "z", 9) == 9


def test_interp_nans_1d_f32():
    out = tu.interp_nans_1d_f32([1, np.nan, 3, np.inf, 5])
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1, 2, 3, 4, 5])
    assert (tu.interp_nans_1d_f32([np.nan, 1, np.nan]) == 0).all()
    y = np.array([1, 2], dtype=np.float32)
    assert tu.interp_nans_1d_f32(y) is y


def test_interp_nans_or_nan():
    np.testing.assert_allclose(tu.interp_nans_or_nan([0, np.nan, 2]), [0, 1, 2])
    assert np.isnan(tu.interp_nans_or_nan([np.nan, 1, np.nan])).all()


def test_robust_std_matches_gaussian_sigma():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, 20000)
    assert abs(tu.robust_std(x) - 2.0) < 0.1
    assert np.isnan(tu.robust_std([np.nan]))
    assert tu.robust_std(np.r_[x, np.nan, 1e6]) < 2.2


def test_standardize_rows():
    X = np.array([[1, 2, 3], [5, 5, 5], [np.nan, 1, 3]], dtype=float)
    Z = tu.standardize_rows(X)
    assert Z.dtype == np.float32 and np.isfinite(Z).all()
    assert abs(Z[0].mean()) < 1e-6 and abs(Z[0].std() - 1) < 1e-5
    assert (Z[1] == 0).all()


def test_block_average_trace():
    t = np.arange(100) / 100.0
    x = np.arange(100, dtype=float)
    xb, tb = tu.block_average_trace(x, t, 25)
    assert len(xb) == 25 and len(tb) == 25
    assert xb[0] == 1.5
    xs, ts = tu.block_average_trace(x, t, None)
    assert len(xs) == 100
    xs, ts = tu.block_average_trace(x, t, 1000)
    assert len(xs) == 100


def test_find_timebase_segments():
    fs = 100.0
    t = np.r_[np.arange(300) / fs, 10 + np.arange(300) / fs, 20 + np.arange(50) / fs]
    seg, audit = tu.find_timebase_segments(t, fs, min_segment_sec=2.0)
    assert seg.tolist() == [[0, 300], [300, 600]]  # 0.5 s tail dropped
    assert audit["n_detected_gaps"] == 2 and audit["n_kept_segments"] == 2
