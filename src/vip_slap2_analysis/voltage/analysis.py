import numpy as np
from pathlib import Path
from vip_slap2_analysis.plotting.plot_session_heatmap import (
    IM_COLORS,
    DEFAULT_X_TICK_PARAMS,
    DEFAULT_Y_TICK_PARAMS,
    _merge_kwargs,
    _robust_row_zscore,
    _fill_nan_rowwise,
    _smooth_rows,
    _compute_dt,
    _safe_percentiles,
    _build_image_color_map,
    load_stimulus_events,
    load_running_speed,
    build_stimulus_locked_feature_mats,
    compute_sort_orders,
    build_pc1_trace_for_session,
)

def resolve_voltage_mean_npz(asset, trace_variant="dff_robust_f0_trial", mean_npz=None):
    if mean_npz is not None:
        path = Path(mean_npz)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    path = Path(asset.derived_dir) / "voltage" / f"voltage_mean_{trace_variant}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Mean voltage NPZ not found: {path}\n"
            "Run voltage extraction with write_sequence/write_single_trials as needed, or point mean_npz at an existing file."
        )
    return path


def load_voltage_mean_npz(asset, trace_variant="dff_robust_f0_trial", mean_npz=None):
    path = resolve_voltage_mean_npz(asset, trace_variant=trace_variant, mean_npz=mean_npz)
    pkg = np.load(path, allow_pickle=True)["data"][0]
    return pkg, path


def _coerce_event_mat_timebase(mat, t, *, context="event response"):
    """Return a 2-D ROI-by-time matrix and 1-D timebase with matching time lengths.

    Existing voltage_mean_*.npz files can have a one-sample mismatch because the
    extraction summary uses rounded sample counts while its saved time vector was
    generated with np.arange over floating-point endpoints. For plotting, the
    safest behavior is to preserve the event matrix and trim the longer axis.
    """
    mat = np.asarray(mat, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)

    if mat.ndim != 2:
        raise ValueError(f"Expected a 2-D ROI-by-time matrix for {context}; got shape {mat.shape}")
    if t.ndim != 1 or t.size == 0:
        raise ValueError(f"Expected a non-empty 1-D timebase for {context}; got shape {t.shape}")

    n_mat = int(mat.shape[1])
    n_t = int(t.size)
    if n_mat == n_t:
        return mat, t

    n = min(n_mat, n_t)
    # print(
    #     f"Warning: {context} has {n_mat} matrix time samples but {n_t} timebase samples; "
    #     f"trimming both to {n}."
    # )
    return mat[:, :n], t[:n]


def _baseline_subtract_mat(mat, t, baseline_window=(-0.25, 0.0)):
    mat, t = _coerce_event_mat_timebase(mat, t, context="baseline subtraction")
    mask = (t >= baseline_window[0]) & (t < baseline_window[1])
    if not np.any(mask):
        print(f"Warning: no baseline samples found in baseline_window={baseline_window}; skipping baseline subtraction.")
        return mat
    baseline = np.nanmean(mat[:, mask], axis=1, keepdims=True)
    return mat - baseline