from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from vip_slap2_analysis.glutamate.alignment import (
    extract_image_intervals,
    filter_ordered_images_to_epochs,
    load_corrected_bonsai_csv,
    load_imaging_epochs_csv,
)
from vip_slap2_analysis.glutamate.summary import GlutamateSummary


IM_COLORS = [
    "#c5cae9",
    "#ffcdd2",
    "#c8e6c9",
    "#ffe0b2",
    "#e1bee7",
    "#d7ccc8",
    "#9fd3f2",
]

DEFAULT_X_TICK_PARAMS = dict(axis="x", which="major", reset=True, top=False, labelsize=12)
DEFAULT_Y_TICK_PARAMS = dict(axis="y", which="major", reset=True, right=False, labelsize=12)


# -----------------------------------------------------------------------------
# small utilities
# -----------------------------------------------------------------------------


def _merge_kwargs(base: Optional[dict], override: Optional[dict]) -> dict:
    out = dict(base or {})
    out.update(override or {})
    return out



def _robust_row_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=1, keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(scale < eps, 1.0, scale)
    return (x - med) / scale



def _fill_nan_rowwise(x: np.ndarray, fill_value: str = "median") -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    for i in range(x.shape[0]):
        row = x[i]
        if fill_value == "median":
            val = np.nanmedian(row)
        elif fill_value == "mean":
            val = np.nanmean(row)
        elif fill_value == "zero":
            val = 0.0
        else:
            raise ValueError("fill_value must be 'median', 'mean', or 'zero'")
        if not np.isfinite(val):
            val = 0.0
        row[np.isnan(row)] = val
        x[i] = row
    return x



def _smooth_rows(x: np.ndarray, sigma_samples: float = 0) -> np.ndarray:
    if sigma_samples is None or sigma_samples <= 0:
        return x
    return gaussian_filter1d(x, sigma=sigma_samples, axis=1, mode="nearest")



def _compute_dt(tb: Iterable[float]) -> float:
    tb = np.asarray(tb, dtype=float)
    if tb.size >= 2:
        dt = float(np.nanmedian(np.diff(tb)))
        if np.isfinite(dt) and dt > 0:
            return dt
    return 1.0



def _safe_percentiles(x: np.ndarray, q: Tuple[float, float] = (2, 98)) -> Tuple[float, float]:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(vals, q)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)



def _clip_time_window(t: np.ndarray, xlim_sec: Optional[Tuple[float, float]]) -> slice:
    if xlim_sec is None:
        return slice(None)
    t0, t1 = xlim_sec
    mask = (t >= t0) & (t <= t1)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError(f"No data in requested xlim_sec={xlim_sec}")
    return slice(idx[0], idx[-1] + 1)



def _short_image_label(name: str, max_len: int = 18) -> str:
    base = os.path.basename(str(name))
    base = base.replace(".tiff", "").replace(".tif", "")
    base = base.replace("stimuliImages_", "")
    if len(base) > max_len:
        return base[: max_len - 1] + "…"
    return base



def _build_image_color_map(ordered_images, im_colors):
    unique = []
    seen = set()
    for evt in ordered_images:
        nm = evt.image_name
        if nm not in seen:
            unique.append(nm)
            seen.add(nm)
    return {img: im_colors[i % len(im_colors)] for i, img in enumerate(unique)}


# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------


def load_stimulus_events(asset) -> Dict[str, Any]:
    stim_df = load_corrected_bonsai_csv(asset.bonsai_event_log_csv)
    _, ordered_images = extract_image_intervals(stim_df)

    epoch_csv = Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv"
    epoch_df = load_imaging_epochs_csv(epoch_csv) if epoch_csv.exists() else None

    if epoch_df is not None:
        ordered_images = filter_ordered_images_to_epochs(
            ordered_images,
            epoch_df,
            pre_time=0.25,
            post_time=0.50,
        )
        session_start_sec = float(epoch_df.iloc[0]["start_time"])
    else:
        time_col = "corrected_timestamp" if "corrected_timestamp" in stim_df.columns else "corrected_timestamps"
        session_start_sec = float(stim_df[time_col].min())

    return {
        "ordered_images": ordered_images,
        "session_start_sec": session_start_sec,
    }

def load_running_speed(
    asset,
    session_start_sec: float,
    xlim_sec: Optional[Tuple[float, float]] = None,
    encoder_path: Optional[str] = None,
    encoder_col: str = "Encoder",
    wheel_radius_cm: float = 8.0,
    encoder_units: str = "ticks",
    speed_units: str = "cm/s",
    median_filter_kernel: int = 51,
    absolute_speed: bool = True,
    ticks_per_revolution: Optional[float] = None,
    smooth_speed_sigma: Optional[float] = 3.0,
    time_zero: str = "first_sample",   # NEW
    time_offset_sec: float = 0.0,      # NEW
) -> Dict[str, np.ndarray]:
    """
    Load encoder data and convert to running speed.

    Parameters
    ----------
    time_zero :
        How to zero the encoder time axis.
        - "first_sample": subtract the first encoder timestamp
        - "session_start": subtract session_start_sec
        - "none": keep raw encoder timestamps
    time_offset_sec :
        Additional offset applied after zeroing.
    """
    if encoder_path is None:
        if getattr(asset, "photodiode_pkl", None) is None:
            raise FileNotFoundError("asset.photodiode_pkl is missing, so encoder.pkl could not be inferred")
        encoder_path = str(Path(asset.photodiode_pkl).with_name("encoder.pkl"))

    encoder_path = Path(encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    df = pd.read_pickle(encoder_path)
    if encoder_col not in df.columns:
        raise KeyError(f"Encoder column '{encoder_col}' not found in {encoder_path}")

    raw_time = df.index.to_numpy(dtype=float)
    pos = df[encoder_col].to_numpy(dtype=float)

    if raw_time.size == 0:
        return {"time_sec": np.array([]), "speed": np.array([]), "path": str(encoder_path)}

    if time_zero == "first_sample":
        time_sec = raw_time - raw_time[0]
    elif time_zero == "session_start":
        time_sec = raw_time - float(session_start_sec)
    elif time_zero == "none":
        time_sec = raw_time.copy()
    else:
        raise ValueError("time_zero must be one of: 'first_sample', 'session_start', 'none'")

    time_sec = time_sec + float(time_offset_sec)

    kernel = int(median_filter_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        pos = medfilt(pos, kernel_size=kernel)

    dt = np.gradient(time_sec)
    dt[~np.isfinite(dt)] = np.nan
    dt[dt <= 0] = np.nan

    dpos = np.gradient(pos)

    if encoder_units == "degrees":
        dist_cm = np.deg2rad(dpos) * float(wheel_radius_cm)
    elif encoder_units == "radians":
        dist_cm = dpos * float(wheel_radius_cm)
    elif encoder_units == "cm":
        dist_cm = dpos
    elif encoder_units == "m":
        dist_cm = dpos * 100.0
    elif encoder_units == "ticks":
        if ticks_per_revolution is None:
            raise ValueError("ticks_per_revolution must be provided when encoder_units='ticks'")
        cm_per_tick = (2.0 * np.pi * float(wheel_radius_cm)) / float(ticks_per_revolution)
        dist_cm = dpos * cm_per_tick
    else:
        raise ValueError("encoder_units must be one of: 'ticks', 'degrees', 'radians', 'cm', 'm'")

    speed = dist_cm / dt

    if absolute_speed:
        speed = np.abs(speed)

    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    if smooth_speed_sigma is not None and smooth_speed_sigma > 0:
        speed = gaussian_filter1d(speed, sigma=float(smooth_speed_sigma))

    if speed_units == "m/s":
        speed = speed / 100.0
    elif speed_units != "cm/s":
        raise ValueError("speed_units must be 'cm/s' or 'm/s'")

    if xlim_sec is not None:
        mask = (time_sec >= xlim_sec[0]) & (time_sec <= xlim_sec[1])
        time_sec = time_sec[mask]
        speed = speed[mask]

    return {"time_sec": time_sec, "speed": speed, "path": str(encoder_path)}



# -----------------------------------------------------------------------------
# matrix construction
# -----------------------------------------------------------------------------


def build_session_glutamate_mats(
    asset,
    signal: str = "dF",
    mode: str = "ls",
    channels: str = "glutamate",
    normalize_rows: Optional[str] = "zscore",
):
    summary = GlutamateSummary(asset.summary_mat)

    dmd_mats = {}
    dmd_timebases = {}
    dmd_dt = {}

    for dmd in range(1, summary.n_dmds + 1):
        n_syn = summary.n_synapses[dmd - 1]
        if n_syn == 0:
            continue

        valid_trials = np.where(summary.keep_trials[dmd - 1])[0] + 1
        if len(valid_trials) == 0:
            continue

        ref_trial = int(valid_trials[0])
        ref_traces, _ = summary.get_traces(
            dmd=dmd,
            trial=ref_trial,
            signal=signal,
            mode=mode,
            channels=channels,
            squeeze_channels=True,
            return_frame_lines=True,
        )
        ref_traces = np.asarray(ref_traces, dtype=float)
        if ref_traces.ndim == 1:
            ref_traces = ref_traces[:, None]
        ref_len = ref_traces.shape[0]

        try:
            dt = _compute_dt(summary.timebase(dmd=dmd, trial=ref_trial))
        except Exception:
            dt = 1.0

        blocks = []
        tblocks = []
        t_cursor = 0.0

        for trial in range(1, summary.n_trials + 1):
            valid = bool(summary.keep_trials[dmd - 1, trial - 1])

            if valid:
                traces = summary.get_traces(
                    dmd=dmd,
                    trial=trial,
                    signal=signal,
                    mode=mode,
                    channels=channels,
                    squeeze_channels=True,
                )
                traces = np.asarray(traces, dtype=float)
                if traces.ndim == 1:
                    traces = traces[:, None]

                if traces.shape[1] != n_syn and traces.shape[0] == n_syn:
                    traces = traces.T

                if traces.shape[1] < n_syn:
                    pad = np.full((traces.shape[0], n_syn - traces.shape[1]), np.nan)
                    traces = np.concatenate([traces, pad], axis=1)
                elif traces.shape[1] > n_syn:
                    traces = traces[:, :n_syn]

                block = traces.T
                block_len = block.shape[1]
            else:
                block_len = ref_len
                block = np.full((n_syn, block_len), np.nan)

            blocks.append(block)
            tblocks.append(t_cursor + np.arange(block_len) * dt)
            t_cursor += block_len * dt

        mat = np.concatenate(blocks, axis=1)
        t = np.concatenate(tblocks)

        if normalize_rows == "zscore":
            mat = _robust_row_zscore(mat)

        dmd_mats[dmd] = mat
        dmd_timebases[dmd] = t
        dmd_dt[dmd] = dt

    return summary, dmd_mats, dmd_timebases, dmd_dt


# -----------------------------------------------------------------------------
# stimulus-locked feature matrices
# -----------------------------------------------------------------------------


def _extract_triggered_stack(session_mat, session_t, event_times_rel, t_pre, t_post, dt):
    n_syn, n_time = session_mat.shape
    n_pre = int(round(t_pre / dt))
    n_post = int(round(t_post / dt))
    rel_idx = np.arange(-n_pre, n_post)
    t_rel = rel_idx * dt

    stacks = []
    for et in event_times_rel:
        center = np.searchsorted(session_t, et)
        idx = center + rel_idx
        if idx[0] < 0 or idx[-1] >= n_time:
            continue
        stacks.append(session_mat[:, idx])

    if len(stacks) == 0:
        return np.empty((0, n_syn, len(t_rel))), t_rel

    return np.stack(stacks, axis=0), t_rel



def _baseline_subtract_stack(stack, t_rel, baseline_window=(-0.25, 0.0)):
    if stack.size == 0:
        return stack
    mask = (t_rel >= baseline_window[0]) & (t_rel < baseline_window[1])
    if not np.any(mask):
        return stack
    baseline = np.nanmean(stack[..., mask], axis=-1, keepdims=True)
    return stack - baseline



def build_stimulus_locked_feature_mats(
    dmd_mats,
    dmd_timebases,
    dmd_dt,
    ordered_images,
    session_start_sec,
    t_pre: float = 0.5,
    t_post: float = 1.0,
    baseline_subtract: bool = True,
    baseline_window: Tuple[float, float] = (-0.25, 0.0),
    smooth_sigma: float = 2.0,
):
    image_names = [evt.image_name for evt in ordered_images]
    unique_images = list(dict.fromkeys(image_names))
    pooled_times = [float(evt.onset) - float(session_start_sec) for evt in ordered_images]

    image_to_times = {
        img: [float(evt.onset) - float(session_start_sec) for evt in ordered_images if evt.image_name == img]
        for img in unique_images
    }

    pooled = {}
    per_image = {}
    t_rels = {}

    for dmd, mat in dmd_mats.items():
        t = dmd_timebases[dmd]
        dt = dmd_dt[dmd]

        stack, t_rel = _extract_triggered_stack(mat, t, pooled_times, t_pre, t_post, dt)
        if baseline_subtract:
            stack = _baseline_subtract_stack(stack, t_rel, baseline_window)
        pooled_mean = np.nanmean(stack, axis=0) if stack.size else np.full((mat.shape[0], len(t_rel)), np.nan)

        blocks = []
        for img in unique_images:
            st, _ = _extract_triggered_stack(mat, t, image_to_times[img], t_pre, t_post, dt)
            if baseline_subtract:
                st = _baseline_subtract_stack(st, t_rel, baseline_window)
            blk = np.nanmean(st, axis=0) if st.size else np.full((mat.shape[0], len(t_rel)), np.nan)
            blocks.append(blk)

        per_img = np.concatenate(blocks, axis=1)
        pooled_mean = _smooth_rows(pooled_mean, smooth_sigma)
        per_img = _smooth_rows(per_img, smooth_sigma)

        pooled[dmd] = pooled_mean
        per_image[dmd] = per_img
        t_rels[dmd] = t_rel

    return {
        "pooled": pooled,
        "per_image": per_image,
        "t_rel": t_rels,
        "unique_images": unique_images,
    }


# -----------------------------------------------------------------------------
# sorting
# -----------------------------------------------------------------------------


def _sort_rows_by_feature_matrix(feature_mat, metric="correlation", method="average"):
    x = np.asarray(feature_mat, dtype=float)
    if x.shape[0] <= 2:
        return np.arange(x.shape[0]), None

    x = _fill_nan_rowwise(x, fill_value="median")
    row_std = np.std(x, axis=1)
    x = x.copy()
    x[row_std == 0] += 1e-12

    d = pdist(x, metric=metric)
    d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)

    if np.allclose(d, 0):
        return np.arange(x.shape[0]), None

    z = linkage(d, method=method)
    return leaves_list(z), None



def _sort_rows_by_pc1(feature_mat, nan_fill="median"):
    x = np.asarray(feature_mat, dtype=float)

    if x.shape[0] <= 1:
        return np.arange(x.shape[0]), {"scores": np.zeros(x.shape[0]), "pca": None}

    x = _fill_nan_rowwise(x, fill_value=nan_fill)
    col_mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - col_mean

    if np.allclose(np.std(x_centered, axis=0), 0):
        scores = np.zeros(x.shape[0])
        return np.arange(x.shape[0]), {"scores": scores, "pca": None}

    pca = PCA(n_components=1)
    scores = pca.fit_transform(x_centered).ravel()
    order = np.argsort(-scores)
    return order, {"scores": scores, "pca": pca}



def _sort_rows_by_rastermap(feature_mat: np.ndarray, rastermap_kwargs: Optional[dict] = None):
    try:
        from rastermap import Rastermap
    except ImportError as e:
        raise ImportError(
            "Rastermap sorting requested, but the 'rastermap' package is not installed. "
            "Install it in your environment first."
        ) from e

    x = np.asarray(feature_mat, dtype=float)
    if x.shape[0] <= 1:
        return np.arange(x.shape[0]), {"embedding": None, "model": None}

    x = _fill_nan_rowwise(x, fill_value="median")
    x = x - np.mean(x, axis=1, keepdims=True)

    kwargs = {
        "n_PCs": min(200, x.shape[1], max(2, x.shape[0] - 1)),
        "n_clusters": min(100, max(10, x.shape[0] // 2)),
        "locality": 0.75,
        "time_lag_window": 0,
    }
    kwargs.update(rastermap_kwargs or {})

    model = Rastermap(**kwargs)
    embedding = np.asarray(model.fit_transform(x))

    if hasattr(model, "isort") and model.isort is not None:
        order = np.asarray(model.isort)
    else:
        scores = embedding[:, 0] if embedding.ndim > 1 else embedding
        order = np.argsort(scores)

    return order, {"embedding": embedding, "model": model}



def compute_sort_orders(
    dmd_mats: Dict[int, np.ndarray],
    features: Dict[str, Any],
    sort_by: Optional[str],
    feature_smooth_sigma: float = 2.0,
    rastermap_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:
    dmds = sorted(dmd_mats.keys())
    dmd_order = {}
    dmd_sort_meta = {}

    for dmd in dmds:
        if sort_by == "stimulus_locked_per_image":
            order, meta = _sort_rows_by_feature_matrix(features["per_image"][dmd])
        elif sort_by == "stimulus_locked_pooled":
            order, meta = _sort_rows_by_feature_matrix(features["pooled"][dmd])
        elif sort_by == "raw_correlation":
            basis = _smooth_rows(_fill_nan_rowwise(dmd_mats[dmd]), feature_smooth_sigma)
            order, meta = _sort_rows_by_feature_matrix(basis)
        elif sort_by == "pc1_per_image":
            order, meta = _sort_rows_by_pc1(features["per_image"][dmd])
        elif sort_by == "pc1_pooled":
            order, meta = _sort_rows_by_pc1(features["pooled"][dmd])
        elif sort_by == "pc1_raw":
            basis = _smooth_rows(_fill_nan_rowwise(dmd_mats[dmd]), feature_smooth_sigma)
            order, meta = _sort_rows_by_pc1(basis)
        elif sort_by == "rastermap_per_image":
            order, meta = _sort_rows_by_rastermap(features["per_image"][dmd], rastermap_kwargs=rastermap_kwargs)
        elif sort_by == "rastermap_pooled":
            order, meta = _sort_rows_by_rastermap(features["pooled"][dmd], rastermap_kwargs=rastermap_kwargs)
        elif sort_by == "rastermap_raw":
            basis = _smooth_rows(_fill_nan_rowwise(dmd_mats[dmd]), feature_smooth_sigma)
            order, meta = _sort_rows_by_rastermap(basis, rastermap_kwargs=rastermap_kwargs)
        elif sort_by is None:
            order = np.arange(dmd_mats[dmd].shape[0])
            meta = None
        else:
            raise ValueError(
                "sort_by must be one of: 'stimulus_locked_per_image', 'stimulus_locked_pooled', "
                "'raw_correlation', 'pc1_per_image', 'pc1_pooled', 'pc1_raw', "
                "'rastermap_per_image', 'rastermap_pooled', 'rastermap_raw', or None"
            )

        dmd_order[dmd] = np.asarray(order)
        dmd_sort_meta[dmd] = meta

    return {"dmd_order": dmd_order, "dmd_sort_meta": dmd_sort_meta}



def build_pc1_trace_for_session(
    session_mat: np.ndarray,
    session_t: np.ndarray,
    smooth_sigma: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Return the temporal PC1 component of a synapse x time matrix."""
    x = _smooth_rows(_fill_nan_rowwise(session_mat), smooth_sigma)
    x = x - np.mean(x, axis=0, keepdims=True)
    pca = PCA(n_components=1)
    pca.fit(x)
    trace = np.asarray(pca.components_[0], dtype=float)
    return {"time_sec": session_t, "trace": trace, "pca": pca}


# -----------------------------------------------------------------------------
# plotting: session heatmap only
# -----------------------------------------------------------------------------


def plot_glutamate_session(
    asset,
    signal: str = "dF",
    mode: str = "ls",
    channels: str = "glutamate",
    xlim_sec: Optional[Tuple[float, float]] = None,
    normalize_rows: Optional[str] = "zscore",
    display_smooth_sigma: float = 1.5,
    feature_smooth_sigma: float = 2.0,
    sort_by: Optional[str] = "pc1_raw",
    show_image_bar: bool = True,
    show_sort_trace: bool = False,
    sort_trace_source_dmd: Optional[int] = None,
    sort_trace_height: float = 0.7,
    show_running_trace: bool = False,
    running_trace_height: float = 0.7,
    encoder_path: Optional[str] = None,
    encoder_col: str = "Encoder",
    wheel_radius_cm: float = 8.0,
    encoder_units: str = "cm",
    speed_units: str = "cm/s",
    running_median_filter_kernel: int = 51,
    running_absolute_speed: bool = True,
    ticks_per_revolution: Optional[float] = None,
    running_smooth_speed_sigma: Optional[float] = 3.0,
    image_flash_duration: float = 0.25,
    im_colors=IM_COLORS,
    cmap_session: str = "Greens",
    session_percentiles: Tuple[float, float] = (10, 99),
    figsize_width: float = 16,
    synapse_height: float = 0.11,
    min_whole_height: float = 0.6,
    image_bar_height: float = 0.35,
    dmd_gap_height: float = 0.24,
    show_row_labels: bool = True,
    running_time_zero: str = "first_sample",
    running_time_offset_sec: float = 0.0,
    # label strings
    image_bar_ylabel: str = "Image",
    session_xlabel: str = "Session time from imaging epoch start (s)",
    session_ylabel_template: str = "DMD {dmd}\n({depth_txt})\n{n_syn} syn",
    running_ylabel: str = "Speed\n(cm/s)",
    sort_trace_ylabel: str = "PC1",
    cbar_label: Optional[str] = None,
    # label kwargs
    label_kwargs: Optional[dict] = None,
    image_bar_ylabel_kwargs: Optional[dict] = None,
    session_xlabel_kwargs: Optional[dict] = None,
    session_ylabel_kwargs: Optional[dict] = None,
    running_ylabel_kwargs: Optional[dict] = None,
    sort_trace_ylabel_kwargs: Optional[dict] = None,
    cbar_label_kwargs: Optional[dict] = None,
    # tick params
    x_tick_params: Optional[dict] = None,
    y_tick_params: Optional[dict] = None,
    rastermap_kwargs: Optional[dict] = None,
):
    """
    Plot session heatmaps split by DMD for an arbitrary session time window.

    Supported sort_by:
        - 'stimulus_locked_per_image'
        - 'stimulus_locked_pooled'
        - 'raw_correlation'
        - 'pc1_per_image'
        - 'pc1_pooled'
        - 'pc1_raw'
        - 'rastermap_per_image'
        - 'rastermap_pooled'
        - 'rastermap_raw'
        - None

    Notes
    -----
    The optional top PC1 trace is only plotted when sort_by == 'pc1_raw', because
    that is the only mode where the sorting basis lives directly on session time.
    Rastermap sorting does not plot a PC1 trace.
    """
    stim = load_stimulus_events(asset)
    ordered_images = stim["ordered_images"]
    session_start_sec = stim["session_start_sec"]

    summary, dmd_mats, dmd_timebases, dmd_dt = build_session_glutamate_mats(
        asset=asset,
        signal=signal,
        mode=mode,
        channels=channels,
        normalize_rows=normalize_rows,
    )

    features = build_stimulus_locked_feature_mats(
        dmd_mats=dmd_mats,
        dmd_timebases=dmd_timebases,
        dmd_dt=dmd_dt,
        ordered_images=ordered_images,
        session_start_sec=session_start_sec,
        t_pre=0.5,
        t_post=1.0,
        baseline_subtract=True,
        baseline_window=(-0.25, 0.0),
        smooth_sigma=feature_smooth_sigma,
    )

    sort_info = compute_sort_orders(
        dmd_mats=dmd_mats,
        features=features,
        sort_by=sort_by,
        feature_smooth_sigma=feature_smooth_sigma,
        rastermap_kwargs=rastermap_kwargs,
    )
    dmd_order = sort_info["dmd_order"]
    dmd_sort_meta = sort_info["dmd_sort_meta"]

    dmds = sorted(dmd_mats.keys())

    dmd_session = {}
    dmd_display_time = {}
    for dmd in dmds:
        mat = dmd_mats[dmd][dmd_order[dmd]]
        mat = _smooth_rows(mat, display_smooth_sigma)
        sl = _clip_time_window(dmd_timebases[dmd], xlim_sec)
        dmd_session[dmd] = mat[:, sl]
        dmd_display_time[dmd] = dmd_timebases[dmd][sl]

    session_vals = np.concatenate([np.ravel(np.nan_to_num(dmd_session[d], nan=0.0)) for d in dmds])
    svmin, svmax = _safe_percentiles(session_vals, session_percentiles)

    if sort_trace_source_dmd is None:
        sort_trace_source_dmd = dmds[0]

    x0 = dmd_display_time[dmds[0]][0]
    x1 = dmd_display_time[dmds[0]][-1]

    pc1_trace = None
    if show_sort_trace and sort_by == "pc1_raw":
        src_t = dmd_timebases[sort_trace_source_dmd]
        src_sl = _clip_time_window(src_t, xlim_sec)
        pc1_trace = build_pc1_trace_for_session(
            session_mat=dmd_mats[sort_trace_source_dmd][:, src_sl],
            session_t=src_t[src_sl],
            smooth_sigma=feature_smooth_sigma,
        )

    running = None
    if show_running_trace:
        running = load_running_speed(
            asset=asset,
            session_start_sec=session_start_sec,
            xlim_sec=(x0, x1),
            encoder_path=encoder_path,
            encoder_col=encoder_col,
            wheel_radius_cm=wheel_radius_cm,
            encoder_units=encoder_units,
            speed_units=speed_units,
            median_filter_kernel=running_median_filter_kernel,
            absolute_speed=running_absolute_speed,
            ticks_per_revolution=ticks_per_revolution,
            smooth_speed_sigma=running_smooth_speed_sigma,
            time_zero=running_time_zero,
            time_offset_sec=running_time_offset_sec,
        )

    rows = []
    heights = []

    if show_running_trace:
        rows.append(("running", dmds[0]))
        heights.append(running_trace_height)

    if show_sort_trace and pc1_trace is not None:
        rows.append(("sort_trace", dmds[0]))
        heights.append(sort_trace_height)

    if show_image_bar:
        rows.append(("image", dmds[0]))
        heights.append(image_bar_height)

    for i, dmd in enumerate(dmds):
        n_syn = dmd_session[dmd].shape[0]
        rows.append(("whole", dmd))
        heights.append(max(min_whole_height, n_syn * synapse_height))
        if i < len(dmds) - 1:
            rows.append(("gap", dmd))
            heights.append(dmd_gap_height)

    fig = plt.figure(figsize=(figsize_width, sum(heights) + 0.4), constrained_layout=True)
    gs = GridSpec(
        nrows=len(rows),
        ncols=2,
        figure=fig,
        width_ratios=[40, 1.4],
        height_ratios=heights,
    )

    axes = {}
    sharex_ax = None
    session_im = None

    for r, (kind, dmd) in enumerate(rows):
        if kind == "gap":
            ax = fig.add_subplot(gs[r, 0])
            ax.axis("off")
            axes[(kind, dmd)] = ax
            continue

        if kind == "whole" and sharex_ax is None:
            ax = fig.add_subplot(gs[r, 0])
            sharex_ax = ax
        else:
            ax = fig.add_subplot(gs[r, 0], sharex=sharex_ax)

        axes[(kind, dmd)] = ax

    label_kwargs = dict(label_kwargs or {})
    image_bar_ylabel_kwargs = _merge_kwargs(label_kwargs, image_bar_ylabel_kwargs)
    session_xlabel_kwargs = _merge_kwargs(label_kwargs, session_xlabel_kwargs)
    session_ylabel_kwargs = _merge_kwargs(label_kwargs, session_ylabel_kwargs)
    running_ylabel_kwargs = _merge_kwargs(label_kwargs, running_ylabel_kwargs)
    sort_trace_ylabel_kwargs = _merge_kwargs(label_kwargs, sort_trace_ylabel_kwargs)
    cbar_label_kwargs = _merge_kwargs(label_kwargs, cbar_label_kwargs)

    x_tick_params = _merge_kwargs(DEFAULT_X_TICK_PARAMS, x_tick_params)
    y_tick_params = _merge_kwargs(DEFAULT_Y_TICK_PARAMS, y_tick_params)

    if show_running_trace and running is not None:
        ax = axes[("running", dmds[0])]
        ax.plot(running["time_sec"], running["speed"], color="k", lw=0.8)
        ax.set_xlim(x0, x1)
        ax.set_ylabel(running_ylabel, **running_ylabel_kwargs)
        ax.tick_params(**x_tick_params)
        ax.tick_params(**y_tick_params)
        ax.tick_params(axis="x", labelbottom=False)

    if show_sort_trace and pc1_trace is not None:
        ax = axes[("sort_trace", dmds[0])]
        ax.plot(pc1_trace["time_sec"], pc1_trace["trace"], color="k", lw=0.8)
        ax.set_xlim(x0, x1)
        ax.set_ylabel(sort_trace_ylabel, **sort_trace_ylabel_kwargs)
        ax.tick_params(**x_tick_params)
        ax.tick_params(**y_tick_params)
        ax.tick_params(axis="x", labelbottom=False)

    if show_image_bar:
        ax = axes[("image", dmds[0])]
        color_map = _build_image_color_map(ordered_images, im_colors)
        for evt in ordered_images:
            onset_rel = float(evt.onset) - float(session_start_sec)
            if onset_rel + image_flash_duration < x0 or onset_rel > x1:
                continue
            ax.axvspan(
                onset_rel,
                onset_rel + image_flash_duration,
                color=color_map.get(evt.image_name, "#cccccc"),
                lw=0,
                alpha=1.0,
            )
        ax.set_xlim(x0, x1)
        ax.set_yticks([])
        ax.set_ylabel(image_bar_ylabel, rotation=0, ha="right", va="center", labelpad=14, **image_bar_ylabel_kwargs)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

    last_whole_ax = None
    for dmd in dmds:
        depth = None
        try:
            if dmd - 1 < len(summary.dmd_zs):
                depth = summary.dmd_zs[dmd - 1]
        except Exception:
            pass

        depth_txt = f"{int(depth)} µm" if depth is not None and np.isfinite(depth) else "depth ?"
        n_syn = dmd_session[dmd].shape[0]
        left_label = session_ylabel_template.format(dmd=dmd, depth_txt=depth_txt, n_syn=n_syn)

        ax = axes[("whole", dmd)]
        t = dmd_display_time[dmd]
        mat = np.nan_to_num(dmd_session[dmd], nan=0.0)
        im = ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap_session,
            vmin=svmin,
            vmax=svmax,
            extent=[t[0], t[-1], mat.shape[0], 0],
        )
        if session_im is None:
            session_im = im

        if show_row_labels:
            ax.set_ylabel(left_label, **session_ylabel_kwargs)
        else:
            ax.set_ylabel("")

        ax.tick_params(**x_tick_params)
        ax.tick_params(**y_tick_params)
        ax.tick_params(axis="x", labelbottom=False)
        last_whole_ax = ax

    if last_whole_ax is not None:
        last_whole_ax.tick_params(axis="x", labelbottom=True)
        last_whole_ax.set_xlabel(session_xlabel, **session_xlabel_kwargs)

    cax = fig.add_subplot(gs[0, 1])
    if session_im is not None:
        cb = fig.colorbar(session_im, cax=cax)
        if cbar_label is None:
            cbar_label = f"{signal}/{mode}" + (" (row norm)" if normalize_rows else "")
        cb.set_label(cbar_label, **cbar_label_kwargs)
        cb.ax.tick_params(labelsize=y_tick_params.get("labelsize", 12))
    else:
        cax.axis("off")
        cb = None

    return {
        "fig": fig,
        "axes": axes,
        "summary": summary,
        "dmd_order": dmd_order,
        "dmd_sort_meta": dmd_sort_meta,
        "dmd_session": dmd_session,
        "dmd_display_time": dmd_display_time,
        "session_cbar": cb,
        "stimulus_events": stim,
        "pc1_trace": pc1_trace,
        "running": running,
    }


# -----------------------------------------------------------------------------
# plotting: stimulus-locked heatmaps (kept separate)
# -----------------------------------------------------------------------------


def plot_stimulus_locked_heatmaps(
    asset,
    signal: str = "dF",
    mode: str = "ls",
    channels: str = "glutamate",
    normalize_rows: Optional[str] = "zscore",
    feature_smooth_sigma: float = 2.0,
    sort_by: Optional[str] = "stimulus_locked_per_image",
    t_pre: float = 0.5,
    t_post: float = 1.0,
    baseline_subtract: bool = True,
    baseline_window: Tuple[float, float] = (-0.25, 0.0),
    show_pooled: bool = True,
    show_per_image: bool = True,
    cmap_feature: str = "coolwarm",
    feature_percentiles: Tuple[float, float] = (2, 98),
    figsize_width: float = 12,
    synapse_height: float = 0.10,
    pooled_height: float = 1.0,
    per_image_height: float = 1.3,
    dmd_gap_height: float = 0.20,
    max_image_label_len: int = 16,
    row_label_template: str = "DMD {dmd}\n({depth_txt})\n{n_syn} syn",
    x_tick_params: Optional[dict] = None,
    y_tick_params: Optional[dict] = None,
    label_kwargs: Optional[dict] = None,
    xlabel_kwargs: Optional[dict] = None,
    ylabel_kwargs: Optional[dict] = None,
    cbar_label: str = "Stimulus-locked response",
    cbar_label_kwargs: Optional[dict] = None,
    rastermap_kwargs: Optional[dict] = None,
):
    summary, dmd_mats, dmd_timebases, dmd_dt = build_session_glutamate_mats(
        asset=asset,
        signal=signal,
        mode=mode,
        channels=channels,
        normalize_rows=normalize_rows,
    )

    stim = load_stimulus_events(asset)
    features = build_stimulus_locked_feature_mats(
        dmd_mats=dmd_mats,
        dmd_timebases=dmd_timebases,
        dmd_dt=dmd_dt,
        ordered_images=stim["ordered_images"],
        session_start_sec=stim["session_start_sec"],
        t_pre=t_pre,
        t_post=t_post,
        baseline_subtract=baseline_subtract,
        baseline_window=baseline_window,
        smooth_sigma=feature_smooth_sigma,
    )

    sort_info = compute_sort_orders(
        dmd_mats=dmd_mats,
        features=features,
        sort_by=sort_by,
        feature_smooth_sigma=feature_smooth_sigma,
        rastermap_kwargs=rastermap_kwargs,
    )
    dmd_order = sort_info["dmd_order"]

    dmds = sorted(dmd_mats.keys())
    dmd_pooled = {d: features["pooled"][d][dmd_order[d]] for d in dmds}
    dmd_per_image = {d: features["per_image"][d][dmd_order[d]] for d in dmds}

    vals = np.concatenate(
        [np.ravel(np.nan_to_num(dmd_pooled[d], nan=0.0)) for d in dmds] +
        [np.ravel(np.nan_to_num(dmd_per_image[d], nan=0.0)) for d in dmds]
    )
    fvmin, fvmax = _safe_percentiles(vals, feature_percentiles)

    rows = []
    heights = []
    for i, dmd in enumerate(dmds):
        n_syn = dmd_pooled[dmd].shape[0]
        if show_pooled:
            rows.append(("pooled", dmd))
            heights.append(max(0.7, n_syn * synapse_height, pooled_height))
        if show_per_image:
            rows.append(("per_image", dmd))
            heights.append(max(0.8, n_syn * synapse_height, per_image_height))
        if i < len(dmds) - 1:
            rows.append(("gap", dmd))
            heights.append(dmd_gap_height)

    fig = plt.figure(figsize=(figsize_width, sum(heights) + 0.4), constrained_layout=True)
    gs = GridSpec(len(rows), 2, figure=fig, width_ratios=[40, 1.4], height_ratios=heights)

    axes = {}
    feature_im = None
    label_kwargs = dict(label_kwargs or {})
    xlabel_kwargs = _merge_kwargs(label_kwargs, xlabel_kwargs)
    ylabel_kwargs = _merge_kwargs(label_kwargs, ylabel_kwargs)
    cbar_label_kwargs = _merge_kwargs(label_kwargs, cbar_label_kwargs)
    x_tick_params = _merge_kwargs(DEFAULT_X_TICK_PARAMS, x_tick_params)
    y_tick_params = _merge_kwargs(DEFAULT_Y_TICK_PARAMS, y_tick_params)

    unique_images = features["unique_images"]
    short_labels = [_short_image_label(x, max_len=max_image_label_len) for x in unique_images]

    for r, (kind, dmd) in enumerate(rows):
        if kind == "gap":
            ax = fig.add_subplot(gs[r, 0])
            ax.axis("off")
            axes[(kind, dmd)] = ax
            continue
        ax = fig.add_subplot(gs[r, 0])
        axes[(kind, dmd)] = ax

        depth = None
        try:
            if dmd - 1 < len(summary.dmd_zs):
                depth = summary.dmd_zs[dmd - 1]
        except Exception:
            pass
        depth_txt = f"{int(depth)} µm" if depth is not None and np.isfinite(depth) else "depth ?"
        label = row_label_template.format(dmd=dmd, depth_txt=depth_txt, n_syn=dmd_pooled[dmd].shape[0])

        if kind == "pooled":
            mat = np.nan_to_num(dmd_pooled[dmd], nan=0.0)
            t_rel = features["t_rel"][dmd]
            im = ax.imshow(
                mat,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap_feature,
                vmin=fvmin,
                vmax=fvmax,
                extent=[t_rel[0], t_rel[-1], mat.shape[0], 0],
            )
            ax.axvline(0, color="k", lw=0.7, alpha=0.7)
            ax.set_xlabel("Time from image onset (s)", **xlabel_kwargs)
        else:
            mat = np.nan_to_num(dmd_per_image[dmd], nan=0.0)
            t_rel = features["t_rel"][dmd]
            n_win = len(t_rel)
            im = ax.imshow(
                mat,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap_feature,
                vmin=fvmin,
                vmax=fvmax,
                extent=[0, mat.shape[1], mat.shape[0], 0],
            )
            centers = []
            for i, _img in enumerate(unique_images):
                start = i * n_win
                zero_idx = start + np.searchsorted(t_rel, 0)
                ax.axvline(start, color="k", lw=0.45, alpha=0.35)
                ax.axvline(zero_idx, color="k", lw=0.7, alpha=0.7)
                centers.append(start + n_win / 2)
            ax.axvline(len(unique_images) * n_win, color="k", lw=0.45, alpha=0.35)
            ax.set_xticks(centers)
            ax.set_xticklabels(short_labels, rotation=40, ha="right")
            ax.set_xlabel("Image identity", **xlabel_kwargs)

        feature_im = im
        ax.set_ylabel(label, **ylabel_kwargs)
        ax.tick_params(**x_tick_params)
        ax.tick_params(**y_tick_params)

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(feature_im, cax=cax)
    cb.set_label(cbar_label, **cbar_label_kwargs)
    cb.ax.tick_params(labelsize=y_tick_params.get("labelsize", 12))

    return {
        "fig": fig,
        "axes": axes,
        "summary": summary,
        "dmd_order": dmd_order,
        "dmd_pooled": dmd_pooled,
        "dmd_per_image": dmd_per_image,
        "feature_cbar": cb,
    }
