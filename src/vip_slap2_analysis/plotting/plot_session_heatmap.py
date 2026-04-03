import os
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PNW_cmap import PNW_cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, leaves_list
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
from vip_slap2_analysis.glutamate.summary import GlutamateSummary
from vip_slap2_analysis.glutamate.alignment import (
    load_corrected_bonsai_csv,
    load_imaging_epochs_csv,
    extract_image_intervals,
    filter_ordered_images_to_epochs,
)

import seaborn as sns
sns.set_style('white')
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)



IM_COLORS = [
    '#c5cae9', '#ffcdd2', '#c8e6c9', '#ffe0b2',
    '#e1bee7', '#d7ccc8', '#9fd3f2'
]


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _robust_row_zscore(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=1, keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(scale < eps, 1.0, scale)
    return (x - med) / scale


def _fill_nan_rowwise(x, fill_value="median"):
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


def _smooth_rows(x, sigma_samples=0):
    if sigma_samples is None or sigma_samples <= 0:
        return x
    return gaussian_filter1d(x, sigma=sigma_samples, axis=1, mode="nearest")


def _compute_dt(tb):
    tb = np.asarray(tb, dtype=float)
    if tb.size >= 2:
        dt = float(np.nanmedian(np.diff(tb)))
        if np.isfinite(dt) and dt > 0:
            return dt
    return 1.0


def _safe_percentiles(x, q=(2, 98)):
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(vals, q)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _clip_time_window(t, xlim_sec):
    if xlim_sec is None:
        return slice(None)
    t0, t1 = xlim_sec
    mask = (t >= t0) & (t <= t1)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError(f"No data in requested xlim_sec={xlim_sec}")
    return slice(idx[0], idx[-1] + 1)


def _short_image_label(name, max_len=18):
    base = os.path.basename(str(name))
    base = base.replace(".tiff", "").replace(".tif", "")
    base = base.replace("stimuliImages_", "")
    if len(base) > max_len:
        return base[:max_len - 1] + "…"
    return base


def _sort_rows_by_feature_matrix(feature_mat, metric="correlation", method="average"):
    x = np.asarray(feature_mat, dtype=float)
    if x.shape[0] <= 2:
        return np.arange(x.shape[0])

    x = _fill_nan_rowwise(x, fill_value="median")
    row_std = np.std(x, axis=1)
    x = x.copy()
    x[row_std == 0] += 1e-12

    d = pdist(x, metric=metric)
    d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=1.0)

    if np.allclose(d, 0):
        return np.arange(x.shape[0])

    z = linkage(d, method=method)
    return leaves_list(z)


def _sort_rows_by_pc1(feature_mat, nan_fill="median"):
    """
    Sort rows by their score on the first principal component.
    """
    x = np.asarray(feature_mat, dtype=float)

    if x.shape[0] <= 1:
        return np.arange(x.shape[0]), np.zeros(x.shape[0]), None

    if nan_fill == "median":
        x = _fill_nan_rowwise(x, fill_value="median")
    elif nan_fill == "mean":
        x = _fill_nan_rowwise(x, fill_value="mean")
    elif nan_fill == "zero":
        x = _fill_nan_rowwise(x, fill_value="zero")
    else:
        raise ValueError("nan_fill must be 'median', 'mean', or 'zero'")

    col_mean = np.mean(x, axis=0, keepdims=True)
    x_centered = x - col_mean

    if np.allclose(np.std(x_centered, axis=0), 0):
        scores = np.zeros(x.shape[0])
        order = np.arange(x.shape[0])
        return order, scores, None

    pca = PCA(n_components=1)
    scores = pca.fit_transform(x_centered).ravel()
    order = np.argsort(-scores)
    return order, scores, pca


# ---------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------

def load_stimulus_events(asset):
    stim_df = load_corrected_bonsai_csv(asset.bonsai_event_log_csv)
    _, ordered_images = extract_image_intervals(stim_df)

    epoch_csv = asset.qc_dir / "behavior" / "imaging_epochs.csv"
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


def _build_image_color_map(ordered_images, im_colors):
    unique = []
    seen = set()
    for evt in ordered_images:
        nm = evt.image_name
        if nm not in seen:
            unique.append(nm)
            seen.add(nm)
    return {img: im_colors[i % len(im_colors)] for i, img in enumerate(unique)}


# ---------------------------------------------------------------------
# session matrix construction
# ---------------------------------------------------------------------

def build_session_glutamate_mats(
    asset,
    signal="dF",
    mode="ls",
    channels="glutamate",
    normalize_rows="zscore",
):  
    summary = GlutamateSummary(asset.summary_mat)

    print(f'loaded data from {asset.summary_mat}')
    
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


# ---------------------------------------------------------------------
# stimulus-triggered feature matrices
# ---------------------------------------------------------------------

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
    t_pre=0.5,
    t_post=1.0,
    baseline_subtract=True,
    baseline_window=(-0.25, 0.0),
    smooth_sigma=2.0,
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


# ---------------------------------------------------------------------
# main plotting function
# ---------------------------------------------------------------------

def plot_glutamate_session(
    asset,
    signal="dF",
    mode="ls",
    channels="glutamate",
    xlim_sec=None,
    normalize_rows="zscore",
    display_smooth_sigma=1.5,
    feature_smooth_sigma=2.0,
    sort_by="stimulus_locked_per_image",
    t_pre=0.5,
    t_post=1.0,
    baseline_subtract=True,
    baseline_window=(-0.25, 0.0),
    show_image_bar=True,
    show_whole_session=True,
    show_pooled_stimulus_heatmap=True,
    show_per_image_heatmap=True,
    cmap_session="viridis",
    cmap_feature="coolwarm",
    session_percentiles=(2, 98),
    feature_percentiles=(2, 98),
    image_flash_duration=0.25,
    im_colors=IM_COLORS,
    figsize_width=14,
    synapse_height=0.12,
    min_whole_height=1.5,
    pooled_height=1.0,
    per_image_height=1.3,
    image_bar_height=0.28,
    dmd_gap_height=0.22,
    show_titles=False,
    show_row_labels=True,
    max_image_label_len=16,
):
    """
    sort_by options:
        - "stimulus_locked_per_image"
        - "stimulus_locked_pooled"
        - "raw_correlation"
        - "pc1_per_image"
        - "pc1_pooled"
        - "pc1_raw"
        - None
    """
    summary, dmd_mats, dmd_timebases, dmd_dt = build_session_glutamate_mats(
        asset=asset,
        signal=signal,
        mode=mode,
        channels=channels,
        normalize_rows=normalize_rows,
    )

    stim = load_stimulus_events(asset)
    ordered_images = stim["ordered_images"]
    session_start_sec = stim["session_start_sec"]

    features = build_stimulus_locked_feature_mats(
        dmd_mats=dmd_mats,
        dmd_timebases=dmd_timebases,
        dmd_dt=dmd_dt,
        ordered_images=ordered_images,
        session_start_sec=session_start_sec,
        t_pre=t_pre,
        t_post=t_post,
        baseline_subtract=baseline_subtract,
        baseline_window=baseline_window,
        smooth_sigma=feature_smooth_sigma,
    )

    dmds = sorted(dmd_mats.keys())

    dmd_order = {}
    dmd_sort_values = {}
    dmd_pca_models = {}

    for dmd in dmds:
        if sort_by == "stimulus_locked_per_image":
            basis = features["per_image"][dmd]
            order = _sort_rows_by_feature_matrix(basis)
            dmd_sort_values[dmd] = None
            dmd_pca_models[dmd] = None

        elif sort_by == "stimulus_locked_pooled":
            basis = features["pooled"][dmd]
            order = _sort_rows_by_feature_matrix(basis)
            dmd_sort_values[dmd] = None
            dmd_pca_models[dmd] = None

        elif sort_by == "raw_correlation":
            basis = _smooth_rows(_fill_nan_rowwise(dmd_mats[dmd]), feature_smooth_sigma)
            order = _sort_rows_by_feature_matrix(basis)
            dmd_sort_values[dmd] = None
            dmd_pca_models[dmd] = None

        elif sort_by == "pc1_per_image":
            basis = features["per_image"][dmd]
            order, scores, pca_model = _sort_rows_by_pc1(basis)
            dmd_sort_values[dmd] = scores
            dmd_pca_models[dmd] = pca_model

        elif sort_by == "pc1_pooled":
            basis = features["pooled"][dmd]
            order, scores, pca_model = _sort_rows_by_pc1(basis)
            dmd_sort_values[dmd] = scores
            dmd_pca_models[dmd] = pca_model

        elif sort_by == "pc1_raw":
            basis = _smooth_rows(_fill_nan_rowwise(dmd_mats[dmd]), feature_smooth_sigma)
            order, scores, pca_model = _sort_rows_by_pc1(basis)
            dmd_sort_values[dmd] = scores
            dmd_pca_models[dmd] = pca_model

        elif sort_by is None:
            order = np.arange(dmd_mats[dmd].shape[0])
            dmd_sort_values[dmd] = None
            dmd_pca_models[dmd] = None

        else:
            raise ValueError(
                "sort_by must be one of: "
                "'stimulus_locked_per_image', 'stimulus_locked_pooled', "
                "'raw_correlation', 'pc1_per_image', 'pc1_pooled', 'pc1_raw', or None"
            )

        dmd_order[dmd] = order

    dmd_session = {}
    dmd_display_time = {}
    dmd_pooled = {}
    dmd_per_image = {}

    for dmd in dmds:
        mat = dmd_mats[dmd][dmd_order[dmd]]
        mat = _smooth_rows(mat, display_smooth_sigma)

        sl = _clip_time_window(dmd_timebases[dmd], xlim_sec)
        dmd_session[dmd] = mat[:, sl]
        dmd_display_time[dmd] = dmd_timebases[dmd][sl]
        dmd_pooled[dmd] = features["pooled"][dmd][dmd_order[dmd]]
        dmd_per_image[dmd] = features["per_image"][dmd][dmd_order[dmd]]

    session_vals = np.concatenate([np.ravel(np.nan_to_num(dmd_session[d], nan=0.0)) for d in dmds])
    feature_vals = np.concatenate(
        [np.ravel(np.nan_to_num(dmd_pooled[d], nan=0.0)) for d in dmds] +
        [np.ravel(np.nan_to_num(dmd_per_image[d], nan=0.0)) for d in dmds]
    )

    svmin, svmax = _safe_percentiles(session_vals, session_percentiles)
    fvmin, fvmax = _safe_percentiles(feature_vals, feature_percentiles)

    rows = []
    heights = []

    for i, dmd in enumerate(dmds):
        n_syn = dmd_session[dmd].shape[0]

        if i == 0 and show_image_bar:
            rows.append(("image", dmd))
            heights.append(image_bar_height)

        if show_whole_session:
            rows.append(("whole", dmd))
            heights.append(max(min_whole_height, n_syn * synapse_height))

        if show_pooled_stimulus_heatmap:
            rows.append(("pooled", dmd))
            heights.append(pooled_height)

        if show_per_image_heatmap:
            rows.append(("per_image", dmd))
            heights.append(per_image_height)

        if i < len(dmds) - 1:
            rows.append(("gap", dmd))
            heights.append(dmd_gap_height)

    fig = plt.figure(figsize=(figsize_width, sum(heights) + 0.4), constrained_layout=True)
    gs = GridSpec(
        nrows=len(rows),
        ncols=2,
        figure=fig,
        width_ratios=[40, 1.5],
        height_ratios=heights,
    )

    axes = {}
    sharex_ax = None
    session_im = None
    feature_im = None

    for r, (kind, dmd) in enumerate(rows):
        if kind == "gap":
            ax = fig.add_subplot(gs[r, 0])
            ax.axis("off")
            axes[(kind, dmd)] = ax
            continue

        if kind == "image":
            ax = fig.add_subplot(gs[r, 0], sharex=sharex_ax)
        elif kind == "whole":
            if sharex_ax is None:
                ax = fig.add_subplot(gs[r, 0])
                sharex_ax = ax
            else:
                ax = fig.add_subplot(gs[r, 0], sharex=sharex_ax)
        else:
            ax = fig.add_subplot(gs[r, 0])

        axes[(kind, dmd)] = ax

    if show_image_bar:
        ax = axes[("image", dmds[0])]
        x0 = dmd_display_time[dmds[0]][0]
        x1 = dmd_display_time[dmds[0]][-1]
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
        ax.set_ylabel("Image", rotation=0, ha="right", va="center", labelpad=14)
        if show_titles:
            ax.set_title("Image presentations", fontsize=11, loc="left", pad=4)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

    unique_images = features["unique_images"]
    short_labels = [_short_image_label(x, max_len=max_image_label_len) for x in unique_images]

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
        left_label = f"DMD {dmd}\n({depth_txt})\n{n_syn} syn"

        if show_whole_session:
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

            ax.set_ylabel(left_label if show_row_labels else "")
            if show_titles:
                ax.set_title(f"DMD {dmd} whole session", loc="left", fontsize=10, pad=3)
            else:
                ax.text(
                    0.0, 1.01,
                    f"DMD {dmd} · session",
                    transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=10
                )

            ax.tick_params(axis="x", labelbottom=False)
            last_whole_ax = ax

        if show_pooled_stimulus_heatmap:
            ax = axes[("pooled", dmd)]
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
            if feature_im is None:
                feature_im = im

            ax.axvline(0, color="k", lw=0.7, alpha=0.7)
            ax.set_ylabel(left_label if show_row_labels else "")
            if show_titles:
                ax.set_title(f"DMD {dmd} pooled image-triggered", loc="left", fontsize=10, pad=3)
            else:
                ax.text(
                    0.0, 1.01,
                    "pooled image-triggered",
                    transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9
                )

            ax.set_xlabel("Time from image onset (s)")
            ax.tick_params(axis="y", length=0)

        if show_per_image_heatmap:
            ax = axes[("per_image", dmd)]
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
            if feature_im is None:
                feature_im = im

            centers = []
            for i, _img in enumerate(unique_images):
                start = i * n_win
                stop = (i + 1) * n_win
                zero_idx = start + np.searchsorted(t_rel, 0)

                ax.axvline(start, color="k", lw=0.45, alpha=0.35)
                ax.axvline(zero_idx, color="k", lw=0.7, alpha=0.7)
                centers.append((start + stop) / 2)

            ax.axvline(len(unique_images) * n_win, color="k", lw=0.45, alpha=0.35)

            ax.set_ylabel(left_label if show_row_labels else "")
            if show_titles:
                ax.set_title(f"DMD {dmd} by image identity", loc="left", fontsize=10, pad=3)
            else:
                ax.text(
                    0.0, 1.01,
                    "by image identity",
                    transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9
                )

            if dmd == dmds[-1]:
                ax.set_xticks(centers)
                ax.set_xticklabels(short_labels, rotation=40, ha="right")
                ax.set_xlabel("Image identity")
            else:
                ax.set_xticks([])
                ax.set_xlabel("")

    if last_whole_ax is not None:
        last_whole_ax.tick_params(axis="x", labelbottom=True)
        last_whole_ax.set_xlabel("Session time from imaging epoch start (s)")

    if session_im is not None:
        cax = fig.add_subplot(gs[1, 1] if show_image_bar else gs[0, 1])
        cb = fig.colorbar(session_im, cax=cax)
        cb.set_label(f"{signal}/{mode}" + (" (row norm)" if normalize_rows else ""))
    else:
        cb = None

    feature_row = None
    for i, (kind, dmd) in enumerate(rows):
        if kind in ("pooled", "per_image"):
            feature_row = i
            break

    if feature_im is not None and feature_row is not None:
        cax2 = fig.add_subplot(gs[feature_row, 1])
        cb2 = fig.colorbar(feature_im, cax=cax2)
        cb2.set_label("Stimulus-locked response")
    else:
        cb2 = None

    return {
        "fig": fig,
        "axes": axes,
        "summary": summary,
        "dmd_order": dmd_order,
        "dmd_sort_values": dmd_sort_values,
        "dmd_pca_models": dmd_pca_models,
        "dmd_session": dmd_session,
        "dmd_pooled": dmd_pooled,
        "dmd_per_image": dmd_per_image,
        "session_cbar": cb,
        "feature_cbar": cb2,
    }

### EXAMPLE USEAGE

# out = plot_glutamate_session_structure_clean(
#     asset, # from session_registry.py
#     signal="dF",
#     mode="ls",
#     channels="glutamate",
#     xlim_sec=(140,200),
#     normalize_rows='zscore',
#     display_smooth_sigma=1.5,
#     feature_smooth_sigma=2.0,
#     sort_by="pc1_per_image",
#     t_pre=0.5,
#     t_post=1.0,
#     baseline_subtract=True,
#     baseline_window=(-0.25, 0.0),
#     show_image_bar=True,
#     show_whole_session=True,
#     show_pooled_stimulus_heatmap=False,
#     show_per_image_heatmap=False,
#     cmap_session="Greens",
#     cmap_feature="coolwarm",
#     session_percentiles=(10, 99),
#     feature_percentiles=(10, 99),
#     image_flash_duration=0.25,
#     im_colors=IM_COLORS,
#     figsize_width=25,
#     synapse_height=0.1,
#     min_whole_height=0.5,
#     pooled_height=0.75,
#     per_image_height=0.75,
#     image_bar_height=0.8,
#     dmd_gap_height=0.22,
#     show_titles=False,
#     show_row_labels=True,
#     max_image_label_len=16,
# )
# plt.show()

# filen = f'{today_str}_{asset.session_id}_heatmap'
# # save_figure(out['fig'],os.path.join(save_path,filen),formats=['.pdf'],dpi=300)