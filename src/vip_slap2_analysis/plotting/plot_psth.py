import os
import numpy as np
import matplotlib.pyplot as plt
from vip_slap2_analysis.voltage import analysis
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

def plot_voltage_mean_image_response_heatmap(
    asset,
    *,
    trace_variant="dff_robust_f0_trial",
    mean_npz=None,
    dmd=1,
    image_name=None,
    use_roi_qc=False,
    normalize_rows="zscore",
    baseline_subtract=True,
    baseline_window=(-0.25, 0.0),
    smooth_sigma=2.0,
    sort_by="pc1_raw",
    cmap="coolwarm",
    percentiles=(2, 98),
    figsize=(4, 4),
    xlabel="Time from image onset (s)",
    ylabel="Dendrite ROI",
    cbar_label="Mean voltage response",
    label_kwargs=None,
    title_kwargs=None,
    cbar_label_kwargs=None,
    x_tick_params=None,
    y_tick_params=None,
):
    pkg, path = analysis.load_voltage_mean_npz(asset, trace_variant=trace_variant, mean_npz=mean_npz)
    dmd_key = f"DMD{int(dmd)}"
    if dmd_key not in pkg or "image_identity" not in pkg[dmd_key]:
        raise KeyError(f"{dmd_key}/image_identity not found in {path}")

    image_dict = pkg[dmd_key]["image_identity"]
    if image_name is None:
        image_name = next(iter(image_dict.keys()))
    if image_name not in image_dict:
        raise KeyError(f"Image {image_name!r} not found. Available images: {list(image_dict.keys())}")

    mat = np.asarray(image_dict[image_name]["mean"], dtype=float)
    t = np.asarray(pkg["timebase_sec"]["image"], dtype=float)

    if use_roi_qc and "valid_rois_mask" in pkg[dmd_key]:
        mask = np.asarray(pkg[dmd_key]["valid_rois_mask"], dtype=bool).reshape(-1)
        if mask.size == mat.shape[0]:
            mat = mat[mask]
        # else:

            # print(
            #     f"Warning: valid_rois_mask length {mask.size} does not match matrix rows {mat.shape[0]}; "
            #     "plotting without applying ROI QC mask."
            # )

    mat, t = analysis._coerce_event_mat_timebase(mat, t, context=f"{dmd_key}/{image_name}")

    if baseline_subtract:
        mat = analysis._baseline_subtract_mat(mat, t, baseline_window=baseline_window)
    if normalize_rows == "zscore":
        mat = analysis._robust_row_zscore(mat)
    elif normalize_rows in (None, False, "none"):
        pass
    else:
        raise ValueError("normalize_rows must be 'zscore', None, False, or 'none'.")

    mat = analysis._smooth_rows(mat, smooth_sigma)

    if sort_by is not None:
        sort_info = analysis.compute_sort_orders(
            dmd_mats={int(dmd): mat},
            features={"pooled": {int(dmd): mat}, "per_image": {int(dmd): mat}},
            sort_by=sort_by,
            feature_smooth_sigma=0,
        )
        order = sort_info["dmd_order"][int(dmd)]
        mat = mat[order]
    else:
        order = np.arange(mat.shape[0])

    vmin, vmax = analysis._safe_percentiles(mat, percentiles)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        np.nan_to_num(mat, nan=0.0),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[t[0], t[-1], mat.shape[0], 0],
    )
    ax.axvline(0, color="k", lw=0.8, alpha=0.8)

    label_kwargs = dict(label_kwargs or {})
    title_kwargs = analysis._merge_kwargs(label_kwargs, title_kwargs)
    cbar_label_kwargs = analysis._merge_kwargs(label_kwargs, cbar_label_kwargs)
    x_tick_params = analysis._merge_kwargs(DEFAULT_X_TICK_PARAMS, x_tick_params)
    y_tick_params = analysis._merge_kwargs(DEFAULT_Y_TICK_PARAMS, y_tick_params)

    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)
    ax.set_title(f"{dmd_key}: {os.path.basename(str(image_name))}", **title_kwargs)
    ax.tick_params(**x_tick_params)
    ax.tick_params(**y_tick_params)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label, **cbar_label_kwargs)
    
    fig.tight_layout()
    return {"fig": fig, "ax": ax, "cbar": cb, "data": mat, "timebase_sec": t, "image_name": image_name, "path": path, "order": order}