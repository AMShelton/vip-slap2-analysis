from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.multivariate.manova import MANOVA
except Exception:
    MANOVA = None


EPS = 1e-12


@dataclass
class GlutamateAnalysisConfig:
    """Configuration for glutamate response analyses.

    Default windows assume image-aligned traces sampled at 200 Hz with stimulus onset
    at sample 50 and a 250 ms pre/post comparison window.
    """

    alpha: float = 0.05
    min_events_activation: int = 8
    min_events_tuning_per_image: int = 5
    min_images_for_tuning: int = 4
    min_images_for_sequence: int = 4
    min_positions_for_sequence: int = 3
    n_shuffles_tuning: int = 2000
    random_seed: int = 0

    image_pre_samples: tuple[int, int] = (0, 50)
    image_post_samples: tuple[int, int] = (50, 100)
    change_pre_samples: tuple[int, int] = (100, 150)
    change_post_samples: tuple[int, int] = (150, 200)
    omission_pre_samples: tuple[int, int] = (200, 250)
    omission_post_samples: tuple[int, int] = (250, 300)

    sequence_pre_samples: tuple[int, int] = (0, 50)
    sequence_post_samples: tuple[int, int] = (50, 100)
    sequence_peak_window_samples: int = 10
    sequence_n_quantile_bins: int = 3
    sequence_min_count_per_position: int = 1
    sequence_label_slope_frac: float = 0.35
    sequence_label_min_abs_slope: float = 25.0
    tuning_min_effect_fve: float = 0.05
    tuning_method: str = "hybrid"  # {"fve", "manova", "hybrid"}
    manova_stat: str = "Wilks' lambda"
    manova_max_timepoints: int = 20
    manova_use_post_only: bool = False
    manova_interpolate_nans: bool = True
    manova_max_nan_fraction_per_trial: float = 0.1

    sequence_rank_by: str = "hybrid_preference"   # {"hybrid_preference", "selectivity_score", "response_amplitude"}
    sequence_norm_strategy: str = "r0_abs"        # {"r0_abs", "max_abs", "none"}


@dataclass
class GlutamateAnalysisPaths:
    single_trial_npz: Path
    mean_npz: Path
    sequence_npz: Path
    output_dir: Path


def _bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not np.any(finite):
        return out
    pf = pvals[finite]
    order = np.argsort(pf)
    ranked = pf[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out_f = np.empty_like(q)
    out_f[order] = q
    out[finite] = out_f
    return out


def _safe_wilcoxon_zero(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.allclose(x, 0.0, equal_nan=True):
        return 1.0
    try:
        return float(stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def _safe_kruskal(groups: Sequence[np.ndarray]) -> float:
    valid = []
    for g in groups:
        g = np.asarray(g, dtype=float)
        g = g[np.isfinite(g)]
        if g.size > 0:
            valid.append(g)
    if len(valid) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*valid).pvalue)
    except ValueError:
        return np.nan


def _basename_stimulus(stim_name: str) -> str:
    stim = str(stim_name).replace("\\", "/")
    leaf = stim.split("/")[-1]
    return leaf.replace(".tiff", "")


def _load_npz_dict(path: str | Path) -> dict[str, Any]:
    arr = np.load(Path(path), allow_pickle=True)
    if "data" not in arr.files:
        raise KeyError(f"Expected top-level key 'data' in {path}, found {arr.files}")
    return arr["data"].item()


def resolve_glutamate_analysis_paths(
    session_dir_or_analysis_dir: str | Path,
    output_dir: str | Path | None = None,
) -> GlutamateAnalysisPaths:
    base = Path(session_dir_or_analysis_dir)
    analysis_dir = base if base.name == "analysis" else base / "analysis"
    derived = analysis_dir / "derived" / "glutamate"
    out = Path(output_dir) if output_dir is not None else analysis_dir / "derived" / "glutamate" / "glutamate_analysis"
    return GlutamateAnalysisPaths(
        single_trial_npz=derived / "glutamate_single_trial_df.npz",
        mean_npz=derived / "glutamate_mean_df.npz",
        sequence_npz=derived / "glutamate_sequence_df.npz",
        output_dir=out,
    )


def _window_metric(windows: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    windows = np.asarray(windows, dtype=float)
    pre_auc = np.nansum(windows[..., pre[0]:pre[1]], axis=-1)
    post_auc = np.nansum(windows[..., post[0]:post[1]], axis=-1)
    return post_auc - pre_auc


def _nanmean_last_axis(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    denom = valid.sum(axis=-1)
    numer = np.nansum(arr, axis=-1)
    out = np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=float), where=denom > 0)
    return out


def _window_delta_mean(windows: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    windows = np.asarray(windows, dtype=float)
    pre_mean = _nanmean_last_axis(windows[..., pre[0]:pre[1]])
    post_mean = _nanmean_last_axis(windows[..., post[0]:post[1]])
    return post_mean - pre_mean


def _rolling_nanmean_1d(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_rolling_nanmean_1d expects a 1D array.")
    if x.size == 0:
        return np.array([], dtype=float)
    window = int(max(1, min(window, x.size)))
    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    numer = np.convolve(x0, kernel, mode="valid")
    denom = np.convolve(valid, kernel, mode="valid")
    out = np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=float), where=denom > 0)
    return out


def _peak_window_response(trace: np.ndarray, pre: tuple[int, int], post: tuple[int, int], peak_window_samples: int) -> float:
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("_peak_window_response expects a 1D trace.")
    pre_seg = trace[pre[0]:pre[1]]
    post_seg = trace[post[0]:post[1]]
    baseline = float(np.nanmean(pre_seg)) if pre_seg.size else np.nan
    if not np.isfinite(baseline) or post_seg.size == 0:
        return np.nan
    peak_mean = _rolling_nanmean_1d(post_seg, peak_window_samples)
    if peak_mean.size == 0 or not np.any(np.isfinite(peak_mean)):
        return np.nan
    return float(np.nanmax(peak_mean) - baseline)


def _sequence_metric_from_mean(
    mean_traces: np.ndarray,
    pre: tuple[int, int],
    post: tuple[int, int],
    peak_window_samples: int,
) -> np.ndarray:
    mean_traces = np.asarray(mean_traces, dtype=float)
    if mean_traces.ndim == 1:
        return np.array([_peak_window_response(mean_traces, pre=pre, post=post, peak_window_samples=peak_window_samples)], dtype=float)
    out = np.full(mean_traces.shape[:-1], np.nan, dtype=float)
    for idx in np.ndindex(mean_traces.shape[:-1]):
        out[idx] = _peak_window_response(mean_traces[idx], pre=pre, post=post, peak_window_samples=peak_window_samples)
    return out


def build_event_response_table(
    single_trial_npz: str | Path,
    config: GlutamateAnalysisConfig | None = None,
) -> pd.DataFrame:
    config = config or GlutamateAnalysisConfig()
    root = _load_npz_dict(single_trial_npz)
    meta = root.get("metadata", {})
    rows: list[dict[str, Any]] = []

    family_windows = {
        "image": (config.image_pre_samples, config.image_post_samples),
        "change": (config.change_pre_samples, config.change_post_samples),
        "omission": (config.omission_pre_samples, config.omission_post_samples),
    }

    for dmd_name, dmd_data in root.items():
        if not str(dmd_name).startswith("DMD"):
            continue
        synapse_ids = np.asarray(dmd_data.get("synapse_ids", []))

        for stim_name, windows in dmd_data["image_identity"].items():
            delta_auc = _window_metric(windows, *family_windows["image"])
            delta_mean = _window_delta_mean(windows, *family_windows["image"])
            n_events = windows.shape[0]
            for event_idx in range(n_events):
                for syn_idx, syn_id in enumerate(synapse_ids):
                    rows.append(
                        {
                            "session_id": meta.get("session_id"),
                            "subject_id": meta.get("subject_id"),
                            "dmd": dmd_name,
                            "synapse_id": str(syn_id),
                            "stimulus_family": "image",
                            "stimulus_name": str(stim_name),
                            "stimulus_label": _basename_stimulus(str(stim_name)),
                            "event_index": int(event_idx),
                            "delta_auc": float(delta_auc[event_idx, syn_idx]),
                            "delta_mean": float(delta_mean[event_idx, syn_idx]),
                        }
                    )

        for family in ("change", "omission"):
            windows = np.asarray(dmd_data[family], dtype=float)
            delta_auc = _window_metric(windows, *family_windows[family])
            delta_mean = _window_delta_mean(windows, *family_windows[family])
            n_events = windows.shape[0]
            for event_idx in range(n_events):
                for syn_idx, syn_id in enumerate(synapse_ids):
                    rows.append(
                        {
                            "session_id": meta.get("session_id"),
                            "subject_id": meta.get("subject_id"),
                            "dmd": dmd_name,
                            "synapse_id": str(syn_id),
                            "stimulus_family": family,
                            "stimulus_name": family,
                            "stimulus_label": family,
                            "event_index": int(event_idx),
                            "delta_auc": float(delta_auc[event_idx, syn_idx]),
                            "delta_mean": float(delta_mean[event_idx, syn_idx]),
                        }
                    )

    return pd.DataFrame(rows)


def classify_activation(
    single_trial_npz: str | Path,
    config: GlutamateAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or GlutamateAnalysisConfig()
    event_df = build_event_response_table(single_trial_npz=single_trial_npz, config=config)

    rows: list[dict[str, Any]] = []
    for (session_id, subject_id, dmd, synapse_id, family), grp in event_df.groupby(
        ["session_id", "subject_id", "dmd", "synapse_id", "stimulus_family"], sort=False
    ):
        values = grp["delta_auc"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        n_events = int(values.size)
        median_delta = float(np.nanmedian(values)) if n_events else np.nan
        mean_delta = float(np.nanmean(values)) if n_events else np.nan
        p_value = np.nan
        response_class = "no_change"
        effect_direction = "none"

        if n_events >= config.min_events_activation:
            p_value = _safe_wilcoxon_zero(values)
            if np.isfinite(p_value) and p_value < config.alpha:
                if median_delta > 0:
                    response_class = "activated"
                    effect_direction = "up"
                elif median_delta < 0:
                    response_class = "deactivated"
                    effect_direction = "down"

        rows.append(
            {
                "session_id": session_id,
                "subject_id": subject_id,
                "dmd": dmd,
                "synapse_id": str(synapse_id),
                "stimulus_family": family,
                "n_events": n_events,
                "median_delta_auc": median_delta,
                "mean_delta_auc": mean_delta,
                "effect_direction": effect_direction,
                "p_value": p_value,
                "response_class": response_class,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df["q_value_within_synapse"] = np.nan
    for (_, _, _), idx in summary_df.groupby(["session_id", "dmd", "synapse_id"]).groups.items():
        summary_df.loc[list(idx), "q_value_within_synapse"] = _bh_fdr(summary_df.loc[list(idx), "p_value"])

    summary_df["response_class"] = np.where(
        summary_df["q_value_within_synapse"].lt(config.alpha) & summary_df["median_delta_auc"].gt(0),
        "activated",
        np.where(
            summary_df["q_value_within_synapse"].lt(config.alpha) & summary_df["median_delta_auc"].lt(0),
            "deactivated",
            "no_change",
        ),
    )
    summary_df["effect_direction"] = np.where(
        summary_df["response_class"].eq("activated"),
        "up",
        np.where(summary_df["response_class"].eq("deactivated"), "down", "none"),
    )

    return event_df, summary_df


def _compute_fve(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    if values.size < 2:
        return np.nan
    overall_var = float(np.var(values))
    if overall_var <= EPS:
        return 0.0
    pred = np.empty_like(values)
    for lab in np.unique(labels):
        mask = labels == lab
        pred[mask] = np.mean(values[mask])
    resid = values - pred
    return float(1.0 - np.var(resid) / overall_var)


def _interp_nans_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if x.ndim != 1:
        raise ValueError("_interp_nans_1d expects a 1D array.")
    nans = ~np.isfinite(x)
    if not nans.any():
        return x
    good = np.flatnonzero(~nans)
    if good.size == 0:
        return x
    idx = np.arange(x.size)
    x[nans] = np.interp(idx[nans], idx[good], x[good])
    return x


def _select_manova_timepoints(n_obs: int, n_groups: int, n_time: int, max_timepoints: int) -> np.ndarray:
    max_dv = max(2, min(int(max_timepoints), int(n_obs - n_groups - 1), int(n_time)))
    if max_dv < 2:
        return np.array([], dtype=int)
    return np.linspace(0, n_time - 1, max_dv, dtype=int)


def _run_manova_trace_test(
    traces: np.ndarray,
    labels: np.ndarray,
    *,
    stat_name: str = "Wilks' lambda",
    max_timepoints: int = 20,
    interpolate_nans: bool = True,
    max_nan_fraction_per_trial: float = 0.1,
) -> dict[str, Any]:
    out = {
        "p_manova": np.nan,
        "f_manova": np.nan,
        "stat_value_manova": np.nan,
        "num_df_manova": np.nan,
        "den_df_manova": np.nan,
        "n_trials_manova": 0,
        "n_groups_manova": 0,
        "n_timepoints_used_manova": 0,
        "manova_stat_used": stat_name,
    }
    if MANOVA is None:
        return out

    X = np.asarray(traces, dtype=float)
    y = np.asarray(labels)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        return out

    keep = np.ones(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        frac_nan = np.mean(~np.isfinite(X[i]))
        if frac_nan > max_nan_fraction_per_trial:
            keep[i] = False
        elif interpolate_nans and frac_nan > 0:
            X[i] = _interp_nans_1d(X[i])

    X = X[keep]
    y = y[keep]
    if X.shape[0] < 3:
        return out

    finite_col = np.any(np.isfinite(X), axis=0)
    X = X[:, finite_col]
    if X.shape[1] < 2:
        return out
    col_var = np.nanvar(X, axis=0)
    X = X[:, col_var > 0]
    if X.shape[1] < 2:
        return out

    groups, counts = np.unique(y, return_counts=True)
    valid_groups = groups[counts >= 2]
    mask = np.isin(y, valid_groups)
    X = X[mask]
    y = y[mask]
    n_obs = X.shape[0]
    n_groups = np.unique(y).size
    if n_groups < 2 or n_obs <= n_groups + 2:
        return out

    idx = _select_manova_timepoints(n_obs=n_obs, n_groups=n_groups, n_time=X.shape[1], max_timepoints=max_timepoints)
    if idx.size < 2:
        return out
    Xs = X[:, idx]
    cols = [f"t{i:03d}" for i in range(Xs.shape[1])]
    wide = pd.DataFrame(Xs, columns=cols)
    wide["stimulus"] = pd.Categorical(y)

    lhs = " + ".join(cols)
    formula = f"{lhs} ~ C(stimulus)"
    try:
        manova = MANOVA.from_formula(formula, data=wide)
        res = manova.mv_test()
        stat = res.results["C(stimulus)"]["stat"].copy()
        if stat_name not in stat.index:
            stat_name_use = "Wilks' lambda" if "Wilks' lambda" in stat.index else str(stat.index[0])
        else:
            stat_name_use = stat_name
        row = stat.loc[stat_name_use]
        out.update(
            {
                "p_manova": float(pd.to_numeric(row.get("Pr > F"), errors="coerce")),
                "f_manova": float(pd.to_numeric(row.get("F Value"), errors="coerce")),
                "stat_value_manova": float(pd.to_numeric(row.get("Value"), errors="coerce")),
                "num_df_manova": float(pd.to_numeric(row.get("Num DF"), errors="coerce")),
                "den_df_manova": float(pd.to_numeric(row.get("Den DF"), errors="coerce")),
                "n_trials_manova": int(n_obs),
                "n_groups_manova": int(n_groups),
                "n_timepoints_used_manova": int(Xs.shape[1]),
                "manova_stat_used": stat_name_use,
            }
        )
    except Exception:
        return out
    return out


def _extract_image_trial_traces(
    single_trial_npz: str | Path,
    *,
    post_only: bool = False,
) -> pd.DataFrame:
    root = _load_npz_dict(single_trial_npz)
    meta = root.get("metadata", {})
    rows: list[dict[str, Any]] = []
    for dmd_name, dmd_data in root.items():
        if not str(dmd_name).startswith("DMD"):
            continue
        synapse_ids = np.asarray(dmd_data.get("synapse_ids", []))
        for stim_name, windows in dmd_data["image_identity"].items():
            windows = np.asarray(windows, dtype=float)
            if post_only:
                windows = windows[..., 50:100]
            n_events, n_syn, _ = windows.shape
            for event_idx in range(n_events):
                for syn_idx, syn_id in enumerate(synapse_ids):
                    rows.append(
                        {
                            "session_id": meta.get("session_id"),
                            "subject_id": meta.get("subject_id"),
                            "dmd": dmd_name,
                            "synapse_id": str(syn_id),
                            "stimulus_name": str(stim_name),
                            "stimulus_label": _basename_stimulus(str(stim_name)),
                            "event_index": int(event_idx),
                            "trace": windows[event_idx, syn_idx].astype(float, copy=True),
                        }
                    )
    return pd.DataFrame(rows)


def analyze_image_tuning(
    single_trial_npz: str | Path,
    activation_summary_df: pd.DataFrame,
    config: GlutamateAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or GlutamateAnalysisConfig()
    rng = np.random.default_rng(config.random_seed)
    event_df = build_event_response_table(single_trial_npz=single_trial_npz, config=config)
    img_df = event_df[event_df["stimulus_family"].eq("image")].copy()

    active = activation_summary_df[
        activation_summary_df["stimulus_family"].eq("image")
        & activation_summary_df["response_class"].eq("activated")
    ][["session_id", "dmd", "synapse_id"]].drop_duplicates()

    img_df = img_df.merge(active.assign(is_image_activated=True), on=["session_id", "dmd", "synapse_id"], how="inner")

    trace_df = _extract_image_trial_traces(
        single_trial_npz,
        post_only=config.manova_use_post_only,
    )
    if not trace_df.empty:
        trace_df = trace_df.merge(active.assign(is_image_activated=True), on=["session_id", "dmd", "synapse_id"], how="inner")

    per_image = (
        img_df.groupby(["session_id", "subject_id", "dmd", "synapse_id", "stimulus_name", "stimulus_label"], as_index=False)
        .agg(
            n_trials=("delta_auc", lambda x: int(np.isfinite(x).sum())),
            mean_response=("delta_auc", "mean"),
            median_response=("delta_auc", "median"),
            std_response=("delta_auc", "std"),
        )
    )

    rows: list[dict[str, Any]] = []
    for (session_id, subject_id, dmd, synapse_id), grp in img_df.groupby(
        ["session_id", "subject_id", "dmd", "synapse_id"], sort=False
    ):
        counts = grp.groupby("stimulus_name")["delta_auc"].apply(lambda x: int(np.isfinite(x).sum()))
        valid_images = counts[counts >= config.min_events_tuning_per_image].index.to_numpy()
        gvalid = grp[grp["stimulus_name"].isin(valid_images)].copy()
        values = gvalid["delta_auc"].to_numpy(dtype=float)
        labels = gvalid["stimulus_label"].to_numpy()

        fve = np.nan
        p_shuffle = np.nan
        p_kw = np.nan
        p_manova = np.nan
        f_manova = np.nan
        stat_value_manova = np.nan
        num_df_manova = np.nan
        den_df_manova = np.nan
        n_trials_manova = 0
        n_groups_manova = 0
        n_timepoints_used_manova = 0
        manova_stat_used = config.manova_stat

        is_tuned = False
        preferred_image = None
        preferred_mean = np.nan
        preferred_median = np.nan
        pref_vs_rest = np.nan
        pref_vs_next = np.nan

        if len(valid_images) >= config.min_images_for_tuning and np.isfinite(values).sum() > 1:
            fve = _compute_fve(values, labels)
            groups = [
                gvalid.loc[gvalid["stimulus_label"].eq(lbl), "delta_auc"].to_numpy(dtype=float)
                for lbl in np.unique(labels)
            ]
            p_kw = _safe_kruskal(groups)
            if np.isfinite(fve):
                null = np.empty(config.n_shuffles_tuning, dtype=float)
                for i in range(config.n_shuffles_tuning):
                    null[i] = _compute_fve(values, rng.permutation(labels))
                p_shuffle = float((1.0 + np.sum(null >= fve)) / (config.n_shuffles_tuning + 1.0))

            stats_by_image = (
                gvalid.groupby("stimulus_label")["delta_auc"]
                .agg([("mean_response", "mean"), ("median_response", "median")])
                .sort_values("mean_response", ascending=False)
            )
            if not stats_by_image.empty:
                preferred_image = str(stats_by_image.index[0])
                preferred_mean = float(stats_by_image.iloc[0]["mean_response"])
                preferred_median = float(stats_by_image.iloc[0]["median_response"])
                rest_vals = gvalid.loc[gvalid["stimulus_label"].ne(preferred_image), "delta_auc"].to_numpy(dtype=float)
                pref_vals = gvalid.loc[gvalid["stimulus_label"].eq(preferred_image), "delta_auc"].to_numpy(dtype=float)
                pref_vs_rest = float(np.nanmedian(pref_vals) - np.nanmedian(rest_vals)) if rest_vals.size else np.nan
                if len(stats_by_image) > 1:
                    pref_vs_next = float(stats_by_image.iloc[0]["mean_response"] - stats_by_image.iloc[1]["mean_response"])

            if not trace_df.empty:
                tgrp = trace_df[
                    (trace_df["session_id"] == session_id)
                    & (trace_df["dmd"] == dmd)
                    & (trace_df["synapse_id"] == synapse_id)
                    & (trace_df["stimulus_name"].isin(valid_images))
                ].copy()
                if not tgrp.empty:
                    traces = np.stack(tgrp["trace"].to_list(), axis=0)
                    manova = _run_manova_trace_test(
                        traces=traces,
                        labels=tgrp["stimulus_label"].to_numpy(),
                        stat_name=config.manova_stat,
                        max_timepoints=config.manova_max_timepoints,
                        interpolate_nans=config.manova_interpolate_nans,
                        max_nan_fraction_per_trial=config.manova_max_nan_fraction_per_trial,
                    )
                    p_manova = manova["p_manova"]
                    f_manova = manova["f_manova"]
                    stat_value_manova = manova["stat_value_manova"]
                    num_df_manova = manova["num_df_manova"]
                    den_df_manova = manova["den_df_manova"]
                    n_trials_manova = manova["n_trials_manova"]
                    n_groups_manova = manova["n_groups_manova"]
                    n_timepoints_used_manova = manova["n_timepoints_used_manova"]
                    manova_stat_used = manova["manova_stat_used"]

            method = str(config.tuning_method).lower()
            if method == "fve":
                is_tuned = bool(
                    np.isfinite(p_shuffle)
                    and np.isfinite(p_kw)
                    and p_shuffle < config.alpha
                    and p_kw < config.alpha
                    and np.isfinite(fve)
                    and fve >= config.tuning_min_effect_fve
                )
            elif method == "manova":
                is_tuned = bool(np.isfinite(p_manova) and p_manova < config.alpha)
            elif method == "hybrid":
                is_tuned = bool(
                    np.isfinite(p_manova)
                    and p_manova < config.alpha
                    and np.isfinite(fve)
                    and fve >= config.tuning_min_effect_fve
                )
            else:
                raise ValueError(f"Unknown tuning_method={config.tuning_method!r}. Use 'fve', 'manova', or 'hybrid'.")

        rows.append(
            {
                "session_id": session_id,
                "subject_id": subject_id,
                "dmd": dmd,
                "synapse_id": str(synapse_id),
                "n_image_trials": int(np.isfinite(values).sum()),
                "n_images_tested": int(len(valid_images)),
                "fve_image": fve,
                "p_shuffle_fve": p_shuffle,
                "p_kw": p_kw,
                "p_manova": p_manova,
                "f_manova": f_manova,
                "stat_value_manova": stat_value_manova,
                "num_df_manova": num_df_manova,
                "den_df_manova": den_df_manova,
                "n_trials_manova": n_trials_manova,
                "n_groups_manova": n_groups_manova,
                "n_timepoints_used_manova": n_timepoints_used_manova,
                "manova_stat_used": manova_stat_used,
                "preferred_image": preferred_image,
                "preferred_mean": preferred_mean,
                "preferred_median": preferred_median,
                "preferred_vs_rest_effect": pref_vs_rest,
                "preferred_vs_next_effect": pref_vs_next,
                "is_tuned": is_tuned,
                "tuning_method": str(config.tuning_method).lower(),
            }
        )

    tuning_df = pd.DataFrame(rows)
    if not tuning_df.empty:
        tuning_df["q_shuffle_fve"] = _bh_fdr(tuning_df["p_shuffle_fve"])
        tuning_df["q_kw"] = _bh_fdr(tuning_df["p_kw"])
        tuning_df["q_manova"] = _bh_fdr(tuning_df["p_manova"])
        method = str(config.tuning_method).lower()
        if method == "fve":
            tuning_df["is_tuned"] = (
                tuning_df["q_shuffle_fve"].lt(config.alpha)
                & tuning_df["q_kw"].lt(config.alpha)
                & tuning_df["fve_image"].ge(config.tuning_min_effect_fve)
            )
        elif method == "manova":
            tuning_df["is_tuned"] = tuning_df["q_manova"].lt(config.alpha)
        elif method == "hybrid":
            tuning_df["is_tuned"] = (
                tuning_df["q_manova"].lt(config.alpha)
                & tuning_df["fve_image"].ge(config.tuning_min_effect_fve)
            )

    return per_image, tuning_df


def _weighted_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        w = np.asarray(w, dtype=float)
        valid &= np.isfinite(w) & (w > 0)
    if valid.sum() < 2:
        return np.nan
    x = x[valid]
    y = y[valid]
    if w is None:
        w = np.ones_like(x)
    else:
        w = w[valid]
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    denom = np.sum(w * (x - xbar) ** 2)
    if denom <= EPS:
        return np.nan
    return float(np.sum(w * (x - xbar) * (y - ybar)) / denom)


def _normalize_sequence_responses(values: np.ndarray, strategy: str, r0: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    strategy = str(strategy).lower()
    if strategy == "none":
        return values.copy()
    if strategy == "max_abs":
        denom = np.nanmax(np.abs(values))
    else:
        denom = abs(float(r0))
    if not np.isfinite(denom) or denom <= EPS:
        return np.full_like(values, np.nan, dtype=float)
    return values / denom


def _assign_quantile_bins(positions: np.ndarray, n_bins: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    n = positions.size
    out = np.full(n, -1, dtype=int)
    if n == 0:
        return out
    n_bins = int(max(1, min(n_bins, n)))
    order = np.argsort(positions)
    splits = np.array_split(order, n_bins)
    for bin_idx, inds in enumerate(splits):
        out[np.asarray(inds, dtype=int)] = int(bin_idx)
    return out


def _epoch_label_from_bin(bin_idx: int, n_bins: int) -> str:
    if n_bins <= 1:
        return "all"
    if bin_idx == 0:
        return "early"
    if bin_idx == n_bins - 1:
        return "late"
    return f"mid_{bin_idx}"


def _summarize_binned_sequence(
    positions: np.ndarray,
    responses: np.ndarray,
    responses_norm: np.ndarray,
    counts: np.ndarray,
    n_bins: int,
) -> tuple[pd.DataFrame, float, float, float, float, float, float]:
    positions = np.asarray(positions, dtype=float)
    responses = np.asarray(responses, dtype=float)
    responses_norm = np.asarray(responses_norm, dtype=float)
    counts = np.asarray(counts, dtype=float)

    bin_ids = _assign_quantile_bins(positions, n_bins=n_bins)
    rows = []
    for bin_idx in sorted(np.unique(bin_ids)):
        mask = bin_ids == bin_idx
        pos_bin = positions[mask]
        resp_bin = responses[mask]
        norm_bin = responses_norm[mask]
        cnt_bin = counts[mask]
        valid_resp = np.isfinite(resp_bin) & np.isfinite(cnt_bin) & (cnt_bin > 0)
        if not np.any(valid_resp):
            continue
        pos_center = float(np.average(pos_bin[valid_resp], weights=cnt_bin[valid_resp]))
        mean_resp = float(np.average(resp_bin[valid_resp], weights=cnt_bin[valid_resp]))
        valid_norm = valid_resp & np.isfinite(norm_bin)
        mean_norm = float(np.average(norm_bin[valid_norm], weights=cnt_bin[valid_norm])) if np.any(valid_norm) else np.nan
        rows.append(
            {
                "bin_index": int(bin_idx),
                "epoch_label": _epoch_label_from_bin(int(bin_idx), int(bin_ids.max()) + 1),
                "position_center": pos_center,
                "response_amplitude": mean_resp,
                "response_amplitude_norm": mean_norm,
                "counts": float(np.nansum(cnt_bin[valid_resp])),
                "n_positions_in_bin": int(np.sum(mask)),
            }
        )

    binned = pd.DataFrame(rows)
    if binned.empty:
        return binned, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    overall_slope = _weighted_slope(
        binned["position_center"].to_numpy(dtype=float),
        binned["response_amplitude"].to_numpy(dtype=float),
        binned["counts"].to_numpy(dtype=float),
    )
    overall_slope_norm = _weighted_slope(
        binned["position_center"].to_numpy(dtype=float),
        binned["response_amplitude_norm"].to_numpy(dtype=float),
        binned["counts"].to_numpy(dtype=float),
    )

    if len(binned) >= 2:
        split = max(2, int(np.ceil(len(binned) / 2.0)))
        early_df = binned.iloc[:split].copy()
        late_df = binned.iloc[-split:].copy()
        early_slope = _weighted_slope(
            early_df["position_center"].to_numpy(dtype=float),
            early_df["response_amplitude"].to_numpy(dtype=float),
            early_df["counts"].to_numpy(dtype=float),
        )
        late_slope = _weighted_slope(
            late_df["position_center"].to_numpy(dtype=float),
            late_df["response_amplitude"].to_numpy(dtype=float),
            late_df["counts"].to_numpy(dtype=float),
        )
    else:
        early_slope = np.nan
        late_slope = np.nan

    early_mean = float(binned.iloc[0]["response_amplitude"])
    late_mean = float(binned.iloc[-1]["response_amplitude"])
    return binned, overall_slope, overall_slope_norm, early_slope, late_slope, early_mean, late_mean


def _classify_sequence_pattern(
    early_slope: float,
    late_slope: float,
    overall_slope: float,
    *,
    min_abs_slope: float,
    slope_frac: float,
) -> str:
    vals = np.array([early_slope, late_slope, overall_slope], dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return "stable"
    max_abs = float(np.nanmax(np.abs(vals[finite])))
    thr = max(float(min_abs_slope), float(slope_frac) * max_abs)

    def _state(v: float) -> int:
        if not np.isfinite(v) or abs(v) < thr:
            return 0
        return 1 if v > 0 else -1

    e = _state(early_slope)
    l = _state(late_slope)
    o = _state(overall_slope)

    if e == 0 and l == 0 and o == 0:
        return "stable"
    if e < 0 and l > 0:
        return "biphasic_rebound"
    if e > 0 and l < 0:
        return "biphasic_adapting"
    if e < 0 and l == 0:
        return "early_adapting"
    if e > 0 and l == 0:
        return "early_facilitating"
    if e == 0 and l < 0:
        return "late_adapting"
    if e == 0 and l > 0:
        return "late_facilitating"
    if e < 0 and l < 0:
        return "adapting"
    if e > 0 and l > 0:
        return "facilitating"
    if o < 0:
        return "adapting"
    if o > 0:
        return "facilitating"
    return "mixed"


def _build_sequence_rank_table(
    tuning_per_image_df: pd.DataFrame | None,
    tuning_summary_df: pd.DataFrame | None,
    *,
    rank_by: str = "hybrid_preference",
) -> pd.DataFrame:
    base_cols = ["session_id", "dmd", "synapse_id", "stimulus_name"]
    if tuning_per_image_df is None or tuning_per_image_df.empty:
        return pd.DataFrame(
            columns=base_cols
            + [
                "stimulus_label",
                "image_selectivity_score",
                "ranking_score",
                "image_rank_within_synapse",
                "rank_basis",
                "is_preferred_ranked_image",
            ]
        )

    rank_df = tuning_per_image_df.copy()
    if "subject_id" in rank_df.columns:
        rank_df = rank_df.drop(columns=["subject_id"])

    if tuning_summary_df is not None and not tuning_summary_df.empty:
        rank_df = rank_df.merge(
            tuning_summary_df[["session_id", "dmd", "synapse_id", "is_tuned"]],
            on=["session_id", "dmd", "synapse_id"],
            how="left",
        )
    else:
        rank_df["is_tuned"] = False

    rank_df["is_tuned"] = rank_df["is_tuned"].fillna(False)
    rank_df["image_selectivity_score"] = np.nan

    group_keys = ["session_id", "dmd", "synapse_id"]
    for _, idx in rank_df.groupby(group_keys).groups.items():
        idx = list(idx)
        vals = rank_df.loc[idx, "mean_response"].to_numpy(dtype=float)
        scores = np.full(len(idx), np.nan, dtype=float)
        for i in range(len(idx)):
            others = np.delete(vals, i)
            scores[i] = vals[i] if others.size == 0 else vals[i] - float(np.nanmean(others))
        rank_df.loc[idx, "image_selectivity_score"] = scores

    rank_scores = []
    rank_basis = []
    strategy = str(rank_by).lower()
    for _, row in rank_df.iterrows():
        tuned = bool(row["is_tuned"])
        if strategy == "selectivity_score":
            rank_scores.append(row["image_selectivity_score"])
            rank_basis.append("selectivity_score")
        elif strategy == "response_amplitude":
            rank_scores.append(row["mean_response"])
            rank_basis.append("response_amplitude")
        else:
            if tuned:
                rank_scores.append(row["image_selectivity_score"])
                rank_basis.append("selectivity_score")
            else:
                rank_scores.append(row["mean_response"])
                rank_basis.append("response_amplitude")

    rank_df["ranking_score"] = rank_scores
    rank_df["rank_basis"] = rank_basis
    rank_df["image_rank_within_synapse"] = (
        rank_df.groupby(group_keys)["ranking_score"]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    rank_df["is_preferred_ranked_image"] = rank_df["image_rank_within_synapse"].eq(1)

    return rank_df[
        [
            "session_id",
            "dmd",
            "synapse_id",
            "stimulus_name",
            "stimulus_label",
            "image_selectivity_score",
            "ranking_score",
            "image_rank_within_synapse",
            "rank_basis",
            "is_preferred_ranked_image",
        ]
    ].copy()


def analyze_sequence_dynamics(
    sequence_npz: str | Path,
    activation_summary_df: pd.DataFrame,
    config: GlutamateAnalysisConfig | None = None,
    tuning_per_image_df: pd.DataFrame | None = None,
    tuning_summary_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or GlutamateAnalysisConfig()
    root = _load_npz_dict(sequence_npz)
    meta = root.get("metadata", {})

    active = activation_summary_df[
        activation_summary_df["stimulus_family"].eq("image")
        & activation_summary_df["response_class"].eq("activated")
    ][["session_id", "dmd", "synapse_id"]].drop_duplicates()

    rank_df = _build_sequence_rank_table(
        tuning_per_image_df=tuning_per_image_df,
        tuning_summary_df=tuning_summary_df,
        rank_by=config.sequence_rank_by,
    )

    position_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dmd_name, dmd_data in root.items():
        if not str(dmd_name).startswith("DMD"):
            continue
        synapse_ids = np.asarray(dmd_data.get("synapse_ids", []))

        for syn_idx, syn_id in enumerate(synapse_ids):
            active_mask = (
                active["session_id"].eq(meta.get("session_id"))
                & active["dmd"].eq(dmd_name)
                & active["synapse_id"].eq(str(syn_id))
            )
            if not active_mask.any():
                continue

            overall_slopes: list[float] = []
            early_slopes: list[float] = []
            late_slopes: list[float] = []
            image_adaptation: list[float] = []
            image_terminal_jump: list[float] = []
            image_r0: list[float] = []
            image_rlast: list[float] = []
            image_rterminal: list[float] = []
            image_early_minus_late: list[float] = []

            for stim_name, seq_data in dmd_data["image_identity"].items():
                repeated = seq_data["repeated"]
                repeated_mean = np.asarray(repeated["mean"], dtype=float)[:, syn_idx, :]
                positions = np.asarray(repeated["positions"], dtype=float)
                counts = np.asarray(repeated["counts"], dtype=float)

                repeated_resp = _sequence_metric_from_mean(
                    repeated_mean,
                    pre=config.sequence_pre_samples,
                    post=config.sequence_post_samples,
                    peak_window_samples=config.sequence_peak_window_samples,
                )

                valid = (
                    np.isfinite(repeated_resp)
                    & np.isfinite(positions)
                    & np.isfinite(counts)
                    & (counts >= config.sequence_min_count_per_position)
                )
                if valid.sum() < config.min_positions_for_sequence:
                    continue

                pos_valid = positions[valid]
                resp_valid = repeated_resp[valid]
                counts_valid = counts[valid]

                order = np.argsort(pos_valid)
                pos_valid = pos_valid[order]
                resp_valid = resp_valid[order]
                counts_valid = counts_valid[order]

                r0 = float(resp_valid[0])
                rlast = float(resp_valid[-1])

                resp_norm = _normalize_sequence_responses(
                    resp_valid,
                    strategy=config.sequence_norm_strategy,
                    r0=r0,
                )

                (
                    binned_df,
                    overall_slope,
                    overall_slope_norm,
                    early_slope,
                    late_slope,
                    early_mean,
                    late_mean,
                ) = _summarize_binned_sequence(
                    positions=pos_valid,
                    responses=resp_valid,
                    responses_norm=resp_norm,
                    counts=counts_valid,
                    n_bins=config.sequence_n_quantile_bins,
                )
                if binned_df.empty:
                    continue

                adaptation_idx = float((r0 - rlast) / (abs(r0) + abs(rlast) + EPS))

                terminal_mean = np.asarray(seq_data["terminal"]["mean"], dtype=float)[syn_idx, :]
                rterminal = float(
                    _sequence_metric_from_mean(
                        terminal_mean,
                        pre=config.sequence_pre_samples,
                        post=config.sequence_post_samples,
                        peak_window_samples=config.sequence_peak_window_samples,
                    )[0]
                )
                terminal_jump = float(rterminal - rlast)

                sequence_label = _classify_sequence_pattern(
                    early_slope=early_slope,
                    late_slope=late_slope,
                    overall_slope=overall_slope,
                    min_abs_slope=config.sequence_label_min_abs_slope,
                    slope_frac=config.sequence_label_slope_frac,
                )

                overall_slopes.append(overall_slope)
                early_slopes.append(early_slope)
                late_slopes.append(late_slope)
                image_adaptation.append(adaptation_idx)
                image_terminal_jump.append(terminal_jump)
                image_r0.append(r0)
                image_rlast.append(rlast)
                image_rterminal.append(rterminal)
                image_early_minus_late.append(early_mean - late_mean)

                bin_lookup = {
                    int(row["bin_index"]): {
                        "epoch_label": row["epoch_label"],
                        "binned_position_center": row["position_center"],
                        "binned_response_amplitude": row["response_amplitude"],
                        "binned_response_amplitude_norm": row["response_amplitude_norm"],
                        "binned_counts": row["counts"],
                    }
                    for _, row in binned_df.iterrows()
                }
                raw_bin_ids = _assign_quantile_bins(pos_valid, n_bins=config.sequence_n_quantile_bins)

                for p, rv, rvn, c, bin_idx in zip(pos_valid, resp_valid, resp_norm, counts_valid, raw_bin_ids):
                    info = bin_lookup.get(int(bin_idx), {})
                    position_rows.append(
                        {
                            "session_id": meta.get("session_id"),
                            "subject_id": meta.get("subject_id"),
                            "dmd": dmd_name,
                            "synapse_id": str(syn_id),
                            "stimulus_name": str(stim_name),
                            "stimulus_label": _basename_stimulus(str(stim_name)),
                            "position_category": "repeated",
                            "sequence_position": int(p),
                            "counts": float(c),
                            "n_sequences": int(repeated.get("n_sequences", 0)),
                            "response_amplitude": float(rv),
                            "response_amplitude_norm": float(rvn) if np.isfinite(rvn) else np.nan,
                            "delta_from_r0": float(rv - r0),
                            "r0": r0,
                            "sequence_slope": overall_slope,
                            "sequence_slope_norm": overall_slope_norm,
                            "overall_slope": overall_slope,
                            "overall_slope_norm": overall_slope_norm,
                            "early_slope": early_slope,
                            "late_slope": late_slope,
                            "sequence_label": sequence_label,
                            "quantile_bin": int(bin_idx),
                            "epoch_label": info.get("epoch_label"),
                            "binned_position_center": info.get("binned_position_center"),
                            "binned_response_amplitude": info.get("binned_response_amplitude"),
                            "binned_response_amplitude_norm": info.get("binned_response_amplitude_norm"),
                            "binned_counts": info.get("binned_counts"),
                        }
                    )

                terminal_norm = _normalize_sequence_responses(
                    np.array([rterminal], dtype=float),
                    strategy=config.sequence_norm_strategy,
                    r0=r0,
                )[0]
                position_rows.append(
                    {
                        "session_id": meta.get("session_id"),
                        "subject_id": meta.get("subject_id"),
                        "dmd": dmd_name,
                        "synapse_id": str(syn_id),
                        "stimulus_name": str(stim_name),
                        "stimulus_label": _basename_stimulus(str(stim_name)),
                        "position_category": "terminal",
                        "sequence_position": int(np.nanmax(pos_valid) + 1),
                        "counts": float(repeated.get("n_sequences", 0)),
                        "n_sequences": int(repeated.get("n_sequences", 0)),
                        "response_amplitude": rterminal,
                        "response_amplitude_norm": float(terminal_norm) if np.isfinite(terminal_norm) else np.nan,
                        "delta_from_r0": float(rterminal - r0),
                        "r0": r0,
                        "sequence_slope": overall_slope,
                        "sequence_slope_norm": overall_slope_norm,
                        "overall_slope": overall_slope,
                        "overall_slope_norm": overall_slope_norm,
                        "early_slope": early_slope,
                        "late_slope": late_slope,
                        "sequence_label": sequence_label,
                        "quantile_bin": np.nan,
                        "epoch_label": "terminal",
                        "binned_position_center": np.nan,
                        "binned_response_amplitude": np.nan,
                        "binned_response_amplitude_norm": np.nan,
                        "binned_counts": np.nan,
                    }
                )

                per_image_rows.append(
                    {
                        "session_id": meta.get("session_id"),
                        "subject_id": meta.get("subject_id"),
                        "dmd": dmd_name,
                        "synapse_id": str(syn_id),
                        "stimulus_name": str(stim_name),
                        "stimulus_label": _basename_stimulus(str(stim_name)),
                        "n_positions": int(valid.sum()),
                        "n_sequences": int(repeated.get("n_sequences", 0)),
                        "r0": r0,
                        "rlast": rlast,
                        "rterminal": rterminal,
                        "terminal_minus_last": terminal_jump,
                        "early_mean": early_mean,
                        "late_mean": late_mean,
                        "early_minus_late": float(early_mean - late_mean),
                        "adaptation_index": adaptation_idx,
                        "sequence_slope": overall_slope,
                        "sequence_slope_norm": overall_slope_norm,
                        "overall_slope": overall_slope,
                        "overall_slope_norm": overall_slope_norm,
                        "early_slope": early_slope,
                        "late_slope": late_slope,
                        "sequence_label": sequence_label,
                    }
                )

            slopes = np.asarray(overall_slopes, dtype=float)
            if slopes.size == 0:
                continue

            median_overall = float(np.nanmedian(slopes))
            median_early = float(np.nanmedian(np.asarray(early_slopes, dtype=float)))
            median_late = float(np.nanmedian(np.asarray(late_slopes, dtype=float)))
            p_slope = _safe_wilcoxon_zero(slopes)

            summary_rows.append(
                {
                    "session_id": meta.get("session_id"),
                    "subject_id": meta.get("subject_id"),
                    "dmd": dmd_name,
                    "synapse_id": str(syn_id),
                    "n_images_with_sequences": int(slopes.size),
                    "median_seq_slope": median_overall,
                    "median_overall_slope": median_overall,
                    "median_early_slope": median_early,
                    "median_late_slope": median_late,
                    "median_adaptation_index": float(np.nanmedian(image_adaptation)),
                    "median_r0": float(np.nanmedian(image_r0)),
                    "median_rlast": float(np.nanmedian(image_rlast)),
                    "median_rterminal": float(np.nanmedian(image_rterminal)),
                    "median_terminal_minus_last": float(np.nanmedian(image_terminal_jump)),
                    "median_early_minus_late": float(np.nanmedian(image_early_minus_late)),
                    "seq_p": p_slope,
                    "sequence_class": _classify_sequence_pattern(
                        early_slope=median_early,
                        late_slope=median_late,
                        overall_slope=median_overall,
                        min_abs_slope=config.sequence_label_min_abs_slope,
                        slope_frac=config.sequence_label_slope_frac,
                    ),
                }
            )

    position_df = pd.DataFrame(position_rows)
    per_image_df = pd.DataFrame(per_image_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not rank_df.empty:
        if not per_image_df.empty:
            per_image_df = per_image_df.merge(
                rank_df,
                on=["session_id", "dmd", "synapse_id", "stimulus_name", "stimulus_label"],
                how="left",
            )
        if not position_df.empty:
            position_df = position_df.merge(
                rank_df,
                on=["session_id", "dmd", "synapse_id", "stimulus_name", "stimulus_label"],
                how="left",
            )

    if not summary_df.empty:
        summary_df["seq_q"] = _bh_fdr(summary_df["seq_p"])
    else:
        summary_df["seq_q"] = []

    return position_df, per_image_df, summary_df


def save_analysis_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, df in tables.items():
        csv_path = output_path / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        written[f"{name}_csv"] = csv_path
        parquet_path = output_path / f"{name}.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            written[f"{name}_parquet"] = parquet_path
        except Exception:
            pass
    if metadata is not None:
        meta_path = output_path / "glutamate_analysis_metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        written["metadata_json"] = meta_path
    return written


def run_glutamate_analysis(
    session_dir_or_analysis_dir: str | Path,
    output_dir: str | Path | None = None,
    config: GlutamateAnalysisConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or GlutamateAnalysisConfig()
    paths = resolve_glutamate_analysis_paths(session_dir_or_analysis_dir, output_dir=output_dir)

    activation_event_df, activation_summary_df = classify_activation(paths.single_trial_npz, config=config)
    tuning_per_image_df, tuning_summary_df = analyze_image_tuning(
        paths.single_trial_npz,
        activation_summary_df=activation_summary_df,
        config=config,
    )
    sequence_position_df, sequence_per_image_df, sequence_summary_df = analyze_sequence_dynamics(
        paths.sequence_npz,
        activation_summary_df=activation_summary_df,
        config=config,
        tuning_per_image_df=tuning_per_image_df,
        tuning_summary_df=tuning_summary_df,
    )

    metadata = {
        "analysis_name": "glutamate_response_analysis",
        "config": asdict(config),
        "inputs": {
            "single_trial_npz": str(paths.single_trial_npz),
            "mean_npz": str(paths.mean_npz),
            "sequence_npz": str(paths.sequence_npz),
        },
        "outputs": {
            "output_dir": str(paths.output_dir),
        },
    }

    save_analysis_tables(
        {
            "activation_event_table": activation_event_df,
            "activation_summary_table": activation_summary_df,
            "tuning_per_image_table": tuning_per_image_df,
            "tuning_summary_table": tuning_summary_df,
            "sequence_position_table": sequence_position_df,
            "sequence_per_image_table": sequence_per_image_df,
            "sequence_summary_table": sequence_summary_df,
        },
        output_dir=paths.output_dir,
        metadata=metadata,
    )

    return {
        "activation_event_table": activation_event_df,
        "activation_summary_table": activation_summary_df,
        "tuning_per_image_table": tuning_per_image_df,
        "tuning_summary_table": tuning_summary_df,
        "sequence_position_table": sequence_position_df,
        "sequence_per_image_table": sequence_per_image_df,
        "sequence_summary_table": sequence_summary_df,
        "metadata": pd.DataFrame([metadata]),
    }


__all__ = [
    "GlutamateAnalysisConfig",
    "GlutamateAnalysisPaths",
    "resolve_glutamate_analysis_paths",
    "build_event_response_table",
    "classify_activation",
    "analyze_image_tuning",
    "analyze_sequence_dynamics",
    "save_analysis_tables",
    "run_glutamate_analysis",
]