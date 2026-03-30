from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


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
    sequence_early_positions: tuple[int, int] = (0, 1)
    sequence_late_n_positions: int = 2
    tuning_min_effect_fve: float = 0.05


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
    out = Path(output_dir) if output_dir is not None else analysis_dir / "derived" / "glutamate_analysis"
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


def _sequence_metric_from_mean(mean_traces: np.ndarray, pre: tuple[int, int], post: tuple[int, int]) -> np.ndarray:
    return _window_metric(mean_traces, pre=pre, post=post)


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
    for (session_id, dmd, synapse_id), idx in summary_df.groupby(["session_id", "dmd", "synapse_id"]).groups.items():
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
        counts = grp.groupby("stimulus_name")["delta_auc"].apply(lambda x: np.isfinite(x).sum())
        valid_images = counts[counts >= config.min_events_tuning_per_image].index.to_numpy()
        gvalid = grp[grp["stimulus_name"].isin(valid_images)].copy()
        values = gvalid["delta_auc"].to_numpy(dtype=float)
        labels = gvalid["stimulus_label"].to_numpy()

        fve = np.nan
        p_shuffle = np.nan
        p_kw = np.nan
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

            is_tuned = bool(
                np.isfinite(p_shuffle)
                and np.isfinite(p_kw)
                and p_shuffle < config.alpha
                and p_kw < config.alpha
                and np.isfinite(fve)
                and fve >= config.tuning_min_effect_fve
            )

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
                "preferred_image": preferred_image,
                "preferred_mean": preferred_mean,
                "preferred_median": preferred_median,
                "preferred_vs_rest_effect": pref_vs_rest,
                "preferred_vs_next_effect": pref_vs_next,
                "is_tuned": is_tuned,
            }
        )

    tuning_df = pd.DataFrame(rows)
    if not tuning_df.empty:
        tuning_df["q_shuffle_fve"] = _bh_fdr(tuning_df["p_shuffle_fve"])
        tuning_df["q_kw"] = _bh_fdr(tuning_df["p_kw"])
        tuning_df["is_tuned"] = (
            tuning_df["q_shuffle_fve"].lt(config.alpha)
            & tuning_df["q_kw"].lt(config.alpha)
            & tuning_df["fve_image"].ge(config.tuning_min_effect_fve)
        )
    else:
        tuning_df["q_shuffle_fve"] = []
        tuning_df["q_kw"] = []

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


def analyze_sequence_dynamics(
    sequence_npz: str | Path,
    activation_summary_df: pd.DataFrame,
    config: GlutamateAnalysisConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or GlutamateAnalysisConfig()
    root = _load_npz_dict(sequence_npz)
    meta = root.get("metadata", {})

    active = activation_summary_df[
        activation_summary_df["stimulus_family"].eq("image")
        & activation_summary_df["response_class"].eq("activated")
    ][["session_id", "dmd", "synapse_id"]].drop_duplicates()

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

            image_slopes: list[float] = []
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
                )
                valid = np.isfinite(repeated_resp) & (counts > 0)
                if valid.sum() < config.min_positions_for_sequence:
                    continue

                pos_valid = positions[valid]
                resp_valid = repeated_resp[valid]
                counts_valid = counts[valid]
                slope = _weighted_slope(pos_valid, resp_valid, counts_valid)
                if not np.isfinite(slope):
                    continue

                order = np.argsort(pos_valid)
                pos_valid = pos_valid[order]
                resp_valid = resp_valid[order]
                counts_valid = counts_valid[order]
                r0 = float(resp_valid[0])
                rlast = float(resp_valid[-1])
                late_n = min(config.sequence_late_n_positions, resp_valid.size)
                early_mask = np.isin(pos_valid, np.asarray(config.sequence_early_positions))
                early_vals = resp_valid[early_mask]
                if early_vals.size == 0:
                    early_vals = resp_valid[: min(2, resp_valid.size)]
                late_vals = resp_valid[-late_n:]
                early_mean = float(np.nanmean(early_vals))
                late_mean = float(np.nanmean(late_vals))
                adaptation_idx = float((r0 - rlast) / (abs(r0) + abs(rlast) + EPS))

                terminal_mean = np.asarray(seq_data["terminal"]["mean"], dtype=float)[syn_idx, :]
                rterminal = float(
                    _sequence_metric_from_mean(
                        terminal_mean,
                        pre=config.sequence_pre_samples,
                        post=config.sequence_post_samples,
                    )
                )
                terminal_jump = float(rterminal - rlast)

                image_slopes.append(slope)
                image_adaptation.append(adaptation_idx)
                image_terminal_jump.append(terminal_jump)
                image_r0.append(r0)
                image_rlast.append(rlast)
                image_rterminal.append(rterminal)
                image_early_minus_late.append(early_mean - late_mean)

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
                        "sequence_slope": slope,
                    }
                )

            slopes = np.asarray(image_slopes, dtype=float)
            if slopes.size == 0:
                continue
            p_slope = _safe_wilcoxon_zero(slopes)
            median_slope = float(np.nanmedian(slopes))
            seq_class = "stable"
            if np.isfinite(p_slope) and p_slope < config.alpha:
                if median_slope > 0:
                    seq_class = "facilitating"
                elif median_slope < 0:
                    seq_class = "adapting"

            summary_rows.append(
                {
                    "session_id": meta.get("session_id"),
                    "subject_id": meta.get("subject_id"),
                    "dmd": dmd_name,
                    "synapse_id": str(syn_id),
                    "n_images_with_sequences": int(slopes.size),
                    "median_seq_slope": median_slope,
                    "median_adaptation_index": float(np.nanmedian(image_adaptation)),
                    "median_r0": float(np.nanmedian(image_r0)),
                    "median_rlast": float(np.nanmedian(image_rlast)),
                    "median_rterminal": float(np.nanmedian(image_rterminal)),
                    "median_terminal_minus_last": float(np.nanmedian(image_terminal_jump)),
                    "median_early_minus_late": float(np.nanmedian(image_early_minus_late)),
                    "seq_p": p_slope,
                    "sequence_class": seq_class,
                }
            )

    per_image_df = pd.DataFrame(per_image_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["seq_q"] = _bh_fdr(summary_df["seq_p"])
        summary_df["sequence_class"] = np.where(
            summary_df["seq_q"].lt(config.alpha) & summary_df["median_seq_slope"].gt(0),
            "facilitating",
            np.where(
                summary_df["seq_q"].lt(config.alpha) & summary_df["median_seq_slope"].lt(0),
                "adapting",
                "stable",
            ),
        )
    else:
        summary_df["seq_q"] = []
    return per_image_df, summary_df


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
    sequence_per_image_df, sequence_summary_df = analyze_sequence_dynamics(
        paths.sequence_npz,
        activation_summary_df=activation_summary_df,
        config=config,
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
