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

#========================== VOLTAGE ANALYSIS ============================================================================

#========================== SPIKE FEATURE TABLES ========================================================================

import json
from itertools import combinations

import h5py
import pandas as pd


def _crossing_time(time_ms, waveform, peak_idx, fraction, side):
    target = fraction * waveform[peak_idx]
    if not np.isfinite(target) or target <= 0:
        return np.nan

    if side == "left":
        hits = np.flatnonzero(waveform[:peak_idx + 1] <= target)
        if not len(hits) or hits[-1] >= peak_idx:
            return np.nan
        i, j = hits[-1], hits[-1] + 1
    else:
        hits = np.flatnonzero(waveform[peak_idx + 1:] <= target)
        if not len(hits):
            return np.nan
        j = peak_idx + 1 + hits[0]
        i = j - 1

    y0, y1 = waveform[i], waveform[j]
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
        return float(time_ms[i])
    alpha = (target - y0) / (y1 - y0)
    return float(time_ms[i] + alpha * (time_ms[j] - time_ms[i]))


def _trapz(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    if len(y) < 2:
        return np.nan
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))


def _nearest_local_maximum(y, index, radius):
    lo, hi = max(0, index - radius), min(len(y), index + radius + 1)
    segment = np.asarray(y[lo:hi], float)
    if len(segment) < 3:
        return int(index)
    local = np.flatnonzero(
        (segment[1:-1] >= segment[:-2]) &
        (segment[1:-1] > segment[2:])
    ) + 1
    if not len(local):
        return int(index)
    absolute = lo + local
    return int(absolute[np.argmin(np.abs(absolute - index))])


def _waveform_features(
    y,
    spike_index,
    fs,
    *,
    pre_ms,
    post_ms,
    peak_refine_ms,
    baseline_window_ms,
    auc_window_ms,
):
    pre = int(round(pre_ms * fs / 1000))
    post = int(round(post_ms * fs / 1000))
    refine = int(round(peak_refine_ms * fs / 1000))
    peak = _nearest_local_maximum(y, int(spike_index), refine)

    if peak - pre < 0 or peak + post >= len(y):
        return {}

    b0 = peak + int(round(baseline_window_ms[0] * fs / 1000))
    b1 = peak + int(round(baseline_window_ms[1] * fs / 1000))
    if b0 < 0 or b1 <= b0:
        return {}

    baseline = float(np.nanmedian(y[b0:b1]))
    waveform = np.asarray(y[peak - pre:peak + post + 1], float) - baseline
    time_ms = np.arange(-pre, post + 1) / fs * 1000
    center = pre
    amplitude = float(waveform[center])

    if not np.isfinite(amplitude) or amplitude <= 0:
        return {
            "raw_peak_index": peak,
            "spike_amplitude_dff": amplitude,
            "spike_auc_dff_ms": np.nan,
            "spike_half_width_ms": np.nan,
            "spike_rise10_90_ms": np.nan,
            "spike_decay90_10_ms": np.nan,
        }

    l10 = _crossing_time(time_ms, waveform, center, 0.10, "left")
    l50 = _crossing_time(time_ms, waveform, center, 0.50, "left")
    l90 = _crossing_time(time_ms, waveform, center, 0.90, "left")
    r90 = _crossing_time(time_ms, waveform, center, 0.90, "right")
    r50 = _crossing_time(time_ms, waveform, center, 0.50, "right")
    r10 = _crossing_time(time_ms, waveform, center, 0.10, "right")

    auc_mask = (
        (time_ms >= auc_window_ms[0]) &
        (time_ms <= auc_window_ms[1]) &
        np.isfinite(waveform)
    )

    return {
        "raw_peak_index": peak,
        "spike_amplitude_dff": amplitude,
        "spike_auc_dff_ms": _trapz(waveform[auc_mask], time_ms[auc_mask]),
        "spike_half_width_ms": r50 - l50 if np.isfinite(r50) and np.isfinite(l50) else np.nan,
        "spike_rise10_90_ms": l90 - l10 if np.isfinite(l90) and np.isfinite(l10) else np.nan,
        "spike_decay90_10_ms": r10 - r90 if np.isfinite(r10) and np.isfinite(r90) else np.nan,
    }


def _observed_intervals(timebase, sample_epoch=None):
    t = np.asarray(timebase, float)
    if len(t) < 2:
        return []

    dt = float(np.nanmedian(np.diff(t)))
    epoch = (
        np.ones(len(t), dtype=int)
        if sample_epoch is None
        else np.asarray(sample_epoch)
    )

    breaks = (
        (epoch[1:] != epoch[:-1]) |
        (np.diff(t) > 5 * dt) |
        (np.diff(t) <= 0)
    )
    edges = np.flatnonzero(breaks) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(t)]

    return [
        (float(t[a]), float(t[b - 1] + dt))
        for a, b in zip(starts, stops)
        if epoch[a] > 0 and b > a
    ]


def _intersect_intervals(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        stop = min(a[i][1], b[j][1])
        if stop > start:
            out.append((start, stop))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _filter_intervals(times, intervals):
    times = np.asarray(times, float)
    if not len(times) or not intervals:
        return np.array([], float)
    keep = np.zeros(len(times), bool)
    for start, stop in intervals:
        keep |= (times >= start) & (times < stop)
    return times[keep]


def _fraction_near(a, b, dt):
    a, b = np.asarray(a, float), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return np.nan
    j = np.searchsorted(b, a)
    right = (j < len(b)) & (np.abs(b[np.minimum(j, len(b) - 1)] - a) <= dt)
    jm1 = np.maximum(j - 1, 0)
    left = (j > 0) & (np.abs(b[jm1] - a) <= dt)
    return float(np.mean(left | right))


def _fraction_time_tiled(times, dt, intervals):
    times = _filter_intervals(times, intervals)
    total_time = sum(stop - start for start, stop in intervals)
    if total_time <= 0 or not len(times):
        return 0.0

    covered = 0.0
    for start, stop in intervals:
        local = times[(times >= start) & (times < stop)]
        if not len(local):
            continue
        tiles = np.c_[
            np.maximum(local - dt, start),
            np.minimum(local + dt, stop),
        ]
        tiles = tiles[np.argsort(tiles[:, 0])]
        s, e = tiles[0]
        for s2, e2 in tiles[1:]:
            if s2 <= e:
                e = max(e, e2)
            else:
                covered += e - s
                s, e = s2, e2
        covered += e - s
    return float(covered / total_time)


def sttc(a, b, dt, intervals):
    """Spike Time Tiling Coefficient of Cutts & Eglen (2014).

    ``dt`` is the half-window: a spike is coincident when another spike lies
    within ±dt. Observation gaps are excluded through ``intervals``.
    """
    a = _filter_intervals(a, intervals)
    b = _filter_intervals(b, intervals)
    if not len(a) or not len(b):
        return np.nan

    pa, pb = _fraction_near(a, b, dt), _fraction_near(b, a, dt)
    ta = _fraction_time_tiled(a, dt, intervals)
    tb = _fraction_time_tiled(b, dt, intervals)

    terms = []
    for p, t in ((pa, tb), (pb, ta)):
        denominator = 1 - p * t
        terms.append((p - t) / denominator if abs(denominator) > 1e-12 else np.nan)
    return float(np.nanmean(terms)) if np.isfinite(terms).any() else np.nan


def _coincidence_fraction(a, b, dt, intervals):
    a = _filter_intervals(a, intervals)
    b = _filter_intervals(b, intervals)
    if not len(a) and not len(b):
        return np.nan
    if not len(a) or not len(b):
        return 0.0
    return float(np.mean([_fraction_near(a, b, dt), _fraction_near(b, a, dt)]))


def _binned_count_correlation(a, b, bin_s, intervals):
    counts_a, counts_b = [], []
    for start, stop in intervals:
        n_bins = int(np.floor((stop - start) / bin_s))
        if n_bins < 1:
            continue
        edges = start + np.arange(n_bins + 1) * bin_s
        counts_a.extend(np.histogram(a, bins=edges)[0])
        counts_b.extend(np.histogram(b, bins=edges)[0])

    if len(counts_a) < 2 or np.std(counts_a) == 0 or np.std(counts_b) == 0:
        return np.nan
    return float(np.corrcoef(counts_a, counts_b)[0, 1])


def _add_event_labels(spikes, compound_isi_ms, burst_min_spikes, isolation_ms):
    spikes = spikes.sort_values(["epoch", "spike_time_sec"]).reset_index(drop=True).copy()
    if spikes.empty:
        return spikes

    spikes["isi_prev_ms"] = np.nan
    spikes["isi_next_ms"] = np.nan
    spikes["event_index"] = -1

    event_index = 0
    for _, idx in spikes.groupby("epoch", sort=False).groups.items():
        idx = np.asarray(list(idx), int)
        times = spikes.loc[idx, "spike_time_sec"].to_numpy(float)
        isi = np.diff(times) * 1000
        spikes.loc[idx[1:], "isi_prev_ms"] = isi
        spikes.loc[idx[:-1], "isi_next_ms"] = isi

        starts = np.r_[True, isi > compound_isi_ms]
        local_events = np.cumsum(starts) - 1 + event_index
        spikes.loc[idx, "event_index"] = local_events
        event_index = int(local_events[-1]) + 1

    event_counts = spikes.groupby("event_index").size()
    spikes["event_n_spikes"] = spikes["event_index"].map(event_counts).astype(int)
    spikes["event_type"] = np.select(
        [
            spikes["event_n_spikes"].eq(1),
            spikes["event_n_spikes"].eq(2),
            spikes["event_n_spikes"].ge(burst_min_spikes),
        ],
        ["singleton", "doublet", "burst"],
        default="compound",
    )
    spikes["is_compound"] = spikes["event_n_spikes"].ge(2)
    spikes["is_burst"] = spikes["event_n_spikes"].ge(burst_min_spikes)

    prev_ok = spikes["isi_prev_ms"].isna() | spikes["isi_prev_ms"].gt(isolation_ms)
    next_ok = spikes["isi_next_ms"].isna() | spikes["isi_next_ms"].gt(isolation_ms)
    spikes["is_isolated"] = prev_ok & next_ok
    return spikes


def _event_table(spikes):
    if spikes.empty:
        return pd.DataFrame()

    rows = []
    for event_index, group in spikes.groupby("event_index", sort=True):
        start = float(group["spike_time_sec"].min())
        stop = float(group["spike_time_sec"].max())
        rows.append({
            "event_index": int(event_index),
            "event_start_sec": start,
            "event_stop_sec": stop,
            "event_duration_ms": (stop - start) * 1000,
            "n_spikes": int(len(group)),
            "event_type": group["event_type"].iloc[0],
            "is_compound": bool(group["is_compound"].iloc[0]),
            "is_burst": bool(group["is_burst"].iloc[0]),
            "mean_within_event_isi_ms": (
                float(np.diff(group["spike_time_sec"].to_numpy(float)).mean() * 1000)
                if len(group) > 1 else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _roi_summary(roi_row, spike_table, event_table, recording_duration_s):
    isolated = spike_table[spike_table["is_isolated"]] if len(spike_table) else spike_table
    bursts = event_table[event_table["is_burst"]] if len(event_table) else event_table

    isi = spike_table["isi_prev_ms"].dropna().to_numpy(float) if len(spike_table) else np.array([])
    n_spikes = int(len(spike_table))
    n_events = int(len(event_table))
    n_compound_events = int(event_table["is_compound"].sum()) if n_events else 0
    n_bursts = int(event_table["is_burst"].sum()) if n_events else 0

    row = roi_row.to_dict() if hasattr(roi_row, "to_dict") else dict(roi_row)
    row.update({
        "recording_duration_s": float(recording_duration_s),
        "n_spikes": n_spikes,
        "spike_rate_hz": n_spikes / recording_duration_s if recording_duration_s > 0 else np.nan,
        "median_isi_ms": float(np.nanmedian(isi)) if len(isi) else np.nan,
        "isi_cv": float(np.nanstd(isi, ddof=1) / np.nanmean(isi)) if len(isi) > 1 and np.nanmean(isi) > 0 else np.nan,
        "n_isolated_spikes": int(len(isolated)),
        "median_spike_half_width_ms": isolated["spike_half_width_ms"].median() if len(isolated) else np.nan,
        "median_spike_amplitude_dff": isolated["spike_amplitude_dff"].median() if len(isolated) else np.nan,
        "median_spike_auc_dff_ms": isolated["spike_auc_dff_ms"].median() if len(isolated) else np.nan,
        "median_spike_rise10_90_ms": isolated["spike_rise10_90_ms"].median() if len(isolated) else np.nan,
        "median_spike_decay90_10_ms": isolated["spike_decay90_10_ms"].median() if len(isolated) else np.nan,
        "n_events": n_events,
        "event_rate_hz": n_events / recording_duration_s if recording_duration_s > 0 else np.nan,
        "compound_spike_fraction": float(spike_table["is_compound"].mean()) if n_spikes else np.nan,
        "compound_event_fraction": n_compound_events / n_events if n_events else np.nan,
        "burst_spike_fraction": float(spike_table["is_burst"].mean()) if n_spikes else np.nan,
        "burst_event_fraction": n_bursts / n_events if n_events else np.nan,
        "burst_event_rate_hz": n_bursts / recording_duration_s if recording_duration_s > 0 else np.nan,
        "median_burst_duration_ms": bursts["event_duration_ms"].median() if len(bursts) else np.nan,
        "median_burst_n_spikes": bursts["n_spikes"].median() if len(bursts) else np.nan,
    })
    return row


def build_session_analysis_tables(
    trace_h5,
    rois,
    spikes,
    *,
    isolation_ms=50.0,
    waveform_pre_ms=5.0,
    waveform_post_ms=15.0,
    waveform_peak_refine_ms=2.0,
    waveform_baseline_window_ms=(-5.0, -1.0),
    waveform_auc_window_ms=(-2.0, 10.0),
    compound_isi_ms=20.0,
    burst_min_spikes=3,
    spike_sttc_dt_ms=10.0,
    burst_sttc_dt_ms=40.0,
    spike_count_bin_ms=100.0,
):
    """Build per-spike, per-event, per-ROI, and pairwise synchrony tables for one session."""
    rois = rois.copy()
    spikes = spikes.copy()
    roi_rows, spike_tables, event_tables = [], [], []
    train_store, intervals_by_dmd = {}, {}

    with h5py.File(Path(trace_h5), "r") as h5:
        for dmd, dmd_rois in rois.groupby("dmd", sort=True):
            dmd = int(dmd)
            group = h5[f"DMD{dmd}"]
            timebase = np.asarray(group["timebase_sec"][:], float)
            fs = 1.0 / np.nanmedian(np.diff(timebase[:min(len(timebase), 100000)]))
            sample_epoch = group["sample_epoch"][:] if "sample_epoch" in group else None
            intervals = _observed_intervals(timebase, sample_epoch)
            intervals_by_dmd[dmd] = intervals
            duration_s = sum(stop - start for start, stop in intervals)

            dff = group["dff"]
            time_last = dff.shape[-1] == len(timebase)
            if not time_last and dff.shape[0] != len(timebase):
                raise ValueError(f"DMD{dmd}/dff shape {dff.shape} does not match timebase")

            for _, roi_row in dmd_rois.iterrows():
                roi = int(roi_row["roi"])
                y = np.asarray(dff[roi, :] if time_last else dff[:, roi], float)
                s = spikes[(spikes["dmd"] == dmd) & (spikes["roi"] == roi)].copy()
                s = _add_event_labels(s, compound_isi_ms, burst_min_spikes, isolation_ms)

                waveform_rows = [
                    _waveform_features(
                        y,
                        row.spike_index,
                        fs,
                        pre_ms=waveform_pre_ms,
                        post_ms=waveform_post_ms,
                        peak_refine_ms=waveform_peak_refine_ms,
                        baseline_window_ms=waveform_baseline_window_ms,
                        auc_window_ms=waveform_auc_window_ms,
                    )
                    for row in s.itertuples(index=False)
                ]
                if len(s):
                    waveform_columns = [
                        "raw_peak_index", "spike_amplitude_dff", "spike_auc_dff_ms",
                        "spike_half_width_ms", "spike_rise10_90_ms",
                        "spike_decay90_10_ms",
                    ]
                    waveform = pd.DataFrame(waveform_rows, index=s.index).reindex(
                        columns=waveform_columns
                    )
                    for column in waveform_columns:
                        s[column] = waveform[column]

                    s["event_id"] = (
                        s["session_id"].astype(str) + "_DMD" + str(dmd) +
                        "_ROI" + str(roi) + "_E" + s["event_index"].astype(str)
                    )
                    spike_tables.append(s)

                events = _event_table(s)
                if len(events):
                    events.insert(0, "roi", roi)
                    events.insert(0, "dmd", dmd)
                    events["event_id"] = (
                        str(roi_row["session_id"]) + "_DMD" + str(dmd) +
                        "_ROI" + str(roi) + "_E" + events["event_index"].astype(str)
                    )
                    for column in [
                        "subject_id", "session_id", "session_label", "session_order",
                        "session_type", "depth_um", "cell_id", "global_cell_id",
                        "manually_registered",
                    ]:
                        if column in roi_row:
                            events[column] = roi_row[column]
                    event_tables.append(events)

                roi_rows.append(_roi_summary(roi_row, s, events, duration_s))
                burst_onsets = (
                    events.loc[events["is_burst"], "event_start_sec"].to_numpy(float)
                    if len(events) else np.array([], float)
                )
                train_store[(dmd, roi)] = {
                    "spike_times": s["spike_time_sec"].to_numpy(float) if len(s) else np.array([], float),
                    "isolated_times": s.loc[s["is_isolated"], "spike_time_sec"].to_numpy(float) if len(s) else np.array([], float),
                    "burst_onsets": burst_onsets,
                    "roi_row": roi_row,
                }

    spike_features = pd.concat(spike_tables, ignore_index=True) if spike_tables else pd.DataFrame()
    event_features = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    roi_features = pd.DataFrame(roi_rows)

    pair_rows = []
    for key_a, key_b in combinations(sorted(train_store), 2):
        a, b = train_store[key_a], train_store[key_b]
        row_a, row_b = a["roi_row"], b["roi_row"]

        order_a = (str(row_a.get("cell_id", "")), key_a)
        order_b = (str(row_b.get("cell_id", "")), key_b)
        if order_b < order_a:
            key_a, key_b = key_b, key_a
            a, b = b, a
            row_a, row_b = row_b, row_a

        intervals = _intersect_intervals(
            intervals_by_dmd[int(key_a[0])],
            intervals_by_dmd[int(key_b[0])],
        )
        if not intervals:
            continue

        spike_dt = spike_sttc_dt_ms / 1000
        burst_dt = burst_sttc_dt_ms / 1000
        count_bin = spike_count_bin_ms / 1000

        ga = str(row_a.get("global_cell_id", "") or "")
        gb = str(row_b.get("global_cell_id", "") or "")
        registered_pair = bool(ga and gb and ga != "nan" and gb != "nan")
        longitudinal_pair_id = ""
        if registered_pair:
            ca, cb = sorted([ga, gb])
            longitudinal_pair_id = f"{row_a['subject_id']}:{ca}|{cb}"

        pair_rows.append({
            "subject_id": row_a.get("subject_id"),
            "session_id": row_a.get("session_id"),
            "session_label": row_a.get("session_label"),
            "session_order": row_a.get("session_order"),
            "session_type": row_a.get("session_type", ""),
            "pair_id": (
                f"{row_a.get('session_id')}:DMD{key_a[0]}_ROI{key_a[1]}|"
                f"DMD{key_b[0]}_ROI{key_b[1]}"
            ),
            "longitudinal_pair_id": longitudinal_pair_id,
            "manually_registered_pair": registered_pair,
            "dmd_a": int(key_a[0]),
            "roi_a": int(key_a[1]),
            "cell_id_a": row_a.get("cell_id", ""),
            "global_cell_id_a": ga,
            "depth_a_um": row_a.get("depth_um", np.nan),
            "dmd_b": int(key_b[0]),
            "roi_b": int(key_b[1]),
            "cell_id_b": row_b.get("cell_id", ""),
            "global_cell_id_b": gb,
            "depth_b_um": row_b.get("depth_um", np.nan),
            "same_dmd": bool(key_a[0] == key_b[0]),
            "same_global_cell": bool(registered_pair and ga == gb),
            "depth_separation_um": abs(float(row_a.get("depth_um", np.nan)) - float(row_b.get("depth_um", np.nan))),
            "overlap_duration_s": sum(stop - start for start, stop in intervals),
            "spike_coincidence_fraction_10ms": _coincidence_fraction(
                a["spike_times"], b["spike_times"], spike_dt, intervals
            ),
            "spike_sttc_10ms": sttc(
                a["spike_times"], b["spike_times"], spike_dt, intervals
            ),
            "isolated_spike_coincidence_fraction_10ms": _coincidence_fraction(
                a["isolated_times"], b["isolated_times"], spike_dt, intervals
            ),
            "isolated_spike_sttc_10ms": sttc(
                a["isolated_times"], b["isolated_times"], spike_dt, intervals
            ),
            "burst_onset_coincidence_fraction_40ms": _coincidence_fraction(
                a["burst_onsets"], b["burst_onsets"], burst_dt, intervals
            ),
            "burst_onset_sttc_40ms": sttc(
                a["burst_onsets"], b["burst_onsets"], burst_dt, intervals
            ),
            "spike_count_corr_100ms": _binned_count_correlation(
                a["spike_times"], b["spike_times"], count_bin, intervals
            ),
        })

    return {
        "roi_features": roi_features,
        "spike_features": spike_features,
        "event_features": event_features,
        "synchrony_pairs": pd.DataFrame(pair_rows),
    }


def build_analysis_tables(sessions, rois, spikes, **kwargs):
    """Build feature tables across sessions while keeping all pair metrics within-session."""
    collected = {
        "roi_features": [],
        "spike_features": [],
        "event_features": [],
        "synchrony_pairs": [],
    }

    for session in sessions.itertuples(index=False):
        print(f"Features: {session.session_id}, {session.session_label}")
        session_rois = rois[
            (rois["session_id"] == session.session_id) &
            rois["included"]
        ].copy()
        session_spikes = spikes[spikes["session_id"] == session.session_id].copy()
        tables = build_session_analysis_tables(
            session.trace_h5,
            session_rois,
            session_spikes,
            **kwargs,
        )
        for name, table in tables.items():
            if len(table):
                collected[name].append(table)

    out = {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in collected.items()
    }

    pairs = out["synchrony_pairs"]
    if len(pairs):
        valid = pairs["longitudinal_pair_id"].astype(str).ne("")
        counts = (
            pairs.loc[valid]
            .groupby("longitudinal_pair_id")["session_id"]
            .nunique()
        )
        pairs["pair_session_count"] = pairs["longitudinal_pair_id"].map(counts).fillna(1).astype(int)
        pairs["tracked_across_days"] = pairs["pair_session_count"].ge(2)

    return out


def save_analysis_tables(
    sessions,
    tables,
    *,
    subdir=Path("voltage") / "ephys_characterization",
    parameters=None,
):
    """Save each session's feature tables beneath that session's derived directory."""
    subdir = Path(subdir)

    for session in sessions.itertuples(index=False):
        output_dir = Path(session.derived_dir) / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, table in tables.items():
            if "session_id" not in table.columns:
                continue
            subset = table[table["session_id"].astype(str) == str(session.session_id)]
            subset.to_csv(output_dir / f"{name}.csv", index=False)

        if parameters is not None:
            serializable = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in parameters.items()
            }
            with open(output_dir / "analysis_parameters.json", "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)