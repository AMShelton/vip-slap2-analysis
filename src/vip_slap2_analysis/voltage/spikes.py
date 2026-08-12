"""Optical spike detection for ASAP voltage traces.

A conservative high-pass peak detector provides seed spikes. When enough seed
spikes are available, their median waveform is used as a matched filter to
rescue lower-amplitude events with the same shape.
"""

from pathlib import Path
from typing import Iterable, Mapping, Optional

import h5py
import numpy as np
import pandas as pd
from scipy import signal

DETECTOR_VERSION = "template_v1"

_SPIKE_COLUMNS = [
    "spike_index",
    "spike_time_sec",
    "epoch",
    "peak_dff",
    "detection_dff",
    "noise_dff",
    "peak_snr",
    "prominence_snr",
    "width_ms",
    "template_snr",
    "detection_method",
]


def _runs(labels):
    labels = np.asarray(labels)
    edges = np.flatnonzero(np.diff(labels) != 0) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(labels)]
    return [(int(a), int(b), labels[a]) for a, b in zip(starts, stops) if labels[a] > 0]


def _fill_nans(y):
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return None
    if not finite.all():
        y = np.interp(np.arange(len(y)), np.flatnonzero(finite), y[finite])
    return y


def _highpass(y, fs, cutoff_hz, order):
    if cutoff_hz is None or cutoff_hz <= 0:
        return y - np.median(y)
    if cutoff_hz >= fs / 2:
        raise ValueError("highpass_hz must be below Nyquist")
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    try:
        x = signal.sosfiltfilt(sos, y)
    except ValueError:
        x = y - np.median(y)
    return x - np.median(x)


def _noise_sigma(x):
    """Robust Gaussian-equivalent sigma from the full centered trace."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _seed_peaks(x, sigma, fs, *, height_sigma, prominence_sigma,
                refractory_ms, min_width_ms, prominence_window_ms):
    distance = max(1, int(round(refractory_ms * fs / 1000)))
    min_width = max(1.0, min_width_ms * fs / 1000)
    wlen = max(3, int(round(prominence_window_ms * fs / 1000)))
    wlen += 1 - wlen % 2
    wlen = min(wlen, len(x) if len(x) % 2 else len(x) - 1)

    if wlen < 3:
        return np.array([], dtype=int)

    peaks, _ = signal.find_peaks(
        x,
        height=height_sigma * sigma,
        prominence=prominence_sigma * sigma,
        distance=distance,
        width=min_width,
        wlen=wlen,
    )
    return peaks.astype(int)


def _build_template(segment_data, fs, *, pre_ms, post_ms, min_spikes, max_spikes):
    pre = max(1, int(round(pre_ms * fs / 1000)))
    post = max(1, int(round(post_ms * fs / 1000)))
    waves = []

    for item in segment_data:
        x = item["x"]
        for peak in item["seed_peaks"]:
            if peak < pre or peak + post >= len(x):
                continue
            wave = x[peak - pre : peak + post + 1].copy()
            wave -= np.median(wave[:pre])
            amp = wave[pre]
            if np.isfinite(amp) and amp > 0:
                waves.append(wave / amp)

    if len(waves) < min_spikes:
        return None, pre

    if len(waves) > max_spikes:
        take = np.linspace(0, len(waves) - 1, max_spikes).astype(int)
        waves = [waves[i] for i in take]

    template = np.median(np.asarray(waves), axis=0)
    template -= template.mean()
    norm = np.linalg.norm(template)
    if not np.isfinite(norm) or norm <= 0:
        return None, pre
    return template / norm, pre


def _template_candidates(x, template, peak_offset, fs, *, template_sigma,
                         refractory_ms, refine_ms):
    if template is None or len(x) < len(template):
        return np.array([], dtype=int), np.array([], dtype=float), None

    score = signal.oaconvolve(x, template[::-1], mode="valid")
    score -= np.median(score)
    score_sigma = _noise_sigma(score)
    if not np.isfinite(score_sigma) or score_sigma <= 0:
        return np.array([], dtype=int), np.array([], dtype=float), score

    distance = max(1, int(round(refractory_ms * fs / 1000)))
    starts, props = signal.find_peaks(
        score,
        height=template_sigma * score_sigma,
        distance=distance,
    )

    candidates = starts.astype(int) + int(peak_offset)
    scores = props["peak_heights"] / score_sigma

    # Snap matched-filter events onto actual local maxima. This guarantees that
    # downstream prominence/width measurements are evaluated at valid peaks.
    refine = max(1, int(round(refine_ms * fs / 1000)))
    local_peaks = signal.find_peaks(x)[0]
    refined = {}

    for idx, snr in zip(candidates, scores):
        nearby = local_peaks[
            (local_peaks >= idx - refine) &
            (local_peaks <= idx + refine)
        ]
        if not len(nearby):
            continue

        peak = int(nearby[np.argmin(np.abs(nearby - idx))])
        refined[peak] = max(refined.get(peak, -np.inf), float(snr))

    return (
        np.asarray(list(refined), dtype=int),
        np.asarray(list(refined.values()), dtype=float),
        score,
    )


def _peak_metrics(x, peaks, sigma, fs, prominence_window_ms):
    if not len(peaks):
        return np.array([]), np.array([])

    wlen = max(3, int(round(prominence_window_ms * fs / 1000)))
    wlen += 1 - wlen % 2
    wlen = min(wlen, len(x) if len(x) % 2 else len(x) - 1)

    prominences = signal.peak_prominences(x, peaks, wlen=wlen)[0]
    widths = signal.peak_widths(x, peaks, rel_height=0.5)[0] * 1000 / fs
    return prominences / sigma, widths


def detect_spikes(
    y,
    fs,
    *,
    timebase=None,
    sample_epoch=None,
    trial_lengths=None,
    highpass_hz=10.0,
    height_sigma=3.0,
    prominence_sigma=0.5,
    refractory_ms=1.0,
    min_width_ms=0.2,
    prominence_window_ms=25.0,
    trial_edge_ms=2.0,
    filter_order=3,
    template_sigma=3.5,
    template_pre_ms=3.0,
    template_post_ms=7.0,
    template_min_spikes=10,
    template_max_spikes=1000,
    template_refine_ms=1.5,
):
    """Detect positive optical spikes in one ROI trace.

    Seed spikes are detected by amplitude/prominence. A median seed waveform is
    then matched against the high-pass trace to rescue weaker events. Raw dF/F
    is used only for ``peak_dff``.
    """
    y = _fill_nans(y)
    if y is None:
        return pd.DataFrame(columns=_SPIKE_COLUMNS)

    fs = float(fs)
    if timebase is not None and len(timebase) != len(y):
        raise ValueError("timebase and trace must have the same length")
    if sample_epoch is not None and len(sample_epoch) != len(y):
        raise ValueError("sample_epoch and trace must have the same length")

    segments = [(0, len(y), 1)] if sample_epoch is None else _runs(sample_epoch)
    segment_data = []

    for start, stop, epoch in segments:
        segment = y[start:stop]
        if len(segment) < 3:
            continue
        x = _highpass(segment, fs, highpass_hz, filter_order)
        sigma = _noise_sigma(x)
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        seeds = _seed_peaks(
            x, sigma, fs,
            height_sigma=height_sigma,
            prominence_sigma=prominence_sigma,
            refractory_ms=refractory_ms,
            min_width_ms=min_width_ms,
            prominence_window_ms=prominence_window_ms,
        )
        segment_data.append({
            "start": start,
            "stop": stop,
            "epoch": int(epoch),
            "x": x,
            "sigma": sigma,
            "seed_peaks": seeds,
        })

    template, peak_offset = _build_template(
        segment_data,
        fs,
        pre_ms=template_pre_ms,
        post_ms=template_post_ms,
        min_spikes=template_min_spikes,
        max_spikes=template_max_spikes,
    )

    rows = []
    refractory = max(1, int(round(refractory_ms * fs / 1000)))

    for item in segment_data:
        start = item["start"]
        epoch = item["epoch"]
        x = item["x"]
        sigma = item["sigma"]
        seeds = item["seed_peaks"]

        candidates, candidate_snr, score = _template_candidates(
            x, template, peak_offset, fs,
            template_sigma=template_sigma,
            refractory_ms=refractory_ms,
            refine_ms=template_refine_ms,
        )

        accepted = [(int(p), "seed", np.nan) for p in seeds]
        occupied = list(seeds.astype(int))

        # Strongest template matches get first claim on the refractory window.
        for p, snr in sorted(zip(candidates, candidate_snr), key=lambda z: z[1], reverse=True):
            if all(abs(int(p) - q) >= refractory for q in occupied):
                accepted.append((int(p), "template", float(snr)))
                occupied.append(int(p))

        accepted.sort(key=lambda z: z[0])
        peaks = np.asarray([p for p, _, _ in accepted], dtype=int)
        methods = [m for _, m, _ in accepted]
        template_snrs = [s for _, _, s in accepted]

        prominence_snrs, widths = _peak_metrics(
            x, peaks, sigma, fs, prominence_window_ms
        )

        # Report template SNR for seed spikes too when a template exists.
        if template is not None and score is not None:
            score_sigma = _noise_sigma(score)
            for i, (p, method, snr) in enumerate(accepted):
                if method == "seed":
                    j = p - peak_offset
                    if 0 <= j < len(score) and score_sigma > 0:
                        template_snrs[i] = float(score[j] / score_sigma)

        for p, method, template_snr, prom_snr, width in zip(
            peaks, methods, template_snrs, prominence_snrs, widths
        ):
            idx = int(start + p)
            rows.append({
                "spike_index": idx,
                "spike_time_sec": float(timebase[idx]) if timebase is not None else idx / fs,
                "epoch": epoch,
                "peak_dff": float(y[idx]),
                "detection_dff": float(x[p]),
                "noise_dff": sigma,
                "peak_snr": float(x[p] / sigma),
                "prominence_snr": float(prom_snr),
                "width_ms": float(width),
                "template_snr": float(template_snr) if np.isfinite(template_snr) else np.nan,
                "detection_method": method,
            })

    spikes = pd.DataFrame(rows, columns=_SPIKE_COLUMNS)
    if spikes.empty or trial_lengths is None or trial_edge_ms <= 0:
        return spikes

    edge_samples = max(1, int(round(trial_edge_ms * fs / 1000)))
    boundaries = np.cumsum(np.asarray(trial_lengths, dtype=int))[:-1]
    samples = spikes["spike_index"].to_numpy()
    keep = np.ones(len(spikes), dtype=bool)
    for boundary in boundaries:
        keep &= np.abs(samples - boundary) > edge_samples
    return spikes.loc[keep].reset_index(drop=True)


def extract_session_spikes(
    trace_h5,
    *,
    rois: Optional[Mapping[int, Iterable[int]]] = None,
    **detection_kwargs,
):
    """Detect spikes for selected ROIs in a processed session-trace H5."""
    tables = []

    with h5py.File(Path(trace_h5), "r") as h5:
        for dmd_key in sorted(k for k in h5 if k.startswith("DMD")):
            dmd = int(dmd_key.replace("DMD", ""))
            group = h5[dmd_key]
            timebase = np.asarray(group["timebase_sec"][:], dtype=float)
            fs = 1.0 / np.nanmedian(np.diff(timebase[: min(len(timebase), 100000)]))
            dff = group["dff"]

            if dff.shape[-1] == len(timebase):
                n_rois = dff.shape[0]

                def read_trace(roi):
                    return dff[int(roi), :]

            elif dff.shape[0] == len(timebase):
                n_rois = dff.shape[1]

                def read_trace(roi):
                    return dff[:, int(roi)]

            else:
                raise ValueError(
                    f"{dmd_key}/dff shape {dff.shape} does not match timebase"
                )

            roi_indices = range(n_rois) if rois is None else rois.get(dmd, [])
            sample_epoch = group["sample_epoch"][:] if "sample_epoch" in group else None
            trial_lengths = (
                group["trial_lengths_samples"][:]
                if "trial_lengths_samples" in group
                else None
            )

            for roi in roi_indices:
                roi = int(roi)
                if roi < 0 or roi >= n_rois:
                    raise IndexError(
                        f"DMD{dmd} ROI {roi} is outside 0..{n_rois - 1}"
                    )

                table = detect_spikes(
                    read_trace(roi),
                    fs,
                    timebase=timebase,
                    sample_epoch=sample_epoch,
                    trial_lengths=trial_lengths,
                    **detection_kwargs,
                )
                if len(table):
                    table.insert(0, "roi", roi)
                    table.insert(0, "dmd", dmd)
                    tables.append(table)

    if tables:
        return pd.concat(tables, ignore_index=True)

    return pd.DataFrame(columns=["dmd", "roi"] + _SPIKE_COLUMNS)
# -----------------------------------------------------------------------------
# Persistent session-level spike tables
# -----------------------------------------------------------------------------


def _normalize_roi_selection(rois):
    if rois is None:
        return None
    return {
        str(int(dmd)): sorted(int(roi) for roi in values)
        for dmd, values in sorted(rois.items())
    }


def _jsonable_detection_kwargs(kwargs):
    out = {}
    for key, value in sorted(kwargs.items()):
        if isinstance(value, np.generic):
            value = value.item()
        out[str(key)] = value
    return out


def _spike_cache_metadata(trace_h5, rois, detection_kwargs):
    import os

    trace_h5 = Path(trace_h5)
    stat = trace_h5.stat()
    return {
        "detector_version": DETECTOR_VERSION,
        "source_trace_h5": str(trace_h5),
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "roi_selection": _normalize_roi_selection(rois),
        "detection_kwargs": _jsonable_detection_kwargs(detection_kwargs),
    }


def _cache_paths(trace_h5, cache_path=None):
    trace_h5 = Path(trace_h5)
    if cache_path is None:
        cache_path = trace_h5.parent / f"spikes_{DETECTOR_VERSION}.parquet"
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    return cache_path, meta_path


def _load_cached_spikes(cache_path):
    cache_path = Path(cache_path)
    if cache_path.suffix == ".parquet":
        return pd.read_parquet(cache_path)
    if cache_path.name.endswith(".csv.gz"):
        return pd.read_csv(cache_path)
    raise ValueError(f"Unsupported spike-cache format: {cache_path}")


def _save_cached_spikes(table, cache_path):
    """Save parquet when available; fall back to compressed CSV."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table.to_parquet(cache_path, index=False)
        return cache_path
    except (ImportError, ModuleNotFoundError, ValueError):
        fallback = cache_path.with_suffix(".csv.gz")
        table.to_csv(fallback, index=False, compression="gzip")
        return fallback


def load_or_extract_session_spikes(
    trace_h5,
    *,
    rois=None,
    cache_path=None,
    force=False,
    **detection_kwargs,
):
    """Load a valid cached spike table or run the canonical detector and persist it.

    Cache validity depends on detector version, source H5 size/mtime, requested ROI
    selection, and detector parameters. A cache can therefore never be silently
    reused after the trace file, ROI inclusion, or detector settings change.
    """
    import json

    cache_path, meta_path = _cache_paths(trace_h5, cache_path=cache_path)
    expected = _spike_cache_metadata(trace_h5, rois, detection_kwargs)

    candidates = [cache_path]
    if cache_path.suffix == ".parquet":
        candidates.append(cache_path.with_suffix(".csv.gz"))

    if not force and meta_path.exists():
        with meta_path.open("r") as f:
            observed = json.load(f)
        if observed == expected:
            for candidate in candidates:
                if candidate.exists():
                    return _load_cached_spikes(candidate)

    table = extract_session_spikes(trace_h5, rois=rois, **detection_kwargs)
    written = _save_cached_spikes(table, cache_path)
    with meta_path.open("w") as f:
        json.dump(expected, f, indent=2)

    # If parquet failed, make the metadata path discoverable beside the fallback too.
    if written != cache_path:
        fallback_meta = written.with_suffix(written.suffix + ".meta.json")
        if fallback_meta != meta_path:
            with fallback_meta.open("w") as f:
                json.dump(expected, f, indent=2)
    return table


def build_spike_table(
    sessions,
    rois,
    *,
    force=False,
    detection_kwargs=None,
):
    """Build the cross-session spike table used by ephys/DoC analyses."""
    detection_kwargs = dict(detection_kwargs or {})
    tables = []

    for session in sessions.itertuples(index=False):
        session_rois = rois[
            rois["session_id"].astype(str).eq(str(session.session_id)) & rois["included"]
        ]
        roi_lookup = {
            int(dmd): group["roi"].astype(int).tolist()
            for dmd, group in session_rois.groupby("dmd")
        }
        table = load_or_extract_session_spikes(
            session.trace_h5,
            rois=roi_lookup,
            force=force,
            **detection_kwargs,
        )
        table.insert(0, "session_id", str(session.session_id))
        tables.append(table)

    spikes = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if spikes.empty:
        return spikes

    keep = [
        "subject_id",
        "session_id",
        "session_label",
        "session_order",
        "session_type",
        "dmd",
        "roi",
        "depth_um",
        "depth_group",
        "cell_id",
        "global_cell_id",
        "manually_registered",
    ]
    return spikes.merge(
        rois[keep],
        on=["session_id", "dmd", "roi"],
        how="left",
    )
