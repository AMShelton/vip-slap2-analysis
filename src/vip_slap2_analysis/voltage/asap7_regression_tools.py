
"""
ASAP7 regression/alignment helpers.

Design goal: keep the notebook concise while making time alignment explicit.

Core time convention
--------------------
All behavioral, stimulus, running, and imaging variables are converted to
session-relative HARP seconds:

    session_time_sec = absolute_harp_time - session_harp_t0_abs

For the current ASAP7 behavior-processing outputs, the safest session t0 is
usually the first encoder sample. imaging_epochs.csv and voltage QC event times
are often already session-relative; absolute HARP vectors are detected and
shifted automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Mapping, Sequence
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

from vip_slap2_analysis.common.epoch_alignment import (
    DEFAULT_MIN_EPOCH_DURATION_SEC,
    accepted_epoch_dataframe,
)

from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

try:
    import h5py
except Exception:
    h5py = None


@dataclass
class SessionPaths:
    trace_h5: Optional[Path]
    qc_dir: Path
    encoder_pkl: Path
    harp_df_csv: Optional[Path] = None
    bonsai_event_log_csv: Optional[Path] = None


def _as_path(x):
    return None if x is None else Path(x)


def _maybe_get(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return default


def resolve_session_paths(
    asset: Any = None,
    *,
    trace_h5: Optional[str | Path] = None,
    qc_dir: Optional[str | Path] = None,
    encoder_pkl: Optional[str | Path] = None,
    harp_df_csv: Optional[str | Path] = None,
    bonsai_event_log_csv: Optional[str | Path] = None,
    trace_variant: str = "dff_robust_f0_trial",
) -> SessionPaths:
    """Resolve the files needed for alignment/regression from an asset or manual paths."""
    if trace_h5 is None and asset is not None:
        derived_dir = _maybe_get(asset, "derived_dir")
        candidates = []
        if derived_dir is not None:
            derived_dir = Path(derived_dir)
            candidates.extend([
                derived_dir / "voltage" / f"voltage_session_traces_{trace_variant}.h5",
                derived_dir / f"voltage_session_traces_{trace_variant}.h5",
            ])
        mod = _maybe_get(asset, "modality_assets", {})
        if isinstance(mod, Mapping) and "voltage" in mod and isinstance(mod["voltage"], Mapping):
            if mod["voltage"].get("trace_h5") is not None:
                candidates.append(Path(mod["voltage"]["trace_h5"]))
        for c in candidates:
            if c.exists():
                trace_h5 = c
                break

    if qc_dir is None:
        qc_dir = _maybe_get(asset, "qc_dir")
    if qc_dir is None:
        raise ValueError("qc_dir could not be resolved. Pass qc_dir=... or provide asset.qc_dir.")
    qc_dir = Path(qc_dir)

    if encoder_pkl is None:
        photodiode = _maybe_get(asset, "photodiode_pkl")
        if photodiode is not None:
            candidate = Path(photodiode).with_name("encoder.pkl")
            if candidate.exists():
                encoder_pkl = candidate
    if encoder_pkl is None:
        # local/attached data fallback
        candidates = [
            qc_dir / "behavior" / "encoder.pkl",
            qc_dir.parent / "encoder.pkl",
            Path("encoder.pkl"),
        ]
        for c in candidates:
            if c.exists():
                encoder_pkl = c
                break
    if encoder_pkl is None:
        raise ValueError("encoder.pkl could not be resolved. Pass encoder_pkl=...")

    if harp_df_csv is None:
        harp_df_csv = _maybe_get(asset, "harp_df_csv")
    if harp_df_csv is None:
        candidates = [
            qc_dir / "behavior" / "HARP_df.csv",
            qc_dir.parent / "HARP_df.csv",
            Path("HARP_df.csv"),
        ]
        for c in candidates:
            if c.exists():
                harp_df_csv = c
                break

    if bonsai_event_log_csv is None:
        bonsai_event_log_csv = _maybe_get(asset, "bonsai_event_log_csv")
    if bonsai_event_log_csv is None:
        candidates = [
            qc_dir / "behavior" / "bonsai_event_log.csv",
            qc_dir.parent / "bonsai_event_log.csv",
            Path("bonsai_event_log.csv"),
        ]
        for c in candidates:
            if c.exists():
                bonsai_event_log_csv = c
                break

    return SessionPaths(
        trace_h5=_as_path(trace_h5),
        qc_dir=qc_dir,
        encoder_pkl=Path(encoder_pkl),
        harp_df_csv=_as_path(harp_df_csv),
        bonsai_event_log_csv=_as_path(bonsai_event_log_csv),
    )


def find_file(qc_dir: str | Path, pattern: str) -> Optional[Path]:
    qc_dir = Path(qc_dir)
    hits = sorted(qc_dir.rglob(pattern))
    return hits[0] if hits else None


def normalize_to_session_time(
    x: Sequence[float],
    session_harp_t0_abs: float,
    *,
    name: str = "times",
    assume_relative_if_small: bool = True,
) -> np.ndarray:
    """
    Convert a time vector to session-relative HARP seconds.

    Heuristic:
    - absolute HARP values are large and near session_harp_t0_abs, so subtract t0.
    - small values are treated as already session-relative.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr

    med = float(np.nanmedian(arr))
    t0 = float(session_harp_t0_abs)

    if abs(med - t0) < 10_000 or med > 10_000:
        return arr - t0

    if assume_relative_if_small:
        return arr

    return arr - t0


def _odd_kernel(k: int | None) -> int:
    if k is None or int(k) <= 1:
        return 0
    k = int(k)
    return k if k % 2 else k + 1


def unwrap_modular_counter(values: np.ndarray, period: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return values.copy()
    d = np.diff(values)
    half = period / 2.0
    d2 = d.copy()
    d2[d > half] -= period
    d2[d < -half] += period
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    out[1:] = values[0] + np.cumsum(d2)
    return out


def load_encoder_running(
    encoder_pkl: str | Path,
    *,
    session_harp_t0_abs: Optional[float] = None,
    ticks_per_revolution: float = 8192.0,
    wheel_radius_cm: float = 4.69,
    median_filter_kernel: int = 51,
    smooth_sigma_samples: float = 3.0,
    absolute_speed: bool = True,
) -> tuple[pd.DataFrame, float]:
    """
    Load encoder.pkl and return running in session-relative HARP seconds.

    Returns
    -------
    running_df, session_harp_t0_abs
    """
    enc = pd.read_pickle(encoder_pkl)
    if "Encoder" not in enc.columns:
        raise KeyError(f"Encoder column not found. Columns: {list(enc.columns)}")
    raw_t_abs = enc.index.to_numpy(dtype=float)
    if session_harp_t0_abs is None:
        session_harp_t0_abs = float(raw_t_abs[0])

    raw = enc["Encoder"]
    vals = raw.to_numpy(dtype=float)
    dtype = raw.dtype

    period = float(2 ** (8 * np.dtype(dtype).itemsize)) if np.issubdtype(dtype, np.integer) else None
    pos = unwrap_modular_counter(vals, period) if period is not None else vals.copy()

    k = _odd_kernel(median_filter_kernel)
    pos_filt = medfilt(pos, kernel_size=k) if k > 1 else pos

    t = raw_t_abs - float(session_harp_t0_abs)
    angle = pos_filt * (2.0 * np.pi / float(ticks_per_revolution))
    vel_rad_s = np.gradient(angle) / np.gradient(t)
    vel_rad_s[~np.isfinite(vel_rad_s)] = 0.0
    vel_cm_s = vel_rad_s * float(wheel_radius_cm)
    if smooth_sigma_samples and smooth_sigma_samples > 0:
        vel_cm_s = gaussian_filter1d(vel_cm_s, sigma=float(smooth_sigma_samples), mode="nearest")

    speed_cm_s = np.abs(vel_cm_s) if absolute_speed else vel_cm_s
    accel_cm_s2 = np.gradient(vel_cm_s) / np.gradient(t)
    accel_cm_s2[~np.isfinite(accel_cm_s2)] = 0.0
    if smooth_sigma_samples and smooth_sigma_samples > 0:
        accel_cm_s2 = gaussian_filter1d(accel_cm_s2, sigma=float(smooth_sigma_samples), mode="nearest")

    out = pd.DataFrame({
        "time_sec": t,
        "harp_time_abs": raw_t_abs,
        "encoder_raw": vals,
        "encoder_unwrapped": pos,
        "velocity_cm_s": vel_cm_s,
        "speed_cm_s": speed_cm_s,
        "accel_cm_s2": accel_cm_s2,
    })
    out.attrs["session_harp_t0_abs"] = float(session_harp_t0_abs)
    out.attrs["counter_period"] = period
    out.attrs["ticks_per_revolution"] = float(ticks_per_revolution)
    out.attrs["wheel_radius_cm"] = float(wheel_radius_cm)
    return out, float(session_harp_t0_abs)


def load_harp_df_summary(harp_df_csv: Optional[str | Path], session_harp_t0_abs: float) -> dict:
    if harp_df_csv is None:
        return {}
    df = pd.read_csv(harp_df_csv, usecols=["time", "DI3", "MessageType"])
    time_abs = df["time"].to_numpy(float)
    time_rel = time_abs - float(session_harp_t0_abs)
    return {
        "first_harp_abs": float(time_abs[0]),
        "last_harp_abs": float(time_abs[-1]),
        "first_session_sec": float(time_rel[0]),
        "last_session_sec": float(time_rel[-1]),
        "n_rows": int(len(df)),
    }


def load_imaging_epochs(
    qc_dir: str | Path,
    session_harp_t0_abs: float,
    *,
    min_epoch_duration_sec: float = DEFAULT_MIN_EPOCH_DURATION_SEC,
) -> pd.DataFrame:
    p = find_file(qc_dir, "imaging_epochs.csv")
    if p is None:
        raise FileNotFoundError(f"No imaging_epochs.csv found under {qc_dir}")
    raw_epochs = pd.read_csv(p).copy()
    epochs = accepted_epoch_dataframe(
        raw_epochs,
        min_duration_sec=min_epoch_duration_sec,
        require_any=True,
    )
    for col in ["start_time", "end_time"]:
        if col not in epochs:
            raise KeyError(f"{p} missing required column {col!r}")
        epochs[col + "_sec"] = normalize_to_session_time(
            epochs[col].to_numpy(float),
            session_harp_t0_abs,
            name=col,
        )
    epochs["duration_sec"] = epochs["end_time_sec"] - epochs["start_time_sec"]
    epochs.attrs["path"] = str(p)
    epochs.attrs["min_epoch_duration_sec"] = float(min_epoch_duration_sec)
    return epochs


def _flatten_numeric(x) -> np.ndarray:
    out = []
    def rec(v):
        if isinstance(v, Mapping):
            for vv in v.values(): rec(vv)
        elif isinstance(v, (list, tuple, np.ndarray)):
            for vv in v: rec(vv)
        else:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    out.append(fv)
            except Exception:
                pass
    rec(x)
    return np.asarray(out, dtype=float)


def load_voltage_qc_events(
    qc_dir: str | Path,
    *,
    event_source_dmd: str = "auto",
    session_harp_t0_abs: Optional[float] = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Load image/change/omission onset times from voltage_extraction_qc*.json.

    Uses stimulus_onsets_used_for_extraction, not windows_sec.
    """
    p = find_file(qc_dir, "voltage_extraction_qc*.json")
    if p is None:
        raise FileNotFoundError(f"No voltage_extraction_qc*.json found under {qc_dir}")

    with open(p, "r") as f:
        qc = json.load(f)

    per_dmd = qc.get("per_dmd", {})
    if not per_dmd:
        raise KeyError(f"{p} has no per_dmd field.")

    if event_source_dmd == "auto":
        dmd = "DMD1" if "DMD1" in per_dmd else sorted(per_dmd.keys())[0]
    else:
        dmd = event_source_dmd

    stim = per_dmd[dmd]["stimulus_onsets_used_for_extraction"]
    image = np.sort(_flatten_numeric(stim.get("image_identity", {})))
    change = np.sort(_flatten_numeric(stim.get("change", [])))
    omission = np.sort(_flatten_numeric(stim.get("omission", [])))

    events = {"image": image, "change": change, "omission": omission}

    if session_harp_t0_abs is not None:
        # Usually these are already session-relative. This only shifts if they are absolute.
        events = {
            k: normalize_to_session_time(v, session_harp_t0_abs, name=k)
            for k, v in events.items()
        }

    meta = {
        "path": str(p),
        "event_source_dmd": dmd,
        "windows_sec_present_but_ignored": "windows_sec" in qc,
        "event_counts_in_qc": qc.get("event_counts", {}),
        "n_loaded": {k: int(len(v)) for k, v in events.items()},
    }
    return events, meta


def build_timing_context(
    paths: SessionPaths,
    *,
    ticks_per_revolution: float = 8192.0,
    wheel_radius_cm: float = 4.69,
    session_harp_t0_abs: Optional[float] = None,
    event_source_dmd: str = "auto",
) -> dict:
    """Load and align running, imaging epochs, and event onsets."""
    running, t0 = load_encoder_running(
        paths.encoder_pkl,
        session_harp_t0_abs=session_harp_t0_abs,
        ticks_per_revolution=ticks_per_revolution,
        wheel_radius_cm=wheel_radius_cm,
    )
    epochs = load_imaging_epochs(paths.qc_dir, t0)
    events, event_meta = load_voltage_qc_events(
        paths.qc_dir,
        event_source_dmd=event_source_dmd,
        session_harp_t0_abs=t0,
    )
    harp_summary = load_harp_df_summary(paths.harp_df_csv, t0)

    ctx = {
        "session_harp_t0_abs": t0,
        "running": running,
        "epochs": epochs,
        "events": events,
        "event_meta": event_meta,
        "harp_summary": harp_summary,
        "paths": paths,
    }
    return ctx


def alignment_summary_table(ctx: dict) -> pd.DataFrame:
    rows = []
    run = ctx["running"]
    rows.append({
        "stream": "encoder/running",
        "start_sec": float(run["time_sec"].min()),
        "end_sec": float(run["time_sec"].max()),
        "n": len(run),
    })
    for _, r in ctx["epochs"].iterrows():
        rows.append({
            "stream": f"imaging_epoch_{int(r.get('epoch_index', len(rows)))}",
            "start_sec": float(r["start_time_sec"]),
            "end_sec": float(r["end_time_sec"]),
            "n": np.nan,
        })
    for name, arr in ctx["events"].items():
        rows.append({
            "stream": name,
            "start_sec": float(np.min(arr)) if len(arr) else np.nan,
            "end_sec": float(np.max(arr)) if len(arr) else np.nan,
            "n": int(len(arr)),
        })
    return pd.DataFrame(rows)


def assert_timing_alignment(ctx: dict, *, slack_sec: float = 1.0) -> None:
    tab = alignment_summary_table(ctx)
    run0 = tab.loc[tab["stream"] == "encoder/running", "start_sec"].iloc[0]
    run1 = tab.loc[tab["stream"] == "encoder/running", "end_sec"].iloc[0]
    ep0 = float(ctx["epochs"]["start_time_sec"].min())
    ep1 = float(ctx["epochs"]["end_time_sec"].max())
    if not (run0 - slack_sec <= ep0 <= run1 + slack_sec and run0 - slack_sec <= ep1 <= run1 + slack_sec):
        raise AssertionError(f"Imaging epoch [{ep0:.3f}, {ep1:.3f}] is not covered by running [{run0:.3f}, {run1:.3f}]")
    for name, arr in ctx["events"].items():
        if len(arr) == 0:
            continue
        frac_in_epoch = np.mean((arr >= ep0 - slack_sec) & (arr <= ep1 + slack_sec))
        if frac_in_epoch < 0.95:
            raise AssertionError(f"Only {100*frac_in_epoch:.1f}% of {name} events fall in imaging epoch.")
    return None


def plot_alignment_overview(ctx: dict, *, t0: Optional[float] = None, duration_sec: float = 60.0):
    import matplotlib.pyplot as plt
    run = ctx["running"]
    events = ctx["events"]
    epochs = ctx["epochs"]

    if t0 is None:
        t0 = max(float(epochs["start_time_sec"].min()) - 2.0, float(run["time_sec"].min()))
    t1 = t0 + duration_sec

    fig, axes = plt.subplots(3, 1, figsize=(12, 6.2), sharex=True, constrained_layout=True)

    idx = (run["time_sec"] >= t0) & (run["time_sec"] <= t1)
    axes[0].plot(run.loc[idx, "time_sec"], run.loc[idx, "speed_cm_s"], lw=1)
    axes[0].set_ylabel("speed cm/s")
    axes[0].set_title("Alignment check: running, imaging epoch, event onsets")

    for _, ep in epochs.iterrows():
        for ax in axes:
            lo, hi = float(ep["start_time_sec"]), float(ep["end_time_sec"])
            if hi >= t0 and lo <= t1:
                ax.axvspan(max(lo, t0), min(hi, t1), alpha=0.08)

    y_positions = {"image": 0, "change": 1, "omission": 2}
    for name, y in y_positions.items():
        arr = events.get(name, np.array([]))
        arr = arr[(arr >= t0) & (arr <= t1)]
        axes[1].vlines(arr, y - 0.35, y + 0.35, lw=0.8)
    axes[1].set_yticks(list(y_positions.values()))
    axes[1].set_yticklabels(list(y_positions.keys()))
    axes[1].set_ylabel("events")

    # Histogram of all image intervals in the full session
    img = events.get("image", np.array([]))
    d = np.diff(np.sort(img))
    d = d[(d > 0.2) & (d < 2.0)]
    axes[2].hist(d, bins=60)
    axes[2].set_xlabel("session-relative HARP time (s)")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"Image interval check: median Δt={np.nanmedian(d):.4f} s, cadence={1/np.nanmedian(d):.4f} Hz" if len(d) else "Image interval check")
    return fig, axes


def interpolate_running_to_time(ctx: dict, t: np.ndarray) -> pd.DataFrame:
    run = ctx["running"]
    t = np.asarray(t, dtype=float)
    out = pd.DataFrame({"time_sec": t})
    for c in ["velocity_cm_s", "speed_cm_s", "accel_cm_s2"]:
        out[c] = np.interp(t, run["time_sec"].to_numpy(float), run[c].to_numpy(float), left=np.nan, right=np.nan)
    return out


def _decode_strings(a):
    out = []
    for x in a:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=str)


def _infer_layout(y_ds, t_len):
    if y_ds.shape[-1] == t_len:
        return "roi_time", y_ds.shape[0], y_ds.shape[-1]
    if y_ds.shape[0] == t_len:
        return "time_roi", y_ds.shape[1], y_ds.shape[0]
    raise ValueError(f"Cannot infer trace layout from shape {y_ds.shape} and time length {t_len}")


def infer_voltage_timebase_session_time(
    timebase_sec: np.ndarray,
    epochs: pd.DataFrame,
    session_harp_t0_abs: float,
    *,
    mode: str = "auto",
) -> tuple[np.ndarray, str]:
    """
    Map voltage H5 timebase to session-relative HARP seconds.

    mode:
      - auto
      - already_session
      - imaging_relative
      - absolute_harp
    """
    t = np.asarray(timebase_sec, dtype=float)
    ep0 = float(epochs["start_time_sec"].min())
    ep1 = float(epochs["end_time_sec"].max())
    ep_dur = ep1 - ep0

    if mode == "absolute_harp":
        return t - float(session_harp_t0_abs), mode
    if mode == "already_session":
        return t, mode
    if mode == "imaging_relative":
        return t + ep0, mode
    if mode != "auto":
        raise ValueError("mode must be auto, already_session, imaging_relative, or absolute_harp")

    med = float(np.nanmedian(t))
    if med > 10_000 or abs(med - session_harp_t0_abs) < 10_000:
        return t - float(session_harp_t0_abs), "absolute_harp_auto"

    start, stop = float(t[0]), float(t[-1])
    dur = stop - start

    score_already = abs(start - ep0) + abs(stop - ep1)
    score_imrel = abs(start - 0.0) + abs(dur - ep_dur)

    # If the timebase starts near 0 but is approximately an imaging-epoch duration,
    # it is imaging-relative and must be shifted by imaging start.
    if score_imrel + 2.0 < score_already:
        return t + ep0, "imaging_relative_auto"

    return t, "already_session_auto"


def _read_roi(y_ds, layout, i, n_time):
    return np.asarray(y_ds[i, :n_time] if layout == "roi_time" else y_ds[:n_time, i], dtype=np.float32)


def _fill_nans(y):
    y = np.asarray(y, dtype=np.float32)
    ok = np.isfinite(y)
    if ok.all():
        return y
    if ok.sum() < 2:
        return np.zeros_like(y)
    x = np.arange(len(y))
    z = y.copy()
    z[~ok] = np.interp(x[~ok], x[ok], y[ok]).astype(np.float32)
    return z


def _downsample(y, down):
    y = _fill_nans(y)
    if down <= 1:
        return y
    try:
        return signal.resample_poly(y, up=1, down=down, window=("kaiser", 8.6), padtype="line").astype(np.float32)
    except TypeError:
        return signal.resample_poly(y, up=1, down=down, window=("kaiser", 8.6)).astype(np.float32)


def load_voltage_lowfreq(
    trace_h5: str | Path,
    ctx: dict,
    *,
    signal_name: str = "dff",
    target_fs_hz: float = 100.0,
    valid_rois_only: bool = True,
    max_seconds: Optional[float] = None,
    voltage_time_mode: str = "auto",
) -> dict:
    """
    Load H5 voltage traces into downsampled ROI x time arrays with session-relative time.
    """
    if h5py is None:
        raise ImportError("h5py is required to load voltage traces.")
    trace_h5 = Path(trace_h5)
    out = {"trace_h5": trace_h5, "signal_name": signal_name, "dmd": {}}
    with h5py.File(trace_h5, "r") as h5:
        dmd_keys = sorted([k for k in h5.keys() if k.upper().startswith("DMD")])
        if not dmd_keys:
            raise KeyError("No DMD groups found in trace H5.")
        for dmd in dmd_keys:
            g = h5[dmd]
            if signal_name not in g:
                raise KeyError(f"{dmd} missing dataset {signal_name!r}; available: {list(g.keys())}")
            y_ds = g[signal_name]
            t_raw = np.asarray(g["timebase_sec"][:], dtype=float)
            if max_seconds is not None:
                # rough trim before downsampling
                dt0 = np.nanmedian(np.diff(t_raw[:min(len(t_raw), 100000)]))
                n_keep = min(len(t_raw), int(math.ceil(max_seconds / dt0)))
                t_raw = t_raw[:n_keep]
            else:
                n_keep = len(t_raw)

            t_session, time_mode_used = infer_voltage_timebase_session_time(
                t_raw, ctx["epochs"], ctx["session_harp_t0_abs"], mode=voltage_time_mode
            )
            fs_src = 1.0 / float(np.nanmedian(np.diff(t_raw[:min(len(t_raw), 200000)])))
            down = max(1, int(round(fs_src / float(target_fs_hz))))
            fs_low = fs_src / down

            layout, n_rois_total, _ = _infer_layout(y_ds, len(g["timebase_sec"]))
            roi_idx = np.arange(n_rois_total)
            if valid_rois_only and "valid_rois_mask" in g:
                roi_idx = np.flatnonzero(np.asarray(g["valid_rois_mask"][:], dtype=bool))
            roi_ids = _decode_strings(g["roi_ids"][:]) if "roi_ids" in g else np.asarray([f"{dmd}_{i}" for i in range(n_rois_total)], dtype=str)
            roi_ids = roi_ids[roi_idx]

            t_low = _downsample(t_session[:n_keep].astype(np.float32), down).astype(float)
            X = np.empty((len(roi_idx), len(t_low)), dtype=np.float32)
            for j, i in enumerate(roi_idx):
                y = _read_roi(y_ds, layout, int(i), n_keep)
                yy = _downsample(y, down)
                if len(yy) > len(t_low):
                    yy = yy[:len(t_low)]
                elif len(yy) < len(t_low):
                    yy = np.pad(yy, (0, len(t_low)-len(yy)), mode="edge")
                X[j] = yy

            out["dmd"][dmd] = {
                "X": X,
                "time_sec": t_low,
                "roi_ids": roi_ids,
                "source_roi_indices": roi_idx,
                "fs_source_hz": fs_src,
                "fs_hz": fs_low,
                "decimation": down,
                "time_mode_used": time_mode_used,
            }
    return out


def robust_zscore_rows(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float32)
    med = np.nanmedian(X, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), axis=1, keepdims=True)
    return (X - med) / np.maximum(1.4826 * mad, eps)


def make_drift_design(t: np.ndarray, n_basis: int = 12) -> tuple[np.ndarray, list[str], list[str]]:
    t = np.asarray(t, dtype=float)
    x = (t - np.nanmin(t)) / max(np.nanmax(t) - np.nanmin(t), 1e-9)
    cols = [np.ones_like(x)]
    names = ["intercept"]
    groups = ["intercept"]
    centers = np.linspace(0, 1, n_basis)
    width = 1.0 / max(n_basis - 1, 1)
    for i, c in enumerate(centers):
        cols.append(np.exp(-0.5 * ((x - c) / width) ** 2))
        names.append(f"drift_rbf_{i:02d}")
        groups.append("drift")
    return np.column_stack(cols), names, groups


def add_lagged_continuous_predictors(
    Xs: list[np.ndarray],
    names: list[str],
    groups: list[str],
    t: np.ndarray,
    series: pd.DataFrame,
    variables=("speed_cm_s", "velocity_cm_s", "accel_cm_s2"),
    lags_sec=(-1.0, -0.5, 0.0, 0.5, 1.0),
):
    ts = series["time_sec"].to_numpy(float)
    for var in variables:
        vals = series[var].to_numpy(float)
        vals = np.nan_to_num(vals, nan=0.0)
        vals = (vals - np.nanmean(vals)) / max(np.nanstd(vals), 1e-9)
        for lag in lags_sec:
            col = np.interp(t - lag, ts, vals, left=0.0, right=0.0)
            Xs.append(col)
            names.append(f"{var}_lag{lag:+.2f}s")
            groups.append("running")


def add_event_boxcar_predictors(
    Xs: list[np.ndarray],
    names: list[str],
    groups: list[str],
    t: np.ndarray,
    events: np.ndarray,
    event_name: str,
    window=(-0.5, 1.5),
    bin_width=0.05,
):
    """
    Add event-aligned boxcar/FIR predictors.

    Uses searchsorted slice assignment rather than time x event boolean masks, so
    this remains fast for thousands of image flashes.
    """
    events = np.asarray(events, dtype=float)
    events = events[np.isfinite(events)]
    if len(events) == 0:
        return

    t = np.asarray(t, dtype=float)
    edges = np.arange(window[0], window[1] + 1e-12, bin_width)

    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        col = np.zeros(t.shape[0], dtype=np.float32)

        starts = np.searchsorted(t, events + lo, side="left")
        stops = np.searchsorted(t, events + hi, side="left")
        valid = (stops > starts) & (stops > 0) & (starts < len(t))

        for s, e in zip(starts[valid], stops[valid]):
            col[max(int(s), 0):min(int(e), len(t))] = 1.0

        Xs.append(col)
        names.append(f"{event_name}_{lo:+.2f}_{hi:+.2f}")
        groups.append(event_name)


def build_regression_design(
    t: np.ndarray,
    ctx: dict,
    *,
    n_drift_basis: int = 12,
    running_lags_sec=(-1.0, -0.5, 0.0, 0.5, 1.0),
    event_window=(-0.5, 1.5),
    event_bin_width=0.05,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build dense design matrix for voltage regression."""
    X0, names, groups = make_drift_design(t, n_basis=n_drift_basis)
    Xs = [X0[:, i] for i in range(X0.shape[1])]
    running_t = interpolate_running_to_time(ctx, t)
    add_lagged_continuous_predictors(Xs, names, groups, t, running_t, lags_sec=running_lags_sec)
    for ev_name in ["image", "change", "omission"]:
        add_event_boxcar_predictors(Xs, names, groups, t, ctx["events"].get(ev_name, []), ev_name,
                                    window=event_window, bin_width=event_bin_width)
    X = np.column_stack(Xs).astype(np.float32)

    # Drop zero-variance non-intercept columns.
    keep = np.ones(X.shape[1], dtype=bool)
    for j, (nm, gr) in enumerate(zip(names, groups)):
        if gr != "intercept" and np.nanstd(X[:, j]) < 1e-12:
            keep[j] = False
    X = X[:, keep]
    meta = pd.DataFrame({"name": np.asarray(names)[keep], "group": np.asarray(groups)[keep]})
    return X, meta


def standardize_design(X: np.ndarray, meta: pd.DataFrame):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    no_scale = meta["group"].to_numpy() == "intercept"
    sd[no_scale] = 1.0
    mu[no_scale] = 0.0
    sd = np.maximum(sd, 1e-9)
    return ((X - mu) / sd).astype(np.float32), mu, sd


def fit_ridge_models(
    X: np.ndarray,
    Y_roi_time: np.ndarray,
    meta: pd.DataFrame,
    *,
    alpha: float = 10.0,
    models: Optional[Mapping[str, Sequence[str]]] = None,
) -> dict:
    """
    Fit ridge models to Y (ROI x time) using dense design X (time x predictors).
    Returns predictions/residuals for each model.
    """
    if models is None:
        models = {
            "drift": ["intercept", "drift"],
            "drift_running": ["intercept", "drift", "running"],
            "drift_stimulus": ["intercept", "drift", "image", "change", "omission"],
            "full": ["intercept", "drift", "running", "image", "change", "omission"],
        }

    Xz, x_mu, x_sd = standardize_design(X, meta)
    Y = np.asarray(Y_roi_time, dtype=np.float32).T  # time x roi
    y_mu = np.nanmean(Y, axis=0, keepdims=True)
    Yc = Y - y_mu
    ss_tot = np.sum(Yc ** 2, axis=0) + 1e-12

    out = {"models": {}, "x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu.squeeze(), "meta": meta.copy()}
    groups = meta["group"].to_numpy()

    for model_name, keep_groups in models.items():
        keep = np.isin(groups, list(keep_groups))
        Xm = Xz[:, keep]
        reg = alpha * np.eye(Xm.shape[1], dtype=np.float64)
        intercept_cols = np.flatnonzero(groups[keep] == "intercept")
        reg[intercept_cols, intercept_cols] = 0.0
        A = Xm.T @ Xm + reg
        B = Xm.T @ Y
        coef = np.linalg.solve(A, B)
        pred = Xm @ coef
        resid = Y - pred
        ss_res = np.sum((Y - pred) ** 2, axis=0)
        r2 = 1.0 - ss_res / ss_tot
        out["models"][model_name] = {
            "groups": list(keep_groups),
            "predictor_mask": keep,
            "coef": coef.astype(np.float32),
            "prediction": pred.T.astype(np.float32),
            "residual": resid.T.astype(np.float32),
            "r2": r2.astype(np.float32),
        }
    return out


def summarize_r2(fits_by_dmd: Mapping[str, dict]) -> pd.DataFrame:
    rows = []
    for dmd, fit in fits_by_dmd.items():
        for model, d in fit["models"].items():
            for i, r2 in enumerate(d["r2"]):
                rows.append({"dmd": dmd, "model": model, "roi_index": i, "r2": float(r2)})
    return pd.DataFrame(rows)


def welch_psd_rows(X: np.ndarray, fs_hz: float, *, window_sec: float = 256.0):
    nperseg = min(X.shape[1], max(16, int(round(window_sec * fs_hz))))
    noverlap = nperseg // 2
    f, P = signal.welch(X, fs=fs_hz, axis=1, nperseg=nperseg, noverlap=noverlap,
                        detrend="constant", scaling="density", average="median")
    return f, P.astype(np.float32)


def bandpower_from_psd(f, P, bands: Mapping[str, tuple[float, float]]):
    rows = []
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        bp = np.trapz(P[:, m], f[m], axis=1) if m.sum() >= 2 else np.full(P.shape[0], np.nan)
        for i, v in enumerate(bp):
            rows.append({"band": name, "roi_index": i, "bandpower": float(v), "log10_bandpower": float(np.log10(v + 1e-20))})
    return pd.DataFrame(rows)


def plot_r2_summary(r2_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    order = ["drift", "drift_running", "drift_stimulus", "full"]
    order = [x for x in order if x in set(r2_df["model"])]
    dmds = sorted(r2_df["dmd"].unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(order))
    width = 0.8 / max(1, len(dmds))
    for j, dmd in enumerate(dmds):
        vals = [r2_df.query("dmd == @dmd and model == @m")["r2"].to_numpy(float) for m in order]
        means = [np.nanmean(v) for v in vals]
        sems = [np.nanstd(v, ddof=1)/np.sqrt(max(np.isfinite(v).sum(),1)) for v in vals]
        ax.bar(x - 0.4 + width/2 + j*width, means, width, yerr=sems, capsize=3, label=dmd)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("in-sample R²")
    ax.set_title("Regression model fit by DMD")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax


def plot_raw_vs_residual_psd(voltage: dict, fits_by_dmd: Mapping[str, dict], *, model="full", fmax=30.0, window_sec=256.0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for dmd, vd in voltage["dmd"].items():
        X_raw = robust_zscore_rows(vd["X"])
        X_res = robust_zscore_rows(fits_by_dmd[dmd]["models"][model]["residual"])
        f, P_raw = welch_psd_rows(X_raw, vd["fs_hz"], window_sec=window_sec)
        _, P_res = welch_psd_rows(X_res, vd["fs_hz"], window_sec=window_sec)
        m = (f > 0) & (f <= fmax)
        ax.plot(f[m], np.nanmean(P_raw[:, m], axis=0), lw=1.2, alpha=0.65, label=f"{dmd} raw")
        ax.plot(f[m], np.nanmean(P_res[:, m], axis=0), lw=2.0, label=f"{dmd} residual")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title(f"Raw vs {model} residual PSD")
    ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax


def plot_population_coherence_before_after(voltage: dict, fits_by_dmd: Mapping[str, dict], *, model="full", fmax=30.0, window_sec=256.0):
    import matplotlib.pyplot as plt
    dmds = sorted(voltage["dmd"].keys())
    if len(dmds) < 2:
        raise ValueError("Need at least two DMDs.")
    d1, d2 = dmds[:2]
    fs = voltage["dmd"][d1]["fs_hz"]
    y1_raw = robust_zscore_rows(voltage["dmd"][d1]["X"]).mean(axis=0)
    y2_raw = robust_zscore_rows(voltage["dmd"][d2]["X"]).mean(axis=0)
    y1_res = robust_zscore_rows(fits_by_dmd[d1]["models"][model]["residual"]).mean(axis=0)
    y2_res = robust_zscore_rows(fits_by_dmd[d2]["models"][model]["residual"]).mean(axis=0)
    nperseg = min(len(y1_raw), max(16, int(round(window_sec * fs))))
    f, c_raw = signal.coherence(y1_raw, y2_raw, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    _, c_res = signal.coherence(y1_res, y2_res, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    m = (f > 0) & (f <= fmax)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(f[m], c_raw[m], lw=1.6, label="raw")
    ax.plot(f[m], c_res[m], lw=1.8, label=f"{model} residual")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("DMD population coherence")
    ax.set_title("Cross-DMD coherence before/after residualization")
    ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
    return fig, ax


def save_regression_outputs(out_dir: str | Path, voltage: dict, fits_by_dmd: Mapping[str, dict], design_meta: pd.DataFrame, r2_df: pd.DataFrame, *, model="full"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    r2_df.to_csv(out_dir / "regression_r2.csv", index=False)
    design_meta.to_csv(out_dir / "design_columns.csv", index=False)
    npz = {}
    for dmd, vd in voltage["dmd"].items():
        npz[f"{dmd}_time_sec"] = vd["time_sec"]
        npz[f"{dmd}_roi_ids"] = vd["roi_ids"]
        npz[f"{dmd}_raw"] = vd["X"]
        npz[f"{dmd}_{model}_residual"] = fits_by_dmd[dmd]["models"][model]["residual"]
        npz[f"{dmd}_{model}_prediction"] = fits_by_dmd[dmd]["models"][model]["prediction"]
    np.savez_compressed(out_dir / f"regression_{model}_outputs.npz", **npz)
    return out_dir
