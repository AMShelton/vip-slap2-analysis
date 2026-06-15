"""
Trace reconstruction and stimulus-alignment helpers for glutamate imaging.

This module contains low-level utilities for parsing corrected Bonsai event logs,
filtering events to valid imaging epochs, reconstructing continuous DMD traces
from trial-wise SLAP2 outputs, aligning traces to stimulus onsets, and summarizing
fixed-length or ragged event tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from vip_slap2_analysis.glutamate.summary import GlutamateSummary


Interval = Tuple[float, float]
StimIntervalList = List[Interval]
StimIntervalDict = Dict[str, StimIntervalList]


@dataclass
class EventWindows:
    """
    Pre- and post-event window durations for each stimulus family.
    
    Each tuple stores ``(pre_seconds, post_seconds)`` and is used to determine the
    length of aligned event snippets.
    """
    image: Tuple[float, float] = (0.25, 0.50)
    change: Tuple[float, float] = (1.00, 0.75)
    omission: Tuple[float, float] = (1.00, 1.50)


@dataclass
class OrderedImageEvent:
    """
    Metadata for one image presentation in corrected Bonsai event order.
    
    The ``is_change_target`` flag is later set for images immediately following a
    change event.
    """
    event_idx: int
    image_name: str
    onset: float
    offset: Optional[float]
    is_change_target: bool = False


@dataclass
class ReconstructedTraceBundle:
    """
    Continuous DMD trace bundle reconstructed from trial-wise SLAP2 output.
    
    The bundle stores traces, sample times, trial validity, trial lengths, and session
    time bounds needed for event alignment.
    """
    traces: np.ndarray                 # (n_rois, n_total_samples)
    timebase_sec: np.ndarray          # (n_total_samples,)
    trial_valid_mask: np.ndarray      # (n_trials,)
    trial_lengths_samples: np.ndarray # (n_trials,)
    trial_starts_sec: np.ndarray      # (n_trials,)
    session_start_sec: float
    session_end_sec: float
    reconstructed_duration_sec: float
    metadata: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def load_corrected_bonsai_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a behavior event log containing HARP-corrected Bonsai timestamps.
    
    The function validates that a corrected timestamp column and stimulus ``Value``
    column are present before returning the DataFrame.
    """
    df = pd.read_csv(path)
    if "corrected_timestamp" not in df.columns and "corrected_timestamps" not in df.columns:
        raise ValueError(
            f"{path} does not contain corrected timestamp columns. "
            f"Expected 'corrected_timestamp' or 'corrected_timestamps'."
        )
    if "Value" not in df.columns:
        raise ValueError(f"{path} does not contain a 'Value' column.")
    return df


def load_imaging_epochs_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load imaging epoch boundaries used to filter behavior events.
    
    The returned DataFrame must contain ``start_time`` and ``end_time`` columns in the
    same corrected timebase as the event log.
    """
    df = pd.read_csv(path)
    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing epoch columns: {sorted(missing)}")
    return df


def _time_col(stim_df: pd.DataFrame) -> str:
    """
    Return the corrected timestamp column used by a stimulus DataFrame.
    
    Both singular and plural historical column names are supported.
    """
    if "corrected_timestamp" in stim_df.columns:
        return "corrected_timestamp"
    if "corrected_timestamps" in stim_df.columns:
        return "corrected_timestamps"
    raise ValueError("No corrected timestamp column found.")


# -----------------------------------------------------------------------------
# Stimulus parsing
# -----------------------------------------------------------------------------

def _normalize_value_series(stim_df: pd.DataFrame) -> pd.Series:
    """Return stimulus values as non-null strings for robust event parsing."""
    return stim_df["Value"].fillna("").astype(str)


def _is_image_value(v: str) -> bool:
    """
    Return True when a stimulus value represents an image file presentation.
    
    Photodiode diagnostic strings are explicitly excluded even if they contain image
    file-like text.
    """
    vl = v.lower()
    return vl.endswith((".tif", ".tiff")) and ("photodiode" not in vl)


def _is_change_value(v: str) -> bool:
    """Return True when a stimulus value marks a change event."""
    return "changeflash" in v.lower() or v.lower() == "change"


def _is_omission_value(v: str) -> bool:
    """Return True when a stimulus value marks an omitted-image event."""
    vl = v.lower()
    return vl == "omission" or "omission" in vl


def _is_nonstimulus_value(v: str) -> bool:
    """
    Return True for behavior-log rows that should not be treated as stimuli.
    
    This helper captures frame markers, gray periods, intertrial markers, and
    photodiode events.
    """
    vl = v.lower()
    return (
        vl in {"frame", "endframe", "endflash", "gray", "intertrial"}
        or "photodiode" in vl
    )


def extract_image_intervals(
    stim_df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
) -> Tuple[StimIntervalDict, List[OrderedImageEvent]]:
    """
    Extract image-onset events from Bonsai. Every image row is treated as one flash onset.
    """
    tcol = time_col or _time_col(stim_df)
    values = _normalize_value_series(stim_df)

    image_names = sorted([v for v in values.unique() if _is_image_value(v)])
    image_times: StimIntervalDict = {name: [] for name in image_names}
    ordered: List[OrderedImageEvent] = []

    event_idx = 0
    for _, row in stim_df.iterrows():
        val = str(row["Value"])
        if not _is_image_value(val):
            continue
        onset = float(row[tcol])
        image_times[val].append((onset, np.nan))
        ordered.append(
            OrderedImageEvent(
                event_idx=event_idx,
                image_name=val,
                onset=onset,
                offset=None,
                is_change_target=False,
            )
        )
        event_idx += 1

    return image_times, ordered


def extract_change_intervals(
    stim_df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
) -> StimIntervalList:
    """
    Extract corrected onset times for change events from a stimulus DataFrame.
    
    Intervals are returned as ``(onset, NaN)`` pairs to match the shared interval
    schema used by alignment helpers.
    """
    tcol = time_col or _time_col(stim_df)
    out: StimIntervalList = []
    for _, row in stim_df.iterrows():
        val = str(row["Value"])
        if _is_change_value(val):
            out.append((float(row[tcol]), np.nan))
    return out


def extract_omission_intervals(
    stim_df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
) -> StimIntervalList:
    """
    Extract corrected onset times for omission events from a stimulus DataFrame.
    
    Intervals are returned as ``(onset, NaN)`` pairs to match the shared interval
    schema used by alignment helpers.
    """
    tcol = time_col or _time_col(stim_df)
    out: StimIntervalList = []
    for _, row in stim_df.iterrows():
        val = str(row["Value"])
        if _is_omission_value(val):
            out.append((float(row[tcol]), np.nan))
    return out


def extract_ordered_change_targets(
    stim_df: pd.DataFrame,
    ordered_images: List[OrderedImageEvent],
    *,
    time_col: Optional[str] = None,
) -> List[int]:
    """
    Mark the first image event after each ChangeFlash as a change-target image.
    Returns ordered-image indices of those target images.
    """
    tcol = time_col or _time_col(stim_df)

    image_rows: List[Tuple[int, str, float]] = []
    change_rows: List[Tuple[int, float]] = []

    for i, row in stim_df.iterrows():
        val = str(row["Value"])
        if _is_image_value(val):
            image_rows.append((i, val, float(row[tcol])))
        elif _is_change_value(val):
            change_rows.append((i, float(row[tcol])))

    if not image_rows or not change_rows:
        return []

    change_target_ord_idxs: List[int] = []
    img_ptr = 0
    for change_i, _ in change_rows:
        while img_ptr < len(image_rows) and image_rows[img_ptr][0] <= change_i:
            img_ptr += 1
        if img_ptr >= len(image_rows):
            break
        next_onset = image_rows[img_ptr][2]
        for k, evt in enumerate(ordered_images):
            if k in change_target_ord_idxs:
                continue
            if np.isclose(evt.onset, next_onset):
                evt.is_change_target = True
                change_target_ord_idxs.append(k)
                break

    return change_target_ord_idxs


# -----------------------------------------------------------------------------
# Epoch filtering
# -----------------------------------------------------------------------------

def _epoch_bounds(epoch_df: pd.DataFrame, pre: float, post: float) -> List[Tuple[float, float]]:
    """
    Compute event-onset bounds that allow full pre/post windows within imaging epochs.
    
    Epoch starts are shifted forward by the pre-window and epoch ends are shifted
    backward by the post-window.
    """
    bounds = []
    for _, row in epoch_df.iterrows():
        a = float(row["start_time"]) + pre
        b = float(row["end_time"]) - post
        if a < b:
            bounds.append((a, b))
    return bounds


def _in_any_bound(t: float, bounds: List[Tuple[float, float]]) -> bool:
    """Return True when a timestamp falls within any valid epoch bound."""
    for a, b in bounds:
        if a <= t <= b:
            return True
    return False


def filter_intervals_to_epochs(
    stim_times: Union[StimIntervalDict, StimIntervalList],
    epoch_df: pd.DataFrame,
    *,
    pre_time: float,
    post_time: float,
) -> Union[StimIntervalDict, StimIntervalList]:
    """
    Filter stimulus intervals to events whose full extraction window is in an epoch.
    
    The function preserves whether the input is a dictionary of image identities or a
    single list of event intervals.
    """
    keep_bounds = _epoch_bounds(epoch_df, pre_time, post_time)

    if isinstance(stim_times, dict):
        out: StimIntervalDict = {}
        for name, intervals in stim_times.items():
            out[name] = [(t0, t1) for (t0, t1) in intervals if _in_any_bound(float(t0), keep_bounds)]
        return out

    return [(t0, t1) for (t0, t1) in stim_times if _in_any_bound(float(t0), keep_bounds)]


def filter_ordered_images_to_epochs(
    ordered_images: List[OrderedImageEvent],
    epoch_df: pd.DataFrame,
    *,
    pre_time: float,
    post_time: float,
) -> List[OrderedImageEvent]:
    """
    Filter ordered image events to those with full extraction windows inside epochs.
    """
    keep_bounds = _epoch_bounds(epoch_df, pre_time, post_time)
    return [evt for evt in ordered_images if _in_any_bound(evt.onset, keep_bounds)]


# -----------------------------------------------------------------------------
# Trace reconstruction
# -----------------------------------------------------------------------------

def _trial_trace_as_syn_by_time(
    exp: GlutamateSummary,
    dmd: int,
    trial: int,
    *,
    signal: str = "dF",
    mode: str = "ls",
    channel: int = 0,
) -> np.ndarray:
    """
    Return a trial trace matrix as (n_synapses, n_time).

    Accepts either:
    - 2D traces: (time, roi) or (roi, time)
    - 3D traces: (time, roi, channel)

    For 3D traces, selects the requested channel (default: channel 0,
    assumed glutamate/green for glutamate extraction).
    """
    x = exp.get_traces(
        dmd=dmd,
        trial=trial,
        signal=signal,
        mode=mode,
        squeeze_channels=False,
    )

    x = np.asarray(x, dtype=float)

    if x.ndim == 3:
        if x.shape[2] <= channel:
            raise ValueError(
                f"Requested channel {channel} but trace shape is {x.shape}"
            )
        x = x[:, :, channel]

    if x.ndim != 2:
        raise ValueError(f"Expected 2D trace matrix after channel selection, got shape {x.shape}")

    n0, n1 = x.shape
    expected_n_syn = int(exp.n_synapses[dmd - 1]) if len(exp.n_synapses) >= dmd else None

    if expected_n_syn is not None and n1 == expected_n_syn:
        return x.T
    if expected_n_syn is not None and n0 == expected_n_syn:
        return x
    if n0 >= n1:
        return x.T
    return x


def reconstruct_dmd_session_traces(
    exp: GlutamateSummary,
    dmd: int,
    *,
    im_rate_hz: float,
    epoch_start_sec: float,
    signal: str = "dF",
    mode: str = "ls",
) -> ReconstructedTraceBundle:
    """
    Reconstruct a session-wide trace by concatenating all trials in order.
    Invalid trials are represented by NaN blocks of the inferred trial length.
    """
    n_trials = int(exp.n_trials)
    valid_set = set(int(t) for t in exp.valid_trials[dmd - 1])

    valid_trial_data: Dict[int, np.ndarray] = {}
    valid_lengths: List[int] = []
    n_syn_expected = int(exp.n_synapses[dmd - 1]) if len(exp.n_synapses) >= dmd else None

    for trial in range(1, n_trials + 1):
        if trial not in valid_set:
            continue
        arr = _trial_trace_as_syn_by_time(
                exp,
                dmd=dmd,
                trial=trial,
                signal=signal,
                mode=mode,
                channel=0,
            )
        valid_trial_data[trial] = arr
        valid_lengths.append(arr.shape[1])
        if n_syn_expected is None:
            n_syn_expected = arr.shape[0]

    if not valid_trial_data:
        return ReconstructedTraceBundle(
            traces=np.empty((0, 0), dtype=float),
            timebase_sec=np.empty((0,), dtype=float),
            trial_valid_mask=np.zeros((n_trials,), dtype=bool),
            trial_lengths_samples=np.zeros((n_trials,), dtype=int),
            trial_starts_sec=np.zeros((n_trials,), dtype=float),
            session_start_sec=float(epoch_start_sec),
            session_end_sec=float(epoch_start_sec),
            reconstructed_duration_sec=0.0,
        )

    if n_syn_expected is None:
        n_syn_expected = next(iter(valid_trial_data.values())).shape[0]

    default_len = int(round(float(np.median(valid_lengths))))
    trial_lengths = np.full((n_trials,), default_len, dtype=int)
    for trial, arr in valid_trial_data.items():
        trial_lengths[trial - 1] = int(arr.shape[1])

    total_samples = int(np.sum(trial_lengths))
    traces = np.full((n_syn_expected, total_samples), np.nan, dtype=float)
    trial_valid_mask = np.zeros((n_trials,), dtype=bool)
    trial_starts_sec = np.zeros((n_trials,), dtype=float)

    pos = 0
    for trial in range(1, n_trials + 1):
        L = int(trial_lengths[trial - 1])
        trial_starts_sec[trial - 1] = float(epoch_start_sec + pos / im_rate_hz)
        if trial in valid_trial_data:
            arr = valid_trial_data[trial]
            s = min(n_syn_expected, arr.shape[0])
            LL = min(L, arr.shape[1])
            traces[:s, pos:pos + LL] = arr[:s, :LL]
            trial_valid_mask[trial - 1] = True
        pos += L

    timebase_sec = epoch_start_sec + np.arange(total_samples, dtype=float) / float(im_rate_hz)
    return ReconstructedTraceBundle(
        traces=traces,
        timebase_sec=timebase_sec,
        trial_valid_mask=trial_valid_mask,
        trial_lengths_samples=trial_lengths,
        trial_starts_sec=trial_starts_sec,
        session_start_sec=float(epoch_start_sec),
        session_end_sec=float(timebase_sec[-1]) if total_samples else float(epoch_start_sec),
        reconstructed_duration_sec=float(total_samples / im_rate_hz),
    )


# -----------------------------------------------------------------------------
# Session-wide alignment
# -----------------------------------------------------------------------------

def _resolve_bundle_timebase(bundle) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Return traces, a nominal session start, and an optional explicit sample timebase.

    If an explicit per-sample timebase is present and valid, downstream alignment
    should use it instead of assuming onset->sample conversion is exactly
    (t - session_start) * fs.
    """
    if isinstance(bundle, dict):
        traces = np.asarray(bundle["traces"], dtype=float)
        session_start_sec = float(bundle.get("session_start_sec", np.nan))
        tb = bundle.get("timebase_sec", None)
    else:
        traces = np.asarray(bundle.traces, dtype=float)
        session_start_sec = float(getattr(bundle, "session_start_sec", np.nan))
        tb = getattr(bundle, "timebase_sec", None)

    timebase_sec: Optional[np.ndarray]
    if tb is None:
        timebase_sec = None
    else:
        timebase_sec = np.asarray(tb, dtype=float).reshape(-1)
        if (
            traces.ndim != 2
            or timebase_sec.ndim != 1
            or timebase_sec.size != traces.shape[1]
            or timebase_sec.size == 0
            or not np.all(np.isfinite(timebase_sec))
            or np.any(np.diff(timebase_sec) <= 0)
        ):
            timebase_sec = None

    if not np.isfinite(session_start_sec):
        if timebase_sec is not None and timebase_sec.size:
            session_start_sec = float(timebase_sec[0])
        else:
            session_start_sec = 0.0

    return traces, np.asarray([session_start_sec], dtype=float), session_start_sec, timebase_sec


def _nearest_sample_index(
    onset_sec: float,
    *,
    session_start_sec: float,
    im_rate_hz: float,
    n_total_samples: int,
    timebase_sec: Optional[np.ndarray] = None,
) -> int:
    """
    Convert an event time to the nearest sample center.

    When a valid explicit sample timebase is available, use nearest-neighbor search
    on that timebase. Otherwise, fall back to the nominal fixed-rate mapping.
    """
    if n_total_samples <= 0:
        return 0

    if timebase_sec is not None and timebase_sec.size == n_total_samples:
        idx = int(np.searchsorted(timebase_sec, float(onset_sec), side="left"))
        if idx <= 0:
            return 0
        if idx >= n_total_samples:
            return n_total_samples - 1
        prev_idx = idx - 1
        if abs(float(timebase_sec[idx]) - float(onset_sec)) < abs(float(onset_sec) - float(timebase_sec[prev_idx])):
            return idx
        return prev_idx

    center = int(round((float(onset_sec) - float(session_start_sec)) * float(im_rate_hz)))
    return max(0, min(center, n_total_samples - 1))


def align_traces_to_session_intervals(
    bundle,
    stim_times,
    *,
    im_rate_hz: float,
    pre_time: float,
    post_time: float,
    return_used_onsets: bool = True,
):
    """
    Extract event-aligned trace snippets for stimulus intervals.
    
    The function supports image-identity dictionaries and flat interval lists, using
    an explicit bundle timebase when available and otherwise falling back to fixed-rate
    sample conversion.
    """
    n_pre = int(round(pre_time * im_rate_hz))
    n_post = int(round(post_time * im_rate_hz))
    n_win = n_pre + n_post

    traces, _session_start_arr, session_start_sec, timebase_sec = _resolve_bundle_timebase(bundle)

    def _extract_one_list(intervals):
        """Extract aligned snippets and retained onset times for one interval list."""
        snippets = []
        kept_onsets = []

        for onset, _ in intervals:
            center = _nearest_sample_index(
                float(onset),
                session_start_sec=session_start_sec,
                im_rate_hz=im_rate_hz,
                n_total_samples=traces.shape[1],
                timebase_sec=timebase_sec,
            )
            start = center - n_pre
            stop = start + n_win
            if start < 0 or stop > traces.shape[1]:
                continue
            snippets.append(traces[:, start:stop])
            kept_onsets.append(float(onset))

        if len(snippets) == 0:
            arr = np.full((0, traces.shape[0], n_win), np.nan, dtype=float)
            onsets = np.array([], dtype=float)
        else:
            arr = np.stack(snippets, axis=0)
            onsets = np.asarray(kept_onsets, dtype=float)

        return arr, onsets

    if isinstance(stim_times, dict):
        aligned = {}
        onset_dict = {}
        for k, v in stim_times.items():
            arr, onsets = _extract_one_list(v)
            aligned[k] = arr
            onset_dict[k] = onsets
        if return_used_onsets:
            return aligned, onset_dict
        return aligned

    arr, onsets = _extract_one_list(stim_times)
    if return_used_onsets:
        return arr, onsets
    return arr


# -----------------------------------------------------------------------------
# Sequence construction and summaries
# -----------------------------------------------------------------------------

def summarize_event_tensor(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Summarize an event tensor with mean, standard deviation, event count, and finite
    counts.
    
    The expected input shape is events by synapses by time.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (n_events, n_synapses, n_time), got {x.shape}")
    n_events = int(x.shape[0])
    counts = np.sum(np.isfinite(x), axis=0)
    out_shape = x.shape[1:]
    return {
        "mean": np.nanmean(x, axis=0) if n_events else np.full(out_shape, np.nan),
        "std": np.nanstd(x, axis=0, ddof=0) if n_events else np.full(out_shape, np.nan),
        "n_events": np.array(n_events, dtype=int),
        "n_finite": counts.astype(int),
    }


def tolerant_summary_ragged(arrays: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    arrays: list of (seq_len_i, n_syn, n_time)
    Returns tolerant mean/std/count across ragged sequence-position axis.
    """
    if len(arrays) == 0:
        return {
            "mean": np.empty((0, 0, 0), dtype=float),
            "std": np.empty((0, 0, 0), dtype=float),
            "counts": np.empty((0,), dtype=int),
        }

    max_len = max(a.shape[0] for a in arrays)
    n_syn, n_time = arrays[0].shape[1], arrays[0].shape[2]
    stack = np.full((len(arrays), max_len, n_syn, n_time), np.nan, dtype=float)
    for i, a in enumerate(arrays):
        stack[i, : a.shape[0]] = a

    valid = np.isfinite(stack).any(axis=(2, 3))
    counts = valid.sum(axis=0).astype(int)
    return {
        "mean": np.nanmean(stack, axis=0),
        "std": np.nanstd(stack, axis=0, ddof=0),
        "counts": counts,
    }


def build_change_locked_sequences(ordered_images: List[OrderedImageEvent]) -> Dict[str, Dict[str, Any]]:
    """
    Build change-locked ordered image sequences keyed by changed-to image identity.

    For each change-target image:
      - prechange: ordered images at positions -2, -1
      - repeated: from the changed-to image (position 0) up to the image immediately
        before the next change-target image
      - terminal: the next change-target image, held out separately
    """
    out: Dict[str, Dict[str, Any]] = {}
    change_idxs = [i for i, evt in enumerate(ordered_images) if evt.is_change_target]

    for j, idx in enumerate(change_idxs):
        if idx < 2:
            continue
        evt0 = ordered_images[idx]
        img = evt0.image_name
        out.setdefault(img, {"prechange": [], "repeated": [], "terminal": []})

        pre = [ordered_images[idx - 2], ordered_images[idx - 1]]
        if j + 1 < len(change_idxs):
            next_idx = change_idxs[j + 1]
            repeated = ordered_images[idx:next_idx]
            terminal = ordered_images[next_idx]
        else:
            repeated = ordered_images[idx:]
            terminal = ordered_images[-1]

        out[img]["prechange"].append(pre)
        out[img]["repeated"].append(repeated)
        out[img]["terminal"].append(terminal)

    return out