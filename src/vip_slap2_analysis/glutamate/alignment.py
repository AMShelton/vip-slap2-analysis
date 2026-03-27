from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from vip_slap2_analysis.glutamate.summary import GlutamateSummary
from vip_slap2_analysis.utils.utils import tolerant_mean


Interval = Tuple[float, float]
StimIntervalList = List[Interval]
StimIntervalDict = Dict[str, StimIntervalList]


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class EventWindows:
    image: Tuple[float, float] = (0.25, 0.50)
    change: Tuple[float, float] = (1.00, 0.75)
    omission: Tuple[float, float] = (1.00, 1.50)


@dataclass
class OrderedImageEvent:
    event_idx: int
    image_name: str
    onset: float
    offset: Optional[float]
    is_change_target: bool = False


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def load_corrected_bonsai_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "corrected_timestamp" not in df.columns and "corrected_timestamps" not in df.columns:
        raise ValueError(
            f"{path} does not contain corrected timestamp columns. "
            "Expected 'corrected_timestamp' or 'corrected_timestamps'."
        )
    return df


def load_imaging_epochs_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing epoch columns: {sorted(missing)}")
    return df


def _time_col(stim_df: pd.DataFrame) -> str:
    if "corrected_timestamp" in stim_df.columns:
        return "corrected_timestamp"
    if "corrected_timestamps" in stim_df.columns:
        return "corrected_timestamps"
    raise ValueError("No corrected timestamp column found.")


# -----------------------------------------------------------------------------
# Stimulus parsing
# -----------------------------------------------------------------------------

def extract_image_intervals(
    stim_df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
) -> Tuple[StimIntervalDict, List[OrderedImageEvent]]:
    """
    Parse image onset/offset intervals from the Bonsai event table.

    Returns
    -------
    image_times
        Dict mapping image_name -> list of (onset, offset)
    ordered_image_events
        Flat list preserving appearance order
    """
    tcol = time_col or _time_col(stim_df)
    values = stim_df["Value"].astype(str)

    image_names = sorted([v for v in values.unique() if v.lower().endswith((".tif", ".tiff"))])
    image_times: StimIntervalDict = {name: [] for name in image_names}
    ordered: List[OrderedImageEvent] = []

    i = 0
    event_idx = 0
    while i < len(stim_df):
        row = stim_df.iloc[i]
        val = str(row["Value"])
        if val.lower().endswith((".tif", ".tiff")):
            onset = float(row[tcol])
            offset = np.nan
            j = i + 1
            while j < len(stim_df):
                if str(stim_df.iloc[j]["Value"]) == "EndFlash":
                    offset = float(stim_df.iloc[j][tcol])
                    break
                j += 1

            image_times[val].append((onset, offset))
            ordered.append(
                OrderedImageEvent(
                    event_idx=event_idx,
                    image_name=val,
                    onset=onset,
                    offset=offset if np.isfinite(offset) else None,
                    is_change_target=False,
                )
            )
            event_idx += 1
            i = max(i + 1, j + 1)
        else:
            i += 1

    return image_times, ordered


def extract_named_intervals(
    stim_df: pd.DataFrame,
    target_value: str,
    *,
    time_col: Optional[str] = None,
) -> StimIntervalList:
    """
    Return a list of (onset, offset) for rows whose Value contains target_value.
    Offset is the next EndFlash if present, otherwise NaN.
    """
    tcol = time_col or _time_col(stim_df)
    out: StimIntervalList = []

    i = 0
    while i < len(stim_df):
        row = stim_df.iloc[i]
        if target_value in str(row["Value"]):
            onset = float(row[tcol])
            offset = np.nan
            j = i + 1
            while j < len(stim_df):
                if str(stim_df.iloc[j]["Value"]) == "EndFlash":
                    offset = float(stim_df.iloc[j][tcol])
                    break
                j += 1
            out.append((onset, offset))
            i = max(i + 1, j + 1)
        else:
            i += 1

    return out


def extract_ordered_change_targets(
    stim_df: pd.DataFrame,
    ordered_images: List[OrderedImageEvent],
    *,
    time_col: Optional[str] = None,
) -> List[int]:
    """
    Mark which ordered image events are the first image after each ChangeFlash.
    Returns the list of ordered-image indices corresponding to change-target images.
    """
    tcol = time_col or _time_col(stim_df)

    image_rows = []
    for i, row in stim_df.iterrows():
        val = str(row["Value"])
        if val.lower().endswith((".tif", ".tiff")):
            image_rows.append((i, val, float(row[tcol])))

    change_rows = []
    for i, row in stim_df.iterrows():
        if "ChangeFlash" in str(row["Value"]):
            change_rows.append((i, float(row[tcol])))

    if not image_rows or not change_rows:
        return []

    onset_to_ordidx = {evt.onset: k for k, evt in enumerate(ordered_images)}
    change_target_ord_idxs = []

    for change_i, change_t in change_rows:
        next_img = None
        for row_i, img_name, onset in image_rows:
            if row_i > change_i:
                next_img = (img_name, onset)
                break
        if next_img is not None and next_img[1] in onset_to_ordidx:
            ord_idx = onset_to_ordidx[next_img[1]]
            ordered_images[ord_idx].is_change_target = True
            change_target_ord_idxs.append(ord_idx)

    return change_target_ord_idxs


# -----------------------------------------------------------------------------
# Epoch filtering
# -----------------------------------------------------------------------------

def _epoch_bounds(
    epoch_df: pd.DataFrame,
    pre: float,
    post: float,
) -> List[Tuple[float, float]]:
    bounds = []
    for _, row in epoch_df.iterrows():
        a = float(row["start_time"]) + pre
        b = float(row["end_time"]) - post
        if a < b:
            bounds.append((a, b))
    return bounds


def _in_any_bound(t: float, bounds: List[Tuple[float, float]]) -> bool:
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
    keep_bounds = _epoch_bounds(epoch_df, pre_time, post_time)

    if isinstance(stim_times, dict):
        out: StimIntervalDict = {}
        for name, intervals in stim_times.items():
            kept = [(t0, t1) for (t0, t1) in intervals if _in_any_bound(float(t0), keep_bounds)]
            if kept:
                out[name] = kept
        return out

    return [(t0, t1) for (t0, t1) in stim_times if _in_any_bound(float(t0), keep_bounds)]


def filter_ordered_images_to_epochs(
    ordered_images: List[OrderedImageEvent],
    epoch_df: pd.DataFrame,
    *,
    pre_time: float,
    post_time: float,
) -> List[OrderedImageEvent]:
    keep_bounds = _epoch_bounds(epoch_df, pre_time, post_time)
    return [evt for evt in ordered_images if _in_any_bound(evt.onset, keep_bounds)]


# -----------------------------------------------------------------------------
# Trace collection
# -----------------------------------------------------------------------------

def collect_dmd_trial_traces(
    exp: GlutamateSummary,
    dmd: int,
    *,
    signal: str = "dF",
    mode: str = "ls",
) -> np.ndarray:
    """
    Returns array of shape (n_synapses, n_trials_total, n_samples), NaN-filled for invalid trials.
    """
    dmd0 = dmd - 1
    valid_trials = list(exp.valid_trials[dmd0])
    n_trials_total = int(exp.n_trials)

    if len(valid_trials) == 0:
        return np.empty((0, 0, 0), dtype=float)

    ref = exp.get_traces(dmd=dmd, trial=int(valid_trials[0]), signal=signal, mode=mode, squeeze_channels=True)
    if ref.ndim != 2:
        raise ValueError(f"Expected 2D trace matrix, got shape {ref.shape}")

    n_samples, n_syn = ref.shape
    out = np.full((n_syn, n_trials_total, n_samples), np.nan, dtype=float)

    for trial in range(1, n_trials_total + 1):
        try:
            x = exp.get_traces(dmd=dmd, trial=trial, signal=signal, mode=mode, squeeze_channels=True)
            s = min(n_syn, x.shape[1])
            L = min(n_samples, x.shape[0])
            out[:s, trial - 1, :L] = x[:L, :s].T
        except Exception:
            continue

    return out


# -----------------------------------------------------------------------------
# Event-to-trace alignment
# -----------------------------------------------------------------------------

def _find_epoch_index_for_onset(onset: float, epoch_df: pd.DataFrame) -> Optional[int]:
    hits = np.where((epoch_df["start_time"].to_numpy() <= onset) &
                    (onset <= epoch_df["end_time"].to_numpy()))[0]
    if len(hits) == 0:
        return None
    return int(hits[0])


def align_traces_to_intervals(
    traces_stl: np.ndarray,
    stim_times: Union[StimIntervalDict, StimIntervalList],
    epoch_df: pd.DataFrame,
    *,
    im_rate_hz: float,
    pre_time: float,
    post_time: float,
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Align traces to onset times.

    traces_stl : (n_synapses, n_trials, n_samples)
    epoch_df   : ordered epochs, assumed to correspond to trial axis when in trial mode
    """
    n_pre = int(round(pre_time * im_rate_hz))
    n_post = int(round(post_time * im_rate_hz))
    n_win = n_pre + n_post

    def _extract_one_list(intervals: StimIntervalList) -> np.ndarray:
        snippets = []
        for onset, _ in intervals:
            epoch_idx = _find_epoch_index_for_onset(float(onset), epoch_df)
            if epoch_idx is None:
                continue
            if epoch_idx >= traces_stl.shape[1]:
                continue

            trial_start = float(epoch_df.iloc[epoch_idx]["start_time"])
            center_sample = int(round((float(onset) - trial_start) * im_rate_hz))
            start = center_sample - n_pre
            stop = start + n_win
            if start < 0 or stop > traces_stl.shape[2]:
                continue

            snippets.append(traces_stl[:, epoch_idx, start:stop])

        if len(snippets) == 0:
            return np.full((0, traces_stl.shape[0], n_win), np.nan, dtype=float)
        return np.stack(snippets, axis=0)  # (n_events, n_syn, n_time)

    if isinstance(stim_times, dict):
        return {k: _extract_one_list(v) for k, v in stim_times.items()}
    return _extract_one_list(stim_times)


# -----------------------------------------------------------------------------
# Sequence construction
# -----------------------------------------------------------------------------

def build_change_locked_sequences(
    ordered_images: List[OrderedImageEvent],
) -> Dict[str, Dict[str, Any]]:
    """
    Build ragged image sequences keyed by changed-to image identity.

    For each change-target image:
      - include two preceding image presentations: positions -2, -1
      - include all subsequent image presentations from the changed-to image up to and including
        the next changed-to image (held out separately as sequence_terminal)

    Returns
    -------
    dict
        {
          image_name: {
            "prechange": [list[OrderedImageEvent], ...],   # always len 2 if complete
            "sequence": [list[OrderedImageEvent], ...],    # 0..terminal inclusive
            "terminal": [OrderedImageEvent, ...],          # final held-out next-change image
          }
        }
    """
    out: Dict[str, Dict[str, Any]] = {}

    change_ord_idxs = [i for i, evt in enumerate(ordered_images) if evt.is_change_target]
    for k, ord_idx in enumerate(change_ord_idxs):
        evt0 = ordered_images[ord_idx]
        image_name = evt0.image_name

        if image_name not in out:
            out[image_name] = {"prechange": [], "sequence": [], "terminal": []}

        if ord_idx < 2:
            continue

        pre_events = [ordered_images[ord_idx - 2], ordered_images[ord_idx - 1]]

        if k + 1 < len(change_ord_idxs):
            next_change_idx = change_ord_idxs[k + 1]
            seq_events = ordered_images[ord_idx: next_change_idx + 1]
            terminal_evt = ordered_images[next_change_idx]
        else:
            seq_events = ordered_images[ord_idx:]
            terminal_evt = seq_events[-1]

        out[image_name]["prechange"].append(pre_events)
        out[image_name]["sequence"].append(seq_events)
        out[image_name]["terminal"].append(terminal_evt)

    return out


def summarize_event_tensor(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    x: (n_events, n_synapses, n_time)
    """
    return {
        "mean": np.nanmean(x, axis=0),
        "std": np.nanstd(x, axis=0, ddof=0),
        "n": np.sum(np.isfinite(x).any(axis=2), axis=0),
    }


def tolerant_mean_over_ragged(
    arrays: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    arrays: list of arrays with shape (n_synapses, n_time) but varying first axis length
            along the sequence-position dimension before this function is called.
    """
    if len(arrays) == 0:
        return np.empty((0, 0, 0)), np.empty((0, 0, 0))
    stacked_mean, stacked_std = tolerant_mean(arrays, axis=0)
    return stacked_mean, stacked_std