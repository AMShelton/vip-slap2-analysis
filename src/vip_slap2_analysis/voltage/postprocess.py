"""Postprocess voltage-summary ROI traces for downstream analysis.

This module provides utilities for concatenating per-trial voltage ROI traces
while retaining trial-to-session slice bookkeeping. It assumes a
:class:`vip_slap2_analysis.voltage.summary.VoltageSummary`-like object exposing
valid trials, ROI counts, ROI traces, and discard-frame masks.

Shape convention
----------------
Low-level voltage trace readers return arrays as ``(n_samples, n_rois)``.  This
is the canonical orientation for voltage traces throughout ``voltage.summary``
and ``voltage.postprocess``.  Functions that need ROI-major arrays for event
alignment should transpose explicitly at the boundary rather than relying on an
ambiguous reader orientation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _as_time_by_roi(
    traces: np.ndarray,
    *,
    n_rois: int,
    dmd: int,
    trial: int,
) -> np.ndarray:
    """Validate that a trace block is two-dimensional ``(n_samples, n_rois)``.

    Parameters
    ----------
    traces : array-like
        Trace block returned by ``VoltageSummary.get_roi_traces``.
    n_rois : int
        Expected number of DMD-local ROIs.
    dmd, trial : int
        One-indexed DMD and trial values used only for informative errors.

    Returns
    -------
    np.ndarray
        Validated two-dimensional trace block.
    """
    x = np.asarray(traces)
    if x.ndim == 1:
        if n_rois != 1:
            raise ValueError(
                "Voltage trace block is 1D but more than one ROI was expected "
                f"for dmd={dmd}, trial={trial}: shape={x.shape}, n_rois={n_rois}."
            )
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(
            "Voltage trace blocks must be 2D with shape (n_samples, n_rois); "
            f"got shape={x.shape} for dmd={dmd}, trial={trial}."
        )

    if x.shape[1] != n_rois:
        # A common legacy mistake was to treat arrays as ROI x time.  Do not
        # silently transpose here because a session with n_samples == n_rois
        # would be ambiguous and because orientation bugs corrupt event timing.
        orientation_hint = ""
        if x.shape[0] == n_rois and x.shape[1] != n_rois:
            orientation_hint = " The block looks ROI x time; transpose upstream."
        raise ValueError(
            "Voltage trace blocks must use shape (n_samples, n_rois). "
            f"Expected {n_rois} ROI columns for dmd={dmd}, trial={trial}, "
            f"but got shape={x.shape}.{orientation_hint}"
        )

    return x


def _align_discard_mask(mask: np.ndarray, n_samples: int) -> np.ndarray:
    """Return a one-dimensional boolean discard mask with length ``n_samples``."""
    discard = np.asarray(mask).astype(bool).squeeze()
    if discard.ndim != 1:
        discard = discard.reshape(-1)

    if discard.size == n_samples:
        return discard
    if discard.size > n_samples:
        return discard[:n_samples]

    aligned = np.zeros(n_samples, dtype=bool)
    aligned[: discard.size] = discard
    return aligned


def concat_rois_across_trials(
    vs,
    dmd: int = 1,
    drop_discarded: bool = True,
    dtype=np.float32,
    return_array: bool = False,
):
    """Concatenate DMD-local voltage ROI traces across valid trials.

    ``VoltageSummary.get_roi_traces`` is expected to return each trial as
    ``(n_samples, n_rois)``.  This function preserves that orientation while
    concatenating in time.

    Parameters
    ----------
    vs : VoltageSummary-like
        Object exposing ``valid_trials``, ``n_rois``, ``get_roi_traces``, and
        ``get_discard_frames``.
    dmd : int, default=1
        One-indexed DMD number.
    drop_discarded : bool, default=True
        If True, remove samples flagged by the trial's discard-frame mask.
    dtype : numpy dtype, default=np.float32
        Output dtype requested from ``get_roi_traces``.
    return_array : bool, default=False
        If False, preserve the historical return type: a list of length
        ``n_rois``, with one concatenated one-dimensional array per ROI.  If
        True, return a single array with shape ``(n_total_samples, n_rois)``.

    Returns
    -------
    roi_traces : list[np.ndarray] or np.ndarray
        If ``return_array=False``, a list of per-ROI one-dimensional traces.  If
        ``return_array=True``, a time-by-ROI array with shape
        ``(n_total_samples, n_rois)``.
    trial_slices : list[tuple[int, slice]]
        Each tuple contains the one-indexed trial number and the corresponding
        slice into the concatenated time axis.

    Notes
    -----
    The historical implementation used ``X.shape[1]`` as the segment length and
    appended ``X[r, :]`` for each ROI, which treated ``X`` as ROI x time despite
    the documented ``(n_samples, n_rois)`` convention.  That silently produced
    trial slices of length ``n_rois`` and concatenated across ROI columns rather
    than time samples.  This implementation uses ``X.shape[0]`` for time and
    ``X[:, r]`` for ROI traces.
    """
    dmd0 = int(dmd) - 1
    valid_trials = list(vs.valid_trials[dmd0])
    n_rois = int(vs.n_rois[dmd0])

    chunks_by_roi: List[List[np.ndarray]] = [[] for _ in range(n_rois)]
    chunks_time_by_roi: List[np.ndarray] = []
    trial_slices: List[Tuple[int, slice]] = []
    t_cursor = 0

    for trial in valid_trials:
        trial = int(trial)
        x = vs.get_roi_traces(
            dmd=dmd,
            trial=trial,
            drop_discarded=False,
            dtype=dtype,
        )
        x = _as_time_by_roi(x, n_rois=n_rois, dmd=dmd, trial=trial)

        if drop_discarded:
            discard = vs.get_discard_frames(dmd=dmd, trial=trial)
            discard = _align_discard_mask(discard, x.shape[0])
            x = x[~discard, :]

        seg_len = int(x.shape[0])
        trial_slices.append((trial, slice(t_cursor, t_cursor + seg_len)))
        t_cursor += seg_len

        if return_array:
            chunks_time_by_roi.append(x)
        else:
            for roi_idx in range(n_rois):
                chunks_by_roi[roi_idx].append(x[:, roi_idx])

    if return_array:
        if chunks_time_by_roi:
            roi_traces = np.concatenate(chunks_time_by_roi, axis=0).astype(dtype, copy=False)
        else:
            roi_traces = np.empty((0, n_rois), dtype=dtype)
        return roi_traces, trial_slices

    roi_traces = [
        np.concatenate(chunks, axis=0).astype(dtype, copy=False)
        if chunks
        else np.array([], dtype=dtype)
        for chunks in chunks_by_roi
    ]

    return roi_traces, trial_slices
