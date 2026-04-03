from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


ArrayLike2D = np.ndarray


def infer_max_trial_length(trials: Sequence[ArrayLike2D | None]) -> int:
    lengths = [int(tr.shape[-1]) for tr in trials if tr is not None]
    if not lengths:
        raise ValueError("Could not infer trial length: no non-empty trials were provided.")
    return max(lengths)


def infer_n_signals(trials: Sequence[ArrayLike2D | None]) -> int:
    counts = [int(tr.shape[0]) for tr in trials if tr is not None]
    if not counts:
        raise ValueError("Could not infer signal count: no non-empty trials were provided.")
    return max(counts)


def pad_trial_to_length(
    trial: ArrayLike2D,
    *,
    target_length: int,
    n_signals: int | None = None,
    fill_value: float = np.nan,
) -> ArrayLike2D:
    trial = np.asarray(trial, dtype=float)
    if trial.ndim != 2:
        raise ValueError(f"Expected a 2D array shaped (n_signals, time). Got {trial.shape}.")

    if n_signals is None:
        n_signals = int(trial.shape[0])

    out = np.full((int(n_signals), int(target_length)), fill_value, dtype=float)
    rcopy = min(out.shape[0], trial.shape[0])
    tcopy = min(out.shape[1], trial.shape[1])
    out[:rcopy, :tcopy] = trial[:rcopy, :tcopy]
    return out


def stack_trials_padded(
    trials: Sequence[ArrayLike2D | None],
    *,
    fill_length: int | None = None,
    n_signals: int | None = None,
    fill_value: float = np.nan,
) -> ArrayLike2D:
    """
    Convert a sequence of per-trial 2D arrays into a padded stack.

    Parameters
    ----------
    trials:
        Sequence of arrays shaped ``(n_signals, time)`` or ``None`` for invalid trials.
    fill_length:
        Length used for padding and for invalid trials. If omitted, inferred as the
        maximum valid-trial length.
    n_signals:
        Number of signals/ROIs. If omitted, inferred from the valid trials.
    """
    if fill_length is None:
        fill_length = infer_max_trial_length(trials)
    if n_signals is None:
        n_signals = infer_n_signals(trials)

    out = np.full((len(trials), int(n_signals), int(fill_length)), fill_value, dtype=float)
    for i, trial in enumerate(trials):
        if trial is None:
            continue
        out[i] = pad_trial_to_length(
            trial,
            target_length=int(fill_length),
            n_signals=int(n_signals),
            fill_value=fill_value,
        )
    return out


def concatenate_trial_stack(trial_stack: ArrayLike2D) -> ArrayLike2D:
    """Flatten a padded stack of shape (n_trials, n_signals, T) into (n_signals, n_trials*T)."""
    trial_stack = np.asarray(trial_stack, dtype=float)
    if trial_stack.ndim != 3:
        raise ValueError(f"Expected trial_stack with ndim=3. Got shape {trial_stack.shape}.")
    n_trials, n_signals, n_time = trial_stack.shape
    return np.transpose(trial_stack, (1, 0, 2)).reshape(n_signals, n_trials * n_time)


def trial_lengths(trials: Sequence[ArrayLike2D | None], *, invalid_fill_length: int | None = None) -> list[int | None]:
    out: list[int | None] = []
    for tr in trials:
        if tr is None:
            out.append(None if invalid_fill_length is None else int(invalid_fill_length))
        else:
            out.append(int(np.asarray(tr).shape[-1]))
    return out


def trial_start_times_seconds(n_trials: int, trial_length_samples: int, fs_hz: float) -> np.ndarray:
    return np.arange(int(n_trials), dtype=float) * (float(trial_length_samples) / float(fs_hz))
