from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Literal, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


EncoderUnits = Literal["ticks", "degrees", "radians", "cm", "m"]
TimeZero = Literal["first_sample", "session_start", "none"]
UnwrapPosition = Union[bool, Literal["auto"]]


def _odd_kernel(kernel: Optional[int]) -> int:
    """Return a positive odd median-filter kernel size, or 0 to disable."""
    if kernel is None:
        return 0
    kernel = int(kernel)
    if kernel <= 1:
        return 0
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def _infer_counter_period(
    values: np.ndarray,
    dtype: np.dtype,
    *,
    unwrap_position: UnwrapPosition = "auto",
    counter_period: Optional[float] = None,
    counter_bits: Optional[int] = None,
) -> Optional[float]:
    """Infer the period of a wrapping integer encoder counter.

    HARP encoder pickles produced by this repo usually store the ``Encoder``
    column as a signed int16 counter. That counter is cumulative only modulo
    2**16, so it jumps from +32767 to -32768 (or the reverse) when it wraps.
    """
    if counter_period is not None:
        period = float(counter_period)
        if not np.isfinite(period) or period <= 0:
            raise ValueError("counter_period must be a positive finite number")
        return period

    if counter_bits is not None:
        bits = int(counter_bits)
        if bits <= 0:
            raise ValueError("counter_bits must be positive")
        return float(2**bits)

    if unwrap_position is False:
        return None

    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return float(2 ** (8 * dtype.itemsize))

    if unwrap_position is True:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        span = float(np.nanmax(finite) - np.nanmin(finite))
        # Float exports are uncommon, but sometimes integer counters are cast to
        # float before saving. Only infer common ADC/counter widths when the data
        # span is consistent with that representation.
        common_periods = (2**16, 2**32)
        for period in common_periods:
            if 0.35 * period <= span <= 1.05 * period:
                return float(period)
        raise ValueError(
            "unwrap_position=True was requested, but counter_period/counter_bits "
            "could not be inferred from a non-integer encoder column. Pass "
            "counter_period=... or counter_bits=..., or use unwrap_position='auto'."
        )

    if unwrap_position != "auto":
        raise ValueError("unwrap_position must be True, False, or 'auto'")

    return None


def unwrap_encoder_position(
    position: np.ndarray,
    *,
    counter_period: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unwrap a modular encoder-position trace.

    Parameters
    ----------
    position:
        Raw position values from the encoder counter.
    counter_period:
        Counter period. For signed int16 HARP encoder data this is 65536.

    Returns
    -------
    unwrapped, diagnostics
        ``unwrapped`` is a cumulative position trace in the same units as the
        raw counter. ``diagnostics`` reports wrap locations and corrections.
    """
    position = np.asarray(position, dtype=float)
    if position.size == 0:
        return position.copy(), {
            "counter_period": float(counter_period),
            "n_wraps": 0,
            "wrap_indices": np.array([], dtype=int),
            "wrap_deltas": np.array([], dtype=float),
        }

    period = float(counter_period)
    if not np.isfinite(period) or period <= 0:
        raise ValueError("counter_period must be a positive finite number")

    d_raw = np.diff(position)
    d_unwrapped = d_raw.copy()
    half_period = period / 2.0

    high = d_unwrapped > half_period
    low = d_unwrapped < -half_period
    d_unwrapped[high] -= period
    d_unwrapped[low] += period

    unwrapped = np.empty_like(position, dtype=float)
    unwrapped[0] = position[0]
    if position.size > 1:
        unwrapped[1:] = position[0] + np.cumsum(d_unwrapped)

    wrap_mask = high | low
    diagnostics = {
        "counter_period": period,
        "n_wraps": int(np.sum(wrap_mask)),
        # Index i means the wrap was between samples i and i+1.
        "wrap_indices": np.flatnonzero(wrap_mask).astype(int),
        "wrap_deltas": d_raw[wrap_mask].astype(float),
    }
    return unwrapped, diagnostics


def _safe_velocity(y: np.ndarray, time_sec: np.ndarray) -> np.ndarray:
    """Numerically differentiate y with respect to time_sec."""
    y = np.asarray(y, dtype=float)
    time_sec = np.asarray(time_sec, dtype=float)
    if y.size == 0:
        return np.array([], dtype=float)
    if y.size == 1:
        return np.zeros_like(y, dtype=float)

    dt = np.gradient(time_sec)
    dt[~np.isfinite(dt)] = np.nan
    dt[dt <= 0] = np.nan

    dy = np.gradient(y)
    velocity = dy / dt
    velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
    return velocity


def compute_encoder_velocity(
    encoder_path: str | Path,
    encoder_col: str = "Encoder",
    wheel_radius_cm: float = 8.0,
    encoder_units: EncoderUnits = "ticks",
    ticks_per_revolution: Optional[float] = None,
    session_start_sec: Optional[float] = None,
    time_zero: TimeZero = "first_sample",
    time_offset_sec: float = 0.0,
    median_filter_kernel: int = 51,
    smooth_sigma_samples: Optional[float] = 3.0,
    absolute_velocity: bool = False,
    unwrap_position: UnwrapPosition = "auto",
    counter_period: Optional[float] = None,
    counter_bits: Optional[int] = None,
    max_abs_linear_velocity_cm_s: Optional[float] = None,
) -> dict[str, np.ndarray | str | float | int | None]:
    """
    Load encoder.pkl and compute angular and linear velocity.

    The behavior-processing pathway saves HARP ``AnalogData['Encoder']`` values
    to ``encoder.pkl``. For the HARP files used in this project, that column is
    often a signed int16 modular counter, not an already-unwrapped distance
    trace. In that case the raw position jumps from +32767 to -32768 whenever
    the counter wraps. Differentiating that wrapped trace creates enormous
    one-sample velocity artifacts. This function unwraps modular integer
    counters by default before filtering and differentiating.

    Parameters
    ----------
    encoder_path:
        Path to encoder.pkl.
    encoder_col:
        Encoder column name.
    wheel_radius_cm:
        Radius of the running wheel in cm.
    encoder_units:
        Units of the *unwrapped* encoder position. For HARP counter data use
        ``"ticks"`` and provide ``ticks_per_revolution``.
    ticks_per_revolution:
        Encoder counts per wheel revolution. Required when
        ``encoder_units='ticks'``.
    time_zero:
        How to zero the encoder time axis: ``"first_sample"``,
        ``"session_start"``, or ``"none"``.
    median_filter_kernel:
        Optional median filter applied to the unwrapped position before
        differentiating. Set to <= 1 to disable.
    smooth_sigma_samples:
        Optional Gaussian smoothing sigma applied to velocity traces after
        differentiating.
    absolute_velocity:
        If True, return non-negative angular and linear velocity traces.
    unwrap_position:
        ``"auto"`` unwraps integer encoder columns using their dtype width,
        e.g. int16 -> 65536. True forces unwrapping and may require
        ``counter_period`` or ``counter_bits`` for non-integer columns. False
        disables unwrapping.
    counter_period, counter_bits:
        Explicit modular-counter period, or number of counter bits.
    max_abs_linear_velocity_cm_s:
        Optional artifact guard. Values with absolute linear velocity above this
        threshold are set to NaN before final smoothing, then converted to zero
        if they remain non-finite. Usually unnecessary after unwrapping.

    Returns
    -------
    dict
        Contains time, raw/wrapped position, unwrapped position, filtered
        position, angle, signed velocity, speed, and wrap diagnostics.
    """
    encoder_path = Path(encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    df = pd.read_pickle(encoder_path)
    if encoder_col not in df.columns:
        raise KeyError(
            f"Encoder column {encoder_col!r} not found in {encoder_path}. "
            f"Available columns: {list(df.columns)}"
        )

    raw_time = df.index.to_numpy(dtype=float)
    series = df[encoder_col]
    raw_dtype = series.dtype
    position_raw = series.to_numpy(dtype=float)

    empty = {
        "time_sec": np.array([], dtype=float),
        "position_raw": np.array([], dtype=float),
        "position_unwrapped": np.array([], dtype=float),
        "position_filtered": np.array([], dtype=float),
        "angle_rad": np.array([], dtype=float),
        "position_velocity_per_s": np.array([], dtype=float),
        "position_speed_per_s": np.array([], dtype=float),
        "angular_velocity_rad_s": np.array([], dtype=float),
        "linear_velocity_cm_s": np.array([], dtype=float),
        "linear_speed_cm_s": np.array([], dtype=float),
        "path": str(encoder_path),
        "counter_period": None,
        "n_wraps": 0,
        "wrap_indices": np.array([], dtype=int),
        "wrap_deltas": np.array([], dtype=float),
    }
    if raw_time.size == 0:
        return empty

    if time_zero == "first_sample":
        time_sec = raw_time - raw_time[0]
    elif time_zero == "session_start":
        if session_start_sec is None:
            raise ValueError("session_start_sec is required when time_zero='session_start'")
        time_sec = raw_time - float(session_start_sec)
    elif time_zero == "none":
        time_sec = raw_time.copy()
    else:
        raise ValueError("time_zero must be 'first_sample', 'session_start', or 'none'")

    time_sec = time_sec + float(time_offset_sec)

    inferred_period = _infer_counter_period(
        position_raw,
        np.dtype(raw_dtype),
        unwrap_position=unwrap_position,
        counter_period=counter_period,
        counter_bits=counter_bits,
    )
    if inferred_period is not None:
        position_unwrapped, unwrap_diag = unwrap_encoder_position(
            position_raw,
            counter_period=inferred_period,
        )
    else:
        position_unwrapped = position_raw.copy()
        unwrap_diag = {
            "counter_period": None,
            "n_wraps": 0,
            "wrap_indices": np.array([], dtype=int),
            "wrap_deltas": np.array([], dtype=float),
        }

    position = position_unwrapped.copy()
    kernel = _odd_kernel(median_filter_kernel)
    if kernel > 1:
        position = medfilt(position, kernel_size=kernel)

    position_velocity_per_s = _safe_velocity(position, time_sec)

    if encoder_units == "ticks":
        if ticks_per_revolution is None:
            raise ValueError("ticks_per_revolution is required when encoder_units='ticks'")
        angle_rad = position * (2.0 * np.pi / float(ticks_per_revolution))
    elif encoder_units == "degrees":
        angle_rad = np.deg2rad(position)
    elif encoder_units == "radians":
        angle_rad = position
    elif encoder_units == "cm":
        angle_rad = position / float(wheel_radius_cm)
    elif encoder_units == "m":
        angle_rad = (position * 100.0) / float(wheel_radius_cm)
    else:
        raise ValueError("encoder_units must be 'ticks', 'degrees', 'radians', 'cm', or 'm'")

    angular_velocity_rad_s = _safe_velocity(angle_rad, time_sec)
    linear_velocity_cm_s = angular_velocity_rad_s * float(wheel_radius_cm)

    if max_abs_linear_velocity_cm_s is not None:
        threshold = float(max_abs_linear_velocity_cm_s)
        if threshold <= 0 or not np.isfinite(threshold):
            raise ValueError("max_abs_linear_velocity_cm_s must be positive and finite")
        bad = np.abs(linear_velocity_cm_s) > threshold
        linear_velocity_cm_s = linear_velocity_cm_s.copy()
        angular_velocity_rad_s = angular_velocity_rad_s.copy()
        linear_velocity_cm_s[bad] = np.nan
        angular_velocity_rad_s[bad] = np.nan

    if smooth_sigma_samples is not None and smooth_sigma_samples > 0:
        # Fill NaNs as zero before smoothing. In normal use after unwrapping there
        # should be no NaNs; this mostly supports the optional artifact guard.
        angular_velocity_rad_s = gaussian_filter1d(
            np.nan_to_num(angular_velocity_rad_s, nan=0.0, posinf=0.0, neginf=0.0),
            sigma=float(smooth_sigma_samples),
            mode="nearest",
        )
        linear_velocity_cm_s = gaussian_filter1d(
            np.nan_to_num(linear_velocity_cm_s, nan=0.0, posinf=0.0, neginf=0.0),
            sigma=float(smooth_sigma_samples),
            mode="nearest",
        )
    else:
        angular_velocity_rad_s = np.nan_to_num(
            angular_velocity_rad_s, nan=0.0, posinf=0.0, neginf=0.0
        )
        linear_velocity_cm_s = np.nan_to_num(
            linear_velocity_cm_s, nan=0.0, posinf=0.0, neginf=0.0
        )

    if absolute_velocity:
        angular_velocity_rad_s = np.abs(angular_velocity_rad_s)
        linear_velocity_cm_s = np.abs(linear_velocity_cm_s)

    return {
        "time_sec": time_sec,
        "position_raw": position_raw,
        "position_unwrapped": position_unwrapped,
        "position_filtered": position,
        "angle_rad": angle_rad,
        "position_velocity_per_s": position_velocity_per_s,
        "position_speed_per_s": np.abs(position_velocity_per_s),
        "angular_velocity_rad_s": angular_velocity_rad_s,
        "linear_velocity_cm_s": linear_velocity_cm_s,
        "linear_speed_cm_s": np.abs(linear_velocity_cm_s),
        "path": str(encoder_path),
        "counter_period": unwrap_diag["counter_period"],
        "n_wraps": unwrap_diag["n_wraps"],
        "wrap_indices": unwrap_diag["wrap_indices"],
        "wrap_deltas": unwrap_diag["wrap_deltas"],
    }
