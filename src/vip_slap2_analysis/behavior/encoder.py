from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


EncoderUnits = Literal["ticks", "degrees", "radians", "cm", "m"]
TimeZero = Literal["first_sample", "session_start", "none"]


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
) -> dict[str, np.ndarray | str]:
    """
    Load encoder.pkl and compute angular and linear velocity.

    Assumes encoder.pkl is a pandas DataFrame whose index is time in seconds and
    whose encoder_col gives cumulative encoder position.

    Returns
    -------
    dict with:
        time_sec
        position_raw
        position_filtered
        angle_rad
        angular_velocity_rad_s
        linear_velocity_cm_s
        linear_speed_cm_s
        path
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
    position_raw = df[encoder_col].to_numpy(dtype=float)

    if raw_time.size == 0:
        return {
            "time_sec": np.array([]),
            "position_raw": np.array([]),
            "position_filtered": np.array([]),
            "angle_rad": np.array([]),
            "angular_velocity_rad_s": np.array([]),
            "linear_velocity_cm_s": np.array([]),
            "linear_speed_cm_s": np.array([]),
            "path": str(encoder_path),
        }

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

    position = position_raw.copy()
    kernel = int(median_filter_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        position = medfilt(position, kernel_size=kernel)

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

    dt = np.gradient(time_sec)
    dt[~np.isfinite(dt)] = np.nan
    dt[dt <= 0] = np.nan

    angular_velocity_rad_s = np.gradient(angle_rad) / dt
    linear_velocity_cm_s = angular_velocity_rad_s * float(wheel_radius_cm)

    angular_velocity_rad_s = np.nan_to_num(
        angular_velocity_rad_s, nan=0.0, posinf=0.0, neginf=0.0
    )
    linear_velocity_cm_s = np.nan_to_num(
        linear_velocity_cm_s, nan=0.0, posinf=0.0, neginf=0.0
    )

    if smooth_sigma_samples is not None and smooth_sigma_samples > 0:
        angular_velocity_rad_s = gaussian_filter1d(
            angular_velocity_rad_s,
            sigma=float(smooth_sigma_samples),
            mode="nearest",
        )
        linear_velocity_cm_s = gaussian_filter1d(
            linear_velocity_cm_s,
            sigma=float(smooth_sigma_samples),
            mode="nearest",
        )

    if absolute_velocity:
        angular_velocity_rad_s = np.abs(angular_velocity_rad_s)
        linear_velocity_cm_s = np.abs(linear_velocity_cm_s)

    return {
        "time_sec": time_sec,
        "position_raw": position_raw,
        "position_filtered": position,
        "angle_rad": angle_rad,
        "angular_velocity_rad_s": angular_velocity_rad_s,
        "linear_velocity_cm_s": linear_velocity_cm_s,
        "linear_speed_cm_s": np.abs(linear_velocity_cm_s),
        "path": str(encoder_path),
    }