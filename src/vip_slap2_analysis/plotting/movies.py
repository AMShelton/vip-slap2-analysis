from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import imageio.v3 as iio
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, filtfilt


ArrayLike = np.ndarray
PathLike = Union[str, Path]


@dataclass
class MovieRenderConfig:
    """
    Configuration for rendering SLAP2 glutamate dF movies.

    Parameters
    ----------
    input_frame_rate_hz
        Native sampling rate of the input movie in Hz.
    output_frame_rate_hz
        Frame rate of the output video.
    downsample_factor_time
        Number of factor-2 temporal downsampling steps. A value of 2 means
        the movie is averaged in time twice, so effective time compression is 4x.
    baseline_window_s
        Approximate temporal window for slow baseline estimation.
    channel_index
        Which channel to render if movie has shape (Y, X, C, T).
    padding_px
        Padding around ROI-defined crop.
    activity_percentile
        Percentile used to normalize the red activity channel.
    structure_percentile
        Percentile used to normalize the cyan structure channel.
    median_filter_xy
        Spatial median filtering kernel for activity map.
    gamma
        Gamma correction applied to final RGB frames.
    quality
        Video quality for ffmpeg-backed writers if supported.
    codec
        Video codec. "libx264" is a good default for mp4.
    """
    input_frame_rate_hz: float
    output_frame_rate_hz: float = 10.0
    downsample_factor_time: int = 2
    baseline_window_s: float = 0.25
    channel_index: int = 0
    padding_px: int = 20
    activity_percentile: float = 99.97
    structure_percentile: float = 99.99
    median_filter_xy: int = 3
    gamma: float = 1.0
    quality: int = 8
    codec: str = "libx264"


def load_slap2_movie_from_tiffs(
    dmd1_tiff: PathLike,
    dmd2_tiff: Optional[PathLike] = None,
    *,
    n_channels: int = 2,
    combine_dmds: bool = True,
) -> np.ndarray:
    """
    Load SLAP2 TIFF movie(s).

    Returns
    -------
    movie : np.ndarray
        Shape:
        - (Y, X, C, T) if TIFF(s) contain interleaved channels
        - combined vertically across DMDs if combine_dmds=True and dmd2_tiff is provided
    """
    dmd1 = np.asarray(tifffile.imread(dmd1_tiff), dtype=np.float32)
    if dmd1.ndim != 3:
        raise ValueError(f"Expected DMD1 TIFF to be 3D (Y, X, frames), got {dmd1.shape}")

    def _reshape_channels(arr: np.ndarray, n_ch: int) -> np.ndarray:
        y, x, n_frames = arr.shape
        usable = (n_frames // n_ch) * n_ch
        if usable == 0:
            raise ValueError("Movie has too few frames to reshape into channels.")
        if usable != n_frames:
            arr = arr[:, :, :usable]
        return arr.reshape(y, x, n_ch, usable // n_ch)

    dmd1 = _reshape_channels(dmd1, n_channels)

    if dmd2_tiff is None:
        return dmd1

    dmd2 = np.asarray(tifffile.imread(dmd2_tiff), dtype=np.float32)
    if dmd2.ndim != 3:
        raise ValueError(f"Expected DMD2 TIFF to be 3D (Y, X, frames), got {dmd2.shape}")
    dmd2 = _reshape_channels(dmd2, n_channels)

    # Match X and T dimensions.
    y1, x1, c1, t1 = dmd1.shape
    y2, x2, c2, t2 = dmd2.shape
    if c1 != c2:
        raise ValueError(f"Channel mismatch between DMD1 ({c1}) and DMD2 ({c2})")

    x_max = max(x1, x2)
    t_min = min(t1, t2)

    def _pad_x(arr: np.ndarray, x_target: int) -> np.ndarray:
        y, x, c, t = arr.shape
        if x == x_target:
            return arr[:, :, :, :t_min]
        out = np.full((y, x_target, c, t_min), np.nan, dtype=arr.dtype)
        out[:, :x, :, :] = arr[:, :, :, :t_min]
        return out

    dmd1 = _pad_x(dmd1, x_max)
    dmd2 = _pad_x(dmd2, x_max)

    if combine_dmds:
        return np.concatenate([dmd1, dmd2], axis=0)

    return np.stack([dmd1, dmd2], axis=0)


def _downsample_time_by_2(movie: np.ndarray) -> np.ndarray:
    """
    Temporal average downsample by factor 2.

    Expects shape (Y, X, T).
    """
    if movie.ndim != 3:
        raise ValueError(f"Expected (Y, X, T), got {movie.shape}")

    t = movie.shape[-1]
    usable = (t // 2) * 2
    if usable == 0:
        raise ValueError("Movie too short to downsample by factor 2.")
    movie = movie[:, :, :usable]
    return 0.5 * (movie[:, :, 0::2] + movie[:, :, 1::2])


def temporal_downsample(movie: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Repeated factor-2 temporal downsampling.

    Expects shape (Y, X, T).
    """
    out = movie
    for _ in range(n_steps):
        out = _downsample_time_by_2(out)
    return out


def _fill_nans_with_mean_image(movie: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replace NaNs in a movie with mean image values for baseline estimation.

    Returns
    -------
    movie_filled, mean_image, setnan
    """
    mean_image = np.nanmean(movie, axis=2)
    mean_image = np.where(np.isnan(mean_image), 0.0, mean_image)

    nan_fraction = np.mean(np.isnan(movie), axis=2)
    setnan = nan_fraction > 0.7
    mean_image[setnan] = 0.0

    movie_filled = movie.copy()
    nan_mask = np.isnan(movie_filled)
    if np.any(nan_mask):
        mean_stack = np.repeat(mean_image[:, :, None], movie.shape[2], axis=2)
        movie_filled[nan_mask] = mean_stack[nan_mask]

    return movie_filled, mean_image, setnan


def _compute_f0_movie(
    movie: np.ndarray,
    *,
    baseline_window_frames: int,
    butter_order: int = 4,
) -> np.ndarray:
    """
    Approximate Kaspar's MATLAB F0 estimation:
    - light temporal smoothing
    - temporal median filter
    - repeated low-pass filtfilt on min(smoothed, F0)

    Expects NaN-free movie of shape (Y, X, T).
    """
    if movie.ndim != 3:
        raise ValueError(f"Expected (Y, X, T), got {movie.shape}")

    y, x, t = movie.shape
    if t < 5:
        raise ValueError("Movie too short for baseline estimation.")

    # mild temporal smoothing
    e1 = gaussian_filter1d(movie, sigma=1.0, axis=2, mode="nearest")

    # median over time
    med_win = max(3, min(25, t if t % 2 == 1 else t - 1))
    f0 = median_filter(e1, size=(1, 1, med_win), mode="nearest")

    cutoff = min(0.99, max(1e-4, baseline_window_frames / max(2, t)))
    b, a = butter(butter_order, cutoff, btype="low")

    for _ in range(3):
        f0 = filtfilt(
            b,
            a,
            np.minimum(e1, f0),
            axis=2,
            padtype="odd",
            padlen=min(3 * max(len(a), len(b)), t - 1),
        )

    lead = int(np.ceil(baseline_window_frames))
    if lead > 0 and lead + 1 < t:
        f0[:, :, :lead] = f0[:, :, lead][:, :, None]

    return f0


def _normalize_percentile(arr: np.ndarray, q: float, eps: float = 1e-8) -> np.ndarray:
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    scale = np.nanpercentile(arr[finite], q)
    scale = max(scale, eps)
    out = arr / scale
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _compute_structure_and_activity(
    movie: np.ndarray,
    *,
    baseline_window_frames: int,
    activity_percentile: float,
    structure_percentile: float,
    median_filter_xy: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute structure (cyan) and activity (red) channels.

    Returns
    -------
    structure : np.ndarray
        Shape (Y, X, T), normalized 0..1
    activity : np.ndarray
        Shape (Y, X, T), normalized 0..1
    """
    movie_filled, mean_image, setnan = _fill_nans_with_mean_image(movie)
    f0 = _compute_f0_movie(movie_filled, baseline_window_frames=baseline_window_frames)
    df = movie_filled - f0

    soma_im = np.maximum(50.0, mean_image + 50.0)

    # activity map
    if median_filter_xy > 1:
        act = median_filter(df, size=(median_filter_xy, median_filter_xy, 1), mode="nearest")
    else:
        act = df.copy()

    finite = np.isfinite(act)
    baseline = np.nanmedian(act[finite]) if np.any(finite) else 0.0
    act = np.maximum(0.0, act - baseline)
    act[np.repeat(setnan[:, :, None], movie.shape[2], axis=2)] = 0.0
    act = act / np.maximum(soma_im[:, :, None], 1e-8)
    act = _normalize_percentile(act, activity_percentile)

    # structure map
    st = np.sqrt(np.maximum(0.0, f0 / np.maximum(soma_im[:, :, None], 1e-8)))
    finite_st = np.isfinite(st)
    if np.any(finite_st):
        st = st - np.nanpercentile(st[finite_st], 10.0)
    st = np.maximum(st, 0.0)
    st = 0.75 * _normalize_percentile(st, structure_percentile)
    st[np.repeat(setnan[:, :, None], movie.shape[2], axis=2)] = 0.0
    st = np.clip(st, 0.0, 1.0)

    return st.astype(np.float32), act.astype(np.float32)


def _compose_rgb_frames(
    structure: np.ndarray,
    activity: np.ndarray,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Compose RGB frames:
    red = activity
    green/blue = cyan structure attenuated by activity
    """
    red = np.clip(activity, 0.0, 1.0)
    cyan = np.clip(structure, 0.0, 1.0)

    rgb = np.stack(
        [
            red,
            cyan * np.sqrt(1.0 - red),
            cyan * np.sqrt(1.0 - red),
        ],
        axis=-1,
    )

    rgb = np.clip(rgb, 0.0, 1.0)
    if gamma != 1.0:
        rgb = np.power(rgb, 1.0 / gamma)

    return (255.0 * rgb).astype(np.uint8)


def crop_movie_to_bbox(movie: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop movie to (y0, y1, x0, x1).
    Supports (Y, X, T) or (Y, X, C, T).
    """
    y0, y1, x0, x1 = bbox
    if movie.ndim == 3:
        return movie[y0:y1, x0:x1, :]
    if movie.ndim == 4:
        return movie[y0:y1, x0:x1, :, :]
    raise ValueError(f"Unsupported movie shape {movie.shape}")


def bbox_from_mask(mask: np.ndarray, padding_px: int = 0) -> Tuple[int, int, int, int]:
    """
    Bounding box from a 2D boolean mask.
    """
    yy, xx = np.where(mask > 0)
    if len(yy) == 0:
        raise ValueError("Mask is empty; cannot compute bounding box.")

    y0 = max(0, int(yy.min()) - padding_px)
    y1 = int(yy.max()) + 1 + padding_px
    x0 = max(0, int(xx.min()) - padding_px)
    x1 = int(xx.max()) + 1 + padding_px
    return y0, y1, x0, x1


def bbox_from_masks_union(
    roi_masks: Sequence[np.ndarray],
    image_shape: Tuple[int, int],
    padding_px: int = 0,
) -> Tuple[int, int, int, int]:
    """
    Bounding box covering the union of provided ROI masks.
    """
    if len(roi_masks) == 0:
        raise ValueError("roi_masks is empty.")

    union = np.zeros(image_shape, dtype=bool)
    for m in roi_masks:
        if m.shape != image_shape:
            raise ValueError(f"ROI mask shape {m.shape} does not match image shape {image_shape}")
        union |= m.astype(bool)

    y0, y1, x0, x1 = bbox_from_mask(union, padding_px=padding_px)
    y1 = min(y1, image_shape[0])
    x1 = min(x1, image_shape[1])
    return y0, y1, x0, x1


def select_movie_time_window(
    movie: np.ndarray,
    *,
    frame_rate_hz: float,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> np.ndarray:
    """
    Subselect movie in time. Works on (Y, X, T) and (Y, X, C, T).
    """
    if movie.ndim == 3:
        t_dim = 2
        n_frames = movie.shape[2]
    elif movie.ndim == 4:
        t_dim = 3
        n_frames = movie.shape[3]
    else:
        raise ValueError(f"Unsupported movie shape {movie.shape}")

    if start_frame is None and start_s is not None:
        start_frame = int(np.floor(start_s * frame_rate_hz))
    if end_frame is None and end_s is not None:
        end_frame = int(np.ceil(end_s * frame_rate_hz))

    start_frame = 0 if start_frame is None else max(0, start_frame)
    end_frame = n_frames if end_frame is None else min(n_frames, end_frame)

    slicer = [slice(None)] * movie.ndim
    slicer[t_dim] = slice(start_frame, end_frame)
    return movie[tuple(slicer)]


def render_glutamate_df_movie(
    output_path: PathLike,
    *,
    config: MovieRenderConfig,
    movie: Optional[np.ndarray] = None,
    dmd1_tiff: Optional[PathLike] = None,
    dmd2_tiff: Optional[PathLike] = None,
    n_channels: int = 2,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    roi_mask: Optional[np.ndarray] = None,
    roi_masks: Optional[Sequence[np.ndarray]] = None,
    crop_bbox: Optional[Tuple[int, int, int, int]] = None,
    overwrite: bool = False,
    draw_timestamp: bool = False,
) -> Path:
    """
    Render a cyan/red glutamate movie in the style of renderDfMovieSLAP2.m.

    Parameters
    ----------
    output_path
        Output video path, usually .mp4
    config
        Rendering configuration.
    movie
        Optional in-memory movie. Expected shape:
        - (Y, X, T), or
        - (Y, X, C, T)
    dmd1_tiff, dmd2_tiff
        Optional TIFF inputs if movie is not provided.
    n_channels
        Number of interleaved channels in TIFF input.
    start_s, end_s
        Time window to render, in seconds.
    start_frame, end_frame
        Frame window to render, in native input frames.
    roi_mask
        Single boolean mask for crop selection.
    roi_masks
        Multiple ROI masks; crop uses union of all masks.
    crop_bbox
        Explicit crop box (y0, y1, x0, x1).
    overwrite
        Whether to overwrite existing output.
    draw_timestamp
        Placeholder switch; currently timestamp text is not burned in.

    Returns
    -------
    Path
        Output movie path.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Set overwrite=True to replace it.")

    if movie is None:
        if dmd1_tiff is None:
            raise ValueError("Provide either `movie` or `dmd1_tiff`.")
        movie = load_slap2_movie_from_tiffs(
            dmd1_tiff=dmd1_tiff,
            dmd2_tiff=dmd2_tiff,
            n_channels=n_channels,
            combine_dmds=True,
        )

    movie = np.asarray(movie, dtype=np.float32)

    # Standardize to (Y, X, T)
    if movie.ndim == 4:
        if config.channel_index >= movie.shape[2]:
            raise ValueError(
                f"channel_index={config.channel_index} out of bounds for movie shape {movie.shape}"
            )
        movie = movie[:, :, config.channel_index, :]
    elif movie.ndim != 3:
        raise ValueError(f"Expected movie shape (Y, X, T) or (Y, X, C, T), got {movie.shape}")

    # Time window in native frames
    movie = select_movie_time_window(
        movie,
        frame_rate_hz=config.input_frame_rate_hz,
        start_s=start_s,
        end_s=end_s,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    # Crop
    if crop_bbox is not None:
        movie = crop_movie_to_bbox(movie, crop_bbox)
    elif roi_mask is not None:
        bbox = bbox_from_mask(roi_mask, padding_px=config.padding_px)
        movie = crop_movie_to_bbox(movie, bbox)
    elif roi_masks is not None:
        bbox = bbox_from_masks_union(
            roi_masks=roi_masks,
            image_shape=movie.shape[:2],
            padding_px=config.padding_px,
        )
        movie = crop_movie_to_bbox(movie, bbox)

    # Temporal downsample
    movie_ds = temporal_downsample(movie, config.downsample_factor_time)
    eff_frame_rate = config.input_frame_rate_hz / (2 ** config.downsample_factor_time)

    baseline_window_frames = max(3, int(np.ceil(config.baseline_window_s * eff_frame_rate)))

    structure, activity = _compute_structure_and_activity(
        movie_ds,
        baseline_window_frames=baseline_window_frames,
        activity_percentile=config.activity_percentile,
        structure_percentile=config.structure_percentile,
        median_filter_xy=config.median_filter_xy,
    )

    rgb = _compose_rgb_frames(structure, activity, gamma=config.gamma)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # imageio expects frames as iterable of (Y, X, 3)
    iio.imwrite(
        output_path,
        rgb.transpose(2, 0, 1, 3),  # -> (T, Y, X, 3)
        fps=config.output_frame_rate_hz,
        codec=config.codec,
    )

    return output_path