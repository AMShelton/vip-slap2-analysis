from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import imageio.v3 as iio
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.signal import butter, filtfilt

PathLike = Union[str, Path]
ChannelSpec = Union[int, Literal["auto"]]


@dataclass
class MovieRenderConfig:
    """
    Configuration for rendering SLAP2 glutamate dF movies.
    """
    input_frame_rate_hz: float
    output_frame_rate_hz: float = 10.0
    downsample_factor_time: int = 0
    baseline_window_s: float = 0.25
    baseline_mode: str = "fast"  # {"fast", "full"}
    channel_index: int = 0
    padding_px: int = 20
    activity_percentile: float = 99.5
    structure_percentile: float = 99.5
    median_filter_xy: int = 3
    gamma: float = 1.2
    codec: str = "libx264"

    # Orientation transforms in display space
    transpose_xy: bool = False
    flip_ud: bool = False
    flip_lr: bool = False

    # Overlays
    show_timer: bool = True
    show_scale_bar: bool = True
    pixel_size_um: float = 0.25
    scale_bar_um: float = 20.0
    overlay_margin_px: int = 50
    overlay_fontsize: int = 120
    overlay_linewidth: int = 20
    overlay_text_color: Tuple[int, int, int] = (255, 255, 255)
    overlay_bar_color: Tuple[int, int, int] = (255, 255, 255)

    # High-quality rendering options
    activity_gaussian_sigma_px: float = 0.8
    structure_gaussian_sigma_px: float = 0.6
    activity_temporal_sigma_frames: float = 0.75
    upsample_factor: float = 1.0
    upsample_mode: str = "bicubic"  # {"nearest", "bilinear", "bicubic"}
    video_crf: int = 12
    video_preset: str = "slow"


def _vprint(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg)
        
def _resolve_overlay_fontsize(config: MovieRenderConfig, frame_h: int) -> int:
    return max(config.overlay_fontsize, int(round(0.045 * frame_h)))

def _resolve_overlay_linewidth(config: MovieRenderConfig, frame_h: int, frame_w: int) -> int:
    return max(config.overlay_linewidth, int(round(0.006 * min(frame_h, frame_w))))

def _resolve_overlay_margin(config: MovieRenderConfig, frame_h: int, frame_w: int) -> int:
    return max(config.overlay_margin_px, int(round(0.03 * min(frame_h, frame_w))))

def _progress(verbose: bool, step: int, total: int, msg: str) -> None:
    if not verbose:
        return
    bar_len = 24
    filled = int(round(bar_len * step / total))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"[{bar}] {step}/{total}  {msg}")


def inspect_tiff_layout(path: PathLike) -> None:
    path = str(path)
    with tifffile.TiffFile(path) as tf:
        print(f"path: {path}")
        print("series count:", len(tf.series))
        for i, s in enumerate(tf.series):
            print(f"series {i}: shape={s.shape}, axes={getattr(s, 'axes', None)}")
        print("pages:", len(tf.pages))

    arr = tifffile.memmap(path)
    print("memmap shape:", arr.shape, "dtype:", arr.dtype)


def _read_tiff_memmap(path: PathLike, *, verbose: bool = False) -> np.ndarray:
    path = str(path)
    _vprint(verbose, f"Opening TIFF: {path}")
    try:
        arr = tifffile.memmap(path)
        _vprint(verbose, f"Opened as memmap with shape={arr.shape}, dtype={arr.dtype}")
        return arr
    except Exception as exc:
        _vprint(verbose, f"memmap failed ({exc}); falling back to tifffile.imread(...)")
        arr = tifffile.imread(path)
        _vprint(verbose, f"Loaded fully into memory with shape={arr.shape}, dtype={arr.dtype}")
        return arr


def infer_page_stack_channels(
    arr: np.ndarray,
    *,
    requested_n_channels: ChannelSpec = "auto",
    verbose: bool = False,
) -> int:
    """
    Infer channel count for page-stacked TIFFs.

    If explicitly provided as int, use that.
    If "auto":
      - prefer 2 when page count is divisible by 2
      - otherwise use 1
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected page stack with shape (pages, Y, X), got {arr.shape}")

    if isinstance(requested_n_channels, int):
        if requested_n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        return requested_n_channels

    n_pages = arr.shape[0]
    if n_pages % 2 == 0:
        if verbose:
            print(f"Auto-detected n_channels=2 from page count {n_pages}")
        return 2

    if verbose:
        print(f"Auto-detected n_channels=1 from page count {n_pages}")
    return 1


def apply_orientation_transform(
    arr: np.ndarray,
    *,
    transpose_xy: bool = False,
    flip_ud: bool = False,
    flip_lr: bool = False,
) -> np.ndarray:
    """
    Apply orientation transforms in display space.

    Supported shapes:
    - (Y, X)
    - (Y, X, T)
    - (Y, X, C, T)
    """
    if arr.ndim == 2:
        if transpose_xy:
            arr = arr.T
        if flip_ud:
            arr = np.flip(arr, axis=0)
        if flip_lr:
            arr = np.flip(arr, axis=1)
        return arr

    if arr.ndim == 3:
        if transpose_xy:
            arr = np.transpose(arr, (1, 0, 2))
        if flip_ud:
            arr = np.flip(arr, axis=0)
        if flip_lr:
            arr = np.flip(arr, axis=1)
        return arr

    if arr.ndim == 4:
        if transpose_xy:
            arr = np.transpose(arr, (1, 0, 2, 3))
        if flip_ud:
            arr = np.flip(arr, axis=0)
        if flip_lr:
            arr = np.flip(arr, axis=1)
        return arr

    raise ValueError(f"Unsupported array shape {arr.shape}")


def _time_to_page_range(
    start_frame: int,
    end_frame: int,
    n_channels: int,
) -> tuple[int, int]:
    return start_frame * n_channels, end_frame * n_channels


def _resolve_frame_window(
    *,
    frame_rate_hz: float,
    start_s: Optional[float],
    end_s: Optional[float],
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> tuple[Optional[int], Optional[int]]:
    if start_frame is None and start_s is not None:
        start_frame = int(np.floor(start_s * frame_rate_hz))
    if end_frame is None and end_s is not None:
        end_frame = int(np.ceil(end_s * frame_rate_hz))

    if start_frame is not None:
        start_frame = max(0, start_frame)
    if end_frame is not None:
        end_frame = max(0, end_frame)

    return start_frame, end_frame


def _load_page_stack_subset(
    path: PathLike,
    *,
    n_channels: ChannelSpec = "auto",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    crop_bbox: Optional[tuple[int, int, int, int]] = None,
    dtype=np.float32,
    verbose: bool = False,
) -> np.ndarray:
    """
    Load subset of a page-stacked TIFF and return (Y, X, C, T).

    Expected TIFF layout:
    - 1-channel: (pages, Y, X) with pages = time
    - 2-channel: page order t0-ch0, t0-ch1, t1-ch0, t1-ch1, ...
    """
    arr = _read_tiff_memmap(path, verbose=verbose)

    if arr.ndim != 3:
        raise ValueError(f"Expected page stack with shape (pages, Y, X), got {arr.shape}")

    n_pages, _, _ = arr.shape
    resolved_n_channels = infer_page_stack_channels(
        arr,
        requested_n_channels=n_channels,
        verbose=verbose,
    )

    if n_pages % resolved_n_channels != 0:
        raise ValueError(
            f"Page stack has {n_pages} pages, which is not divisible by "
            f"n_channels={resolved_n_channels}"
        )

    n_timepoints = n_pages // resolved_n_channels
    start_frame = 0 if start_frame is None else max(0, start_frame)
    end_frame = n_timepoints if end_frame is None else min(n_timepoints, end_frame)

    if end_frame <= start_frame:
        raise ValueError(
            f"Invalid frame window [{start_frame}, {end_frame}) for movie with {n_timepoints} frames."
        )

    start_page, end_page = _time_to_page_range(start_frame, end_frame, resolved_n_channels)

    _vprint(
        verbose,
        f"Loading frames [{start_frame}:{end_frame}] -> pages [{start_page}:{end_page}] "
        f"from total {n_timepoints} frames with {resolved_n_channels} channel(s)"
    )

    if crop_bbox is None:
        subset = np.asarray(arr[start_page:end_page], dtype=dtype)
    else:
        y0, y1, x0, x1 = crop_bbox
        subset = np.asarray(arr[start_page:end_page, y0:y1, x0:x1], dtype=dtype)

    n_subset_pages, y_sub, x_sub = subset.shape
    t = n_subset_pages // resolved_n_channels
    subset = subset.reshape(t, resolved_n_channels, y_sub, x_sub)   # (T, C, Y, X)
    subset = np.transpose(subset, (2, 3, 1, 0))                     # (Y, X, C, T)

    _vprint(verbose, f"Loaded subset shape: {subset.shape}")
    return subset


def load_slap2_movie_from_tiffs(
    dmd1_tiff: Optional[PathLike] = None,
    dmd2_tiff: Optional[PathLike] = None,
    *,
    n_channels: ChannelSpec = "auto",
    combine_dmds: bool = True,
    dtype: np.dtype = np.float32,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    crop_bbox: Optional[tuple[int, int, int, int]] = None,
    dmd_selection: str = "both",
    verbose: bool = False,
) -> np.ndarray:
    """
    Load SLAP2 TIFF movie(s) into (Y, X, C, T).

    crop_bbox here is interpreted in raw TIFF page coordinates.
    """
    dmd_selection = dmd_selection.lower()
    if dmd_selection not in {"dmd1", "dmd2", "both"}:
        raise ValueError("dmd_selection must be one of {'dmd1', 'dmd2', 'both'}")

    if dmd_selection in {"dmd1", "both"} and dmd1_tiff is None:
        raise ValueError("dmd1_tiff is required for dmd_selection='dmd1' or 'both'")
    if dmd_selection in {"dmd2", "both"} and dmd2_tiff is None:
        raise ValueError("dmd2_tiff is required for dmd_selection='dmd2' or 'both'")

    if dmd_selection == "dmd1":
        return _load_page_stack_subset(
            dmd1_tiff,
            n_channels=n_channels,
            start_frame=start_frame,
            end_frame=end_frame,
            crop_bbox=crop_bbox,
            dtype=dtype,
            verbose=verbose,
        )

    if dmd_selection == "dmd2":
        return _load_page_stack_subset(
            dmd2_tiff,
            n_channels=n_channels,
            start_frame=start_frame,
            end_frame=end_frame,
            crop_bbox=crop_bbox,
            dtype=dtype,
            verbose=verbose,
        )

    _vprint(verbose, "Loading DMD1 subset...")
    dmd1 = _load_page_stack_subset(
        dmd1_tiff,
        n_channels=n_channels,
        start_frame=start_frame,
        end_frame=end_frame,
        crop_bbox=crop_bbox,
        dtype=dtype,
        verbose=verbose,
    )

    _vprint(verbose, "Loading DMD2 subset...")
    dmd2 = _load_page_stack_subset(
        dmd2_tiff,
        n_channels=n_channels,
        start_frame=start_frame,
        end_frame=end_frame,
        crop_bbox=crop_bbox,
        dtype=dtype,
        verbose=verbose,
    )

    if not combine_dmds:
        return np.stack([dmd1, dmd2], axis=0)

    y1, x1, c1, t1 = dmd1.shape
    y2, x2, c2, t2 = dmd2.shape
    if c1 != c2:
        raise ValueError(f"Channel mismatch: DMD1 has {c1}, DMD2 has {c2}")

    x_max = max(x1, x2)
    t_min = min(t1, t2)

    def _pad_x(arr: np.ndarray, x_target: int, t_target: int) -> np.ndarray:
        y, x, c, t = arr.shape
        out = np.full((y, x_target, c, t_target), np.nan, dtype=arr.dtype)
        out[:, :x, :, :] = arr[:, :, :, :t_target]
        return out

    dmd1 = _pad_x(dmd1, x_max, t_min)
    dmd2 = _pad_x(dmd2, x_max, t_min)
    combined = np.concatenate([dmd1, dmd2], axis=0)

    _vprint(verbose, f"Returning combined movie with shape: {combined.shape}")
    return combined


def _downsample_time_by_2(movie: np.ndarray) -> np.ndarray:
    if movie.ndim != 3:
        raise ValueError(f"Expected (Y, X, T), got {movie.shape}")

    t = movie.shape[-1]
    usable = (t // 2) * 2
    if usable == 0:
        raise ValueError("Movie too short to downsample by factor 2.")

    movie = movie[:, :, :usable]
    return 0.5 * (movie[:, :, 0::2] + movie[:, :, 1::2])


def temporal_downsample(movie: np.ndarray, n_steps: int) -> np.ndarray:
    out = movie
    for _ in range(n_steps):
        out = _downsample_time_by_2(out)
    return out


def _fill_nans_with_mean_image(movie: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_counts = np.sum(~np.isnan(movie), axis=2)
    sum_image = np.nansum(movie, axis=2)
    mean_image = np.divide(
        sum_image,
        valid_counts,
        out=np.zeros_like(sum_image, dtype=np.float32),
        where=valid_counts > 0,
    )

    nan_fraction = np.mean(np.isnan(movie), axis=2)
    setnan = nan_fraction > 0.7
    mean_image[setnan] = 0.0

    movie_filled = movie.copy()
    nan_mask = np.isnan(movie_filled)
    if np.any(nan_mask):
        mean_stack = np.repeat(mean_image[:, :, None], movie.shape[2], axis=2)
        movie_filled[nan_mask] = mean_stack[nan_mask]

    return movie_filled, mean_image, setnan


def _compute_f0_movie_full(
    movie: np.ndarray,
    *,
    baseline_window_frames: int,
    butter_order: int = 4,
) -> np.ndarray:
    if movie.ndim != 3:
        raise ValueError(f"Expected (Y, X, T), got {movie.shape}")

    _, _, t = movie.shape
    if t < 5:
        raise ValueError("Movie too short for baseline estimation.")

    e1 = gaussian_filter1d(movie, sigma=1.0, axis=2, mode="nearest")
    med_win = max(3, min(25, t if t % 2 == 1 else t - 1))
    f0 = median_filter(e1, size=(1, 1, med_win), mode="nearest")

    cutoff = min(0.99, max(1e-4, baseline_window_frames / max(2, t)))
    b, a = butter(butter_order, cutoff, btype="low")
    padlen = min(3 * max(len(a), len(b)), t - 1)

    for _ in range(3):
        f0 = filtfilt(
            b,
            a,
            np.minimum(e1, f0),
            axis=2,
            padtype="odd",
            padlen=padlen,
        )

    lead = int(np.ceil(baseline_window_frames))
    if lead > 0 and lead + 1 < t:
        f0[:, :, :lead] = f0[:, :, lead][:, :, None]

    return f0


def _compute_f0_movie_fast(
    movie: np.ndarray,
    *,
    baseline_window_frames: int,
) -> np.ndarray:
    if movie.ndim != 3:
        raise ValueError(f"Expected (Y, X, T), got {movie.shape}")

    e1 = gaussian_filter1d(movie, sigma=1.0, axis=2, mode="nearest")
    med_win = baseline_window_frames
    if med_win % 2 == 0:
        med_win += 1

    max_valid = movie.shape[2] if movie.shape[2] % 2 == 1 else movie.shape[2] - 1
    med_win = max(3, min(med_win, max_valid))
    return median_filter(e1, size=(1, 1, med_win), mode="nearest")


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
    baseline_mode: str,
    activity_percentile: float,
    structure_percentile: float,
    median_filter_xy: int,
    activity_gaussian_sigma_px: float,
    structure_gaussian_sigma_px: float,
    activity_temporal_sigma_frames: float,
) -> Tuple[np.ndarray, np.ndarray]:
    movie_filled, mean_image, setnan = _fill_nans_with_mean_image(movie)

    baseline_mode = baseline_mode.lower()
    if baseline_mode == "fast":
        f0 = _compute_f0_movie_fast(movie_filled, baseline_window_frames=baseline_window_frames)
    elif baseline_mode == "full":
        f0 = _compute_f0_movie_full(movie_filled, baseline_window_frames=baseline_window_frames)
    else:
        raise ValueError("baseline_mode must be one of {'fast', 'full'}")

    df = movie_filled - f0
    soma_im = np.maximum(50.0, mean_image + 50.0)

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

    st = np.sqrt(np.maximum(0.0, f0 / np.maximum(soma_im[:, :, None], 1e-8)))
    finite_st = np.isfinite(st)
    if np.any(finite_st):
        st = st - np.nanpercentile(st[finite_st], 10.0)
    st = np.maximum(st, 0.0)
    st = 0.75 * _normalize_percentile(st, structure_percentile)
    st[np.repeat(setnan[:, :, None], movie.shape[2], axis=2)] = 0.0
    st = np.clip(st, 0.0, 1.0)

    # High-quality smoothing
    if activity_gaussian_sigma_px > 0:
        act = np.stack(
            [gaussian_filter(act[:, :, i], sigma=activity_gaussian_sigma_px) for i in range(act.shape[2])],
            axis=2,
        )

    if structure_gaussian_sigma_px > 0:
        st = np.stack(
            [gaussian_filter(st[:, :, i], sigma=structure_gaussian_sigma_px) for i in range(st.shape[2])],
            axis=2,
        )

    if activity_temporal_sigma_frames > 0:
        act = gaussian_filter1d(
            act,
            sigma=activity_temporal_sigma_frames,
            axis=2,
            mode="nearest",
        )

    act = np.clip(act, 0.0, 1.0).astype(np.float32)
    st = np.clip(st, 0.0, 1.0).astype(np.float32)

    return st, act


def _compose_rgb_frames(
    structure: np.ndarray,
    activity: np.ndarray,
    *,
    gamma: float = 1.0,
) -> np.ndarray:
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


def _pad_rgb_frames_to_even(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB array with shape (Y, X, T, 3), got {rgb.shape}")

    y, x, t, c = rgb.shape
    pad_y = y % 2
    pad_x = x % 2

    if pad_y == 0 and pad_x == 0:
        return rgb

    out = np.zeros((y + pad_y, x + pad_x, t, c), dtype=rgb.dtype)
    out[:y, :x, :, :] = rgb
    return out


def _get_pil_resample(mode: str):
    mode = mode.lower()
    if mode == "nearest":
        return Image.Resampling.NEAREST
    if mode == "bilinear":
        return Image.Resampling.BILINEAR
    if mode == "bicubic":
        return Image.Resampling.BICUBIC
    raise ValueError("upsample_mode must be one of {'nearest', 'bilinear', 'bicubic'}")


def _upsample_frames(
    frames: np.ndarray,
    factor: float,
    mode: str,
) -> np.ndarray:
    if factor == 1.0:
        return frames
    if factor <= 0:
        raise ValueError("upsample_factor must be > 0")

    resample = _get_pil_resample(mode)
    out = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i])
        new_size = (
            int(round(img.size[0] * factor)),
            int(round(img.size[1] * factor)),
        )
        img = img.resize(new_size, resample=resample)
        out.append(np.asarray(img))
    return np.stack(out, axis=0)


def crop_movie_to_bbox(movie: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    if movie.ndim == 3:
        return movie[y0:y1, x0:x1, :]
    if movie.ndim == 4:
        return movie[y0:y1, x0:x1, :, :]
    raise ValueError(f"Unsupported movie shape {movie.shape}")


def bbox_from_mask(mask: np.ndarray, padding_px: int = 0) -> Tuple[int, int, int, int]:
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


def preview_oriented_mean_image(
    movie_yxct: np.ndarray,
    *,
    channel_index: int = 0,
    transpose_xy: bool = False,
    flip_ud: bool = False,
    flip_lr: bool = False,
):
    import matplotlib.pyplot as plt

    mean_im = np.nanmean(movie_yxct[:, :, channel_index, :], axis=2)
    mean_im = apply_orientation_transform(
        mean_im,
        transpose_xy=transpose_xy,
        flip_ud=flip_ud,
        flip_lr=flip_lr,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(np.nan_to_num(mean_im), cmap="viridis", origin="upper")
    plt.title("Oriented mean image")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def _get_default_font(fontsize: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try common scalable fonts first
    candidates = [
        "Arial.ttf",
        "arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    ]

    for name in candidates:
        try:
            return ImageFont.truetype(name, fontsize)
        except Exception:
            pass

    # Try matplotlib's bundled DejaVu Sans if available
    try:
        from matplotlib import font_manager
        font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        if font_path:
            return ImageFont.truetype(font_path, fontsize)
    except Exception:
        pass

    # Last resort
    return ImageFont.load_default()


def _draw_timer(
    draw: ImageDraw.ImageDraw,
    *,
    frame_idx: int,
    native_frame_rate_hz: float,
    margin_px: int,
    fontsize: int,
    text_color: Tuple[int, int, int],
) -> None:
    elapsed_s = frame_idx / native_frame_rate_hz
    text = f"t = {elapsed_s:0.3f} s"
    font = _get_default_font(fontsize)
    draw.text((margin_px, margin_px), text, font=font, fill=text_color)


def _draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    *,
    image_size_xy: Tuple[int, int],
    pixel_size_um: float,
    scale_bar_um: float,
    margin_px: int,
    linewidth: int,
    fontsize: int,
    bar_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
) -> None:
    width_px, height_px = image_size_xy
    bar_len_px = int(round(scale_bar_um / pixel_size_um))
    if bar_len_px <= 0:
        return

    font = _get_default_font(fontsize)
    label = f"{int(scale_bar_um) if float(scale_bar_um).is_integer() else scale_bar_um:g} µm"

    x1 = width_px - margin_px
    x0 = max(margin_px, x1 - bar_len_px)
    y = height_px - margin_px

    draw.line((x0, y, x1, y), fill=bar_color, width=linewidth)

    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = font.getsize(label)

    tx = x1 - text_w
    ty = y - text_h - max(4, linewidth + 2)
    draw.text((tx, ty), label, font=font, fill=text_color)


def _apply_overlays_to_frames(
    frames: np.ndarray,
    *,
    native_frame_rate_hz: float,
    config: MovieRenderConfig,
    verbose: bool = False,
) -> np.ndarray:
    """
    Apply timer and scale bar overlays to frames of shape (T, Y, X, 3).
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (T, Y, X, 3), got {frames.shape}")

    if not config.show_timer and not config.show_scale_bar:
        return frames

    out = frames.copy()
    n_frames = out.shape[0]
    
    effective_pixel_size_um = config.pixel_size_um / max(config.upsample_factor, 1.0)

    for i in range(n_frames):
        img = Image.fromarray(out[i])
        draw = ImageDraw.Draw(img)
        width_px, height_px = img.size
        fontsize = _resolve_overlay_fontsize(config, height_px)
        linewidth = _resolve_overlay_linewidth(config, height_px, width_px)
        margin_px = _resolve_overlay_margin(config, height_px, width_px)
        if config.show_timer:
            _draw_timer(
                draw,
                frame_idx=i,
                native_frame_rate_hz=native_frame_rate_hz,
                margin_px=margin_px,
                fontsize=fontsize,
                text_color=config.overlay_text_color,
            )

        if config.show_scale_bar:
            _draw_scale_bar(
                draw,
                image_size_xy=(width_px, height_px),
                pixel_size_um=effective_pixel_size_um,
                scale_bar_um=config.scale_bar_um,
                margin_px=margin_px,
                linewidth=linewidth,
                fontsize=fontsize,
                bar_color=config.overlay_bar_color,
                text_color=config.overlay_text_color,
            )

        out[i] = np.asarray(img)

        if verbose and (i == 0 or (i + 1) % max(1, n_frames // 5) == 0 or i == n_frames - 1):
            print(f"Overlayed {i + 1}/{n_frames} frames")

    return out


def render_glutamate_df_movie(
    output_path: PathLike,
    *,
    config: MovieRenderConfig,
    movie: Optional[np.ndarray] = None,
    dmd1_tiff: Optional[PathLike] = None,
    dmd2_tiff: Optional[PathLike] = None,
    n_channels: ChannelSpec = "auto",
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    roi_mask: Optional[np.ndarray] = None,
    roi_masks: Optional[Sequence[np.ndarray]] = None,
    crop_bbox: Optional[Tuple[int, int, int, int]] = None,
    dmd_selection: str = "both",
    overwrite: bool = False,
    draw_timestamp: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Render a cyan/red glutamate movie.

    Behavior:
    - If orientation transforms are requested, crop_bbox is interpreted
      in final displayed coordinates.
    - If no orientation transforms are requested, crop_bbox may be used
      during load for speed.
    """
    del draw_timestamp

    total_steps = 8
    step = 0

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Set overwrite=True to replace it.")

    resolved_start_frame, resolved_end_frame = _resolve_frame_window(
        frame_rate_hz=config.input_frame_rate_hz,
        start_s=start_s,
        end_s=end_s,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    orientation_requested = config.transpose_xy or config.flip_ud or config.flip_lr

    load_crop_bbox = None
    if not orientation_requested:
        if crop_bbox is not None:
            load_crop_bbox = crop_bbox
        elif roi_mask is not None:
            load_crop_bbox = bbox_from_mask(roi_mask, padding_px=config.padding_px)

    step += 1
    _progress(verbose, step, total_steps, "Loading movie subset")
    loaded_from_tiff = movie is None
    if movie is None:
        if dmd1_tiff is None and dmd_selection in {"dmd1", "both"}:
            raise ValueError("dmd1_tiff is required.")
        if dmd2_tiff is None and dmd_selection in {"dmd2", "both"}:
            raise ValueError("dmd2_tiff is required.")

        movie = load_slap2_movie_from_tiffs(
            dmd1_tiff=dmd1_tiff,
            dmd2_tiff=dmd2_tiff,
            n_channels=n_channels,
            combine_dmds=True,
            start_frame=resolved_start_frame,
            end_frame=resolved_end_frame,
            crop_bbox=load_crop_bbox,
            dmd_selection=dmd_selection,
            verbose=verbose,
        )
    else:
        movie = np.asarray(movie, dtype=np.float32)

    step += 1
    _progress(verbose, step, total_steps, "Selecting channel / standardizing shape")
    movie = np.asarray(movie, dtype=np.float32)

    if movie.ndim == 4:
        if config.channel_index >= movie.shape[2]:
            raise ValueError(
                f"channel_index={config.channel_index} out of bounds for movie shape {movie.shape}"
            )
        movie = movie[:, :, config.channel_index, :]
    elif movie.ndim != 3:
        raise ValueError(f"Expected movie shape (Y, X, T) or (Y, X, C, T), got {movie.shape}")

    _vprint(verbose, f"Movie shape after channel selection: {movie.shape}")

    step += 1
    if loaded_from_tiff:
        _progress(verbose, step, total_steps, "Time window already applied during load")
    else:
        _progress(verbose, step, total_steps, "Selecting time window")
        movie = select_movie_time_window(
            movie,
            frame_rate_hz=config.input_frame_rate_hz,
            start_s=start_s,
            end_s=end_s,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    _vprint(verbose, f"Movie shape after time selection: {movie.shape}")

    if movie.shape[2] < 2:
        raise ValueError("Selected movie window is too short.")

    step += 1
    _progress(verbose, step, total_steps, "Applying orientation transform")
    movie = apply_orientation_transform(
        movie,
        transpose_xy=config.transpose_xy,
        flip_ud=config.flip_ud,
        flip_lr=config.flip_lr,
    )
    _vprint(verbose, f"Movie shape after orientation transform: {movie.shape}")

    step += 1
    _progress(verbose, step, total_steps, "Cropping in display space if needed")
    if orientation_requested:
        if crop_bbox is not None:
            movie = crop_movie_to_bbox(movie, crop_bbox)
        elif roi_mask is not None:
            oriented_mask = apply_orientation_transform(
                roi_mask,
                transpose_xy=config.transpose_xy,
                flip_ud=config.flip_ud,
                flip_lr=config.flip_lr,
            )
            bbox = bbox_from_mask(oriented_mask, padding_px=config.padding_px)
            movie = crop_movie_to_bbox(movie, bbox)
        elif roi_masks is not None:
            oriented_masks = [
                apply_orientation_transform(
                    m,
                    transpose_xy=config.transpose_xy,
                    flip_ud=config.flip_ud,
                    flip_lr=config.flip_lr,
                )
                for m in roi_masks
            ]
            bbox = bbox_from_masks_union(
                roi_masks=oriented_masks,
                image_shape=movie.shape[:2],
                padding_px=config.padding_px,
            )
            movie = crop_movie_to_bbox(movie, bbox)
    else:
        if load_crop_bbox is None and roi_masks is not None:
            bbox = bbox_from_masks_union(
                roi_masks=roi_masks,
                image_shape=movie.shape[:2],
                padding_px=config.padding_px,
            )
            movie = crop_movie_to_bbox(movie, bbox)

    _vprint(verbose, f"Movie shape after crop: {movie.shape}")

    step += 1
    _progress(verbose, step, total_steps, "Temporal downsampling")
    if config.downsample_factor_time > 0:
        movie_ds = temporal_downsample(movie, config.downsample_factor_time)
    else:
        movie_ds = movie

    eff_frame_rate = config.input_frame_rate_hz / (2 ** config.downsample_factor_time)
    baseline_window_frames = max(3, int(np.ceil(config.baseline_window_s * eff_frame_rate)))

    _vprint(verbose, f"Movie shape after downsampling: {movie_ds.shape}")
    _vprint(verbose, f"Effective frame rate: {eff_frame_rate:.3f} Hz")
    _vprint(verbose, f"Baseline window: {baseline_window_frames} frames")
    _vprint(verbose, f"Baseline mode: {config.baseline_mode}")

    step += 1
    _progress(verbose, step, total_steps, "Computing structure/activity volumes")
    structure, activity = _compute_structure_and_activity(
        movie_ds,
        baseline_window_frames=baseline_window_frames,
        baseline_mode=config.baseline_mode,
        activity_percentile=config.activity_percentile,
        structure_percentile=config.structure_percentile,
        median_filter_xy=config.median_filter_xy,
        activity_gaussian_sigma_px=config.activity_gaussian_sigma_px,
        structure_gaussian_sigma_px=config.structure_gaussian_sigma_px,
        activity_temporal_sigma_frames=config.activity_temporal_sigma_frames,
    )

    step += 1
    _progress(verbose, step, total_steps, "Composing RGB / upsampling / overlays / writing video")
    rgb = _compose_rgb_frames(structure, activity, gamma=config.gamma)
    rgb = _pad_rgb_frames_to_even(rgb)

    frames = np.transpose(rgb, (2, 0, 1, 3))  # (T, Y, X, 3)

    if config.upsample_factor != 1.0:
        _vprint(
            verbose,
            f"Upsampling frames by {config.upsample_factor}x using {config.upsample_mode}"
        )
        frames = _upsample_frames(frames, config.upsample_factor, config.upsample_mode)

    frames = _apply_overlays_to_frames(
        frames,
        native_frame_rate_hz=eff_frame_rate,
        config=config,
        verbose=verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    _vprint(verbose, f"Writing {frames.shape[0]} frames to: {output_path}")
    _vprint(verbose, f"Output frame size: {frames.shape[2]} x {frames.shape[1]}")
    iio.imwrite(
        output_path,
        frames,
        fps=config.output_frame_rate_hz,
        codec=config.codec,
        macro_block_size=1,
        ffmpeg_params=[
            "-crf", str(config.video_crf),
            "-preset", str(config.video_preset),
            "-pix_fmt", "yuv420p",
        ],
    )

    _vprint(verbose, f"Done: {output_path}")
    return output_path