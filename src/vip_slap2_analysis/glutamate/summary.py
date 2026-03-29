from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import h5py

import warnings
import pandas as pd
from scipy.interpolate import PchipInterpolator

from vip_slap2_analysis.io.matv73 import MatV73File, bytes_to_str


ChannelSpec = Union[None, int, str, Sequence[Union[int, str]]]


@dataclass
class UnmixResult:
    ca_unmixed: np.ndarray      # (n_rois, n_samples)
    beta: np.ndarray            # (n_rois,)
    intercept: np.ndarray       # (n_rois,)
    method: str


@dataclass
class DffResult:
    dff: np.ndarray             # (n_rois, n_samples)
    baseline: np.ndarray        # (n_rois, n_samples)
    method: str


def _nan_pad_artifacts_by_diff(
    x: np.ndarray,
    std_factor: float = 20.0,
    nan_pad: int = 10,
) -> np.ndarray:
    """
    Port of ExperimentSummary.process_ca_trace artifact masking:
    detect big jumps in diff(x) and NaN-pad around them.
    """
    x = np.asarray(x, float).copy()
    x[np.isinf(x)] = np.nan
    if x.ndim != 1:
        raise ValueError("_nan_pad_artifacts_by_diff expects 1D trace")

    d = np.diff(x)
    thresh = np.nanmedian(d) + std_factor * np.nanstd(d)

    y = x.copy()
    for i in range(1, len(d) - 1):
        if np.abs(d[i]) > thresh:
            a = max(0, i - nan_pad)
            b = min(len(y), i + nan_pad)
            y[a:b] = np.nan
    return y


def _moving_average_reflect(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[win:] - c[:-win]) / win


def _baseline_percentile_filter(
    x: np.ndarray,
    fs_hz: float,
    window_s: float = 4.0,
    q: float = 10.0,
    smooth_s: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Practical stand-in for "compute_f0(... hull_window ... denoise_window ...)".
    Uses a percentile filter to follow the lower envelope + optional smoothing.
    """
    x = np.asarray(x, float)
    n = x.size
    win = max(3, int(round(window_s * fs_hz)))
    if win % 2 == 0:
        win += 1

    # fill NaNs for filtering, but preserve mask later
    if np.isfinite(x).any():
        fill = np.nanmedian(x)
    else:
        fill = 0.0
    xf = x.copy()
    xf[~np.isfinite(xf)] = fill

    try:
        from scipy.ndimage import percentile_filter
        base = percentile_filter(xf, percentile=q, size=win, mode="reflect")
    except Exception:
        # fallback: coarse baseline via moving average
        base = _moving_average_reflect(xf, win)

    # optional smoothing of baseline
    smooth = max(1, int(round(smooth_s * fs_hz)))
    base = _moving_average_reflect(base, smooth)

    return np.maximum(base, eps)

def _movmean_nan(x: np.ndarray, win: int) -> np.ndarray:
    """
    NaN-aware moving mean using pandas. Returns same length as x.
    """
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    return (
        pd.Series(x)
        .rolling(window=win, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def compute_f0(Fin, denoise_window: int, hull_window: int):
    """
    Baseline estimator (hull-like) used by your colleague.

    Args
    ----
    Fin : array_like, shape (T, ...) with time along axis 0 (NaNs allowed)
    denoise_window : int
        Median filter window (time samples).
    hull_window : int
        Window that controls the “convex hull”-like operation.

    Returns
    -------
    F0 : ndarray, same shape as Fin
    """
    F = np.asarray(Fin)
    orig_shape = F.shape
    if F.ndim == 1:
        F = F[:, None]
    else:
        F = F.reshape(F.shape[0], -1)

    T, C = F.shape
    if T < 4:
        return np.ones_like(Fin, dtype=float) * np.nanmean(Fin, axis=0, keepdims=True)

    hull_window = int(min(hull_window, T // 4))
    delta_des = max(4.0, denoise_window / 6.0)

    sample_times = np.rint(np.linspace(0, T - 1, num=int(np.ceil(T / delta_des) + 1))).astype(int)
    n_samps_in_hull = int(np.ceil(hull_window / delta_des))

    # rolling median denoise
    F0 = (
        pd.DataFrame(F)
        .rolling(window=denoise_window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    origsz = F0.shape
    F0 = F0.reshape(origsz[0], -1)

    for cix in range(F0.shape[1]):
        if np.all(np.isnan(F0[:, cix])):
            continue

        F00 = np.full((sample_times.shape[0], n_samps_in_hull), np.nan)
        for dix in range(n_samps_in_hull, 0, -1):
            xi = sample_times[dix - 1 :: n_samps_in_hull]
            F00[:, dix - 1] = np.interp(sample_times, xi, F0[xi, cix], left=np.nan, right=np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            FF = np.nanmin(F00, axis=1)

        doubt = np.sum(~np.isnan(F00), axis=1) < int(np.ceil(n_samps_in_hull / 2))
        if np.sum(~doubt) > 2:
            FF[doubt] = np.nan

        win = 2 * int(np.ceil(n_samps_in_hull / 2.0)) + 1
        fill = _movmean_nan(FF, win)
        nan_mask = np.isnan(FF)
        FF[nan_mask] = fill[nan_mask]
        FF = _movmean_nan(FF, win)

        nan_mask = np.isnan(FF)
        if np.any(nan_mask):
            FF = np.interp(sample_times, sample_times[~nan_mask], FF[~nan_mask])

        pchip = PchipInterpolator(sample_times, FF, extrapolate=True)
        F0[:, cix] = pchip(np.arange(T))

    return F0.reshape(orig_shape)


def unmix_ca_with_glu_hp_regress(
    ca: np.ndarray,
    glu: np.ndarray,
    fs_hz: float,
    hp_window_s: float = 0.2,
    ridge: float = 1e-6,
    min_finite_frac: float = 0.5,
) -> UnmixResult:
    """
    Remove glutamate-shaped contamination from Ca by fitting on high-pass components:
      ca_hp ≈ beta * glu_hp + intercept
    and subtracting only the fitted high-pass glutamate nuisance from the raw Ca trace.

    Notes
    -----
    The fitted coefficient is estimated on high-pass traces, so we subtract
    ``beta * glu_hp`` rather than ``beta * glu``. Subtracting the full glutamate
    trace can introduce slow step-like offsets and remove meaningful biological
    structure from the Ca signal.
    """
    ca = np.asarray(ca, float)
    glu = np.asarray(glu, float)
    if ca.shape != glu.shape or ca.ndim != 2:
        raise ValueError("ca and glu must be (n_rois, n_samples) and match shape")

    n_rois, _ = ca.shape
    win = max(3, int(round(hp_window_s * fs_hz)))
    if win % 2 == 0:
        win += 1

    beta = np.full(n_rois, np.nan)
    intercept = np.full(n_rois, np.nan)
    out = ca.copy()

    for i in range(n_rois):
        ca_i = ca[i]
        glu_i = glu[i]
        finite = np.isfinite(ca_i) & np.isfinite(glu_i)
        if finite.mean() < min_finite_frac:
            continue

        ca_f = ca_i.copy()
        glu_f = glu_i.copy()
        ca_f[~np.isfinite(ca_f)] = np.nanmedian(ca_f[finite])
        glu_f[~np.isfinite(glu_f)] = np.nanmedian(glu_f[finite])

        ca_lp = _moving_average_reflect(ca_f, win)
        glu_lp = _moving_average_reflect(glu_f, win)
        ca_hp = ca_f - ca_lp
        glu_hp = glu_f - glu_lp

        x = glu_hp[finite]
        y = ca_hp[finite]
        X = np.vstack([x, np.ones_like(x)]).T
        A = X.T @ X
        A[0, 0] += ridge
        b = X.T @ y
        b0, b1 = np.linalg.solve(A, b)

        beta[i] = b0
        intercept[i] = b1
        out[i, finite] = ca_i[finite] - beta[i] * glu_hp[finite]

    return UnmixResult(out, beta, intercept, method="hp_regress")


def _as_1d_bool(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(bool).squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


def _align_bool_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """Ensure mask is 1D length n. Truncate if longer; pad False if shorter."""
    mask = _as_1d_bool(mask)
    if mask.size == n:
        return mask
    if mask.size > n:
        return mask[:n]
    out = np.zeros(n, dtype=bool)
    out[: mask.size] = mask
    return out


@dataclass
class GlutamateSummary:
    """
    Fast, lazy loader for summarize_LoCo.m ExperimentSummary files.

    Expected layout (MATLAB -v7.3 / HDF5):
      - top-level group: exptSummary
      - exptSummary/E: ref-typed dataset (MATLAB cell array) indexing per-trial groups
      - per-trial group commonly contains:
          dF/events, dF/denoised, dF/ls
          F0, SNR, footprints, discardFrames
          frameLines
          ROIs/F, ROIs/Fsvd
          global/F
      - summary images (ref-cells):
          meanIM{dmd}, actIM{dmd}, selPix{dmd}, userROIs{dmd}
    Notes:
      - Some files reserve a 2-channel axis even when channel 2 is all-NaNs.
      - For synapse/source traces we normalize to (samples, rois, channels) internally.
      - For *user ROI* traces we return (rois, channels, samples) to match legacy ExperimentSummary.
    """

    file_path: Union[str, Path]
    keep_open: bool = True
    swap_xy_images: bool = True
    cache_e_groups: bool = True
    max_group_cache: int = 64  # cap for cached dereferenced E groups

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        self._mat = MatV73File(self.file_path, keep_open=self.keep_open)

        if "exptSummary" not in self._mat.f:
            raise KeyError(
                f"Top-level variable 'exptSummary' not found. Keys: {list(self._mat.f.keys())}"
            )

        self.n_trials: int = 0
        self.n_dmds: int = 0
        self.keep_trials: np.ndarray
        self.valid_trials: List[List[int]] = []
        self.trials_to_analyze: List[int] = []
        self.n_synapses: List[int] = []
        self.dmd_zs: np.ndarray = np.array([np.nan, np.nan], dtype=float)

        # E layout can be (dmd, trial) or (trial, dmd)
        self._E_layout: str = "dmd_trial"

        # caching
        self._e_cache: Dict[Tuple[int, int], h5py.Group] = {}
        self._e_cache_order: List[Tuple[int, int]] = []

        # lazy metadata
        self._metadata: Optional[Dict[str, Any]] = None
        self._align_params: Optional[Dict[str, Any]] = None

        self._get_info()

    # ----------------- lifecycle -----------------

    def close(self) -> None:
        self._mat.close()

    # ----------------- structure inference -----------------

    def _get_info(self) -> None:
        f = self._mat.f
        E = f["exptSummary"]["E"]

        if len(E.shape) != 2:
            raise ValueError(f"Unexpected exptSummary/E shape: {E.shape}")

        # Z planes if present
        try:
            Z = f["exptSummary"]["Z"][()]
            self.dmd_zs = np.array(Z).astype(float).flatten()
        except Exception:
            pass

        s0, s1 = E.shape

        # Heuristic: DMD count small; trials larger
        if s0 <= 4 and s1 > s0:
            self._E_layout = "dmd_trial"
            self.n_dmds, self.n_trials = int(s0), int(s1)
        elif s1 <= 4 and s0 > s1:
            self._E_layout = "trial_dmd"
            self.n_trials, self.n_dmds = int(s0), int(s1)
        else:
            # ambiguous; default to dmd_trial
            self._E_layout = "dmd_trial"
            self.n_dmds, self.n_trials = int(s0), int(s1)

        self.keep_trials = np.full((self.n_dmds, self.n_trials), False)

        # Determine valid trials cheaply by dereferencing and touching keys
        for dmd0 in range(self.n_dmds):
            for trial0 in range(self.n_trials):
                ref = self._E_ref(dmd0, trial0)
                if ref is None:
                    continue
                try:
                    node = self._mat.deref(ref)
                    if isinstance(node, h5py.Group):
                        _ = list(node.keys())
                        self.keep_trials[dmd0, trial0] = True
                except Exception:
                    pass

        self.valid_trials = [
            list(1 + np.argwhere(self.keep_trials[dmd0])[:, 0])
            for dmd0 in range(self.n_dmds)
        ]
        self.trials_to_analyze = list(1 + np.where(self.keep_trials.any(axis=0))[0])

        # Infer synapse count per DMD using dF/ls from first valid trial
        self.n_synapses = [0 for _ in range(self.n_dmds)]
        for dmd0 in range(self.n_dmds):
            v = np.argwhere(self.keep_trials[dmd0])
            if v.size == 0:
                continue
            trial0 = int(v[0, 0])
            try:
                g = self._E_group(dmd0, trial0)
                ds = g["dF"]["ls"]
                self.n_synapses[dmd0] = self._infer_n_rois_from_trace_dataset(ds, g=g)
            except Exception:
                self.n_synapses[dmd0] = 0

    def _E_ref(self, dmd0: int, trial0: int) -> Optional[h5py.Reference]:
        E = self._mat.f["exptSummary"]["E"]
        if self._E_layout == "dmd_trial":
            ref = E[dmd0, trial0]
        else:
            ref = E[trial0, dmd0]

        if ref is None:
            return None
        try:
            if int(ref) == 0:  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        return ref

    def _E_group(self, dmd0: int, trial0: int) -> h5py.Group:
        key = (dmd0, trial0)
        if self.cache_e_groups and key in self._e_cache:
            return self._e_cache[key]

        ref = self._E_ref(dmd0, trial0)
        if ref is None:
            raise ValueError(f"No E ref for dmd={dmd0+1}, trial={trial0+1}")

        node = self._mat.deref(ref)
        if not isinstance(node, h5py.Group):
            raise TypeError("E ref did not dereference to a Group")

        if self.cache_e_groups:
            self._e_cache[key] = node
            self._e_cache_order.append(key)
            if len(self._e_cache_order) > self.max_group_cache:
                old = self._e_cache_order.pop(0)
                self._e_cache.pop(old, None)

        return node

    # ----------------- grab aData structure from summary .mat file -----------------

    def _aData_ref(self, dmd0: int, trial0: int) -> Optional[h5py.Reference]:
        f = self._mat.f
        if "aData" not in f["exptSummary"]:
            return None
        A = f["exptSummary"]["aData"]
        if self._E_layout == "dmd_trial":
            ref = A[dmd0, trial0]
        else:
            ref = A[trial0, dmd0]

        if ref is None:
            return None
        try:
            if int(ref) == 0:  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        return ref

    def _aData_group(self, dmd0: int, trial0: int) -> Optional[h5py.Group]:
        ref = self._aData_ref(dmd0, trial0)
        if ref is None:
            return None
        node = self._mat.deref(ref)
        return node if isinstance(node, h5py.Group) else None

    # ----------------- metadata (lazy) -----------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """Decode exptSummary/params into a python dict (lazy)."""
        if self._metadata is None:
            self._metadata = self._read_params_group("exptSummary/params")
        return self._metadata

    @property
    def align_params(self) -> Dict[str, Any]:
        """Decode exptSummary/trialTable/alignParams into a python dict (lazy)."""
        if self._align_params is None:
            try:
                self._align_params = self._read_params_group("exptSummary/trialTable/alignParams")
            except Exception:
                self._align_params = {}
        return self._align_params

    def _read_params_group(self, group_path: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        g = self._mat.f[group_path]
        for k, ds in g.items():
            try:
                if isinstance(ds, h5py.Dataset):
                    # attempt string decode
                    try:
                        if h5py.check_dtype(vlen=ds.dtype) is str or ds.dtype.kind in ("S", "O"):
                            sval = ds.asstr()[()]
                            if isinstance(sval, np.ndarray):
                                sval = "".join(np.ravel(sval).tolist())
                            out[k] = str(sval).rstrip("\x00")
                            continue
                    except Exception:
                        pass

                    val = ds[()]
                    val = bytes_to_str(val)
                    if isinstance(val, np.ndarray):
                        val2 = np.squeeze(val)
                        out[k] = val2.tolist() if val2.ndim > 0 else val2.item()
                    else:
                        out[k] = val
            except Exception:
                pass
        return out

    # ----------------- channel selection -----------------

    def _normalize_channels(self, channels: ChannelSpec, n_channels: int) -> Optional[np.ndarray]:
        """
        Return 0-based channel indices or None meaning 'all'.

        Accepted: None, int, "green"/"red", list/tuple of those.
        """
        if channels is None:
            return None

        if isinstance(channels, (list, tuple, np.ndarray)):
            idxs: List[int] = []
            for c in channels:
                sub = self._normalize_channels(c, n_channels)
                if sub is None:
                    return None
                idxs.extend(sub.tolist())
            return np.array(sorted(set(idxs)), dtype=int)

        if isinstance(channels, str):
            c = channels.lower()
            if c in ("g", "green", "glu", "glutamate"):
                return np.array([0], dtype=int)
            if c in ("r", "red", "ca", "rcamp", "calcium"):
                return np.array([1], dtype=int)
            raise ValueError(f"Unknown channel spec: {channels}")

        if isinstance(channels, (int, np.integer)):
            c = int(channels)
            # accept matlab 1-indexed
            if c in (1, 2, 3, 4) and (c - 1) < n_channels:
                return np.array([c - 1], dtype=int)
            # accept python 0-indexed
            if 0 <= c < n_channels:
                return np.array([c], dtype=int)
            raise ValueError(f"Channel index out of range: {channels} (n_channels={n_channels})")

        raise TypeError(f"Unsupported channels type: {type(channels)}")

    # ----------------- trace helpers -----------------

    def _infer_n_rois_from_trace_dataset(self, ds: h5py.Dataset, g: Optional[h5py.Group] = None) -> int:
        """
        Infer number of sources/rois in a synapse-trace dataset.

        Works for shapes:
          - (time, rois)
          - (time, rois, ch) or (time, ch, rois) or (ch, time, rois)
        """
        shape = tuple(ds.shape)

        # time length if frameLines exists
        n_time = None
        if g is not None and "frameLines" in g:
            try:
                n_time = int(np.asarray(g["frameLines"][()]).size)
            except Exception:
                n_time = None

        if len(shape) == 2:
            if n_time is not None:
                if shape[0] == n_time:
                    return int(shape[1])
                if shape[1] == n_time:
                    return int(shape[0])
            # fallback assume time is larger axis
            return int(shape[1]) if shape[0] >= shape[1] else int(shape[0])

        if len(shape) == 3:
            # if time known, ROI is axis that is not time and not small-channel axis
            if n_time is not None:
                axes = list(range(3))
                time_ax = next((ax for ax, s in enumerate(shape) if s == n_time), None)
                if time_ax is not None:
                    axes.remove(time_ax)
                    # choose roi axis among remaining as larger (channels usually <= 4)
                    roi_ax = max(axes, key=lambda ax: shape[ax])
                    return int(shape[roi_ax])
            # fallback: ROI dimension is the middle-sized or largest non-channel
            return int(sorted(shape)[-2])  # usually rois
        return 0

    def _normalize_raw_to_time_roi_ch(
        self,
        raw: np.ndarray,
        n_channels: int,
        channels: Optional[np.ndarray],
        n_time: Optional[int],
    ) -> np.ndarray:
        """
        Normalize raw to (time, rois, ch).
        """
        x = np.asarray(raw)

        if x.ndim == 2:
            # (time, rois) or (rois, time)
            if n_time is not None:
                if x.shape[0] == n_time:
                    pass
                elif x.shape[1] == n_time:
                    x = x.T
                else:
                    # fallback: assume time larger
                    if x.shape[0] < x.shape[1]:
                        x = x.T
            else:
                if x.shape[0] < x.shape[1]:
                    x = x.T
            return x[:, :, None]

        if x.ndim != 3:
            raise ValueError(f"Expected 2D or 3D traces; got {x.shape}")

        # channel axis: prefer structural n_channels, not len(channels) (len(channels) may be 1)
        ch_ax = None

        if n_channels > 1:
            for ax, s in enumerate(x.shape):
                if s == n_channels:
                    ch_ax = ax
                    break

        # fall back to len(channels) only if it is informative (>1)
        if ch_ax is None and channels is not None and len(channels) > 1:
            for ax, s in enumerate(x.shape):
                if s == len(channels):
                    ch_ax = ax
                    break

        if ch_ax is None:
            ones = [ax for ax, s in enumerate(x.shape) if s == 1]
            ch_ax = ones[0] if ones else 0

        # time axis
        time_ax = None
        if n_time is not None:
            for ax, s in enumerate(x.shape):
                if ax != ch_ax and s == n_time:
                    time_ax = ax
                    break
        if time_ax is None:
            non_ch = [ax for ax in range(3) if ax != ch_ax]
            time_ax = max(non_ch, key=lambda ax: x.shape[ax])

        roi_ax = [ax for ax in range(3) if ax not in (time_ax, ch_ax)][0]

        x = np.moveaxis(x, [time_ax, roi_ax, ch_ax], [0, 1, 2])
        return x

    def _read_trace_dataset(
        self,
        ds: h5py.Dataset,
        frame_lines: Optional[np.ndarray],
        n_channels: int,
        channels: Optional[np.ndarray],
        t_slice: Optional[slice],
        roi_inds: Optional[Sequence[int]],
    ) -> np.ndarray:
        """
        Read ds (optionally sliced) and return normalized (time, rois, ch).
        """
        shape = tuple(ds.shape)
        ndim = len(shape)

        n_time = None
        if frame_lines is not None and frame_lines.size > 0:
            n_time = int(frame_lines.size)

        # Identify a plausible time axis in native ds
        def find_time_axis() -> Optional[int]:
            if n_time is None:
                return None
            for ax, s in enumerate(shape):
                if s == n_time:
                    return ax
            return None

        time_ax = find_time_axis()

        # Identify channel axis candidates in native ds
        ch_axes = [ax for ax, s in enumerate(shape) if s == n_channels] if n_channels > 1 else []
        ch_ax = None
        # Identify channel axis candidates in native ds
        ch_axes = [ax for ax, s in enumerate(shape) if s == n_channels] if n_channels > 1 else []
        ch_ax = None

        if ndim == 3:
            # 1) Prefer an axis that matches the STRUCTURAL channel count (e.g. 2)
            if ch_axes:
                if time_ax is not None:
                    cand = [ax for ax in ch_axes if ax != time_ax]
                    ch_ax = cand[0] if cand else ch_axes[0]
                else:
                    ch_ax = ch_axes[0]

            # 2) Only if we couldn't find that, fall back to len(channels)
            #    (dangerous when len(channels)==1 and ROI axis is also 1)
            if ch_ax is None and channels is not None and len(channels) != 1:
                for ax, s in enumerate(shape):
                    if s == len(channels):
                        ch_ax = ax
                        break

            # 3) Last resort: pick a singleton axis
            if ch_ax is None:
                ones = [ax for ax, s in enumerate(shape) if s == 1]
                ch_ax = ones[0] if ones else None

                if time_ax is None:
                    axes = list(range(ndim))
                    if ch_ax is not None and ch_ax in axes:
                        axes.remove(ch_ax)
                    time_ax = max(axes, key=lambda ax: shape[ax])

                if ndim == 2:
                    roi_ax = 1 - time_ax
                else:
                    remaining = [ax for ax in range(3) if ax not in (time_ax, ch_ax)]
                    roi_ax = remaining[0] if remaining else (0 if time_ax != 0 else 1)

        sel = [slice(None)] * ndim
        if t_slice is not None:
            sel[time_ax] = t_slice
        if roi_inds is not None:
            sel[roi_ax] = list(roi_inds)
        if ndim == 3 and ch_ax is not None and channels is not None:
            sel[ch_ax] = channels.tolist()

        raw = np.asarray(ds[tuple(sel)])
        return self._normalize_raw_to_time_roi_ch(raw, n_channels=n_channels, channels=channels, n_time=n_time)

    @staticmethod
    def _squeeze_channels(x: np.ndarray, squeeze_channels: bool) -> np.ndarray:
        if squeeze_channels and x.ndim == 3 and x.shape[2] == 1:
            return x[:, :, 0]
        return x

    def _get_motion_regressors(
        self,
        dmd: int,
        trial: int,
        target_len: int,
        use_fields: Sequence[str] = ("onlineXshift", "onlineYshift", "onlineZshift", "motionDSr", "motionDSc"),
        motion_step_thresh_z: float = 3.0,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build nuisance regressors aligned to the ROI trace length.

        Returns
        -------
        Xz : (target_len, P) float
            Z-scored regressors (mean 0, std 1 per column where possible).
        names : list[str]
            Names for each regressor column.

        Notes
        -----
        - Pulls vectors from aData struct fields, interpolates each to `target_len`.
        - Adds derived terms for dx/dy: quadratic + interaction + derivatives.
        - Adds speed/accel terms.
        - Adds a *signed* step regressor that can model down-then-up plateau artifacts
        (unlike a monotonic cumsum-only step).
        """
        dmd0, trial0 = dmd - 1, trial - 1
        g = self._aData_group(dmd0, trial0)
        if g is None:
            return np.zeros((target_len, 0), float), []

        regs: List[np.ndarray] = []
        names: List[str] = []

        def _read_vec(key: str) -> Optional[np.ndarray]:
            if key not in g:
                return None
            v = np.asarray(g[key][()]).squeeze()
            if v.ndim != 1:
                v = v.reshape(-1)
            v = v.astype(float)
            return v

        # --- raw motion vectors
        raw: Dict[str, np.ndarray] = {}
        for k in use_fields:
            v = _read_vec(k)
            if v is None or v.size < 2 or not np.isfinite(v).any():
                continue
            raw[k] = v

        if len(raw) == 0:
            return np.zeros((target_len, 0), float), []

        # --- interpolate each to target_len
        t_tgt = np.linspace(0, 1, target_len)
        for k, v in raw.items():
            t_src = np.linspace(0, 1, v.size)
            vv = v.copy()
            if np.any(~np.isfinite(vv)):
                finite = np.isfinite(vv)
                if finite.sum() < 2:
                    continue
                vv[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), vv[finite])
            vi = np.interp(t_tgt, t_src, vv)
            regs.append(vi)
            names.append(k)

        X = np.stack(regs, axis=1)  # (T, P)

        # --- derived regressors + step/speed
        have_dxdy = ("onlineXshift" in names) and ("onlineYshift" in names)
        if have_dxdy:
            dx = X[:, names.index("onlineXshift")]
            dy = X[:, names.index("onlineYshift")]

            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            # add quadratic + interaction + first derivatives
            regs2 = [dx**2, dy**2, dx * dy, ddx, ddy]
            names2 = ["dx2", "dy2", "dxdy", "ddx", "ddy"]
            X = np.concatenate([X, np.stack(regs2, axis=1)], axis=1)
            names.extend(names2)

            # optionally include z velocity in speed if available
            if "onlineZshift" in names:
                dz = X[:, names.index("onlineZshift")]
                ddz = np.gradient(dz)
                speed = np.sqrt(ddx**2 + ddy**2 + ddz**2)
                accel = np.sqrt(np.gradient(ddx)**2 + np.gradient(ddy)**2 + np.gradient(ddz)**2)
            else:
                speed = np.sqrt(ddx**2 + ddy**2)
                accel = np.sqrt(np.gradient(ddx)**2 + np.gradient(ddy)**2)

            # speed/accel regressors
            X = np.concatenate([X, speed[:, None], accel[:, None]], axis=1)
            names.extend(["speed", "accel"])

            # --- SIGNED motion step regressor (can go up AND down)
            # detect bursts of motion using speed z-score
            v_mu = np.nanmean(speed)
            v_sd = np.nanstd(speed)
            zv = (speed - v_mu) / (v_sd + 1e-12)
            events = zv > float(motion_step_thresh_z)

            # determine motion direction on event frames (unit direction of [ddx,ddy] (and ddz if present))
            dirx = ddx / (speed + 1e-12)
            diry = ddy / (speed + 1e-12)

            # choose a consistent sign reference from the mean direction during events
            if np.any(events):
                mx = float(np.nanmean(dirx[events]))
                my = float(np.nanmean(diry[events]))
                denom = np.sqrt(mx * mx + my * my) + 1e-12
                mx /= denom
                my /= denom
            else:
                mx, my = 1.0, 0.0  # arbitrary, won't matter if no events

            proj = dirx * mx + diry * my  # roughly in [-1,1]
            imp = np.zeros_like(dx, dtype=float)
            if np.any(events):
                imp[events] = np.sign(proj[events])
                # handle exact zeros
                imp[(events) & (imp == 0)] = 1.0

            step_signed = np.cumsum(imp)

            # add signed step (z-scored later with all columns)
            X = np.concatenate([X, step_signed[:, None]], axis=1)
            names.append("motion_step_signed")

        # --- z-score columns (helps ridge behave)
        Xz = X.copy()
        for j in range(Xz.shape[1]):
            col = Xz[:, j]
            mu = np.nanmean(col)
            sd = np.nanstd(col)
            if np.isfinite(sd) and sd > 0:
                Xz[:, j] = (col - mu) / sd
            else:
                Xz[:, j] = 0.0

        return Xz, names


    # ----------------- synapse/source traces -----------------

    def get_traces(
        self,
        dmd: int = 1,
        trial: int = 1,
        signal: str = "dF",
        mode: str = "ls",
        channels: ChannelSpec = None,
        t_slice: Optional[slice] = None,
        roi_inds: Optional[Sequence[int]] = None,
        drop_discarded: bool = False,
        return_frame_lines: bool = False,
        dtype: Optional[np.dtype] = None,
        force_n_channels: Optional[int] = None,
        pad_value: float = np.nan,
        drop_nan_channels: bool = False,
        squeeze_channels: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Load synapse/source traces.

        Normalized return is (samples, rois, channels) internally.
        If squeeze_channels=True and channels==1 => returns (samples, rois).

        Parameters
        ----------
        signal : {'dF','dFF','F0'}
            Note: many LoCo outputs do NOT contain dFF. If missing, raises KeyError.
        mode : e.g. 'ls', 'denoised', 'events'
        """
        dmd0, trial0 = dmd - 1, trial - 1
        if trial <= 0 or dmd <= 0:
            raise ValueError("dmd and trial must be 1-indexed positive integers")
        if not self.keep_trials[dmd0, trial0]:
            raise ValueError(f"Trial {trial} is invalid in DMD {dmd}")

        g = self._E_group(dmd0, trial0)

        # frameLines gives time axis length if present
        frame_lines = None
        if "frameLines" in g:
            try:
                frame_lines = np.asarray(g["frameLines"][()]).squeeze()
            except Exception:
                frame_lines = None

        # Determine number of channels for this dataset:
        # Many files reserve 2 channels (second is NaNs). If the dataset is 2D => 1 channel.
        # We'll infer from shape.
        if signal == "F0":
            ds = g["F0"]
        elif signal in ("dF", "dFF"):
            if signal not in g:
                raise KeyError(
                    f"'{signal}' not found for dmd={dmd}, trial={trial}. "
                    f"Available top-level keys: {list(g.keys())}"
                )
            if mode not in g[signal]:
                raise KeyError(
                    f"Mode '{mode}' not found under '{signal}' for dmd={dmd}, trial={trial}. "
                    f"Available modes: {list(g[signal].keys())}"
                )
            ds = g[signal][mode]
        else:
            raise ValueError(f"Unknown signal '{signal}'. Expected 'dF', 'dFF', or 'F0'.")

        # infer structural channels from dataset shape (not from content)
        ds_shape = tuple(ds.shape)
        n_channels_struct = 1 if len(ds_shape) == 2 else (2 if 2 in ds_shape else 1)
        ch_sel = self._normalize_channels(channels, n_channels_struct)

        raw = self._read_trace_dataset(
            ds,
            frame_lines=frame_lines,
            n_channels=n_channels_struct,
            channels=ch_sel,
            t_slice=t_slice,
            roi_inds=roi_inds,
        )  # (time, rois, ch)

        if drop_discarded and "discardFrames" in g:
            df = np.asarray(g["discardFrames"][()])
            df = _align_bool_mask(df, raw.shape[0])
            raw = raw[~df, :, :]

        # Optionally drop channels that are entirely NaN (content-based)
        if drop_nan_channels and raw.ndim == 3 and raw.shape[2] > 1:
            keep = [ch for ch in range(raw.shape[2]) if np.isfinite(raw[:, :, ch]).any()]
            if len(keep) == 0:
                keep = [0]
            raw = raw[:, :, keep]

        # Optionally force a fixed channel dimension (pad with NaNs)
        if force_n_channels is not None and raw.ndim == 3 and raw.shape[2] != force_n_channels:
            if raw.shape[2] > force_n_channels:
                raw = raw[:, :, :force_n_channels]
            else:
                pad = np.full(
                    (raw.shape[0], raw.shape[1], force_n_channels - raw.shape[2]),
                    pad_value,
                    dtype=raw.dtype,
                )
                raw = np.concatenate([raw, pad], axis=2)

        if dtype is not None:
            raw = raw.astype(dtype, copy=False)

        out = self._squeeze_channels(raw, squeeze_channels=squeeze_channels)

        if return_frame_lines:
            if frame_lines is None:
                frame_lines = np.array([], dtype=float)
            if t_slice is not None and frame_lines.size > 0:
                frame_lines = frame_lines[t_slice]
            return out, frame_lines

        return out

    # ----------------- user ROI traces -----------------

    def get_user_roi_traces(
        self,
        dmd: int = 1,
        trial: int = 1,
        trace_type: str = "F",
        channels: ChannelSpec = None,
        t_slice: Optional[slice] = None,
        roi_inds: Optional[Sequence[int]] = None,
        dtype: Optional[np.dtype] = None,
        squeeze_channels: bool = False,
    ) -> np.ndarray:
        """
        Load user ROI traces from E/ROIs/<trace_type>.

        UPDATED RETURN FORMAT (per your request):
          - returns (n_rois, n_channels, n_samples)

        Notes
        -----
        Under the hood, LoCo often stores ROIs traces in a few possible layouts.
        We normalize internally to (time, rois, ch), then *transpose* to (rois, ch, time).

        If squeeze_channels=True and n_channels==1, returns (n_rois, n_samples).
        """
        dmd0, trial0 = dmd - 1, trial - 1
        if trial <= 0 or dmd <= 0:
            raise ValueError("dmd and trial must be 1-indexed positive integers")
        if not self.keep_trials[dmd0, trial0]:
            raise ValueError(f"Trial {trial} is invalid in DMD {dmd}")

        g = self._E_group(dmd0, trial0)

        if "ROIs" not in g or trace_type not in g["ROIs"]:
            raise KeyError(f"ROIs/{trace_type} not found for dmd={dmd}, trial={trial}")

        ds = g["ROIs"][trace_type]
        ds_shape = tuple(ds.shape)
        n_channels_struct = 1 if len(ds_shape) == 2 else (2 if 2 in ds_shape else 1)
        ch_sel = self._normalize_channels(channels, n_channels_struct)

        frame_lines = None
        if "frameLines" in g:
            try:
                frame_lines = np.asarray(g["frameLines"][()]).squeeze()
            except Exception:
                frame_lines = None

        # internal normalized: (time, rois, ch)
        raw = self._read_trace_dataset(
            ds,
            frame_lines=frame_lines,
            n_channels=n_channels_struct,
            channels=ch_sel,
            t_slice=t_slice,
            roi_inds=roi_inds,
        )

        if dtype is not None:
            raw = raw.astype(dtype, copy=False)

        # Convert to requested format: (rois, ch, time)
        out = np.transpose(raw, (1, 2, 0))

        if squeeze_channels and out.ndim == 3 and out.shape[1] == 1:
            out = out[:, 0, :]  # (rois, time)

        return out

    # ----------------- images and footprints -----------------

    def _cell_item(self, name: str, dmd0: int) -> Optional[Union[h5py.Dataset, h5py.Group]]:
        f = self._mat.f
        if name not in f["exptSummary"]:
            return None

        cell = f["exptSummary"][name]
        if not isinstance(cell, h5py.Dataset) or cell.dtype != h5py.ref_dtype:
            return cell

        for (i, j) in [(dmd0, 0), (0, dmd0)]:
            try:
                ref = cell[i, j]
                if ref is None:
                    continue
                try:
                    if int(ref) == 0:  # type: ignore[arg-type]
                        continue
                except Exception:
                    pass
                return self._mat.deref(ref)
            except Exception:
                continue
        return None

    def get_summary_image(self, dmd: int, image_type: str) -> np.ndarray:
        """
        Load exptSummary meanIM or actIM for given DMD.

        image_type: 'meanIM' or 'actIM'
        """
        if image_type not in ("meanIM", "actIM"):
            raise ValueError("image_type must be 'meanIM' or 'actIM'")

        dmd0 = dmd - 1
        node = self._cell_item(image_type, dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"exptSummary/{image_type}{{{dmd}}} not found or unexpected type")

        img = np.asarray(node[()])
        if self.swap_xy_images and img.ndim >= 2:
            img = np.swapaxes(img, 0, 1)
        return img

    def get_sel_pix(self, dmd: int) -> np.ndarray:
        dmd0 = dmd - 1
        node = self._cell_item("selPix", dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"exptSummary/selPix{{{dmd}}} not found or unexpected type")

        m = np.asarray(node[()]).astype(bool)
        if self.swap_xy_images and m.ndim == 2:
            m = np.swapaxes(m, 0, 1)
        return m

    def get_footprints(self, dmd: int, trial: int) -> np.ndarray:
        """
        Load synapse footprints.

        Handles:
          - dense footprints: (n_rois, y, x) or (y, x, n_rois)
          - sparse LoCo footprints: (n_rois, n_selected_pixels) (+ selPix mask)
        Returns: (n_rois, y, x) in python display orientation.
        """
        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)

        if "footprints" not in g:
            raise KeyError("footprints not found in trial group")

        fp = np.asarray(g["footprints"][()])

        if fp.ndim == 2:
            sel = self.get_sel_pix(dmd)
            sel_flat = sel.reshape(-1)
            true_ind = np.where(sel_flat > 0)[0]

            if fp.shape[1] == len(true_ind):
                fp_rois_by_sel = fp
            elif fp.shape[0] == len(true_ind):
                fp_rois_by_sel = fp.T
            else:
                fp_rois_by_sel = fp

            n_rois = fp_rois_by_sel.shape[0]
            recon = np.full((n_rois, sel_flat.size), np.nan, dtype=float)
            recon[:, true_ind] = fp_rois_by_sel
            fp3 = recon.reshape((n_rois, sel.shape[0], sel.shape[1]))

        elif fp.ndim == 3:
            # (n_rois, y, x) or (y, x, n_rois)
            if fp.shape[0] == self.n_synapses[dmd0]:
                fp3 = fp
            elif fp.shape[-1] == self.n_synapses[dmd0]:
                fp3 = np.moveaxis(fp, -1, 0)
            else:
                fp3 = fp
        else:
            raise ValueError(f"Unexpected footprints shape: {fp.shape}")

        if self.swap_xy_images and fp3.ndim == 3:
            fp3 = np.swapaxes(fp3, 1, 2)

        return fp3

    # ----------------- convenience -----------------

    def timebase(self, dmd: int, trial: int, hz_key: str = "analyzeHz") -> np.ndarray:
        """
        Uniform timebase in seconds using params.analyzeHz when available.
        """
        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)

        n = None
        if "frameLines" in g:
            try:
                n = int(np.asarray(g["frameLines"][()]).size)
            except Exception:
                n = None

        if n is None:
            # fallback: infer from dF/ls if possible
            if "dF" in g and "ls" in g["dF"]:
                ds = g["dF"]["ls"]
                n = int(max(ds.shape))
            elif "ROIs" in g and "F" in g["ROIs"]:
                ds = g["ROIs"]["F"]
                n = int(max(ds.shape))
            else:
                raise ValueError("Could not infer time axis length for timebase()")

        hz = np.nan
        try:
            hz = float(self.metadata.get(hz_key, np.nan))
        except Exception:
            hz = np.nan

        if not np.isfinite(hz) or hz <= 0:
            return np.arange(n, dtype=float)
        return np.arange(n, dtype=float) / hz
        
    def get_soma_glu_ca_traces(
        self,
        dmd: int = 1,
        trial: int = 1,
        trace_type: str = "Fsvd",
        roi_inds: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience loader for somatic user ROI traces.
        Returns (glu, ca) each shaped (n_rois, n_samples), guaranteed.
        """
        glu = self.get_user_roi_traces(
            dmd=dmd, trial=trial, trace_type=trace_type,
            channels="glutamate", roi_inds=roi_inds, squeeze_channels=False
        )
        ca = self.get_user_roi_traces(
            dmd=dmd, trial=trial, trace_type=trace_type,
            channels="calcium", roi_inds=roi_inds, squeeze_channels=False
        )

        glu = np.asarray(glu, float)
        ca = np.asarray(ca, float)

        def _to_rois_by_time(x: np.ndarray) -> np.ndarray:
            # expected raw formats:
            #  (rois, time)                      -> OK
            #  (rois, ch, time) with ch==1       -> squeeze ch
            #  (time,)                           -> make (1, time)
            #  (time, rois)                      -> transpose if time axis larger
            if x.ndim == 1:
                return x[None, :]
            if x.ndim == 3:
                # (rois, ch, time) expected
                if x.shape[1] == 1:
                    return x[:, 0, :]
                # if somehow multiple channels slipped through, take first
                return x[:, 0, :]
            if x.ndim == 2:
                # decide whether it's (rois, time) or (time, rois)
                # time usually larger
                if x.shape[0] < x.shape[1]:
                    return x
                else:
                    # ambiguous; use frameLines length if possible could be better
                    # heuristic: if first dim is huge, it's time
                    return x.T
            raise ValueError(f"Unexpected trace shape for soma traces: {x.shape}")

        glu2 = _to_rois_by_time(glu)
        ca2 = _to_rois_by_time(ca)

        # final sanity: shapes must match in time axis
        T = min(glu2.shape[1], ca2.shape[1])
        glu2 = glu2[:, :T]
        ca2 = ca2[:, :T]

        # match ROI counts conservatively
        n = min(glu2.shape[0], ca2.shape[0])
        return glu2[:n], ca2[:n]
    
    def _estimate_ca_baseline(
        self,
        x: np.ndarray,
        fs_hz: float,
        *,
        baseline_method: Literal["hull", "percentile"] = "percentile",
        denoise_window_s: float = 2.0,
        hull_window_s: float = 90.0,
        baseline_window_s: float = 20.0,
        baseline_q: float = 15.0,
        baseline_smooth_s: float = 2.0,
        eps: float = 1e-6,
        f0_floor_frac: float = 0.15,
    ) -> np.ndarray:
        """Estimate a fluorescence-domain baseline for one ROI trace."""
        x = np.asarray(x, float)
        finite = np.isfinite(x)
        if finite.mean() < 0.5:
            return np.full_like(x, np.nan, dtype=float)

        if baseline_method == "hull":
            denoise_w = max(3, int(round(denoise_window_s * fs_hz)))
            hull_w = max(denoise_w + 1, int(round(hull_window_s * fs_hz)))
            base = compute_f0(x, denoise_window=denoise_w, hull_window=hull_w).reshape(-1)
            method = "hull"
        else:
            base = _baseline_percentile_filter(
                x,
                fs_hz=fs_hz,
                window_s=baseline_window_s,
                q=baseline_q,
                smooth_s=baseline_smooth_s,
                eps=eps,
            )
            method = "percentile_filter"

        x_ref = np.nanpercentile(x[finite], 20) if np.any(finite) else np.nan
        floor = max(eps, f0_floor_frac * x_ref) if np.isfinite(x_ref) else eps
        base = np.maximum(base, floor)
        base[~finite] = np.nan
        return base

    @staticmethod
    def _glu_hp_nuisance(
        glu: np.ndarray,
        fs_hz: float,
        hp_window_s: float,
    ) -> np.ndarray:
        """Build a standardized high-pass glutamate nuisance regressor."""
        g = np.asarray(glu, float).reshape(-1)
        win = max(3, int(round(hp_window_s * fs_hz)))
        if win % 2 == 0:
            win += 1

        gf = g.copy()
        finite = np.isfinite(gf)
        if finite.any():
            gf[~finite] = np.nanmedian(gf[finite])
        else:
            return np.zeros_like(gf)

        g_hp = gf - _moving_average_reflect(gf, win)
        mu = np.nanmean(g_hp)
        sd = np.nanstd(g_hp)
        if np.isfinite(sd) and sd > 0:
            g_hp = (g_hp - mu) / (sd + 1e-12)
        else:
            g_hp = np.zeros_like(g_hp)
        return g_hp

    def _process_soma_ca_trial(
        self,
        ca: np.ndarray,
        fs_hz: float,
        glu: Optional[np.ndarray] = None,
        *,
        X_motion: Optional[np.ndarray] = None,
        motion_names: Optional[Sequence[str]] = None,
        mask_artifacts: bool = True,
        std_factor: float = 20.0,
        nan_pad: int = 10,
        unmix: bool = True,
        hp_window_s: float = 0.2,
        ridge: float = 1e-6,
        motion_correct: bool = True,
        motion_ridge: float = 1e-1,
        use_glu_as_motion_regressor: bool = False,
        glu_motion_hp_window_s: float = 0.5,
        motion_regress_on: Literal["dF", "F"] = "dF",
        compute_dff: bool = True,
        baseline_method: Literal["hull", "percentile"] = "percentile",
        denoise_window_s: float = 2.0,
        hull_window_s: float = 90.0,
        f0_floor_frac: float = 0.05,
        baseline_window_s: float = 10.0,
        baseline_q: float = 10.0,
        baseline_smooth_s: float = 1.0,
        eps: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Process one trial of paired soma Ca / glutamate traces while preserving
        the legacy return contract used by extraction and QC.
        """
        ca = np.asarray(ca, float)
        if ca.ndim != 2:
            raise ValueError("ca must be (n_rois, n_samples)")

        n_rois, n_time = ca.shape
        if glu is not None:
            glu = np.asarray(glu, float)
            if glu.shape != ca.shape:
                raise ValueError(f"glu must match ca shape; got {glu.shape} vs {ca.shape}")

        ca_clean = ca.copy()
        if mask_artifacts:
            for i in range(n_rois):
                if np.isfinite(ca_clean[i]).any():
                    ca_clean[i] = _nan_pad_artifacts_by_diff(
                        ca_clean[i], std_factor=std_factor, nan_pad=nan_pad
                    )

        unmix_res: Optional[UnmixResult] = None
        ca_unmixed = ca_clean.copy()
        if unmix:
            if glu is None:
                raise ValueError("glu must be provided when unmix=True")
            unmix_res = unmix_ca_with_glu_hp_regress(
                ca=ca_clean,
                glu=glu,
                fs_hz=fs_hz,
                hp_window_s=hp_window_s,
                ridge=ridge,
            )
            ca_unmixed = unmix_res.ca_unmixed

        motion_names_list = list(motion_names) if motion_names is not None else []
        beta_motion = None

        if X_motion is None:
            X_motion = np.zeros((n_time, 0), dtype=float)
        else:
            X_motion = np.asarray(X_motion, float)
            if X_motion.ndim == 1:
                X_motion = X_motion[:, None]
            if X_motion.shape[0] != n_time:
                raise ValueError(
                    f"X_motion must have shape (n_time, P) with n_time={n_time}; got {X_motion.shape}"
                )

        baseline = np.full_like(ca_unmixed, np.nan, dtype=float)
        dff = np.full_like(ca_unmixed, np.nan, dtype=float)
        ca_mc = ca_unmixed.copy()
        dff_method = baseline_method if baseline_method == "hull" else "percentile_filter"

        if motion_correct and motion_regress_on == "dF":
            dF = np.full_like(ca_unmixed, np.nan, dtype=float)
            dF_mc = np.full_like(ca_unmixed, np.nan, dtype=float)
            beta_rows: List[np.ndarray] = []
            names_out = list(motion_names_list)
            if use_glu_as_motion_regressor:
                names_out = names_out + ["glu_hp"]

            for i in range(n_rois):
                base = self._estimate_ca_baseline(
                    ca_unmixed[i],
                    fs_hz,
                    baseline_method=baseline_method,
                    denoise_window_s=denoise_window_s,
                    hull_window_s=hull_window_s,
                    baseline_window_s=baseline_window_s,
                    baseline_q=baseline_q,
                    baseline_smooth_s=baseline_smooth_s,
                    eps=eps,
                    f0_floor_frac=f0_floor_frac,
                )
                baseline[i] = base
                finite = np.isfinite(ca_unmixed[i]) & np.isfinite(base)
                if finite.mean() < 0.5:
                    beta_rows.append(np.full((X_motion.shape[1] + (1 if use_glu_as_motion_regressor else 0) + 1,), np.nan))
                    continue

                dF_i = np.full(n_time, np.nan, dtype=float)
                dF_i[finite] = ca_unmixed[i, finite] - base[finite]
                dF[i] = dF_i

                Xi = X_motion
                if use_glu_as_motion_regressor:
                    if glu is None:
                        raise ValueError("glu must be provided when use_glu_as_motion_regressor=True")
                    g_hp = self._glu_hp_nuisance(glu[i], fs_hz=fs_hz, hp_window_s=glu_motion_hp_window_s)
                    Xi = np.column_stack([Xi, g_hp]) if Xi.size else g_hp[:, None]

                y_resid, b_mc = regress_out_(dF_i, Xi, ridge=motion_ridge)
                dF_mc[i] = y_resid
                beta_rows.append(b_mc)

                finite2 = np.isfinite(y_resid) & np.isfinite(base)
                if finite2.mean() >= 0.5:
                    dff[i, finite2] = y_resid[finite2] / base[finite2]
                    ca_mc[i, finite2] = base[finite2] + y_resid[finite2]
                    ca_mc[i, ~finite2] = np.nan
                else:
                    ca_mc[i] = np.nan

            beta_motion = np.stack(beta_rows, axis=0) if beta_rows else None
            motion_names_list = names_out

        else:
            if motion_correct and X_motion.size:
                beta_rows = []
                names_out = list(motion_names_list)
                if use_glu_as_motion_regressor:
                    names_out = names_out + ["glu_hp"]

                for i in range(n_rois):
                    Xi = X_motion
                    if use_glu_as_motion_regressor:
                        if glu is None:
                            raise ValueError("glu must be provided when use_glu_as_motion_regressor=True")
                        g_hp = self._glu_hp_nuisance(glu[i], fs_hz=fs_hz, hp_window_s=glu_motion_hp_window_s)
                        Xi = np.column_stack([Xi, g_hp]) if Xi.size else g_hp[:, None]
                    y_resid, b_mc = regress_out_(ca_unmixed[i], Xi, ridge=motion_ridge)
                    ca_mc[i] = y_resid
                    beta_rows.append(b_mc)
                beta_motion = np.stack(beta_rows, axis=0) if beta_rows else None
                motion_names_list = names_out

            if compute_dff:
                for i in range(n_rois):
                    base = self._estimate_ca_baseline(
                        ca_mc[i],
                        fs_hz,
                        baseline_method=baseline_method,
                        denoise_window_s=denoise_window_s,
                        hull_window_s=hull_window_s,
                        baseline_window_s=baseline_window_s,
                        baseline_q=baseline_q,
                        baseline_smooth_s=baseline_smooth_s,
                        eps=eps,
                        f0_floor_frac=f0_floor_frac,
                    )
                    baseline[i] = base
                    finite = np.isfinite(ca_mc[i]) & np.isfinite(base)
                    if finite.mean() < 0.5:
                        continue
                    dff[i, finite] = (ca_mc[i, finite] - base[finite]) / base[finite]

        dff_res: Optional[DffResult]
        if compute_dff:
            dff_res = DffResult(dff=dff, baseline=baseline, method=dff_method)
        else:
            dff_res = None

        return {
            "ca_clean": ca_clean,
            "unmix": unmix_res,
            "ca_unmixed": ca_unmixed,
            "ca_mc": ca_mc,
            "beta_motion": beta_motion,
            "motion_names": motion_names_list,
            "dff": dff_res,
        }

    def process_ca_trace_extended(
        self,
        ca: np.ndarray,
        fs_hz: float,
        glu: Optional[np.ndarray] = None,
        *,
        mask_artifacts: bool = True,
        std_factor: float = 20.0,
        nan_pad: int = 10,
        unmix: bool = True,
        hp_window_s: float = 0.2,
        ridge: float = 1e-6,
        motion_correct: bool = True,
        motion_ridge: float = 1e-1,
        use_motion_fields: Sequence[str] = ("onlineXshift", "onlineYshift", "motionDSr", "motionDSc"),
        use_glu_as_motion_regressor: bool = False,
        glu_motion_hp_window_s: float = 0.5,
        compute_dff: bool = True,
        motion_regress_on: Literal["dF", "F"] = "dF",
        baseline_method: Literal["hull", "percentile"] = "percentile",
        denoise_window_s: float = 2.0,
        hull_window_s: float = 90.0,
        f0_floor_frac: float = 0.15,
        baseline_window_s: float = 20.0,
        baseline_q: float = 15.0,
        baseline_smooth_s: float = 2.0,
        eps: float = 1e-6,
        X_motion: Optional[np.ndarray] = None,
        motion_names: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """
        Backward-compatible public wrapper for single-trial Ca processing.

        Notes
        -----
        - Glutamate loading and return keys are preserved.
        - Motion correction now operates when `X_motion` is supplied.
        - `use_motion_fields` is retained for API compatibility, but motion-field
        construction is handled by `get_processed_soma_ca(...)` /
        `get_processed_soma_ca_all_trials(...)`.
        """
        _ = use_motion_fields
        return self._process_soma_ca_trial(
            ca=ca,
            glu=glu,
            fs_hz=fs_hz,
            X_motion=X_motion,
            motion_names=motion_names,
            mask_artifacts=mask_artifacts,
            std_factor=std_factor,
            nan_pad=nan_pad,
            unmix=unmix,
            hp_window_s=hp_window_s,
            ridge=ridge,
            motion_correct=motion_correct,
            motion_ridge=motion_ridge,
            use_glu_as_motion_regressor=use_glu_as_motion_regressor,
            glu_motion_hp_window_s=glu_motion_hp_window_s,
            motion_regress_on=motion_regress_on,
            compute_dff=compute_dff,
            baseline_method=baseline_method,
            denoise_window_s=denoise_window_s,
            hull_window_s=hull_window_s,
            f0_floor_frac=f0_floor_frac,
            baseline_window_s=baseline_window_s,
            baseline_q=baseline_q,
            baseline_smooth_s=baseline_smooth_s,
            eps=eps,
        )

    def get_processed_soma_ca(
        self,
        dmd: int = 1,
        trial: int = 1,
        trace_type: str = "Fsvd",
        roi_inds: Optional[Sequence[int]] = None,
        fs_hz: Optional[float] = None,
        motion_correct: bool = True,
        motion_use_fields: Sequence[str] = ("onlineXshift", "onlineYshift", "onlineZshift", "motionDSr", "motionDSc"),
        motion_step_thresh_z: float = 3.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Load one trial of soma ROI Glu/Ca traces and run processing."""
        if fs_hz is None:
            fs_hz = float(self.metadata.get("analyzeHz", np.nan))
        if not np.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError("fs_hz must be provided or present in metadata['analyzeHz'].")

        glu, ca = self.get_soma_glu_ca_traces(dmd=dmd, trial=trial, trace_type=trace_type, roi_inds=roi_inds)
        X_motion, mot_names = self._get_motion_regressors(
            dmd=dmd,
            trial=trial,
            target_len=ca.shape[1],
            use_fields=motion_use_fields,
            motion_step_thresh_z=motion_step_thresh_z,
        ) if motion_correct else (None, None)

        return self.process_ca_trace_extended(
            ca=ca,
            glu=glu,
            fs_hz=fs_hz,
            motion_correct=motion_correct,
            X_motion=X_motion,
            motion_names=mot_names,
            **kwargs,
        )

    def _first_valid_trial(self, dmd0: int) -> Optional[int]:
        """Return 0-based trial index of first valid trial for this DMD, else None."""
        v = np.argwhere(self.keep_trials[dmd0])
        if v.size == 0:
            return None
        return int(v[0, 0])

    def _ref_trial_shape_user_rois(
        self,
        dmd: int,
        trace_type: str = "Fsvd",
        roi_inds: Optional[Sequence[int]] = None,
    ) -> Tuple[int, int]:
        """
        Determine (n_rois, n_time) from the first *readable* valid trial.

        This is more defensive than the original implementation:
        - it iterates over valid trials until it finds a usable user-ROI trace matrix
        - it skips malformed/missing user-ROI exports
        - it raises a clear error if no readable manual soma ROI traces exist
        """
        dmd0 = dmd - 1
        keep = np.asarray(self.keep_trials[dmd0], dtype=bool)
        valid_trials = np.flatnonzero(keep)

        if valid_trials.size == 0:
            raise ValueError(f"No valid trials found for dmd={dmd}")

        last_err: Optional[Exception] = None

        for t0 in valid_trials:
            try:
                x = self.get_user_roi_traces(
                    dmd=dmd,
                    trial=int(t0) + 1,
                    trace_type=trace_type,
                    roi_inds=roi_inds,
                )
                x = np.asarray(x)

                if x.ndim == 3:
                    n_rois = x.shape[0]
                    n_time = x.shape[2]
                    if n_rois > 0 and n_time > 0:
                        return int(n_rois), int(n_time)

                elif x.ndim == 2:
                    n_rois = x.shape[0]
                    n_time = x.shape[1]
                    if n_rois > 0 and n_time > 0:
                        return int(n_rois), int(n_time)

            except Exception as e:
                last_err = e
                continue

        msg = (
            f"No readable user ROI trace matrices found for dmd={dmd}. "
            f"This session may lack exported manual soma ROI traces in SummaryLoCo."
        )
        if last_err is not None:
            msg += f" Last error: {repr(last_err)}"
        raise ValueError(msg)

    def get_processed_soma_ca_all_trials(
        self,
        dmd: int = 1,
        trace_type: str = "Fsvd",
        roi_inds: Optional[Sequence[int]] = None,
        fs_hz: Optional[float] = None,
        pad_to: Literal["ref", "max_valid", "none"] = "ref",
        include_invalid: bool = True,
        *,
        motion_correct: bool = True,
        motion_ridge: float = 1e-1,
        motion_use_fields: Sequence[str] = ("onlineXshift", "onlineYshift", "onlineZshift", "motionDSr", "motionDSc"),
        motion_step_thresh_z: float = 3.0,
        use_glu_as_motion_regressor: bool = False,
        glu_motion_hp_window_s: float = 0.5,
        eps: float = 1e-6,
        motion_regress_on: Literal["dF", "F"] = "dF",
        baseline_method: Literal["hull", "percentile"] = "percentile",
        denoise_window_s: float = 2.0,
        hull_window_s: float = 90.0,
        f0_floor_frac: float = 0.15,
        baseline_window_s: float = 20.0,
        baseline_q: float = 15.0,
        baseline_smooth_s: float = 2.0,
        max_session_minutes: Optional[float] = None,
        **proc_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process soma ROI Ca traces across all trials while preserving the existing
        return contract used by QC and data extraction.

        Returns arrays shaped (n_trials, n_rois, Tpad) when pad_to != "none".
        When `max_session_minutes` is provided, later samples are left as NaN once
        the cumulative kept duration exceeds that cutoff.
        """
        if fs_hz is None:
            fs_hz = float(self.metadata.get("analyzeHz", np.nan))
        if not np.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError("fs_hz must be provided or present in metadata['analyzeHz'].")

        n_rois_ref, T_ref = self._ref_trial_shape_user_rois(dmd=dmd, trace_type=trace_type, roi_inds=roi_inds)

        if pad_to == "ref":
            Tpad = T_ref
        elif pad_to == "max_valid":
            Ts: List[int] = []
            for tr in range(1, self.n_trials + 1):
                if self.keep_trials[dmd - 1, tr - 1]:
                    x = self.get_user_roi_traces(dmd=dmd, trial=tr, trace_type=trace_type, roi_inds=roi_inds)
                    T = x.shape[2] if x.ndim == 3 else x.shape[1]
                    Ts.append(int(T))
            Tpad = max(Ts) if Ts else T_ref
        elif pad_to == "none":
            Tpad = -1
        else:
            raise ValueError(f"pad_to must be 'ref', 'max_valid', or 'none'. Got: {pad_to}")

        if pad_to == "none":
            ca_clean_list: List[Optional[np.ndarray]] = [None] * self.n_trials
            ca_unmixed_list: List[Optional[np.ndarray]] = [None] * self.n_trials
            ca_mc_list: List[Optional[np.ndarray]] = [None] * self.n_trials
            baseline_list: List[Optional[np.ndarray]] = [None] * self.n_trials
            dff_list: List[Optional[np.ndarray]] = [None] * self.n_trials
            beta_unmix = np.full((self.n_trials, n_rois_ref), np.nan, dtype=float)
        else:
            ca_clean = np.full((self.n_trials, n_rois_ref, Tpad), np.nan, dtype=float)
            ca_unmixed = np.full((self.n_trials, n_rois_ref, Tpad), np.nan, dtype=float)
            ca_mc = np.full((self.n_trials, n_rois_ref, Tpad), np.nan, dtype=float)
            baseline = np.full((self.n_trials, n_rois_ref, Tpad), np.nan, dtype=float)
            dff = np.full((self.n_trials, n_rois_ref, Tpad), np.nan, dtype=float)
            beta_unmix = np.full((self.n_trials, n_rois_ref), np.nan, dtype=float)

        beta_motion: List[Optional[np.ndarray]] = [None] * self.n_trials
        motion_names: List[Optional[List[str]]] = [None] * self.n_trials

        remaining_samples: Optional[int]
        if max_session_minutes is None:
            remaining_samples = None
        else:
            remaining_samples = max(0, int(round(float(max_session_minutes) * 60.0 * fs_hz)))

        for tr in range(1, self.n_trials + 1):
            valid = bool(self.keep_trials[dmd - 1, tr - 1])
            if not valid:
                if not include_invalid:
                    if pad_to != "none":
                        raise ValueError("include_invalid=False requires pad_to='none' (so invalid can be None).")
                    continue
                continue

            glu, ca = self.get_soma_glu_ca_traces(dmd=dmd, trial=tr, trace_type=trace_type, roi_inds=roi_inds)
            if glu.shape[0] != n_rois_ref:
                n_use = min(glu.shape[0], n_rois_ref)
                glu = glu[:n_use]
                ca = ca[:n_use]

            Ttr = ca.shape[1]
            Xmot, mot_names = self._get_motion_regressors(
                dmd=dmd,
                trial=tr,
                target_len=Ttr,
                use_fields=motion_use_fields,
                motion_step_thresh_z=motion_step_thresh_z,
            ) if motion_correct else (None, None)

            out = self._process_soma_ca_trial(
                ca=ca,
                glu=glu,
                fs_hz=fs_hz,
                X_motion=Xmot,
                motion_names=mot_names,
                motion_correct=motion_correct,
                motion_ridge=motion_ridge,
                use_glu_as_motion_regressor=use_glu_as_motion_regressor,
                glu_motion_hp_window_s=glu_motion_hp_window_s,
                motion_regress_on=motion_regress_on,
                compute_dff=True,
                baseline_method=baseline_method,
                denoise_window_s=denoise_window_s,
                hull_window_s=hull_window_s,
                f0_floor_frac=f0_floor_frac,
                baseline_window_s=baseline_window_s,
                baseline_q=baseline_q,
                baseline_smooth_s=baseline_smooth_s,
                eps=eps,
                **proc_kwargs,
            )

            ca_clean_tr = out["ca_clean"]
            ca_unmixed_tr = out["ca_unmixed"]
            ca_mc_tr = out["ca_mc"]
            dff_res = out.get("dff", None)
            baseline_tr = dff_res.baseline if dff_res is not None else None
            dff_tr = dff_res.dff if dff_res is not None else None

            beta_tr = out["unmix"].beta if out.get("unmix", None) is not None else None
            if beta_tr is not None:
                beta_unmix[tr - 1, : beta_tr.shape[0]] = beta_tr
            beta_motion[tr - 1] = out.get("beta_motion", None)
            motion_names[tr - 1] = list(out.get("motion_names", [])) if out.get("motion_names", None) is not None else None

            if remaining_samples is not None:
                keep_this_trial = max(0, min(Ttr, remaining_samples))
                remaining_samples -= keep_this_trial
                if keep_this_trial < Ttr:
                    for arr in (ca_clean_tr, ca_unmixed_tr, ca_mc_tr):
                        arr[:, keep_this_trial:] = np.nan
                    if baseline_tr is not None:
                        baseline_tr[:, keep_this_trial:] = np.nan
                    if dff_tr is not None:
                        dff_tr[:, keep_this_trial:] = np.nan

            if pad_to == "none":
                ca_clean_list[tr - 1] = ca_clean_tr
                ca_unmixed_list[tr - 1] = ca_unmixed_tr
                ca_mc_list[tr - 1] = ca_mc_tr
                baseline_list[tr - 1] = baseline_tr
                dff_list[tr - 1] = dff_tr
            else:
                tcopy = min(Ttr, Tpad)
                rcopy = min(ca_clean_tr.shape[0], n_rois_ref)
                ca_clean[tr - 1, :rcopy, :tcopy] = ca_clean_tr[:rcopy, :tcopy]
                ca_unmixed[tr - 1, :rcopy, :tcopy] = ca_unmixed_tr[:rcopy, :tcopy]
                ca_mc[tr - 1, :rcopy, :tcopy] = ca_mc_tr[:rcopy, :tcopy]
                if baseline_tr is not None:
                    baseline[tr - 1, :rcopy, :tcopy] = baseline_tr[:rcopy, :tcopy]
                if dff_tr is not None:
                    dff[tr - 1, :rcopy, :tcopy] = dff_tr[:rcopy, :tcopy]

        if pad_to == "none":
            return {
                "ca_clean": ca_clean_list,
                "ca_unmixed": ca_unmixed_list,
                "ca_mc": ca_mc_list,
                "baseline": baseline_list,
                "dff": dff_list,
                "beta_unmix": beta_unmix,
                "beta_motion": beta_motion,
                "motion_names": motion_names,
                "session_sample_stop": None if max_session_minutes is None else max(0, int(round(float(max_session_minutes) * 60.0 * fs_hz))),
            }

        return {
            "ca_clean": ca_clean,
            "ca_unmixed": ca_unmixed,
            "ca_mc": ca_mc,
            "baseline": baseline,
            "dff": dff,
            "beta_unmix": beta_unmix,
            "beta_motion": beta_motion,
            "motion_names": motion_names,
            "session_sample_stop": None if max_session_minutes is None else max(0, int(round(float(max_session_minutes) * 60.0 * fs_hz))),
        }

#---------------- Motion regression --------------------------------------

def regress_out_(
    y: np.ndarray,
    X: np.ndarray,
    ridge: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    y: (T,) signal
    X: (T, P) nuisance regressors
    Returns:
      y_resid: (T,)
      beta: (P+1,) including intercept
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.size == 0:
        return y.copy(), np.zeros((0,), float)

    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if finite.sum() < max(10, X.shape[1] + 2):
        return y.copy(), np.zeros((X.shape[1] + 1,), float) * np.nan

    Xf = X[finite]
    yf = y[finite]

    # add intercept
    A = np.column_stack([Xf, np.ones(Xf.shape[0])])
    # ridge on regressors only, not intercept
    ATA = A.T @ A
    ATA[:-1, :-1] += ridge * np.eye(A.shape[1] - 1)
    ATy = A.T @ yf
    beta = np.linalg.solve(ATA, ATy)

    yhat = (np.column_stack([X, np.ones(X.shape[0])]) @ beta)
    y_resid = y.copy()
    y_resid[finite] = y[finite] - yhat[finite]
    return y_resid, beta

def add_lags(X: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Return [X(t-l), ...] concatenated. Pads with edge values."""
    T, P = X.shape
    outs = []
    for l in lags:
        if l == 0:
            outs.append(X)
        elif l > 0:
            pad = np.repeat(X[:1], l, axis=0)
            outs.append(np.vstack([pad, X[:-l]]))
        else:
            l2 = -l
            pad = np.repeat(X[-1:], l2, axis=0)
            outs.append(np.vstack([X[l2:], pad]))
    return np.concatenate(outs, axis=1)

