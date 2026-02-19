# vip_slap2_analysis/glutamate/summary.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import h5py

from vip_slap2_analysis.io.matv73 import MatV73File, bytes_to_str


ChannelSpec = Union[None, int, str, Sequence[Union[int, str]]]


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

        # channel axis: match selected channels length if provided, else match n_channels if possible
        ch_ax = None
        if channels is not None:
            for ax, s in enumerate(x.shape):
                if s == len(channels):
                    ch_ax = ax
                    break

        if ch_ax is None and n_channels > 1:
            for ax, s in enumerate(x.shape):
                if s == n_channels:
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
        if ndim == 3:
            if channels is not None:
                # after selection, channel axis might equal len(channels)
                for ax, s in enumerate(shape):
                    if s == len(channels):
                        ch_ax = ax
                        break
            if ch_ax is None and ch_axes:
                if time_ax is not None:
                    cand = [ax for ax in ch_axes if ax != time_ax]
                    ch_ax = cand[0] if cand else ch_axes[0]
                else:
                    ch_ax = ch_axes[0]
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

    # ----------------- user ROI traces (UPDATED FORMAT) -----------------

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
