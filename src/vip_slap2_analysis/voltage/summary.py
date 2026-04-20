# vip_slap2_analysis/voltage/summary.py
"""Read MATLAB voltage-summary files and expose ROI traces lazily.

This module provides :class:`VoltageSummary`, a lightweight HDF5/MATLAB v7.3
reader for voltage-imaging summary files. It supports both the original
per-DMD/per-trial ``summary/E`` layout and newer flat session-wide trace exports.
The public API uses 1-indexed DMD and trial arguments to stay consistent with the
MATLAB pipeline while returning NumPy arrays suitable for downstream Python
postprocessing and QC.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, List

import numpy as np
import h5py

from vip_slap2_analysis.io.matv73 import MatV73File


@dataclass
class VoltageSummary:
    """
    Lazy loader for summarize_Voltage MAT files.

    Supports two layouts:

    1) Event/trial layout
       summary/E -> per-(dmd, trial) groups containing ROIs/F, discardFrames, etc.

    2) Flat-traces layout
       summary/traces -> single session-wide traces matrix
       summary/nAnalysisROIs -> number of ROIs per DMD
       summary/masks -> per-DMD ROI masks
       summary/refIM -> per-DMD reference image

    Public API remains 1-indexed for dmd/trial, matching MATLAB conventions.
    For flat-traces files, only trial=1 is valid.
    """

    file_path: Union[str, Path]
    keep_open: bool = True
    swap_xy_images: bool = True  # convenience for ref images / masks display orientation

    def __post_init__(self) -> None:
        """Open the MAT file and infer the supported voltage-summary layout.

        The constructor initializes shared session metadata such as number of
        DMDs, trials, samples, ROI counts, valid-trial masks, and ROI offsets.
        It does not eagerly load large trace arrays; those remain HDF5-backed
        until requested by accessor methods.
        """
        self.file_path = Path(self.file_path)
        self._mat = MatV73File(self.file_path, keep_open=self.keep_open)

        if "summary" not in self._mat.f:
            raise KeyError(f"Top-level variable 'summary' not found. Keys: {list(self._mat.f.keys())}")

        self.n_trials: int = 0
        self.n_dmds: int = 0
        self.n_samples: int = 0
        self.keep_trials: np.ndarray
        self.valid_trials: List[List[int]] = []
        self.n_rois: List[int] = []
        self._E_layout: str = "dmd_trial"   # only used for event/trial files
        self._summary_layout: str = "unknown"  # "event_trial" or "flat_traces"
        self._trace_axis: Optional[str] = None  # "rois_samples" or "samples_rois"
        self._roi_offsets: List[int] = []

        self._get_info()

    # ----------------- lifecycle -----------------

    def close(self) -> None:
        """Close the underlying HDF5 file handle."""
        self._mat.close()

    # ----------------- structure inference -----------------

    @staticmethod
    def _is_ref_dataset(node: object) -> bool:
        """Return True when ``node`` stores MATLAB-style HDF5 references."""
        return (
            isinstance(node, h5py.Dataset)
            and (node.dtype == h5py.ref_dtype or getattr(node.dtype, "kind", None) == "O")
        )

    def _get_info(self) -> None:
        """Detect summary layout and populate object-level shape metadata."""
        summary = self._mat.f["summary"]

        if "E" in summary:
            self._summary_layout = "event_trial"
            self._init_from_E(summary)
            return

        if "traces" in summary and "nAnalysisROIs" in summary:
            self._summary_layout = "flat_traces"
            self._init_from_flat_summary(summary)
            return

        raise KeyError(
            "Could not find a supported voltage-summary layout under 'summary'. "
            f"Available keys: {list(summary.keys())}"
        )

    def _init_from_E(self, summary: h5py.Group) -> None:
        """
        Original summarize_Voltage layout:
          summary/E -> 2D ref array of per-(dmd, trial) groups
        """
        E = summary["E"]

        if len(E.shape) != 2:
            raise ValueError(f"Unexpected shape for summary/E: {E.shape}")

        s0, s1 = E.shape

        # Heuristic: DMD count is small (<=4 typically), trials larger.
        if s0 <= 4 and s1 > s0:
            self._E_layout = "dmd_trial"
            self.n_dmds, self.n_trials = int(s0), int(s1)
        elif s1 <= 4 and s0 > s1:
            self._E_layout = "trial_dmd"
            self.n_trials, self.n_dmds = int(s0), int(s1)
        else:
            self._E_layout = "dmd_trial"
            self.n_dmds, self.n_trials = int(s0), int(s1)

        self.keep_trials = np.full((self.n_dmds, self.n_trials), False)

        # Mark valid trials by checking dereferenceability
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

        # Determine ROI count from first valid trial for each DMD
        self.n_rois = [0 for _ in range(self.n_dmds)]
        for dmd0 in range(self.n_dmds):
            idx = np.argwhere(self.keep_trials[dmd0])
            if idx.size == 0:
                continue
            trial0 = int(idx[0, 0])
            try:
                g = self._E_group(dmd0, trial0)
                F = g["ROIs"]["F"]  # expected: (n_samples, n_rois)
                if len(F.shape) != 2:
                    self.n_rois[dmd0] = 0
                else:
                    self.n_rois[dmd0] = int(F.shape[1])  # fixed: ROI dimension, not sample dimension
            except Exception:
                self.n_rois[dmd0] = 0

        self.valid_trials = [
            list(1 + np.argwhere(self.keep_trials[dmd0])[:, 0])
            for dmd0 in range(self.n_dmds)
        ]

        # Infer sample count from first valid trial
        for dmd0 in range(self.n_dmds):
            idx = np.argwhere(self.keep_trials[dmd0])
            if idx.size == 0:
                continue
            trial0 = int(idx[0, 0])
            try:
                g = self._E_group(dmd0, trial0)
                self.n_samples = int(g["ROIs"]["F"].shape[0])
                break
            except Exception:
                pass

    def _init_from_flat_summary(self, summary: h5py.Group) -> None:
        """
        Flat session-wide layout:
          summary/traces shape is either (n_total_rois, n_samples) or (n_samples, n_total_rois)
          summary/nAnalysisROIs gives ROI count per DMD
        """
        traces = summary["traces"]
        if len(traces.shape) != 2:
            raise ValueError(f"Unexpected shape for summary/traces: {traces.shape}")

        nrois_raw = np.asarray(summary["nAnalysisROIs"][()]).astype(int).squeeze()
        if nrois_raw.ndim == 0:
            nrois_raw = np.array([int(nrois_raw)])
        self.n_rois = [int(x) for x in np.ravel(nrois_raw).tolist()]
        self.n_dmds = len(self.n_rois)
        total_rois = int(np.sum(self.n_rois))

        s0, s1 = map(int, traces.shape)
        if s0 == total_rois:
            self._trace_axis = "rois_samples"
            self.n_samples = s1
        elif s1 == total_rois:
            self._trace_axis = "samples_rois"
            self.n_samples = s0
        else:
            raise ValueError(
                "Could not reconcile summary/traces with summary/nAnalysisROIs. "
                f"traces shape={traces.shape}, nAnalysisROIs={self.n_rois}"
            )

        self.n_trials = 1
        self.keep_trials = np.ones((self.n_dmds, 1), dtype=bool)
        self.valid_trials = [[1] for _ in range(self.n_dmds)]
        self._roi_offsets = [0] + list(np.cumsum(self.n_rois))

    # ----------------- event/trial helpers -----------------

    def _E_ref(self, dmd0: int, trial0: int) -> Optional[h5py.Reference]:
        """Return the HDF5 reference for a zero-indexed DMD/trial pair."""
        if self._summary_layout != "event_trial":
            raise ValueError("summary/E is not present in this file; this is a flat-traces voltage summary.")

        E = self._mat.f["summary"]["E"]

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
        """Dereference a zero-indexed DMD/trial entry from ``summary/E``."""
        ref = self._E_ref(dmd0, trial0)
        if ref is None:
            raise ValueError(f"No E ref for dmd={dmd0+1}, trial={trial0+1}")

        node = self._mat.deref(ref)
        if not isinstance(node, h5py.Group):
            raise TypeError("E ref did not dereference to a Group")
        return node

    # ----------------- params -----------------

    def analyze_hz(self) -> float:
        """
        summary/params/analyzeHz if present, otherwise NaN.
        """
        try:
            hz = self._mat.f["summary"]["params"]["analyzeHz"][()]
            return float(np.array(hz).squeeze())
        except Exception:
            return float("nan")

    # ----------------- helpers -----------------

    @staticmethod
    def _align_bool_mask(mask: np.ndarray, n: int) -> np.ndarray:
        """
        Ensure mask is 1D length n.
        If longer: truncate. If shorter: pad False.
        """
        mask = np.asarray(mask).astype(bool).squeeze()
        if mask.ndim != 1:
            mask = mask.reshape(-1)

        if mask.size == n:
            return mask
        if mask.size > n:
            return mask[:n]
        out = np.zeros(n, dtype=bool)
        out[: mask.size] = mask
        return out

    def _require_trial1_for_flat_summary(self, trial: int) -> None:
        """Validate flat-trace access, where the whole session is exposed as trial 1."""
        if self._summary_layout == "flat_traces" and trial != 1:
            raise ValueError(
                "This voltage summary has no per-trial E structure. "
                "Use trial=1 for the flattened session-wide traces."
            )

    def _flat_roi_row_inds(self, dmd: int, roi_inds: Optional[Sequence[int]]) -> np.ndarray:
        """Map DMD-local ROI indices to rows in the flat session-wide trace matrix."""
        dmd0 = dmd - 1
        if dmd0 < 0 or dmd0 >= self.n_dmds:
            raise IndexError(f"dmd must be in [1, {self.n_dmds}]")

        start = self._roi_offsets[dmd0]
        stop = self._roi_offsets[dmd0 + 1]

        if roi_inds is None:
            return np.arange(start, stop, dtype=int)

        roi_inds = np.asarray(list(roi_inds), dtype=int)
        if np.any(roi_inds < 0) or np.any(roi_inds >= self.n_rois[dmd0]):
            raise IndexError(f"roi_inds out of range for dmd {dmd}")
        return start + roi_inds

    # ----------------- traces -----------------

    def get_roi_traces(
        self,
        dmd: int,
        trial: int,
        roi_inds: Optional[Sequence[int]] = None,
        t_slice: Optional[slice] = None,
        drop_discarded: bool = False,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Returns ROI traces with shape (n_samples, n_rois).

        For flat-traces summaries, only trial=1 is valid.
        """
        if t_slice is None:
            t_slice = slice(None)

        if self._summary_layout == "flat_traces":
            self._require_trial1_for_flat_summary(trial)
            traces = self._mat.f["summary"]["traces"]
            row_inds = self._flat_roi_row_inds(dmd, roi_inds)

            if self._trace_axis == "rois_samples":
                x = np.asarray(traces[row_inds, t_slice]).T
            else:
                x = np.asarray(traces[t_slice, row_inds])

            x = np.atleast_2d(x)
            if x.shape[0] != self.n_samples and x.shape[1] == self.n_samples:
                x = x.T

            if drop_discarded:
                df = self.get_discard_frames(dmd=dmd, trial=trial)
                df = self._align_bool_mask(df, x.shape[0])
                x = x[~df, :]

            if dtype is not None:
                x = x.astype(dtype, copy=False)
            return x

        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        F = g["ROIs"]["F"]  # (n_samples, n_rois)

        if roi_inds is None:
            x = np.asarray(F[t_slice, :])
        else:
            roi_inds = list(roi_inds)
            x = np.asarray(F[t_slice, roi_inds])

        if drop_discarded:
            df = self.get_discard_frames(dmd=dmd, trial=trial)
            df = self._align_bool_mask(df, x.shape[0])
            x = x[~df, :]

        if dtype is not None:
            x = x.astype(dtype, copy=False)

        return x

    def get_roi_weights(
        self,
        dmd: int,
        trial: int,
        roi_inds: Optional[Sequence[int]] = None,
        t_slice: Optional[slice] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Load ROI weights from E/ROIs/weight.

        Returns shape: (n_samples, n_rois)
        """
        if self._summary_layout == "flat_traces":
            raise KeyError("summary/ROIs/weight is not present in flat-traces voltage summaries.")

        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        W = g["ROIs"]["weight"]

        if t_slice is None:
            t_slice = slice(None)

        if roi_inds is None:
            x = np.asarray(W[t_slice, :])
        else:
            roi_inds = list(roi_inds)
            x = np.asarray(W[t_slice, roi_inds])

        if dtype is not None:
            x = x.astype(dtype, copy=False)

        return x

    def get_global_trace(
        self,
        dmd: int,
        trial: int,
        t_slice: Optional[slice] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """
        Load global weighted trace: E/global/F

        Returns shape: (n_samples,)
        """
        if self._summary_layout == "flat_traces":
            raise KeyError("summary/global/F is not present in flat-traces voltage summaries.")

        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        F = g["global"]["F"]

        if t_slice is None:
            t_slice = slice(None)

        x = np.asarray(F[t_slice]).squeeze()
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    def get_discard_frames(self, dmd: int, trial: int) -> np.ndarray:
        """
        Load discardFrames if present.

        For flat-traces summaries, returns all-False since discardFrames are not stored.
        """
        if self._summary_layout == "flat_traces":
            self._require_trial1_for_flat_summary(trial)
            return np.zeros(self.n_samples, dtype=bool)

        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        df = g["discardFrames"]
        return np.asarray(df[()]).astype(bool).squeeze()

    def get_motion(
        self,
        dmd: int,
        trial: int,
        keys: Optional[Sequence[str]] = None,
        t_slice: Optional[slice] = None,
        dtype: Optional[np.dtype] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load upsampled motion: E/upsampledMotion/<field>

        For flat-traces summaries, returns {}.
        """
        if self._summary_layout == "flat_traces":
            return {}

        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        um = g["upsampledMotion"]

        if keys is None:
            keys = list(um.keys())

        if t_slice is None:
            t_slice = slice(None)

        out: Dict[str, np.ndarray] = {}
        for k in keys:
            if k not in um:
                continue
            arr = np.asarray(um[k][t_slice]).squeeze()
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            out[k] = arr
        return out

    # ----------------- cell-array backed images -----------------

    def _cell_item(self, cell_name: str, dmd0: int) -> Optional[Union[h5py.Dataset, h5py.Group]]:
        """
        summary/<cell_name> is often a MATLAB cell array stored as an HDF5 ref dataset.
        Dereference element {dmd}.
        """
        summary = self._mat.f["summary"]
        if cell_name not in summary:
            return None

        cell = summary[cell_name]
        if not self._is_ref_dataset(cell):
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

    def get_ref_plane(self, dmd: int) -> np.ndarray:
        """
        Supports either summary/refPlane{dmd} or summary/refIM{dmd}.
        """
        dmd0 = dmd - 1
        node = self._cell_item("refPlane", dmd0)
        if node is None:
            node = self._cell_item("refIM", dmd0)

        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"summary/refPlane{{{dmd}}} or summary/refIM{{{dmd}}} not found or unexpected type")

        arr = np.asarray(node[()])
        if self.swap_xy_images and arr.ndim >= 2:
            arr = np.swapaxes(arr, 0, 1)
        return arr

    def get_roi_masks(self, dmd: int) -> np.ndarray:
        """
        Returns masks as (y, x, n_rois) boolean.

        Handles either:
          - (y, x, n_rois)
          - (n_rois, y, x)
        """
        dmd0 = dmd - 1
        node = self._cell_item("masks", dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"summary/masks{{{dmd}}} not found or unexpected type")

        m = np.asarray(node[()])
        expected = self.n_rois[dmd0] if dmd0 < len(self.n_rois) else None

        if m.ndim == 3 and expected is not None:
            if m.shape[0] == expected:
                m = np.moveaxis(m, 0, -1)  # (n_rois, y, x) -> (y, x, n_rois)
            elif m.shape[-1] == expected:
                pass

        if self.swap_xy_images and m.ndim >= 2:
            m = np.swapaxes(m, 0, 1)
        return m.astype(bool)

    def get_user_roi_label_image(self, dmd: int) -> np.ndarray:
        """
        summary/userROIs{dmd} label image if present.
        Otherwise reconstruct a label image from masks.
        """
        dmd0 = dmd - 1
        node = self._cell_item("userROIs", dmd0)
        if node is not None and isinstance(node, h5py.Dataset):
            img = np.asarray(node[()])
            if self.swap_xy_images and img.ndim >= 2:
                img = np.swapaxes(img, 0, 1)
            return img

        masks = self.get_roi_masks(dmd)
        label_img = np.zeros(masks.shape[:2], dtype=np.int32)
        for i in range(masks.shape[2]):
            label_img[masks[:, :, i]] = i + 1
        return label_img

    # ----------------- convenience -----------------

    def timebase(self, dmd: int, trial: int) -> np.ndarray:
        """
        Uniform timebase in seconds based on analyzeHz when available.
        If analyzeHz is missing, returns sample indices.
        """
        if self._summary_layout == "flat_traces":
            self._require_trial1_for_flat_summary(trial)
            n = self.n_samples
        else:
            dmd0, trial0 = dmd - 1, trial - 1
            g = self._E_group(dmd0, trial0)
            n = int(g["ROIs"]["F"].shape[0])

        hz = self.analyze_hz()
        if not np.isfinite(hz) or hz <= 0:
            return np.arange(n, dtype=float)
        return np.arange(n, dtype=float) / hz
