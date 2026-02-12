# vip_slap2_analysis/voltage/summary.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import h5py

from vip_slap2_analysis.io.matv73 import MatV73File


@dataclass
class VoltageSummary:
    """
    Fast, lazy loader for summarize_Voltage output MAT files (~GB scale).

    ASSUMPTION (per your debug):
      - ROI traces are stored as samples x rois (n_samples, n_rois)
        i.e. E/ROIs/F has shape (n_samples, n_rois)
      - discardFrames is per-sample and should align to axis 0 of F

    Public API uses 1-indexed (dmd, trial), matching MATLAB conventions.
    """

    file_path: Union[str, Path]
    keep_open: bool = True
    swap_xy_images: bool = True  # convenience for refPlane/masks display orientation

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        self._mat = MatV73File(self.file_path, keep_open=self.keep_open)

        if "summary" not in self._mat.f:
            raise KeyError(f"Top-level variable 'summary' not found. Keys: {list(self._mat.f.keys())}")

        self.n_trials: int = 0
        self.n_dmds: int = 0
        self.keep_trials: np.ndarray
        self.valid_trials: List[List[int]] = []
        self.n_rois: List[int] = []
        self._E_layout: str = "dmd_trial"  # or "trial_dmd"

        self._get_info()

    # ----------------- lifecycle -----------------

    def close(self) -> None:
        self._mat.close()

    # ----------------- core structure inference -----------------

    def _get_info(self) -> None:
        """
        Quickly infer:
          - E layout and (n_dmds, n_trials)
          - keep_trials mask (dereferenceable E entries)
          - n_rois per dmd from first valid trial using F.shape[1]
        """
        f = self._mat.f
        E = f["summary"]["E"]

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
            # ambiguous; default to dmd_trial (matches old ExperimentSummary patterns)
            self._E_layout = "dmd_trial"
            self.n_dmds, self.n_trials = int(s0), int(s1)

        self.keep_trials = np.full((self.n_dmds, self.n_trials), False)

        # Mark valid trials by checking dereferenceability (fast; doesn't read big arrays)
        for dmd0 in range(self.n_dmds):
            for trial0 in range(self.n_trials):
                ref = self._E_ref(dmd0, trial0)
                if ref is None:
                    continue
                try:
                    node = self._mat.deref(ref)
                    if isinstance(node, h5py.Group):
                        # touch minimal metadata only
                        _ = list(node.keys())
                        self.keep_trials[dmd0, trial0] = True
                except Exception:
                    pass

        # Determine ROIs count from first valid trial for each dmd
        self.n_rois = [0 for _ in range(self.n_dmds)]
        for dmd0 in range(self.n_dmds):
            idx = np.argwhere(self.keep_trials[dmd0])
            if idx.size == 0:
                continue
            trial0 = int(idx[0, 0])
            try:
                g = self._E_group(dmd0, trial0)
                F = g["ROIs"]["F"]  # (n_samples, n_rois)
                if len(F.shape) != 2:
                    self.n_rois[dmd0] = 0
                else:
                    self.n_rois[dmd0] = int(F.shape[0])  # <-- ROI dimension
            except Exception:
                self.n_rois[dmd0] = 0

        self.valid_trials = [
            list(1 + np.argwhere(self.keep_trials[dmd0])[:, 0])
            for dmd0 in range(self.n_dmds)
        ]

    def _E_ref(self, dmd0: int, trial0: int) -> Optional[h5py.Reference]:
        """
        Return the object reference for E at (dmd0, trial0) (both 0-indexed),
        handling E layout.
        """
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
        summary/params/analyzeHz if present
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

    # ----------------- traces (samples x rois) -----------------

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
        Load ROI traces from E/ROIs/F.

        Returns shape: (n_samples, n_rois)

        Parameters
        ----------
        roi_inds : sequence[int] or None
            0-indexed ROI indices to select. If None, returns all ROIs.
        t_slice : slice or None
            Slice along samples axis (axis 0).
        drop_discarded : bool
            If True, drops samples where discardFrames is True.
        """
        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        F = g["ROIs"]["F"]  # (n_samples, n_rois)

        if t_slice is None:
            t_slice = slice(None)

        # Read only requested slice/columns (minimize I/O)
        if roi_inds is None:
            x = np.asarray(F[t_slice, :])
        else:
            roi_inds = list(roi_inds)
            x = np.asarray(F[t_slice, roi_inds])

        # Apply discard mask in the same sample space (axis 0)
        if drop_discarded:
            df = self.get_discard_frames(dmd=dmd, trial=trial)  # full length
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
        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        W = g["ROIs"]["weight"]  # (n_samples, n_rois)

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
        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        F = g["global"]["F"]  # (n_samples,) or (n_samples,1)

        if t_slice is None:
            t_slice = slice(None)

        x = np.asarray(F[t_slice]).squeeze()
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    def get_discard_frames(self, dmd: int, trial: int) -> np.ndarray:
        """
        Load discardFrames: E/discardFrames

        Returns shape: (n_samples,)
        """
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

        Returns dict[field] -> (n_samples,)
        """
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
        summary/<cell_name> is typically a MATLAB cell array stored as an HDF5 ref dataset.
        We dereference element {dmd}.
        """
        f = self._mat.f
        if cell_name not in f["summary"]:
            return None

        cell = f["summary"][cell_name]
        if not isinstance(cell, h5py.Dataset) or cell.dtype != h5py.ref_dtype:
            # not a ref-cell; return directly
            return cell

        # Try common layouts (dmd,0) or (0,dmd)
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
        summary/refPlane{dmd}
        """
        dmd0 = dmd - 1
        node = self._cell_item("refPlane", dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"summary/refPlane{{{dmd}}} not found or unexpected type")

        arr = np.asarray(node[()])
        if self.swap_xy_images and arr.ndim >= 2:
            arr = np.swapaxes(arr, 0, 1)
        return arr

    def get_roi_masks(self, dmd: int) -> np.ndarray:
        """
        summary/masks{dmd}

        MATLAB: (y, x, n_rois)
        Returns: (y, x, n_rois) boolean by default (native MATLAB order).
        If you prefer (n_rois, y, x), you can moveaxis externally.
        """
        dmd0 = dmd - 1
        node = self._cell_item("masks", dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"summary/masks{{{dmd}}} not found or unexpected type")

        m = np.asarray(node[()])  # (y, x, n_rois)
        if self.swap_xy_images and m.ndim >= 2:
            m = np.swapaxes(m, 0, 1)
        return m.astype(bool)

    def get_user_roi_label_image(self, dmd: int) -> np.ndarray:
        """
        summary/userROIs{dmd} label image (pixel -> ROI id)
        """
        dmd0 = dmd - 1
        node = self._cell_item("userROIs", dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"summary/userROIs{{{dmd}}} not found or unexpected type")

        img = np.asarray(node[()])
        if self.swap_xy_images and img.ndim >= 2:
            img = np.swapaxes(img, 0, 1)
        return img

    # ----------------- convenience -----------------

    def timebase(self, dmd: int, trial: int) -> np.ndarray:
        """
        Uniform timebase in seconds based on analyzeHz and the number of samples in E/ROIs/F.
        """
        dmd0, trial0 = dmd - 1, trial - 1
        g = self._E_group(dmd0, trial0)
        n = int(g["ROIs"]["F"].shape[0])

        hz = self.analyze_hz()
        if not np.isfinite(hz) or hz <= 0:
            return np.arange(n, dtype=float)
        return np.arange(n, dtype=float) / hz
