from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vip-slap2-analysis.io.matv73 import MatV73


@dataclass
class VoltageSummary:
    """
    Reader for summarize_Voltage output:
      ExperimentSummary/Summary_Voltage-*.mat  (saved with variable 'summary')
    """
    mat_path: Path
    swap_xy: bool = True  # mimic common "imshow-ready" convention if desired

    def __post_init__(self) -> None:
        self._mat = MatV73(self.mat_path)
        root = self._mat.read_var("summary")
        if not isinstance(root, dict):
            raise TypeError("Expected 'summary' to be a struct-like dict.")
        self._root: Dict[str, Any] = root

        # Infer sizes
        self._E = self._root.get("E", None)
        if self._E is None:
            raise KeyError("Missing summary.E in MAT file.")
        # E is cell-like nested list [trial][dmd] in our reader
        self.n_trials = len(self._E)
        self.n_dmds = len(self._E[0]) if self.n_trials > 0 else 0

    @property
    def params(self) -> Dict[str, Any]:
        return self._root.get("params", {})

    @property
    def trial_table(self) -> Dict[str, Any]:
        return self._root.get("trialTable", {})

    @property
    def analyze_hz(self) -> float:
        p = self.params
        # summarize_Voltage sets params.analyzeHz
        return float(p.get("analyzeHz", np.nan))

    def timebase(self, trial: int, dmd: int) -> np.ndarray:
        """
        Returns timebase in seconds assuming uniform sampling at analyzeHz.
        Starts at 0 because absolute start requires linerateHz + firstLine.
        """
        y = self.roi_traces(trial=trial, dmd=dmd)
        n = y.shape[-1]
        hz = self.analyze_hz
        if not np.isfinite(hz) or hz <= 0:
            return np.arange(n, dtype=float)
        return np.arange(n, dtype=float) / hz

    def _get_E(self, trial: int, dmd: int) -> Dict[str, Any]:
        """
        trial/dmd are 1-indexed for user-facing API.
        """
        t = trial - 1
        d = dmd - 1
        if t < 0 or t >= self.n_trials:
            raise IndexError(f"trial={trial} out of range [1..{self.n_trials}]")
        if d < 0 or d >= self.n_dmds:
            raise IndexError(f"dmd={dmd} out of range [1..{self.n_dmds}]")
        E = self._E[t][d]
        if E is None:
            raise ValueError(f"No data for trial={trial}, dmd={dmd}")
        if not isinstance(E, dict):
            raise TypeError(f"Expected struct-like dict at E[{trial},{dmd}]")
        return E

    def roi_traces(self, trial: int, dmd: int) -> np.ndarray:
        """
        Returns ROI traces as (n_rois, n_time).
        MATLAB: E.ROIs.F is (nSamps, nROIs)
        """
        E = self._get_E(trial, dmd)
        rois = E.get("ROIs", {})
        F = rois.get("F", None)
        if F is None:
            raise KeyError("Missing E.ROIs.F")
        F = np.asarray(F)
        if F.ndim != 2:
            raise ValueError(f"Unexpected E.ROIs.F shape: {F.shape}")
        return F.T  # (n_rois, n_time)

    def roi_weights(self, trial: int, dmd: int) -> np.ndarray:
        """
        Returns ROI weights as (n_rois, n_time)
        """
        E = self._get_E(trial, dmd)
        rois = E.get("ROIs", {})
        W = rois.get("weight", None)
        if W is None:
            raise KeyError("Missing E.ROIs.weight")
        W = np.asarray(W)
        return W.T

    def global_trace(self, trial: int, dmd: int) -> np.ndarray:
        """
        Returns global weighted trace as (n_time,)
        """
        E = self._get_E(trial, dmd)
        g = E.get("global", {})
        F = g.get("F", None)
        if F is None:
            raise KeyError("Missing E.global.F")
        return np.asarray(F).squeeze()

    def discard_frames(self, trial: int, dmd: int) -> np.ndarray:
        E = self._get_E(trial, dmd)
        df = E.get("discardFrames", None)
        if df is None:
            raise KeyError("Missing E.discardFrames")
        return np.asarray(df).astype(bool).squeeze()

    def motion(self, trial: int, dmd: int) -> Dict[str, np.ndarray]:
        """
        Returns upsampled motion dict: online shifts, recNegErr, etc.
        """
        E = self._get_E(trial, dmd)
        um = E.get("upsampledMotion", {})
        if not isinstance(um, dict):
            raise TypeError("Expected E.upsampledMotion to be a struct-like dict")
        return {k: np.asarray(v).squeeze() for k, v in um.items()}

    def ref_plane(self, dmd: int) -> np.ndarray:
        """
        summary.refPlane is a cell {dmd} of a 3D image.
        summarize_Voltage permutes it as (x, y, ch) in MATLAB.
        For imshow-ready arrays, swap to (y, x, ch) if swap_xy=True.
        """
        ref_cell = self._root.get("refPlane", None)
        if ref_cell is None:
            raise KeyError("Missing summary.refPlane")
        im = np.asarray(ref_cell[dmd - 1][0])  # our cell reader returns [row][col]
        if im.ndim == 2:
            return im.T if self.swap_xy else im
        if im.ndim == 3:
            return np.swapaxes(im, 0, 1) if self.swap_xy else im
        raise ValueError(f"Unexpected refPlane dims: {im.shape}")

    def roi_masks(self, dmd: int) -> np.ndarray:
        """
        Returns masks as (n_rois, y, x) for convenience.
        MATLAB: summary.masks{dmd} is (rows=y, cols=x, n_rois)
        """
        masks_cell = self._root.get("masks", None)
        if masks_cell is None:
            raise KeyError("Missing summary.masks")
        m = np.asarray(masks_cell[dmd - 1][0])
        if m.ndim != 3:
            raise ValueError(f"Unexpected masks dims: {m.shape}")
        # (y, x, n_rois) -> (n_rois, y, x)
        m = np.moveaxis(m, -1, 0)
        if self.swap_xy:
            m = np.swapaxes(m, -2, -1)  # (n_rois, x, y) -> (n_rois, y, x)
        return m.astype(bool)

    def superpixel_footprints(self, dmd: int) -> Optional[np.ndarray]:
        """
        Returns superpixel footprints (n_sp, y, x) if present.
        MATLAB: (rows, cols, n_sp)
        """
        fp_cell = self._root.get("footprints", None)
        if fp_cell is None:
            return None
        fp = fp_cell[dmd - 1][0]
        if fp is None:
            return None
        fp = np.asarray(fp)
        if fp.ndim != 3:
            return None
        fp = np.moveaxis(fp, -1, 0)  # (n_sp, y, x) if swap_xy=False
        if self.swap_xy:
            fp = np.swapaxes(fp, -2, -1)
        return fp
