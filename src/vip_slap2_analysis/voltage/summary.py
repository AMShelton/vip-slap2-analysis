"""Lazy accessors for SLAP2 dendritic-voltage summary and trace outputs.

This module provides :class:`VoltageSummary`, a lightweight reader for voltage
imaging outputs created by the MATLAB ``extractDendrites_new.m`` pipeline and
older single-file voltage summaries.  The new pipeline stores lightweight
metadata in ``dendriticVoltageSummary-<timestamp>.mat`` and large traces in a
paired ``dendriticVoltageTraces-<timestamp>.h5`` file.  ``VoltageSummary`` accepts
the summary ``.mat`` path, discovers the paired trace ``.h5`` file when possible,
and exposes trial-sliced or continuous traces through one consistent Python API.

Public methods use 1-indexed ``dmd`` and ``trial`` arguments to match MATLAB and
other summary readers in ``vip_slap2_analysis``.  Trace arrays are returned as
``(n_samples, n_rois)`` unless otherwise noted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from vip_slap2_analysis.io.matv73 import MatV73File, bytes_to_str


PathLike = Union[str, Path]


@dataclass
class VoltageSummary:
    """Lazy loader for dendritic-voltage MATLAB summaries and HDF5 traces.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a voltage summary ``.mat`` file.  For new ``extractDendrites_new``
        outputs, this should be ``dendriticVoltageSummary-<timestamp>.mat``.
    trace_path : str or pathlib.Path, optional
        Explicit path to the paired trace ``.h5`` file.  If omitted, the class first
        checks ``summary.outputH5`` and then searches beside ``file_path`` for a file
        with the same timestamp, such as ``dendriticVoltageTraces-<timestamp>.h5``.
    keep_open : bool
        Keep HDF5 file handles open between calls.
    swap_xy_images : bool
        Swap the first two image/mask axes on read for display consistency with the
        glutamate summary reader.

    Notes
    -----
    Three layouts are supported:

    1. ``split_h5``: new separated metadata/traces format from
       ``extractDendrites_new.m``.
    2. ``flat_traces``: older single-file summaries with ``summary/traces``.
    3. ``event_trial``: older per-DMD/per-trial summaries with ``summary/E``.
    """

    file_path: PathLike
    trace_path: Optional[PathLike] = None
    keep_open: bool = True
    swap_xy_images: bool = True

    def __post_init__(self) -> None:
        """Open metadata/traces and infer the available storage layout."""
        self.file_path = Path(self.file_path)
        self._mat = MatV73File(self.file_path, keep_open=self.keep_open)

        if "summary" not in self._mat.f:
            raise KeyError(
                f"Top-level variable 'summary' not found. Keys: {list(self._mat.f.keys())}"
            )

        self.trace_path = Path(self.trace_path) if self.trace_path is not None else None
        self._h5: Optional[h5py.File] = None

        self.n_trials: int = 0
        self.n_dmds: int = 0
        self.n_samples: int = 0
        self.keep_trials: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.valid_trials: List[List[int]] = []
        self.n_rois: List[int] = []
        self.n_total_rois: int = 0
        self.roi_global_offsets: List[int] = []
        self.trial_epoch: Optional[np.ndarray] = None
        self.n_epochs: int = 0

        self._summary_layout: str = "unknown"
        self._trace_axis: Optional[str] = None
        self._E_layout: str = "dmd_trial"
        self._metadata: Optional[Dict[str, Any]] = None
        self._h5_attrs: Optional[Dict[str, Any]] = None

        self._get_info()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close open metadata and trace HDF5 handles."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._mat.close()

    def __enter__(self) -> "VoltageSummary":
        """Return ``self`` for context-manager use."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close file handles when leaving a context manager."""
        self.close()

    # ------------------------------------------------------------------
    # structure inference
    # ------------------------------------------------------------------

    @staticmethod
    def _is_ref_dataset(node: object) -> bool:
        """Return True when ``node`` stores MATLAB-style HDF5 references."""
        return (
            isinstance(node, h5py.Dataset)
            and (node.dtype == h5py.ref_dtype or getattr(node.dtype, "kind", None) == "O")
        )

    @staticmethod
    def _scalar(node: h5py.Dataset, default: Any = None) -> Any:
        """Read a scalar-like HDF5 dataset and decode simple MATLAB values."""
        try:
            val = bytes_to_str(node[()])
            if isinstance(val, np.ndarray):
                val = np.squeeze(val)
                if val.ndim == 0:
                    return val.item()
                return val.tolist()
            return val
        except Exception:
            return default

    def _get_info(self) -> None:
        """Detect the summary layout and populate shape/bookkeeping metadata."""
        summary = self._mat.f["summary"]

        if self._looks_like_split_h5(summary):
            self._summary_layout = "split_h5"
            self._init_from_split_h5(summary)
            return

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

    def _looks_like_split_h5(self, summary: h5py.Group) -> bool:
        """Return True for new metadata-only summaries paired with trace HDF5 files."""
        if "outputH5" in summary or "nTotalROIs" in summary or "roiGlobalOffsets" in summary:
            return True
        if self.trace_path is not None:
            return True
        return False

    def _init_from_split_h5(self, summary: h5py.Group) -> None:
        """Initialize metadata for ``extractDendrites_new`` split MAT/H5 outputs."""
        self.n_dmds = int(self._scalar(summary["nDMDs"], 0)) if "nDMDs" in summary else 0
        self.n_trials = int(self._scalar(summary["nTrials"], 0)) if "nTrials" in summary else 0

        if "nAnalysisROIs" not in summary:
            raise KeyError("summary/nAnalysisROIs is required for split-H5 voltage summaries.")
        nrois_raw = np.asarray(summary["nAnalysisROIs"][()]).astype(int).squeeze()
        if nrois_raw.ndim == 0:
            nrois_raw = np.array([int(nrois_raw)])
        self.n_rois = [int(x) for x in np.ravel(nrois_raw)]
        if self.n_dmds <= 0:
            self.n_dmds = len(self.n_rois)
        self.n_total_rois = int(np.sum(self.n_rois))

        if "roiGlobalOffsets" in summary:
            offsets = np.asarray(summary["roiGlobalOffsets"][()]).astype(int).squeeze()
            self.roi_global_offsets = [int(x) for x in np.ravel(offsets)]
        else:
            self.roi_global_offsets = [0] + list(np.cumsum(self.n_rois[:-1]).astype(int))

        self.trace_path = self._resolve_trace_path(summary)
        self._h5 = h5py.File(self.trace_path, "r")
        if "traces" not in self._h5:
            raise KeyError(f"Trace file does not contain '/traces': {self.trace_path}")

        trial_keys = self._trial_dataset_keys()
        continuous_keys = self._continuous_dataset_keys()
        if len(trial_keys) == 0 and len(continuous_keys) == 0:
            raise KeyError(
                "Trace file contains '/traces' but no trial_XXXX datasets or continuous/DMD datasets."
            )

        # Preserve the MATLAB metadata trial count whenever it is available.
        # Continuous-only H5 outputs have no /traces/trial_XXXX datasets, but the
        # metadata still knows how many behavioral/acquisition trials existed.
        if self.n_trials <= 0:
            self.n_trials = len(trial_keys) if trial_keys else 1
        elif trial_keys:
            self.n_trials = max(self.n_trials, len(trial_keys))

        self.keep_trials = np.zeros((self.n_dmds, max(self.n_trials, 1)), dtype=bool)
        if trial_keys:
            available = set(trial_keys)
            for trial in range(1, self.n_trials + 1):
                name = f"trial_{trial:04d}"
                if name in available:
                    self.keep_trials[:, trial - 1] = True
        else:
            # Continuous traces can be read for any trial index conceptually because
            # the trial argument is ignored in trace_mode='continuous'.
            self.keep_trials[:, :] = True

        self.valid_trials = [
            list(1 + np.flatnonzero(self.keep_trials[dmd0])) for dmd0 in range(self.n_dmds)
        ]

        if "trialEpoch" in summary:
            try:
                te = np.asarray(summary["trialEpoch"][()]).astype(int).squeeze()
                self.trial_epoch = np.ravel(te).astype(int)
                if self.trial_epoch.size:
                    self.n_epochs = int(np.nanmax(self.trial_epoch))
            except Exception:
                self.trial_epoch = None
        elif "trialTable" in summary and isinstance(summary["trialTable"], h5py.Group) and "epoch" in summary["trialTable"]:
            try:
                te = np.asarray(summary["trialTable"]["epoch"][()]).astype(int).squeeze()
                self.trial_epoch = np.ravel(te).astype(int)
                if self.trial_epoch.size:
                    self.n_epochs = int(np.nanmax(self.trial_epoch))
            except Exception:
                self.trial_epoch = None
        if "nEpochs" in summary:
            try:
                self.n_epochs = int(self._scalar(summary["nEpochs"], self.n_epochs))
            except Exception:
                pass

        self.n_samples = self._infer_representative_sample_count()

    def _resolve_trace_path(self, summary: h5py.Group) -> Path:
        """Find the paired ``dendriticVoltageTraces-<timestamp>.h5`` file."""
        if self.trace_path is not None:
            p = Path(self.trace_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"Explicit trace_path does not exist: {p}")

        candidates: List[Path] = []

        if "outputH5" in summary:
            out_h5 = self._decode_matlab_string_dataset(summary["outputH5"])
            if out_h5:
                candidates.append(Path(out_h5))
                candidates.append(self.file_path.parent / Path(out_h5).name)

        timestamp = self._timestamp_from_summary_name(self.file_path.name)
        if timestamp:
            candidates.extend(
                [
                    self.file_path.with_name(f"dendriticVoltageTraces-{timestamp}.h5"),
                    self.file_path.with_name(f"*Traces*{timestamp}*.h5"),
                ]
            )

        candidates.extend(sorted(self.file_path.parent.glob("dendriticVoltageTraces-*.h5")))
        candidates.extend(sorted(self.file_path.parent.glob("*Traces*.h5")))

        expanded: List[Path] = []
        for p in candidates:
            if "*" in str(p):
                expanded.extend(sorted(p.parent.glob(p.name)))
            else:
                expanded.append(p)

        seen: set[Path] = set()
        for p in expanded:
            p = p.expanduser()
            if p in seen:
                continue
            seen.add(p)
            if p.exists():
                return p

        raise FileNotFoundError(
            "Could not locate paired trace HDF5 file. Pass trace_path=... or place "
            "dendriticVoltageTraces-<timestamp>.h5 beside the summary .mat file."
        )

    @staticmethod
    def _timestamp_from_summary_name(name: str) -> Optional[str]:
        """Extract ``YYMMDD-HHMMSS`` timestamp from a dendritic-voltage filename."""
        m = re.search(r"dendriticVoltageSummary[-_](\d{6}-\d{6})", name)
        if m:
            return m.group(1)
        m = re.search(r"(\d{6}-\d{6})", name)
        return m.group(1) if m else None

    @staticmethod
    def _decode_matlab_string_dataset(ds: h5py.Dataset) -> str:
        """Decode MATLAB char/string datasets into a Python string."""
        try:
            val = bytes_to_str(ds[()])
            if isinstance(val, bytes):
                return val.decode(errors="ignore")
            if isinstance(val, str):
                return val.rstrip("\x00")
            arr = np.asarray(val)
            if arr.dtype.kind in ("U", "S", "O"):
                return "".join(np.ravel(arr).astype(str).tolist()).rstrip("\x00")
            if np.issubdtype(arr.dtype, np.number):
                return "".join(chr(int(x)) for x in np.ravel(arr) if int(x) != 0)
        except Exception:
            pass
        return ""

    def _trial_dataset_keys(self) -> List[str]:
        """Return sorted ``/traces/trial_XXXX`` dataset names."""
        if self._h5 is None or "traces" not in self._h5:
            return []
        keys = [k for k in self._h5["traces"].keys() if re.match(r"trial_\d{4}$", k)]
        return sorted(keys)

    def _continuous_dataset_keys(self) -> List[str]:
        """Return sorted ``/traces/continuous/DMD#`` dataset names."""
        if self._h5 is None or "traces/continuous" not in self._h5:
            return []
        keys = [k for k in self._h5["traces/continuous"].keys() if re.match(r"DMD\d+$", k)]
        return sorted(keys, key=lambda x: int(x.replace("DMD", "")))

    def _infer_representative_sample_count(self) -> int:
        """Infer a representative sample count from available trace datasets."""
        if self._h5 is None:
            return 0
        trial_keys = self._trial_dataset_keys()
        if trial_keys:
            ds = self._h5["traces"][trial_keys[0]]
            return self._infer_time_len_from_split_dataset(ds, self.n_total_rois)
        continuous_keys = self._continuous_dataset_keys()
        if continuous_keys:
            key = continuous_keys[0]
            ds = self._h5["traces/continuous"][key]
            match = re.search(r"DMD(\d+)$", key)
            dmd = int(match.group(1)) if match else 1
            return self._infer_time_len_from_split_dataset(ds, self.n_rois[dmd - 1])
        return 0

    def _init_from_flat_summary(self, summary: h5py.Group) -> None:
        """Initialize older single-file flat ``summary/traces`` outputs."""
        traces = summary["traces"]
        if len(traces.shape) != 2:
            raise ValueError(f"Unexpected shape for summary/traces: {traces.shape}")

        nrois_raw = np.asarray(summary["nAnalysisROIs"][()]).astype(int).squeeze()
        if nrois_raw.ndim == 0:
            nrois_raw = np.array([int(nrois_raw)])
        self.n_rois = [int(x) for x in np.ravel(nrois_raw).tolist()]
        self.n_dmds = len(self.n_rois)
        self.n_total_rois = int(np.sum(self.n_rois))
        self.roi_global_offsets = [0] + list(np.cumsum(self.n_rois[:-1]).astype(int))

        s0, s1 = map(int, traces.shape)
        if s0 == self.n_total_rois:
            self._trace_axis = "rois_samples"
            self.n_samples = s1
        elif s1 == self.n_total_rois:
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

    def _init_from_E(self, summary: h5py.Group) -> None:
        """Initialize original per-DMD/per-trial ``summary/E`` layout."""
        E = summary["E"]
        if len(E.shape) != 2:
            raise ValueError(f"Unexpected shape for summary/E: {E.shape}")

        s0, s1 = E.shape
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

        self.n_rois = [0 for _ in range(self.n_dmds)]
        for dmd0 in range(self.n_dmds):
            idx = np.argwhere(self.keep_trials[dmd0])
            if idx.size == 0:
                continue
            try:
                g = self._E_group(dmd0, int(idx[0, 0]))
                F = g["ROIs"]["F"]
                self.n_rois[dmd0] = int(F.shape[1] if F.shape[0] >= F.shape[1] else F.shape[0])
            except Exception:
                self.n_rois[dmd0] = 0

        self.n_total_rois = int(np.sum(self.n_rois))
        self.roi_global_offsets = [0] + list(np.cumsum(self.n_rois[:-1]).astype(int))
        self.valid_trials = [list(1 + np.argwhere(self.keep_trials[dmd0])[:, 0]) for dmd0 in range(self.n_dmds)]

        for dmd0 in range(self.n_dmds):
            idx = np.argwhere(self.keep_trials[dmd0])
            if idx.size == 0:
                continue
            try:
                g = self._E_group(dmd0, int(idx[0, 0]))
                self.n_samples = int(g["ROIs"]["F"].shape[0])
                break
            except Exception:
                pass

    # ------------------------------------------------------------------
    # metadata access
    # ------------------------------------------------------------------

    @property
    def layout(self) -> str:
        """Storage layout: ``split_h5``, ``flat_traces``, or ``event_trial``."""
        return self._summary_layout

    @property
    def metadata(self) -> Dict[str, Any]:
        """Decode and cache simple values under ``summary`` and ``summary/params``."""
        if self._metadata is None:
            out: Dict[str, Any] = {}
            summary = self._mat.f["summary"]
            for key in ("createdAt", "completedAt", "sourceTrialTable", "sessionDir", "outputMode", "storageMode", "outputDir", "outputH5", "summaryPath", "isContinuousAcquisition", "complete"):
                if key in summary and isinstance(summary[key], h5py.Dataset):
                    val = self._scalar(summary[key])
                    if isinstance(val, list):
                        val = self._decode_matlab_string_dataset(summary[key])
                    out[key] = val
            if "params" in summary and isinstance(summary["params"], h5py.Group):
                out["params"] = self._read_group_scalars(summary["params"])
            self._metadata = out
        return self._metadata

    @property
    def h5_attrs(self) -> Dict[str, Any]:
        """Root attributes from the paired trace HDF5 file, if present."""
        if self._h5_attrs is None:
            out: Dict[str, Any] = {}
            if self._h5 is not None:
                for k, v in self._h5.attrs.items():
                    out[k] = bytes_to_str(v)
            self._h5_attrs = out
        return self._h5_attrs

    def _dmd_cell_item(self, cell_name: str, dmd0: int) -> Optional[Union[h5py.Dataset, h5py.Group]]:
        """Dereference a DMD-indexed MATLAB cell array stored under ``summary/dmd``."""
        summary = self._mat.f["summary"]
        if "dmd" not in summary or cell_name not in summary["dmd"]:
            return None
        cell = summary["dmd"][cell_name]
        if not self._is_ref_dataset(cell):
            return cell

        # MATLAB cell arrays can appear transposed depending on save path.  Try
        # both common orientations before giving up.
        for i, j in ((dmd0, 0), (0, dmd0)):
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

    def get_dmd_metadata(self, dmd: int) -> Dict[str, Any]:
        """Return decoded per-DMD acquisition metadata when present.

        Newer ``extractDendrites_new.m`` summaries store SLAP2 acquisition
        metadata under ``summary/dmd/metadata{dmd}``.  This includes the true
        line-scan rate (``lineRateHz``), which is preferred over historical
        hard-coded defaults for voltage time alignment.
        """
        dmd0 = self._validate_dmd(dmd)
        node = self._dmd_cell_item("metadata", dmd0)
        if node is None or not isinstance(node, h5py.Group):
            return {}
        return self._read_group_scalars(node)

    def get_line_rate_hz(self, dmd: Optional[int] = None) -> float:
        """Return the per-DMD or representative SLAP2 line rate in Hz.

        If ``dmd`` is omitted, the median of available finite per-DMD line rates
        is returned.  ``NaN`` is returned when the metadata is unavailable.
        """
        rates: List[float] = []
        dmds = [int(dmd)] if dmd is not None else list(range(1, int(self.n_dmds) + 1))
        for d in dmds:
            md = self.get_dmd_metadata(d)
            for key in ("lineRateHz", "line_rate_hz", "sample_rate_hz", "fs_hz"):
                if key not in md:
                    continue
                try:
                    val = float(np.asarray(md[key]).squeeze())
                except Exception:
                    continue
                if np.isfinite(val) and val > 0:
                    rates.append(val)
                    break
        return float(np.median(rates)) if rates else float("nan")

    def _read_numeric_summary_group(self, group_name: str, *, dmd: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Read numeric datasets from a named ``summary`` subgroup.

        Object/reference datasets, such as filenames, are intentionally skipped.
        For DMD-column fields shaped ``(n_trials, n_dmds)``, passing ``dmd``
        selects the requested 1-indexed DMD column.
        """
        summary = self._mat.f["summary"]
        if group_name not in summary or not isinstance(summary[group_name], h5py.Group):
            return {}
        group = summary[group_name]
        dmd0 = None if dmd is None else self._validate_dmd(dmd)
        out: Dict[str, np.ndarray] = {}
        for key, node in group.items():
            if not isinstance(node, h5py.Dataset):
                continue
            if node.dtype == h5py.ref_dtype or getattr(node.dtype, "kind", None) == "O":
                continue
            arr = np.asarray(node[()])
            if dmd0 is not None and arr.ndim == 2 and arr.shape[1] == self.n_dmds:
                arr = arr[:, dmd0]
            arr = np.asarray(arr).squeeze()
            out[key] = arr
        return out

    def get_trial_line_ranges(self, dmd: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Return numeric ``summary/trialLineRanges`` arrays when present.

        The returned arrays use MATLAB's 1-indexed line numbering as stored in the
        source summary.  Downstream code should subtract one before converting
        line indices to Python sample offsets.
        """
        return self._read_numeric_summary_group("trialLineRanges", dmd=dmd)

    def get_trial_timing_table(self, dmd: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Return numeric ``summary/trialTable`` arrays when present.

        Filename/object columns are omitted.  MATLAB datenum columns, when
        present, are left in MATLAB datenum units so callers can decide how to
        align them to behavior time.
        """
        return self._read_numeric_summary_group("trialTable", dmd=dmd)

    def _read_group_scalars(self, group: h5py.Group) -> Dict[str, Any]:
        """Read scalar-like datasets from a metadata group into a dictionary."""
        out: Dict[str, Any] = {}
        for k, node in group.items():
            if isinstance(node, h5py.Dataset):
                val = self._scalar(node)
                if isinstance(val, list):
                    s = self._decode_matlab_string_dataset(node)
                    out[k] = s if s else val
                else:
                    out[k] = val
        return out

    def analyze_hz(self) -> float:
        """Return ``summary/params/analyzeHz`` when present, otherwise NaN."""
        try:
            hz = self._mat.f["summary"]["params"]["analyzeHz"][()]
            return float(np.asarray(hz).squeeze())
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # index helpers
    # ------------------------------------------------------------------

    def _validate_dmd(self, dmd: int) -> int:
        """Validate a 1-indexed DMD argument and return its 0-indexed value."""
        dmd0 = int(dmd) - 1
        if dmd0 < 0 or dmd0 >= self.n_dmds:
            raise IndexError(f"dmd must be in [1, {self.n_dmds}], got {dmd}")
        return dmd0

    def _validate_trial(self, trial: int) -> int:
        """Validate a 1-indexed trial argument and return its 0-indexed value."""
        trial0 = int(trial) - 1
        if trial0 < 0 or trial0 >= self.n_trials:
            raise IndexError(f"trial must be in [1, {self.n_trials}], got {trial}")
        return trial0

    def _global_roi_cols(self, dmd: int, roi_inds: Optional[Sequence[int]]) -> np.ndarray:
        """Map DMD-local ROI indices to global HDF5 trial-dataset columns."""
        dmd0 = self._validate_dmd(dmd)
        start = int(self.roi_global_offsets[dmd0])
        n_rois = int(self.n_rois[dmd0])
        if roi_inds is None:
            return start + np.arange(n_rois, dtype=int)
        roi_inds_arr = np.asarray(list(roi_inds), dtype=int)
        if np.any(roi_inds_arr < 0) or np.any(roi_inds_arr >= n_rois):
            raise IndexError(f"roi_inds out of range for dmd {dmd}; n_rois={n_rois}")
        return start + roi_inds_arr

    def _local_roi_cols(self, dmd: int, roi_inds: Optional[Sequence[int]]) -> np.ndarray:
        """Return DMD-local ROI columns for continuous datasets."""
        dmd0 = self._validate_dmd(dmd)
        n_rois = int(self.n_rois[dmd0])
        if roi_inds is None:
            return np.arange(n_rois, dtype=int)
        roi_inds_arr = np.asarray(list(roi_inds), dtype=int)
        if np.any(roi_inds_arr < 0) or np.any(roi_inds_arr >= n_rois):
            raise IndexError(f"roi_inds out of range for dmd {dmd}; n_rois={n_rois}")
        return roi_inds_arr

    @staticmethod
    def _slice_len(slc: slice, n: int) -> int:
        """Return the number of elements selected by a slice of length ``n``."""
        return len(range(*slc.indices(n)))

    @staticmethod
    def _infer_split_orientation(ds: h5py.Dataset, n_rois_expected: int) -> str:
        """Infer whether h5py sees split-H5 data as ROI x time or time x ROI.

        MATLAB writes these datasets as ``[n_lines x n_rois]``. In practice, h5py
        can expose the same datasets as ``(n_rois, n_lines)``, which is why the QC
        notebook reads the whole dataset and transposes it. This keeps the public
        API stable as ``(n_samples, n_rois)`` for either orientation.
        """
        if len(ds.shape) != 2:
            raise ValueError(f"Expected a 2D trace dataset, got shape {ds.shape}")
        s0, s1 = map(int, ds.shape)
        n_rois_expected = int(n_rois_expected)
        if s0 == n_rois_expected and s1 != n_rois_expected:
            return "rois_time"
        if s1 == n_rois_expected and s0 != n_rois_expected:
            return "time_rois"
        return "rois_time" if s0 <= s1 else "time_rois"

    @classmethod
    def _infer_time_len_from_split_dataset(cls, ds: h5py.Dataset, n_rois_expected: int) -> int:
        """Return the time/line dimension length for a split-H5 dataset."""
        orient = cls._infer_split_orientation(ds, n_rois_expected)
        return int(ds.shape[1] if orient == "rois_time" else ds.shape[0])

    @classmethod
    def _read_split_columns(
        cls,
        ds: h5py.Dataset,
        rows: slice,
        cols: np.ndarray,
        n_rois_expected: int,
    ) -> np.ndarray:
        """Read selected ROI columns from split-H5 data as ``(time, rois)``."""
        cols = np.asarray(cols, dtype=int)
        orient = cls._infer_split_orientation(ds, n_rois_expected)
        n_time = int(ds.shape[1] if orient == "rois_time" else ds.shape[0])
        if cols.size == 0:
            return np.empty((cls._slice_len(rows, n_time), 0), dtype=ds.dtype)

        order = np.argsort(cols)
        sorted_cols = cols[order]
        if orient == "rois_time":
            x = np.asarray(ds[sorted_cols, rows]).T
        else:
            x = np.asarray(ds[rows, sorted_cols])
        inv = np.argsort(order)
        return x[:, inv]

    @staticmethod
    def _read_columns(ds: h5py.Dataset, rows: slice, cols: np.ndarray) -> np.ndarray:
        """Read selected HDF5 columns and restore the requested order."""
        cols = np.asarray(cols, dtype=int)
        if cols.size == 0:
            return np.empty((len(range(*rows.indices(ds.shape[0]))), 0), dtype=ds.dtype)
        order = np.argsort(cols)
        sorted_cols = cols[order]
        x = np.asarray(ds[rows, sorted_cols])
        inv = np.argsort(order)
        return x[:, inv]

    @staticmethod
    def _align_bool_mask(mask: np.ndarray, n: int) -> np.ndarray:
        """Ensure a boolean mask is one-dimensional with length ``n``."""
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

    # ------------------------------------------------------------------
    # trace access
    # ------------------------------------------------------------------

    def available_trace_modes(self) -> List[str]:
        """Return available split-H5 trace modes: ``trial`` and/or ``continuous``."""
        if self._summary_layout != "split_h5" or self._h5 is None:
            return []
        modes: List[str] = []
        if self._trial_dataset_keys():
            modes.append("trial")
        if self._continuous_dataset_keys():
            modes.append("continuous")
        return modes

    def get_roi_traces(
        self,
        dmd: int = 1,
        trial: int = 1,
        roi_inds: Optional[Sequence[int]] = None,
        t_slice: Optional[slice] = None,
        drop_discarded: bool = False,
        dtype: Optional[np.dtype] = None,
        trace_mode: str = "auto",
    ) -> np.ndarray:
        """Return ROI traces as ``(n_samples, n_rois)``.

        Parameters
        ----------
        dmd, trial : int
            1-indexed DMD and trial.  ``trial`` is ignored only when
            ``trace_mode='continuous'``.
        roi_inds : sequence of int, optional
            DMD-local 0-indexed ROI indices to load.  ``None`` loads all ROIs for
            the requested DMD.
        t_slice : slice, optional
            Sample/line slice along the time axis.
        drop_discarded : bool
            Preserve API compatibility.  Older ``summary/E`` files use stored
            ``discardFrames``.  New split-H5 outputs currently return all-False masks.
        dtype : numpy dtype, optional
            Cast output without copying when possible.
        trace_mode : {"auto", "trial", "continuous"}
            For split-H5 outputs, choose trial-sliced or continuous datasets.
            ``auto`` prefers trial datasets when present.
        """
        if t_slice is None:
            t_slice = slice(None)

        if self._summary_layout == "split_h5":
            x = self._get_split_h5_traces(
                dmd=dmd,
                trial=trial,
                roi_inds=roi_inds,
                t_slice=t_slice,
                trace_mode=trace_mode,
            )
        elif self._summary_layout == "flat_traces":
            x = self._get_flat_traces(dmd=dmd, trial=trial, roi_inds=roi_inds, t_slice=t_slice)
        else:
            x = self._get_E_traces(dmd=dmd, trial=trial, roi_inds=roi_inds, t_slice=t_slice)

        if drop_discarded:
            df = self.get_discard_frames(dmd=dmd, trial=trial, trace_mode=trace_mode)
            df = self._align_bool_mask(df, x.shape[0])
            x = x[~df, :]

        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    def get_traces(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Alias for :meth:`get_roi_traces` for consistency with other summaries."""
        return self.get_roi_traces(*args, **kwargs)

    def _get_split_h5_traces(
        self,
        dmd: int,
        trial: int,
        roi_inds: Optional[Sequence[int]],
        t_slice: slice,
        trace_mode: str,
    ) -> np.ndarray:
        """Read traces from new paired HDF5 files."""
        if self._h5 is None:
            raise RuntimeError("Trace HDF5 file is not open.")

        mode = trace_mode.lower()
        if mode not in ("auto", "trial", "continuous"):
            raise ValueError("trace_mode must be 'auto', 'trial', or 'continuous'.")

        trial_keys = self._trial_dataset_keys()
        continuous_keys = self._continuous_dataset_keys()
        if mode == "auto":
            mode = "trial" if trial_keys else "continuous"

        if mode == "trial":
            trial0 = self._validate_trial(trial)
            ds_name = f"traces/trial_{trial0 + 1:04d}"
            if ds_name not in self._h5:
                raise KeyError(f"Trial dataset not found: /{ds_name}")
            cols = self._global_roi_cols(dmd, roi_inds)
            return self._read_split_columns(
                self._h5[ds_name], t_slice, cols, self.n_total_rois
            )

        self._validate_dmd(dmd)
        ds_name = f"traces/continuous/DMD{dmd}"
        if ds_name not in self._h5:
            raise KeyError(f"Continuous dataset not found: /{ds_name}")
        cols = self._local_roi_cols(dmd, roi_inds)
        return self._read_split_columns(
            self._h5[ds_name], t_slice, cols, self.n_rois[dmd - 1]
        )

    def _get_flat_traces(
        self,
        dmd: int,
        trial: int,
        roi_inds: Optional[Sequence[int]],
        t_slice: slice,
    ) -> np.ndarray:
        """Read traces from older single-file ``summary/traces`` outputs."""
        if trial != 1:
            raise ValueError("Flat-trace voltage summaries expose the session as trial=1 only.")
        traces = self._mat.f["summary"]["traces"]
        row_inds = self._global_roi_cols(dmd, roi_inds)
        if self._trace_axis == "rois_samples":
            x = np.asarray(traces[row_inds, t_slice]).T
        else:
            x = self._read_columns(traces, t_slice, row_inds)
        return np.atleast_2d(x)

    def _get_E_traces(
        self,
        dmd: int,
        trial: int,
        roi_inds: Optional[Sequence[int]],
        t_slice: slice,
    ) -> np.ndarray:
        """Read traces from original ``summary/E/ROIs/F`` outputs."""
        dmd0 = self._validate_dmd(dmd)
        trial0 = self._validate_trial(trial)
        g = self._E_group(dmd0, trial0)
        F = g["ROIs"]["F"]
        if roi_inds is None:
            x = np.asarray(F[t_slice, :])
        else:
            x = np.asarray(F[t_slice, list(roi_inds)])
        return np.atleast_2d(x)

    def get_trace_dataset(self, dmd: int = 1, trial: int = 1, trace_mode: str = "auto") -> h5py.Dataset:
        """Return the underlying HDF5 trace dataset for advanced lazy access."""
        if self._summary_layout != "split_h5" or self._h5 is None:
            raise ValueError("get_trace_dataset is only available for split-H5 outputs.")
        mode = trace_mode.lower()
        if mode == "auto":
            mode = "trial" if self._trial_dataset_keys() else "continuous"
        if mode == "trial":
            trial0 = self._validate_trial(trial)
            return self._h5[f"traces/trial_{trial0 + 1:04d}"]
        if mode == "continuous":
            self._validate_dmd(dmd)
            return self._h5[f"traces/continuous/DMD{dmd}"]
        raise ValueError("trace_mode must be 'auto', 'trial', or 'continuous'.")

    def get_trial_traces(
        self,
        trial: int = 1,
        roi_inds: Optional[Sequence[int]] = None,
        t_slice: Optional[slice] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """Return a full trial-sliced dataset as ``(n_samples, n_total_rois)``.

        ``roi_inds`` are global 0-indexed ROI columns in the trial dataset.  Use
        :meth:`get_roi_traces` to select ROIs by DMD-local indices.
        """
        if self._summary_layout != "split_h5" or self._h5 is None:
            raise ValueError("get_trial_traces requires a split-H5 voltage summary.")
        if t_slice is None:
            t_slice = slice(None)
        trial0 = self._validate_trial(trial)
        ds = self._h5[f"traces/trial_{trial0 + 1:04d}"]
        if roi_inds is None:
            cols = np.arange(self.n_total_rois, dtype=int)
        else:
            cols = np.asarray(list(roi_inds), dtype=int)
        x = self._read_split_columns(ds, t_slice, cols, self.n_total_rois)
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    def get_roi_traces_all_trials(
        self,
        dmd: int = 1,
        roi_inds: Optional[Sequence[int]] = None,
        t_slice: Optional[slice] = None,
        include_invalid: bool = True,
        pad_to: str = "max_valid",
        dtype: Optional[np.dtype] = None,
    ) -> Union[np.ndarray, List[Optional[np.ndarray]]]:
        """Load one DMD's trial-sliced traces across trials.

        Parameters
        ----------
        pad_to : {"max_valid", "none"}
            ``"max_valid"`` returns a padded array shaped
            ``(n_trials, n_samples, n_rois)``.  ``"none"`` returns a list of per-trial
            arrays, with invalid trials as ``None`` when ``include_invalid=False``.
        """
        if pad_to not in ("max_valid", "none"):
            raise ValueError("pad_to must be 'max_valid' or 'none'.")

        dmd0 = self._validate_dmd(dmd)
        trials = range(1, self.n_trials + 1)
        out_list: List[Optional[np.ndarray]] = []
        max_t = 0
        n_roi = len(self._local_roi_cols(dmd, roi_inds))

        for tr in trials:
            valid = bool(self.keep_trials[dmd0, tr - 1])
            if not valid:
                out_list.append(None if not include_invalid else np.full((0, n_roi), np.nan))
                continue
            try:
                x = self.get_roi_traces(
                    dmd=dmd,
                    trial=tr,
                    roi_inds=roi_inds,
                    t_slice=t_slice,
                    dtype=dtype,
                    trace_mode="trial",
                )
            except Exception:
                if not include_invalid:
                    out_list.append(None)
                    continue
                x = np.full((0, n_roi), np.nan)
            max_t = max(max_t, int(x.shape[0]))
            out_list.append(x)

        if pad_to == "none":
            return out_list

        arr = np.full(
            (self.n_trials, max_t, n_roi),
            np.nan,
            dtype=(dtype if dtype is not None else float),
        )
        for i, x in enumerate(out_list):
            if x is None or x.size == 0:
                continue
            arr[i, : x.shape[0], : x.shape[1]] = x
        return arr

    # ------------------------------------------------------------------
    # older event-trial extras and shared masks/images
    # ------------------------------------------------------------------

    def _E_ref(self, dmd0: int, trial0: int) -> Optional[h5py.Reference]:
        """Return the HDF5 reference for a zero-indexed DMD/trial pair."""
        if self._summary_layout != "event_trial":
            raise ValueError("summary/E is not present for this voltage summary.")
        E = self._mat.f["summary"]["E"]
        ref = E[dmd0, trial0] if self._E_layout == "dmd_trial" else E[trial0, dmd0]
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
            raise ValueError(f"No E ref for dmd={dmd0 + 1}, trial={trial0 + 1}")
        node = self._mat.deref(ref)
        if not isinstance(node, h5py.Group):
            raise TypeError("E ref did not dereference to a Group")
        return node

    def get_roi_weights(
        self,
        dmd: int,
        trial: int,
        roi_inds: Optional[Sequence[int]] = None,
        t_slice: Optional[slice] = None,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        """Load ``E/ROIs/weight`` from older event-trial summaries."""
        if self._summary_layout != "event_trial":
            raise KeyError("ROI weights are only stored in older event-trial voltage summaries.")
        if t_slice is None:
            t_slice = slice(None)
        g = self._E_group(self._validate_dmd(dmd), self._validate_trial(trial))
        W = g["ROIs"]["weight"]
        x = np.asarray(W[t_slice, :] if roi_inds is None else W[t_slice, list(roi_inds)])
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
        """Load ``E/global/F`` from older event-trial summaries."""
        if self._summary_layout != "event_trial":
            raise KeyError("Global traces are only stored in older event-trial voltage summaries.")
        if t_slice is None:
            t_slice = slice(None)
        g = self._E_group(self._validate_dmd(dmd), self._validate_trial(trial))
        x = np.asarray(g["global"]["F"][t_slice]).squeeze()
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    def get_discard_frames(self, dmd: int, trial: int, trace_mode: str = "auto") -> np.ndarray:
        """Return stored discard-frame mask, or all-False for split/flat outputs."""
        if self._summary_layout == "event_trial":
            g = self._E_group(self._validate_dmd(dmd), self._validate_trial(trial))
            if "discardFrames" in g:
                return np.asarray(g["discardFrames"][()]).astype(bool).squeeze()
        n = self.get_roi_traces(dmd=dmd, trial=trial, roi_inds=[], trace_mode=trace_mode).shape[0]
        return np.zeros(n, dtype=bool)

    def get_motion(
        self,
        dmd: int,
        trial: int,
        keys: Optional[Sequence[str]] = None,
        t_slice: Optional[slice] = None,
        dtype: Optional[np.dtype] = None,
    ) -> Dict[str, np.ndarray]:
        """Load ``E/upsampledMotion`` fields from older event-trial summaries."""
        if self._summary_layout != "event_trial":
            return {}
        if t_slice is None:
            t_slice = slice(None)
        g = self._E_group(self._validate_dmd(dmd), self._validate_trial(trial))
        if "upsampledMotion" not in g:
            return {}
        um = g["upsampledMotion"]
        if keys is None:
            keys = list(um.keys())
        out: Dict[str, np.ndarray] = {}
        for key in keys:
            if key not in um:
                continue
            arr = np.asarray(um[key][t_slice]).squeeze()
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            out[key] = arr
        return out

    def _cell_item(self, cell_name: str, dmd0: int) -> Optional[Union[h5py.Dataset, h5py.Group]]:
        """Dereference DMD-indexed MATLAB cell arrays stored under ``summary``."""
        summary = self._mat.f["summary"]
        if cell_name not in summary:
            return None
        cell = summary[cell_name]
        if not self._is_ref_dataset(cell):
            return cell
        for i, j in ((dmd0, 0), (0, dmd0)):
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
        """Return DMD reference image from ``summary/refPlane`` or ``summary/refIM``."""
        dmd0 = self._validate_dmd(dmd)
        node = self._cell_item("refPlane", dmd0)
        if node is None:
            node = self._cell_item("refIM", dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"summary/refPlane{{{dmd}}} or summary/refIM{{{dmd}}} not found")
        arr = np.asarray(node[()])
        if self.swap_xy_images and arr.ndim >= 2:
            arr = np.swapaxes(arr, 0, 1)
        return arr

    def get_mean_image(self, dmd: int) -> np.ndarray:
        """Alias for :meth:`get_ref_plane` for workflow compatibility."""
        return self.get_ref_plane(dmd)

    def get_activity_image(self, dmd: int) -> np.ndarray:
        """Return an activity/label image when available, otherwise ROI labels."""
        dmd0 = self._validate_dmd(dmd)
        for name in ("actIM", "maskImages", "userROIs"):
            node = self._cell_item(name, dmd0)
            if isinstance(node, h5py.Dataset):
                arr = np.asarray(node[()])
                if self.swap_xy_images and arr.ndim >= 2:
                    arr = np.swapaxes(arr, 0, 1)
                return arr
        return self.get_user_roi_label_image(dmd)

    def get_roi_masks(self, dmd: int) -> np.ndarray:
        """Return masks as ``(y, x, n_rois)`` boolean array."""
        dmd0 = self._validate_dmd(dmd)
        node = self._cell_item("masks", dmd0)
        if node is None or not isinstance(node, h5py.Dataset):
            raise KeyError(f"summary/masks{{{dmd}}} not found or unexpected type")
        masks = np.asarray(node[()])
        expected = self.n_rois[dmd0]
        if masks.ndim == 3:
            if masks.shape[0] == expected:
                masks = np.moveaxis(masks, 0, -1)
            elif masks.shape[-1] == expected:
                pass
        if self.swap_xy_images and masks.ndim >= 2:
            masks = np.swapaxes(masks, 0, 1)
        return masks.astype(bool)

    def get_user_roi_label_image(self, dmd: int) -> np.ndarray:
        """Return stored ROI label image or reconstruct one from masks."""
        dmd0 = self._validate_dmd(dmd)
        for name in ("userROIs", "maskImages"):
            node = self._cell_item(name, dmd0)
            if isinstance(node, h5py.Dataset):
                img = np.asarray(node[()])
                if self.swap_xy_images and img.ndim >= 2:
                    img = np.swapaxes(img, 0, 1)
                return img
        masks = self.get_roi_masks(dmd)
        label_img = np.zeros(masks.shape[:2], dtype=np.int32)
        for i in range(masks.shape[2]):
            label_img[masks[:, :, i]] = i + 1
        return label_img

    # ------------------------------------------------------------------
    # trial metadata helpers
    # ------------------------------------------------------------------

    def trial_dataset_attrs(self, trial: int) -> Dict[str, Any]:
        """Return HDF5 attributes for ``/traces/trial_XXXX``."""
        if self._summary_layout != "split_h5" or self._h5 is None:
            return {}
        trial0 = self._validate_trial(trial)
        ds_name = f"traces/trial_{trial0 + 1:04d}"
        if ds_name not in self._h5:
            return {}
        return {k: bytes_to_str(v) for k, v in self._h5[ds_name].attrs.items()}

    def continuous_dataset_attrs(self, dmd: int) -> Dict[str, Any]:
        """Return HDF5 attributes for ``/traces/continuous/DMD#``."""
        if self._summary_layout != "split_h5" or self._h5 is None:
            return {}
        self._validate_dmd(dmd)
        ds_name = f"traces/continuous/DMD{dmd}"
        if ds_name not in self._h5:
            return {}
        return {k: bytes_to_str(v) for k, v in self._h5[ds_name].attrs.items()}

    def timebase(self, dmd: int = 1, trial: int = 1, trace_mode: str = "auto") -> np.ndarray:
        """Return a uniform timebase in seconds when ``analyzeHz`` is available.

        For split-H5 files, the length is taken from the selected trial or continuous
        trace dataset.  If ``analyzeHz`` is unavailable, sample/line indices are returned.
        """
        n = int(self.get_roi_traces(dmd=dmd, trial=trial, roi_inds=[], trace_mode=trace_mode).shape[0])
        hz = self.analyze_hz()
        if not np.isfinite(hz) or hz <= 0:
            return np.arange(n, dtype=float)
        return np.arange(n, dtype=float) / hz
