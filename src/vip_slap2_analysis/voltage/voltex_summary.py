"""Reader for SLAP2-Voltage-Extraction ``VoltageSummary.h5`` (format 0.2.x).

Schema verified in ``SLAP2-Voltage-Extraction/processVoltage.m`` (``main`` @
434490c, 2026-11-03; identical on ``feature/motion-censoring`` and ``dev``):

* root: ``/format_version`` (string, ``"0.2.3"``), ``/params/*``, ``/row_major`` = 0;
* ``/Path{n}`` per DMD with voltage ROIs (a DMD with no ROIs has no group);
* ``/Path{n}/sample_rate_hz`` double (``1/linePeriod_s`` from ``.meta``);
* ``/Path{n}/Z_depths`` double;
* ``/Path{n}/frame_info/trial_num_frames`` int32 ``nTrials x 1`` (0 = empty/invalid
  trial), ``trial_epoch`` int32 ``nTrials x 1``, ``frame_line_idxs`` int32
  ``totalSamples x 1`` (1-based source line, ``first:last`` per trial,
  concatenated), ``sample_epoch`` int32 ``totalSamples x 1``, optional
  ``motion_censor`` uint8 ``totalSamples x 1`` (1 = censored);
* ``/Path{n}/sources/spatial/{profiles, coords, source_roi_index, roi_pixel_count}``;
* ``/Path{n}/sources/temporal/{raw_f, f0, dff}`` ``nROIs x 1 x totalSamples``
  (MATLAB) -> h5py ``(totalSamples, 1, nROIs)``; ``dff`` carries attrs
  ``sign_convention``, ``indicator_direction``, ``indicator_name``.

Samples are lines; no timestamps are stored. The companion ``trial_table.h5``
(``buildTrialTableSLAP2.m`` of the voltage repo) uses *inclusive* 1-based line
ranges (``/slap2_info/line_range_convention == "inclusive"``).

The class mirrors the :class:`~vip_slap2_analysis.voltage.summary.VoltageSummary`
API used by ``voltage.postprocess`` / ``voltage.qc`` / ``voltage.extraction``:
``n_dmds, n_trials, n_rois, n_total_rois, roi_global_offsets, valid_trials,
trial_epoch, n_epochs, layout, metadata, h5_attrs, available_trace_modes,
get_roi_traces(dmd, trial, roi_inds, t_slice, drop_discarded, dtype, trace_mode)``
(time-major ``(n_samples, n_rois)``), ``get_discard_frames``, ``get_trial_source_lines``,
``get_trial_line_ranges``, ``get_dmd_metadata``, ``get_dmd_epoch_metadata``,
``get_dmd_epoch_durations_sec``, ``get_line_rate_hz``, ``get_trace_dataset``,
``_infer_time_len_from_split_dataset``.

The Voltex ``dff``/``f0`` datasets are exposed through ``trace_kind`` so the
downstream pipeline can either re-derive dF/F from ``raw_f`` (default, identical
to the legacy path) or read Voltex's own ``dff``. The two are never mixed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np

PathLike = Union[str, Path]

REQUIRED_PATH_DATASETS = (
    "sample_rate_hz",
    "frame_info/trial_num_frames",
    "frame_info/frame_line_idxs",
    "sources/temporal/raw_f",
)
OPTIONAL_PATH_DATASETS = (
    "frame_info/trial_epoch",
    "frame_info/sample_epoch",
    "frame_info/motion_censor",
    "sources/temporal/f0",
    "sources/temporal/dff",
    "sources/spatial/profiles",
    "sources/spatial/coords",
    "sources/spatial/source_roi_index",
    "sources/spatial/roi_pixel_count",
    "Z_depths",
)


def _decode(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8", "ignore")
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("S", "O"):
            flat = x.ravel()
            return _decode(flat[0]) if flat.size == 1 else [_decode(v) for v in flat]
        flat = x.ravel()
        return flat.tolist() if flat.size > 1 else (flat[0].item() if flat.size else None)
    if isinstance(x, np.generic):
        return x.item()
    return x


def _read_1d(ds: h5py.Dataset, sl: slice) -> np.ndarray:
    """Slice a MATLAB ``N x 1`` vector stored as h5py ``(1, N)`` (or ``(N,)``/``(N, 1)``)."""
    if ds.ndim == 1:
        return np.asarray(ds[sl]).ravel()
    if ds.ndim == 2 and ds.shape[0] == 1:
        return np.asarray(ds[0, sl]).ravel()
    if ds.ndim == 2 and ds.shape[1] == 1:
        return np.asarray(ds[sl, 0]).ravel()
    return np.asarray(ds[()]).ravel()[sl]


def validate_voltex_summary_schema(path: PathLike) -> Dict[str, Any]:
    """Report whether a ``VoltageSummary.h5`` has the datasets this reader needs."""
    path = Path(path)
    report: Dict[str, Any] = {"path": str(path), "compatible": False, "paths": {}, "problems": [], "format_version": None}
    if not path.exists():
        report["problems"].append("file does not exist")
        return report
    try:
        with h5py.File(path, "r") as f:
            if "format_version" in f:
                report["format_version"] = str(_decode(f["format_version"][()]))
            else:
                report["problems"].append("missing /format_version")
            if "row_major" in f:
                rm = int(np.asarray(f["row_major"][()]).ravel()[0])
                report["row_major"] = rm
                if rm != 0:
                    report["problems"].append(f"unexpected row_major={rm}")
            paths = sorted(k for k in f.keys() if k.startswith("Path") and isinstance(f[k], h5py.Group))
            if not paths:
                report["problems"].append("no /Path{n} groups (no voltage ROIs?)")
            for pg in paths:
                g = f[pg]
                entry: Dict[str, Any] = {"datasets": {}, "missing_required": []}
                for ds in REQUIRED_PATH_DATASETS:
                    if ds in g:
                        entry["datasets"][ds] = {"shape": list(g[ds].shape), "dtype": str(g[ds].dtype)}
                    else:
                        entry["missing_required"].append(ds)
                for ds in OPTIONAL_PATH_DATASETS:
                    if ds in g:
                        entry["datasets"][ds] = {"shape": list(g[ds].shape), "dtype": str(g[ds].dtype)}
                if entry["missing_required"]:
                    report["problems"].append(f"{pg}: missing {entry['missing_required']}")
                else:
                    tnf = np.asarray(g["frame_info/trial_num_frames"][()]).ravel()
                    n = int(tnf.sum())
                    raw = g["sources/temporal/raw_f"]
                    if int(np.prod(g["frame_info/frame_line_idxs"].shape)) != n:
                        report["problems"].append(f"{pg}: frame_line_idxs length != sum(trial_num_frames)")
                    if raw.ndim != 3 or raw.shape[0] != n:
                        report["problems"].append(f"{pg}: raw_f shape {list(raw.shape)} not (totalSamples={n}, 1, nROIs)")
                    entry.update({"n_trials": int(tnf.size), "n_samples": n, "n_rois": int(raw.shape[-1]) if raw.ndim == 3 else None,
                                  "sample_rate_hz": float(np.asarray(g["sample_rate_hz"][()]).ravel()[0])})
                report["paths"][pg] = entry
    except OSError as exc:
        report["problems"].append(f"cannot open as HDF5: {exc}")
        return report
    report["compatible"] = not report["problems"]
    return report


@dataclass
class VoltexSummary:
    """Lazy reader for Voltex ``VoltageSummary.h5`` with the ``VoltageSummary`` API.

    Parameters
    ----------
    file_path
        ``VoltageSummary.h5`` (or the pilot ``voltage_summary.h5``).
    trial_table_path
        Optional ``trial_table.h5``. Defaults to ``<dir>/../trial_table.h5`` then
        ``<dir>/trial_table.h5``.
    dynamic_data_dir
        Optional raw ``slap2/dynamic_data`` dir for lines-per-cycle / cycle counts.
    trace_kind
        ``"raw_f"`` (default; downstream code computes dF/F itself as for legacy
        products), ``"dff"`` or ``"f0"`` to read Voltex's stored transforms.
    apply_motion_censor
        If True, ``frame_info/motion_censor`` is reported through
        ``get_discard_frames`` so censored samples are dropped like discard frames.
    """

    file_path: PathLike
    trial_table_path: Optional[PathLike] = None
    dynamic_data_dir: Optional[PathLike] = None
    trace_kind: str = "raw_f"
    apply_motion_censor: bool = False
    keep_open: bool = True
    swap_xy_images: bool = True

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        report = validate_voltex_summary_schema(self.file_path)
        if not report["compatible"]:
            raise ValueError(f"{self.file_path}: incompatible Voltex summary schema: {report['problems']}")
        if self.trace_kind not in ("raw_f", "dff", "f0"):
            raise ValueError("trace_kind must be 'raw_f', 'dff', or 'f0'")
        self.schema_report = report
        self.layout = "voltex_h5"
        self.summary_format = "voltex_voltage_summary_h5"
        self._h5: Optional[h5py.File] = h5py.File(self.file_path, "r") if self.keep_open else None
        self.path_dmds: List[int] = sorted(int(k.replace("Path", "")) for k in report["paths"])
        self.n_dmds: int = max(self.path_dmds) if self.path_dmds else 0
        self._tnf: Dict[int, np.ndarray] = {}
        self._offsets: Dict[int, np.ndarray] = {}
        self._trial_epoch_by_dmd: Dict[int, np.ndarray] = {}
        self.n_rois: List[int] = [0] * self.n_dmds
        self._sample_rate: Dict[int, float] = {}
        self._dff_attrs: Dict[int, Dict[str, Any]] = {}
        with self._open() as f:
            for d in self.path_dmds:
                g = f[f"Path{d}"]
                tnf = np.asarray(g["frame_info/trial_num_frames"][()]).ravel().astype(int)
                self._tnf[d] = tnf
                self._offsets[d] = np.concatenate([[0], np.cumsum(tnf)])
                if "frame_info/trial_epoch" in g:
                    self._trial_epoch_by_dmd[d] = np.asarray(g["frame_info/trial_epoch"][()]).ravel().astype(int)
                self.n_rois[d - 1] = int(g["sources/temporal/raw_f"].shape[-1])
                self._sample_rate[d] = float(np.asarray(g["sample_rate_hz"][()]).ravel()[0])
                if "sources/temporal/dff" in g:
                    self._dff_attrs[d] = {k: _decode(np.asarray(v)) for k, v in g["sources/temporal/dff"].attrs.items()}
            self._params = {k: _decode(np.asarray(v[()])) for k, v in f["params"].items() if isinstance(v, h5py.Dataset)} if "params" in f else {}
            self._format_version = report.get("format_version")
        self.n_total_rois = int(sum(self.n_rois))
        self.roi_global_offsets = [0] + list(np.cumsum(self.n_rois[:-1]).astype(int))
        self.n_trials = int(max((v.size for v in self._tnf.values()), default=0))
        self.keep_trials = np.zeros((self.n_dmds, max(self.n_trials, 1)), dtype=bool)
        for d in self.path_dmds:
            self.keep_trials[d - 1, : self._tnf[d].size] = self._tnf[d] > 0
        self.valid_trials = [list(1 + np.flatnonzero(self.keep_trials[d])) for d in range(self.n_dmds)]
        te = next(iter(self._trial_epoch_by_dmd.values()), None)
        self.trial_epoch: Optional[np.ndarray] = te.astype(int) if te is not None else None
        self.n_epochs = int(np.max(te[te > 0])) if te is not None and np.any(te > 0) else (1 if self.n_trials else 0)
        self.trial_table_path = self._resolve_trial_table(self.trial_table_path)
        self._trial_table = self._read_trial_table(self.trial_table_path) if self.trial_table_path else {}
        self.n_samples = int(self._offsets[self.path_dmds[0]][-1]) if self.path_dmds else 0
        self._acq_cache: Dict[int, Dict[str, Any]] = {}

    # -- lifecycle ----------------------------------------------------------
    def _open(self):
        if self._h5 is not None:
            class _Keep:
                def __init__(s, f): s.f = f
                def __enter__(s): return s.f
                def __exit__(s, *a): return False
            return _Keep(self._h5)
        return h5py.File(self.file_path, "r")

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "VoltexSummary":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- metadata ------------------------------------------------------------
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"format_version": self._format_version, "params": dict(self._params), "trace_kind": self.trace_kind,
                "dff_attrs": {str(k): v for k, v in self._dff_attrs.items()}, "summary_format": self.summary_format}

    @property
    def h5_attrs(self) -> Dict[str, Any]:
        return {"format_version": self._format_version}

    def _resolve_trial_table(self, explicit: Optional[PathLike]) -> Optional[Path]:
        if explicit is not None:
            p = Path(explicit)
            return p if p.exists() else None
        base = self.file_path.parent
        for _ in range(3):
            cand = base / "trial_table.h5"
            if cand.exists():
                return cand
            base = base.parent
        return None

    @staticmethod
    def _read_trial_table(path: Path) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            for key in ("slap2_info/first_line", "slap2_info/last_line", "epoch", "true_trial_ix"):
                if key in f:
                    out[key.split("/")[-1]] = np.atleast_2d(np.asarray(f[key][()], dtype=float))
            if "slap2_info/line_range_convention" in f:
                out["line_range_convention"] = str(_decode(f["slap2_info/line_range_convention"][()]))
            for key in ("slap2_info/trial_end_time_from_pc", "slap2_info/trial_start_time_inferred"):
                if key in f:
                    out[key.split("/")[-1]] = np.asarray(f[key][()], dtype=float).ravel()
        return out

    @property
    def trial_table(self) -> Dict[str, Any]:
        return dict(self._trial_table)

    def available_trace_modes(self) -> List[str]:
        return ["trial"]

    # -- indexing ------------------------------------------------------------
    def _validate_dmd(self, dmd: int) -> int:
        d = int(dmd)
        if d not in self._tnf:
            raise IndexError(f"DMD {dmd} has no voltage ROIs in {self.file_path.name} (paths: {self.path_dmds})")
        return d - 1

    def _validate_trial(self, trial: int) -> int:
        t0 = int(trial) - 1
        if t0 < 0 or t0 >= self.n_trials:
            raise IndexError(f"trial must be in [1, {self.n_trials}], got {trial}")
        return t0

    def _sample_slice(self, dmd: int, trial: int) -> slice:
        self._validate_dmd(dmd)
        t0 = self._validate_trial(trial)
        if t0 >= self._tnf[int(dmd)].size or self._tnf[int(dmd)][t0] <= 0:
            raise ValueError(f"Trial {trial} is invalid in DMD {dmd}")
        return slice(int(self._offsets[int(dmd)][t0]), int(self._offsets[int(dmd)][t0 + 1]))

    @staticmethod
    def _infer_time_len_from_split_dataset(ds: h5py.Dataset, n_rois_expected: int) -> int:
        return int(ds.shape[0])

    def get_trace_dataset(self, dmd: int = 1, trial: int = 1, trace_mode: str = "trial", epoch: Optional[int] = None) -> h5py.Dataset:
        """Return a view-like object with ``shape == (n_trial_samples, 1, n_rois)``."""
        sl = self._sample_slice(dmd, trial)
        f = self._h5 if self._h5 is not None else h5py.File(self.file_path, "r")
        ds = f[f"Path{int(dmd)}/sources/temporal/{self.trace_kind}"]

        class _View:
            shape = (sl.stop - sl.start,) + tuple(ds.shape[1:])
            ndim = ds.ndim
        return _View()

    # -- traces --------------------------------------------------------------
    def get_roi_traces(
        self,
        dmd: int = 1,
        trial: int = 1,
        roi_inds: Optional[Sequence[int]] = None,
        t_slice: Optional[slice] = None,
        drop_discarded: bool = False,
        dtype: Optional[np.dtype] = None,
        trace_mode: str = "trial",
        epoch: Optional[int] = None,
        trace_kind: Optional[str] = None,
    ) -> np.ndarray:
        """Return ``(n_samples, n_rois)`` for one DMD/trial from ``raw_f`` (or ``dff``/``f0``)."""
        kind = trace_kind or self.trace_kind
        sl = self._sample_slice(dmd, trial)
        if t_slice is not None:
            a = sl.start + (t_slice.start or 0)
            b = sl.start + (t_slice.stop if t_slice.stop is not None else (sl.stop - sl.start))
            sl = slice(a, min(b, sl.stop))
        with self._open() as f:
            ds = f[f"Path{int(dmd)}/sources/temporal/{kind}"]
            if roi_inds is None:
                x = np.asarray(ds[sl, 0, :])
            else:
                x = np.asarray(ds[sl, 0, :])[:, list(roi_inds)]
        if drop_discarded:
            mask = self.get_discard_frames(dmd, trial)
            if t_slice is not None:
                mask = mask[(t_slice.start or 0):(t_slice.stop if t_slice.stop is not None else None)]
            if mask.size == x.shape[0]:
                x = x[~mask, :]
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return np.atleast_2d(x)

    def get_traces(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.get_roi_traces(*args, **kwargs)

    def has_trace_kind(self, kind: str, dmd: Optional[int] = None) -> bool:
        """True when ``sources/temporal/<kind>`` exists for ``dmd`` (or for every DMD with ROIs)."""
        dmds = [int(dmd)] if dmd is not None else [d for d in range(1, int(self.n_dmds) + 1) if self.n_rois[d - 1] > 0]
        with self._open() as f:
            return all(f"Path{d}/sources/temporal/{kind}" in f for d in dmds) and bool(dmds)

    @property
    def product_dff_available(self) -> bool:
        """Voltex wrote its own polarity-corrected ``dff`` (and ``f0``) for every DMD with ROIs."""
        try:
            return self.has_trace_kind("dff") and self.has_trace_kind("f0")
        except Exception:
            return False

    def using(self, kind: str) -> "_TraceKindContext":
        """Context manager that temporarily switches the default ``trace_kind`` (raw_f / dff / f0)."""
        return _TraceKindContext(self, kind)

    def get_discard_frames(self, dmd: int, trial: int, trace_mode: str = "trial") -> np.ndarray:
        """All-False unless ``apply_motion_censor`` and ``frame_info/motion_censor`` exist."""
        sl = self._sample_slice(dmd, trial)
        n = sl.stop - sl.start
        if not self.apply_motion_censor:
            return np.zeros((n,), dtype=bool)
        with self._open() as f:
            key = f"Path{int(dmd)}/frame_info/motion_censor"
            if key not in f:
                return np.zeros((n,), dtype=bool)
            return _read_1d(f[key], sl).astype(bool)

    def get_motion_censor_mask(self, dmd: int) -> Optional[np.ndarray]:
        with self._open() as f:
            key = f"Path{int(dmd)}/frame_info/motion_censor"
            return np.asarray(f[key][()]).ravel().astype(bool) if key in f else None

    def get_trial_source_lines(self, dmd: int, trial: int) -> np.ndarray:
        sl = self._sample_slice(dmd, trial)
        with self._open() as f:
            return _read_1d(f[f"Path{int(dmd)}/frame_info/frame_line_idxs"], sl).astype(np.int64)

    def get_trial_line_ranges(self, dmd: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Per-trial first/last line and sample counts (from ``frame_line_idxs``)."""
        d = int(dmd) if dmd is not None else self.path_dmds[0]
        self._validate_dmd(d)
        tnf = self._tnf[d]
        first = np.full(tnf.size, np.nan); last = np.full(tnf.size, np.nan)
        with self._open() as f:
            fl = np.asarray(f[f"Path{d}/frame_info/frame_line_idxs"][()]).ravel()
            for i in range(tnf.size):
                if tnf[i] > 0:
                    first[i] = float(fl[int(self._offsets[d][i])]); last[i] = float(fl[int(self._offsets[d][i + 1]) - 1])
        return {"trialFirstLineGlobal": first, "trialLastLineGlobal": last, "nLines": tnf.astype(float)}

    def get_sample_epoch(self, dmd: int) -> Optional[np.ndarray]:
        with self._open() as f:
            key = f"Path{int(dmd)}/frame_info/sample_epoch"
            return np.asarray(f[key][()]).ravel().astype(int) if key in f else None

    def get_roi_masks(self, dmd: int) -> np.ndarray:
        with self._open() as f:
            key = f"Path{int(dmd)}/sources/spatial/profiles"
            if key not in f:
                return np.zeros((0, 0, 0), dtype=bool)
            p = np.asarray(f[key][()])  # h5py (W, H, 1, nROIs) from MATLAB nROIs x 1 x rows x cols
        p = np.squeeze(p, axis=2) if p.ndim == 4 else p
        masks = np.transpose(p, (2, 1, 0)) if p.ndim == 3 else p  # -> (nROIs, rows, cols)
        if self.swap_xy_images and masks.ndim == 3:
            masks = np.transpose(masks, (0, 2, 1))
        return masks > 0

    # -- acquisition metadata (for clock QC / registration) ------------------
    def _acquisition(self, dmd: int) -> Dict[str, Any]:
        dmd = int(dmd)
        if dmd in self._acq_cache:
            return self._acq_cache[dmd]
        sample_rate = self._sample_rate.get(dmd, float("nan"))
        out: Dict[str, Any] = {"metadata": {"lineRateHz": sample_rate, "source": "VoltageSummary.h5:/Path/sample_rate_hz"}, "epochs": []}
        lpc = None
        timing_by_prefix: Dict[str, Any] = {}
        if self.dynamic_data_dir is not None and Path(self.dynamic_data_dir).exists():
            from vip_slap2_analysis.io.slap2_dat import list_acquisition_prefixes, read_dmd_acquisition_timing
            for prefix in list_acquisition_prefixes(self.dynamic_data_dir):
                try:
                    timing_by_prefix[prefix] = read_dmd_acquisition_timing(self.dynamic_data_dir, dmd, prefix=prefix)
                except FileNotFoundError:
                    continue
            if timing_by_prefix:
                lpc = int(next(iter(timing_by_prefix.values())).lines_per_cycle)
                out["metadata"]["linesPerCycle"] = lpc
                out["metadata"]["source"] = "raw_dat_headers+VoltageSummary.h5"
        # epochs from the summary's own trial_epoch/frame_line_idxs
        te = self._trial_epoch_by_dmd.get(dmd)
        tnf = self._tnf[dmd]
        if te is None:
            te = np.ones(tnf.size, dtype=int)
        ranges = self.get_trial_line_ranges(dmd)
        prefixes = list(timing_by_prefix.keys())
        for i, ep in enumerate(sorted(set(int(x) for x in te if x > 0)), start=1):
            idx = np.flatnonzero((te == ep) & (tnf > 0))
            if idx.size == 0:
                continue
            last_line = float(np.nanmax(ranges["trialLastLineGlobal"][idx]))
            row: Dict[str, Any] = {"epochIdx": int(ep), "available": True, "totalNumLines": int(last_line),
                                   "firstTrial": int(idx[0] + 1), "lastTrial": int(idx[-1] + 1),
                                   "metadata": {"lineRateHz": sample_rate}}
            if lpc:
                row["linesPerCycle"] = lpc
                row["nCycles"] = int(last_line // lpc)
            if i - 1 < len(prefixes):
                t = timing_by_prefix[prefixes[i - 1]]
                row["rawDatCycles"] = int(t.n_cycles); row["rawDatPrefix"] = prefixes[i - 1]; row["nSourceGaps"] = int(t.gap_indices.size)
                row["nCycles"] = int(t.n_cycles)
                row["totalNumLines"] = int(t.n_cycles * t.lines_per_cycle)
            if sample_rate and np.isfinite(sample_rate):
                row["durationSec"] = float(row["totalNumLines"] / sample_rate)
            out["epochs"].append(row)
        self._acq_cache[dmd] = out
        return out

    def get_dmd_metadata(self, dmd: int) -> Dict[str, Any]:
        self._validate_dmd(dmd)
        return dict(self._acquisition(dmd)["metadata"])

    def get_dmd_epoch_metadata(self, dmd: int) -> List[Dict[str, Any]]:
        self._validate_dmd(dmd)
        return [dict(r) for r in self._acquisition(dmd)["epochs"]]

    def get_dmd_epoch_durations_sec(self, dmd: int, *, sample_rate_hz: Optional[float] = None) -> Dict[int, float]:
        rate = float(sample_rate_hz) if sample_rate_hz else self._sample_rate.get(int(dmd), float("nan"))
        out: Dict[int, float] = {}
        for r in self._acquisition(dmd)["epochs"]:
            if r.get("totalNumLines") and np.isfinite(rate) and rate > 0:
                out[int(r["epochIdx"])] = float(r["totalNumLines"] / rate)
        return out

    def get_line_rate_hz(self, dmd: Optional[int] = None) -> float:
        if dmd is not None:
            return float(self._sample_rate.get(int(dmd), float("nan")))
        vals = [v for v in self._sample_rate.values() if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def analyze_hz(self) -> float:
        return self.get_line_rate_hz()


def find_voltex_summary_candidates(session_dir: PathLike) -> List[Path]:
    """``VoltageSummary.h5`` / ``voltage_summary.h5`` below a session directory, newest first."""
    root = Path(session_dir)
    hits = list(root.rglob("VoltageSummary.h5")) + list(root.rglob("voltage_summary.h5"))
    return sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)


class _TraceKindContext:
    def __init__(self, vs: "VoltexSummary", kind: str) -> None:
        if kind not in ("raw_f", "dff", "f0"):
            raise ValueError("trace kind must be 'raw_f', 'dff' or 'f0'")
        self.vs, self.kind, self._prev = vs, kind, vs.trace_kind

    def __enter__(self) -> "VoltexSummary":
        self.vs.trace_kind = self.kind
        return self.vs

    def __exit__(self, *exc: Any) -> None:
        self.vs.trace_kind = self._prev

