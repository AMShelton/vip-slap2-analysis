"""Reader for GIAnT ``experiment_summary.h5`` with a ``GlutamateSummary``-compatible API.

Schema (verified against ``GIAnT-MATLAB/dependencies/io/saveExperimentSummaryH5.m``
and ``saveStructToH5.m`` on branch ``fix/aind-slap2-dynamic-data``, 2026-11):

* ``saveStructToH5`` writes dataspace dims as ``fliplr(size(x))``, so a MATLAB
  array ``(a x b x c)`` appears to h5py as ``(c, b, a)``; ``/row_major`` is 0.
* ``/Path{n}`` = DMD ``n`` (row of the trial table). Groups are omitted when a
  DMD produced no valid trials.
* ``/Path{n}/frame_info/trial_num_frames`` int32 ``nTrials x 1`` (0 for skipped
  trials); ``frame_line_idxs`` int32 ``totalFrames x 1`` = concatenated
  ``E.frameLines`` (1-based global source line numbers, ``ceil(first:dt:last)``);
  ``discard_frames`` int8 ``totalFrames x 1``.
* ``/Path{n}/sources/temporal/{dF_ls, dF_denoised, events, F0}`` double
  ``nSources x nChannels x totalFrames`` in MATLAB -> h5py ``(totalFrames,
  nChannels, nSources)``. The channel axis is the *full* acquisition channel
  list in original order; only the analyzed channels are finite (others NaN).
  ``/params/activityChannel`` (1-based) is the activity (glutamate) channel.
* ``/Path{n}/sources/spatial/coords`` double ``nSources x 3`` ``[z, y, x]``
  0-based centroids; ``profiles`` single ``nSources x 1 x H x W``.
* ``/Path{n}/user_rois/F`` double ``nRois x nChannels x totalFrames``.
* No timebase is stored; ``/params/analyzeHz`` is the nominal frame rate.
* Companion ``trial_table.h5`` (``buildTrialTableSLAP2.m``): ``/epoch`` and
  ``/true_trial_ix`` double ``nDMDs x nTrials``, ``/filename`` string
  ``nDMDs x nTrials``, ``/slap2_info/first_line,last_line`` double
  ``nDMDs x nTrials`` (GIAnT's continuous-mode convention is half-open:
  ``last_line(i) == first_line(i+1)``), ``/slap2_info/trial_end_time_from_pc``
  (MATLAB datenum).

This reader exposes the subset of :class:`~vip_slap2_analysis.glutamate.summary.GlutamateSummary`
used by ``common.alignment`` / ``glutamate.extraction`` / ``common.clock_qc``:
``n_dmds``, ``n_trials``, ``valid_trials``, ``n_synapses``, ``trial_epoch``,
``n_epochs``, ``get_traces(dmd, trial, signal, mode, ..., squeeze_channels)``
returning ``(time, rois, channels)``, ``get_trial_source_lines``,
``get_dmd_metadata``, ``get_dmd_epoch_metadata``, ``get_dmd_epoch_durations_sec``,
``get_line_rate_hz``. Acquisition metadata (line rate, lines per cycle, cycle
counts) is **not** in the GIAnT files; it is taken from raw ``.meta``/``.dat``
headers when ``dynamic_data_dir`` is given, otherwise those methods return
empty results and clock QC reports ``unavailable`` rather than guessing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]

REQUIRED_PATH_DATASETS = (
    "frame_info/trial_num_frames",
    "frame_info/frame_line_idxs",
    "sources/temporal/dF_ls",
    "sources/temporal/F0",
)
OPTIONAL_PATH_DATASETS = (
    "frame_info/discard_frames",
    "sources/temporal/dF_denoised",
    "sources/temporal/events",
    "sources/spatial/coords",
    "sources/spatial/profiles",
    "user_rois/F",
    "user_rois/labels",
    "global/F",
)


def _decode_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", "ignore")
    if isinstance(x, np.ndarray):
        flat = x.ravel()
        if flat.size == 1:
            return _decode_str(flat[0])
        return "".join(_decode_str(v) for v in flat)
    return str(x)


def _read_1d(ds: h5py.Dataset, sl: slice) -> np.ndarray:
    """Slice a MATLAB ``N x 1`` vector stored as h5py ``(1, N)`` (or ``(N,)``/``(N, 1)``)."""
    if ds.ndim == 1:
        return np.asarray(ds[sl]).ravel()
    if ds.ndim == 2 and ds.shape[0] == 1:
        return np.asarray(ds[0, sl]).ravel()
    if ds.ndim == 2 and ds.shape[1] == 1:
        return np.asarray(ds[sl, 0]).ravel()
    return np.asarray(ds[()]).ravel()[sl]


def validate_giant_summary_schema(path: PathLike) -> Dict[str, Any]:
    """Inspect an ``experiment_summary.h5`` and report schema compatibility.

    Returns a JSON-serializable dict with ``compatible`` (bool), the datasets
    found per Path, shapes, ``row_major``, ``params`` scalars and ``problems``.
    Existence of the file is never sufficient: a missing required dataset or an
    unexpected ``row_major`` flag marks the file incompatible.
    """
    path = Path(path)
    report: Dict[str, Any] = {"path": str(path), "compatible": False, "paths": {}, "problems": [], "params": {}}
    if not path.exists():
        report["problems"].append("file does not exist")
        return report
    try:
        with h5py.File(path, "r") as f:
            rm = f["row_major"][()] if "row_major" in f else None
            report["row_major"] = None if rm is None else int(np.asarray(rm).ravel()[0])
            if report["row_major"] not in (0, None):
                report["problems"].append(f"unexpected row_major={report['row_major']} (GIAnT writes 0)")
            if "params" in f and isinstance(f["params"], h5py.Group):
                for k, v in f["params"].items():
                    if isinstance(v, h5py.Dataset) and v.size <= 8:
                        val = v[()]
                        try:
                            if v.dtype.kind in ("S", "O"):
                                report["params"][k] = _decode_str(val)
                            else:
                                arr = np.asarray(val).ravel()
                                report["params"][k] = arr.tolist() if arr.size > 1 else float(arr[0])
                        except Exception:
                            pass
            else:
                report["problems"].append("missing /params group")
            path_groups = sorted(k for k in f.keys() if k.startswith("Path") and isinstance(f[k], h5py.Group))
            if not path_groups:
                report["problems"].append("no /Path{n} groups")
            for pg in path_groups:
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
                    fli = g["frame_info/frame_line_idxs"]
                    dfl = g["sources/temporal/dF_ls"]
                    n_frames = int(tnf.sum())
                    if int(np.prod(fli.shape)) != n_frames:
                        report["problems"].append(f"{pg}: frame_line_idxs has {int(np.prod(fli.shape))} entries, trial_num_frames sums to {n_frames}")
                    if dfl.ndim != 3 or dfl.shape[0] != n_frames:
                        report["problems"].append(f"{pg}: dF_ls shape {list(dfl.shape)} does not start with totalFrames={n_frames} (expected h5py order (frames, channels, sources))")
                    entry["n_trials"] = int(tnf.size)
                    entry["n_frames"] = n_frames
                    entry["n_sources"] = int(dfl.shape[2]) if dfl.ndim == 3 else None
                    entry["n_channels"] = int(dfl.shape[1]) if dfl.ndim == 3 else None
                report["paths"][pg] = entry
    except OSError as exc:
        report["problems"].append(f"cannot open as HDF5: {exc}")
        return report
    report["compatible"] = not report["problems"]
    return report


@dataclass
class GiantExperimentSummary:
    """Lazy reader for GIAnT ``experiment_summary.h5`` (+ optional ``trial_table.h5``).

    Parameters
    ----------
    file_path
        Path to ``experiment_summary.h5``.
    trial_table_path
        Optional ``trial_table.h5``; when omitted, ``<dir>/trial_table.h5`` and
        ``<dir>/../trial_table.h5`` are tried. Needed for per-trial epoch labels.
    dynamic_data_dir
        Optional raw ``slap2/dynamic_data`` directory. Supplies acquisition
        metadata (``lineRateHz``, ``linesPerCycle``, cycle counts per epoch) from
        ``.meta``/``.dat`` headers for clock QC and cycle registration.
    keep_open
        Keep the HDF5 handle open between calls.
    """

    file_path: PathLike
    trial_table_path: Optional[PathLike] = None
    dynamic_data_dir: Optional[PathLike] = None
    keep_open: bool = True

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path)
        report = validate_giant_summary_schema(self.file_path)
        if not report["compatible"]:
            raise ValueError(f"{self.file_path}: incompatible GIAnT summary schema: {report['problems']}")
        self.schema_report = report
        self._f: Optional[h5py.File] = h5py.File(self.file_path, "r") if self.keep_open else None
        self.layout = "giant_h5"
        self.summary_format = "giant_experiment_summary_h5"

        self.path_names: List[str] = sorted(report["paths"].keys(), key=lambda s: int(s.replace("Path", "")))
        self.path_dmds: List[int] = [int(s.replace("Path", "")) for s in self.path_names]
        self.n_dmds: int = max(self.path_dmds) if self.path_dmds else 0
        self._tnf: Dict[int, np.ndarray] = {}
        self._offsets: Dict[int, np.ndarray] = {}
        with self._open() as f:
            for dmd in self.path_dmds:
                tnf = np.asarray(f[f"Path{dmd}/frame_info/trial_num_frames"][()]).ravel().astype(int)
                self._tnf[dmd] = tnf
                self._offsets[dmd] = np.concatenate([[0], np.cumsum(tnf)])
            self.activity_channel: int = int(np.asarray(f["params/activityChannel"][()]).ravel()[0]) if "params/activityChannel" in f else 1
            self.analyze_hz: float = float(np.asarray(f["params/analyzeHz"][()]).ravel()[0]) if "params/analyzeHz" in f else float("nan")
        self.n_trials: int = int(max((v.size for v in self._tnf.values()), default=0))
        # keep_trials[dmd0, trial0] True when the trial contributed frames
        self.keep_trials = np.zeros((self.n_dmds, self.n_trials), dtype=bool)
        for dmd in self.path_dmds:
            self.keep_trials[dmd - 1, : self._tnf[dmd].size] = self._tnf[dmd] > 0
        self.valid_trials: List[List[int]] = [list(1 + np.flatnonzero(self.keep_trials[d])) for d in range(self.n_dmds)]
        self.trials_to_analyze: List[int] = list(1 + np.flatnonzero(self.keep_trials.any(axis=0)))
        self.n_synapses: List[int] = []
        with self._open() as f:
            for d in range(1, self.n_dmds + 1):
                self.n_synapses.append(int(f[f"Path{d}/sources/temporal/dF_ls"].shape[2]) if f"Path{d}" in f else 0)

        self.trial_table_path = self._resolve_trial_table(self.trial_table_path)
        self.trial_epoch: Optional[np.ndarray] = None
        self.trial_epoch_source: str = "unavailable"
        self.n_epochs: int = 0
        self._trial_table: Dict[str, Any] = {}
        if self.trial_table_path is not None:
            self._trial_table = self._read_trial_table(self.trial_table_path)
            te = self._trial_table.get("epoch")
            if te is not None and te.size:
                self.trial_epoch = te.astype(int)
                self.trial_epoch_source = "trial_table.h5:/epoch"
                self.n_epochs = int(np.max(self.trial_epoch[self.trial_epoch > 0])) if np.any(self.trial_epoch > 0) else 0
        if self.trial_epoch is None and self.n_trials:
            self.trial_epoch = np.ones((self.n_trials,), dtype=int)
            self.trial_epoch_source = "assumed_single_epoch"
            self.n_epochs = 1
        self.dmd_zs = np.full((max(self.n_dmds, 2),), np.nan)
        with self._open() as f:
            for d in self.path_dmds:
                if f"Path{d}/Z_depths" in f:
                    z = np.asarray(f[f"Path{d}/Z_depths"][()]).ravel()
                    if z.size:
                        self.dmd_zs[d - 1] = float(z[0])
        self._acq_cache: Dict[int, Dict[str, Any]] = {}

    # -- lifecycle ----------------------------------------------------------
    def _open(self):
        if self._f is not None:
            class _Keep:
                def __init__(s, f): s.f = f
                def __enter__(s): return s.f
                def __exit__(s, *a): return False
            return _Keep(self._f)
        return h5py.File(self.file_path, "r")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "GiantExperimentSummary":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- trial table -------------------------------------------------------
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
            def arr(key):
                if key not in f:
                    return None
                a = np.asarray(f[key][()])
                # MATLAB (nDMDs x nTrials) -> h5py (nTrials, nDMDs); take DMD1 row for epoch/trial ix
                return a
            ep = arr("epoch")
            if ep is not None:
                ep = np.atleast_2d(ep)
                out["epoch"] = ep[:, 0] if ep.shape[1] <= 4 and ep.shape[0] >= ep.shape[1] else ep[0, :]
            tt = arr("true_trial_ix")
            if tt is not None:
                tt = np.atleast_2d(tt)
                out["true_trial_ix"] = tt[:, 0] if tt.shape[1] <= 4 and tt.shape[0] >= tt.shape[1] else tt[0, :]
            for key in ("slap2_info/first_line", "slap2_info/last_line"):
                a = arr(key)
                if a is not None:
                    out[key.split("/")[-1]] = np.atleast_2d(a)
            for key in ("slap2_info/trial_end_time_from_pc", "slap2_info/trial_start_time_inferred"):
                a = arr(key)
                if a is not None:
                    out[key.split("/")[-1]] = np.asarray(a).ravel()
            if "slap2_info/line_range_convention" in f:
                out["line_range_convention"] = _decode_str(f["slap2_info/line_range_convention"][()])
            if "filename" in f:
                try:
                    fn = f["filename"][()]
                    out["filename"] = np.vectorize(_decode_str)(fn) if np.asarray(fn).size else None
                except Exception:
                    out["filename"] = None
        return out

    @property
    def trial_table(self) -> Dict[str, Any]:
        return dict(self._trial_table)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self.schema_report.get("params", {}))

    # -- traces --------------------------------------------------------------
    def _frame_slice(self, dmd: int, trial: int) -> slice:
        if dmd not in self._tnf:
            raise IndexError(f"DMD {dmd} has no Path group in {self.file_path.name}")
        trial0 = int(trial) - 1
        if trial0 < 0 or trial0 >= self._tnf[dmd].size:
            raise IndexError(f"trial must be in [1, {self._tnf[dmd].size}], got {trial}")
        if self._tnf[dmd][trial0] <= 0:
            raise ValueError(f"Trial {trial} is invalid in DMD {dmd}")
        return slice(int(self._offsets[dmd][trial0]), int(self._offsets[dmd][trial0 + 1]))

    def get_trial_source_lines(self, dmd: int, trial: int) -> Optional[np.ndarray]:
        """1-based global source line of every frame of one trial (``frame_line_idxs``)."""
        sl = self._frame_slice(int(dmd), int(trial))
        with self._open() as f:
            lines = _read_1d(f[f"Path{int(dmd)}/frame_info/frame_line_idxs"], sl)
        return lines.astype(np.int64)

    def get_discard_frames(self, dmd: int, trial: int) -> np.ndarray:
        sl = self._frame_slice(int(dmd), int(trial))
        with self._open() as f:
            key = f"Path{int(dmd)}/frame_info/discard_frames"
            if key not in f:
                return np.zeros((sl.stop - sl.start,), dtype=bool)
            return _read_1d(f[key], sl).astype(bool)

    def get_traces(
        self,
        dmd: int = 1,
        trial: int = 1,
        signal: str = "dF",
        mode: str = "ls",
        channels: Any = None,
        t_slice: Optional[slice] = None,
        roi_inds: Optional[Sequence[int]] = None,
        drop_discarded: bool = False,
        return_frame_lines: bool = False,
        dtype: Optional[np.dtype] = None,
        force_n_channels: Optional[int] = None,
        pad_value: float = np.nan,
        drop_nan_channels: bool = False,
        squeeze_channels: bool = True,
    ):
        """Load source traces as ``(time, rois, channels)``, channel 0 = activity channel.

        ``signal`` in {"dF", "dFF", "F0"}; ``mode`` in {"ls", "denoised", "events"}.
        ``dFF`` is derived as ``dF_<mode> / F0`` (GIAnT stores no dFF). Channels are
        reordered so index 0 is ``params/activityChannel`` followed by the other
        channels in original order; all-NaN channels (not analyzed) are kept unless
        ``drop_nan_channels``.
        """
        dmd = int(dmd)
        sl = self._frame_slice(dmd, int(trial))
        if t_slice is not None:
            start = sl.start + (t_slice.start or 0)
            stop = sl.start + (t_slice.stop if t_slice.stop is not None else (sl.stop - sl.start))
            sl = slice(start, min(stop, sl.stop))
        name_map = {"ls": "dF_ls", "denoised": "dF_denoised", "events": "events"}
        with self._open() as f:
            g = f[f"Path{dmd}/sources/temporal"]
            if signal == "F0":
                raw = np.asarray(g["F0"][sl, :, :], dtype=float)
            elif signal == "dF":
                if mode not in name_map or name_map[mode] not in g:
                    raise KeyError(f"mode {mode!r} not available in {list(g.keys())}")
                raw = np.asarray(g[name_map[mode]][sl, :, :], dtype=float)
            elif signal == "dFF":
                if mode not in name_map or name_map[mode] not in g:
                    raise KeyError(f"mode {mode!r} not available in {list(g.keys())}")
                num = np.asarray(g[name_map[mode]][sl, :, :], dtype=float)
                den = np.asarray(g["F0"][sl, :, :], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    raw = np.where(np.abs(den) > 1e-8, num / den, np.nan)
            else:
                raise ValueError(f"Unknown signal {signal!r}; expected 'dF', 'dFF', or 'F0'")
        # h5py order (time, channels, sources) -> (time, rois, channels)
        raw = np.transpose(raw, (0, 2, 1))
        n_ch = raw.shape[2]
        order = [self.activity_channel - 1] + [c for c in range(n_ch) if c != self.activity_channel - 1]
        order = [c for c in order if 0 <= c < n_ch]
        raw = raw[:, :, order]
        if roi_inds is not None:
            raw = raw[:, list(roi_inds), :]
        if channels is not None:
            ch = [int(channels)] if np.isscalar(channels) else [int(c) for c in channels]
            raw = raw[:, :, ch]
        if drop_discarded:
            mask = self.get_discard_frames(dmd, trial)
            if t_slice is not None:
                mask = mask[(t_slice.start or 0):(t_slice.stop if t_slice.stop is not None else None)]
            if mask.size == raw.shape[0]:
                raw = raw[~mask, :, :]
        if drop_nan_channels and raw.shape[2] > 1:
            keep = [c for c in range(raw.shape[2]) if np.isfinite(raw[:, :, c]).any()] or [0]
            raw = raw[:, :, keep]
        if force_n_channels is not None and raw.shape[2] != force_n_channels:
            if raw.shape[2] > force_n_channels:
                raw = raw[:, :, :force_n_channels]
            else:
                pad = np.full((raw.shape[0], raw.shape[1], force_n_channels - raw.shape[2]), pad_value)
                raw = np.concatenate([raw, pad], axis=2)
        if dtype is not None:
            raw = raw.astype(dtype, copy=False)
        out = raw[:, :, 0] if (squeeze_channels and raw.shape[2] == 1) else raw
        if return_frame_lines:
            lines = self.get_trial_source_lines(dmd, trial)
            if t_slice is not None:
                lines = lines[(t_slice.start or 0):(t_slice.stop if t_slice.stop is not None else None)]
            return out, lines
        return out

    def get_user_roi_traces(self, dmd: int, trial: int, *, t_slice: Optional[slice] = None) -> Optional[np.ndarray]:
        """``user_rois/F`` for one trial as ``(time, rois, channels)`` or None."""
        dmd = int(dmd)
        sl = self._frame_slice(dmd, int(trial))
        with self._open() as f:
            key = f"Path{dmd}/user_rois/F"
            if key not in f:
                return None
            raw = np.asarray(f[key][sl, :, :], dtype=float)
        return np.transpose(raw, (0, 2, 1))

    def get_source_coords(self, dmd: int) -> np.ndarray:
        """0-based ``[z, y, x]`` centroids, ``(n_sources, 3)``."""
        with self._open() as f:
            key = f"Path{int(dmd)}/sources/spatial/coords"
            if key not in f:
                return np.zeros((0, 3))
            a = np.asarray(f[key][()], dtype=float)
        # MATLAB nSources x 3 -> h5py (3, nSources)
        return a.T if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] != 3 else a

    # -- acquisition metadata ------------------------------------------------
    def _acquisition(self, dmd: int) -> Dict[str, Any]:
        """Per-DMD acquisition metadata from raw .meta/.dat headers (cached)."""
        dmd = int(dmd)
        if dmd in self._acq_cache:
            return self._acq_cache[dmd]
        out: Dict[str, Any] = {"metadata": {}, "epochs": []}
        if self.dynamic_data_dir is not None and Path(self.dynamic_data_dir).exists():
            from vip_slap2_analysis.io.slap2_dat import (
                find_meta_file, list_acquisition_prefixes, read_dmd_acquisition_timing, read_meta_summary,
            )
            prefixes = list_acquisition_prefixes(self.dynamic_data_dir)
            for i, prefix in enumerate(prefixes, start=1):
                row: Dict[str, Any] = {"epochIdx": i, "available": True, "prefix": prefix}
                meta_path = find_meta_file(self.dynamic_data_dir, dmd, prefix=prefix)
                meta = read_meta_summary(meta_path) if meta_path is not None else {}
                try:
                    timing = read_dmd_acquisition_timing(self.dynamic_data_dir, dmd, prefix=prefix)
                    row.update({
                        "nCycles": int(timing.n_cycles),
                        "linesPerCycle": int(timing.lines_per_cycle),
                        "totalNumLines": int(timing.n_cycles * timing.lines_per_cycle),
                        "fpgaCyclePeriod_s": float(timing.median_cycle_period_s),
                        "nSourceGaps": int(timing.gap_indices.size),
                        "firstDatFile": timing.files[0] if timing.files else None,
                    })
                except FileNotFoundError:
                    row["available"] = False
                lr = meta.get("lineRateHz")
                if lr is None and row.get("fpgaCyclePeriod_s") and row.get("linesPerCycle"):
                    lr = row["linesPerCycle"] / row["fpgaCyclePeriod_s"]
                row["metadata"] = {"lineRateHz": lr, "linePeriod_s": meta.get("linePeriod_s"),
                                   "acqDuration_s": meta.get("acqDuration_s"), "acqDurationCycles": meta.get("acqDurationCycles"),
                                   "metaPath": meta.get("meta_path")}
                if row.get("totalNumLines") and lr:
                    row["durationSec"] = float(row["totalNumLines"] / lr)
                    row["cycleRateHz"] = float(lr / row["linesPerCycle"])
                out["epochs"].append(row)
            if out["epochs"]:
                first = out["epochs"][0]
                out["metadata"] = {"lineRateHz": first["metadata"].get("lineRateHz"),
                                   "linesPerCycle": first.get("linesPerCycle"),
                                   "source": "raw_dat_meta_headers"}
        self._acq_cache[dmd] = out
        return out

    def get_dmd_metadata(self, dmd: int) -> Dict[str, Any]:
        return dict(self._acquisition(dmd)["metadata"])

    def get_dmd_epoch_metadata(self, dmd: int) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._acquisition(dmd)["epochs"]]

    def get_dmd_epoch_durations_sec(self, dmd: int) -> Dict[int, float]:
        return {int(r["epochIdx"]): float(r["durationSec"]) for r in self._acquisition(dmd)["epochs"] if r.get("durationSec")}

    def get_line_rate_hz(self, dmd: Optional[int] = None) -> float:
        dmds = [int(dmd)] if dmd is not None else list(range(1, self.n_dmds + 1))
        rates = [self.get_dmd_metadata(d).get("lineRateHz") for d in dmds]
        rates = [float(r) for r in rates if r]
        return float(np.median(rates)) if rates else float("nan")

    def get_dmd_cycle_rate_hz(self, dmd: int) -> float:
        md = self.get_dmd_metadata(dmd)
        lr, lpc = md.get("lineRateHz"), md.get("linesPerCycle")
        return float(lr) / float(lpc) if lr and lpc else float("nan")


def find_giant_summary_candidates(session_dir: PathLike) -> List[Path]:
    """All ``experiment_summary.h5`` files below a session directory, newest first."""
    hits = list(Path(session_dir).rglob("experiment_summary.h5"))
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
