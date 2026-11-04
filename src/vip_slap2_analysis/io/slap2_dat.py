"""Header-only readers for raw SLAP2 ``.dat`` acquisition files and ``.meta`` files.

The SLAP2 FPGA writes one line header per scan line. Each header carries a
64-bit FPGA timestamp (``fpgaSystemClock_Hz`` ticks), a line number, flags such
as ``startOfAcq``/``startOfCycle``, and the online motion offsets. These headers
are the only source-side record of *when* each acquisition cycle happened, so
they are the independent anchor used by
:mod:`vip_slap2_analysis.common.cycle_registration` to decide whether a HARP DI3
pulse train covers the start and end of a SLAP2 acquisition.

Only headers are read (``numpy.memmap`` + gathered 32-byte windows); pixel data
are never loaded. The binary layout mirrors
``Slap2DataReader/+slap2/+util/@DataFile`` (``loadFileHeaderV2.m``,
``parseLineHeader.m``): file format version 2 only.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

MAGIC_NUMBER = 322379495
FILE_HEADER_FIELDS = (
    "firstCycleOffsetBytes",
    "lineHeaderSizeBytes",
    "laserPathIdx",
    "bytesPerCycle",
    "linesPerCycle",
    "superPixelsPerCycle",
    "dmdPixelsPerRow",
    "dmdPixelsPerColumn",
    "numChannels",
    "channelMask",
    "numSlices",
    "channelsInterleave",
    "fpgaSystemClock_Hz",
    "referenceTimestamp_lower",
    "referenceTimestamp_upper",
)
# Line-header flag bits (1-indexed in MATLAB ``bitget``; 0-indexed masks here).
FLAG_START_OF_ACQ = 1 << 0
FLAG_END_OF_ACQ = 1 << 1
FLAG_START_OF_CYCLE = 1 << 3
FLAG_END_OF_CYCLE = 1 << 4
FLAG_DMD_SYNC_PIN = 1 << 5
FLAG_SYNC_PENDING = 1 << 6

DAT_NAME_RE = re.compile(
    r"^(?P<prefix>.+?)_DMD(?P<dmd>\d+)(?P<trial>-TRIAL\d+)?(?:-CYCLE-(?P<cycle>\d+))?\.dat$",
    re.IGNORECASE,
)
META_NAME_RE = re.compile(r"^(?P<prefix>.+?)_DMD(?P<dmd>\d+)\.meta$", re.IGNORECASE)


@dataclass
class DatFileHeader:
    """Decoded SLAP2 ``.dat`` file header plus derived cycle count."""

    path: str
    file_version: int
    firstCycleOffsetBytes: int
    lineHeaderSizeBytes: int
    bytesPerCycle: int
    linesPerCycle: int
    numChannels: int
    fpgaSystemClock_Hz: int
    referenceTimestamp: int
    fileSizeBytes: int
    numCycles: int
    totalNumLines: int
    extra: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_dat_file_header(path: PathLike) -> DatFileHeader:
    """Decode the file header of a SLAP2 ``.dat`` file (format version 2)."""
    path = Path(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = np.frombuffer(f.read(4096), dtype="<u4")
        if head.size >= 3 and int(head[2]) > 4096:
            f.seek(0)
            head = np.frombuffer(f.read(int(head[2]) + 16), dtype="<u4")
    if head.size < 4 or int(head[0]) != MAGIC_NUMBER:
        raise ValueError(f"{path}: not a SLAP2 data file (bad magic number)")
    version = int(head[1])
    if version != 2:
        raise ValueError(f"{path}: unsupported SLAP2 file format version {version}")
    n_entries = int(head[2]) // 4
    if n_entries > head.size or int(head[n_entries - 1]) != MAGIC_NUMBER:
        raise ValueError(f"{path}: corrupt file header (end magic number missing)")
    pairs = head[3 : n_entries - 1].reshape(-1, 2)
    fields: Dict[str, int] = {}
    extra: Dict[str, int] = {}
    for key, value in pairs:
        key = int(key)
        if key < len(FILE_HEADER_FIELDS):
            fields[FILE_HEADER_FIELDS[key]] = int(value)
        else:
            extra[f"field{key}"] = int(value)
    required = ("firstCycleOffsetBytes", "lineHeaderSizeBytes", "bytesPerCycle", "linesPerCycle", "fpgaSystemClock_Hz")
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"{path}: file header lacks fields {missing}")
    ref = 0
    if "referenceTimestamp_lower" in fields and "referenceTimestamp_upper" in fields:
        ref = (fields["referenceTimestamp_upper"] << 32) | fields["referenceTimestamp_lower"]
    num_cycles = max(0, (size - fields["firstCycleOffsetBytes"]) // fields["bytesPerCycle"])
    return DatFileHeader(
        path=str(path),
        file_version=version,
        firstCycleOffsetBytes=fields["firstCycleOffsetBytes"],
        lineHeaderSizeBytes=fields["lineHeaderSizeBytes"],
        bytesPerCycle=fields["bytesPerCycle"],
        linesPerCycle=fields["linesPerCycle"],
        numChannels=int(fields.get("numChannels", 0)),
        fpgaSystemClock_Hz=fields["fpgaSystemClock_Hz"],
        referenceTimestamp=int(ref),
        fileSizeBytes=int(size),
        numCycles=int(num_cycles),
        totalNumLines=int(num_cycles * fields["linesPerCycle"]),
        extra=extra,
    )


def read_cycle_start_headers(path: PathLike, header: Optional[DatFileHeader] = None) -> Dict[str, np.ndarray]:
    """Return the first-line header of every cycle in one ``.dat`` file.

    Returns arrays ``timestamp`` (uint64 FPGA ticks), ``lineNumber``,
    ``acqNumber`` and ``flags`` (int64), one entry per cycle. Reading is a
    single gather of 32 bytes per cycle from a memory map.
    """
    h = header or read_dat_file_header(path)
    n = int(h.numCycles)
    out = {
        "timestamp": np.zeros((n,), dtype=np.uint64),
        "lineNumber": np.zeros((n,), dtype=np.int64),
        "acqNumber": np.zeros((n,), dtype=np.int64),
        "flags": np.zeros((n,), dtype=np.int64),
    }
    if n == 0:
        return out
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    starts = h.firstCycleOffsetBytes + np.arange(n, dtype=np.int64) * int(h.bytesPerCycle)
    window = mm[starts[:, None] + np.arange(32, dtype=np.int64)[None, :]]
    window = np.ascontiguousarray(window)
    u32 = window[:, :24].reshape(n, 6, 4).copy().view("<u4").reshape(n, 6)
    magic = u32[:, 1]
    if not np.all(magic == MAGIC_NUMBER):
        bad = int(np.flatnonzero(magic != MAGIC_NUMBER)[0])
        raise ValueError(f"{path}: corrupt line header at cycle {bad} (bad magic number)")
    out["lineNumber"] = u32[:, 2].astype(np.int64)
    out["acqNumber"] = u32[:, 3].astype(np.int64)
    out["flags"] = u32[:, 4].astype(np.int64)
    out["timestamp"] = window[:, 24:32].copy().view("<u8").reshape(n)
    return out


def _dat_sort_key(path: Path) -> tuple:
    m = DAT_NAME_RE.match(path.name)
    cycle = int(m.group("cycle")) if (m and m.group("cycle") is not None) else -1
    return (path.name.lower().split("-cycle-")[0], cycle, path.name.lower())


def find_dmd_dat_files(dynamic_data_dir: PathLike, dmd: int, *, prefix: Optional[str] = None) -> List[Path]:
    """List the ``.dat`` chunk files of one DMD in cycle order.

    When several acquisitions (prefixes) exist in the directory and ``prefix`` is
    not given, all are returned grouped by prefix then cycle; callers that need a
    single acquisition should pass ``prefix`` (e.g. ``'acquisition_090424'``).
    """
    d = Path(dynamic_data_dir)
    out: List[Path] = []
    for p in d.glob("*.dat"):
        m = DAT_NAME_RE.match(p.name)
        if not m or int(m.group("dmd")) != int(dmd):
            continue
        if prefix is not None and m.group("prefix") != prefix:
            continue
        out.append(p)
    return sorted(out, key=_dat_sort_key)


def list_acquisition_prefixes(dynamic_data_dir: PathLike) -> List[str]:
    """Return distinct acquisition prefixes (e.g. ``acquisition_090424``) in a directory."""
    d = Path(dynamic_data_dir)
    prefixes = set()
    for p in d.glob("*.dat"):
        m = DAT_NAME_RE.match(p.name)
        if m:
            prefixes.add(m.group("prefix"))
    return sorted(prefixes)


@dataclass
class DmdAcquisitionTiming:
    """Source-side cycle timing for one DMD of one acquisition, from ``.dat`` headers."""

    dmd: int
    prefix: Optional[str]
    files: List[str]
    lines_per_cycle: int
    fpga_clock_hz: int
    reference_timestamp: int
    n_cycles: int
    cycle_start_s: np.ndarray            # seconds since referenceTimestamp, one per cycle
    cycle_flags: np.ndarray
    cycle_line_number: np.ndarray
    n_start_of_acq_flags: int
    first_cycle_has_start_flag: bool
    median_cycle_period_s: float
    gap_indices: np.ndarray              # cycle index i where start[i+1]-start[i] > 1.5*median
    gap_durations_s: np.ndarray
    chunk_cycle_counts: List[int]
    has_partial_trailing_cycle: bool = False   # last chunk's size is not a whole number of cycles

    @property
    def duration_s(self) -> float:
        if self.n_cycles < 2:
            return 0.0
        return float(self.cycle_start_s[-1] - self.cycle_start_s[0] + self.median_cycle_period_s)

    def to_dict(self, *, include_arrays: bool = False) -> Dict[str, Any]:
        out = {
            "dmd": int(self.dmd),
            "prefix": self.prefix,
            "files": list(self.files),
            "lines_per_cycle": int(self.lines_per_cycle),
            "fpga_clock_hz": int(self.fpga_clock_hz),
            "reference_timestamp": int(self.reference_timestamp),
            "n_cycles": int(self.n_cycles),
            "n_start_of_acq_flags": int(self.n_start_of_acq_flags),
            "first_cycle_has_start_flag": bool(self.first_cycle_has_start_flag),
            "median_cycle_period_s": float(self.median_cycle_period_s),
            "duration_s": float(self.duration_s),
            "n_gaps": int(self.gap_indices.size),
            "gap_indices": self.gap_indices.astype(int).tolist(),
            "gap_durations_s": self.gap_durations_s.astype(float).tolist(),
            "chunk_cycle_counts": [int(x) for x in self.chunk_cycle_counts],
            "has_partial_trailing_cycle": bool(self.has_partial_trailing_cycle),
        }
        if include_arrays:
            out["cycle_start_s"] = self.cycle_start_s.astype(float).tolist()
        return out


def read_dmd_acquisition_timing(
    dynamic_data_dir: PathLike,
    dmd: int,
    *,
    prefix: Optional[str] = None,
    files: Optional[Sequence[PathLike]] = None,
    gap_factor: float = 1.5,
) -> DmdAcquisitionTiming:
    """Concatenate cycle-start timestamps across the ``-CYCLE-`` chunk files of one DMD.

    All chunk files of an acquisition share ``referenceTimestamp`` and
    ``fpgaSystemClock_Hz``; a mismatch raises because it means the files belong
    to different acquisitions.
    """
    paths = [Path(p) for p in files] if files is not None else find_dmd_dat_files(dynamic_data_dir, dmd, prefix=prefix)
    if not paths:
        raise FileNotFoundError(f"No DMD{dmd} .dat files under {dynamic_data_dir} (prefix={prefix!r})")
    if prefix is None:
        prefixes = {DAT_NAME_RE.match(p.name).group("prefix") for p in paths if DAT_NAME_RE.match(p.name)}
        if len(prefixes) > 1:
            raise ValueError(
                f"DMD{dmd} .dat files span several acquisitions {sorted(prefixes)}; pass prefix=..."
            )
        prefix = next(iter(prefixes)) if prefixes else None

    headers = [read_dat_file_header(p) for p in paths]
    h0 = headers[0]
    for h in headers[1:]:
        if h.referenceTimestamp != h0.referenceTimestamp or h.fpgaSystemClock_Hz != h0.fpgaSystemClock_Hz or h.linesPerCycle != h0.linesPerCycle:
            raise ValueError(f"{h.path}: file header inconsistent with {h0.path}; not the same acquisition")
    ts_parts, flag_parts, line_parts = [], [], []
    for p, h in zip(paths, headers):
        c = read_cycle_start_headers(p, h)
        ts_parts.append(c["timestamp"]); flag_parts.append(c["flags"]); line_parts.append(c["lineNumber"])
    ts = np.concatenate(ts_parts) if ts_parts else np.zeros((0,), dtype=np.uint64)
    flags = np.concatenate(flag_parts) if flag_parts else np.zeros((0,), dtype=np.int64)
    lines = np.concatenate(line_parts) if line_parts else np.zeros((0,), dtype=np.int64)
    clk = float(h0.fpgaSystemClock_Hz)
    rel = (ts.astype(np.float64) - float(h0.referenceTimestamp)) / clk
    if rel.size > 1:
        d = np.diff(rel)
        med = float(np.median(d))
        gap_idx = np.flatnonzero(d > float(gap_factor) * med)
        gap_dur = d[gap_idx]
    else:
        med = float("nan"); gap_idx = np.zeros((0,), dtype=int); gap_dur = np.zeros((0,), dtype=float)
    n_start = int(np.sum((flags & FLAG_START_OF_ACQ) > 0))
    return DmdAcquisitionTiming(
        dmd=int(dmd),
        prefix=prefix,
        files=[str(p) for p in paths],
        lines_per_cycle=int(h0.linesPerCycle),
        fpga_clock_hz=int(h0.fpgaSystemClock_Hz),
        reference_timestamp=int(h0.referenceTimestamp),
        n_cycles=int(rel.size),
        cycle_start_s=rel,
        cycle_flags=flags,
        cycle_line_number=lines,
        n_start_of_acq_flags=n_start,
        first_cycle_has_start_flag=bool(rel.size and (flags[0] & FLAG_START_OF_ACQ)),
        median_cycle_period_s=med,
        gap_indices=np.asarray(gap_idx, dtype=int),
        gap_durations_s=np.asarray(gap_dur, dtype=float),
        chunk_cycle_counts=[int(h.numCycles) for h in headers],
        has_partial_trailing_cycle=bool((headers[-1].fileSizeBytes - headers[-1].firstCycleOffsetBytes) % headers[-1].bytesPerCycle),
    )


# ---------------------------------------------------------------------------
# .meta (MATLAB v7.3) summary
# ---------------------------------------------------------------------------

def _matlab_datetime_ms(f, ref_name: str = "fpgaTimeReference") -> Optional[float]:
    """Best-effort decode of a MATLAB ``datetime`` saved in v7.3 (ms since epoch).

    MATLAB stores the datetime payload under ``#refs#`` as a complex double whose
    real part is milliseconds since 1970-01-01 UTC. We find a complex-valued
    dataset with a plausible epoch-ms magnitude. Returns None when not found.
    """
    try:
        refs = f["#refs#"]
    except KeyError:
        return None
    candidates: List[float] = []
    for name, obj in refs.items():
        try:
            import h5py
            if isinstance(obj, h5py.Dataset) and obj.dtype.names and "real" in obj.dtype.names:
                v = obj[()]
                real = float(np.asarray(v["real"]).ravel()[0])
                if 1.0e12 < real < 4.0e12:
                    candidates.append(real)
        except Exception:
            continue
    if not candidates:
        return None
    return float(candidates[0])


def read_meta_summary(meta_path: PathLike) -> Dict[str, Any]:
    """Read the small scalar acquisition settings from a SLAP2 ``.meta`` (MATLAB v7.3) file.

    Returned keys: ``acqDurationCycles`` (per DMD list), ``acqDuration_s``,
    ``linePeriod_s``, ``lineRateHz`` (ParsePlan), ``linesPerCycle`` (ParsePlan),
    ``acquisitionPathIdx``/``acquisitionPathName`` and, when decodable,
    ``fpgaTimeReference_iso`` (wall-clock of FPGA tick 0, *SLAP2 PC clock*).
    """
    import h5py

    meta_path = Path(meta_path)
    out: Dict[str, Any] = {"meta_path": str(meta_path)}
    with h5py.File(meta_path, "r") as f:
        def scalar(key: str):
            if key not in f:
                return None
            v = np.asarray(f[key][()]).ravel()
            return v.tolist() if v.size > 1 else (float(v[0]) if v.size else None)
        out["acqDurationCycles"] = scalar("acqDurationCycles")
        out["acqDuration_s"] = scalar("acqDuration_s")
        out["linePeriod_s"] = scalar("linePeriod_s")
        out["acquisitionPathIdx"] = scalar("acquisitionPathIdx")
        out["samplesPerLine"] = scalar("samplesPerLine")
        for key, name in (("AcquisitionContainer/ParsePlan/lineRateHz", "lineRateHz"),
                          ("AcquisitionContainer/ParsePlan/linesPerCycle", "linesPerCycle")):
            if key in f:
                v = np.asarray(f[key][()]).ravel()
                out[name] = float(v[0]) if v.size else None
        if "acquisitionPathName" in f:
            raw = np.asarray(f["acquisitionPathName"][()]).ravel()
            try:
                out["acquisitionPathName"] = "".join(chr(int(c)) for c in raw if int(c) != 0)
            except Exception:
                out["acquisitionPathName"] = None
        ms = _matlab_datetime_ms(f)
        if ms is not None:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=ms)
            out["fpgaTimeReference_epoch_ms"] = ms
            out["fpgaTimeReference_iso_utc"] = dt.isoformat()
    if out.get("linePeriod_s") and not out.get("lineRateHz"):
        out["lineRateHz"] = 1.0 / float(out["linePeriod_s"])
    return out


def find_meta_file(dynamic_data_dir: PathLike, dmd: int, *, prefix: Optional[str] = None) -> Optional[Path]:
    """Locate ``<prefix>_DMD<n>.meta`` in a dynamic_data directory."""
    d = Path(dynamic_data_dir)
    hits = []
    for p in d.glob("*.meta"):
        m = META_NAME_RE.match(p.name)
        if m and int(m.group("dmd")) == int(dmd) and (prefix is None or m.group("prefix") == prefix):
            hits.append(p)
    return sorted(hits)[0] if hits else None
