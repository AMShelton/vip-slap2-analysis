"""Synthetic SLAP2/HARP/GIAnT/Voltex fixtures with known ground truth.

Everything here mirrors the on-disk conventions verified in the MATLAB writers:

* ``.dat``: format v2 file header (field/value pairs), one 40-byte line header per
  line (``lineSizeBytes, magic, lineNumber, acqNumber, flags, xOff, yOff,
  timestamp(u64), zOffset, zFeedback``) followed by ``n_pixels`` int16 samples.
* HARP register dumps: 13-byte ``DigitalInputState`` (reg 32) and 18-byte
  ``AnalogData`` (reg 44) messages with ``u32 seconds + u16 32-us ticks``.
* HDF5 written by MATLAB ``saveStructToH5``: dims are ``fliplr(size)``, so a
  MATLAB ``(a, b, c)`` array is created here with shape ``(c, b, a)``; ``/row_major`` = 0.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

MAGIC = 322379495
FPGA_CLOCK_HZ = 200_000_000
LINE_RATE_HZ = 10686.6669921875
FIELD_IDS = {
    "firstCycleOffsetBytes": 0, "lineHeaderSizeBytes": 1, "laserPathIdx": 2, "bytesPerCycle": 3,
    "linesPerCycle": 4, "superPixelsPerCycle": 5, "dmdPixelsPerRow": 6, "dmdPixelsPerColumn": 7,
    "numChannels": 8, "channelMask": 9, "numSlices": 10, "channelsInterleave": 11,
    "fpgaSystemClock_Hz": 12, "referenceTimestamp_lower": 13, "referenceTimestamp_upper": 14,
}


# ---------------------------------------------------------------------------
# raw .dat / .meta
# ---------------------------------------------------------------------------

def write_dat_chunks(
    dynamic_data_dir: Path,
    *,
    prefix: str,
    dmd: int,
    n_cycles: int,
    lines_per_cycle: int,
    reference_timestamp: int,
    line_period_s: float = 1.0 / LINE_RATE_HZ,
    cycles_per_chunk: int = 1000,
    n_pixels: int = 8,
    cycle_jitter_ticks: int = 0,
    seed: int = 0,
) -> Tuple[List[Path], np.ndarray]:
    """Write ``<prefix>_DMD<n>-TRIAL000001-CYCLE-XXXXXX.dat`` chunks; return (paths, cycle_start_s)."""
    rng = np.random.default_rng(seed)
    dynamic_data_dir.mkdir(parents=True, exist_ok=True)
    line_hdr = 40
    line_bytes = line_hdr + 2 * n_pixels
    bytes_per_cycle = line_bytes * lines_per_cycle
    fields = {
        "firstCycleOffsetBytes": 0, "lineHeaderSizeBytes": line_hdr, "laserPathIdx": dmd - 1,
        "bytesPerCycle": bytes_per_cycle, "linesPerCycle": lines_per_cycle, "superPixelsPerCycle": n_pixels,
        "dmdPixelsPerRow": 1280, "dmdPixelsPerColumn": 800, "numChannels": 1, "channelMask": 1,
        "numSlices": 1, "channelsInterleave": 0, "fpgaSystemClock_Hz": FPGA_CLOCK_HZ,
        "referenceTimestamp_lower": reference_timestamp & 0xFFFFFFFF,
        "referenceTimestamp_upper": reference_timestamp >> 32,
    }
    pairs = [(FIELD_IDS[k], v) for k, v in fields.items()]
    header_words = [MAGIC, 2, 4 * (3 + 2 * len(pairs) + 1)] + [w for p in pairs for w in p] + [MAGIC]
    header = struct.pack("<%dI" % len(header_words), *header_words)
    fields["firstCycleOffsetBytes"] = len(header)
    # rebuild with correct offset
    pairs = [(FIELD_IDS[k], v) for k, v in fields.items()]
    header_words = [MAGIC, 2, 4 * (3 + 2 * len(pairs) + 1)] + [w for p in pairs for w in p] + [MAGIC]
    header = struct.pack("<%dI" % len(header_words), *header_words)
    assert len(header) == fields["firstCycleOffsetBytes"]

    ticks_per_line = line_period_s * FPGA_CLOCK_HZ
    cycle_start = np.zeros(n_cycles, dtype=np.float64)
    paths: List[Path] = []
    cyc = 0
    pix = np.zeros(n_pixels, dtype=np.int16).tobytes()
    while cyc < n_cycles:
        n_here = min(cycles_per_chunk, n_cycles - cyc)
        path = dynamic_data_dir / f"{prefix}_DMD{dmd}-TRIAL000001-CYCLE-{cyc:06d}.dat"
        with open(path, "wb") as f:
            f.write(header)
            for c in range(cyc, cyc + n_here):
                base = reference_timestamp + int(round(c * lines_per_cycle * ticks_per_line))
                if cycle_jitter_ticks:
                    base += int(rng.integers(-cycle_jitter_ticks, cycle_jitter_ticks + 1))
                cycle_start[c] = (base - reference_timestamp) / FPGA_CLOCK_HZ
                for l in range(lines_per_cycle):
                    flags = 0
                    if l == 0:
                        flags |= 1 << 3
                        if c == 0:
                            flags |= 1
                    if l == lines_per_cycle - 1:
                        flags |= 1 << 4
                    ts = base + int(round(l * ticks_per_line))
                    f.write(struct.pack("<5I2hQ2f", line_bytes, MAGIC, l, 1, flags, 0, 0, ts, 0.0, 0.0))
                    f.write(pix)
        paths.append(path)
        cyc += n_here
    return paths, cycle_start


def write_meta_file(dynamic_data_dir: Path, *, prefix: str, dmd: int, n_cycles_by_dmd: Sequence[int],
                    lines_per_cycle: int, line_rate_hz: float = LINE_RATE_HZ, duration_s: float = 45.0) -> Path:
    """Write a minimal MATLAB-v7.3-like ``.meta`` with the scalars the reader uses."""
    path = dynamic_data_dir / f"{prefix}_DMD{dmd}.meta"
    with h5py.File(path, "w") as f:
        f.create_dataset("acqDurationCycles", data=np.asarray(n_cycles_by_dmd, dtype=float).reshape(1, -1))
        f.create_dataset("acqDuration_s", data=np.array([[duration_s]]))
        f.create_dataset("linePeriod_s", data=np.array([[1.0 / line_rate_hz]]))
        f.create_dataset("acquisitionPathIdx", data=np.array([[float(dmd)]]))
        g = f.create_group("AcquisitionContainer/ParsePlan")
        g.create_dataset("lineRateHz", data=np.array([[line_rate_hz]]))
        g.create_dataset("linesPerCycle", data=np.array([[lines_per_cycle]], dtype=np.uint64))
        refs = f.create_group("#refs#")
        dt = np.dtype([("real", "<f8"), ("imag", "<f8")])
        refs.create_dataset("Kf", data=np.array([(1.7930917e12, 0.0)], dtype=dt))
    return path


# ---------------------------------------------------------------------------
# HARP register dumps
# ---------------------------------------------------------------------------

def _harp_msg(address: int, payload_type: int, t_s: float, payload: bytes) -> bytes:
    sec = int(np.floor(t_s))
    frac = int(round((t_s - sec) / 32e-6))
    if frac >= 31250:
        sec += 1
        frac -= 31250
    body = bytes([address, 0, payload_type]) + struct.pack("<IH", sec, frac) + payload
    length = len(body) + 1  # + checksum
    msg = bytes([1, length]) + body
    checksum = sum(msg) & 0xFF
    return msg + bytes([checksum])


def write_harp_di3(path: Path, rise_times: Sequence[float], *, pulse_width_s: float = 0.001,
                   record_start_s: float, extra_initial_low_msg: bool = True) -> None:
    """``Behavior_32.bin`` with DI3 (bit 3) pulses; initial low state message at record_start."""
    msgs = []
    if extra_initial_low_msg:
        msgs.append(_harp_msg(32, 0x11, record_start_s, bytes([0])))
    for t in rise_times:
        msgs.append(_harp_msg(32, 0x11, t, bytes([8])))
        msgs.append(_harp_msg(32, 0x11, t + pulse_width_s, bytes([0])))
    path.write_bytes(b"".join(msgs))


def write_harp_analog(path: Path, *, record_start_s: float, record_end_s: float, rate_hz: float = 1000.0,
                      photodiode: Optional[np.ndarray] = None) -> np.ndarray:
    """``Behavior_44.bin`` with 3 int16 channels at ``rate_hz``; returns sample times."""
    t = np.arange(record_start_s, record_end_s, 1.0 / rate_hz)
    pd_vals = photodiode if photodiode is not None else np.zeros(t.size, dtype=np.int16)
    msgs = [_harp_msg(44, 0x92, float(ti), struct.pack("<3h", int(pd_vals[i]), 0, 0)) for i, ti in enumerate(t)]
    path.write_bytes(b"".join(msgs))
    return t


# ---------------------------------------------------------------------------
# GIAnT experiment_summary.h5 + trial_table.h5 (MATLAB-flipped dims)
# ---------------------------------------------------------------------------

def write_giant_experiment_summary(
    path: Path,
    *,
    trial_frames: Sequence[int],
    n_sources: int,
    n_channels: int = 2,
    activity_channel: int = 1,
    analyze_hz: float = 200.0,
    frame_lines_by_trial: Optional[Sequence[np.ndarray]] = None,
    dF_by_trial: Optional[Sequence[np.ndarray]] = None,   # each (frames, sources) for the activity channel
    F0_value: float = 10.0,
    n_dmds: int = 1,
) -> None:
    """Write a GIAnT-like summary: ``/Path1`` (and optionally ``/Path2`` copy)."""
    tnf = np.asarray(trial_frames, dtype=np.int32)
    total = int(tnf.sum())
    if frame_lines_by_trial is None:
        frame_lines = np.arange(1, total + 1, dtype=np.int32)
    else:
        frame_lines = np.concatenate([np.asarray(x, dtype=np.int32) for x in frame_lines_by_trial])
    assert frame_lines.size == total
    with h5py.File(path, "w") as f:
        f.create_dataset("row_major", data=np.array([[0]], dtype=np.uint8))
        p = f.create_group("params")
        p.create_dataset("activityChannel", data=np.array([[float(activity_channel)]]))
        p.create_dataset("analyzeHz", data=np.array([[analyze_hz]]))
        p.create_dataset("numChannels", data=np.array([[float(n_channels)]]))
        for d in range(1, n_dmds + 1):
            g = f.create_group(f"Path{d}")
            fi = g.create_group("frame_info")
            fi.create_dataset("trial_num_frames", data=tnf.reshape(1, -1))          # MATLAB nTrials x 1 -> (1, nTrials)
            fi.create_dataset("frame_line_idxs", data=frame_lines.reshape(1, -1))   # (1, totalFrames)
            fi.create_dataset("discard_frames", data=np.zeros((1, total), dtype=np.int8))
            st = g.create_group("sources/temporal")
            dF = np.full((total, n_channels, n_sources), np.nan)  # h5py (frames, channels, sources)
            if dF_by_trial is None:
                vals = np.arange(total, dtype=float)[:, None] + 100.0 * np.arange(n_sources)[None, :]
            else:
                vals = np.concatenate([np.asarray(x, dtype=float) for x in dF_by_trial], axis=0)
            dF[:, activity_channel - 1, :] = vals
            st.create_dataset("dF_ls", data=dF)
            st.create_dataset("dF_denoised", data=dF * 0.5)
            st.create_dataset("events", data=np.zeros_like(dF))
            F0 = np.full((total, n_channels, n_sources), np.nan)
            F0[:, activity_channel - 1, :] = F0_value
            st.create_dataset("F0", data=F0)
            sp = g.create_group("sources/spatial")
            sp.create_dataset("coords", data=np.stack([np.zeros(n_sources), np.arange(n_sources) + 10.0, np.arange(n_sources) + 20.0]))  # (3, nSources)
            ur = g.create_group("user_rois")
            ur.create_dataset("F", data=np.ones((total, n_channels, 2)))


def write_giant_trial_table(path: Path, *, epoch_by_trial: Sequence[int], first_line: Sequence[float],
                            last_line: Sequence[float], n_dmds: int = 2, inclusive: bool = False) -> None:
    ep = np.asarray(epoch_by_trial, dtype=float)
    n = ep.size
    with h5py.File(path, "w") as f:
        f.create_dataset("row_major", data=np.array([[0]], dtype=np.uint8))
        f.create_dataset("epoch", data=np.tile(ep, (1, 1)).T.repeat(n_dmds, axis=1) if False else np.tile(ep.reshape(n, 1), (1, n_dmds)))  # (nTrials, nDMDs)
        f.create_dataset("true_trial_ix", data=np.tile(np.arange(1, n + 1, dtype=float).reshape(n, 1), (1, n_dmds)))
        si = f.create_group("slap2_info")
        si.create_dataset("first_line", data=np.tile(np.asarray(first_line, dtype=float).reshape(n, 1), (1, n_dmds)))
        si.create_dataset("last_line", data=np.tile(np.asarray(last_line, dtype=float).reshape(n, 1), (1, n_dmds)))
        if inclusive:
            si.create_dataset("line_range_convention", data=np.bytes_("inclusive"))


# ---------------------------------------------------------------------------
# Voltex VoltageSummary.h5
# ---------------------------------------------------------------------------

def write_voltex_summary(
    path: Path,
    *,
    trial_lines: Sequence[int],
    trial_epoch: Sequence[int],
    n_rois: int,
    sample_rate_hz: float = LINE_RATE_HZ,
    first_lines: Optional[Sequence[int]] = None,
    raw_by_trial: Optional[Sequence[np.ndarray]] = None,  # each (lines, rois)
    motion_censor: Optional[np.ndarray] = None,
    indicator: str = "ASAP7y",
    n_dmds: int = 1,
) -> None:
    tnf = np.asarray(trial_lines, dtype=np.int32)
    total = int(tnf.sum())
    if first_lines is None:
        first_lines = np.concatenate([[1], 1 + np.cumsum(tnf)[:-1]])
    lines = np.concatenate([np.arange(fl, fl + n, dtype=np.int32) for fl, n in zip(first_lines, tnf)])
    te = np.asarray(trial_epoch, dtype=np.int32)
    sample_epoch = np.concatenate([np.full(n, e, dtype=np.int32) for n, e in zip(tnf, te)])
    with h5py.File(path, "w") as f:
        f.create_dataset("format_version", data=np.bytes_("0.2.3"))
        f.create_dataset("row_major", data=np.array([[0]], dtype=np.uint8))
        p = f.create_group("params")
        p.create_dataset("indicator", data=np.bytes_(indicator))
        p.create_dataset("f0Percentile", data=np.array([[50.0]]))
        for d in range(1, n_dmds + 1):
            g = f.create_group(f"Path{d}")
            g.create_dataset("sample_rate_hz", data=np.array([[sample_rate_hz]]))
            g.create_dataset("Z_depths", data=np.array([[0.5]]))
            fi = g.create_group("frame_info")
            fi.create_dataset("trial_num_frames", data=tnf.reshape(1, -1))
            fi.create_dataset("trial_epoch", data=te.reshape(1, -1))
            fi.create_dataset("frame_line_idxs", data=lines.reshape(1, -1))
            fi.create_dataset("sample_epoch", data=sample_epoch.reshape(1, -1))
            if motion_censor is not None:
                fi.create_dataset("motion_censor", data=np.asarray(motion_censor, dtype=np.uint8).reshape(1, -1))
            st = g.create_group("sources/temporal")
            if raw_by_trial is None:
                raw = (np.arange(total, dtype=np.float32)[:, None] + 1000.0 * np.arange(n_rois, dtype=np.float32)[None, :])
            else:
                raw = np.concatenate([np.asarray(x, dtype=np.float32) for x in raw_by_trial], axis=0)
            st.create_dataset("raw_f", data=raw.reshape(total, 1, n_rois), chunks=(min(total, 1000), 1, 1))
            st.create_dataset("f0", data=np.full((total, 1, n_rois), 100.0, dtype=np.float32))
            dff = st.create_dataset("dff", data=(raw.reshape(total, 1, n_rois) - 100.0) / 100.0)
            dff.attrs["sign_convention"] = np.bytes_("depolarization_positive")
            dff.attrs["indicator_direction"] = np.bytes_("decreases")
            dff.attrs["indicator_name"] = np.bytes_(indicator)
            sp = g.create_group("sources/spatial")
            sp.create_dataset("profiles", data=np.ones((4, 3, 1, n_rois), dtype=np.float32))  # MATLAB nROIs x 1 x 3 x 4
            sp.create_dataset("coords", data=np.zeros((3, n_rois)))
            sp.create_dataset("source_roi_index", data=np.arange(1, n_rois + 1, dtype=np.int32).reshape(1, -1))
            sp.create_dataset("roi_pixel_count", data=np.full((1, n_rois), 12, dtype=np.int32))


def write_voltex_trial_table(path: Path, *, epoch_by_trial: Sequence[int], first_line: Sequence[int], last_line: Sequence[int]) -> None:
    write_giant_trial_table(path, epoch_by_trial=epoch_by_trial, first_line=first_line, last_line=last_line, n_dmds=2, inclusive=True)
