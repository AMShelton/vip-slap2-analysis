"""Minimal decoder for HARP register binaries (``Behavior_<reg>.bin``).

The behavior pipeline normally reads HARP data through the ``harp`` package and
caches ``extracted_files/HARP_df.csv`` / ``photodiode.pkl``. Cycle registration
needs two things that those caches do not preserve reliably: the exact
*recording* start/end times of the digital-input stream (independent of any
pulses) and a dependency-free read path for sessions whose caches are missing.

HARP message layout (binary protocol 1.x):
``[msgType u8][length u8][address u8][port u8][payloadType u8]
[timestamp: seconds u32, 32-us ticks u16 (present when payloadType & 0x10)]
[payload ...][checksum u8]``; ``length`` counts every byte after it except the
``msgType``/``length`` pair itself, so a message occupies ``length + 2`` bytes.
``DigitalInputState`` (register 32) is a one-byte bitmask where DI3 is bit 3.
``AnalogData`` (register 44) carries three signed 16-bit values; AnalogInput0
(photodiode) is the first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
HARP_TICK_S = 32e-6
DIGITAL_INPUT_STATE_REGISTER = 32
ANALOG_DATA_REGISTER = 44
DI3_MASK = 0x08


def _payload_itemsize(payload_type: int) -> int:
    base = payload_type & 0x0F
    return {1: 1, 2: 2, 4: 4, 8: 8}.get(base, 1)


def read_harp_register(path: PathLike) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Decode one HARP register file into ``(timestamps_s, payload_values, info)``.

    ``payload_values`` is ``(n_messages, n_elements)`` with the register's native
    signed/unsigned integer type. Messages without a timestamp get NaN times.
    Messages are assumed to be of constant length (true for register dumps).
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size < 2:
        return np.zeros((0,)), np.zeros((0, 1), dtype=np.int64), {"n_messages": 0}
    length = int(raw[1])
    msg_len = length + 2
    if msg_len <= 0 or raw.size % msg_len != 0:
        # Fall back to a slow variable-length walk if the dump is not uniform.
        return _read_harp_register_slow(raw)
    n = raw.size // msg_len
    msgs = raw.reshape(n, msg_len)
    payload_type = int(msgs[0, 4])
    has_ts = bool(payload_type & 0x10)
    itemsize = _payload_itemsize(payload_type)
    signed = bool(payload_type & 0x80)
    float_type = bool(payload_type & 0x40)
    p0 = 5 + (6 if has_ts else 0)
    payload_bytes = msg_len - p0 - 1
    n_elem = max(1, payload_bytes // itemsize)
    if has_ts:
        sec = msgs[:, 5:9].copy().view("<u4").reshape(n).astype(np.float64)
        frac = msgs[:, 9:11].copy().view("<u2").reshape(n).astype(np.float64)
        t = sec + frac * HARP_TICK_S
    else:
        t = np.full((n,), np.nan)
    if float_type and itemsize == 4:
        dt = "<f4"
    else:
        dt = {1: "i1" if signed else "u1", 2: "<i2" if signed else "<u2", 4: "<i4" if signed else "<u4", 8: "<i8" if signed else "<u8"}[itemsize]
    vals = msgs[:, p0:p0 + n_elem * itemsize].copy().view(dt).reshape(n, n_elem)
    info = {
        "n_messages": int(n),
        "address": int(msgs[0, 2]),
        "payload_type": payload_type,
        "message_length_bytes": int(msg_len),
        "n_elements": int(n_elem),
        "has_timestamp": has_ts,
    }
    return t, vals, info


def _read_harp_register_slow(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    times = []
    vals = []
    i = 0
    n = raw.size
    addr = None
    while i + 5 <= n:
        length = int(raw[i + 1])
        total = length + 2
        if total <= 0 or i + total > n:
            break
        payload_type = int(raw[i + 4])
        addr = int(raw[i + 2])
        p = i + 5
        if payload_type & 0x10:
            sec = int.from_bytes(bytes(raw[p:p + 4]), "little")
            frac = int.from_bytes(bytes(raw[p + 4:p + 6]), "little")
            times.append(sec + frac * HARP_TICK_S)
            p += 6
        else:
            times.append(np.nan)
        vals.append(int(raw[p]))
        i += total
    return np.asarray(times, dtype=float), np.asarray(vals, dtype=np.int64).reshape(-1, 1), {
        "n_messages": len(times), "address": addr, "variable_length": True,
    }


@dataclass
class DigitalInputRecording:
    """DI3 edges and recording bounds from a HARP ``DigitalInputState`` dump."""

    record_start_s: float
    record_end_s: float
    initial_state_high: bool
    rise_times_s: np.ndarray
    fall_times_s: np.ndarray
    n_messages: int
    source: str

    @property
    def lead_s(self) -> float:
        """Recording time before the first rising edge (NaN when no edges)."""
        return float(self.rise_times_s[0] - self.record_start_s) if self.rise_times_s.size else float("nan")

    @property
    def tail_s(self) -> float:
        """Recording time after the last rising edge (NaN when no edges)."""
        return float(self.record_end_s - self.rise_times_s[-1]) if self.rise_times_s.size else float("nan")

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "record_start_s": float(self.record_start_s),
            "record_end_s": float(self.record_end_s),
            "record_duration_s": float(self.record_end_s - self.record_start_s),
            "initial_state_high": bool(self.initial_state_high),
            "n_rising_edges": int(self.rise_times_s.size),
            "n_falling_edges": int(self.fall_times_s.size),
            "first_rise_s": float(self.rise_times_s[0]) if self.rise_times_s.size else None,
            "last_rise_s": float(self.rise_times_s[-1]) if self.rise_times_s.size else None,
            "lead_s": None if not self.rise_times_s.size else self.lead_s,
            "tail_s": None if not self.rise_times_s.size else self.tail_s,
            "n_messages": int(self.n_messages),
        }


def read_digital_input_edges(
    register_path: PathLike,
    *,
    bit_mask: int = DI3_MASK,
    record_end_from: Optional[PathLike] = None,
) -> DigitalInputRecording:
    """Decode DI edges (default DI3) from ``Behavior_32.bin``.

    ``DigitalInputState`` messages are event-driven: a message is emitted on
    every input change, so the last message marks the last *transition*, not the
    end of the recording. Pass ``record_end_from`` (typically ``Behavior_44.bin``,
    the periodic analog stream) to bound the recording end more tightly; when it
    is given, the recording start is also taken as the earlier of the two
    streams' first messages.
    """
    t, vals, info = read_harp_register(register_path)
    if t.size == 0:
        raise ValueError(f"{register_path}: no HARP messages decoded")
    order = np.argsort(t, kind="stable")
    t = t[order]
    state = (vals[order, 0].astype(np.int64) & int(bit_mask)) > 0
    # The first message is the initial state, not a transition: a recording that starts
    # with DI3 already high is mid-pulse, and that partial pulse is not a cycle start.
    prev = np.r_[state[0], state[:-1]]
    rises = t[state & ~prev]
    falls = t[~state & prev]
    start = float(t[0])
    end = float(t[-1])
    if record_end_from is not None and Path(record_end_from).exists():
        t2, _, _ = read_harp_register(record_end_from)
        finite = t2[np.isfinite(t2)]
        if finite.size:
            start = min(start, float(finite.min()))
            end = max(end, float(finite.max()))
    return DigitalInputRecording(
        record_start_s=start,
        record_end_s=end,
        initial_state_high=bool(state[0]),
        rise_times_s=np.asarray(rises, dtype=float),
        fall_times_s=np.asarray(falls, dtype=float),
        n_messages=int(t.size),
        source=str(register_path),
    )


def read_digital_input_edges_from_csv(harp_df_csv: PathLike, *, column: str = "DI3", chunk_size: int = 300_000) -> DigitalInputRecording:
    """Same as :func:`read_digital_input_edges` but from a cached ``HARP_df.csv``."""
    import pandas as pd

    rises, falls = [], []
    prev = False
    first = None
    last = None
    initial = None
    n = 0
    for chunk in pd.read_csv(harp_df_csv, usecols=["time", column], chunksize=chunk_size):
        if not len(chunk):
            continue
        tt = pd.to_numeric(chunk["time"], errors="coerce").to_numpy(dtype=float)
        col = chunk[column]
        if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            st = col.fillna(0).astype(float).to_numpy() > 0.5
        else:
            st = col.astype(str).str.strip().str.lower().isin(("true", "t", "1", "1.0")).to_numpy()
        ok = np.isfinite(tt)
        if first is None and ok.any():
            first = float(tt[ok][0]); initial = bool(st[ok][0])
        if ok.any():
            last = float(tt[ok][-1])
        before = np.r_[prev, st[:-1]]
        rises.append(tt[st & ~before & ok]); falls.append(tt[~st & before & ok])
        prev = bool(st[-1]); n += len(chunk)
    if first is None:
        raise ValueError(f"{harp_df_csv}: no finite HARP timestamps")
    return DigitalInputRecording(
        record_start_s=first, record_end_s=float(last), initial_state_high=bool(initial),
        rise_times_s=np.concatenate(rises) if rises else np.zeros((0,)),
        fall_times_s=np.concatenate(falls) if falls else np.zeros((0,)),
        n_messages=n, source=str(harp_df_csv),
    )


def find_register_file(harp_dir: PathLike, register: int) -> Optional[Path]:
    """Locate ``*_<register>.bin`` inside a ``*.harp`` directory."""
    d = Path(harp_dir)
    hits = sorted(d.glob(f"*_{int(register)}.bin"))
    return hits[0] if hits else None
