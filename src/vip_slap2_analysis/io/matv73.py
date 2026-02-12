from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import h5py


MatValue = Union[np.ndarray, str, float, int, bool, None, Dict[str, Any], List[Any]]


def _is_h5_ref_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype == object and x.size > 0


def _maybe_decode_matlab_string(x: Any) -> Optional[str]:
    """
    Try to decode typical MATLAB string encodings in v7.3 files.
    Returns None if not decodable.
    """
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "ignore").rstrip("\x00")

    if isinstance(x, np.ndarray):
        # MATLAB char stored as uint16/uint8
        if x.dtype == np.uint16:
            return "".join(chr(c) for c in x.flatten() if c != 0).rstrip("\x00")
        if x.dtype == np.uint8:
            return x.tobytes().decode("utf-8", "ignore").rstrip("\x00")

        # h5py can expose variable-length strings as dtype object or bytes
        if x.dtype.kind in ("S",):
            return x.tobytes().decode("utf-8", "ignore").rstrip("\x00")

    return None


@dataclass(frozen=True)
class MatV73:
    """
    Minimal, robust reader for MATLAB -v7.3 .mat (HDF5) files.
    Handles:
      - structs (HDF5 groups with field datasets)
      - cell arrays (object references)
      - numeric arrays / logicals
      - common MATLAB string encodings
    """
    path: Path

    def open(self) -> h5py.File:
        return h5py.File(self.path, "r")

    def read_var(self, var_name: str) -> MatValue:
        """
        Read a top-level variable ('summary', 'exptSummary', etc.)
        """
        with self.open() as f:
            if var_name not in f:
                raise KeyError(f"Variable '{var_name}' not found. Top-level keys: {list(f.keys())}")
            return self._read_node(f[var_name], f)

    def _read_node(self, node: Union[h5py.Dataset, h5py.Group], f: h5py.File) -> MatValue:
        if isinstance(node, h5py.Dataset):
            data = node[()]

            # If dataset is a ref array, interpret as cell array
            if _is_h5_ref_array(data):
                return self._read_cell(data, f)

            # Strings
            s = _maybe_decode_matlab_string(data)
            if s is not None:
                return s

            # Numeric/logical arrays
            if isinstance(data, np.ndarray):
                # squeeze scalar arrays
                if data.shape == (1, 1):
                    return data.squeeze().item()
                return data

            # Scalars
            return data

        # Group: interpret as struct-like container
        if isinstance(node, h5py.Group):
            out: Dict[str, Any] = {}
            for k, v in node.items():
                out[k] = self._read_node(v, f)
            return out

        raise TypeError(f"Unsupported node type: {type(node)}")

    def _read_cell(self, ref_array: np.ndarray, f: h5py.File) -> List[Any]:
        """
        Read MATLAB cell arrays saved as object references.
        MATLAB stores cells with shape (m, n); keep that shape by default.
        Return a nested list [row][col].
        """
        # ensure 2D
        arr = ref_array
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(arr.shape[0], 1)

        out: List[List[Any]] = []
        for r in range(arr.shape[0]):
            row: List[Any] = []
            for c in range(arr.shape[1]):
                ref = arr[r, c]
                if ref == 0:
                    row.append(None)
                    continue
                row.append(self._read_node(f[ref], f))
            out.append(row)

        return out
