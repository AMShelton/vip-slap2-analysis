from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import h5py

def bytes_to_str(x: Any) -> Any:
    """
    Best-effort conversion of common MATLAB v7.3 string encodings.

    Returns:
      - python str if convertible
      - otherwise returns x unchanged
    """
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "ignore").rstrip("\x00")

    if isinstance(x, np.ndarray):
        # MATLAB char arrays often stored as uint16
        if x.dtype == np.uint16:
            s = "".join(chr(int(c)) for c in x.flatten() if int(c) != 0)
            return s.rstrip("\x00")

        # sometimes uint8 bytes
        if x.dtype == np.uint8:
            try:
                return x.tobytes().decode("utf-8", "ignore").rstrip("\x00")
            except Exception:
                return x

        # fixed-width bytes
        if x.dtype.kind == "S":
            try:
                return x.tobytes().decode("utf-8", "ignore").rstrip("\x00")
            except Exception:
                return x

        # h5py vlen strings sometimes appear as object array of bytes
        if x.dtype == object:
            # try a very common case: shape (1,1) containing bytes
            if x.size == 1 and isinstance(x.flat[0], (bytes, np.bytes_)):
                return bytes_to_str(x.flat[0])

    return x


def _is_ref_dtype(dset: h5py.Dataset) -> bool:
    # MATLAB cell arrays and object refs are stored as HDF5 references
    try:
        return dset.dtype == h5py.ref_dtype
    except Exception:
        return False


def _safe_2d_index(shape: Tuple[int, ...], i: int, j: int) -> Tuple[int, int]:
    """
    MATLAB sometimes stores cell arrays as (1, N) or (N, 1) or (N, M).
    This helper lets callers ask for (i,j) but tolerates degenerate dims.
    """
    if len(shape) == 1:
        # treat as (N,1)
        return (i, 0)
    if len(shape) >= 2:
        return (i, j)
    raise ValueError(f"Unsupported shape for 2D index: {shape}")


@dataclass
class MatV73File:
    """
    Thin wrapper around an HDF5-backed MATLAB v7.3 MAT file.

    Key goals:
      - fast: avoid reading large arrays unless explicitly requested
      - ergonomic: dereference MATLAB object refs / cell arrays
      - robust: handle common MATLAB string encodings
    """
    path: Union[str, Path]
    keep_open: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._fh: Optional[h5py.File] = None
        if self.keep_open:
            self.open()

    def open(self) -> h5py.File:
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
        return self._fh

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "MatV73File":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.keep_open:
            self.close()

    @property
    def f(self) -> h5py.File:
        return self.open()

    # ---------- low-level helpers ----------

    def has(self, *keys: str) -> bool:
        """
        Check existence of nested keys, e.g. has("summary","E")
        """
        node: Any = self.f
        for k in keys:
            if k not in node:
                return False
            node = node[k]
        return True

    def g(self, *keys: str) -> h5py.Group:
        """
        Return group at nested path.
        """
        node: Any = self.f
        for k in keys:
            node = node[k]
        if not isinstance(node, h5py.Group):
            raise TypeError(f"Expected Group at {'/'.join(keys)}, got {type(node)}")
        return node

    def d(self, *keys: str) -> h5py.Dataset:
        """
        Return dataset at nested path.
        """
        node: Any = self.f
        for k in keys:
            node = node[k]
        if not isinstance(node, h5py.Dataset):
            raise TypeError(f"Expected Dataset at {'/'.join(keys)}, got {type(node)}")
        return node

    def deref(self, ref: h5py.Reference) -> Union[h5py.Group, h5py.Dataset]:
        """
        Dereference an HDF5 object reference.
        """
        return self.f[ref]

    def cell_ref(self, cell_dset: h5py.Dataset, i: int, j: int = 0) -> Optional[h5py.Reference]:
        """
        Read a single object ref from a MATLAB cell array stored as a ref-typed dataset.
        Returns None for null refs.
        """
        if not _is_ref_dtype(cell_dset):
            raise TypeError("cell_ref called on non-ref dataset")

        ii, jj = _safe_2d_index(cell_dset.shape, i, j)
        ref = cell_dset[ii, jj]
        # null ref can come through as 0 or an invalid reference
        if ref is None:
            return None
        try:
            # h5py.Reference is truthy even when null, so check repr-ish
            if int(ref) == 0:  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        return ref

    def read_scalar(self, dset: h5py.Dataset) -> Any:
        """
        Read small scalar-ish items with minimal overhead.
        """
        x = dset[()]
        x = bytes_to_str(x)
        # squeeze trivial MATLAB scalar arrays
        if isinstance(x, np.ndarray) and x.shape == (1, 1):
            return bytes_to_str(x.squeeze().item())
        return x
