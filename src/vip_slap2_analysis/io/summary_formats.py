"""Content-based identification of SLAP2 summary HDF5 products.

File names are not reliable: the voltage extractor (Voltex) is routinely run
with its output named ``experiment_summary.h5`` under a ``Voltage/`` folder,
the same basename GIAnT uses for glutamate/calcium summaries. The two schemas
are disjoint, so a cheap metadata-only sniff settles it:

* GIAnT: ``/params`` group (``activityChannel``/``analyzeHz``) and
  ``/Path{n}/sources/temporal/dF_ls``;
* Voltex: ``/format_version`` string and ``/Path{n}/sources/temporal/raw_f``
  (+ ``/Path{n}/sample_rate_hz``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import h5py

PathLike = Union[str, Path]

FORMAT_GIANT = "giant_experiment_summary_h5"
FORMAT_VOLTEX = "voltex_voltage_summary_h5"
FORMAT_UNKNOWN = "unknown_h5"
FORMAT_UNREADABLE = "unreadable"


def sniff_summary_h5(path: PathLike) -> Dict[str, object]:
    """Return ``{'format': ..., 'paths': [...], 'detail': str}`` for an HDF5 summary file."""
    p = Path(path)
    out: Dict[str, object] = {"path": str(p), "format": FORMAT_UNKNOWN, "paths": [], "detail": ""}
    try:
        with h5py.File(p, "r") as f:
            path_groups = sorted(k for k in f.keys() if k.startswith("Path") and isinstance(f[k], h5py.Group))
            out["paths"] = path_groups
            has_fv = "format_version" in f
            has_params = "params" in f and isinstance(f["params"], h5py.Group)
            raw_f = any(f"{pg}/sources/temporal/raw_f" in f for pg in path_groups)
            df_ls = any(f"{pg}/sources/temporal/dF_ls" in f for pg in path_groups)
            if raw_f and not df_ls:
                out["format"] = FORMAT_VOLTEX
                out["detail"] = "raw_f temporal datasets" + (" + format_version" if has_fv else "")
            elif df_ls and not raw_f:
                out["format"] = FORMAT_GIANT
                out["detail"] = "dF_ls temporal datasets" + (" + params" if has_params else "")
            elif raw_f and df_ls:
                out["format"] = FORMAT_UNKNOWN
                out["detail"] = "both raw_f and dF_ls present; refusing to classify"
            else:
                out["detail"] = "no recognised temporal datasets" + (" (format_version present)" if has_fv else "")
    except OSError as exc:
        out["format"] = FORMAT_UNREADABLE
        out["detail"] = f"{type(exc).__name__}: {exc}"
    return out


def classify_summary_candidates(paths) -> Dict[str, list]:
    """Group candidate HDF5 files by sniffed format: ``{format: [Path, ...]}``."""
    out: Dict[str, list] = {FORMAT_GIANT: [], FORMAT_VOLTEX: [], FORMAT_UNKNOWN: [], FORMAT_UNREADABLE: []}
    for p in paths:
        fmt = str(sniff_summary_h5(p)["format"])
        out.setdefault(fmt, []).append(Path(p))
    return out
