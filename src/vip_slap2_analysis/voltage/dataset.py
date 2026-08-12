"""Dataset registries for cross-session VIP voltage analyses.

This module keeps notebook setup compact by turning project/session metadata,
processed voltage products, ROI QC, and manual longitudinal registrations into
canonical session and ROI tables.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
import pandas as pd


DEPTH_GROUP_ORDER = ["<100 µm", "100–150 µm", ">150 µm"]


def session_label(row: pd.Series) -> str:
    """Return the compact image-set/day label used in longitudinal analyses."""
    image_set = row.get("image_set", np.nan)
    day = row.get("image_set_day_index", np.nan)
    if pd.notna(image_set) and pd.notna(day):
        return f"{image_set}{int(day)}"
    return str(row.get("session_type", row.get("session_id", "")))


def dmd_depth_um(row, dmd: int) -> float:
    """Resolve cortical depth for one DMD from a registry row or metadata dict."""
    keys = (
        f"dmd{int(dmd)}_depth",
        f"dmd{int(dmd)}_depth_um",
        f"DMD{int(dmd)}_depth",
        f"DMD{int(dmd)}_depth_um",
    )
    for key in keys:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    metadata = row.get("metadata", {}) if hasattr(row, "get") else {}
    if isinstance(metadata, dict):
        for key in keys:
            if key in metadata and pd.notna(metadata[key]):
                return float(metadata[key])
    return np.nan


def depth_group(depth_um: float) -> str:
    """Assign the depth bins shared by the ephys and DoC notebooks."""
    if not np.isfinite(depth_um):
        return ""
    if depth_um < 100:
        return DEPTH_GROUP_ORDER[0]
    if depth_um <= 150:
        return DEPTH_GROUP_ORDER[1]
    return DEPTH_GROUP_ORDER[2]


def _first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def _load_json(path: Optional[Path]) -> dict:
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open("r") as f:
        return json.load(f)


def resolve_voltage_products(asset, trace_variant: str = "dff_robust_f0_trial") -> dict:
    """Resolve processed voltage/behavior products for one session asset."""
    voltage_dir = Path(asset.derived_dir) / "voltage"
    voltage_qc_dir = Path(asset.qc_dir) / "voltage"
    behavior_qc_dir = Path(asset.qc_dir) / "behavior"

    qc_json = _first_existing(
        [
            voltage_qc_dir / f"voltage_extraction_qc_{trace_variant}.json",
            *sorted(voltage_qc_dir.glob("voltage_extraction_qc_*.json")),
        ]
    )
    qc = _load_json(qc_json)
    qc_meta = qc.get("metadata", {}) if isinstance(qc.get("metadata"), dict) else {}

    event_candidates = [
        getattr(asset, "bonsai_event_log_csv", None),
        qc.get("bonsai_event_log_csv"),
        qc_meta.get("bonsai_event_log_csv"),
    ]
    event_csv = _first_existing(
        [Path(x) if x else None for x in event_candidates]
    )

    summary_candidates = [
        asset.get_asset("voltage", "summary_mat") if hasattr(asset, "get_asset") else None,
        qc.get("summary_mat"),
        qc_meta.get("summary_mat"),
    ]
    summary_mat = _first_existing(
        [Path(x) if x else None for x in summary_candidates]
    )

    encoder_pkl = None
    if getattr(asset, "harp_dir", None) is not None:
        encoder_pkl = _first_existing(
            [
                Path(asset.harp_dir) / "extracted_files" / "encoder.pkl",
                *sorted(Path(asset.harp_dir).glob("**/encoder.pkl")),
            ]
        )

    return {
        "trace_h5": voltage_dir / f"voltage_session_traces_{trace_variant}.h5",
        "single_trial_h5": voltage_dir / f"voltage_single_trial_{trace_variant}.h5",
        "mean_npz": voltage_dir / f"voltage_mean_{trace_variant}.npz",
        "sequence_npz": voltage_dir / f"voltage_sequence_{trace_variant}.npz",
        "voltage_qc_json": qc_json,
        "summary_mat": summary_mat,
        "event_csv": event_csv,
        "imaging_epochs_csv": _first_existing([behavior_qc_dir / "imaging_epochs.csv"]),
        "encoder_pkl": encoder_pkl,
    }


def build_voltage_session_table(
    registry,
    *,
    subject_ids: Optional[Sequence[int]] = None,
    paradigms: Optional[Sequence[str]] = None,
    session_types: Optional[Sequence[str]] = None,
    exclude_session_types: Optional[Sequence[str]] = None,
    session_labels: Optional[Sequence[str]] = None,
    session_ids: Optional[Sequence[str]] = None,
    trace_variant: str = "dff_robust_f0_trial",
    require_products: Sequence[str] = (
        "trace_h5",
        "single_trial_h5",
        "mean_npz",
        "sequence_npz",
        "event_csv",
        "imaging_epochs_csv",
    ),
    expected_f0_smooth_sec: Optional[float] = 60.0,
) -> pd.DataFrame:
    """Build the canonical cross-session voltage registry used by notebooks."""
    raw = registry.sessions(
        subject_ids=subject_ids,
        paradigms=paradigms,
        session_types=session_types,
        exclude_session_types=exclude_session_types,
    ).copy()
    raw["session_id"] = raw["session_id"].astype(str)

    if session_ids is not None:
        wanted = {str(x) for x in session_ids}
        raw = raw[raw["session_id"].isin(wanted)]

    raw["session_datetime"] = pd.to_datetime(
        raw["session_id"].str.extract(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")[0],
        format="%Y-%m-%d_%H-%M-%S",
        errors="coerce",
    )
    raw = raw.sort_values(["subject_id", "session_datetime"]).reset_index(drop=True)
    raw["session_order"] = raw.groupby("subject_id").cumcount()
    raw["session_label"] = raw.apply(session_label, axis=1)

    if session_labels is not None:
        raw = raw[raw["session_label"].isin(session_labels)]

    rows = []
    for _, row in raw.iterrows():
        asset = registry.resolve_assets(row)
        products = resolve_voltage_products(asset, trace_variant=trace_variant)
        missing = [
            name
            for name in require_products
            if products.get(name) is None or not Path(products[name]).exists()
        ]
        if missing:
            warnings.warn(f"{asset.session_id}: missing {missing}; skipped")
            continue

        f0_smooth_sec = np.nan
        if products["mean_npz"].exists():
            try:
                with np.load(products["mean_npz"], allow_pickle=True) as npz:
                    pkg = npz["data"][0]
                f0_smooth_sec = float(pkg.get("metadata", {}).get("robust_f0_smooth_sec", np.nan))
            except Exception as exc:
                warnings.warn(f"{asset.session_id}: could not inspect mean NPZ metadata ({exc})")

        if (
            expected_f0_smooth_sec is not None
            and np.isfinite(f0_smooth_sec)
            and not np.isclose(f0_smooth_sec, float(expected_f0_smooth_sec))
        ):
            warnings.warn(
                f"{asset.session_id}: F0 smoother is {f0_smooth_sec:g} s, "
                f"expected {float(expected_f0_smooth_sec):g} s"
            )

        rows.append(
            {
                "subject_id": str(asset.subject_id),
                "session_id": str(asset.session_id),
                "session_label": row["session_label"],
                "session_order": int(row["session_order"]),
                "session_type": row.get("session_type", ""),
                "session_datetime": row["session_datetime"],
                "dmd1_depth_um": dmd_depth_um(row, 1),
                "dmd2_depth_um": dmd_depth_um(row, 2),
                "subject_dir": Path(asset.session_dir).parent,
                "session_dir": Path(asset.session_dir),
                "derived_dir": Path(asset.derived_dir),
                "qc_dir": Path(asset.qc_dir),
                "f0_smooth_sec": f0_smooth_sec,
                **products,
            }
        )

    sessions = pd.DataFrame(rows)
    if sessions.empty:
        raise RuntimeError("No complete processed voltage sessions were found.")
    return sessions.sort_values(["subject_id", "session_order"]).reset_index(drop=True)


def _trace_roi_manifest(trace_h5: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(Path(trace_h5), "r") as h5:
        for dmd_key in sorted(k for k in h5 if str(k).startswith("DMD")):
            dmd = int(str(dmd_key).replace("DMD", ""))
            group = h5[dmd_key]
            n_time = len(group["timebase_sec"])
            dff = group["dff"]
            if dff.shape[-1] == n_time:
                n_rois = int(dff.shape[0])
            elif dff.shape[0] == n_time:
                n_rois = int(dff.shape[1])
            else:
                raise ValueError(
                    f"{trace_h5} {dmd_key}: dff shape {dff.shape} does not match timebase"
                )

            valid = (
                np.asarray(group["valid_rois_mask"][:], dtype=bool)
                if "valid_rois_mask" in group
                else np.ones(n_rois, dtype=bool)
            )
            if len(valid) != n_rois:
                warnings.warn(
                    f"{trace_h5} {dmd_key}: invalid valid_rois_mask length; using all ROIs"
                )
                valid = np.ones(n_rois, dtype=bool)

            for roi in range(n_rois):
                rows.append({"dmd": dmd, "roi": roi, "trace_h5_valid_roi": bool(valid[roi])})
    return pd.DataFrame(rows)


def _boolish(value, default: bool = False) -> bool:
    if pd.isna(value):
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_voltage_roi_table(
    sessions: pd.DataFrame,
    *,
    registration_filename: str = "roi_identity_registration.csv",
    exclude_invalid_rois: bool = True,
) -> pd.DataFrame:
    """Build one row per session-specific ROI and merge longitudinal identities."""
    rows = []
    for session in sessions.itertuples(index=False):
        manifest = _trace_roi_manifest(Path(session.trace_h5))
        for row in manifest.itertuples(index=False):
            depth = float(getattr(session, f"dmd{int(row.dmd)}_depth_um"))
            rows.append(
                {
                    "subject_id": str(session.subject_id),
                    "session_id": str(session.session_id),
                    "session_label": session.session_label,
                    "session_order": int(session.session_order),
                    "session_type": session.session_type,
                    "dmd": int(row.dmd),
                    "roi": int(row.roi),
                    "depth_um": depth,
                    "depth_group": depth_group(depth),
                    "trace_h5_valid_roi": bool(row.trace_h5_valid_roi),
                    "source_roi_label": f"DMD{int(row.dmd)}_ROI{int(row.roi)}",
                }
            )

    rois = pd.DataFrame(rows)
    registration_tables = []
    for session in sessions.drop_duplicates("subject_id").itertuples(index=False):
        path = Path(session.subject_dir) / registration_filename
        if not path.exists():
            warnings.warn(f"No manual ROI registry for mouse {session.subject_id}: {path}")
            continue
        table = pd.read_csv(
            path,
            dtype={"subject_id": str, "session_id": str, "global_cell_id": str},
        )
        table["subject_id"] = str(session.subject_id)
        for column, default in (
            ("global_cell_id", ""),
            ("valid_roi", True),
            ("excluded", False),
            ("confidence", ""),
            ("notes", ""),
        ):
            if column not in table:
                table[column] = default
        keep = [
            "subject_id",
            "session_id",
            "dmd",
            "roi",
            "global_cell_id",
            "valid_roi",
            "excluded",
            "confidence",
            "notes",
        ]
        registration_tables.append(
            table[keep].drop_duplicates(
                ["subject_id", "session_id", "dmd", "roi"], keep="last"
            )
        )

    if registration_tables:
        registration = pd.concat(registration_tables, ignore_index=True).rename(
            columns={"valid_roi": "registration_valid_roi"}
        )
        rois = rois.merge(
            registration,
            on=["subject_id", "session_id", "dmd", "roi"],
            how="left",
        )
    else:
        rois["global_cell_id"] = ""
        rois["registration_valid_roi"] = True
        rois["excluded"] = False
        rois["confidence"] = ""
        rois["notes"] = ""

    rois["global_cell_id"] = rois["global_cell_id"].fillna("").replace("nan", "")
    rois["registration_valid_roi"] = rois["registration_valid_roi"].fillna(True).map(
        lambda x: _boolish(x, default=True)
    )
    rois["excluded"] = rois["excluded"].fillna(False).map(_boolish)
    rois["valid_roi"] = rois["trace_h5_valid_roi"] & rois["registration_valid_roi"]
    rois["manually_registered"] = rois["global_cell_id"].ne("")
    rois["cell_id"] = np.where(
        rois["manually_registered"],
        rois["global_cell_id"],
        rois["session_id"]
        + "_DMD"
        + rois["dmd"].astype(str)
        + "_ROI"
        + rois["roi"].astype(str),
    )
    rois["included"] = (~rois["excluded"]) & (
        rois["valid_roi"] if exclude_invalid_rois else True
    )
    rois["depth_group"] = pd.Categorical(
        rois["depth_group"], categories=DEPTH_GROUP_ORDER, ordered=True
    )

    manual = rois[rois["manually_registered"] & rois["included"]]
    duplicate = manual.duplicated(
        ["subject_id", "session_id", "global_cell_id"], keep=False
    )
    if duplicate.any():
        bad = manual.loc[duplicate, ["subject_id", "session_id", "dmd", "roi", "global_cell_id"]]
        raise ValueError(
            "Manual registration assigns one global_cell_id to multiple ROIs in the same session:\n"
            + bad.to_string(index=False)
        )

    return rois.sort_values(
        ["subject_id", "session_order", "dmd", "roi"]
    ).reset_index(drop=True)
