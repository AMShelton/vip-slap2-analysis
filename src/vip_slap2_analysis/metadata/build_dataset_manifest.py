from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SESSION_RE = re.compile(r"(?P<mouse>\d{6})_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})")


@dataclass
class ManifestConfig:
    dataset_root: Path
    output_csv: str = "dataset_manifest.csv"
    output_json: str = "dataset_manifest.json"
    output_md: str = "DATASET_OVERVIEW.md"

    # Candidate relative paths / filenames to search for QC and analysis outputs.
    glutamate_qc_candidates: Tuple[str, ...] = (
        "analysis/glutamate/glutamate_qc.json",
        "analysis/qc/glutamate_qc.json",
        "glutamate_qc.json",
    )
    calcium_qc_candidates: Tuple[str, ...] = (
        "analysis/calcium/calcium_qc.json",
        "analysis/qc/calcium_qc.json",
        "calcium_qc.json",
    )
    behavior_qc_candidates: Tuple[str, ...] = (
        "analysis/behavior/behavior_qc.json",
        "analysis/qc/behavior_qc.json",
        "behavior_qc.json",
    )

    # Optional registry file if present.
    registry_candidates: Tuple[str, ...] = (
        "VIP_SD_summary.xlsx",
        "vip_sd_summary.xlsx",
    )


def safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def first_existing(base: Path, candidates: Sequence[str]) -> Optional[Path]:
    for rel in candidates:
        p = base / rel
        if p.exists():
            return p
    return None


def discover_session_dirs(dataset_root: Path) -> List[Path]:
    session_dirs: List[Path] = []
    for p in dataset_root.rglob("*"):
        if not p.is_dir():
            continue
        if SESSION_RE.fullmatch(p.name):
            session_dirs.append(p)
    return sorted(session_dirs)


def infer_session_identity(session_dir: Path) -> Dict[str, Any]:
    m = SESSION_RE.fullmatch(session_dir.name)
    if not m:
        return {
            "mouse_id": None,
            "session_id": session_dir.name,
            "session_date": None,
            "session_time": None,
        }
    return {
        "mouse_id": m.group("mouse"),
        "session_id": session_dir.name,
        "session_date": m.group("date"),
        "session_time": m.group("time"),
    }


def maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        val = float(x)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def maybe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def coalesce(*values: Any) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        return v
    return None


def summarize_glutamate_qc(qc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "has_glutamate": bool(qc),
        "glut_qc_path_found": bool(qc),
        "glut_dmd_count": coalesce(
            maybe_int(qc.get("n_dmds")),
            maybe_int(qc.get("dmd_count")),
        ),
        "glut_n_rois": coalesce(
            maybe_int(qc.get("n_rois")),
            maybe_int(qc.get("n_synapses")),
            maybe_int(get_nested(qc, "summary", "n_rois")),
        ),
        "glut_n_good_rois": coalesce(
            maybe_int(qc.get("n_good_rois")),
            maybe_int(qc.get("n_passing_rois")),
            maybe_int(qc.get("n_good_synapses")),
        ),
        "glut_pass_fraction": coalesce(
            maybe_float(qc.get("pass_fraction")),
            maybe_float(qc.get("fraction_good")),
            maybe_float(get_nested(qc, "summary", "pass_fraction")),
        ),
        "glut_median_snr": coalesce(
            maybe_float(qc.get("median_snr")),
            maybe_float(qc.get("median_residual_snr")),
            maybe_float(get_nested(qc, "summary", "median_snr")),
        ),
        "glut_notes": coalesce(
            qc.get("notes"),
            qc.get("summary_note"),
        ),
    }


def summarize_calcium_qc(qc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "has_calcium": bool(qc),
        "calcium_qc_path_found": bool(qc),
        "ca_n_rois": coalesce(
            maybe_int(qc.get("n_rois")),
            maybe_int(qc.get("n_cells")),
            maybe_int(get_nested(qc, "summary", "n_rois")),
        ),
        "ca_n_good_rois": coalesce(
            maybe_int(qc.get("n_good_rois")),
            maybe_int(qc.get("n_passing_rois")),
            maybe_int(qc.get("n_good_cells")),
        ),
        "ca_pass_fraction": coalesce(
            maybe_float(qc.get("pass_fraction")),
            maybe_float(qc.get("fraction_good")),
            maybe_float(get_nested(qc, "summary", "pass_fraction")),
        ),
        "ca_median_snr": coalesce(
            maybe_float(qc.get("median_snr")),
            maybe_float(qc.get("median_residual_snr")),
            maybe_float(get_nested(qc, "summary", "median_snr")),
        ),
        "ca_notes": coalesce(
            qc.get("notes"),
            qc.get("summary_note"),
        ),
    }


def summarize_behavior_qc(qc: Dict[str, Any]) -> Dict[str, Any]:
    ready = coalesce(
        qc.get("ready_for_physiology_extraction"),
        qc.get("alignment_success"),
        qc.get("success"),
    )
    return {
        "behavior_qc_path_found": bool(qc),
        "alignment_ready": bool(ready) if ready is not None else None,
        "alignment_rmse_ms": coalesce(
            maybe_float(qc.get("alignment_rmse_ms")),
            maybe_float(get_nested(qc, "fit", "rmse_ms")),
        ),
        "event_coverage": coalesce(
            maybe_float(qc.get("event_coverage")),
            maybe_float(get_nested(qc, "summary", "event_coverage")),
        ),
        "behavior_notes": coalesce(
            qc.get("notes"),
            qc.get("summary_note"),
        ),
    }


def compute_overall_quality(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], str]:
    """
    Returns:
        score in [0, 1] if computable,
        tier in {A, B, C, FAIL} if computable,
        semicolon-delimited flags
    """
    components: List[float] = []
    weights: List[float] = []
    flags: List[str] = []

    glut_pf = maybe_float(row.get("glut_pass_fraction"))
    ca_pf = maybe_float(row.get("ca_pass_fraction"))
    align_ready = row.get("alignment_ready")
    event_cov = maybe_float(row.get("event_coverage"))

    if glut_pf is not None:
        components.append(np.clip(glut_pf, 0.0, 1.0))
        weights.append(0.45)
        if glut_pf < 0.40:
            flags.append("low_glut_pass_fraction")

    if ca_pf is not None:
        components.append(np.clip(ca_pf, 0.0, 1.0))
        weights.append(0.20)
        if ca_pf < 0.40:
            flags.append("low_calcium_pass_fraction")

    if align_ready is not None:
        components.append(1.0 if bool(align_ready) else 0.0)
        weights.append(0.20)
        if not bool(align_ready):
            flags.append("alignment_failed")

    if event_cov is not None:
        components.append(np.clip(event_cov, 0.0, 1.0))
        weights.append(0.15)
        if event_cov < 0.85:
            flags.append("low_event_coverage")

    if not components or sum(weights) == 0:
        return None, None, "; ".join(flags)

    score = float(np.average(components, weights=weights))

    if score >= 0.85:
        tier = "A"
    elif score >= 0.70:
        tier = "B"
    elif score >= 0.50:
        tier = "C"
    else:
        tier = "FAIL"

    return score, tier, "; ".join(flags)


def build_session_row(session_dir: Path, cfg: ManifestConfig) -> Dict[str, Any]:
    ident = infer_session_identity(session_dir)

    glut_path = first_existing(session_dir, cfg.glutamate_qc_candidates)
    ca_path = first_existing(session_dir, cfg.calcium_qc_candidates)
    beh_path = first_existing(session_dir, cfg.behavior_qc_candidates)

    glut_qc = safe_read_json(glut_path) if glut_path else {}
    ca_qc = safe_read_json(ca_path) if ca_path else {}
    beh_qc = safe_read_json(beh_path) if beh_path else {}

    row: Dict[str, Any] = {
        "row_type": "session",
        "mouse_id": ident["mouse_id"],
        "session_id": ident["session_id"],
        "session_date": ident["session_date"],
        "session_time": ident["session_time"],
        "session_dir": str(session_dir),
        "relative_session_dir": str(session_dir.relative_to(cfg.dataset_root)),
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "glut_qc_path": str(glut_path) if glut_path else None,
        "calcium_qc_path": str(ca_path) if ca_path else None,
        "behavior_qc_path": str(beh_path) if beh_path else None,
    }

    row.update(summarize_glutamate_qc(glut_qc))
    row.update(summarize_calcium_qc(ca_qc))
    row.update(summarize_behavior_qc(beh_qc))

    score, tier, flags = compute_overall_quality(row)
    row["overall_quality_score"] = score
    row["quality_tier"] = tier
    row["flags"] = flags

    row["needs_review"] = bool(
        (tier in {"C", "FAIL"}) if tier is not None else True
    )

    return row


def build_mouse_rows(session_df: pd.DataFrame) -> pd.DataFrame:
    if session_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for mouse_id, g in session_df.groupby("mouse_id", dropna=False):
        usable = g["quality_tier"].isin(["A", "B"]).sum() if "quality_tier" in g else 0
        review = g["needs_review"].fillna(True).sum() if "needs_review" in g else len(g)

        row = {
            "row_type": "mouse",
            "mouse_id": mouse_id,
            "session_id": None,
            "session_date": None,
            "session_time": None,
            "session_dir": None,
            "relative_session_dir": None,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "n_sessions": int(len(g)),
            "n_usable_sessions": int(usable),
            "n_sessions_needing_review": int(review),
            "n_sessions_with_glutamate": int(g["has_glutamate"].fillna(False).sum()),
            "n_sessions_with_calcium": int(g["has_calcium"].fillna(False).sum()),
            "n_alignment_failures": int((g["alignment_ready"] == False).sum()),  # noqa: E712
            "mean_overall_quality_score": maybe_float(g["overall_quality_score"].mean()),
            "median_glut_pass_fraction": maybe_float(g["glut_pass_fraction"].median()),
            "median_ca_pass_fraction": maybe_float(g["ca_pass_fraction"].median()),
            "flags": "",
            "quality_tier": None,
            "needs_review": None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    cols_first = [
        "row_type",
        "mouse_id",
        "session_id",
        "session_date",
        "quality_tier",
        "overall_quality_score",
        "needs_review",
        "flags",
        "has_glutamate",
        "has_calcium",
        "glut_n_rois",
        "glut_n_good_rois",
        "glut_pass_fraction",
        "ca_n_rois",
        "ca_n_good_rois",
        "ca_pass_fraction",
        "alignment_ready",
        "event_coverage",
        "relative_session_dir",
    ]
    cols = [c for c in cols_first if c in df.columns] + [c for c in df.columns if c not in cols_first]
    return df.loc[:, cols]


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_No rows found._"
    clipped = df.head(max_rows).copy()
    return clipped.to_markdown(index=False)


def write_markdown_overview(
    output_path: Path,
    session_df: pd.DataFrame,
    mouse_df: pd.DataFrame,
    dataset_root: Path,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_sessions = len(session_df)
    n_mice = session_df["mouse_id"].nunique(dropna=True) if not session_df.empty else 0
    n_good = session_df["quality_tier"].isin(["A", "B"]).sum() if not session_df.empty else 0
    n_review = session_df["needs_review"].fillna(True).sum() if not session_df.empty else 0

    lines: List[str] = []
    lines.append("# VIP Synaptic Dynamics Dataset Overview")
    lines.append("")
    lines.append(f"**Dataset root:** `{dataset_root}`")
    lines.append(f"**Updated:** {now}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Mice: **{n_mice}**")
    lines.append(f"- Sessions: **{n_sessions}**")
    lines.append(f"- Usable sessions (A/B): **{n_good}**")
    lines.append(f"- Sessions needing review: **{n_review}**")
    lines.append("")

    if not mouse_df.empty:
        lines.append("## Mouse-level summary")
        lines.append("")
        mouse_show = mouse_df[
            [
                c for c in [
                    "mouse_id",
                    "n_sessions",
                    "n_usable_sessions",
                    "n_sessions_needing_review",
                    "n_sessions_with_glutamate",
                    "n_sessions_with_calcium",
                    "n_alignment_failures",
                    "mean_overall_quality_score",
                ]
                if c in mouse_df.columns
            ]
        ].sort_values("mouse_id")
        lines.append(markdown_table(mouse_show, max_rows=200))
        lines.append("")

    if not session_df.empty:
        lines.append("## Session-level summary")
        lines.append("")
        session_show = session_df[
            [
                c for c in [
                    "mouse_id",
                    "session_id",
                    "session_date",
                    "quality_tier",
                    "overall_quality_score",
                    "has_glutamate",
                    "has_calcium",
                    "glut_n_rois",
                    "glut_pass_fraction",
                    "ca_n_rois",
                    "ca_pass_fraction",
                    "alignment_ready",
                    "event_coverage",
                    "flags",
                ]
                if c in session_df.columns
            ]
        ].sort_values(["mouse_id", "session_date", "session_id"])
        lines.append(markdown_table(session_show, max_rows=500))
        lines.append("")

        review_df = session_df[session_df["needs_review"].fillna(True)].copy()
        if not review_df.empty:
            lines.append("## Sessions needing review")
            lines.append("")
            review_show = review_df[
                [
                    c for c in [
                        "mouse_id",
                        "session_id",
                        "quality_tier",
                        "overall_quality_score",
                        "flags",
                        "relative_session_dir",
                    ]
                    if c in review_df.columns
                ]
            ].sort_values(["mouse_id", "session_id"])
            lines.append(markdown_table(review_show, max_rows=500))
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset_manifest(
    dataset_root: str | Path,
    write_outputs: bool = True,
) -> pd.DataFrame:
    cfg = ManifestConfig(dataset_root=Path(dataset_root))
    session_dirs = discover_session_dirs(cfg.dataset_root)

    session_rows = [build_session_row(sd, cfg) for sd in session_dirs]
    session_df = pd.DataFrame(session_rows)
    if not session_df.empty:
        session_df = dataframe_for_display(session_df)

    mouse_df = build_mouse_rows(session_df)
    if not mouse_df.empty:
        mouse_df = dataframe_for_display(mouse_df)

    manifest_df = pd.concat([mouse_df, session_df], ignore_index=True, sort=False)

    if write_outputs:
        csv_path = cfg.dataset_root / cfg.output_csv
        json_path = cfg.dataset_root / cfg.output_json
        md_path = cfg.dataset_root / cfg.output_md

        manifest_df.to_csv(csv_path, index=False)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated": datetime.now().isoformat(timespec="seconds"),
                    "dataset_root": str(cfg.dataset_root),
                    "mouse_rows": mouse_df.replace({np.nan: None}).to_dict(orient="records"),
                    "session_rows": session_df.replace({np.nan: None}).to_dict(orient="records"),
                },
                f,
                indent=2,
            )

        write_markdown_overview(
            output_path=md_path,
            session_df=session_df,
            mouse_df=mouse_df,
            dataset_root=cfg.dataset_root,
        )

    return manifest_df