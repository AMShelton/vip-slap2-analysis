from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class MoveRecord:
    src: Path
    dst: Path
    reason: str
    category: str
    status: str = "PLANNED"


@dataclass
class SessionNames:
    mouse_id: str
    session_stamp: str          # e.g. 826033_2026-02-17_13-13-55
    slap2_stamp: str            # e.g. 2026-02-17_13-13-55 or from slap2_*
    raw_root_name: str          # e.g. 826033_2026-02-17_13-13-55
    processed_root_name: str    # e.g. 826033_2026-02-17_13-13-55_slap2_2026-02-17_13-13-55
    backup_root_name: str       # e.g. slap2_826033_2026-02-17_13-13-55_remaining_data_backup
    harp_dir_name: str = "Behavior.harp"


@dataclass
class ReorgPlan:
    target_session_dir: Path
    raw_root: Path
    processed_root: Path
    backup_root: Path
    records: List[MoveRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add(self, src: Path, dst: Path, reason: str, category: str) -> None:
        self.records.append(MoveRecord(src=src, dst=dst, reason=reason, category=category))

    def planned_sources(self) -> set[Path]:
        return {r.src for r in self.records}


# -----------------------------------------------------------------------------
# Parsing / manifest helpers
# -----------------------------------------------------------------------------

SESSION_RE = re.compile(r"(?P<mouse>\d{6})_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})")
SLAP2_DIR_RE = re.compile(r"slap2_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})", re.IGNORECASE)
DMD_FILE_RE = re.compile(r"^E\d+T\d+DMD\d+_.*", re.IGNORECASE)


def load_manifest_paths(tsv_path: Path) -> List[str]:
    """
    Accepts a simple manifest TSV. If there are multiple columns, the first column
    containing a path-like value is used row-wise.
    """
    rows: List[str] = []
    with tsv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            # Pick first non-empty field
            val = next((x.strip() for x in row if x and x.strip()), None)
            if val:
                rows.append(val)
    return rows


def infer_harp_dir_name_from_example_manifest(example_manifest_tsv: Path) -> str:
    """
    Look for a *.harp directory in the example manifest and preserve its basename.
    Falls back to 'Behavior.harp'.
    """
    try:
        paths = load_manifest_paths(example_manifest_tsv)
    except Exception:
        return "Behavior.harp"

    harp_candidates = []
    for p in paths:
        parts = Path(p).parts
        for part in parts:
            if part.lower().endswith(".harp"):
                harp_candidates.append(part)

    if not harp_candidates:
        return "Behavior.harp"

    # Prefer a directory that contains "Behavior"
    for name in harp_candidates:
        if "behavior" in name.lower():
            return name

    return harp_candidates[0]


def infer_session_names(
    target_session_dir: Path,
    example_manifest_tsv: Path,
    mouse_id: Optional[str] = None,
) -> SessionNames:
    """
    Infer raw / processed / backup directory names.

    raw_root_name:
        nnnnnn_yyyy-mm-dd_hh-mm-ss

    processed_root_name:
        nnnnnn_yyyy-mm-dd_hh-mm-ss_slap2_yyyy-mm-dd_hh-mm-ss

    backup_root_name:
        slap2_nnnnnn_yyyy-mm-dd_hh-mm-ss_remaining_data_backup
    """
    target_session_dir = target_session_dir.resolve()
    base_name = target_session_dir.name

    m = SESSION_RE.search(base_name)
    if not m:
        raise ValueError(
            f"Could not parse target session folder name '{base_name}'. "
            "Expected something like '826033_2026-02-17_13-13-55'."
        )

    parsed_mouse = m.group("mouse")
    session_date = m.group("date")
    session_time = m.group("time")

    if mouse_id is None:
        mouse_id = parsed_mouse

    session_stamp = f"{mouse_id}_{session_date}_{session_time}"

    slap2_dirs = find_slap2_dirs(target_session_dir)
    if slap2_dirs:
        slap2_dir = sorted(slap2_dirs, key=lambda p: p.stat().st_mtime if p.exists() else 0)[0]
        sm = SLAP2_DIR_RE.search(slap2_dir.name)
        if sm:
            slap2_stamp = f"{sm.group('date')}_{sm.group('time')}"
        else:
            slap2_stamp = f"{session_date}_{session_time}"
    else:
        slap2_stamp = f"{session_date}_{session_time}"

    harp_dir_name = infer_harp_dir_name_from_example_manifest(example_manifest_tsv)

    raw_root_name = session_stamp
    processed_root_name = f"{session_stamp}_slap2_{slap2_stamp}"
    backup_root_name = f"slap2_{session_stamp}_remaining_data_backup"

    return SessionNames(
        mouse_id=mouse_id,
        session_stamp=session_stamp,
        slap2_stamp=slap2_stamp,
        raw_root_name=raw_root_name,
        processed_root_name=processed_root_name,
        backup_root_name=backup_root_name,
        harp_dir_name=harp_dir_name,
    )


# -----------------------------------------------------------------------------
# Filesystem discovery helpers
# -----------------------------------------------------------------------------

def iter_immediate_children(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return list(path.iterdir())


def safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except Exception:
        return str(path)


def find_slap2_dirs(target_session_dir: Path) -> List[Path]:
    matches = []
    imaging_root = target_session_dir / "imaging_data" / "SLAP2_data"
    if imaging_root.exists():
        for child in imaging_root.iterdir():
            if child.is_dir() and child.name.lower().startswith("slap2_"):
                matches.append(child)
    else:
        # broader fallback
        for p in target_session_dir.rglob("*"):
            if p.is_dir() and p.name.lower().startswith("slap2_"):
                matches.append(p)
    return matches


def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# -----------------------------------------------------------------------------
# Routing logic
# -----------------------------------------------------------------------------

def ensure_dir(path: Path, execute: bool) -> None:
    if execute:
        path.mkdir(parents=True, exist_ok=True)


def classify_top_level_metadata(path: Path) -> Optional[Tuple[str, str]]:
    """
    Returns (category, reason) or None.
    """
    name = path.name.lower()
    if name in {"instrument.json", "acquisition.json", "subject.json", "session.json", "data_description.json", "procedures.json"}:
        return ("raw", "top-level metadata json")
    if name in {"notes.txt", "readme.txt", "readme.md"}:
        return ("backup", "unclassified top-level note/readme")
    return None


def route_behavior_tree(path: Path, names: SessionNames, raw_root: Path) -> Path:
    """
    Move behavior-related content into raw_root / behavior or behavior-videos.
    """
    lower = path.name.lower()

    if path.is_dir() and lower.endswith(".harp"):
        return raw_root / "behavior" / names.harp_dir_name

    if path.is_dir() and ("video" in lower or "camera" in lower):
        return raw_root / "behavior-videos" / path.name

    if lower.endswith(".csv") and "bonsai" in lower:
        return raw_root / "behavior" / path.name

    if lower.endswith(".json") and "stim" in lower:
        return raw_root / "behavior" / path.name

    if lower.endswith(".mp4") or lower.endswith(".avi"):
        return raw_root / "behavior-videos" / path.name

    return raw_root / "behavior" / path.name


def route_slap2_content(src: Path, slap2_root: Path, processed_root: Path, raw_root: Path) -> Tuple[Path, str, str]:
    """
    Returns (dst, reason, category)
    """
    rel = src.relative_to(slap2_root)
    parts_lower = [p.lower() for p in rel.parts]
    name = src.name
    name_lower = name.lower()

    # Raw-ish acquisition/reference content
    if "reference" in parts_lower or name_lower.startswith("reference"):
        return raw_root / "imaging_data" / "SLAP2_data" / rel, "reference stack / reference content", "raw"

    if name_lower.endswith((".meta", ".cfg", ".ini", ".log")):
        return raw_root / "imaging_data" / "SLAP2_data" / rel, "acquisition sidecar/log/config", "raw"

    if name_lower.endswith((".bin", ".dat", ".raw", ".tif", ".tiff")) and "motion_correction" not in parts_lower:
        return raw_root / "imaging_data" / "SLAP2_data" / rel, "raw binary/image acquisition content", "raw"

    # Explicit processed outputs
    if "motion_correction" in parts_lower:
        idx = parts_lower.index("motion_correction")
        tail = rel.parts[idx + 1:]
        return processed_root / "motion_correction" / Path(*tail), "motion correction output", "processed"

    if "source_extraction" in parts_lower:
        idx = parts_lower.index("source_extraction")
        tail = rel.parts[idx + 1:]
        return processed_root / "source_extraction" / Path(*tail), "source extraction output", "processed"

    if "experimentsummary" in parts_lower:
        idx = parts_lower.index("experimentsummary")
        tail = rel.parts[idx + 1:]
        return processed_root / "source_extraction" / "ExperimentSummary" / Path(*tail), "ExperimentSummary output", "processed"

    if DMD_FILE_RE.match(name):
        return processed_root / "motion_correction" / name, "DMD-aligned processed file", "processed"

    if name == "ANNOTATIONS.mat":
        return processed_root / "source_extraction" / name, "annotation file", "processed"

    if name_lower.startswith("trialtable."):
        return processed_root / name, "trial table", "processed"

    # MATLAB summaries / output mats
    if name_lower.endswith(".mat") and (
        "summary" in name_lower
        or "roi" in name_lower
        or "extract" in name_lower
        or "trace" in name_lower
        or "annotation" in name_lower
    ):
        return processed_root / "source_extraction" / name, "processed matlab output", "processed"

    # Fallback backup
    return processed_root / "unclassified_from_slap2" / rel, "unclassified slap2 content", "processed"


def build_reorganization_plan(
    target_session_dir: Path,
    example_manifest_tsv: Path,
    target_manifest_tsv: Optional[Path] = None,
    mouse_id: Optional[str] = None,
) -> ReorgPlan:
    names = infer_session_names(target_session_dir, example_manifest_tsv, mouse_id=mouse_id)

    # New roots live under the parent directory of the current raw session folder
    session_parent = target_session_dir.parent
    raw_root = session_parent / names.raw_root_name
    processed_root = session_parent / names.processed_root_name
    backup_root = session_parent / names.backup_root_name

    plan = ReorgPlan(
        target_session_dir=target_session_dir,
        raw_root=raw_root,
        processed_root=processed_root,
        backup_root=backup_root,
    )

    if raw_root.exists() and raw_root.resolve() != target_session_dir.resolve():
        plan.warnings.append(f"Raw root already exists: {raw_root}")
    if processed_root.exists():
        plan.warnings.append(f"Processed root already exists: {processed_root}")
    if backup_root.exists():
        plan.warnings.append(f"Backup root already exists: {backup_root}")

    # Top-level organization from current target session dir
    for child in iter_immediate_children(target_session_dir):
        if child.name in {processed_root.name, backup_root.name}:
            continue

        # Keep current target session dir as the eventual raw root name;
        # if it's already the correct name, content remains under it or moves within it.
        meta = classify_top_level_metadata(child)
        if meta is not None:
            category, reason = meta
            if category == "raw":
                dst = raw_root / child.name
            else:
                dst = backup_root / child.name
            if child.resolve() != dst.resolve():
                plan.add(child, dst, reason, category)
            continue

        child_lower = child.name.lower()

        # Behavior content
        if (
            "behavior" in child_lower
            or child_lower.endswith(".harp")
            or "video" in child_lower
            or "camera" in child_lower
        ):
            dst = route_behavior_tree(child, names, raw_root)
            if child.resolve() != dst.resolve():
                plan.add(child, dst, "behavior-related content", "raw")
            continue

        # Imaging content
        if child.name == "imaging_data":
            imaging_root = child
            slap2_root = find_first_existing(
                [
                    imaging_root / "SLAP2_data",
                    imaging_root / "slap2_data",
                ]
            )

            if slap2_root is None:
                # Keep the whole imaging tree in raw if structure is unusual
                dst = raw_root / child.name
                if child.resolve() != dst.resolve():
                    plan.add(child, dst, "unparsed imaging_data tree kept as raw", "raw")
                continue

            # Move everything under imaging_data except processed SLAP2 products
            # We recurse only one level down from SLAP2_data and then finer below.
            for p in slap2_root.rglob("*"):
                if p.is_dir():
                    continue
                # Determine route based on its nearest slap2_* ancestor if present
                slap2_ancestor = None
                for anc in [p.parent, *p.parents]:
                    if anc == slap2_root.parent.parent:
                        break
                    if anc.name.lower().startswith("slap2_"):
                        slap2_ancestor = anc
                        break

                if slap2_ancestor is not None:
                    dst, reason, category = route_slap2_content(
                        src=p,
                        slap2_root=slap2_root,
                        processed_root=processed_root,
                        raw_root=raw_root,
                    )
                else:
                    # non-slap2 files under imaging_data/SLAP2_data go to raw
                    rel = p.relative_to(target_session_dir)
                    dst = raw_root / rel
                    reason = "non-slap2 imaging content"
                    category = "raw"

                if p.resolve() != dst.resolve():
                    plan.add(p, dst, reason, category)

            # Also handle any non-SLAP2_data content within imaging_data by keeping it raw
            for p in imaging_root.iterdir():
                if p.name.lower() in {"slap2_data", "slap2_data"}:
                    continue
                dst = raw_root / "imaging_data" / p.name
                if p.resolve() != dst.resolve():
                    plan.add(p, dst, "non-SLAP2 imaging tree", "raw")
            continue

        # Known loose processed files at top level
        if DMD_FILE_RE.match(child.name):
            dst = processed_root / "motion_correction" / child.name
            plan.add(child, dst, "top-level DMD processed file", "processed")
            continue

        if child.name == "ANNOTATIONS.mat":
            dst = processed_root / "source_extraction" / child.name
            plan.add(child, dst, "top-level annotation file", "processed")
            continue

        if child.name.lower().startswith("trialtable."):
            dst = processed_root / child.name
            plan.add(child, dst, "top-level trial table", "processed")
            continue

        # Anything else gets backed up
        dst = backup_root / child.name
        if child.resolve() != dst.resolve():
            plan.add(child, dst, "unmatched top-level content", "backup")

    # Optional: compare against target manifest only for warning/reporting
    if target_manifest_tsv and target_manifest_tsv.exists():
        try:
            target_manifest_rows = load_manifest_paths(target_manifest_tsv)
            if not target_manifest_rows:
                plan.warnings.append("Target manifest TSV was provided but appears empty.")
        except Exception as e:
            plan.warnings.append(f"Could not parse target manifest TSV: {e}")

    return plan


# -----------------------------------------------------------------------------
# Validation / execution
# -----------------------------------------------------------------------------

def validate_plan(plan: ReorgPlan) -> List[str]:
    errors: List[str] = []

    # No duplicate sources
    srcs = [r.src for r in plan.records]
    if len(srcs) != len(set(srcs)):
        errors.append("Duplicate source paths found in plan.")

    # No duplicate destinations
    dsts = [r.dst for r in plan.records]
    if len(dsts) != len(set(dsts)):
        dupes = find_duplicates(dsts)
        errors.append(f"Duplicate destination paths found: {dupes[:10]}")

    # Destinations should not nest inside their own sources
    for rec in plan.records:
        try:
            rec.dst.relative_to(rec.src)
            errors.append(f"Destination is inside source for move: {rec.src} -> {rec.dst}")
        except Exception:
            pass

    return errors


def find_duplicates(paths: Sequence[Path]) -> List[str]:
    seen: Dict[Path, int] = {}
    dupes: List[str] = []
    for p in paths:
        seen[p] = seen.get(p, 0) + 1
    for p, n in seen.items():
        if n > 1:
            dupes.append(str(p))
    return dupes


def create_destination_roots(plan: ReorgPlan, execute: bool) -> None:
    roots = [
        plan.raw_root,
        plan.raw_root / "behavior",
        plan.raw_root / "behavior-videos",
        plan.raw_root / "imaging_data" / "SLAP2_data",
        plan.processed_root,
        plan.processed_root / "motion_correction",
        plan.processed_root / "source_extraction",
        plan.processed_root / "source_extraction" / "ExperimentSummary",
        plan.backup_root,
    ]
    for root in roots:
        ensure_dir(root, execute=execute)


def move_one(src: Path, dst: Path, execute: bool) -> str:
    if not src.exists():
        return "MISSING_SOURCE"

    if dst.exists():
        # Avoid silent overwrite
        return "DEST_EXISTS"

    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return "MOVED"
    return "DRY_RUN"


def execute_plan(plan: ReorgPlan, execute: bool) -> None:
    create_destination_roots(plan, execute=execute)

    # Sort deep files before parents to reduce folder-move conflicts
    def sort_key(rec: MoveRecord) -> Tuple[int, int, str]:
        depth = len(rec.src.parts)
        is_dir = 0 if rec.src.is_file() else 1
        return (-depth, is_dir, str(rec.src))

    for rec in sorted(plan.records, key=sort_key):
        rec.status = move_one(rec.src, rec.dst, execute=execute)


def cleanup_empty_dirs(root: Path, stop_at: Path, execute: bool) -> None:
    """
    Remove empty directories under root up to but not including stop_at.
    """
    if not root.exists():
        return
    all_dirs = sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in all_dirs:
        if d == stop_at:
            continue
        try:
            next(d.iterdir())
        except StopIteration:
            if execute:
                d.rmdir()


def write_report(plan: ReorgPlan, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["src", "dst", "category", "reason", "status"])
        for rec in plan.records:
            writer.writerow([str(rec.src), str(rec.dst), rec.category, rec.reason, rec.status])


def summarize_plan(plan: ReorgPlan) -> str:
    n_raw = sum(r.category == "raw" for r in plan.records)
    n_processed = sum(r.category == "processed" for r in plan.records)
    n_backup = sum(r.category == "backup" for r in plan.records)
    total = len(plan.records)
    lines = [
        f"Target session: {plan.target_session_dir}",
        f"Raw root:       {plan.raw_root}",
        f"Processed root: {plan.processed_root}",
        f"Backup root:    {plan.backup_root}",
        f"Planned moves:  {total}",
        f"  raw:          {n_raw}",
        f"  processed:    {n_processed}",
        f"  backup:       {n_backup}",
    ]
    if plan.warnings:
        lines.append("Warnings:")
        lines.extend([f"  - {w}" for w in plan.warnings])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reorganize a SLAP2 session directory into raw / processed / remaining-data "
            "roots based on a target session path and an example manifest."
        )
    )
    p.add_argument(
        "target_session_dir",
        type=Path,
        help="Path to the current target raw session directory, e.g. ...\\826033_2026-02-17_13-13-55",
    )
    p.add_argument(
        "example_manifest_tsv",
        type=Path,
        help="TSV manifest from an example session structure.",
    )
    p.add_argument(
        "--target-manifest-tsv",
        type=Path,
        default=None,
        help="Optional TSV manifest of the current target tree for reporting / sanity checks.",
    )
    p.add_argument(
        "--mouse-id",
        type=str,
        default=None,
        help="Optional explicit mouse ID. Otherwise parsed from target folder name.",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform moves. Default is dry-run.",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional output TSV report path. Default writes next to target session parent.",
    )
    p.add_argument(
        "--cleanup-empty-dirs",
        action="store_true",
        help="After execution, remove empty directories left under the old target session tree.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    target_session_dir: Path = args.target_session_dir
    example_manifest_tsv: Path = args.example_manifest_tsv
    target_manifest_tsv: Optional[Path] = args.target_manifest_tsv
    mouse_id: Optional[str] = args.mouse_id
    execute: bool = args.execute

    if not target_session_dir.exists():
        raise FileNotFoundError(f"Target session directory does not exist: {target_session_dir}")
    if not example_manifest_tsv.exists():
        raise FileNotFoundError(f"Example manifest TSV does not exist: {example_manifest_tsv}")
    if target_manifest_tsv is not None and not target_manifest_tsv.exists():
        raise FileNotFoundError(f"Target manifest TSV does not exist: {target_manifest_tsv}")

    plan = build_reorganization_plan(
        target_session_dir=target_session_dir,
        example_manifest_tsv=example_manifest_tsv,
        target_manifest_tsv=target_manifest_tsv,
        mouse_id=mouse_id,
    )

    print(summarize_plan(plan))

    validation_errors = validate_plan(plan)
    if validation_errors:
        print("\nValidation errors:")
        for err in validation_errors:
            print(f"  - {err}")
        raise RuntimeError("Plan validation failed. Refusing to continue.")

    execute_plan(plan, execute=execute)

    if execute and args.cleanup_empty_dirs:
        cleanup_empty_dirs(target_session_dir, stop_at=target_session_dir.parent, execute=True)

    report_path = args.report_path
    if report_path is None:
        mode = "executed" if execute else "dry_run"
        report_path = target_session_dir.parent / f"{target_session_dir.name}_reorganization_{mode}_report.tsv"

    write_report(plan, report_path=report_path)
    print(f"\nReport written to: {report_path}")

    status_counts: Dict[str, int] = {}
    for rec in plan.records:
        status_counts[rec.status] = status_counts.get(rec.status, 0) + 1

    print("\nMove status summary:")
    for k in sorted(status_counts):
        print(f"  {k}: {status_counts[k]}")


if __name__ == "__main__":
    main()