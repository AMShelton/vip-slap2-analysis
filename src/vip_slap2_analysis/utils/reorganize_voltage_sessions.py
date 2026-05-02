"""Reorganize VIP SLAP2 voltage-imaging sessions into canonical folders.

This script is a voltage-specific sibling of the glutamate SLAP2 reorganization
utility. It builds a dry-run or executable move plan that separates raw files,
processed voltage outputs, and overflow/backup content for subjects 826031 and
826032 by default.

Important voltage-specific rule
-------------------------------
Any files inside a ``dendriticVoltageExtraction`` folder are treated as processed
voltage assets. The paired ``dendriticVoltageSummary-*.mat`` and
``dendriticVoltageTraces-*.h5`` files are routed together into the same processed
folder and are never split between raw and overflow destinations.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

ALLOWED_SUBJECT_IDS = {"826031", "826032"}
SESSION_RE = re.compile(
    r"(?P<mouse>\d{6})_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})"
)
SLAP2_DIR_RE = re.compile(r"slap2_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})", re.IGNORECASE)
VOLTAGE_PAIR_RE = re.compile(r"dendriticVoltage(?:Summary|Traces)-(?P<stamp>\d{6}-\d{6})\.(?:mat|h5)$", re.IGNORECASE)


@dataclass
class MoveRecord:
    """One planned or executed filesystem operation."""

    src: Path
    dst: Path
    category: str
    reason: str
    status: str = "PLANNED"


@dataclass
class SessionNames:
    """Canonical destination names inferred from a voltage session."""

    subject_id: str
    session_stamp: str
    slap2_stamp: str
    raw_root_name: str
    processed_root_name: str
    overflow_root_name: str


@dataclass
class ReorgPlan:
    """Move plan for one voltage session."""

    target_session_dir: Path
    raw_root: Path
    processed_root: Path
    overflow_root: Path
    records: list[MoveRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, src: Path, dst: Path, category: str, reason: str) -> None:
        """Append a planned move if source and destination differ."""
        try:
            if src.resolve() == dst.resolve():
                return
        except FileNotFoundError:
            pass
        self.records.append(MoveRecord(src=src, dst=dst, category=category, reason=reason))


# -----------------------------------------------------------------------------
# Discovery and naming
# -----------------------------------------------------------------------------


def infer_session_names(target_session_dir: Path) -> SessionNames:
    """Infer canonical raw/processed/overflow names from a session folder."""
    target_session_dir = target_session_dir.expanduser().resolve()
    match = SESSION_RE.search(target_session_dir.name)
    if match is None:
        raise ValueError(
            "Could not parse session folder name. Expected something like "
            "826031_2026-01-30_15-04-02."
        )

    subject_id = match.group("mouse")
    if subject_id not in ALLOWED_SUBJECT_IDS:
        raise ValueError(
            f"Refusing to plan session for subject {subject_id}. "
            f"Allowed subject IDs are {sorted(ALLOWED_SUBJECT_IDS)}."
        )

    session_stamp = f"{subject_id}_{match.group('date')}_{match.group('time')}"
    slap2_stamp = find_slap2_stamp(target_session_dir) or f"{match.group('date')}_{match.group('time')}"

    return SessionNames(
        subject_id=subject_id,
        session_stamp=session_stamp,
        slap2_stamp=slap2_stamp,
        raw_root_name=session_stamp,
        processed_root_name=f"{session_stamp}_slap2_{slap2_stamp}",
        overflow_root_name=f"slap2_{session_stamp}_remaining_data_backup",
    )


def find_slap2_stamp(target_session_dir: Path) -> Optional[str]:
    """Return the timestamp encoded in an inner ``slap2_*`` folder if present."""
    candidates = [p for p in target_session_dir.rglob("*") if p.is_dir() and p.name.lower().startswith("slap2_")]
    for path in sorted(candidates, key=lambda p: len(p.parts)):
        match = SLAP2_DIR_RE.search(path.name)
        if match:
            return f"{match.group('date')}_{match.group('time')}"
    return None


def iter_children(path: Path) -> Iterable[Path]:
    """Yield immediate children if ``path`` exists."""
    if path.exists():
        yield from path.iterdir()


def find_session_dirs(root: Path, subject_ids: set[str]) -> list[Path]:
    """Find voltage session folders below ``root`` for selected subjects."""
    root = root.expanduser().resolve()
    sessions: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        match = SESSION_RE.fullmatch(path.name)
        if match and match.group("mouse") in subject_ids:
            # Exclude already-created processed/overflow clones by requiring core acquisition folders.
            if (path / "slap2").exists() or (path / "behavior").exists() or (path / "imaging_data").exists():
                sessions.append(path)
    return sorted(set(sessions))


# -----------------------------------------------------------------------------
# Routing logic
# -----------------------------------------------------------------------------


def build_reorganization_plan(target_session_dir: Path) -> ReorgPlan:
    """Build a conservative reorganization plan for one voltage session."""
    target_session_dir = target_session_dir.expanduser().resolve()
    names = infer_session_names(target_session_dir)
    parent = target_session_dir.parent

    plan = ReorgPlan(
        target_session_dir=target_session_dir,
        raw_root=parent / names.raw_root_name,
        processed_root=parent / names.processed_root_name,
        overflow_root=parent / names.overflow_root_name,
    )

    for root in [plan.raw_root, plan.processed_root, plan.overflow_root]:
        if root.exists() and root.resolve() != target_session_dir.resolve():
            plan.warnings.append(f"Destination already exists: {root}")

    for child in iter_children(target_session_dir):
        if child.name in {plan.processed_root.name, plan.overflow_root.name}:
            continue
        route_top_level_child(child, plan)

    validate_voltage_pairs(plan)
    return plan


def route_top_level_child(child: Path, plan: ReorgPlan) -> None:
    """Route one top-level child into raw, processed, or overflow."""
    lower = child.name.lower()

    if child.name in {"behavior", "behavior-videos"}:
        plan.add(child, plan.raw_root / child.name, "raw", "canonical behavior folder")
        return

    if lower.endswith(".harp") or lower in {"softwareevents", "behavior"}:
        plan.add(child, plan.raw_root / "behavior" / child.name, "raw", "behavior acquisition content")
        return

    if "camera" in lower or lower.endswith((".avi", ".mp4")):
        plan.add(child, plan.raw_root / "behavior-videos" / child.name, "raw", "behavior video content")
        return

    if lower == "slap2":
        route_slap2_tree(child, child, plan)
        return

    if lower == "imaging_data":
        route_imaging_data_tree(child, plan)
        return

    if lower in {"instrument.json", "subject.json", "session.json", "acquisition.json", "data_description.json", "procedures.json"}:
        plan.add(child, plan.raw_root / child.name, "raw", "top-level metadata json")
        return

    if is_voltage_processed_asset(child):
        plan.add(child, plan.processed_root / "source_extraction" / child.name, "processed", "loose voltage processed asset")
        return

    if is_motion_or_registered_output(child):
        plan.add(child, plan.processed_root / "motion_correction" / child.name, "processed", "loose motion-correction/registered output")
        return

    plan.add(child, plan.overflow_root / "remaining_root_items" / child.name, "overflow", "unmatched top-level content")


def route_imaging_data_tree(imaging_data: Path, plan: ReorgPlan) -> None:
    """Route contents from legacy ``imaging_data`` trees."""
    slap2_data = imaging_data / "SLAP2_data"
    if slap2_data.exists():
        for child in iter_children(slap2_data):
            if child.is_dir() and child.name.lower().startswith("slap2_"):
                route_slap2_tree(child, child, plan)
            else:
                plan.add(child, plan.overflow_root / "unmapped_slap2" / child.name, "overflow", "unmapped imaging_data/SLAP2_data content")
    else:
        plan.add(imaging_data, plan.overflow_root / "remaining_root_items" / imaging_data.name, "overflow", "unmatched imaging_data tree")


def route_slap2_tree(path: Path, slap2_root: Path, plan: ReorgPlan) -> None:
    """Route every file under a SLAP2 tree while preserving relative paths."""
    for src in path.rglob("*"):
        if src.is_dir():
            continue
        dst, category, reason = route_slap2_file(src, slap2_root, plan)
        plan.add(src, dst, category, reason)


def route_slap2_file(src: Path, slap2_root: Path, plan: ReorgPlan) -> tuple[Path, str, str]:
    """Return destination, category, and reason for one SLAP2 file."""
    rel = src.relative_to(slap2_root)
    lower_name = src.name.lower()
    lower_parts = [part.lower() for part in rel.parts]

    if "dendriticvoltageextraction" in lower_parts:
        idx = lower_parts.index("dendriticvoltageextraction")
        tail = Path(*rel.parts[idx + 1 :]) if len(rel.parts) > idx + 1 else Path(src.name)
        return (
            plan.processed_root / "source_extraction" / "dendriticVoltageExtraction" / tail,
            "processed",
            "dendriticVoltageExtraction processed voltage asset",
        )

    if lower_name.startswith("dendriticvoltage") and lower_name.endswith((".mat", ".h5")):
        return (
            plan.processed_root / "source_extraction" / "dendriticVoltageExtraction" / src.name,
            "processed",
            "loose dendritic voltage processed asset",
        )

    if lower_name.startswith("trialtable."):
        return plan.processed_root / src.name, "processed", "trial table"

    if lower_name.endswith("_alignmentdata.mat") or is_motion_or_registered_output(src):
        return plan.processed_root / "motion_correction" / src.name, "processed", "motion-correction/registered output"

    if lower_name == "annotations.mat" or "experimentsummary" in lower_parts:
        tail = rel
        if "experimentsummary" in lower_parts:
            idx = lower_parts.index("experimentsummary")
            tail = Path("ExperimentSummary") / Path(*rel.parts[idx + 1 :])
        return plan.processed_root / "source_extraction" / tail, "processed", "source extraction metadata/output"

    if lower_name.endswith((".dat", ".meta", ".json", ".ini", ".xml", ".tif", ".tiff")):
        return plan.raw_root / "slap2" / rel, "raw", "raw SLAP2 acquisition/reference/metadata content"

    if lower_name in {"desc_.mat", "desc.mat"}:
        return plan.raw_root / "slap2" / rel, "raw", "raw SLAP2 descriptor"

    return plan.overflow_root / "unmapped_slap2" / rel, "overflow", "unmatched SLAP2 content"


def is_voltage_processed_asset(path: Path) -> bool:
    """Return true for known processed voltage summary/trace outputs."""
    name = path.name.lower()
    return name.startswith("dendriticvoltage") and name.endswith((".mat", ".h5"))


def is_motion_or_registered_output(path: Path) -> bool:
    """Return true for known registered/downsampled/alignment outputs."""
    name = path.name.lower()
    return (
        name.endswith("_alignmentdata.mat")
        or "registered" in name
        or "downsampled" in name
        or re.match(r"^e\d+t\d+dmd\d+_.*", name) is not None
    )


def validate_voltage_pairs(plan: ReorgPlan) -> None:
    """Warn when a dendritic voltage extraction appears to have an unpaired mat/h5."""
    by_stamp: dict[str, set[str]] = {}
    for rec in plan.records:
        if "dendriticVoltageExtraction" not in str(rec.dst):
            continue
        match = VOLTAGE_PAIR_RE.search(rec.src.name)
        if not match:
            continue
        by_stamp.setdefault(match.group("stamp"), set()).add(rec.src.suffix.lower())

    for stamp, suffixes in sorted(by_stamp.items()):
        if suffixes != {".mat", ".h5"}:
            plan.warnings.append(
                f"Voltage extraction timestamp {stamp} is missing a paired .mat/.h5 in planned moves: found {sorted(suffixes)}"
            )


# -----------------------------------------------------------------------------
# Validation, execution, reporting
# -----------------------------------------------------------------------------


def validate_plan(plan: ReorgPlan) -> list[str]:
    """Validate duplicate and dangerous moves before execution."""
    errors: list[str] = []
    sources = [rec.src for rec in plan.records]
    destinations = [rec.dst for rec in plan.records]

    if len(sources) != len(set(sources)):
        errors.append("Duplicate source paths found in plan.")
    duplicate_destinations = sorted({str(p) for p in destinations if destinations.count(p) > 1})
    if duplicate_destinations:
        errors.append(f"Duplicate destinations found, first examples: {duplicate_destinations[:10]}")

    for rec in plan.records:
        try:
            rec.dst.relative_to(rec.src)
            errors.append(f"Destination is inside source: {rec.src} -> {rec.dst}")
        except ValueError:
            pass
    return errors


def create_destination_roots(plan: ReorgPlan, execute: bool) -> None:
    """Create standard destination folders in execution mode."""
    roots = [
        plan.raw_root,
        plan.raw_root / "behavior",
        plan.raw_root / "behavior-videos",
        plan.raw_root / "slap2",
        plan.processed_root,
        plan.processed_root / "motion_correction",
        plan.processed_root / "source_extraction",
        plan.processed_root / "source_extraction" / "dendriticVoltageExtraction",
        plan.overflow_root,
        plan.overflow_root / "remaining_root_items",
        plan.overflow_root / "unmapped_slap2",
    ]
    if execute:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)


def execute_plan(plan: ReorgPlan, execute: bool) -> None:
    """Dry-run or execute all planned moves."""
    create_destination_roots(plan, execute=execute)
    for rec in sorted(plan.records, key=lambda r: (-len(r.src.parts), str(r.src))):
        rec.status = move_one(rec.src, rec.dst, execute=execute)


def move_one(src: Path, dst: Path, execute: bool) -> str:
    """Move one file/directory with overwrite protection."""
    if not src.exists():
        return "MISSING_SOURCE"
    if dst.exists():
        return "DEST_EXISTS"
    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return "MOVED"
    return "DRY_RUN"


def cleanup_empty_dirs(root: Path, execute: bool) -> None:
    """Remove empty folders under ``root`` after successful execution."""
    if not execute or not root.exists():
        return
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()


def write_report(plan: ReorgPlan, report_path: Path) -> Path:
    """Write a TSV report describing the plan and statuses."""
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["src", "dst", "category", "reason", "status"])
        for rec in plan.records:
            writer.writerow([str(rec.src), str(rec.dst), rec.category, rec.reason, rec.status])
    return report_path


def summarize_plan(plan: ReorgPlan) -> str:
    """Return a compact text summary of a plan."""
    counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for rec in plan.records:
        counts[rec.category] = counts.get(rec.category, 0) + 1
        statuses[rec.status] = statuses.get(rec.status, 0) + 1

    lines = [
        f"Target session: {plan.target_session_dir}",
        f"Raw root:       {plan.raw_root}",
        f"Processed root: {plan.processed_root}",
        f"Overflow root:  {plan.overflow_root}",
        f"Planned moves:  {len(plan.records)}",
    ]
    for key in ["raw", "processed", "overflow"]:
        lines.append(f"  {key}: {counts.get(key, 0)}")
    if statuses:
        lines.append("Statuses:")
        for key, count in sorted(statuses.items()):
            lines.append(f"  {key}: {count}")
    if plan.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in plan.warnings)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Reorganize voltage SLAP2 sessions for subjects 826031/826032 into raw, processed, and overflow roots."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-dir", type=Path, help="One session folder, e.g. .../826031_2026-01-30_15-04-02")
    group.add_argument("--root", type=Path, help="Root to search recursively for matching voltage session folders")
    parser.add_argument(
        "--subject-id",
        action="append",
        choices=sorted(ALLOWED_SUBJECT_IDS),
        help="Subject ID to include when using --root. Can be repeated. Defaults to both 826031 and 826032.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move files. Default is dry-run only.")
    parser.add_argument("--cleanup-empty-dirs", action="store_true", help="After execution, remove empty folders left behind.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Directory for TSV reports. Defaults to each session parent.")
    return parser


def run_one(session_dir: Path, execute: bool, cleanup: bool, report_dir: Optional[Path]) -> None:
    """Plan, validate, execute/dry-run, and report one session."""
    plan = build_reorganization_plan(session_dir)
    errors = validate_plan(plan)
    print(summarize_plan(plan))
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError("Plan validation failed; refusing to continue.")

    execute_plan(plan, execute=execute)
    if cleanup:
        cleanup_empty_dirs(session_dir, execute=execute)

    mode = "executed" if execute else "dry_run"
    base_report_dir = report_dir.expanduser().resolve() if report_dir else session_dir.parent
    report_path = base_report_dir / f"{session_dir.name}_voltage_reorganization_{mode}_report.tsv"
    write_report(plan, report_path)
    print(summarize_plan(plan))
    print(f"Report written to: {report_path}\n")


def main() -> None:
    """CLI entry point."""
    args = build_argparser().parse_args()
    subject_ids = set(args.subject_id or sorted(ALLOWED_SUBJECT_IDS))

    if args.session_dir is not None:
        sessions = [args.session_dir.expanduser().resolve()]
    else:
        sessions = find_session_dirs(args.root, subject_ids)

    if not sessions:
        raise FileNotFoundError("No matching voltage session folders found.")

    for session_dir in sessions:
        run_one(
            session_dir=session_dir,
            execute=args.execute,
            cleanup=args.cleanup_empty_dirs,
            report_dir=args.report_dir,
        )


if __name__ == "__main__":
    main()
