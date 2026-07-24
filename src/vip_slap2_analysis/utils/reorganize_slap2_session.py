"""Reorganize raw SLAP2 sessions into canonical raw/processed/backup folders.

The default ``nested`` layout treats the supplied session directory as a
container. For a source directory named::

    826031_2026-01-30_15-04-02

it creates these children *inside that same directory*::

    826031_2026-01-30_15-04-02/
        826031_2026-01-30_15-04-02/                  # canonical raw data
        826031_2026-01-30_15-04-02_slap2_.../        # processed data
        slap2_826031_..._remaining_data_backup/      # unmatched content
        .reorganization_reports/                     # dry-run/execution reports

Use ``--layout sibling`` to reproduce the historical behavior in which the
three canonical folders are created beside the supplied session directory.

Supported source-extraction conventions
---------------------------------------
* Voltage: ``dendriticVoltageExtraction`` containing paired
  ``dendriticVoltageSummary-*.mat`` and ``dendriticVoltageTraces-*.h5`` files.
* Glutamate: ``ExperimentSummary`` (for example ``SummaryLoCo-*.mat``).

Vascular reference images whose names contain variants of
``localvasculature`` or ``VasMap`` remain in the canonical raw session root.

The script is dry-run by default. Always inspect the TSV report before rerunning
with ``--execute``.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

SESSION_RE = re.compile(
    r"(?P<mouse>\d{6})_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})"
)
SLAP2_DIR_RE = re.compile(
    r"slap2_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})",
    re.IGNORECASE,
)
VOLTAGE_PAIR_RE = re.compile(
    r"dendriticVoltage(?:Summary|Traces)-(?P<stamp>\d{6}-\d{6})\.(?:mat|h5)$",
    re.IGNORECASE,
)

TOP_LEVEL_METADATA = {
    "instrument.json",
    "subject.json",
    "session.json",
    "acquisition.json",
    "data_description.json",
    "procedures.json",
    "project.json",
}
REPORT_DIR_NAME = ".reorganization_reports"
SOURCE_EXTRACTION_DIR_NAMES = {
    "dendriticvoltageextraction": "dendriticVoltageExtraction",
    "experimentsummary": "ExperimentSummary",
}


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
    """Canonical destination names inferred from a session folder."""

    subject_id: str
    session_stamp: str
    slap2_stamp: str
    raw_root_name: str
    processed_root_name: str
    overflow_root_name: str


@dataclass
class ReorgPlan:
    """Move plan for one SLAP2 session."""

    target_session_dir: Path
    layout: str
    output_base: Path
    raw_root: Path
    processed_root: Path
    overflow_root: Path
    records: list[MoveRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, src: Path, dst: Path, category: str, reason: str) -> None:
        """Append a planned move if source and destination differ."""
        if _same_path(src, dst):
            return
        self.records.append(
            MoveRecord(src=src, dst=dst, category=category, reason=reason)
        )


# -----------------------------------------------------------------------------
# Discovery and naming
# -----------------------------------------------------------------------------


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a.absolute() == b.absolute()


def infer_session_names(target_session_dir: Path) -> SessionNames:
    """Infer canonical raw/processed/overflow names from a session folder."""
    target_session_dir = target_session_dir.expanduser().resolve()
    match = SESSION_RE.fullmatch(target_session_dir.name)
    if match is None:
        raise ValueError(
            "Could not parse the session folder name. Expected exactly "
            "<six-digit-subject>_YYYY-MM-DD_HH-MM-SS, for example "
            "826031_2026-01-30_15-04-02."
        )

    subject_id = match.group("mouse")
    session_stamp = "%s_%s_%s" % (
        subject_id,
        match.group("date"),
        match.group("time"),
    )
    slap2_stamp = find_slap2_stamp(target_session_dir)
    if slap2_stamp is None:
        # This matches the canonical historical naming convention: the suffix
        # records the session timestamp, not the later acquisition file time.
        slap2_stamp = "%s_%s" % (match.group("date"), match.group("time"))

    return SessionNames(
        subject_id=subject_id,
        session_stamp=session_stamp,
        slap2_stamp=slap2_stamp,
        raw_root_name=session_stamp,
        processed_root_name="%s_slap2_%s" % (session_stamp, slap2_stamp),
        overflow_root_name="slap2_%s_remaining_data_backup" % session_stamp,
    )


def find_slap2_stamp(target_session_dir: Path) -> Optional[str]:
    """Return a timestamp encoded in a legacy inner ``slap2_*`` directory."""
    candidates = [
        p
        for p in target_session_dir.rglob("*")
        if p.is_dir() and p.name.lower().startswith("slap2_")
    ]
    for path in sorted(candidates, key=lambda p: len(p.parts)):
        match = SLAP2_DIR_RE.search(path.name)
        if match:
            return "%s_%s" % (match.group("date"), match.group("time"))
    return None


def iter_children(path: Path) -> Iterable[Path]:
    """Yield immediate children if ``path`` exists."""
    if path.exists():
        yield from path.iterdir()


def has_core_acquisition_content(path: Path) -> bool:
    """Return whether a directory resembles an unorganized acquisition root."""
    return any(
        (path / name).exists()
        for name in ("slap2", "behavior", "behavior-videos", "imaging_data")
    )


def appears_already_nested(target_session_dir: Path, names: SessionNames) -> bool:
    """Detect a completed or partially completed nested layout."""
    raw_child = target_session_dir / names.raw_root_name
    processed_child = target_session_dir / names.processed_root_name
    overflow_child = target_session_dir / names.overflow_root_name
    return raw_child.exists() or processed_child.exists() or overflow_child.exists()


def find_session_dirs(
    root: Path,
    subject_ids: Optional[Set[str]] = None,
) -> list[Path]:
    """Find unorganized SLAP2 session folders below ``root``."""
    root = root.expanduser().resolve()
    sessions: list[Path] = []
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]

    for path in candidates:
        match = SESSION_RE.fullmatch(path.name)
        if match is None:
            continue
        if subject_ids and match.group("mouse") not in subject_ids:
            continue
        if not has_core_acquisition_content(path):
            continue

        # Do not treat the canonical raw child of an already nested container as
        # a fresh source during recursive root searches.
        if path.parent.name == path.name:
            continue
        sessions.append(path)

    return sorted(set(sessions))


# -----------------------------------------------------------------------------
# Routing helpers
# -----------------------------------------------------------------------------


def normalized_name(path: Path) -> str:
    """Return a lowercase alphanumeric representation of a file/folder name."""
    return re.sub(r"[^a-z0-9]+", "", path.stem.lower())


def is_vascular_reference(path: Path) -> bool:
    """Recognize local-vasculature and whole-craniotomy VasMap variants."""
    norm = normalized_name(path)
    return (
        "localvasculature" in norm
        or norm.startswith("vasmap")
        or "vasculaturemap" in norm
        or "vascularmap" in norm
    )


def canonical_source_extraction_dir(path: Path) -> Optional[str]:
    """Return canonical source-extraction directory name when recognized."""
    return SOURCE_EXTRACTION_DIR_NAMES.get(path.name.lower())


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


def is_trial_table(path: Path) -> bool:
    return path.name.lower().startswith("trialtable.")


def copy_tail_after_named_part(rel: Path, part_name: str) -> Path:
    """Return the relative path following a case-insensitive named component."""
    lower_parts = [part.lower() for part in rel.parts]
    idx = lower_parts.index(part_name.lower())
    if len(rel.parts) == idx + 1:
        return Path()
    return Path(*rel.parts[idx + 1 :])


# -----------------------------------------------------------------------------
# Plan construction and routing
# -----------------------------------------------------------------------------


def build_reorganization_plan(
    target_session_dir: Path,
    layout: str = "nested",
) -> ReorgPlan:
    """Build a conservative reorganization plan for one SLAP2 session."""
    target_session_dir = target_session_dir.expanduser().resolve()
    if not target_session_dir.is_dir():
        raise NotADirectoryError("Session directory does not exist: %s" % target_session_dir)
    if layout not in {"nested", "sibling"}:
        raise ValueError("layout must be 'nested' or 'sibling'")

    names = infer_session_names(target_session_dir)
    if layout == "nested" and appears_already_nested(target_session_dir, names):
        raise ValueError(
            "This directory already contains one or more canonical destination "
            "folders and appears to be organized or partially organized: %s" % target_session_dir
        )

    output_base = target_session_dir if layout == "nested" else target_session_dir.parent
    plan = ReorgPlan(
        target_session_dir=target_session_dir,
        layout=layout,
        output_base=output_base,
        raw_root=output_base / names.raw_root_name,
        processed_root=output_base / names.processed_root_name,
        overflow_root=output_base / names.overflow_root_name,
    )

    for root in (plan.raw_root, plan.processed_root, plan.overflow_root):
        if root.exists() and not _same_path(root, target_session_dir):
            plan.warnings.append("Destination already exists: %s" % root)

    destination_names = {
        plan.raw_root.name,
        plan.processed_root.name,
        plan.overflow_root.name,
        REPORT_DIR_NAME,
    }
    for child in iter_children(target_session_dir):
        if child.name in destination_names:
            continue
        route_top_level_child(child, plan)

    validate_voltage_pairs(plan)
    return plan


def route_top_level_child(child: Path, plan: ReorgPlan) -> None:
    """Route one top-level child into raw, processed, or overflow."""
    lower = child.name.lower()

    if lower in {"behavior", "behavior-videos"}:
        plan.add(child, plan.raw_root / child.name, "raw", "canonical behavior folder")
        return

    if lower.endswith(".harp") or lower == "softwareevents":
        plan.add(
            child,
            plan.raw_root / "behavior" / child.name,
            "raw",
            "behavior acquisition content",
        )
        return

    if "camera" in lower or lower.endswith((".avi", ".mp4")):
        plan.add(
            child,
            plan.raw_root / "behavior-videos" / child.name,
            "raw",
            "behavior video content",
        )
        return

    if is_vascular_reference(child):
        plan.add(
            child,
            plan.raw_root / child.name,
            "raw",
            "local vasculature/VasMap reference retained with raw session",
        )
        return

    if lower == "slap2":
        route_slap2_tree(child, child, plan)
        return

    if lower == "imaging_data":
        route_imaging_data_tree(child, plan)
        return

    if lower in TOP_LEVEL_METADATA:
        plan.add(child, plan.raw_root / child.name, "raw", "top-level acquisition metadata")
        return

    canonical_dir = canonical_source_extraction_dir(child) if child.is_dir() else None
    if canonical_dir is not None:
        plan.add(
            child,
            plan.processed_root / "source_extraction" / canonical_dir,
            "processed",
            "%s source-extraction directory" % canonical_dir,
        )
        return

    if is_voltage_processed_asset(child):
        plan.add(
            child,
            plan.processed_root
            / "source_extraction"
            / "dendriticVoltageExtraction"
            / child.name,
            "processed",
            "loose voltage source-extraction asset",
        )
        return

    if is_trial_table(child):
        plan.add(child, plan.processed_root / child.name, "processed", "trial table")
        return

    if child.name.lower() == "annotations.mat":
        plan.add(
            child,
            plan.processed_root / "source_extraction" / child.name,
            "processed",
            "source-extraction annotations",
        )
        return

    if is_motion_or_registered_output(child):
        plan.add(
            child,
            plan.processed_root / "motion_correction" / child.name,
            "processed",
            "loose motion-correction/registered output",
        )
        return

    plan.add(
        child,
        plan.overflow_root / "remaining_root_items" / child.name,
        "overflow",
        "unmatched top-level content",
    )


def route_imaging_data_tree(imaging_data: Path, plan: ReorgPlan) -> None:
    """Route contents from legacy ``imaging_data`` trees."""
    slap2_data = imaging_data / "SLAP2_data"
    if not slap2_data.exists():
        plan.add(
            imaging_data,
            plan.overflow_root / "remaining_root_items" / imaging_data.name,
            "overflow",
            "unmatched imaging_data tree",
        )
        return

    for child in iter_children(slap2_data):
        if child.is_dir() and child.name.lower().startswith("slap2_"):
            route_slap2_tree(child, child, plan)
        else:
            plan.add(
                child,
                plan.overflow_root / "unmapped_slap2" / child.name,
                "overflow",
                "unmapped imaging_data/SLAP2_data content",
            )


def route_slap2_tree(path: Path, slap2_root: Path, plan: ReorgPlan) -> None:
    """Route every file under a SLAP2 tree while preserving relative paths."""
    for src in path.rglob("*"):
        if src.is_dir():
            continue
        dst, category, reason = route_slap2_file(src, slap2_root, plan)
        plan.add(src, dst, category, reason)


def route_slap2_file(
    src: Path,
    slap2_root: Path,
    plan: ReorgPlan,
) -> tuple[Path, str, str]:
    """Return destination, category, and reason for one file under SLAP2."""
    rel = src.relative_to(slap2_root)
    lower_name = src.name.lower()
    lower_parts = [part.lower() for part in rel.parts]

    if "dendriticvoltageextraction" in lower_parts:
        tail = copy_tail_after_named_part(rel, "dendriticVoltageExtraction")
        return (
            plan.processed_root
            / "source_extraction"
            / "dendriticVoltageExtraction"
            / tail,
            "processed",
            "dendriticVoltageExtraction processed voltage asset",
        )

    if "experimentsummary" in lower_parts:
        tail = copy_tail_after_named_part(rel, "ExperimentSummary")
        return (
            plan.processed_root / "source_extraction" / "ExperimentSummary" / tail,
            "processed",
            "ExperimentSummary glutamate source-extraction asset",
        )

    if is_voltage_processed_asset(src):
        return (
            plan.processed_root
            / "source_extraction"
            / "dendriticVoltageExtraction"
            / src.name,
            "processed",
            "loose dendritic voltage processed asset",
        )

    if is_trial_table(src):
        return plan.processed_root / src.name, "processed", "trial table"

    if lower_name == "annotations.mat":
        return (
            plan.processed_root / "source_extraction" / src.name,
            "processed",
            "source-extraction annotations",
        )

    if lower_name.endswith("_alignmentdata.mat") or is_motion_or_registered_output(src):
        return (
            plan.processed_root / "motion_correction" / src.name,
            "processed",
            "motion-correction/registered output",
        )

    if is_vascular_reference(src):
        return (
            plan.raw_root / "slap2" / rel,
            "raw",
            "vascular reference retained with raw SLAP2 content",
        )

    # Launcher logs, MATLAB diaries, and experiment notes are acquisition records
    # even when their extensions are not part of the usual image/data set.
    if any(part in {"launcher_metadata", "notes"} for part in lower_parts):
        return (
            plan.raw_root / "slap2" / rel,
            "raw",
            "SLAP2 launcher metadata or acquisition notes",
        )

    raw_suffixes = {
        ".dat",
        ".meta",
        ".json",
        ".ini",
        ".xml",
        ".tif",
        ".tiff",
        ".yml",
        ".yaml",
        ".csv",
        ".log",
        ".txt",
    }
    if src.suffix.lower() in raw_suffixes or lower_name in {"desc_.mat", "desc.mat"}:
        return (
            plan.raw_root / "slap2" / rel,
            "raw",
            "raw SLAP2 acquisition/reference/metadata content",
        )

    return (
        plan.overflow_root / "unmapped_slap2" / rel,
        "overflow",
        "unmatched SLAP2 content",
    )


def validate_voltage_pairs(plan: ReorgPlan) -> None:
    """Warn when a voltage extraction appears to have an unpaired MAT/H5 file."""
    by_stamp: dict[str, set[str]] = {}
    for rec in plan.records:
        if "dendriticVoltageExtraction" not in str(rec.dst):
            continue
        if rec.src.is_dir():
            files = [p for p in rec.src.rglob("*") if p.is_file()]
        else:
            files = [rec.src]
        for src in files:
            match = VOLTAGE_PAIR_RE.search(src.name)
            if match:
                by_stamp.setdefault(match.group("stamp"), set()).add(src.suffix.lower())

    for stamp, suffixes in sorted(by_stamp.items()):
        if suffixes != {".mat", ".h5"}:
            plan.warnings.append(
                "Voltage extraction timestamp %s is missing a paired .mat/.h5: found %s"
                % (stamp, sorted(suffixes))
            )


# -----------------------------------------------------------------------------
# Validation, execution, and reporting
# -----------------------------------------------------------------------------


def validate_plan(plan: ReorgPlan) -> list[str]:
    """Validate duplicate, overlapping, and dangerous moves."""
    errors: list[str] = []
    sources = [rec.src for rec in plan.records]
    destinations = [rec.dst for rec in plan.records]

    if len(sources) != len(set(sources)):
        errors.append("Duplicate source paths found in plan.")

    duplicate_destinations = sorted(
        {str(path) for path in destinations if destinations.count(path) > 1}
    )
    if duplicate_destinations:
        errors.append(
            "Duplicate destinations found, first examples: %s"
            % duplicate_destinations[:10]
        )

    for rec in plan.records:
        try:
            rec.dst.relative_to(rec.src)
        except ValueError:
            pass
        else:
            errors.append("Destination is inside source: %s -> %s" % (rec.src, rec.dst))

    # A whole-directory move and a nested file move cannot both execute safely.
    source_set = set(sources)
    for source in sources:
        for parent in source.parents:
            if parent in source_set:
                errors.append(
                    "Overlapping source moves found: %s and nested source %s"
                    % (parent, source)
                )
                break

    return errors


def create_destination_roots(plan: ReorgPlan, execute: bool) -> None:
    """Create standard destination folders in execution mode."""
    roots = [
        plan.raw_root,
        plan.raw_root / "slap2",
        plan.processed_root,
        plan.processed_root / "motion_correction",
        plan.processed_root / "source_extraction",
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
    """Remove empty source folders under ``root`` after execution."""
    if not execute or not root.exists():
        return
    protected = {REPORT_DIR_NAME}
    directories = [p for p in root.rglob("*") if p.is_dir()]
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        if path.name in protected:
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()


def write_report(plan: ReorgPlan, report_path: Path) -> Path:
    """Write a TSV report describing the plan and operation statuses."""
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["src", "dst", "category", "reason", "status"])
        for rec in plan.records:
            writer.writerow(
                [str(rec.src), str(rec.dst), rec.category, rec.reason, rec.status]
            )
    return report_path


def summarize_plan(plan: ReorgPlan) -> str:
    """Return a compact text summary of a plan."""
    counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for rec in plan.records:
        counts[rec.category] = counts.get(rec.category, 0) + 1
        statuses[rec.status] = statuses.get(rec.status, 0) + 1

    lines = [
        "Target session: %s" % plan.target_session_dir,
        "Layout:         %s" % plan.layout,
        "Output base:    %s" % plan.output_base,
        "Raw root:       %s" % plan.raw_root,
        "Processed root: %s" % plan.processed_root,
        "Overflow root:  %s" % plan.overflow_root,
        "Planned moves:  %d" % len(plan.records),
    ]
    for key in ("raw", "processed", "overflow"):
        lines.append("  %s: %d" % (key, counts.get(key, 0)))
    if statuses:
        lines.append("Statuses:")
        for key, count in sorted(statuses.items()):
            lines.append("  %s: %d" % (key, count))
    if plan.warnings:
        lines.append("Warnings:")
        lines.extend("  - %s" % warning for warning in plan.warnings)
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize voltage or glutamate SLAP2 sessions into canonical raw, "
            "processed, and backup roots. Dry-run is the default."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-dir",
        type=Path,
        help="One raw session folder, e.g. .../826031_2026-01-30_15-04-02",
    )
    group.add_argument(
        "--root",
        type=Path,
        help="Root to search recursively for raw session folders",
    )
    parser.add_argument(
        "--subject-id",
        action="append",
        default=None,
        help=(
            "Optional six-digit subject filter for --root. Can be repeated. "
            "Without this option, all matching subjects are included."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("nested", "sibling"),
        default="nested",
        help=(
            "nested (default): create canonical folders inside the supplied "
            "session directory. sibling: reproduce the historical layout beside it."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files. Without this flag the script only writes a dry-run report.",
    )
    parser.add_argument(
        "--cleanup-empty-dirs",
        action="store_true",
        help="After execution, remove empty source folders left behind.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=(
            "Directory for TSV reports. Default: "
            "<session>/.reorganization_reports for nested layout, or the session parent for sibling layout."
        ),
    )
    return parser


def validate_subject_ids(subject_ids: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if not subject_ids:
        return None
    invalid = [value for value in subject_ids if re.fullmatch(r"\d{6}", value) is None]
    if invalid:
        raise ValueError("Subject IDs must contain exactly six digits: %s" % invalid)
    return set(subject_ids)


def run_one(
    session_dir: Path,
    execute: bool,
    cleanup: bool,
    report_dir: Optional[Path],
    layout: str,
) -> None:
    """Plan, validate, execute/dry-run, and report one session."""
    plan = build_reorganization_plan(session_dir, layout=layout)
    errors = validate_plan(plan)
    print(summarize_plan(plan))
    if errors:
        print("Validation errors:")
        for error in errors:
            print("  - %s" % error)
        raise RuntimeError("Plan validation failed; refusing to continue.")

    execute_plan(plan, execute=execute)
    if cleanup:
        cleanup_empty_dirs(session_dir, execute=execute)

    mode = "executed" if execute else "dry_run"
    if report_dir is not None:
        base_report_dir = report_dir.expanduser().resolve()
    elif layout == "nested":
        base_report_dir = session_dir.expanduser().resolve() / REPORT_DIR_NAME
    else:
        base_report_dir = session_dir.expanduser().resolve().parent

    report_path = base_report_dir / (
        "%s_slap2_reorganization_%s_report.tsv" % (session_dir.name, mode)
    )
    write_report(plan, report_path)
    print(summarize_plan(plan))
    print("Report written to: %s\n" % report_path)


def main() -> None:
    """CLI entry point."""
    args = build_argparser().parse_args()
    subject_ids = validate_subject_ids(args.subject_id)

    if args.session_dir is not None:
        sessions = [args.session_dir.expanduser().resolve()]
    else:
        sessions = find_session_dirs(args.root, subject_ids=subject_ids)

    if not sessions:
        raise FileNotFoundError("No matching raw SLAP2 session folders found.")

    for session_dir in sessions:
        run_one(
            session_dir=session_dir,
            execute=args.execute,
            cleanup=args.cleanup_empty_dirs,
            report_dir=args.report_dir,
            layout=args.layout,
        )


if __name__ == "__main__":
    main()
