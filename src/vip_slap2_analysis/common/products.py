"""Where derived/QC products live, and how legacy products are set aside.

Canonical layout per session (2026-11-12 decision)::

    analysis/derived/<modality>/            current pipeline products (GIAnT / Voltex based)
    analysis/derived/<modality>/legacy_*/   products of the retired preprocessors, moved here untouched
    analysis/qc/<modality>/  (same rule)    plus analysis/qc/slap2/ (glutamate synapse QC)

``legacy_LoCo`` holds everything derived from ``SummaryLoCo*.mat`` (glutamate,
calcium); ``legacy_extractDendrites`` everything derived from the
``dendriticVoltageSummary/Traces`` pair (voltage). Nothing is deleted or renamed
inside the moved content; ``archive_legacy_outputs`` writes a CSV manifest of
every move next to the session's QC.

Transitional ``reextract_*`` sub-folders (an earlier plan wrote new products
there) are *promoted*: their contents move up to the modality root once the
legacy products are out of the way, and the empty folder is removed.
"""
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Current default voltage product (Voltex dff, trial-mode extraction).
DEFAULT_VOLTAGE_TRACE_VARIANT = "voltex_dff_trial"
# Historical voltage product name (recomputed dF/F from raw F).
LEGACY_VOLTAGE_TRACE_VARIANT = "dff_robust_f0_trial"

LEGACY_DIRNAME = {
    "glutamate": "legacy_LoCo",
    "calcium": "legacy_LoCo",
    "voltage": "legacy_extractDendrites",
    "slap2": "legacy_LoCo",           # qc/slap2 = glutamate synapse QC computed from the legacy summary
}
MODALITY_DIRS = ("glutamate", "calcium", "voltage")
QC_DIRS = ("glutamate", "calcium", "voltage", "slap2")
REEXTRACT_PREFIX = "reextract_"
LEGACY_PREFIX = "legacy_"
# Written into a modality folder once its legacy products have been set aside. From then on the folder's
# root holds current products only, and the planner leaves it alone (products cannot be told apart by name:
# glutamate_mean_dFF_ls.npz is the legacy *and* the GIAnT-based name).
ARCHIVE_MARKER = "_legacy_archived.json"


@dataclass
class Move:
    src: Path
    dst: Path
    kind: str          # "legacy" | "promote"
    family: str        # "derived" | "qc"
    modality: str


@dataclass
class ArchivePlan:
    session_id: str
    moves: List[Move] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_legacy(self) -> int:
        return sum(1 for m in self.moves if m.kind == "legacy")

    @property
    def n_promote(self) -> int:
        return sum(1 for m in self.moves if m.kind == "promote")

    def to_rows(self) -> List[Dict[str, Any]]:
        return [{"session_id": self.session_id, "kind": m.kind, "family": m.family, "modality": m.modality,
                 "src": str(m.src), "dst": str(m.dst)} for m in self.moves]


def _is_legacy_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith(LEGACY_PREFIX)


def _is_reextract_dir(p: Path) -> bool:
    return p.is_dir() and p.name.startswith(REEXTRACT_PREFIX)


def _plan_dir(plan: ArchivePlan, base: Path, *, family: str, modality: str, promote_reextract: bool, force: bool = False) -> None:
    if not base.exists() or not base.is_dir():
        return
    if (base / ARCHIVE_MARKER).exists() and not force:
        plan.notes.append(f"{base}: already archived ({ARCHIVE_MARKER} present); root holds current products, left alone")
        return
    legacy_dir = base / LEGACY_DIRNAME[modality]
    entries = sorted(p for p in base.iterdir())
    legacy_entries = [p for p in entries if not _is_legacy_dir(p) and not _is_reextract_dir(p) and p.name != ARCHIVE_MARKER]
    reextract_dirs = [p for p in entries if _is_reextract_dir(p)]
    for p in legacy_entries:
        dst = legacy_dir / p.name
        if dst.exists():
            plan.skipped.append({"session_id": plan.session_id, "kind": "legacy", "src": str(p), "dst": str(dst), "reason": "destination exists"})
            continue
        plan.moves.append(Move(p, dst, "legacy", family, modality))
    if reextract_dirs and promote_reextract:
        planned_dsts = {m.dst for m in plan.moves}
        for rd in reextract_dirs:
            for p in sorted(rd.iterdir()):
                dst = base / p.name
                # after the legacy moves the root holds only what we promote; a clash means two reextract runs
                if (dst.exists() and dst not in {m.src for m in plan.moves}) or dst in planned_dsts:
                    plan.skipped.append({"session_id": plan.session_id, "kind": "promote", "src": str(p), "dst": str(dst), "reason": "destination exists"})
                    continue
                plan.moves.append(Move(p, dst, "promote", family, modality))
                planned_dsts.add(dst)
            plan.notes.append(f"{rd}: contents promoted to {base}; folder removed if empty")
    elif reextract_dirs:
        plan.notes.append(f"{base}: {len(reextract_dirs)} reextract_* folder(s) left in place (promote_reextract=False)")


def plan_legacy_archive(asset: Any, *, promote_reextract: bool = True, force: bool = False) -> ArchivePlan:
    """Plan the moves that set a session's legacy products aside (nothing is touched).

    Folders that carry ``ARCHIVE_MARKER`` were archived before and are skipped
    unless ``force`` (which would move *current* products into ``legacy_*``).
    """
    plan = ArchivePlan(session_id=str(getattr(asset, "session_id", "")))
    derived = getattr(asset, "derived_dir", None)
    qc = getattr(asset, "qc_dir", None)
    if derived is not None:
        for m in MODALITY_DIRS:
            _plan_dir(plan, Path(derived) / m, family="derived", modality=m, promote_reextract=promote_reextract, force=force)
    if qc is not None:
        for m in QC_DIRS:
            _plan_dir(plan, Path(qc) / m, family="qc", modality=m, promote_reextract=promote_reextract, force=force)
    return plan


def apply_legacy_archive(plan: ArchivePlan, *, manifest_dir: Optional[Path] = None) -> Path:
    """Execute a plan (legacy moves first, then promotions) and write a CSV manifest; returns the manifest path."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    done: List[Dict[str, Any]] = []
    for kind in ("legacy", "promote"):
        for m in [x for x in plan.moves if x.kind == kind]:
            m.dst.parent.mkdir(parents=True, exist_ok=True)
            if m.dst.exists():
                plan.skipped.append({"session_id": plan.session_id, "kind": kind, "src": str(m.src), "dst": str(m.dst), "reason": "destination appeared"})
                continue
            shutil.move(str(m.src), str(m.dst))
            done.append({"session_id": plan.session_id, "kind": kind, "family": m.family, "modality": m.modality,
                         "src": str(m.src), "dst": str(m.dst), "time": stamp})
    # remove emptied reextract_* folders
    for m in plan.moves:
        if m.kind == "promote":
            rd = m.src.parent
            if rd.exists() and rd.is_dir() and not any(rd.iterdir()):
                rd.rmdir()
    if manifest_dir is None:
        manifest_dir = Path(plan.moves[0].dst).parents[2] if plan.moves else Path.cwd()
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"archive_legacy_outputs_{stamp}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "kind", "family", "modality", "src", "dst", "time"])
        w.writeheader()
        for row in done:
            w.writerow(row)
    # mark every modality folder this plan touched (or inspected and found already clean) as archived
    import json
    touched = {Path(m.dst).parent if m.kind == "promote" else Path(m.dst).parents[1] for m in plan.moves}
    for base in touched:
        (base / ARCHIVE_MARKER).write_text(json.dumps({"archived_at": stamp, "manifest": str(manifest),
                                                        "legacy_dir": LEGACY_DIRNAME.get(base.name, "")}, indent=1), encoding="utf-8")
    return manifest


def mark_archived_without_legacy(asset: Any) -> List[Path]:
    """Mark modality folders that hold nothing (fresh session) so later runs never archive current products."""
    import json
    out = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for family, names in (("derived", MODALITY_DIRS), ("qc", QC_DIRS)):
        root = getattr(asset, f"{family}_dir", None)
        if root is None:
            continue
        for m in names:
            base = Path(root) / m
            if base.exists() and not (base / ARCHIVE_MARKER).exists() and not any(p.name != ARCHIVE_MARKER for p in base.iterdir()):
                (base / ARCHIVE_MARKER).write_text(json.dumps({"archived_at": stamp, "manifest": None, "note": "empty at first run"}), encoding="utf-8")
                out.append(base / ARCHIVE_MARKER)
    return out


def describe_plan(plan: ArchivePlan) -> str:
    lines = [f"{plan.session_id}: {plan.n_legacy} legacy move(s), {plan.n_promote} promotion(s), {len(plan.skipped)} skipped"]
    for m in plan.moves:
        lines.append(f"  [{m.kind:7s}] {m.src}  ->  {m.dst}")
    for s in plan.skipped:
        lines.append(f"  [skip   ] {s['src']}  ({s['reason']})")
    for n in plan.notes:
        lines.append(f"  note: {n}")
    return "\n".join(lines)
