"""Move legacy post-processing products out of the way so the re-extraction writes the canonical location.

For every registry session the contents of

    analysis/derived/glutamate/   analysis/qc/glutamate/   analysis/qc/slap2/     -> legacy_LoCo/
    analysis/derived/calcium/     analysis/qc/calcium/                            -> legacy_LoCo/
    analysis/derived/voltage/     analysis/qc/voltage/                            -> legacy_extractDendrites/

(files *and* sub-folders, e.g. glutamate_analysis/, movies/, sync_compare/) are moved into a
``legacy_<preprocessor>/`` folder inside the same directory, with an ``_archive_manifest.json``
listing what moved and when. ``qc/behavior`` and ``qc/sync`` are not touched; nothing is
deleted; raw data is never read.

Safety: a session is skipped (reported ``needs_review``) when the legacy folder already exists,
or when the directory already holds new-pipeline products (``*voltex_dff*`` files, or an
extraction QC JSON carrying ``trace_source_effective``) - the migration is meant to run once,
*before* the re-extraction. An existing ``<OUTPUT_SUBDIR>`` folder from an earlier re-extraction
run is reported; ``--promote-subdir NAME`` moves its contents up to the modality root after the
legacy move.

Dry run by default: prints and writes the plan. ``--apply`` performs the moves.

    python scripts/archive_legacy_derived.py --summary-xlsx Z:/VIP_synaptic_dynamics/VIP_SD_summary.xlsx \
        --out data_inventory/archive_legacy_plan.csv
    python scripts/archive_legacy_derived.py --summary-xlsx ... --apply --out data_inventory/archive_legacy_done.csv
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402

LEGACY_NAME = {"glutamate": "legacy_LoCo", "calcium": "legacy_LoCo", "voltage": "legacy_extractDendrites", "slap2": "legacy_LoCo"}
TARGETS = [("derived", "glutamate"), ("qc", "glutamate"), ("qc", "slap2"), ("derived", "calcium"), ("qc", "calcium"),
           ("derived", "voltage"), ("qc", "voltage")]


def _is_new_product(p: Path) -> bool:
    name = p.name.lower()
    if "voltex_dff" in name:
        return True
    if p.suffix.lower() == ".json" and "extraction_qc" in name:
        try:
            return "trace_source_effective" in json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
    return False


def plan_session(asset, *, promote_subdir: str | None) -> list[dict]:
    rows = []
    for family, modality in TARGETS:
        base = Path(getattr(asset, f"{family}_dir")) / modality
        if not base.exists():
            continue
        legacy = base / LEGACY_NAME[modality]
        entries = sorted(p for p in base.iterdir() if not p.name.startswith("legacy_") and p.name != promote_subdir)
        if not entries:
            continue
        status = "move"
        reason = ""
        if legacy.exists():
            status, reason = "needs_review", f"{legacy.name} already exists"
        elif any(_is_new_product(p) for p in entries):
            status, reason = "needs_review", "new-pipeline products already present at the root"
        for p in entries:
            rows.append({"session_id": asset.session_id, "family": family, "modality": modality, "action": status, "reason": reason,
                         "kind": "dir" if p.is_dir() else "file", "source": str(p), "destination": str(legacy / p.name)})
        if promote_subdir and (base / promote_subdir).exists():
            for p in sorted((base / promote_subdir).iterdir()):
                rows.append({"session_id": asset.session_id, "family": family, "modality": modality,
                             "action": "promote" if status == "move" else "needs_review", "reason": reason or f"from {promote_subdir}",
                             "kind": "dir" if p.is_dir() else "file", "source": str(p), "destination": str(base / p.name)})
    return rows


def apply_rows(rows: list[dict]) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    by_dest_dir: dict[Path, list[dict]] = {}
    for r in rows:
        if r["action"] == "move":
            by_dest_dir.setdefault(Path(r["destination"]).parent, []).append(r)
    for legacy, items in by_dest_dir.items():
        legacy.mkdir(parents=True, exist_ok=True)
        moved = []
        for r in items:
            src, dst = Path(r["source"]), Path(r["destination"])
            if dst.exists():
                r["action"], r["reason"] = "skipped", "destination exists"
                continue
            shutil.move(str(src), str(dst))
            r["action"] = "moved"
            moved.append({"name": src.name, "kind": r["kind"]})
        manifest = {"archived_at": stamp, "reason": "legacy preprocessor products (SummaryLoCo / extractDendrites) moved aside for the "
                    "GIAnT/Voltex-based re-extraction; canonical products now live one level up", "items": moved}
        (legacy / "_archive_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for r in rows:
        if r["action"] == "promote":
            src, dst = Path(r["source"]), Path(r["destination"])
            if dst.exists():
                r["action"], r["reason"] = "skipped", "destination exists"
                continue
            shutil.move(str(src), str(dst))
            r["action"] = "promoted"
    # remove emptied promote folders
    for r in rows:
        if r["action"] == "promoted":
            parent = Path(r["source"]).parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", nargs="*", default=None)
    ap.add_argument("--session-type", nargs="*", default=None)
    ap.add_argument("--promote-subdir", default=None, help="an earlier re-extraction sub-folder whose contents become the root products")
    ap.add_argument("--apply", action="store_true", help="perform the moves (default: dry run)")
    ap.add_argument("--out", default=None, help="plan / result CSV")
    a = ap.parse_args(argv)

    reg = VIPSessionRegistry.from_excel(a.summary_xlsx)
    df = reg.sessions(session_types=a.session_type)
    if a.session:
        df = df[df["session_id"].astype(str).isin(set(a.session))]
    rows: list[dict] = []
    for _, r in df.iterrows():
        try:
            asset = reg.resolve_assets(r)
        except Exception as exc:
            rows.append({"session_id": str(r["session_id"]), "action": "error", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if asset.derived_dir is None or asset.qc_dir is None:
            continue
        rows.extend(plan_session(asset, promote_subdir=a.promote_subdir))
    table = pd.DataFrame(rows)
    if table.empty:
        print("nothing to do")
        return 0
    print(table.groupby(["action", "family", "modality"], dropna=False).size().to_string())
    review = table[table["action"] == "needs_review"]
    if len(review):
        print("\nneeds review:")
        print(review[["session_id", "family", "modality", "reason"]].drop_duplicates().to_string(index=False))
    if a.apply:
        apply_rows(rows)
        table = pd.DataFrame(rows)
        print("\nafter apply:")
        print(table.groupby(["action"], dropna=False).size().to_string())
    else:
        print("\nDRY RUN - nothing moved. Re-run with --apply.")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
