"""Session-level synchronization status table.

Two layers are combined:

1. *Availability* (from the VAST directory manifest ``directory_manifest.tsv`` and
   the session sheet of ``VIP_SD_summary.xlsx``; no data access needed): which
   files a DI3 cycle registration requires (``Behavior_32.bin``/``Behavior_44.bin``,
   raw ``*_DMD1*.dat`` chunks + ``.meta``), which summary products exist
   (legacy MAT / Voltex H5 / GIAnT H5), and the *expected* acquisition start order
   from the workflow history (HARP-first before 2026-07-23, SLAP2-first from
   2026-07-23 until the 2026-10-27 sync-buffer workflow).
2. *Registration results* (optional): the per-session CSV written by
   ``scripts/run_cycle_registration.py`` on a machine that can read VAST.

The output classifies every session as ``corrected`` (identified, offset>0),
``verified_no_offset`` (identified, offset 0), ``identified_with_assumptions``,
``unresolved`` (ambiguous/inconsistent), ``pending_registration`` (data present,
registration not yet run) or ``not_registrable`` (raw HARP or .dat headers
missing), with the evidence columns next to it.

    python scripts/build_sync_status_table.py --summary-xlsx <VIP_SD_summary.xlsx> \
        --manifest <directory_manifest.tsv> [--registration-csv <sync_status.csv>] --out <table.csv>
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SESSION_RE = re.compile(r"(\d{6}_20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
SLAP2_FIRST_START = pd.Timestamp("2026-07-23")
SYNC_BUFFER_WORKFLOW = pd.Timestamp("2026-10-27")


def manifest_inventory(manifest: Path) -> dict:
    inv: dict = defaultdict(lambda: {
        "behavior_32": False, "behavior_44": False, "harp_df_csv": False, "photodiode_pkl": False,
        "bonsai_event_log": False, "dmd1_dat_chunks": 0, "dmd2_dat_chunks": 0, "dmd1_meta": False,
        "imaging_epochs_csv": False, "legacy_voltage_mat": False, "legacy_voltage_h5": False,
        "legacy_summary_loco": False, "voltex_h5_paths": [], "giant_h5_paths": [], "experiment_summary_paths": [],
    })
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2 or row[1] != "file":
                continue
            path = row[0].replace("\\", "/")
            m = SESSION_RE.search(path)
            if not m:
                continue
            sid = m.group(1)
            rec = inv[sid]
            low = path.lower()
            name = low.rsplit("/", 1)[-1]
            if name == "behavior_32.bin": rec["behavior_32"] = True
            elif name == "behavior_44.bin": rec["behavior_44"] = True
            elif name == "harp_df.csv": rec["harp_df_csv"] = True
            elif name == "photodiode.pkl": rec["photodiode_pkl"] = True
            elif name.startswith("bonsai_event_log") and name.endswith(".csv"): rec["bonsai_event_log"] = True
            elif name == "imaging_epochs.csv": rec["imaging_epochs_csv"] = True
            elif "dmd1" in name and "-cycle-" in name and name.endswith(".dat") and "/dynamic_data/" in low and "/reference_stack/" not in low:
                rec["dmd1_dat_chunks"] += 1
            elif "dmd2" in name and "-cycle-" in name and name.endswith(".dat") and "/dynamic_data/" in low and "/reference_stack/" not in low:
                rec["dmd2_dat_chunks"] += 1
            elif name.endswith("_dmd1.meta") and "/dynamic_data/" in low and "/reference_stack/" not in low and "structure_volume" not in name:
                rec["dmd1_meta"] = True
            elif name.startswith("dendriticvoltagesummary") and name.endswith(".mat"): rec["legacy_voltage_mat"] = True
            elif name.startswith("dendriticvoltagetraces") and name.endswith(".h5"): rec["legacy_voltage_h5"] = True
            elif name.startswith("summaryloco") and name.endswith(".mat"): rec["legacy_summary_loco"] = True
            elif name in ("voltagesummary.h5", "voltage_summary.h5"): rec["voltex_h5_paths"].append(path)
            elif name == "experiment_summary.h5":
                # name is ambiguous (Voltex writes experiment_summary.h5 under Voltage/); classify by folder
                (rec["voltex_h5_paths"] if "/voltage/" in low else rec["giant_h5_paths"]).append(path)
                rec["experiment_summary_paths"].append(path)
    return inv


def expected_start_order(date: pd.Timestamp) -> str:
    if pd.isna(date):
        return "unknown"
    if date < SLAP2_FIRST_START:
        return "harp_first_expected"
    if date < SYNC_BUFFER_WORKFLOW:
        return "slap2_first_expected"
    return "harp_first_with_gray_buffer_expected"


def build(summary_xlsx: Path, manifest: Path, registration_csv: Path | None, *, include_glutamate: bool) -> pd.DataFrame:
    sessions = pd.read_excel(summary_xlsx, sheet_name="sessions")
    sessions["session_date"] = pd.to_datetime(sessions["session_date"], errors="coerce")
    inv = manifest_inventory(manifest)
    reg = None
    if registration_csv is not None and Path(registration_csv).exists():
        reg = pd.read_csv(registration_csv)
    rows = []
    for _, s in sessions.iterrows():
        sid = str(s["session_id"])
        ind = f"{s.get('indicator1', '')} {s.get('indicator2', '')}".lower()
        is_voltage = "asap" in ind
        if not is_voltage and not include_glutamate:
            continue
        if str(s.get("session_type", "")).lower() in ("expression_check", "volume_imaging"):
            continue
        r = inv.get(sid, None)
        row = {
            "session_id": sid, "subject_id": s.get("subject_id"), "session_type": s.get("session_type"),
            "indicator1": s.get("indicator1"), "indicator2": s.get("indicator2"), "session_date": s["session_date"].date() if not pd.isna(s["session_date"]) else None,
            "modality": "voltage" if is_voltage else "glutamate/calcium",
            "expected_start_order": expected_start_order(s["session_date"]),
            "on_vast_manifest": r is not None,
        }
        if r is not None:
            row.update({k: v for k, v in r.items() if not k.endswith("_paths")})
            row["voltex_summary_h5"] = "; ".join(r["voltex_h5_paths"])
            row["giant_summary_h5"] = "; ".join(r["giant_h5_paths"])
            row["raw_di3_available"] = bool(r["behavior_32"])
            row["raw_dat_headers_available"] = bool(r["dmd1_dat_chunks"] > 0 and r["dmd1_meta"])
            row["dmd1_cycle_count_bounds"] = (f"{(r['dmd1_dat_chunks'] - 1) * 1000 + 1}..{r['dmd1_dat_chunks'] * 1000}" if r["dmd1_dat_chunks"] else "")
            row["legacy_voltage_pair"] = bool(r["legacy_voltage_mat"] and r["legacy_voltage_h5"])
            registrable = r["behavior_32"] and r["dmd1_dat_chunks"] > 0 and r["dmd1_meta"]
            summary_metadata_fallback = (not registrable) and r["behavior_32"] and (row["legacy_voltage_pair"] or r["voltex_h5_paths"] or r["giant_h5_paths"])
        else:
            registrable = False
            summary_metadata_fallback = False
        # registration results
        status = None
        if reg is not None:
            hits = reg[(reg["session_id"].astype(str) == sid)]
            if len(hits):
                dmd1 = hits[hits.get("dmd", pd.Series(dtype=float)) == 1] if "dmd" in hits else hits
                h = (dmd1.iloc[0] if len(dmd1) else hits.iloc[0]).to_dict()
                status = h.get("status")
                row.update({f"reg_{k}": v for k, v in h.items() if k not in ("session_id",)})
        if status is None:
            if registrable:
                row["sync_classification"] = "pending_registration (raw DI3 + .dat headers present; run scripts/run_cycle_registration.py)"
            elif summary_metadata_fallback:
                row["sync_classification"] = "pending_registration_summary_metadata_only (no raw .dat headers; cycle count from summary)"
            else:
                row["sync_classification"] = "not_registrable (raw Behavior_32.bin or DMD1 .dat/.meta missing from manifest)"
        else:
            offs = [h.get(k) for k in h if k.endswith("_offset") and k.startswith("epoch")]
            offs = [o for o in offs if pd.notna(o)]
            if status == "identified":
                row["sync_classification"] = "corrected (identified, nonzero cycle offset applied)" if any(float(o) > 0 for o in offs) else "verified_no_offset (identified, first pulse = cycle 0)"
            elif status == "identified_with_assumptions":
                row["sync_classification"] = "identified_with_assumptions (count-consistent, no coverage margin; review)"
            elif status in ("ambiguous", "inconsistent", "unavailable"):
                row["sync_classification"] = f"unresolved ({status}: {h.get('limitations', '')})"
            else:
                row["sync_classification"] = f"error ({h.get('limitations', '')})"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["modality", "session_date", "session_id"]).reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--registration-csv", default=None)
    ap.add_argument("--include-glutamate", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    df = build(Path(a.summary_xlsx), Path(a.manifest), Path(a.registration_csv) if a.registration_csv else None, include_glutamate=a.include_glutamate)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df["sync_classification"].value_counts().to_string())
    print(f"wrote {a.out} ({len(df)} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
