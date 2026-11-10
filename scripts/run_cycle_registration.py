"""Register HARP DI3 pulses to SLAP2 acquisition cycles for registry sessions.

Writes ``analysis/qc/sync/cycle_registration_DMD<n>.json`` (+ pulse ``.npz``) per
session/DMD and a session-level status table. Read-only with respect to
physiology, behavior and raw files.

Examples
--------
    python scripts/run_cycle_registration.py --summary-xlsx Z:/VIP_synaptic_dynamics/VIP_SD_summary.xlsx \
        --session 863774_2026-08-04_08-12-47 --dmd 1 2 --out data_inventory/sync_status.csv

    python scripts/run_cycle_registration.py --summary-xlsx ... --indicator ASAP7y ASAP8 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vip_slap2_analysis.common.cycle_registration import run_cycle_registration  # noqa: E402
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402


def registration_row(reg, session_id: str, dmd: int) -> dict:
    row = {
        "session_id": session_id,
        "dmd": dmd,
        "status": reg.status,
        "safe_for_physiology": reg.safe_for_physiology,
        "relationship": reg.relationship,
        "unmatched_di3_trains": len(reg.inputs.get("unmatched_di3_trains", []) or []),
        "wall_clock_offset_estimates": str(reg.wall_clock_evidence.get("epoch_offset_estimates", "")),
        "pc_clock_discrepancy_s": str(reg.wall_clock_evidence.get("pc_clock_discrepancy_s", "")),
        "acquisition_start_order": reg.acquisition_start_order,
        "start_order_evidence": reg.start_order_evidence,
        "harp_n_rising_edges": reg.harp_n_rising_edges,
        "n_source_epochs": reg.n_source_epochs,
        "n_di3_epochs": reg.n_di3_epochs,
        "lines_per_cycle": reg.lines_per_cycle,
        "nominal_cycle_period_s": reg.nominal_cycle_period_s,
        "observed_cycle_period_s": reg.observed_cycle_period_s,
        "period_error_ppm": reg.period_error_ppm,
        "limitations": " | ".join(reg.limitations),
    }
    for e in reg.epochs:
        p = f"epoch{e.source_epoch_index}_"
        row.update({
            p + "status": e.status, p + "anchor": e.anchor, p + "offset": e.cycle_index_offset,
            p + "offset_range": f"{e.candidate_offset_range[0]}..{e.candidate_offset_range[1]}",
            p + "n_source_cycles": e.n_source_cycles, p + "n_observed_pulses": e.n_observed_pulses,
            p + "missing_initial": e.missing_initial_cycles, p + "missing_terminal": e.missing_terminal_cycles,
            p + "internal_missing_pulses": e.n_internal_missing_pulses,
            p + "lead_s": e.lead_s, p + "tail_s": e.tail_s, p + "residual_rms_us": (e.residual_rms_s or 0) * 1e6,
        })
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", nargs="*", default=None, help="session IDs (default: all matching filters)")
    ap.add_argument("--indicator", nargs="*", default=None, help="indicator1 filter, e.g. ASAP7y ASAP8")
    ap.add_argument("--session-type", nargs="*", default=None)
    ap.add_argument("--modality", choices=["voltage", "glutamate"], default=None,
                    help="voltage = indicator1 contains 'ASAP'; glutamate = everything else")
    ap.add_argument("--dmd", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--out", default=None, help="status CSV path")
    ap.add_argument("--dry-run", action="store_true", help="compute but do not write qc/sync files")
    ap.add_argument("--no-raw", action="store_true", help="do not read raw .dat headers (use summary metadata only)")
    ap.add_argument("--no-head-assumption", action="store_true",
                    help="for integer-line DI3 cadences, do NOT take a head-anchored first pulse as line 0 (leaves them ambiguous)")
    args = ap.parse_args(argv)

    reg = VIPSessionRegistry.from_excel(args.summary_xlsx)
    df = reg.sessions(session_types=args.session_type, indicators=args.indicator)
    if args.session:
        df = df[df["session_id"].astype(str).isin(set(args.session))]
    if args.modality:
        is_v = df["indicator1"].astype(str).str.lower().str.contains("asap")
        df = df[is_v] if args.modality == "voltage" else df[~is_v]
    rows = []
    for _, r in df.iterrows():
        sid = str(r["session_id"])
        try:
            asset = reg.resolve_assets(r)
        except Exception as exc:
            rows.append({"session_id": sid, "status": "registry_error", "limitations": f"{type(exc).__name__}: {exc}"})
            continue
        dyn = None if args.no_raw else asset.get_asset("slap2", "dynamic_data_dir")
        done = {}
        for dmd in sorted(args.dmd):
            try:
                summary = None
                if dyn is None:
                    from vip_slap2_analysis.io.summary_loader import load_glutamate_summary, load_voltage_summary
                    try:
                        summary, _ = load_voltage_summary(asset)
                    except Exception:
                        summary, _ = load_glutamate_summary(asset)
                out = run_cycle_registration(asset, dmd=dmd, summary=summary, dynamic_data_dir=dyn, save=not args.dry_run,
                                             assume_first_pulse_is_acquisition_start=not args.no_head_assumption)
                # A DMD whose DI3 cadence is not its own cycle clock is registered through the FPGA
                # timestamps shared with an already-identified DMD (normally DMD2 via DMD1).
                if out.relationship != "direct_dmd_cycle_clock" and dyn is not None:
                    donor = next((r for r in done.values() if r.identified and r.relationship == "direct_dmd_cycle_clock"), None)
                    if donor is not None:
                        out = run_cycle_registration(asset, dmd=dmd, dynamic_data_dir=dyn, save=not args.dry_run, transfer_from=donor)
                done[dmd] = out
                if summary is not None and hasattr(summary, "close"):
                    summary.close()
                rows.append(registration_row(out, sid, dmd))
                print(f"{sid} DMD{dmd}: {out.status} ({out.acquisition_start_order}; {out.relationship}) "
                      f"offsets={[e.cycle_index_offset for e in out.epochs]} ranges={[e.candidate_offset_range for e in out.epochs]}")
                for e in out.epochs:
                    for n in e.notes:
                        print(f"    epoch {e.source_epoch_index}: {n}")
                for l in out.limitations:
                    print(f"    ! {l}")
            except Exception as exc:
                rows.append({"session_id": sid, "dmd": dmd, "status": "error", "limitations": f"{type(exc).__name__}: {exc}"})
                print(f"{sid} DMD{dmd}: ERROR {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)
        # second pass: a DMD processed before its donor (e.g. DI3 is DMD2's clock) gets the transfer now
        donor = next((r for r in done.values() if r.identified and r.relationship == "direct_dmd_cycle_clock"), None)
        if donor is not None and dyn is not None:
            for dmd, out in list(done.items()):
                if dmd != donor.dmd and not out.identified and out.relationship not in ("direct_dmd_cycle_clock", "transferred_from_other_dmd_shared_fpga"):
                    try:
                        out2 = run_cycle_registration(asset, dmd=dmd, dynamic_data_dir=dyn, save=not args.dry_run, transfer_from=donor)
                        done[dmd] = out2
                        rows[:] = [r for r in rows if not (r.get("session_id") == sid and r.get("dmd") == dmd)]
                        rows.append(registration_row(out2, sid, dmd))
                        print(f"{sid} DMD{dmd}: {out2.status} (transferred from DMD{donor.dmd} in second pass) offsets={[e.cycle_index_offset for e in out2.epochs]}")
                    except Exception as exc:
                        print(f"{sid} DMD{dmd}: second-pass transfer failed: {type(exc).__name__}: {exc}")
    table = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    else:
        print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
