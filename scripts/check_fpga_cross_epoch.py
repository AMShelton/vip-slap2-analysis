"""Validate saved DI3 cycle registrations against the FPGA counter shared by all acquisitions.

For every multi-acquisition session this predicts each identified epoch's
offset from the *other* acquisitions' pulses (leave-one-out through the shared
200 MHz FPGA timestamps) and reports whether the counted offset is confirmed;
for ``inconsistent``/``ambiguous`` epochs it says where the FPGA puts the
cycles that have no pulses. Read-only except for
``analysis/qc/sync/fpga_cross_epoch_DMD<n>.json`` per session.

    python scripts/check_fpga_cross_epoch.py --summary-xlsx Z:/VIP_synaptic_dynamics/VIP_SD_summary.xlsx \
        --indicator ASAP7y ASAP8 --out data_inventory/fpga_cross_epoch_voltage.csv
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vip_slap2_analysis.common.cycle_registration import load_cycle_registration  # noqa: E402
from vip_slap2_analysis.common.fpga_cross_epoch import run_fpga_cross_epoch_check  # noqa: E402
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", nargs="*", default=None)
    ap.add_argument("--indicator", nargs="*", default=None)
    ap.add_argument("--session-type", nargs="*", default=None)
    ap.add_argument("--dmd", type=int, default=1, help="DMD whose DI3 registration is checked (default 1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    reg = VIPSessionRegistry.from_excel(args.summary_xlsx)
    df = reg.sessions(session_types=args.session_type, indicators=args.indicator)
    if args.session:
        df = df[df["session_id"].astype(str).isin(set(args.session))]
    rows = []
    for _, r in df.iterrows():
        sid = str(r["session_id"])
        try:
            asset = reg.resolve_assets(r)
            dyn = asset.get_asset("slap2", "dynamic_data_dir")
            saved = load_cycle_registration(asset.qc_dir, args.dmd) if getattr(asset, "qc_dir", None) else None
            if saved is None:
                rows.append({"session_id": sid, "dmd": args.dmd, "result": "no_saved_registration"})
                print(f"{sid}: no saved cycle registration for DMD{args.dmd}")
                continue
            if len(saved.epochs) < 2:
                rows.append({"session_id": sid, "dmd": args.dmd, "result": "single_acquisition", "registration_status": saved.status})
                print(f"{sid}: single acquisition ({saved.status}); nothing to cross-check")
                continue
            if dyn is None:
                rows.append({"session_id": sid, "dmd": args.dmd, "result": "no_dynamic_data", "registration_status": saved.status})
                continue
            chk = run_fpga_cross_epoch_check(asset, dmd=args.dmd, dynamic_data_dir=dyn, reg=saved, save=not args.dry_run)
            for line in chk.summary_lines():
                print(line)
            row = {
                "session_id": sid, "dmd": args.dmd, "result": "checked", "registration_status": saved.status,
                "n_source_epochs": chk.n_source_epochs, "n_identified_epochs": chk.n_identified_epochs,
                "fpga_counter_continuous": chk.fpga_counter_continuous, "global_fit_n_pulses": chk.global_fit_n_pulses,
                "harp_over_fpga_ppm": (chk.global_rate_ratio_harp_over_fpga - 1) * 1e6 if chk.global_rate_ratio_harp_over_fpga else None,
                "global_residual_rms_us": (chk.global_residual_rms_s or float("nan")) * 1e6,
                "global_residual_max_us": (chk.global_residual_max_s or float("nan")) * 1e6,
                "all_identified_offsets_confirmed": chk.all_identified_offsets_confirmed,
                "n_relation_steps": len(chk.relation_steps),
                "relation_steps_ms": "; ".join(f"{st['step_s']*1e3:+.2f}@{st['harp_s']:.3f}" for st in chk.relation_steps),
                "limitations": " | ".join(chk.limitations),
            }
            for e in chk.epochs:
                p = f"epoch{e.source_epoch_index}_"
                row.update({p + "status": e.registration_status, p + "counted_offset": e.counted_offset,
                            p + "offset_residual_cycles": e.offset_residual_cycles, p + "confirmed": e.agrees_within_half_cycle,
                            p + "fpga_offset_estimate": e.fpga_offset_estimate_cycles,
                            p + "cycles_before_first_pulse": e.cycles_predicted_before_first_pulse,
                            p + "cycles_inside_train": e.cycles_predicted_inside_train,
                            p + "cycles_after_last_pulse": e.cycles_predicted_after_last_pulse,
                            p + "n_pulses": e.n_observed_pulses, p + "notes": " | ".join(e.notes)})
            rows.append(row)
        except Exception as exc:
            rows.append({"session_id": sid, "dmd": args.dmd, "result": "error", "limitations": f"{type(exc).__name__}: {exc}"})
            print(f"{sid}: ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
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
