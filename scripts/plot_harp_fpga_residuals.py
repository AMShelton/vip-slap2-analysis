"""Plot DI3 rise time minus the HARP~FPGA affine prediction, per epoch and globally, for one session.

Diagnoses sessions where the shared-FPGA fit is poor (e.g. 826031_2026-02-11:
525 us RMS across acquisitions; DMD2 transfer refused at 400-600 us within an
epoch). A HARP timestamp step shows as a jump in the residual; dropped or
duplicated pulses as isolated spikes; a non-affine clock as curvature.

    python scripts/plot_harp_fpga_residuals.py --summary-xlsx Z:/VIP_synaptic_dynamics/VIP_SD_summary.xlsx \
        --session 826031_2026-02-11_12-42-21 826031_2026-02-02_10-23-53
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vip_slap2_analysis.common.cycle_registration import load_cycle_registration  # noqa: E402
from vip_slap2_analysis.common.fpga_cross_epoch import absolute_fpga_cycle_starts, _pulse_cycle_indices  # noqa: E402
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--dmd", type=int, default=1)
    ap.add_argument("--out-dir", default=None, help="default: <session>/analysis/qc/sync")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    registry = VIPSessionRegistry.from_excel(args.summary_xlsx)
    df = registry.sessions()
    for sid in args.session:
        r = df[df["session_id"].astype(str) == sid]
        if not len(r):
            print(f"{sid}: not in registry"); continue
        asset = registry.resolve_assets(r.iloc[0])
        reg = load_cycle_registration(asset.qc_dir, args.dmd)
        dyn = asset.get_asset("slap2", "dynamic_data_dir")
        if reg is None or dyn is None:
            print(f"{sid}: missing registration or dynamic_data"); continue
        from vip_slap2_analysis.common.cycle_registration import default_timing_cache_dir
        abs_ts, info = absolute_fpga_cycle_starts(dyn, args.dmd, cache_dir=default_timing_cache_dir(asset))
        xs, ys, labels = [], [], []
        for e in reg.epochs:
            if e.cycle_index_offset is None or e.pulse_times_harp_s is None or int(e.source_epoch_index) not in abs_ts:
                continue
            idx, pulses = _pulse_cycle_indices(e)
            ts = abs_ts[int(e.source_epoch_index)]
            ok = idx < ts.size
            xs.append(ts[idx[ok]]); ys.append(pulses[ok]); labels.append(int(e.source_epoch_index))
        if not xs:
            print(f"{sid}: no identified epoch with pulses"); continue
        X = np.concatenate(xs); Y = np.concatenate(ys)
        a, b = np.polyfit(X, Y, 1)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        t0 = X[0]
        for x, y, lab in zip(xs, ys, labels):
            rg = y - (a * x + b)
            ag, bg = np.polyfit(x, y, 1)
            rl = y - (ag * x + bg)
            axes[0].plot(x - t0, rg * 1e6, ".", ms=2, label=f"epoch {lab}: global fit, {np.sqrt(np.mean(rg**2))*1e6:.0f} us RMS")
            axes[1].plot(x - t0, rl * 1e6, ".", ms=2, label=f"epoch {lab}: own fit, {np.sqrt(np.mean(rl**2))*1e6:.0f} us RMS, {(ag-1)*1e6:+.2f} ppm")
            # report the largest residual steps (HARP timestamp discontinuities)
            d = np.diff(rl)
            big = np.flatnonzero(np.abs(d) > 2e-4)
            for i in big[:10]:
                print(f"{sid} epoch {lab}: residual step {d[i]*1e6:+.0f} us at FPGA +{x[i+1]-t0:.3f} s (pulse {i+1})")
            if big.size > 10:
                print(f"{sid} epoch {lab}: ... {big.size} steps > 200 us in total")
        axes[0].set_ylabel("DI3 - (a*FPGA+b) [us]\nsingle fit, all epochs"); axes[0].legend(fontsize=7); axes[0].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("DI3 - own-epoch fit [us]"); axes[1].legend(fontsize=7); axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xlabel("FPGA time since first fitted pulse [s]")
        fig.suptitle(f"{sid} DMD{args.dmd}: HARP vs FPGA residuals (global a-1 = {(a-1)*1e6:+.2f} ppm; refs {[v['reference_timestamp'] for _, v in sorted(info.items())]})", fontsize=9)
        fig.tight_layout()
        out_dir = Path(args.out_dir) if args.out_dir else Path(asset.qc_dir) / "sync"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"harp_fpga_residuals_DMD{args.dmd}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"{sid}: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
