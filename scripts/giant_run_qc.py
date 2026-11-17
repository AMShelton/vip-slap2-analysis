"""Quality check of a GIAnT re-run (registration -> SILo -> experiment_summary.h5) for given sessions.

Three layers, each with explicit expectations (PASS / WARN / FAIL):

1. Registration (motion_correction/*_ALIGNMENTDATA.h5 + trial_table.h5)
   - every alignment file was produced with the NaN-preserving clip (/clipKeepsNaN);
     none with the 2026-11-13 max(x,0) clip (/negativePixelFraction without the marker)
   - xcorr cost ~40 ms/frame (the broken clip gave ~480)
   - registration_failed count per DMD and the shift-std distribution vs params.maxMotionStd_px
   - negativePixelFraction per DMD (how far below zero that path's offset sits)
2. SILo trial validity (Path<n>/trial_qc): valid trials, exclusion reasons, template-correlation
   ranges, localization / extraction failures; coverage vs the 2026-11-12 baseline table.
3. Products (Path<n>/sources): source count, frames, dF_ls / F0 sanity (NaN fraction within valid
   trials, F0 > 0), SNR; which experiment_summary.h5 the registry resolves (all candidates listed,
   newest expected).

    python scripts/giant_run_qc.py --summary-xlsx Z:/VIP_synaptic_dynamics/VIP_SD_summary.xlsx \
        --session 803121_2025-10-29_11-19-29 803121_2025-10-30_11-13-32 803121_2025-11-01_19-00-21 \
        --baseline data_inventory/giant_coverage.csv --out data_inventory/giant_run_qc_803121.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vip_slap2_analysis.io.giant_summary import GiantExperimentSummary  # noqa: E402
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402

_spec = importlib.util.spec_from_file_location("giant_coverage_scan", Path(__file__).with_name("giant_coverage_scan.py"))
_cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cov)  # type: ignore[union-attr]


def _scalar(f, key, default=None):
    try:
        return float(np.asarray(f[key][()]).ravel()[0])
    except Exception:
        return default


def registration_qc(moco_dir: Path, trial_table: Path | None, dmd: int) -> dict:
    files = sorted(moco_dir.glob(f"E*T*DMD{dmd}_ALIGNMENTDATA.h5")) if moco_dir.exists() else []
    out = {"reg_n_files": len(files), "reg_n_nan_clip_bad": 0, "reg_n_clip_ok": 0, "reg_n_legacy_version": 0}
    ms, stds, negfrac, flagged, thr = [], [], [], 0, None
    for fp in files:
        try:
            with h5py.File(fp, "r") as f:
                has_neg = "negativePixelFraction" in f
                has_mark = "clipKeepsNaN" in f
                if has_neg and not has_mark:
                    out["reg_n_nan_clip_bad"] += 1
                elif has_mark:
                    out["reg_n_clip_ok"] += 1
                else:
                    out["reg_n_legacy_version"] += 1
                if has_neg:
                    negfrac.append(_scalar(f, "negativePixelFraction"))
                if "motionStd_px" in f:
                    stds.append(float(np.nanmax(np.asarray(f["motionStd_px"][()], dtype=float))))
                elif "motionDSc" in f:
                    stds.append(float(max(np.nanstd(f["motionDSc"][()]), np.nanstd(f["motionDSr"][()]))))
                if "maxMotionStd_px" in f:
                    thr = _scalar(f, "maxMotionStd_px")
                if "registrationFailed" in f and int(np.asarray(f["registrationFailed"][()]).ravel()[0]):
                    flagged += 1
                if "runtime/correlation_s" in f and "motionDSc" in f:
                    n = int(np.asarray(f["motionDSc"]).size)
                    if n:
                        ms.append(1e3 * _scalar(f, "runtime/correlation_s") / n)
        except Exception as exc:
            out.setdefault("reg_read_errors", []).append(f"{fp.name}: {exc}")
    out.update({
        "reg_xcorr_ms_per_frame_median": float(np.median(ms)) if ms else None,
        "reg_shift_std_px_median": float(np.median(stds)) if stds else None,
        "reg_shift_std_px_max": float(np.max(stds)) if stds else None,
        "reg_max_motion_std_px": thr,
        "reg_n_flagged_in_files": flagged,
        "reg_negative_pixel_fraction_median": float(np.median(negfrac)) if negfrac else None,
    })
    if trial_table and trial_table.exists():
        try:
            with h5py.File(trial_table, "r") as f:
                rf = np.asarray(f["motion_correction/registration_failed"][()])
                rf = rf.reshape(rf.shape) if rf.ndim == 2 else rf
                # saveStructToH5 writes column-major MATLAB arrays; h5py sees (nTrials, nDMDs) or (nDMDs, nTrials)
                if rf.ndim == 2:
                    row = rf[:, dmd - 1] if rf.shape[1] < rf.shape[0] else rf[dmd - 1, :]
                else:
                    row = rf
                out["tt_n_registration_failed"] = int(np.count_nonzero(row))
                out["tt_registration_failed_trials"] = (np.flatnonzero(row) + 1).tolist()
                for k in ("maxMotionStd_px", "clipShift", "useMexWeightedXcorr", "startTime"):
                    p = f"motion_correction/align_params/{k}"
                    if p in f:
                        v = f[p][()]
                        out[f"tt_{k}"] = v.decode() if isinstance(v, bytes) else (float(np.asarray(v).ravel()[0]) if np.asarray(v).dtype.kind in "fiu" else str(np.asarray(v).ravel()[0]))
        except Exception as exc:
            out["tt_error"] = f"{type(exc).__name__}: {exc}"
    return out


def product_qc(summary_h5: Path, dmd: int, coverage: dict) -> dict:
    out = {}
    with h5py.File(summary_h5, "r") as f:
        g = f.get(f"Path{dmd}")
        if g is None:
            return {"prod_error": f"Path{dmd} missing"}
        tnf = np.asarray(g["frame_info/trial_num_frames"][()]).ravel().astype(int)
        dfl = g["sources/temporal/dF_ls"]
        f0 = g["sources/temporal/F0"]
        out["prod_n_sources"] = int(dfl.shape[2]) if dfl.ndim == 3 else None
        out["prod_n_frames"] = int(tnf.sum())
        out["prod_n_valid_trials"] = int((tnf > 0).sum())
        # NaN fraction of dF_ls within valid frames (channel 1); sample up to ~2e6 values
        if dfl.ndim == 3 and dfl.shape[0]:
            step = max(1, dfl.shape[0] // 20000)
            x = np.asarray(dfl[::step, 0, :], dtype=float)
            out["prod_dF_ls_nan_frac"] = float(np.isnan(x).mean())
            out["prod_dF_ls_abs_median"] = float(np.nanmedian(np.abs(x))) if np.isfinite(x).any() else None
            y = np.asarray(f0[::step, 0, :], dtype=float)
            out["prod_F0_median"] = float(np.nanmedian(y)) if np.isfinite(y).any() else None
            out["prod_F0_nonpositive_frac"] = float(np.mean(y[np.isfinite(y)] <= 0)) if np.isfinite(y).any() else None
        if "sources/temporal/SNR" in g:
            snr = np.asarray(g["sources/temporal/SNR"][()], dtype=float).ravel()
            out["prod_snr_median"] = float(np.nanmedian(snr))
            out["prod_snr_min"] = float(np.nanmin(snr))
        for k in ("trialCorrRule", "minTrialCorr", "maxDiscardFrac", "motionThresh", "nanThresh", "sourceDetectionMethod", "startTime", "endTime"):
            p = f"params/{k}"
            if p in f:
                v = f[p][()]
                out[f"silo_{k}"] = v.decode() if isinstance(v, bytes) else (float(np.asarray(v).ravel()[0]) if np.asarray(v).dtype.kind in "fiu" else str(np.asarray(v).ravel()[0]))
    out["prod_valid_fraction"] = coverage.get("valid_fraction_of_acquisition")
    return out


def verdicts(row: dict, baseline: dict | None) -> list[str]:
    v = []
    def mark(ok, msg, warn=False):
        v.append(("PASS " if ok else ("WARN " if warn else "FAIL ")) + msg)
    if row.get("reg_n_files"):
        mark(row["reg_n_nan_clip_bad"] == 0, f"registration clip: {row['reg_n_clip_ok']} NaN-preserving, {row['reg_n_nan_clip_bad']} with the broken max(x,0) clip, {row['reg_n_legacy_version']} pre-clip version")
        ms = row.get("reg_xcorr_ms_per_frame_median")
        if ms is not None:
            mark(ms < 120, f"xcorr {ms:.0f} ms/frame (expect ~40; ~480 meant the NaN bug)", warn=ms < 300)
        nf = row.get("tt_n_registration_failed", row.get("reg_n_flagged_in_files"))
        mark(nf <= max(3, 0.05 * row["reg_n_files"]), f"registration_failed {nf}/{row['reg_n_files']} trials {row.get('tt_registration_failed_trials', '')} at maxMotionStd_px {row.get('reg_max_motion_std_px')}; shift std median/max {row.get('reg_shift_std_px_median')}/{row.get('reg_shift_std_px_max')} px", warn=True)
        npf = row.get("reg_negative_pixel_fraction_median")
        if npf is not None:
            mark(npf < 0.25, f"negative raw pixels {npf:.1%} (offset below zero on this path if large)", warn=True)
    if row.get("qc_rule"):
        reasons = row.get("qc_reasons", {})
        mark(not any(k in reasons for k in ("localization_failed",)), f"trial exclusions: {reasons}", warn=True)
        ef = row.get("qc_extraction_failed_trials") or []
        mark(len(ef) <= 2, f"extraction failures: {ef or 'none'}", warn=True)
        cv, ce = row.get("qc_corr_valid_min"), row.get("qc_corr_excluded_max")
        v.append(f"INFO template corr: valid min/median {cv}/{row.get('qc_corr_valid_median')}, excluded max/median {ce}/{row.get('qc_corr_excluded_median')} (threshold {row.get('qc_corr_thresh')})")
    frac = row.get("valid_fraction")
    if frac is not None:
        base = baseline.get("valid_fraction") if baseline else None
        if base is not None:
            mark(frac >= base - 1e-6, f"coverage {frac:.3f} vs 2026-11-12 baseline {base:.3f} ({row['n_valid_trials']}/{row['n_trials']} trials valid, was {baseline.get('n_valid_trials')}/{baseline.get('n_trials')})")
        mark(frac >= 0.70, f"coverage {frac:.3f} {'>=' if frac >= 0.70 else '<'} 0.70 target", warn=frac >= 0.5)
    if row.get("prod_n_sources") is not None:
        mark(row["prod_n_sources"] > 0, f"{row['prod_n_sources']} sources, {row['prod_n_frames']} frames in {row['prod_n_valid_trials']} valid trials")
        nanf = row.get("prod_dF_ls_nan_frac")
        if nanf is not None:
            mark(nanf < 0.10, f"dF_ls NaN fraction within valid frames {nanf:.1%} (censored frames only expected)", warn=nanf < 0.3)
        f0m, f0bad = row.get("prod_F0_median"), row.get("prod_F0_nonpositive_frac")
        if f0m is not None:
            mark((f0bad or 0) < 0.01 and f0m > 0, f"F0 median {f0m:.3g}, non-positive {f0bad:.2%}", warn=True)
        if row.get("prod_snr_median") is not None:
            v.append(f"INFO source SNR median {row['prod_snr_median']:.2f} (min {row['prod_snr_min']:.2f})")
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--baseline", default=None, help="giant_coverage_scan CSV from before the re-run (2026-11-12)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    reg = VIPSessionRegistry.from_excel(a.summary_xlsx)
    df = reg.sessions()
    df = df[df["session_id"].astype(str).isin(set(a.session))]
    base = pd.read_csv(a.baseline) if a.baseline else None
    rows = []
    for _, r in df.iterrows():
        sid = str(r["session_id"])
        print(f"\n=== {sid} ===")
        asset = reg.resolve_assets(r)
        path = asset.get_asset("glutamate", "summary_h5")
        cands = asset.get_asset("glutamate", "summary_h5_candidates") or []
        if path is None:
            print("  no GIAnT experiment_summary.h5 resolved"); rows.append({"session_id": sid, "status": "no_summary"}); continue
        for c in cands:
            cp = Path(c)
            print(f"  candidate {'*' if cp == Path(path) else ' '} {cp}  ({datetime.fromtimestamp(cp.stat().st_mtime):%Y-%m-%d %H:%M})")
        if len(cands) > 1 and Path(path) != max((Path(c) for c in cands), key=lambda p: p.stat().st_mtime):
            print("  WARN the registry did not pick the newest summary (layout preference beats mtime); check PREFERRED_GIANT_PARTS")
        tt = asset.get_asset("glutamate", "trial_table_h5")
        tt = Path(tt) if tt else None
        moco = (tt.parent / "motion_correction") if tt else Path(path).parent.parent / "motion_correction"
        print(f"  trial table {tt}  motion_correction {'found' if moco.exists() else 'MISSING'}: {moco}")
        g = GiantExperimentSummary(path)
        try:
            for dmd in range(1, int(g.n_dmds) + 1):
                cov = _cov.valid_trial_coverage(g, dmd)
                row = {"session_id": sid, "dmd": dmd, "summary_h5": str(path),
                       "n_trials": cov["n_trials"], "n_valid_trials": cov["n_valid_trials"], "valid_fraction": cov["valid_fraction_of_acquisition"]}
                row.update(registration_qc(moco, tt, dmd))
                row.update(_cov.trial_qc_summary(path, dmd))
                row.update(product_qc(Path(path), dmd, cov))
                b = None
                if base is not None:
                    hit = base[(base["session_id"].astype(str) == sid) & (base["dmd"] == dmd)]
                    if len(hit):
                        b = hit.iloc[0].to_dict()
                print(f"\n  DMD{dmd}")
                for line in verdicts(row, b):
                    print("    " + line)
                rows.append(row)
        finally:
            g.close()
    t = pd.DataFrame(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(a.out, index=False); print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
