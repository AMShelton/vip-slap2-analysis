"""Valid-trial coverage of every GIAnT experiment_summary.h5 in the registry, per DMD.

GIAnT marks trials it could not process as having zero frames; extraction drops every event inside
them. This scan reports, for each kept glutamate session and DMD, how many trials are valid and what
fraction of the acquisition they cover, with the time spans of the invalid trials, so sessions that
need a GIAnT re-run are known before the batch extraction (803496_2025-07-28: DMD2 invalid from 393 s on).

    python scripts/giant_coverage_scan.py --summary-xlsx Z:/VIP_synaptic_dynamics/VIP_SD_summary.xlsx --out data_inventory/giant_coverage.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_modern_legacy_summary import valid_trial_coverage  # noqa: E402
from vip_slap2_analysis.io.giant_summary import GiantExperimentSummary  # noqa: E402
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402

LINE_RATE_FALLBACK_HZ = 10686.6669921875


def per_trial_motion_summary(summary_h5, dmd: int, coverage: dict) -> dict:
    """Motion-censoring and registration-shift statistics from frame_info of the valid trials.

    GIAnT (SILo.m) invalidates a trial when its mean image correlates poorly with the cross-trial
    template or when too many of its pixels are NaN; pixels go NaN where frames were censored by the
    motion rule (``motionThresh`` on the normalised registration error). The censored-frame fraction
    and the offline shift magnitude of the *valid* trials, early vs late, say whether the recording was
    drifting / moving into the region where trials start failing.
    """
    import h5py
    import numpy as np
    out = {"discard_frac_all": None, "discard_frac_last_valid_trial": None, "shift_px_first_valid_trial": None, "shift_px_last_valid_trial": None}
    try:
        with h5py.File(summary_h5, "r") as f:
            g = f[f"Path{int(dmd)}"]
            if "frame_info/discard_frames" not in g:
                return out
            disc = np.asarray(g["frame_info/discard_frames"][()]).ravel().astype(bool)
            tnf = np.asarray(g["frame_info/trial_num_frames"][()]).ravel().astype(int)
            offs = np.r_[0, np.cumsum(tnf)]
            valid = [t for t in range(1, tnf.size + 1) if tnf[t - 1] > 0]
            out["discard_frac_all"] = float(disc.mean()) if disc.size else None
            xs = np.asarray(g["frame_info/offlineXshifts"][()]).ravel() if "frame_info/offlineXshifts" in g else None
            ys = np.asarray(g["frame_info/offlineYshifts"][()]).ravel() if "frame_info/offlineYshifts" in g else None
            def _trial(t):
                sl = slice(int(offs[t - 1]), int(offs[t]))
                d = float(disc[sl].mean()) if sl.stop > sl.start else None
                sh = None
                if xs is not None and ys is not None and xs.size >= sl.stop:
                    sh = float(np.nanmedian(np.hypot(xs[sl], ys[sl])))
                return d, sh
            if valid:
                d0, s0 = _trial(valid[0]); d1, s1 = _trial(valid[-1])
                out.update({"discard_frac_last_valid_trial": d1, "shift_px_first_valid_trial": s0 if s0 is not None else float("nan"),
                            "shift_px_last_valid_trial": s1 if s1 is not None else float("nan")})
    except Exception:
        pass
    return out


def trial_qc_summary(summary_h5, dmd: int) -> dict:
    """Read Path<n>/trial_qc written by the parameterised SILo gate (GIAnT fix/aind-slap2-dynamic-data, 66f7ca8)."""
    import h5py
    import numpy as np
    out = {"qc_rule": None}
    try:
        with h5py.File(summary_h5, "r") as f:
            g = f.get(f"Path{int(dmd)}/trial_qc")
            if g is None or "rule" not in g:
                return out
            def _s(k):
                v = g[k][()]
                if isinstance(v, bytes):
                    return v.decode()
                v = np.asarray(v)
                return "".join(chr(int(c)) for c in v.ravel()) if v.dtype.kind in "iu" else str(v.ravel()[0]) if v.size == 1 else str(v)
            corr = np.asarray(g["template_corr"][()], dtype=float).ravel() if "template_corr" in g else np.array([])
            valid = np.asarray(g["valid"][()], dtype=float).ravel().astype(bool) if "valid" in g else np.zeros(corr.size, bool)
            keep = np.asarray(g["trial_table_keep"][()], dtype=float).ravel().astype(bool) if "trial_table_keep" in g else np.ones(corr.size, bool)
            reasons = _s("excluded_reason") if "excluded_reason" in g else ""
            from collections import Counter
            rc = Counter(r for r in reasons.split(";") if r)
            fin = np.isfinite(corr)
            out.update({
                "qc_rule": _s("rule"), "qc_corr_thresh": float(np.asarray(g["corr_thresh"][()]).ravel()[0]) if "corr_thresh" in g else None,
                "qc_legacy_thresh": float(np.asarray(g["legacy_outlier_thresh"][()]).ravel()[0]) if "legacy_outlier_thresh" in g else None,
                "qc_n_trial_table_rejected": int((~keep).sum()), "qc_n_not_aligned": int((keep & ~fin).sum()),
                "qc_reasons": dict(rc),
                "qc_corr_valid_min": float(np.nanmin(corr[valid])) if valid.any() and fin[valid].any() else None,
                "qc_corr_valid_median": float(np.nanmedian(corr[valid])) if valid.any() and fin[valid].any() else None,
                "qc_corr_excluded_max": float(np.nanmax(corr[keep & fin & ~valid])) if (keep & fin & ~valid).any() else None,
                "qc_corr_excluded_median": float(np.nanmedian(corr[keep & fin & ~valid])) if (keep & fin & ~valid).any() else None,
            })
            # per-trial failures recorded (not fatal) since GIAnT fix/aind-slap2-dynamic-data 2026-11-15
            for key in ("extraction_failed_message", "localization_message"):
                if key in g:
                    msgs = [m for m in _s(key).split(";") if m]
                    if msgs:
                        out[f"qc_{key}"] = Counter(msgs).most_common(3)
            if "extraction_failed_trials" in g:
                out["qc_extraction_failed_trials"] = np.asarray(g["extraction_failed_trials"][()], dtype=float).ravel().astype(int).tolist()
    except Exception as exc:
        out["qc_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", nargs="*", default=None)
    ap.add_argument("--min-fraction", type=float, default=0.95, help="flag DMDs covering less than this fraction")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    reg = VIPSessionRegistry.from_excel(a.summary_xlsx)
    df = reg.sessions()
    if a.session:
        df = df[df["session_id"].astype(str).isin(set(a.session))]
    rows = []
    for _, r in df.iterrows():
        sid = str(r["session_id"])
        if str(r.get("indicator1", "")).lower().startswith("asap"):
            continue
        try:
            asset = reg.resolve_assets(r)
            path = asset.get_asset("glutamate", "summary_h5")
            if path is None or asset.get_asset("glutamate", "summary_format") != "giant_experiment_summary_h5":
                rows.append({"session_id": sid, "status": "no_giant_summary"}); continue
            g = GiantExperimentSummary(path)
            try:
                for dmd in range(1, int(g.n_dmds) + 1):
                    c = valid_trial_coverage(g, dmd)
                    motion = per_trial_motion_summary(path, dmd, c)
                    tq = trial_qc_summary(path, dmd)
                    lr = c["line_rate_hz"] if c["line_rate_hz"] == c["line_rate_hz"] else LINE_RATE_FALLBACK_HZ
                    inv = c["invalid_trials"]
                    spans = "; ".join(f"{x['trial']}:{x['first_line']/lr:.0f}-{x['last_line']/lr:.0f}s" for x in inv if "first_line" in x)
                    frac = c["valid_fraction_of_acquisition"]
                    rows.append({"session_id": sid, "dmd": dmd, "status": "ok" if (frac is None or frac >= a.min_fraction) else "low_coverage",
                                 "n_trials": c["n_trials"], "n_valid_trials": c["n_valid_trials"], "valid_fraction": frac,
                                 "first_invalid_s": (inv[0]["first_line"] / lr) if inv and "first_line" in inv[0] else None,
                                 "invalid_spans": spans, "summary_h5": str(path), **motion, **tq})
                    print(f"{sid} DMD{dmd}: {c['n_valid_trials']}/{c['n_trials']} trials valid, {frac if frac is None else round(frac, 3)} covered"
                          + (f"; invalid: {spans[:160]}" if spans else "")
                          + (f"; censored frames {motion['discard_frac_all']:.1%} (last valid trial {motion['discard_frac_last_valid_trial']:.1%}); "
                             f"median |shift| first/last valid trial {motion['shift_px_first_valid_trial']:.1f}/{motion['shift_px_last_valid_trial']:.1f} px"
                             if motion.get("discard_frac_all") is not None else "")
                          + (f"\n    trial_qc: rule={tq['qc_rule']} corr_thresh={tq.get('qc_corr_thresh')} (legacy would be {tq.get('qc_legacy_thresh')}); "
                             f"trial-table rejected={tq.get('qc_n_trial_table_rejected')}, not aligned={tq.get('qc_n_not_aligned')}, reasons={tq.get('qc_reasons')}; "
                             f"template corr valid min/median={tq.get('qc_corr_valid_min')}/{tq.get('qc_corr_valid_median')}, "
                             f"excluded max/median={tq.get('qc_corr_excluded_max')}/{tq.get('qc_corr_excluded_median')}"
                             if tq.get("qc_rule") else "\n    trial_qc: absent (summary written before the parameterised gate)"))
            finally:
                g.close()
        except Exception as exc:
            rows.append({"session_id": sid, "status": f"error: {type(exc).__name__}: {exc}"})
            print(f"{sid}: ERROR {type(exc).__name__}: {exc}")
    t = pd.DataFrame(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); t.to_csv(a.out, index=False); print(f"wrote {a.out}")
    if "status" in t:
        print(t["status"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
