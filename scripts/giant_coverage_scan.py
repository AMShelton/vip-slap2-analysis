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
                    lr = c["line_rate_hz"] if c["line_rate_hz"] == c["line_rate_hz"] else LINE_RATE_FALLBACK_HZ
                    inv = c["invalid_trials"]
                    spans = "; ".join(f"{x['trial']}:{x['first_line']/lr:.0f}-{x['last_line']/lr:.0f}s" for x in inv if "first_line" in x)
                    frac = c["valid_fraction_of_acquisition"]
                    rows.append({"session_id": sid, "dmd": dmd, "status": "ok" if (frac is None or frac >= a.min_fraction) else "low_coverage",
                                 "n_trials": c["n_trials"], "n_valid_trials": c["n_valid_trials"], "valid_fraction": frac,
                                 "first_invalid_s": (inv[0]["first_line"] / lr) if inv and "first_line" in inv[0] else None,
                                 "invalid_spans": spans, "summary_h5": str(path)})
                    print(f"{sid} DMD{dmd}: {c['n_valid_trials']}/{c['n_trials']} trials valid, {frac if frac is None else round(frac, 3)} covered"
                          + (f"; invalid: {spans[:160]}" if spans else ""))
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
