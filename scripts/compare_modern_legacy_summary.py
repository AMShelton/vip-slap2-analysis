"""Numerically compare a GIAnT ``experiment_summary.h5`` with a legacy ``SummaryLoCo*.mat``.

The modern reader (``GiantExperimentSummary``) and the legacy reader
(``GlutamateSummary``) are opened side by side and, trial by trial, the activity
channel ``dF_ls``/``F0`` traces, frame counts and frame line indices are
compared. The comparison is the gate the project requires before the modern
product becomes the default extraction input; the registry already *prefers*
the modern file, so run this on every session that has both.

Outputs a JSON report (and prints a summary). Nothing is modified.

    python scripts/compare_modern_legacy_summary.py --modern .../experiment_summary.h5 \
        --legacy .../SummaryLoCo-260302-101842.mat --out compare_834788.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vip_slap2_analysis.glutamate.summary import GlutamateSummary  # noqa: E402
from vip_slap2_analysis.io.giant_summary import GiantExperimentSummary  # noqa: E402


def _stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return {"n": int(n), "n_finite_both": 0}
    d = a[ok] - b[ok]
    corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else float("nan")
    return {
        "n": int(n), "n_finite_both": int(ok.sum()), "n_finite_modern_only": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
        "n_finite_legacy_only": int((~np.isfinite(a) & np.isfinite(b)).sum()),
        "max_abs_diff": float(np.max(np.abs(d))), "rms_diff": float(np.sqrt(np.mean(d * d))),
        "rel_rms_diff": float(np.sqrt(np.mean(d * d)) / (np.sqrt(np.mean(b[ok] ** 2)) + 1e-12)),
        "pearson_r": corr,
        "scale_fit_modern_vs_legacy": float(np.dot(a[ok], b[ok]) / (np.dot(b[ok], b[ok]) + 1e-12)),
    }


def compare(modern_path: Path, legacy_path: Path, *, max_trials: int | None = None, dmds=(1, 2)) -> dict:
    report: dict = {"modern": str(modern_path), "legacy": str(legacy_path), "per_dmd": {}, "problems": []}
    g = GiantExperimentSummary(modern_path)
    l = GlutamateSummary(legacy_path)
    report["modern_n_trials"] = g.n_trials; report["legacy_n_trials"] = l.n_trials
    report["modern_n_dmds"] = g.n_dmds; report["legacy_n_dmds"] = l.n_dmds
    report["modern_trial_epoch_source"] = g.trial_epoch_source; report["legacy_trial_epoch_source"] = l.trial_epoch_source
    if g.n_trials != l.n_trials:
        report["problems"].append(f"trial count differs: modern {g.n_trials} vs legacy {l.n_trials}")
    same_segmentation = g.n_trials == l.n_trials

    def _coverage(summary, dmd):
        """Lines covered by valid trials vs all trials of the summary (1-based source lines), with time spans."""
        try:
            lr = float(summary.get_line_rate_hz(dmd))
        except Exception:
            lr = float("nan")
        spans, cov_valid, cov_all, invalid = [], 0, 0, []
        valid = set(int(v) for v in summary.valid_trials[dmd - 1])
        # trial table line ranges (GIAnT trial_table.h5) give the span of trials that have no frames
        tt = getattr(summary, "trial_table", None) or {}
        fl, ll = tt.get("first_line"), tt.get("last_line")

        def _table_span(t):
            if fl is None or ll is None:
                return None
            a = np.atleast_2d(fl); b = np.atleast_2d(ll)
            n = int(summary.n_trials)
            try:
                if a.shape[0] == n:                       # (nTrials, nDMDs): MATLAB nDMDs x nTrials read through h5py
                    j = dmd - 1 if a.shape[1] >= dmd else 0
                    return int(a[t - 1, j]), int(b[t - 1, j])
                if a.shape[1] == n:                       # (nDMDs, nTrials)
                    i = dmd - 1 if a.shape[0] >= dmd else 0
                    return int(a[i, t - 1]), int(b[i, t - 1])
            except Exception:
                return None
            return None

        for t in range(1, int(summary.n_trials) + 1):
            lines = None
            if t in valid:
                try:
                    lines = summary.get_trial_source_lines(dmd, t)
                except Exception:
                    lines = None
            if lines is None or len(lines) == 0:
                span = _table_span(t)
                if span is None:
                    invalid.append({"trial": t, "reason": "no frames; no trial-table span"})
                    continue
                lo, hi = span
                cov_all += hi - lo + 1
                invalid.append({"trial": t, "first_line": lo, "last_line": hi,
                                "t_start_s": lo / lr if np.isfinite(lr) else None, "t_end_s": hi / lr if np.isfinite(lr) else None})
                spans.append((lo, hi))
                continue
            lo, hi = int(np.min(lines)), int(np.max(lines))
            cov_all += hi - lo + 1
            if t in valid:
                cov_valid += hi - lo + 1
            else:
                invalid.append({"trial": t, "first_line": lo, "last_line": hi,
                                "t_start_s": lo / lr if np.isfinite(lr) else None, "t_end_s": hi / lr if np.isfinite(lr) else None})
            spans.append((lo, hi))
        total_lines = (max(h for _, h in spans) - min(l0 for l0, _ in spans) + 1) if spans else 0
        return {"n_trials": int(summary.n_trials), "n_valid_trials": len(valid), "lines_in_valid_trials": int(cov_valid),
                "lines_in_all_trials": int(cov_all), "acquisition_lines_spanned": int(total_lines),
                "valid_fraction_of_acquisition": (cov_valid / total_lines) if total_lines else None,
                "invalid_trials": invalid, "line_rate_hz": lr}

    for dmd in dmds:
        if dmd > g.n_dmds or dmd > l.n_dmds:
            continue
        out = {"modern_n_sources": g.n_synapses[dmd - 1], "legacy_n_sources": l.n_synapses[dmd - 1],
               "modern_valid_trials": g.valid_trials[dmd - 1], "legacy_valid_trials": l.valid_trials[dmd - 1], "trials": [],
               "modern_coverage": _coverage(g, dmd), "legacy_coverage": _coverage(l, dmd)}
        mc, lc = out["modern_coverage"], out["legacy_coverage"]
        if mc["valid_fraction_of_acquisition"] is not None and lc["valid_fraction_of_acquisition"] is not None and \
                abs(mc["valid_fraction_of_acquisition"] - lc["valid_fraction_of_acquisition"]) > 0.01:
            report["problems"].append(
                f"DMD{dmd}: valid-trial coverage of the acquisition differs: modern {mc['valid_fraction_of_acquisition']:.1%} "
                f"({mc['n_valid_trials']}/{mc['n_trials']} trials) vs legacy {lc['valid_fraction_of_acquisition']:.1%} "
                f"({lc['n_valid_trials']}/{lc['n_trials']}); events inside invalid trials are dropped by extraction")
        if out["modern_n_sources"] != out["legacy_n_sources"]:
            report["problems"].append(f"DMD{dmd}: source count differs ({out['modern_n_sources']} vs {out['legacy_n_sources']}); traces compared only on the common prefix, identities NOT assumed equal")
        common = sorted(set(g.valid_trials[dmd - 1]) & set(l.valid_trials[dmd - 1]))
        if set(g.valid_trials[dmd - 1]) != set(l.valid_trials[dmd - 1]):
            report["problems"].append(f"DMD{dmd}: valid trial sets differ")
        if not same_segmentation:
            report["problems"].append(f"DMD{dmd}: trial segmentation differs (modern {g.n_trials} vs legacy {l.n_trials} trials over the same "
                                      "acquisition); per-trial trace statistics are not meaningful and were skipped")
            common = []
        for t in common[: (max_trials or len(common))]:
            row = {"trial": t}
            mg = g.get_traces(dmd, t, signal="dF", mode="ls", squeeze_channels=False)[:, :, 0]
            ml = np.asarray(l.get_traces(dmd, t, signal="dF", mode="ls", squeeze_channels=False), dtype=float)
            if ml.ndim == 3:
                ml = ml[:, :, 0]
            if ml.shape[0] != mg.shape[0] and ml.shape[1] == mg.shape[0]:
                ml = ml.T
            row["modern_shape"] = list(mg.shape); row["legacy_shape"] = list(ml.shape)
            ns = min(mg.shape[1], ml.shape[1]); nt = min(mg.shape[0], ml.shape[0])
            row["dF_ls"] = _stats(mg[:nt, :ns], ml[:nt, :ns])
            fg = g.get_traces(dmd, t, signal="F0", squeeze_channels=False)[:, :, 0]
            fl = np.asarray(l.get_traces(dmd, t, signal="F0", squeeze_channels=False), dtype=float)
            if fl.ndim == 3:
                fl = fl[:, :, 0]
            if fl.shape[0] != fg.shape[0] and fl.shape[1] == fg.shape[0]:
                fl = fl.T
            row["F0"] = _stats(fg[:nt, :ns], fl[:nt, :ns])
            lg = g.get_trial_source_lines(dmd, t); ll = l.get_trial_source_lines(dmd, t)
            row["frame_lines"] = {"modern_n": int(lg.size), "legacy_n": None if ll is None else int(ll.size),
                                  "identical": bool(ll is not None and lg.size == ll.size and np.array_equal(lg, ll))}
            out["trials"].append(row)
        report["per_dmd"][f"DMD{dmd}"] = out
    g.close(); l.close()
    verdict = "equivalent" if not report["problems"] and all(
        all(r["dF_ls"].get("rel_rms_diff", 1) < 1e-6 and r["frame_lines"]["identical"] for r in d["trials"])
        for d in report["per_dmd"].values()) else "differences_found"
    report["verdict"] = verdict
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modern", default=None, help="GIAnT experiment_summary.h5 (or use --summary-xlsx/--session)")
    ap.add_argument("--legacy", default=None, help="legacy SummaryLoCo*.mat (or use --summary-xlsx/--session)")
    ap.add_argument("--summary-xlsx", default=None, help="resolve both files from the session registry")
    ap.add_argument("--session", default=None)
    ap.add_argument("--max-trials", type=int, default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    modern, legacy = a.modern, a.legacy
    if a.summary_xlsx and a.session:
        from vip_slap2_analysis.io.session_registry import VIPSessionRegistry
        reg = VIPSessionRegistry.from_excel(a.summary_xlsx)
        asset = reg.resolve_assets(a.session)
        cands = asset.get_asset("glutamate", "summary_candidates") or {}
        modern = modern or cands.get("giant_experiment_summary_h5") or asset.get_asset("glutamate", "summary_h5")
        legacy = legacy or cands.get("legacy_summary_loco_mat") or getattr(asset, "summary_mat", None)
        print(f"modern: {modern}\nlegacy: {legacy}")
    if not modern or not legacy:
        ap.error("give --modern and --legacy, or --summary-xlsx and --session")
    rep = compare(Path(modern), Path(legacy), max_trials=a.max_trials)
    txt = json.dumps(rep, indent=2, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"wrote {a.out}")
    print("verdict:", rep["verdict"])
    for p in rep["problems"]:
        print(" -", p)
    for k, d in rep["per_dmd"].items():
        for tag in ("modern", "legacy"):
            c = d[f"{tag}_coverage"]
            inv = c["invalid_trials"]
            spans = ", ".join(f"{x['trial']}:{x.get('t_start_s', float('nan')):.0f}-{x.get('t_end_s', float('nan')):.0f}s" for x in inv[:12] if 't_start_s' in x)
            print(f" {k} {tag}: {c['n_valid_trials']}/{c['n_trials']} trials valid, {c['valid_fraction_of_acquisition'] if c['valid_fraction_of_acquisition'] is None else round(c['valid_fraction_of_acquisition'], 3)} of the acquisition covered"
                  + (f"; invalid trials (trial:start-end): {spans}{' ...' if len(inv) > 12 else ''}" if inv else ""))
        for r in d["trials"][:3]:
            print(f" {k} trial {r['trial']}: dF_ls rel_rms={r['dF_ls'].get('rel_rms_diff')} r={r['dF_ls'].get('pearson_r')} frame_lines_identical={r['frame_lines']['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
