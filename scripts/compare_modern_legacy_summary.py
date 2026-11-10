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
    for dmd in dmds:
        if dmd > g.n_dmds or dmd > l.n_dmds:
            continue
        out = {"modern_n_sources": g.n_synapses[dmd - 1], "legacy_n_sources": l.n_synapses[dmd - 1],
               "modern_valid_trials": g.valid_trials[dmd - 1], "legacy_valid_trials": l.valid_trials[dmd - 1], "trials": []}
        if out["modern_n_sources"] != out["legacy_n_sources"]:
            report["problems"].append(f"DMD{dmd}: source count differs ({out['modern_n_sources']} vs {out['legacy_n_sources']}); traces compared only on the common prefix, identities NOT assumed equal")
        common = sorted(set(g.valid_trials[dmd - 1]) & set(l.valid_trials[dmd - 1]))
        if set(g.valid_trials[dmd - 1]) != set(l.valid_trials[dmd - 1]):
            report["problems"].append(f"DMD{dmd}: valid trial sets differ")
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
    ap.add_argument("--modern", required=True); ap.add_argument("--legacy", required=True)
    ap.add_argument("--max-trials", type=int, default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = compare(Path(a.modern), Path(a.legacy), max_trials=a.max_trials)
    txt = json.dumps(rep, indent=2, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"wrote {a.out}")
    print("verdict:", rep["verdict"])
    for p in rep["problems"]:
        print(" -", p)
    for k, d in rep["per_dmd"].items():
        for r in d["trials"][:3]:
            print(f" {k} trial {r['trial']}: dF_ls rel_rms={r['dF_ls'].get('rel_rms_diff')} r={r['dF_ls'].get('pearson_r')} frame_lines_identical={r['frame_lines']['identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
