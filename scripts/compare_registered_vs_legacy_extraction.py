"""Before/after check on a real session: extract voltage events with the historical
first-pulse anchoring and with the DI3 cycle registration, into side folders, and plot
the stimulus-locked averages of both.

Nothing under the session's standard products is touched: outputs go to
``analysis/derived/voltage/sync_compare/{legacy,registered}`` (+ matching qc folders).
This is a *consistency* check — a visual response locked to the stimulus under one
timebase and not the other is strong evidence for that timebase — not the basis of the
registration itself, which rests on DI3 coverage and the raw .dat cycle count.

    python scripts\\compare_registered_vs_legacy_extraction.py --summary-xlsx "Z:\\VIP_synaptic_dynamics\\VIP_SD_summary.xlsx" ^
        --session 863774_2026-08-07_13-29-18 --out-dir "<project folder>\\sync_audit_2026-11-04\\figures"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402
from vip_slap2_analysis.voltage.extraction import process_voltage_extraction  # noqa: E402


def _load_mean(npz: Path):
    return np.load(npz, allow_pickle=True)["data"][0]


def _event_snr(summary: dict, t: np.ndarray, *, pre_end: float = -0.05, resp: tuple = (0.03, 0.25)) -> float:
    """Population-mean |response| in a post-onset window relative to pre-onset SD (unitless)."""
    m = np.asarray(summary["mean"], dtype=float)           # (rois, time)
    if m.size == 0 or not np.isfinite(m).any():
        return float("nan")
    pop = np.nanmean(m, axis=0)
    n = min(pop.size, t.size); pop, t = pop[:n], t[:n]
    base = pop[t < pre_end]
    win = pop[(t >= resp[0]) & (t <= resp[1])]
    sd = np.nanstd(base)
    return float(np.nanmax(np.abs(win - np.nanmean(base))) / sd) if sd > 0 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trace-signal", default="dff_robust_f0")
    ap.add_argument("--dmd", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--max-roi-plot", type=int, default=12)
    a = ap.parse_args(argv)

    reg = VIPSessionRegistry.from_excel(a.summary_xlsx)
    asset = reg.resolve_assets(a.session)
    asset.ensure_dirs()
    runs = {}
    for name, cr, require in (("legacy", None, False), ("registered", "auto", True)):
        print(f"[{name}] extracting ...", flush=True)
        runs[name] = process_voltage_extraction(
            asset, trace_signal=a.trace_signal, overwrite=True, write_single_trials=False, write_sequence=False,
            write_session_traces=False, use_roi_qc=False, cycle_registration=cr,
            require_identified_cycle_alignment=require, output_subdir=f"sync_compare/{name}",
        )
        print(f"[{name}] {runs[name]['status']} -> {runs[name]['mean_npz']}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    leg = _load_mean(Path(runs["legacy"]["mean_npz"])); regd = _load_mean(Path(runs["registered"]["mean_npz"]))
    report = {"session": a.session, "trace_signal": a.trace_signal, "per_dmd": {}}
    for dmd in a.dmd:
        key = f"DMD{dmd}"
        if key not in leg or not leg[key] or key not in regd or not regd[key]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
        rep = {}
        for ax, fam in zip(axes, ("image_identity", "change", "omission")):
            t = np.asarray(leg["timebase_sec"]["image" if fam == "image_identity" else fam], dtype=float)
            if fam == "image_identity":
                # pool all image identities (weighted by event count)
                def pooled(pkg):
                    acc, n = None, 0
                    for img, s in pkg[key][fam].items():
                        k = int(s["n_events"])
                        if k == 0:
                            continue
                        m = np.asarray(s["mean"], dtype=float)
                        acc = m * k if acc is None else acc + m * k; n += k
                    return {"mean": acc / n if n else np.full((1, t.size), np.nan), "n_events": n}
                sl, sr = pooled(leg), pooled(regd)
            else:
                sl, sr = leg[key][fam], regd[key][fam]
            for pkg, label, color in ((sl, "first-pulse anchored (historical)", "0.5"), (sr, "DI3-registered", "tab:blue")):
                pop = np.nanmean(np.asarray(pkg["mean"], dtype=float), axis=0)
                n = min(pop.size, t.size)  # stored window may be one sample shorter than the nominal time vector
                ax.plot(t[:n], pop[:n], color=color, label=f"{label} (n={int(pkg['n_events'])})")
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.set_title(f"{key} {fam}", fontsize=10); ax.set_xlabel("time from onset (s)")
            rep[fam] = {"legacy_snr": _event_snr(sl, t), "registered_snr": _event_snr(sr, t),
                        "n_events_legacy": int(sl["n_events"]), "n_events_registered": int(sr["n_events"])}
        axes[0].set_ylabel(a.trace_signal); axes[0].legend(fontsize=7)
        fig.suptitle(f"{a.session} {key}: population mean, historical vs DI3-registered timebase", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"{a.session}_{key}_before_after.png", dpi=150); plt.close(fig)
        report["per_dmd"][key] = rep
    report["registered_qc_json"] = runs["registered"]["qc_json"]; report["legacy_qc_json"] = runs["legacy"]["qc_json"]
    (out / f"{a.session}_before_after.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["per_dmd"], indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
