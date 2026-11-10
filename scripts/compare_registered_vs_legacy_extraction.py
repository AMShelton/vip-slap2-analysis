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


def per_epoch_comparison(asset, runs, dmds, out: Path, session: str, trace_signal: str) -> dict:
    """Image-locked population averages per acquisition epoch, historical vs registered.

    Events are assigned to behavior epochs by onset time (``qc/behavior/imaging_epochs.csv``);
    population means are accumulated per event from the single-trial H5 so memory stays flat.
    """
    import h5py
    import pandas as pd
    from vip_slap2_analysis.common.epoch_alignment import load_epoch_dataframe_from_asset

    epochs = load_epoch_dataframe_from_asset(asset)
    starts = epochs["start_time"].to_numpy(float); ends = epochs["end_time"].to_numpy(float)
    ids = epochs["epoch_index"].to_numpy(int)
    report = {}
    for dmd in dmds:
        key = f"DMD{dmd}"
        acc = {}
        t_img = None
        for name in ("legacy", "registered"):
            path = runs[name].get("single_h5")
            if not path or not Path(path).exists():
                continue
            with h5py.File(path, "r") as f:
                if key not in f or "image_identity" not in f[key]:
                    continue
                t_img = np.asarray(f["timebase_sec/image"][()], dtype=float)
                grp = f[key]["image_identity"]
                for img in grp.keys():
                    g = grp[img]
                    if "traces" not in g or "onsets_sec" not in g:
                        continue
                    onsets = np.asarray(g["onsets_sec"][()], dtype=float)
                    ds = g["traces"]
                    for i0 in range(0, onsets.size, 64):
                        block = np.asarray(ds[i0:i0 + 64], dtype=float)        # (events, rois, time)
                        pop = np.nanmean(block, axis=1)                          # (events, time)
                        for j, on in enumerate(onsets[i0:i0 + 64]):
                            hit = np.flatnonzero((on >= starts) & (on <= ends))
                            ep = int(ids[hit[0]]) if hit.size else 0
                            d = acc.setdefault((name, ep), {"sum": np.zeros(pop.shape[1]), "n": 0})
                            d["sum"] += np.nan_to_num(pop[j]); d["n"] += 1
        if t_img is None:
            continue
        ep_ids = sorted({ep for (_, ep) in acc.keys() if ep > 0})
        if not ep_ids:
            continue
        fig, axes = plt.subplots(1, len(ep_ids), figsize=(3.2 * len(ep_ids), 3.4), sharey=True, squeeze=False)
        rep = {}
        for ax, ep in zip(axes[0], ep_ids):
            r = {}
            for name, color in (("legacy", "0.5"), ("registered", "tab:blue")):
                d = acc.get((name, ep))
                if not d or d["n"] == 0:
                    continue
                m = d["sum"] / d["n"]; n = min(m.size, t_img.size)
                ax.plot(t_img[:n], m[:n], color=color, lw=1.0, label=f"{name} (n={d['n']})")
                r[f"{name}_snr"] = _event_snr({"mean": m[None, :]}, t_img); r[f"{name}_n"] = int(d["n"])
            ax.axvline(0, color="k", ls="--", lw=0.8); ax.set_title(f"epoch {ep}", fontsize=10); ax.set_xlabel("time from image onset (s)")
            ax.legend(fontsize=7)
            rep[str(ep)] = r
        axes[0][0].set_ylabel(trace_signal)
        fig.suptitle(f"{session} {key}: image-locked population mean per acquisition epoch (historical grey, registered blue)", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"{session}_{key}_before_after_per_epoch.png", dpi=150); plt.close(fig)
        report[key] = rep
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trace-signal", default="dff_robust_f0")
    ap.add_argument("--dmd", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--max-roi-plot", type=int, default=12)
    ap.add_argument("--per-epoch", action="store_true",
                    help="also write single-trial H5s and compare image-locked averages epoch by epoch "
                         "(the registration differs from the historical anchoring only in epochs with a nonzero offset)")
    a = ap.parse_args(argv)

    reg = VIPSessionRegistry.from_excel(a.summary_xlsx)
    asset = reg.resolve_assets(a.session)
    asset.ensure_dirs()
    runs = {}
    for name, cr, require in (("legacy", None, False), ("registered", "auto", True)):
        print(f"[{name}] extracting ...", flush=True)
        runs[name] = process_voltage_extraction(
            asset, trace_signal=a.trace_signal, overwrite=True, write_single_trials=bool(a.per_epoch), write_sequence=False,
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
    if a.per_epoch:
        report["per_epoch"] = per_epoch_comparison(asset, runs, a.dmd, out, a.session, a.trace_signal)
    report["registered_qc_json"] = runs["registered"]["qc_json"]; report["legacy_qc_json"] = runs["legacy"]["qc_json"]
    (out / f"{a.session}_before_after.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["per_dmd"], indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
