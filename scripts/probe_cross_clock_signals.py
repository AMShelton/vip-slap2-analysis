"""Probe for physical signals recorded on BOTH clocks that could register a single-acquisition session.

Run on sessions whose offset is already identified (ground truth), e.g. 863774_2026-08-07 (k=193).
Two candidates, both independent of neural responses:

A. Stimulus-monitor light leaking into the PMTs: per-cycle mean raw intensity (SLAP2 .dat pixels,
   FPGA clock) vs the photodiode (HARP AnalogInput0). Image onsets are frame-exact, so if the leak
   exists the lag scan has a cycle-sharp peak.
B. Brain motion vs locomotion: per-cycle online motion-correction offsets from the .dat line headers
   (xOffset, yOffset, zFeedback; FPGA clock) vs wheel speed from the HARP encoder (AnalogData[1]).
   Physical, but slow (hundreds of ms), so expect a broad peak.

For each, the script reports the correlation at the true alignment and runs a *blind* lag scan over
the whole count-consistent offset range, exactly as an ambiguous session would have to be treated:
the question is whether the scan's peak lands on the true k, and how sharply. Nothing is written
except a PNG and a JSON under analysis/qc/sync/.

    python scripts/probe_cross_clock_signals.py --summary-xlsx Z:/VIP_synaptic_dynamics/VIP_SD_summary.xlsx \
        --session 863774_2026-08-07_13-29-18 852836_2026-09-14_14-52-10
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vip_slap2_analysis.common.cycle_registration import load_cycle_registration, piecewise_fpga_to_harp, detect_relation_steps  # noqa: E402
from vip_slap2_analysis.common.fpga_cross_epoch import _pulse_cycle_indices  # noqa: E402
from vip_slap2_analysis.io.harp_bin import find_register_file, read_harp_register  # noqa: E402
from vip_slap2_analysis.io.session_registry import VIPSessionRegistry  # noqa: E402
from vip_slap2_analysis.io.slap2_dat import find_dmd_dat_files, list_acquisition_prefixes, read_dat_file_header, read_dmd_acquisition_timing  # noqa: E402

LINE_HDR = struct.Struct("<5I2hQ2f")   # lineSizeBytes, magic, lineNumber, acqNumber, flags, xOff, yOff, timestamp, zOffset, zFeedback


def per_cycle_signals(paths):
    """Per-cycle mean pixel intensity and first-line motion offsets for one acquisition (all chunks)."""
    inten, xo, yo, zf = [], [], [], []
    for p in paths:
        h = read_dat_file_header(p)
        mm = np.memmap(p, dtype=np.uint8, mode="r", offset=h.firstCycleOffsetBytes)
        n, bpc, lh = h.numCycles, h.bytesPerCycle, h.lineHeaderSizeBytes
        # line layout from the first cycle (identical for every cycle of an acquisition)
        off, sizes = 0, []
        for _ in range(h.linesPerCycle):
            sz = int(np.frombuffer(mm[off:off + 4].tobytes(), "<u4")[0]); sizes.append(sz); off += sz
        if off != bpc:
            raise ValueError(f"{p}: line sizes sum to {off} but bytesPerCycle is {bpc}")
        cyc = mm[: n * bpc].reshape(n, bpc)
        tot = np.zeros(n); cnt = 0
        offs = np.cumsum([0] + sizes[:-1])
        for o, sz in zip(offs, sizes):
            if sz <= lh:
                continue
            pix = np.ascontiguousarray(cyc[:, o + lh:o + sz]).view("<i2")
            tot += pix.astype(np.float64).sum(1); cnt += pix.shape[1]
        inten.append(tot / max(cnt, 1))
        hdr = np.ascontiguousarray(cyc[:, :LINE_HDR.size])
        rec = np.frombuffer(hdr.tobytes(), dtype=np.dtype([("sz", "<u4"), ("magic", "<u4"), ("line", "<u4"), ("acq", "<u4"), ("flags", "<u4"),
                                                            ("xo", "<i2"), ("yo", "<i2"), ("ts", "<u8"), ("zo", "<f4"), ("zf", "<f4")]))
        xo.append(rec["xo"].astype(float)); yo.append(rec["yo"].astype(float)); zf.append(rec["zf"].astype(float))
    return np.concatenate(inten), np.concatenate(xo), np.concatenate(yo), np.concatenate(zf)


def detrend(x, w):
    k = np.ones(int(w)) / int(w)
    return x - np.convolve(x, k, "same")


def lag_scan(a, b, lags):
    """corr(a[i + lag], b[i]) for each lag (both on the cycle grid)."""
    out = np.full(lags.size, np.nan)
    n = min(a.size, b.size)
    for j, L in enumerate(lags):
        if L >= 0:
            x, y = a[L:n], b[: n - L]
        else:
            x, y = a[: n + L], b[-L:n]
        if x.size > 100 and np.std(x) > 0 and np.std(y) > 0:
            out[j] = np.corrcoef(x, y)[0, 1]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-xlsx", required=True)
    ap.add_argument("--session", nargs="+", required=True)
    ap.add_argument("--dmd", type=int, default=1)
    ap.add_argument("--epoch", type=int, default=1, help="source epoch to probe (default 1, the SLAP2-first one)")
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
        e = next((x for x in reg.epochs if x.source_epoch_index == args.epoch), None) if reg else None
        if e is None or e.cycle_index_offset is None or e.pulse_times_harp_s is None:
            print(f"{sid}: epoch {args.epoch} not identified; this probe needs ground truth"); continue
        prefix = list_acquisition_prefixes(dyn)[args.epoch - 1]
        paths = find_dmd_dat_files(dyn, args.dmd, prefix=prefix)
        timing = read_dmd_acquisition_timing(dyn, args.dmd, prefix=prefix)
        print(f"{sid}: reading {len(paths)} chunks of {prefix} DMD{args.dmd} ({timing.n_cycles} cycles) ...")
        inten, xo, yo, zf = per_cycle_signals(paths)
        N = inten.size
        # HARP time of every cycle through the identified registration (interpolated between pulses)
        idx, pulses = _pulse_cycle_indices(e)
        ok = idx < N
        x, y = timing.cycle_start_s[idx[ok]], pulses[ok]
        cyc_harp = piecewise_fpga_to_harp(timing.cycle_start_s, x, y, detect_relation_steps(x, y))
        period = float(np.median(np.diff(cyc_harp)))
        k_true = int(e.cycle_index_offset)
        # HARP analog: photodiode (ch0) and encoder (ch1)
        t44, vals, _info = read_harp_register(find_register_file(asset.harp_dir, 44))
        vals = np.asarray(vals)
        if vals.ndim != 2 or vals.shape[1] < 1:
            print(f"{sid}: unexpected AnalogData payload shape {vals.shape}"); continue
        order = np.argsort(t44); t44, vals = t44[order], vals[order]
        pd_c = np.interp(cyc_harp, t44, vals[:, 0].astype(float), left=np.nan, right=np.nan)
        enc = None
        if vals.shape[1] > 1:
            d_enc = np.diff(vals[:, 1].astype(float)); d_enc[d_enc > 32768] -= 65536; d_enc[d_enc < -32768] += 65536
            enc = np.r_[0.0, np.cumsum(d_enc)]            # unwrapped encoder position (counts)
        covered = np.isfinite(pd_c)
        print(f"  cycles with HARP coverage: {covered.sum()}/{N}; period {period*1e3:.2f} ms; true k = {k_true}")
        res = {"session_id": sid, "epoch": args.epoch, "true_k": k_true, "n_cycles": N, "period_s": period}

        # --- A: stimulus light leak ---------------------------------------------------------------
        # only the covered part is comparable; emulate the ambiguous case: pulses observed for cycles k..N-1,
        # photodiode known on the pulse grid; scan which cycle offset k' the photodiode series starts at
        pd_obs = pd_c[k_true:]                         # photodiode on the pulse grid: all an ambiguous session has
        pd_obs = pd_obs[: int(np.flatnonzero(np.isfinite(pd_obs)).max()) + 1]
        pd_obs = np.where(np.isfinite(pd_obs), pd_obs, np.nanmean(pd_obs))
        w = max(5, int(round(1.0 / period)))           # 1 s detrend
        a = detrend(inten, w); b = detrend(pd_obs, w)
        lags = np.arange(0, N - pd_obs.size + 1)       # candidate k' (count-consistent range)
        cA = lag_scan(a, b, lags)
        jA = int(np.nanargmax(np.abs(cA)))
        zA = (abs(cA[jA]) - np.nanmean(np.abs(cA))) / np.nanstd(np.abs(cA))
        print(f"  A) intensity vs photodiode: corr at true k = {cA[lags == k_true][0] if (lags == k_true).any() else float('nan'):+.4f}; "
              f"blind scan peak at k' = {lags[jA]} (|r| = {abs(cA[jA]):.4f}, z = {zA:.1f}) -> {'HIT' if lags[jA] == k_true else 'miss by %d cycles' % (lags[jA] - k_true)}")
        # event-triggered intensity at photodiode transitions (true alignment)
        d = np.diff(pd_obs); ev = np.flatnonzero(np.abs(d) > 4 * np.nanstd(d))
        eta = None
        if ev.size:
            win = np.arange(-10, 11)
            segs = [a[k_true + i + win] for i in ev if k_true + i + win[0] >= 0 and k_true + i + win[-1] < N]
            eta = np.mean(segs, axis=0) if segs else None
            if eta is not None:
                print(f"     {ev.size} photodiode transitions; intensity ETA peak |z| = {np.max(np.abs(eta)) / (np.std(a) / np.sqrt(len(segs))):.1f}")
        res["A"] = {"corr_true": float(cA[lags == k_true][0]) if (lags == k_true).any() else None, "peak_k": int(lags[jA]),
                    "peak_abs_corr": float(abs(cA[jA])), "peak_z": float(zA), "hit": bool(lags[jA] == k_true), "n_transitions": int(ev.size)}

        # --- B: motion vs locomotion ----------------------------------------------------------------
        resB = None
        if enc is not None:
            ws = max(5, int(round(0.25 / period)))     # 250 ms smoothing
            pos_c = np.interp(cyc_harp, t44, enc, left=np.nan, right=np.nan)
            speed = np.abs(np.gradient(np.where(np.isfinite(pos_c), pos_c, np.nanmean(pos_c))))
            speed_obs = speed[k_true:k_true + pd_obs.size]
            motion = np.abs(np.gradient(xo)) + np.abs(np.gradient(yo))
            ma = np.convolve(motion, np.ones(ws) / ws, "same"); sb = np.convolve(speed_obs, np.ones(ws) / ws, "same")
            ma = detrend(ma, 20 * ws); sb = detrend(sb, 20 * ws)
            cB = lag_scan(ma, sb, lags)
            jB = int(np.nanargmax(cB))
            # peak width: lags within 50% of peak
            half = np.flatnonzero(cB > 0.5 * cB[jB])
            width = int(half.max() - half.min() + 1) if half.size else -1
            zB = (cB[jB] - np.nanmean(cB)) / np.nanstd(cB)
            print(f"  B) motion vs wheel speed: corr at true k = {cB[lags == k_true][0] if (lags == k_true).any() else float('nan'):+.4f}; "
                  f"blind scan peak at k' = {lags[jB]} (r = {cB[jB]:.4f}, z = {zB:.1f}, half-max width {width} cycles = {width*period:.2f} s) "
                  f"-> {'HIT' if lags[jB] == k_true else 'off by %d cycles' % (lags[jB] - k_true)}")
            resB = {"corr_true": float(cB[lags == k_true][0]) if (lags == k_true).any() else None, "peak_k": int(lags[jB]), "peak_corr": float(cB[jB]),
                    "peak_z": float(zB), "half_max_width_cycles": width, "hit": bool(lags[jB] == k_true)}
        res["B"] = resB

        fig, axes = plt.subplots(3, 1, figsize=(11, 8))
        axes[0].plot(lags, cA, lw=0.6); axes[0].axvline(k_true, color="r", ls="--", label=f"true k={k_true}"); axes[0].set_title("A: per-cycle intensity vs photodiode, blind offset scan"); axes[0].legend(); axes[0].set_ylabel("r")
        if resB is not None:
            axes[1].plot(lags, cB, lw=0.6); axes[1].axvline(k_true, color="r", ls="--"); axes[1].set_title("B: motion-correction offsets vs wheel speed, blind offset scan"); axes[1].set_ylabel("r")
        if eta is not None:
            axes[2].plot(np.arange(-10, 11) * period * 1e3, eta); axes[2].axvline(0, color="r", ls="--"); axes[2].set_title("A: intensity around photodiode transitions at the true alignment"); axes[2].set_xlabel("ms")
        axes[1].set_xlabel("candidate k (cycles)")
        fig.suptitle(f"{sid} DMD{args.dmd} epoch {args.epoch}: cross-clock signal probe", fontsize=10); fig.tight_layout()
        out_dir = Path(asset.qc_dir) / "sync"; out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"cross_clock_probe_DMD{args.dmd}_epoch{args.epoch}.png", dpi=130); plt.close(fig)
        (out_dir / f"cross_clock_probe_DMD{args.dmd}_epoch{args.epoch}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"  wrote {out_dir / f'cross_clock_probe_DMD{args.dmd}_epoch{args.epoch}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
