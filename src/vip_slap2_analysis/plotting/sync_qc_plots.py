"""QC figures for DI3 <-> SLAP2 cycle registration.

All functions take already-computed objects (no file IO, no signal processing)
and return a matplotlib ``Figure``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from vip_slap2_analysis.common.cycle_registration import CycleRegistration


def plot_registration_coverage(reg: CycleRegistration, *, title: Optional[str] = None):
    """HARP recording window, DI3 pulse-train epochs and source-cycle extent per epoch.

    Source epochs are drawn at the HARP times implied by the registration (measured
    cycle times where pulses exist, extrapolated otherwise), so a missing head or
    tail shows as source extent outside the HARP recording window.
    """
    import matplotlib.pyplot as plt

    n_ep = max(1, len(reg.epochs))
    fig, ax = plt.subplots(figsize=(11, 2.2 + 0.9 * n_ep))
    t0 = float(reg.harp_record_start_s or 0.0)
    rs, re_ = reg.harp_record_start_s - t0, reg.harp_record_end_s - t0
    ax.axvspan(rs, re_, color="0.92", label="HARP recording", zorder=0)
    y = 0.0
    for e in reg.epochs:
        p = np.asarray(e.pulse_times_harp_s, dtype=float) - t0 if e.pulse_times_harp_s is not None else np.array([])
        if p.size:
            ax.plot([p[0], p[-1]], [y, y], color="tab:blue", lw=8, solid_capstyle="butt", label="DI3 pulses observed" if y == 0 else None)
        if e.cycle_index_offset is not None and p.size:
            ct, lab = e.cycle_times_harp_s()
            ct = ct - t0
            ax.plot([ct[0], ct[-1]], [y + 0.3, y + 0.3], color="tab:orange", lw=8, solid_capstyle="butt",
                    label="source cycles (registered)" if y == 0 else None)
            for code in (1, 2):
                if np.any(lab == code):
                    seg = ct[lab == code]
                    ax.plot([seg[0], seg[-1]], [y + 0.3, y + 0.3], color="tab:red", lw=8, solid_capstyle="butt",
                            label="source cycles without HARP coverage" if (y == 0 and code == 1) else None)
            txt = f"epoch {e.source_epoch_index}: {e.status}, anchor={e.anchor}, offset={e.cycle_index_offset}"
        else:
            lo, hi = e.candidate_offset_range
            txt = f"epoch {e.source_epoch_index}: {e.status}, anchor={e.anchor}, offset in [{lo}, {hi}]"
        ax.text(0.01, y + 0.55, txt + f"   (N={e.n_source_cycles} cycles, {e.n_observed_pulses} pulses, "
                f"lead={e.lead_s:.2f} s, tail={e.tail_s:.2f} s)", fontsize=8, transform=ax.get_yaxis_transform())
        y += 1.0
    ax.set_ylim(-0.4, n_ep - 0.1)
    ax.set_yticks([])
    ax.set_xlabel("time since HARP recording start (s)")
    ppm = reg.period_error_ppm if reg.period_error_ppm is not None else float("nan")
    ax.set_title(title or f"{reg.session_id or ''} DMD{reg.dmd}: {reg.status} ({reg.acquisition_start_order}; "
                 f"{reg.relationship}; HARP/source rate {ppm:+.1f} ppm)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def plot_pulse_residuals(reg: CycleRegistration, source_cycle_start_s: Optional[np.ndarray] = None):
    """Per-pulse residual of DI3 rise times vs the registered linear cycle model.

    When FPGA cycle-start times (raw ``.dat`` headers) are given, the residual is
    taken against the affine fit DI3 ~ FPGA, which isolates HARP timestamp
    quantization from source-side timing.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    for e in reg.epochs:
        if e.pulse_times_harp_s is None or e.cycle_index_offset is None:
            continue
        p = np.asarray(e.pulse_times_harp_s, dtype=float)
        if source_cycle_start_s is not None and source_cycle_start_s.size >= e.cycle_index_offset + p.size:
            x = source_cycle_start_s[e.cycle_index_offset:e.cycle_index_offset + p.size]
            a, b = np.polyfit(x, p, 1)
            r = (p - (a * x + b)) * 1e6
            lab = f"epoch {e.source_epoch_index}: vs FPGA (slope {(a - 1) * 1e6:+.2f} ppm)"
        else:
            k = np.arange(p.size)
            a, b = np.polyfit(k, p, 1)
            r = (p - (a * k + b)) * 1e6
            lab = f"epoch {e.source_epoch_index}: vs linear model"
        axes[0].plot(p - p[0], r, ".", ms=2, label=lab)
        axes[1].hist(r, bins=40, orientation="horizontal", alpha=0.7)
    axes[0].set_xlabel("time since first pulse (s)"); axes[0].set_ylabel("residual (us)")
    axes[0].axhline(0, color="k", lw=0.5); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("count")
    fig.suptitle(f"{reg.session_id or ''} DMD{reg.dmd}: DI3 rise-time residuals (HARP tick = 32 us)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_truncation_benchmark(rows: Sequence[Dict[str, Any]]):
    """Bar chart of candidate-offset count per truncation scenario (identifiability)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.4))
    names = [r["scenario"] for r in rows]
    counts = [int(r["candidate_count"]) for r in rows]
    colors = ["tab:green" if r["status"].startswith("identified") else ("tab:red" if r["status"] == "inconsistent" else "tab:orange") for r in rows]
    ax.bar(range(len(rows)), counts, color=colors)
    for i, r in enumerate(rows):
        ax.text(i, counts[i] + 0.5, f"{r['status']}\noffset={r.get('offset')}", ha="center", fontsize=7)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("candidate integer offsets"); ax.set_yscale("symlog")
    ax.set_title("Integer cycle-offset identifiability vs. DI3 coverage (2026-10-27 reference pulses)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_before_after_event_alignment(
    *,
    t_rel: np.ndarray,
    legacy_mean: np.ndarray,
    registered_mean: np.ndarray,
    truth_onset_rel_s: float = 0.0,
    title: str = "",
    ylabel: str = "signal",
):
    """Stimulus-locked averages under the historical and registered timebases."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t_rel, legacy_mean, color="0.5", label="first-pulse anchored (historical)")
    ax.plot(t_rel, registered_mean, color="tab:blue", label="DI3-registered")
    ax.axvline(truth_onset_rel_s, color="k", ls="--", lw=0.8, label="true onset")
    ax.set_xlabel("time from stimulus onset (s)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
