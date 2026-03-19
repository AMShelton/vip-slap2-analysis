from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vip_slap2_analysis.plotting.plot_utils import (
    apply_plot_style,
    finalize_and_save_figure,
    get_dmd_colors,
    style_axis,
)


def plot_synapse_qc_summary(
    qc_df: pd.DataFrame,
    save_dir: Union[str, Path],
    prefix: str = "synapse_qc_summary",
) -> None:
    """
    Multi-panel histogram summary of key synapse QC metrics.
    """
    apply_plot_style()
    dmd_colors = get_dmd_colors(palette="Sailboat")

    metrics = [
        ("finite_fraction", "Finite fraction", np.linspace(0, 1.0, 25)),
        ("trace_sigma_robust", "Robust σ", 30),
        ("trace_abs_p99", "|Trace| p99", 30),
        ("residual_snr_db", "Residual SNR (dB)", 30),
        ("quality_score", "Quality score", np.linspace(0, 1.0, 25)),
        ("trace_range_robust", "Robust range", 30),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(6.0, 13.0), sharex=False)
    axes = np.ravel(axes)

    for ax, (col, title, bins) in zip(axes, metrics):
        for dmd in sorted(qc_df["dmd"].dropna().unique()):
            dft = qc_df.loc[qc_df["dmd"] == dmd, col].astype(float).values
            dft = dft[np.isfinite(dft)]
            if dft.size == 0:
                continue

            ax.hist(
                dft,
                bins=bins,
                color=dmd_colors.get(int(dmd), "lightgray"),
                edgecolor=dmd_colors.get(int(dmd), "lightgray"),
                alpha=0.75,
                label=f"DMD{int(dmd)}",
            )

        ax.set_title(title, x=0.0)
        style_axis(ax)

    axes[-1].set_xlabel("Metric value")
    axes[2].set_ylabel("Synapses")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, loc="upper right")

    finalize_and_save_figure(fig, Path(save_dir) / prefix)


def plot_synapse_qc_relationships(
    qc_df: pd.DataFrame,
    save_dir: Union[str, Path],
    prefix: str = "synapse_qc_relationships",
) -> None:
    """
    Scatter relationships between key QC metrics.
    """
    apply_plot_style()
    dmd_colors = get_dmd_colors(palette="Sailboat")

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 13.0), sharex=False)

    pairs = [
        ("trace_abs_p99", "residual_snr_db", "|Trace| p99", "Residual SNR (dB)"),
        ("finite_fraction", "quality_score", "Finite fraction", "Quality score"),
        ("trace_sigma_robust", "residual_snr_db", "Robust σ", "Residual SNR (dB)"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        for dmd in sorted(qc_df["dmd"].dropna().unique()):
            dft = qc_df.loc[qc_df["dmd"] == dmd]
            ax.scatter(
                dft[xcol],
                dft[ycol],
                s=30,
                alpha=0.8,
                color=dmd_colors.get(int(dmd), "lightgray"),
                edgecolor="none",
                label=f"DMD{int(dmd)}",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(frameon=False, loc="best")

    finalize_and_save_figure(fig, Path(save_dir) / prefix)


def plot_synapse_qc_ranked(
    qc_df: pd.DataFrame,
    save_dir: Union[str, Path],
    prefix: str = "synapse_qc_ranked",
) -> None:
    """
    Ranked quality score plot by DMD.
    """
    apply_plot_style()
    dmd_colors = get_dmd_colors(palette="Sailboat")

    dmds = sorted(qc_df["dmd"].dropna().unique())
    fig, axes = plt.subplots(
        len(dmds),
        1,
        figsize=(6.0, max(4.0, 3.0 * len(dmds))),
        sharex=False,
    )
    if len(dmds) == 1:
        axes = [axes]

    for ax, dmd in zip(axes, dmds):
        dft = (
            qc_df.loc[qc_df["dmd"] == dmd]
            .sort_values("quality_score", ascending=False)
            .reset_index(drop=True)
        )

        x = np.arange(1, len(dft) + 1)
        y = dft["quality_score"].values.astype(float)

        ax.plot(x, y, lw=2.5, color=dmd_colors.get(int(dmd), "lightgray"))
        ax.scatter(x, y, s=18, color=dmd_colors.get(int(dmd), "lightgray"))

        ax.set_title(f"DMD{int(dmd)}", x=0.0)
        ax.set_ylabel("Quality score")
        style_axis(ax)

    axes[-1].set_xlabel("Synapse rank within DMD")

    finalize_and_save_figure(fig, Path(save_dir) / prefix)


def make_all_synapse_qc_plots(
    qc_df: pd.DataFrame,
    save_dir: Union[str, Path],
) -> None:
    """
    Generate all standard synapse QC figures.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_synapse_qc_summary(qc_df=qc_df, save_dir=save_dir)
    plot_synapse_qc_relationships(qc_df=qc_df, save_dir=save_dir)
    plot_synapse_qc_ranked(qc_df=qc_df, save_dir=save_dir)