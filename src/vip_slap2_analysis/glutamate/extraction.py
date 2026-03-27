from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.glutamate.summary import GlutamateSummary
from vip_slap2_analysis.glutamate.alignment import (
    EventWindows,
    load_corrected_bonsai_csv,
    load_imaging_epochs_csv,
    extract_image_intervals,
    extract_named_intervals,
    extract_ordered_change_targets,
    filter_intervals_to_epochs,
    filter_ordered_images_to_epochs,
    collect_dmd_trial_traces,
    align_traces_to_intervals,
    build_change_locked_sequences,
    summarize_event_tensor,
)


def _load_synapse_qc_mask(asset: SessionAssets, dmd: int) -> Optional[np.ndarray]:
    if asset.qc_dir is None:
        return None
    p = Path(asset.qc_dir) / f"valid_synapses_dmd{dmd}.npy"
    if p.exists():
        return np.load(p)
    return None


def _apply_synapse_mask(x, mask):
    if mask is None:
        return x
    if isinstance(x, dict):
        return {k: _apply_synapse_mask(v, mask) for k, v in x.items()}
    return x[:, mask, :]


def process_glutamate_extraction(
    asset: SessionAssets,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    use_synapse_qc: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    metadata = metadata or {}
    windows = EventWindows(**metadata.get("prepost_sec", {})) if "prepost_sec" in metadata else EventWindows()

    if asset.summary_mat is None:
        raise FileNotFoundError("asset.summary_mat is missing")
    if asset.bonsai_event_log_csv is None:
        raise FileNotFoundError("asset.bonsai_event_log_csv is missing")
    if asset.qc_dir is None or asset.derived_dir is None:
        raise ValueError("asset.qc_dir and asset.derived_dir must be set")

    glut_dir = Path(asset.derived_dir) / "glutamate"
    glut_qc_dir = Path(asset.qc_dir) / "glutamate"
    glut_dir.mkdir(parents=True, exist_ok=True)
    glut_qc_dir.mkdir(parents=True, exist_ok=True)

    mean_npz = glut_dir / "glutamate_mean_df.npz"
    single_npz = glut_dir / "glutamate_single_trial_df.npz"
    seq_npz = glut_dir / "glutamate_sequence_df.npz"
    qc_json = glut_qc_dir / "glutamate_extraction_qc.json"

    if all(p.exists() for p in [mean_npz, single_npz, seq_npz, qc_json]) and not overwrite:
        return {"status": "exists", "mean_npz": str(mean_npz), "single_npz": str(single_npz), "seq_npz": str(seq_npz)}

    stim_df = load_corrected_bonsai_csv(asset.bonsai_event_log_csv)
    epoch_df = load_imaging_epochs_csv(Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv")

    image_times, ordered_images = extract_image_intervals(stim_df)
    change_times = extract_named_intervals(stim_df, "ChangeFlash")
    omission_times = extract_named_intervals(stim_df, "Omission")
    extract_ordered_change_targets(stim_df, ordered_images)

    image_times_f = filter_intervals_to_epochs(image_times, epoch_df, pre_time=windows.image[0], post_time=windows.image[1])
    change_times_f = filter_intervals_to_epochs(change_times, epoch_df, pre_time=windows.change[0], post_time=windows.change[1])
    omission_times_f = filter_intervals_to_epochs(omission_times, epoch_df, pre_time=windows.omission[0], post_time=windows.omission[1])
    ordered_images_f = filter_ordered_images_to_epochs(ordered_images, epoch_df, pre_time=windows.image[0], post_time=windows.image[1])

    exp = GlutamateSummary(asset.summary_mat)
    if "im_rate_hz" not in metadata:
        raise ValueError(
            "metadata must include 'im_rate_hz' because GlutamateSummary does not expose it as exp.meta."
        )
    im_rate_hz = float(metadata["im_rate_hz"])

    mean_pkg = {"metadata": {}, "timebase_sec": {}, "DMD1": {}, "DMD2": {}}
    single_pkg = {"metadata": {}, "timebase_sec": {}, "DMD1": {}, "DMD2": {}}
    seq_pkg = {"metadata": {}, "DMD1": {}, "DMD2": {}}

    qc = {
        "schema_version": "0.1.0",
        "session_id": asset.session_id,
        "summary_mat": str(asset.summary_mat),
        "bonsai_event_log_csv": str(asset.bonsai_event_log_csv),
        "use_synapse_qc": bool(use_synapse_qc),
        "windows_sec": {
            "image": windows.image,
            "change": windows.change,
            "omission": windows.omission,
        },
        "event_counts": {
            "image_total": int(sum(len(v) for v in image_times.values())),
            "image_kept": int(sum(len(v) for v in image_times_f.values())),
            "change_total": int(len(change_times)),
            "change_kept": int(len(change_times_f)),
            "omission_total": int(len(omission_times)),
            "omission_kept": int(len(omission_times_f)),
        },
        "per_dmd": {},
    }

    for dmd in (1, 2):
        traces = collect_dmd_trial_traces(exp, dmd=dmd, signal="dF", mode="ls")
        if traces.size == 0:
            qc["per_dmd"][f"DMD{dmd}"] = {"skipped": True, "reason": "no valid traces"}
            continue

        syn_mask = _load_synapse_qc_mask(asset, dmd) if use_synapse_qc else None
        n_syn_total = int(traces.shape[0])
        n_syn_kept = int(np.sum(syn_mask)) if syn_mask is not None else n_syn_total

        aligned_images = align_traces_to_intervals(
            traces, image_times_f, epoch_df,
            im_rate_hz=im_rate_hz, pre_time=windows.image[0], post_time=windows.image[1]
        )
        aligned_changes = align_traces_to_intervals(
            traces, change_times_f, epoch_df,
            im_rate_hz=im_rate_hz, pre_time=windows.change[0], post_time=windows.change[1]
        )
        aligned_omissions = align_traces_to_intervals(
            traces, omission_times_f, epoch_df,
            im_rate_hz=im_rate_hz, pre_time=windows.omission[0], post_time=windows.omission[1]
        )

        aligned_images = _apply_synapse_mask(aligned_images, syn_mask)
        aligned_changes = _apply_synapse_mask(aligned_changes, syn_mask)
        aligned_omissions = _apply_synapse_mask(aligned_omissions, syn_mask)

        mean_pkg[f"DMD{dmd}"]["image_identity"] = summarize_event_tensor(
            np.concatenate(list(aligned_images.values()), axis=0) if len(aligned_images) else np.empty((0, n_syn_kept, 0))
        )
        mean_pkg[f"DMD{dmd}"]["change"] = summarize_event_tensor(aligned_changes)
        mean_pkg[f"DMD{dmd}"]["omission"] = summarize_event_tensor(aligned_omissions)
        mean_pkg[f"DMD{dmd}"]["synapse_ids"] = np.array([f"DMD{dmd}_syn{i:04d}" for i in range(n_syn_kept)])
        mean_pkg[f"DMD{dmd}"]["valid_synapses_mask"] = syn_mask if syn_mask is not None else np.ones(n_syn_kept, dtype=bool)

        single_pkg[f"DMD{dmd}"]["image_identity"] = aligned_images
        single_pkg[f"DMD{dmd}"]["change"] = aligned_changes
        single_pkg[f"DMD{dmd}"]["omission"] = aligned_omissions
        single_pkg[f"DMD{dmd}"]["synapse_ids"] = np.array([f"DMD{dmd}_syn{i:04d}" for i in range(n_syn_kept)])

        seq_pkg[f"DMD{dmd}"]["sequence_events"] = build_change_locked_sequences(ordered_images_f)
        seq_pkg[f"DMD{dmd}"]["synapse_ids"] = np.array([f"DMD{dmd}_syn{i:04d}" for i in range(n_syn_kept)])

        qc["per_dmd"][f"DMD{dmd}"] = {
            "n_synapses_total": n_syn_total,
            "n_synapses_kept": n_syn_kept,
            "n_image_identities": int(len(aligned_images)),
            "n_change_events_kept": int(aligned_changes.shape[0]),
            "n_omission_events_kept": int(aligned_omissions.shape[0]),
        }

    base_meta = {
        "schema_version": "0.1.0",
        "session_id": asset.session_id,
        "subject_id": int(asset.subject_id),
        "summary_mat": str(asset.summary_mat),
        "im_rate_hz": im_rate_hz,
        "windows_sec": {
            "image": windows.image,
            "change": windows.change,
            "omission": windows.omission,
        },
        "use_synapse_qc": bool(use_synapse_qc),
    }

    mean_pkg["metadata"] = base_meta
    single_pkg["metadata"] = base_meta
    seq_pkg["metadata"] = base_meta

    mean_pkg["timebase_sec"] = {
        "image": np.arange(-windows.image[0], windows.image[1], 1 / im_rate_hz),
        "change": np.arange(-windows.change[0], windows.change[1], 1 / im_rate_hz),
        "omission": np.arange(-windows.omission[0], windows.omission[1], 1 / im_rate_hz),
    }
    single_pkg["timebase_sec"] = mean_pkg["timebase_sec"]

    np.savez_compressed(mean_npz, data=np.array([mean_pkg], dtype=object))
    np.savez_compressed(single_npz, data=np.array([single_pkg], dtype=object))
    np.savez_compressed(seq_npz, data=np.array([seq_pkg], dtype=object))

    with open(qc_json, "w") as f:
        json.dump(qc, f, indent=2)

    return {
        "status": "ok",
        "mean_npz": str(mean_npz),
        "single_npz": str(single_npz),
        "seq_npz": str(seq_npz),
        "qc_json": str(qc_json),
    }