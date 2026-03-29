from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.glutamate.alignment import (
    EventWindows,
    align_traces_to_session_intervals,
    build_change_locked_sequences,
    extract_change_intervals,
    extract_image_intervals,
    extract_omission_intervals,
    extract_ordered_change_targets,
    filter_intervals_to_epochs,
    filter_ordered_images_to_epochs,
    load_corrected_bonsai_csv,
    load_imaging_epochs_csv,
    summarize_event_tensor,
    tolerant_summary_ragged,
)
from vip_slap2_analysis.glutamate.summary import GlutamateSummary


def _resolve_im_rate_hz(asset: SessionAssets, metadata: Dict[str, Any]) -> float:
    if "im_rate_hz" in metadata:
        return float(metadata["im_rate_hz"])
    if "im_rate_Hz" in metadata:
        return float(metadata["im_rate_Hz"])
    if asset.metadata:
        if "im_rate_hz" in asset.metadata:
            return float(asset.metadata["im_rate_hz"])
        if "im_rate_Hz" in asset.metadata:
            return float(asset.metadata["im_rate_Hz"])
    raise ValueError("Could not resolve imaging rate. Provide metadata['im_rate_hz'].")


def _resolve_indicator2(asset: SessionAssets, metadata: Dict[str, Any]) -> Optional[str]:
    if "indicator2" in metadata:
        return metadata["indicator2"]
    if asset.metadata and "indicator2" in asset.metadata:
        return asset.metadata["indicator2"]
    return None


def _apply_roi_mask_to_array(x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return x
    if x.ndim == 3:
        return x[:, mask, :]
    if x.ndim == 2:
        return x[mask, :]
    raise ValueError(f"Unsupported array shape for ROI masking: {x.shape}")


def _time_vectors(windows: EventWindows, im_rate_hz: float) -> Dict[str, np.ndarray]:
    return {
        "image": np.arange(-windows.image[0], windows.image[1], 1.0 / im_rate_hz),
        "change": np.arange(-windows.change[0], windows.change[1], 1.0 / im_rate_hz),
        "omission": np.arange(-windows.omission[0], windows.omission[1], 1.0 / im_rate_hz),
    }


def _empty_event_array(n_rois: int, n_time: int) -> np.ndarray:
    return np.full((0, n_rois, n_time), np.nan, dtype=float)


def _stack_snippets(snippets: List[np.ndarray], n_rois: int, n_time: int) -> np.ndarray:
    if len(snippets) == 0:
        return _empty_event_array(n_rois, n_time)
    return np.stack(snippets, axis=0)


def _build_sequence_output(
    seq_events: Dict[str, Dict[str, Any]],
    ordered_snippets: Dict[int, np.ndarray],
    *,
    n_rois_kept: int,
    n_time: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for image_name, groups in seq_events.items():
        pre_arrays: List[np.ndarray] = []
        rep_arrays: List[np.ndarray] = []
        term_arrays: List[np.ndarray] = []
        seq_lengths: List[int] = []

        for pre_evts, rep_evts, term_evt in zip(groups["prechange"], groups["repeated"], groups["terminal"]):
            pre_snips = [ordered_snippets.get(evt.event_idx) for evt in pre_evts]
            if all(s is not None for s in pre_snips):
                pre_arrays.append(np.stack(pre_snips, axis=0))

            rep_snips = [ordered_snippets.get(evt.event_idx) for evt in rep_evts]
            rep_snips = [s for s in rep_snips if s is not None]
            if len(rep_snips) > 0:
                rep_arr = np.stack(rep_snips, axis=0)
                rep_arrays.append(rep_arr)
                seq_lengths.append(rep_arr.shape[0])

            t = ordered_snippets.get(term_evt.event_idx)
            if t is not None:
                term_arrays.append(t)

        pre_summary = tolerant_summary_ragged(pre_arrays)
        rep_summary = tolerant_summary_ragged(rep_arrays)
        term_stack = _stack_snippets(term_arrays, n_rois_kept, n_time)
        term_summary = summarize_event_tensor(term_stack)

        out[image_name] = {
            "prechange": {
                **pre_summary,
                "n_sequences": np.array(len(pre_arrays), dtype=int),
                "positions": np.array([-2, -1], dtype=int),
            },
            "repeated": {
                **rep_summary,
                "n_sequences": np.array(len(rep_arrays), dtype=int),
                "sequence_lengths": np.asarray(seq_lengths, dtype=int),
                "positions": np.arange(rep_summary["mean"].shape[0], dtype=int),
            },
            "terminal": {
                **term_summary,
                "n_sequences": np.array(len(term_arrays), dtype=int),
                "position": np.array([999], dtype=int),
            },
        }

    return out


def _reconstruct_ca_session_traces(
    exp: GlutamateSummary,
    dmd: int,
    *,
    im_rate_hz: float,
    epoch_start_sec: float,
    motion_correct: bool = False,
) -> Dict[str, Any]:
    """
    Reconstruct session-wide soma Ca traces by concatenating all trials in order.
    Invalid trials are represented as NaN blocks.
    """
    ca = exp.get_processed_soma_ca_all_trials(
        dmd=dmd,
        motion_correct=motion_correct,
    )

    if ca is None:
        return {
            "traces": np.empty((0, 0), dtype=float),
            "trial_valid_mask": np.zeros((0,), dtype=bool),
            "trial_lengths_samples": np.zeros((0,), dtype=int),
            "session_start_sec": float(epoch_start_sec),
            "timebase_sec": np.empty((0,), dtype=float),
            "reconstructed_duration_sec": 0.0,
        }

    # expected shape: (n_rois, n_trials, n_samples)
    if ca.ndim != 3:
        raise ValueError(f"Expected Ca array shape (n_rois, n_trials, n_samples), got {ca.shape}")

    n_rois, n_trials, n_samples = ca.shape
    if n_rois == 0 or n_trials == 0 or n_samples == 0:
        return {
            "traces": np.empty((0, 0), dtype=float),
            "trial_valid_mask": np.zeros((n_trials,), dtype=bool),
            "trial_lengths_samples": np.full((n_trials,), n_samples, dtype=int),
            "session_start_sec": float(epoch_start_sec),
            "timebase_sec": np.empty((0,), dtype=float),
            "reconstructed_duration_sec": 0.0,
        }

    trial_lengths = np.full((n_trials,), n_samples, dtype=int)
    total_samples = int(np.sum(trial_lengths))
    traces = np.full((n_rois, total_samples), np.nan, dtype=float)
    trial_valid_mask = np.zeros((n_trials,), dtype=bool)

    pos = 0
    for t in range(n_trials):
        block = ca[:, t, :]
        traces[:, pos:pos + n_samples] = block
        trial_valid_mask[t] = np.isfinite(block).any()
        pos += n_samples

    timebase_sec = epoch_start_sec + np.arange(total_samples, dtype=float) / float(im_rate_hz)

    return {
        "traces": traces,
        "trial_valid_mask": trial_valid_mask,
        "trial_lengths_samples": trial_lengths,
        "session_start_sec": float(epoch_start_sec),
        "timebase_sec": timebase_sec,
        "reconstructed_duration_sec": float(total_samples / im_rate_hz),
    }


def process_calcium_extraction(
    asset: SessionAssets,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    use_soma_qc: bool = True,
    motion_correct: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    metadata = metadata or {}
    pp = metadata.get("prepost_sec", {})
    windows = EventWindows(
        image=tuple(pp.get("image", pp.get("image_identity", (0.25, 0.50)))),
        change=tuple(pp.get("change", (1.00, 0.75))),
        omission=tuple(pp.get("omission", (1.00, 1.50))),
    )
    im_rate_hz = _resolve_im_rate_hz(asset, metadata)
    indicator2 = _resolve_indicator2(asset, metadata)

    if asset.summary_mat is None:
        raise FileNotFoundError("asset.summary_mat is missing")
    if asset.bonsai_event_log_csv is None:
        raise FileNotFoundError("asset.bonsai_event_log_csv is missing")
    if asset.qc_dir is None or asset.derived_dir is None:
        raise ValueError("asset.qc_dir and asset.derived_dir must be set")

    cal_dir = Path(asset.derived_dir) / "calcium"
    cal_qc_dir = Path(asset.qc_dir) / "calcium"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_qc_dir.mkdir(parents=True, exist_ok=True)

    mean_npz = cal_dir / "calcium_mean_dff.npz"
    single_npz = cal_dir / "calcium_single_trial_dff.npz"
    seq_npz = cal_dir / "calcium_sequence_dff.npz"
    meta_json = cal_qc_dir / "calcium_extraction_meta.json"

    if all(p.exists() for p in [mean_npz, single_npz, seq_npz, meta_json]) and not overwrite:
        return {
            "status": "exists",
            "mean_npz": str(mean_npz),
            "single_npz": str(single_npz),
            "seq_npz": str(seq_npz),
            "meta_json": str(meta_json),
        }

    stim_df = load_corrected_bonsai_csv(asset.bonsai_event_log_csv)
    epoch_df = load_imaging_epochs_csv(Path(asset.qc_dir) / "behavior" / "imaging_epochs.csv")
    epoch_start_sec = float(epoch_df.iloc[0]["start_time"])
    epoch_end_sec = float(epoch_df.iloc[-1]["end_time"])
    epoch_duration_sec = float(epoch_end_sec - epoch_start_sec)

    image_times, ordered_images = extract_image_intervals(stim_df)
    change_times = extract_change_intervals(stim_df)
    omission_times = extract_omission_intervals(stim_df)
    extract_ordered_change_targets(stim_df, ordered_images)

    image_times_f = filter_intervals_to_epochs(image_times, epoch_df, pre_time=windows.image[0], post_time=windows.image[1])
    change_times_f = filter_intervals_to_epochs(change_times, epoch_df, pre_time=windows.change[0], post_time=windows.change[1])
    omission_times_f = filter_intervals_to_epochs(omission_times, epoch_df, pre_time=windows.omission[0], post_time=windows.omission[1])
    ordered_images_f = filter_ordered_images_to_epochs(ordered_images, epoch_df, pre_time=windows.image[0], post_time=windows.image[1])

    exp = GlutamateSummary(asset.summary_mat)
    tvecs = _time_vectors(windows, im_rate_hz)

    base_meta = {
        "schema_version": "0.1.0",
        "session_id": asset.session_id,
        "subject_id": int(asset.subject_id),
        "summary_mat": str(asset.summary_mat),
        "bonsai_event_log_csv": str(asset.bonsai_event_log_csv),
        "indicator2": indicator2,
        "im_rate_hz": float(im_rate_hz),
        "windows_sec": {
            "image": tuple(float(x) for x in windows.image),
            "change": tuple(float(x) for x in windows.change),
            "omission": tuple(float(x) for x in windows.omission),
        },
        "use_soma_qc": bool(use_soma_qc),
        "motion_correct": bool(motion_correct),
        "epoch_start_sec": epoch_start_sec,
        "epoch_end_sec": epoch_end_sec,
    }

    mean_pkg: Dict[str, Any] = {"metadata": base_meta, "timebase_sec": tvecs, "DMD1": {}, "DMD2": {}}
    single_pkg: Dict[str, Any] = {"metadata": base_meta, "timebase_sec": tvecs, "DMD1": {}, "DMD2": {}}
    seq_pkg: Dict[str, Any] = {"metadata": base_meta, "timebase_sec": {"image": tvecs["image"]}, "DMD1": {}, "DMD2": {}}

    meta_out: Dict[str, Any] = {
        "schema_version": "0.1.0",
        "session_id": asset.session_id,
        "indicator2": indicator2,
        "motion_correct": bool(motion_correct),
        "use_soma_qc": bool(use_soma_qc),
        "windows_sec": base_meta["windows_sec"],
        "event_counts": {
            "image_total": int(sum(len(v) for v in image_times.values())),
            "image_after_epoch_filter": int(sum(len(v) for v in image_times_f.values())),
            "change_total": int(len(change_times)),
            "change_after_epoch_filter": int(len(change_times_f)),
            "omission_total": int(len(omission_times)),
            "omission_after_epoch_filter": int(len(omission_times_f)),
            "n_unique_image_ids_total": int(len(image_times)),
            "n_unique_image_ids_after_epoch_filter": int(len(image_times_f)),
        },
        "epoch_duration_sec": epoch_duration_sec,
        "per_dmd": {},
    }

    seq_events = build_change_locked_sequences(ordered_images_f)

    for dmd in (1, 2):
        bundle = _reconstruct_ca_session_traces(
            exp,
            dmd=dmd,
            im_rate_hz=im_rate_hz,
            epoch_start_sec=epoch_start_sec,
            motion_correct=motion_correct,
        )
        if bundle["traces"].size == 0:
            meta_out["per_dmd"][f"DMD{dmd}"] = {"skipped": True, "reason": "no calcium traces"}
            continue

        roi_mask = None
        if use_soma_qc:
            p = cal_qc_dir / f"valid_soma_rois_dmd{dmd}.npy"
            if p.exists():
                roi_mask = np.load(p)

        n_rois_total = int(bundle["traces"].shape[0])
        if roi_mask is None:
            roi_mask = np.ones((n_rois_total,), dtype=bool)
        n_rois_kept = int(np.sum(roi_mask))
        roi_ids = np.array([f"DMD{dmd}_roi{i:04d}" for i in range(n_rois_total)])[roi_mask]

        aligned_images = align_traces_to_session_intervals(
            bundle,
            image_times_f,
            im_rate_hz=im_rate_hz,
            pre_time=windows.image[0],
            post_time=windows.image[1],
        )
        aligned_changes = align_traces_to_session_intervals(
            bundle,
            change_times_f,
            im_rate_hz=im_rate_hz,
            pre_time=windows.change[0],
            post_time=windows.change[1],
        )
        aligned_omissions = align_traces_to_session_intervals(
            bundle,
            omission_times_f,
            im_rate_hz=im_rate_hz,
            pre_time=windows.omission[0],
            post_time=windows.omission[1],
        )

        aligned_images = {k: _apply_roi_mask_to_array(v, roi_mask) for k, v in aligned_images.items()}
        aligned_changes = _apply_roi_mask_to_array(aligned_changes, roi_mask)
        aligned_omissions = _apply_roi_mask_to_array(aligned_omissions, roi_mask)

        ordered_snippets: Dict[int, np.ndarray] = {}
        n_img_time = len(tvecs["image"])
        for evt in ordered_images_f:
            center = int(round((float(evt.onset) - bundle["session_start_sec"]) * im_rate_hz))
            n_pre = int(round(windows.image[0] * im_rate_hz))
            start = center - n_pre
            stop = start + n_img_time
            if start < 0 or stop > bundle["traces"].shape[1]:
                continue
            ordered_snippets[evt.event_idx] = _apply_roi_mask_to_array(bundle["traces"][:, start:stop], roi_mask)

        mean_pkg[f"DMD{dmd}"]["image_identity"] = {
            img: summarize_event_tensor(arr) for img, arr in aligned_images.items()
        }
        mean_pkg[f"DMD{dmd}"]["change"] = summarize_event_tensor(aligned_changes)
        mean_pkg[f"DMD{dmd}"]["omission"] = summarize_event_tensor(aligned_omissions)
        mean_pkg[f"DMD{dmd}"]["roi_ids"] = roi_ids
        mean_pkg[f"DMD{dmd}"]["valid_rois_mask"] = roi_mask

        single_pkg[f"DMD{dmd}"]["image_identity"] = aligned_images
        single_pkg[f"DMD{dmd}"]["change"] = aligned_changes
        single_pkg[f"DMD{dmd}"]["omission"] = aligned_omissions
        single_pkg[f"DMD{dmd}"]["roi_ids"] = roi_ids
        single_pkg[f"DMD{dmd}"]["valid_rois_mask"] = roi_mask

        seq_pkg[f"DMD{dmd}"]["image_identity"] = _build_sequence_output(
            seq_events,
            ordered_snippets,
            n_rois_kept=n_rois_kept,
            n_time=n_img_time,
        )
        seq_pkg[f"DMD{dmd}"]["roi_ids"] = roi_ids
        seq_pkg[f"DMD{dmd}"]["valid_rois_mask"] = roi_mask

        image_count_by_id = {img: int(arr.shape[0]) for img, arr in aligned_images.items()}
        zero_count_ids = [img for img, cnt in image_count_by_id.items() if cnt == 0]

        meta_out["per_dmd"][f"DMD{dmd}"] = {
            "n_rois_total": n_rois_total,
            "n_rois_kept": n_rois_kept,
            "n_trials_total": int(len(bundle["trial_valid_mask"])),
            "n_trials_valid": int(np.sum(bundle["trial_valid_mask"])),
            "n_trials_invalid": int(np.sum(~bundle["trial_valid_mask"])),
            "trial_lengths_samples": bundle["trial_lengths_samples"].tolist(),
            "reconstructed_duration_sec": float(bundle["reconstructed_duration_sec"]),
            "duration_vs_epoch_error_sec": float(bundle["reconstructed_duration_sec"] - epoch_duration_sec),
            "n_image_ids_extracted": int(len(image_count_by_id)),
            "image_count_by_id": image_count_by_id,
            "zero_count_image_ids": zero_count_ids,
            "n_change_events_extracted": int(aligned_changes.shape[0]),
            "n_omission_events_extracted": int(aligned_omissions.shape[0]),
            "sequence_counts_by_id": {
                img: {
                    "n_prechange_sequences": int(seq_pkg[f"DMD{dmd}"]["image_identity"][img]["prechange"]["n_sequences"]),
                    "n_repeated_sequences": int(seq_pkg[f"DMD{dmd}"]["image_identity"][img]["repeated"]["n_sequences"]),
                    "n_terminal_sequences": int(seq_pkg[f"DMD{dmd}"]["image_identity"][img]["terminal"]["n_sequences"]),
                }
                for img in seq_pkg[f"DMD{dmd}"]["image_identity"].keys()
            },
        }

    np.savez_compressed(mean_npz, data=np.array([mean_pkg], dtype=object))
    np.savez_compressed(single_npz, data=np.array([single_pkg], dtype=object))
    np.savez_compressed(seq_npz, data=np.array([seq_pkg], dtype=object))

    with open(meta_json, "w") as f:
        json.dump(meta_out, f, indent=2)

    return {
        "status": "ok",
        "mean_npz": str(mean_npz),
        "single_npz": str(single_npz),
        "seq_npz": str(seq_npz),
        "meta_json": str(meta_json),
    }