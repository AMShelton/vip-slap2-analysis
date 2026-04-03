from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from vip_slap2_analysis.common.session import SessionAssets
from vip_slap2_analysis.glutamate.summary import GlutamateSummary
from vip_slap2_analysis.packaging.stimulus_events import (
    DEFAULT_EVENT_TIME_COLUMN,
    extract_stimulus_events,
    write_stimulus_events_json,
)
from vip_slap2_analysis.packaging.trial_concat import (
    concatenate_trial_stack,
    stack_trials_padded,
    trial_lengths,
)


def _session_export_root(
    asset: SessionAssets,
    *,
    package_name: str = "soma_calcium",
    base_dir: str | Path | None = None,
) -> Path:
    if base_dir is not None:
        root = Path(base_dir)
    elif asset.derived_dir is not None:
        root = Path(asset.derived_dir) / "packaged" / package_name
    else:
        root = Path(asset.session_dir) / "analysis" / "derived" / "packaged" / package_name
    return root / str(asset.session_id)


def _safe_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    try:
        import pandas as pd  # local import to avoid hard dependency here

        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_safe_jsonable(dict(payload)), f, indent=2)
    return path


def _detect_dmd_depth_um(asset: SessionAssets, dmd: int) -> Optional[float]:
    candidates = (
        f"dmd{dmd}_depth_um",
        f"dmd{dmd}_depth",
        f"depth_dmd{dmd}",
        f"depth_um_dmd{dmd}",
        f"DMD{dmd}_depth_um",
        f"DMD{dmd}_depth",
    )
    for key in candidates:
        if key in asset.metadata:
            val = asset.metadata.get(key)
            try:
                if val is None:
                    return None
                return float(val)
            except Exception:
                continue
    return None


def _guess_session_label(asset: SessionAssets) -> Optional[str]:
    candidates = (
        "session_type",
        "experience_level",
        "familiarity",
        "novelty",
        "image_set",
    )
    for key in candidates:
        if key in asset.metadata:
            val = asset.metadata.get(key)
            if val is not None and str(val) != "nan":
                return str(val)
    return None


def _valid_trial_mask(gs: GlutamateSummary, dmd: int) -> np.ndarray:
    return np.asarray(gs.keep_trials[dmd - 1], dtype=bool)


def _load_raw_soma_calcium_trials(
    gs: GlutamateSummary,
    *,
    dmd: int,
    trace_type: str,
    roi_inds: Optional[Sequence[int]] = None,
) -> list[np.ndarray | None]:
    trials: list[np.ndarray | None] = []
    keep = _valid_trial_mask(gs, dmd)
    for trial in range(1, gs.n_trials + 1):
        if not keep[trial - 1]:
            trials.append(None)
            continue
        _, ca = gs.get_soma_glu_ca_traces(
            dmd=dmd,
            trial=trial,
            trace_type=trace_type,
            roi_inds=roi_inds,
        )
        ca = np.asarray(ca, dtype=float)
        if ca.ndim != 2:
            raise ValueError(
                f"Expected raw calcium trial array with shape (n_rois, time). Got {ca.shape} for dmd={dmd}, trial={trial}."
            )
        trials.append(ca)
    return trials


def _load_processed_soma_calcium_trials(
    gs: GlutamateSummary,
    *,
    dmd: int,
    trace_type: str,
    fs_hz: float,
    roi_inds: Optional[Sequence[int]] = None,
    process_kwargs: Optional[Mapping[str, Any]] = None,
) -> list[np.ndarray | None]:
    process_kwargs = dict(process_kwargs or {})
    out = gs.get_processed_soma_ca_all_trials(
        dmd=dmd,
        trace_type=trace_type,
        roi_inds=roi_inds,
        fs_hz=fs_hz,
        pad_to="none",
        include_invalid=True,
        **process_kwargs,
    )
    dff_trials = out["dff"]
    if len(dff_trials) != gs.n_trials:
        raise ValueError(
            f"Expected {gs.n_trials} processed trials for dmd={dmd}; got {len(dff_trials)}."
        )
    cleaned: list[np.ndarray | None] = []
    for tr in dff_trials:
        if tr is None:
            cleaned.append(None)
        else:
            arr = np.asarray(tr, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"Expected processed calcium trial with shape (n_rois, time). Got {arr.shape}.")
            cleaned.append(arr)
    return cleaned


def _package_trace_family(
    trials: Sequence[np.ndarray | None],
    *,
    fs_hz: float,
) -> Dict[str, Any]:
    trial_stack = stack_trials_padded(trials)
    concat = concatenate_trial_stack(trial_stack)
    fill_length = int(trial_stack.shape[-1])
    return {
        "trial_stack": trial_stack,
        "session_concat": concat,
        "trial_lengths_samples": np.asarray(trial_lengths(trials, invalid_fill_length=fill_length), dtype=int),
        "trial_start_times_s": np.arange(trial_stack.shape[0], dtype=float) * (fill_length / float(fs_hz)),
        "fill_length_samples": fill_length,
        "n_rois": int(trial_stack.shape[1]),
        "n_trials": int(trial_stack.shape[0]),
    }


def _write_trace_npz(
    output_path: str | Path,
    *,
    trace_payload: Mapping[str, Any],
    trace_kind: str,
    dmd: int,
    fs_hz: float,
    roi_axis_name: str = "soma_roi",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        trial_stack=np.asarray(trace_payload["trial_stack"], dtype=float),
        session_concat=np.asarray(trace_payload["session_concat"], dtype=float),
        trial_lengths_samples=np.asarray(trace_payload["trial_lengths_samples"], dtype=int),
        trial_start_times_s=np.asarray(trace_payload["trial_start_times_s"], dtype=float),
        fs_hz=float(fs_hz),
        dmd=int(dmd),
        trace_kind=str(trace_kind),
        roi_axis_name=str(roi_axis_name),
        fill_length_samples=int(trace_payload["fill_length_samples"]),
        n_rois=int(trace_payload["n_rois"]),
        n_trials=int(trace_payload["n_trials"]),
    )
    return output_path


def _common_session_metadata(asset: SessionAssets, gs: GlutamateSummary) -> Dict[str, Any]:
    return {
        "session_id": asset.session_id,
        "subject_id": asset.subject_id,
        "session_dir": str(asset.session_dir),
        "summary_mat": None if asset.summary_mat is None else str(asset.summary_mat),
        "bonsai_event_log_csv": None if asset.bonsai_event_log_csv is None else str(asset.bonsai_event_log_csv),
        "session_label": _guess_session_label(asset),
        "sampling_rate_hz": float(gs.metadata.get("analyzeHz", np.nan)),
        "n_trials": int(gs.n_trials),
        "asset_metadata": asset.metadata,
    }


def package_session_soma_calcium(
    asset: SessionAssets,
    *,
    output_root: str | Path | None = None,
    trace_type: str = "Fsvd",
    dmds: Iterable[int] = (1, 2),
    roi_inds: Optional[Sequence[int]] = None,
    process_kwargs: Optional[Mapping[str, Any]] = None,
    event_time_col: str = DEFAULT_EVENT_TIME_COLUMN,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if asset.summary_mat is None:
        raise FileNotFoundError(f"No SummaryLoCo .mat file was resolved for session {asset.session_id}.")
    if asset.bonsai_event_log_csv is None:
        raise FileNotFoundError(f"No bonsai_event_log.csv was resolved for session {asset.session_id}.")

    session_root = _session_export_root(asset, base_dir=output_root)
    session_root.mkdir(parents=True, exist_ok=True)

    gs = GlutamateSummary(asset.summary_mat, keep_open=True)
    try:
        fs_hz = float(gs.metadata.get("analyzeHz", np.nan))
        if not np.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError(f"Could not resolve a valid analyzeHz for session {asset.session_id}.")

        events = extract_stimulus_events(
            asset.bonsai_event_log_csv,
            time_col=event_time_col,
        )
        write_stimulus_events_json(session_root / "stimulus_events.json", events)

        session_meta = _common_session_metadata(asset, gs)
        session_meta["dmd_exports"] = {}

        for dmd in dmds:
            dmd_dir = session_root / f"DMD{int(dmd)}"
            raw_npz = dmd_dir / "raw_soma_calcium.npz"
            proc_npz = dmd_dir / "processed_soma_calcium_dff.npz"

            if not overwrite and raw_npz.exists() and proc_npz.exists():
                session_meta["dmd_exports"][f"DMD{int(dmd)}"] = {
                    "status": "exists",
                    "depth_um": _detect_dmd_depth_um(asset, int(dmd)),
                    "raw_output": str(raw_npz),
                    "processed_output": str(proc_npz),
                }
                continue

            try:
                raw_trials = _load_raw_soma_calcium_trials(
                    gs,
                    dmd=int(dmd),
                    trace_type=trace_type,
                    roi_inds=roi_inds,
                )
                processed_trials = _load_processed_soma_calcium_trials(
                    gs,
                    dmd=int(dmd),
                    trace_type=trace_type,
                    roi_inds=roi_inds,
                    fs_hz=fs_hz,
                    process_kwargs=process_kwargs,
                )

                raw_payload = _package_trace_family(raw_trials, fs_hz=fs_hz)
                proc_payload = _package_trace_family(processed_trials, fs_hz=fs_hz)

                dmd_dir.mkdir(parents=True, exist_ok=True)
                _write_trace_npz(
                    raw_npz,
                    trace_payload=raw_payload,
                    trace_kind="raw_calcium",
                    dmd=int(dmd),
                    fs_hz=fs_hz,
                )
                _write_trace_npz(
                    proc_npz,
                    trace_payload=proc_payload,
                    trace_kind="processed_calcium_dff",
                    dmd=int(dmd),
                    fs_hz=fs_hz,
                )

                session_meta["dmd_exports"][f"DMD{int(dmd)}"] = {
                    "status": "exported",
                    "depth_um": _detect_dmd_depth_um(asset, int(dmd)),
                    "n_rois": int(raw_payload["n_rois"]),
                    "fill_length_samples": int(raw_payload["fill_length_samples"]),
                    "raw_output": str(raw_npz),
                    "processed_output": str(proc_npz),
                }
            except Exception as exc:
                session_meta["dmd_exports"][f"DMD{int(dmd)}"] = {
                    "status": "skipped",
                    "depth_um": _detect_dmd_depth_um(asset, int(dmd)),
                    "reason": repr(exc),
                }

    _write_json(session_root / "session_metadata.json", session_meta)
    return session_meta


def package_soma_calcium_batch(
    assets: Sequence[SessionAssets],
    *,
    output_root: str | Path | None = None,
    trace_type: str = "Fsvd",
    dmds: Iterable[int] = (1, 2),
    roi_inds: Optional[Sequence[int]] = None,
    process_kwargs: Optional[Mapping[str, Any]] = None,
    event_time_col: str = DEFAULT_EVENT_TIME_COLUMN,
    overwrite: bool = False,
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for asset in assets:
        results.append(
            package_session_soma_calcium(
                asset,
                output_root=output_root,
                trace_type=trace_type,
                dmds=dmds,
                roi_inds=roi_inds,
                process_kwargs=process_kwargs,
                event_time_col=event_time_col,
                overwrite=overwrite,
            )
        )
    return results
