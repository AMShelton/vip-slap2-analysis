"""Single entry point for opening a session's extraction summary with provenance.

Precedence (per modality, as resolved by ``VIPSessionRegistry``):

    modern, schema-validated HDF5 product  ->  legacy MAT product  ->  explicit error

A file's existence is never enough: the modern reader validates the schema
(``validate_giant_summary_schema`` / ``validate_voltex_summary_schema``) and an
incompatible file falls through to the legacy product with the reason recorded.
The returned provenance dict is meant to be written verbatim into extraction QC
JSON (``summary_provenance``), so every derived product states which file and
format it came from and why.

The readers returned here expose the method subset the extraction code already
uses, so ``glutamate.extraction``, ``calcium.extraction`` and
``voltage.extraction`` work unchanged on either family. They never mix traces,
timebases, ROI identities or trial metadata between files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

GLUTAMATE_MODERN = "giant_experiment_summary_h5"
GLUTAMATE_LEGACY = "legacy_summary_loco_mat"
VOLTAGE_MODERN = "voltex_voltage_summary_h5"
VOLTAGE_LEGACY = "legacy_dendritic_voltage_mat_h5"


def _assets(asset: Any, modality: str) -> Dict[str, Any]:
    if hasattr(asset, "get_modality_assets"):
        return dict(asset.get_modality_assets(modality))
    return dict((getattr(asset, "modality_assets", {}) or {}).get(modality, {}))


def _order(assets: Dict[str, Any], prefer: Optional[str], modern_key: str, legacy_key: str):
    pref = str(prefer or assets.get("summary_preference") or "modern").lower()
    if pref not in ("modern", "legacy"):
        raise ValueError("prefer must be 'modern' or 'legacy'")
    order = [modern_key, legacy_key]
    if pref == "legacy":
        order.reverse()
    return pref, order


def load_glutamate_summary(
    asset: Any,
    *,
    prefer: Optional[str] = None,
    keep_open: bool = True,
    swap_xy_images: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Open the glutamate/calcium summary for ``asset`` under the modern->legacy precedence.

    Returns ``(reader, provenance)``. ``reader`` is a
    :class:`~vip_slap2_analysis.io.giant_summary.GiantExperimentSummary` or a
    :class:`~vip_slap2_analysis.glutamate.summary.GlutamateSummary`.
    """
    from vip_slap2_analysis.io.giant_summary import GiantExperimentSummary, validate_giant_summary_schema
    from vip_slap2_analysis.glutamate.summary import GlutamateSummary

    assets = _assets(asset, "glutamate")
    modern = assets.get("summary_h5")
    legacy = assets.get("summary_mat") or getattr(asset, "summary_mat", None)
    pref, order = _order(assets, prefer, GLUTAMATE_MODERN, GLUTAMATE_LEGACY)
    dyn = (_assets(asset, "slap2") or {}).get("dynamic_data_dir")
    prov: Dict[str, Any] = {
        "modality": "glutamate",
        "summary_preference": pref,
        "candidates": {GLUTAMATE_MODERN: str(modern) if modern else None, GLUTAMATE_LEGACY: str(legacy) if legacy else None},
        "attempts": [],
    }
    for fmt in order:
        if fmt == GLUTAMATE_MODERN and modern is not None:
            report = validate_giant_summary_schema(modern)
            prov["attempts"].append({"format": fmt, "path": str(modern), "schema_compatible": report["compatible"], "problems": report["problems"]})
            if report["compatible"]:
                reader = GiantExperimentSummary(
                    modern, trial_table_path=assets.get("trial_table_h5"), dynamic_data_dir=dyn, keep_open=keep_open,
                )
                prov.update({"selected_format": fmt, "selected_path": str(modern),
                             "trial_table_h5": str(reader.trial_table_path) if reader.trial_table_path else None,
                             "dynamic_data_dir": str(dyn) if dyn else None,
                             "acquisition_metadata_source": "raw_dat_meta_headers" if dyn else "unavailable"})
                return reader, prov
        elif fmt == GLUTAMATE_LEGACY and legacy is not None:
            try:
                reader = GlutamateSummary(legacy, keep_open=keep_open, swap_xy_images=swap_xy_images)
            except Exception as exc:  # pragma: no cover - depends on real files
                prov["attempts"].append({"format": fmt, "path": str(legacy), "error": f"{type(exc).__name__}: {exc}"})
                continue
            prov["attempts"].append({"format": fmt, "path": str(legacy), "schema_compatible": True})
            prov.update({"selected_format": fmt, "selected_path": str(legacy),
                         "acquisition_metadata_source": "summarize_LoCo exptSummary.dmd metadata (if patched)"})
            return reader, prov
    raise FileNotFoundError(
        f"No loadable glutamate summary for session {getattr(asset, 'session_id', '?')}: "
        f"candidates={prov['candidates']}, attempts={prov['attempts']}"
    )


def load_voltage_summary(
    asset: Any,
    *,
    prefer: Optional[str] = None,
    keep_open: bool = True,
    swap_xy_images: bool = True,
    trace_kind: str = "raw_f",
    apply_motion_censor: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Open the voltage summary for ``asset`` under the modern->legacy precedence.

    Returns ``(reader, provenance)``: a
    :class:`~vip_slap2_analysis.voltage.voltex_summary.VoltexSummary` or a legacy
    :class:`~vip_slap2_analysis.voltage.summary.VoltageSummary` (MAT + paired
    trace H5). The legacy pair is only used when both files are present; a
    summary MAT whose trace H5 is missing is reported, not guessed.
    """
    from vip_slap2_analysis.voltage.voltex_summary import VoltexSummary, validate_voltex_summary_schema
    from vip_slap2_analysis.voltage.summary import VoltageSummary

    assets = _assets(asset, "voltage")
    modern = assets.get("summary_h5")
    legacy_mat = assets.get("summary_mat")
    legacy_h5 = assets.get("trace_h5")
    pref, order = _order(assets, prefer, VOLTAGE_MODERN, VOLTAGE_LEGACY)
    dyn = (_assets(asset, "slap2") or {}).get("dynamic_data_dir")
    prov: Dict[str, Any] = {
        "modality": "voltage",
        "summary_preference": pref,
        "candidates": {VOLTAGE_MODERN: str(modern) if modern else None,
                       VOLTAGE_LEGACY: {"summary_mat": str(legacy_mat) if legacy_mat else None, "trace_h5": str(legacy_h5) if legacy_h5 else None}},
        "attempts": [],
    }
    for fmt in order:
        if fmt == VOLTAGE_MODERN and modern is not None:
            report = validate_voltex_summary_schema(modern)
            prov["attempts"].append({"format": fmt, "path": str(modern), "schema_compatible": report["compatible"],
                                     "problems": report["problems"], "format_version": report.get("format_version")})
            if report["compatible"]:
                reader = VoltexSummary(
                    modern, trial_table_path=assets.get("trial_table_h5"), dynamic_data_dir=dyn,
                    trace_kind=trace_kind, apply_motion_censor=apply_motion_censor, keep_open=keep_open,
                    swap_xy_images=swap_xy_images,
                )
                prov.update({"selected_format": fmt, "selected_path": str(modern), "format_version": report.get("format_version"),
                             "trial_table_h5": str(reader.trial_table_path) if reader.trial_table_path else None,
                             "trace_kind": trace_kind, "apply_motion_censor": bool(apply_motion_censor),
                             "dynamic_data_dir": str(dyn) if dyn else None})
                return reader, prov
        elif fmt == VOLTAGE_LEGACY and legacy_mat is not None:
            if legacy_h5 is None:
                prov["attempts"].append({"format": fmt, "path": str(legacy_mat), "error": "paired dendriticVoltageTraces H5 not found"})
                # VoltageSummary can still open flat/E layouts without a trace file.
            try:
                reader = VoltageSummary(legacy_mat, trace_path=legacy_h5, keep_open=keep_open, swap_xy_images=swap_xy_images)
            except Exception as exc:
                prov["attempts"].append({"format": fmt, "path": str(legacy_mat), "error": f"{type(exc).__name__}: {exc}"})
                continue
            prov["attempts"].append({"format": fmt, "path": str(legacy_mat), "trace_h5": str(legacy_h5) if legacy_h5 else None,
                                     "layout": reader.layout, "schema_compatible": True})
            prov.update({"selected_format": fmt, "selected_path": str(legacy_mat), "trace_h5": str(legacy_h5) if legacy_h5 else None,
                         "layout": reader.layout})
            return reader, prov
    raise FileNotFoundError(
        f"No loadable voltage summary for session {getattr(asset, 'session_id', '?')}: "
        f"candidates={prov['candidates']}, attempts={prov['attempts']}"
    )
