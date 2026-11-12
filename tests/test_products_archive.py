"""Legacy products move under legacy_*/; reextract_* contents are promoted; new products land at the modality root."""
from pathlib import Path
from types import SimpleNamespace

from vip_slap2_analysis.common.products import (
    DEFAULT_VOLTAGE_TRACE_VARIANT, LEGACY_DIRNAME, apply_legacy_archive, describe_plan, plan_legacy_archive,
)


def _touch(p: Path, text: str = "x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_plan_and_apply_legacy_archive(tmp_path):
    a = tmp_path / "analysis"
    files = {
        "derived/voltage/voltage_mean_dff_robust_f0_trial.npz": "v1",
        "derived/voltage/voltage_session_traces_dff_robust_f0_trial.h5": "v2",
        "derived/glutamate/glutamate_mean_dFF_ls.npz": "g1",
        "derived/glutamate/glutamate_analysis/activation_event_table.csv": "g2",
        "derived/calcium/calcium_mean_dff.npz": "c1",
        "qc/voltage/voltage_qc.json": "q1",
        "qc/slap2/qc_metadata.json": "q2",
        "qc/behavior/imaging_epochs.csv": "keep",
        "qc/sync/cycle_registration_DMD1.json": "keep",
        "derived/regression_residuals/x.csv": "keep",
        # an earlier run wrote new products into a transitional folder
        "derived/voltage/reextract_2026-11/voltage_mean_voltex_dff_trial.npz": "new",
        "qc/voltage/reextract_2026-11/voltage_extraction_qc_voltex_dff_trial.json": "new",
    }
    for rel, txt in files.items():
        _touch(a / rel, txt)
    asset = SimpleNamespace(session_id="900001_2026-01-01_00-00-00", derived_dir=a / "derived", qc_dir=a / "qc")

    plan = plan_legacy_archive(asset)
    txt = describe_plan(plan)
    assert plan.n_legacy == 7 and plan.n_promote == 2 and not plan.skipped, txt
    # untouched: behavior, sync, non-modality derived folders
    assert not any("qc/behavior" in str(m.src) or "qc/sync" in str(m.src) or "regression_residuals" in str(m.src) for m in plan.moves)

    manifest = apply_legacy_archive(plan, manifest_dir=a / "qc")
    assert manifest.exists() and manifest.read_text().count("\n") == 10  # header + 9 moves
    assert (a / "derived/voltage" / LEGACY_DIRNAME["voltage"] / "voltage_mean_dff_robust_f0_trial.npz").read_text() == "v1"
    assert (a / "derived/glutamate" / LEGACY_DIRNAME["glutamate"] / "glutamate_analysis/activation_event_table.csv").read_text() == "g2"
    assert (a / "derived/calcium/legacy_LoCo/calcium_mean_dff.npz").exists()
    assert (a / "qc/slap2/legacy_LoCo/qc_metadata.json").exists()
    assert (a / "derived/voltage" / f"voltage_mean_{DEFAULT_VOLTAGE_TRACE_VARIANT}.npz").read_text() == "new"
    assert (a / "qc/voltage/voltage_extraction_qc_voltex_dff_trial.json").exists()
    assert not (a / "derived/voltage/reextract_2026-11").exists()
    assert (a / "qc/behavior/imaging_epochs.csv").exists() and (a / "derived/regression_residuals/x.csv").exists()
    # root now holds only new products + the legacy folder
    from vip_slap2_analysis.common.products import ARCHIVE_MARKER as _MK
    assert sorted(p.name for p in (a / "derived/voltage").iterdir()) == sorted([LEGACY_DIRNAME["voltage"], f"voltage_mean_{DEFAULT_VOLTAGE_TRACE_VARIANT}.npz", _MK])

    # idempotent: archived folders carry a marker and are left alone, so current products never get archived
    from vip_slap2_analysis.common.products import ARCHIVE_MARKER
    assert (a / "derived/voltage" / ARCHIVE_MARKER).exists() and (a / "qc/slap2" / ARCHIVE_MARKER).exists()
    plan2 = plan_legacy_archive(asset)
    assert not plan2.moves and any("already archived" in n for n in plan2.notes)
    # force would archive the (current) root content: the escape hatch, never the default
    assert plan_legacy_archive(asset, force=True).n_legacy >= 1


def test_archive_skips_when_destination_exists(tmp_path):
    a = tmp_path / "analysis"
    _touch(a / "derived/voltage/voltage_mean_dff_robust_f0_trial.npz", "old")
    _touch(a / "derived/voltage/legacy_extractDendrites/voltage_mean_dff_robust_f0_trial.npz", "already")
    asset = SimpleNamespace(session_id="s", derived_dir=a / "derived", qc_dir=a / "qc")
    plan = plan_legacy_archive(asset)
    assert plan.n_legacy == 0 and len(plan.skipped) == 1 and plan.skipped[0]["reason"] == "destination exists"
