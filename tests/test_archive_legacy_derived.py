"""archive_legacy_derived: legacy products move into legacy_<preprocessor>/, new products block the move."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from archive_legacy_derived import apply_rows, plan_session  # noqa: E402


def _asset(tmp_path):
    a = tmp_path / "analysis"
    (a / "derived" / "voltage" / "movies").mkdir(parents=True)
    (a / "derived" / "voltage" / "voltage_mean_dff_robust_f0_trial.npz").write_bytes(b"x")
    (a / "derived" / "voltage" / "movies" / "m.mp4").write_bytes(b"x")
    (a / "derived" / "glutamate" / "glutamate_analysis").mkdir(parents=True)
    (a / "derived" / "glutamate" / "glutamate_mean_dFF_ls.npz").write_bytes(b"x")
    (a / "qc" / "voltage").mkdir(parents=True); (a / "qc" / "voltage" / "dmd1_recommended_voltage_rois.npy").write_bytes(b"x")
    (a / "qc" / "slap2").mkdir(parents=True); (a / "qc" / "slap2" / "synapse_qc.csv").write_bytes(b"x")
    (a / "qc" / "behavior").mkdir(parents=True); (a / "qc" / "behavior" / "imaging_epochs.csv").write_bytes(b"x")
    (a / "qc" / "sync").mkdir(parents=True); (a / "qc" / "sync" / "cycle_registration_DMD1.json").write_bytes(b"{}")
    (a / "derived" / "voltage" / "reextract_2026-11").mkdir()
    (a / "derived" / "voltage" / "reextract_2026-11" / "voltage_mean_voltex_dff_trial.npz").write_bytes(b"new")
    return SimpleNamespace(session_id="900000_2026-01-01_00-00-00", derived_dir=a / "derived", qc_dir=a / "qc"), a


def test_plan_and_apply_moves_legacy_and_promotes_reextract(tmp_path):
    asset, a = _asset(tmp_path)
    rows = plan_session(asset, promote_subdir="reextract_2026-11")
    actions = {(r["family"], r["modality"], Path(r["source"]).name): r["action"] for r in rows}
    assert actions[("derived", "voltage", "voltage_mean_dff_robust_f0_trial.npz")] == "move"
    assert actions[("derived", "voltage", "movies")] == "move"
    assert actions[("qc", "slap2", "synapse_qc.csv")] == "move"
    assert actions[("derived", "voltage", "voltage_mean_voltex_dff_trial.npz")] == "promote"
    assert not any(r["family"] == "qc" and r["modality"] in ("behavior", "sync") for r in rows)
    apply_rows(rows)
    assert (a / "derived" / "voltage" / "legacy_extractDendrites" / "voltage_mean_dff_robust_f0_trial.npz").exists()
    assert (a / "derived" / "voltage" / "legacy_extractDendrites" / "movies" / "m.mp4").exists()
    assert (a / "derived" / "glutamate" / "legacy_LoCo" / "glutamate_analysis").is_dir()
    assert (a / "qc" / "slap2" / "legacy_LoCo" / "synapse_qc.csv").exists()
    assert (a / "derived" / "voltage" / "voltage_mean_voltex_dff_trial.npz").exists()
    assert not (a / "derived" / "voltage" / "reextract_2026-11").exists()
    assert (a / "qc" / "behavior" / "imaging_epochs.csv").exists() and (a / "qc" / "sync" / "cycle_registration_DMD1.json").exists()
    man = json.loads((a / "derived" / "voltage" / "legacy_extractDendrites" / "_archive_manifest.json").read_text())
    assert {i["name"] for i in man["items"]} == {"voltage_mean_dff_robust_f0_trial.npz", "movies"}
    # second run: nothing left to move, legacy folder present -> no rows for those dirs
    rows2 = plan_session(asset, promote_subdir=None)
    assert all(r["action"] == "needs_review" for r in rows2)       # only the promoted new product remains at root -> review, not move


def test_new_products_at_root_block_the_move(tmp_path):
    asset, a = _asset(tmp_path)
    (a / "derived" / "voltage" / "voltage_extraction_qc_x.json").write_text(json.dumps({"trace_source_effective": "product"}))
    rows = plan_session(asset, promote_subdir=None)
    volt = [r for r in rows if r["family"] == "derived" and r["modality"] == "voltage"]
    assert volt and all(r["action"] == "needs_review" for r in volt)
