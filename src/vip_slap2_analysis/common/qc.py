from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from vip_slap2_analysis.glutamate.summary import GlutamateSummary
from vip_slap2_analysis.common.session import SessionAssets


def robust_noise_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def robust_signal_amplitude(x: np.ndarray, q: float = 99.0) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return np.percentile(np.abs(x), q)


def concatenate_synapse_trials(
    exp: GlutamateSummary,
    dmd: int,
    trace_type: str = "dF",
    trace_mode: str = "ls",
    pad_invalid_trials_with_nan: bool = True,
) -> np.ndarray:
    """
    Returns
    -------
    concat : ndarray
        Shape (n_synapses, total_samples_across_trials)
    """
    chunks = []
    n_syn = exp.n_synapses[dmd - 1]

    ref_len = None
    for tr in range(1, exp.n_trials + 1):
        if tr in exp.valid_trials[dmd - 1]:
            x = exp.get_traces(dmd=dmd, trial=tr, signal=trace_type, mode=trace_mode)
            # expected internal shape ~ (samples, rois, channels)
            if x.ndim == 3:
                x = x[:, :, 0]
            x = np.asarray(x).T  # -> (n_syn, samples)
            if ref_len is None:
                ref_len = x.shape[1]
            chunks.append(x)
        else:
            if pad_invalid_trials_with_nan:
                if ref_len is None:
                    # defer padding until we know trial length
                    continue
                chunks.append(np.full((n_syn, ref_len), np.nan))

    if not chunks:
        return np.empty((n_syn, 0))

    return np.concatenate(chunks, axis=1)


def compute_synapse_snr(
    concat: np.ndarray,
    snr_threshold: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    n_syn = concat.shape[0]
    snr = np.full(n_syn, np.nan)

    for i in range(n_syn):
        x = concat[i]
        sigma = robust_noise_sigma(x)
        signal = robust_signal_amplitude(x)
        if np.isfinite(sigma) and sigma > 0:
            snr[i] = signal / sigma

    keep = snr >= snr_threshold
    return snr, keep


def compute_session_qc(
    assets: SessionAssets,
    min_valid_fraction: float = 0.5,
    snr_threshold: float = 6.0,
    trace_type: str = "dF",
    trace_mode: str = "ls",
) -> dict:
    if assets.summary_mat is None:
        raise FileNotFoundError(f"No SummaryLoCo*.mat found for {assets.session_id}")

    exp = GlutamateSummary(assets.summary_mat)

    out = {
        "session_id": assets.session_id,
        "summary_mat": str(assets.summary_mat),
        "passes_session_qc": True,
        "dmd_results": {},
    }

    for dmd in range(1, exp.n_dmds + 1):
        valid_trials = exp.valid_trials[dmd - 1]
        n_trials = exp.n_trials
        n_valid = len(valid_trials)
        valid_fraction = n_valid / n_trials if n_trials > 0 else np.nan
        passes = bool(valid_fraction >= min_valid_fraction)

        concat = concatenate_synapse_trials(
            exp=exp,
            dmd=dmd,
            trace_type=trace_type,
            trace_mode=trace_mode,
            pad_invalid_trials_with_nan=True,
        )
        snr, keep = compute_synapse_snr(concat, snr_threshold=snr_threshold)

        out["dmd_results"][f"DMD{dmd}"] = {
            "dmd": dmd,
            "n_trials": int(n_trials),
            "n_valid_trials": int(n_valid),
            "valid_fraction": float(valid_fraction),
            "passes_session_qc": passes,
            "snr_by_synapse": snr,
            "valid_synapses": keep,
        }

        if not passes:
            out["passes_session_qc"] = False

    exp.close()
    return out


def save_session_qc(qc: dict, qc_dir: str | Path) -> None:
    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "session_id": qc["session_id"],
        "summary_mat": qc["summary_mat"],
        "passes_session_qc": qc["passes_session_qc"],
        "dmd_results": {},
    }

    for dmd_name, d in qc["dmd_results"].items():
        np.save(qc_dir / f"{dmd_name.lower()}_snr.npy", d["snr_by_synapse"])
        np.save(qc_dir / f"{dmd_name.lower()}_valid_synapses.npy", d["valid_synapses"])

        payload["dmd_results"][dmd_name] = {
            "dmd": d["dmd"],
            "n_trials": d["n_trials"],
            "n_valid_trials": d["n_valid_trials"],
            "valid_fraction": d["valid_fraction"],
            "passes_session_qc": d["passes_session_qc"],
            "snr_file": f"{dmd_name.lower()}_snr.npy",
            "valid_synapses_file": f"{dmd_name.lower()}_valid_synapses.npy",
        }

    with open(qc_dir / "session_qc.json", "w") as f:
        json.dump(payload, f, indent=2)