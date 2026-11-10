
from pathlib import Path
import argparse
import json
import zipfile
import io

import h5py
import numpy as np


KEYWORDS = (
    "time", "timestamp", "trace", "signal",
    "voltage", "fluorescence", "cycle",
    "sample", "epoch", "trial", "dff",
    "dmd", "roi"
)

# Export metadata for every HDF5 dataset, but only
# bounded data samples from likely relevant datasets.
MAX_ARRAY_ELEMENTS = 15000
MAX_DATASETS_EXPORTED = 40
WINDOWS = ("start", "middle", "end")


def safe_json(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [safe_json(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_window(ds, position):
    """Read a bounded window; never load the entire dataset."""
    if ds.ndim == 0:
        return np.asarray(ds[()])

    shape = ds.shape
    if any(dim == 0 for dim in shape):
        return np.array([])

    # Treat the largest dimension as the sampling axis.
    # Record this assumption explicitly in the output.
    axis = int(np.argmax(shape))

    slices = []
    for dimension, length in enumerate(shape):
        if dimension == axis:
            count = min(length, MAX_ARRAY_ELEMENTS)
            if position == "start":
                begin = 0
            elif position == "middle":
                begin = max(0, (length - count) // 2)
            else:
                begin = max(0, length - count)
            slices.append(slice(begin, begin + count))
        else:
            slices.append(slice(0, min(length, 1)))

    return np.asarray(ds[tuple(slices)])


def inspect_h5(path, tag, output):
    records = []
    exported = 0

    with h5py.File(path, "r") as h5:
        root_attrs = {
            k: safe_json(v) for k, v in h5.attrs.items()
        }

        def visitor(name, obj):
            nonlocal exported

            if not isinstance(obj, h5py.Dataset):
                return

            record = {
                "path": name,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "attrs": {
                    k: safe_json(v)
                    for k, v in obj.attrs.items()
                },
                "exported": False,
            }

            is_numeric = obj.dtype.kind in "iufb"
            relevant = any(
                word in name.lower() for word in KEYWORDS
            )

            if (
                is_numeric
                and relevant
                and exported < MAX_DATASETS_EXPORTED
            ):
                record["sampling_axis_assumption"] = (
                    int(np.argmax(obj.shape))
                    if obj.ndim else None
                )
                record["exported"] = True

                for position in WINDOWS:
                    arr = get_window(obj, position)

                    if arr.size > MAX_ARRAY_ELEMENTS:
                        arr = arr.ravel()[:MAX_ARRAY_ELEMENTS]

                    buf = io.BytesIO()
                    np.save(buf, arr, allow_pickle=False)

                    output.writestr(
                        f"{tag}/arrays/"
                        f"{exported:03d}_{position}.npy",
                        buf.getvalue()
                    )

                exported += 1

            records.append(record)

        h5.visititems(visitor)

    return {
        "source": str(path),
        "size_bytes": path.stat().st_size,
        "root_attrs": root_attrs,
        "datasets": records,
        "n_exported": exported,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--output", default="sync_example_export.zip"
    )
    args = parser.parse_args()

    session = Path(args.session)
    if not session.is_dir():
        raise FileNotFoundError(session)

    patterns = {
        "extracted_voltage":
            "dendriticVoltageTraces-*.h5",
        "derived_voltage":
            "voltage_session_traces_dff_robust_f0_trial.h5",
    }

    manifest = {
        "session": str(session),
        "notes": [
            "Read-only export.",
            "No filtering or smoothing applied.",
            "Sampling axis is inferred, not verified.",
            "Array slices are exploratory; see dataset metadata.",
            "HDF5 object paths are retained for audit."
        ],
        "sources": {}
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED
    ) as z:
        for tag, pattern in patterns.items():
            matches = sorted(session.rglob(pattern))

            # Avoid silently selecting an arbitrary source
            # if more than one exists.
            manifest["sources"][tag] = []

            for i, path in enumerate(matches):
                source_tag = f"{tag}_{i}"
                info = inspect_h5(path, source_tag, z)
                manifest["sources"][tag].append(info)

        # Include the existing synchronization QC.
        for relative in (
            "analysis/qc/behavior/alignment_meta.json",
            "analysis/qc/behavior/imaging_epochs.csv",
            "analysis/qc/behavior/di3_pulse_train_detection.json",
            "analysis/qc/voltage/voltage_qc_metadata.json",
        ):
            path = session / relative
            if path.is_file():
                z.write(path, arcname="qc/" + path.name)

        z.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, allow_nan=False)
        )

    print(f"Export created: {output.resolve()}")
    print("Review contents before sharing.")


if __name__ == "__main__":
    main()
