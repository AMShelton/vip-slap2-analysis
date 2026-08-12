# vip-slap2-analysis

`vip-slap2-analysis` contains data-access, preprocessing, quality-control,
analysis, packaging, and plotting utilities for VIP SLAP2 experiments. The
repository is organized around multimodal physiology datasets collected from VIP
interneurons in mouse visual cortex, with an emphasis on glutamate imaging,
calcium imaging, voltage imaging, behavior alignment, and morphology-derived
context.

The main goal of the package is to make the analysis path from raw/session-level
assets to reviewable tables and figures explicit, reproducible, and easy to audit.

## Contents

- [Project scope](#project-scope)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Expected session assets](#expected-session-assets)
- [Glutamate workflow](#glutamate-workflow)
- [Calcium workflow](#calcium-workflow)
- [Behavior alignment and validation](#behavior-alignment-and-validation)
- [Metadata and manifests](#metadata-and-manifests)
- [Morphology utilities](#morphology-utilities)
- [Generated analysis outputs](#generated-analysis-outputs)
- [Coding and documentation standards](#coding-and-documentation-standards)
- [Current development notes](#current-development-notes)

## Project scope

This repository supports ongoing VIP synaptic dynamics analyses. Current use
cases include:

- loading SLAP2 `SummaryLoCo*.mat` / MATLAB v7.3 HDF5 summary files;
- extracting source, user-ROI, glutamate, and calcium traces;
- reconstructing session-long traces from trial-wise SLAP2 outputs;
- aligning physiology traces to BonVision/Bonsai/HARP stimulus events;
- classifying stimulus-evoked activation, deactivation, and no-change responses;
- estimating image tuning and image-response variance explained;
- analyzing repeated-image sequence dynamics, including adaptation and
  facilitation metrics;
- running calcium ROI quality control and extraction;
- generating session-level metadata manifests and quality summaries;
- rendering figures, movies, morphology projections, and Illustrator-friendly
  vector outputs.

The package is intended to be used both interactively from notebooks and as a
library of reusable functions for batch processing.

## Repository layout

```text
vip-slap2-analysis/
├── notebooks/
│   ├── analysis/          # exploratory and batch analysis notebooks
│   ├── behavior/          # behavior/HARP alignment notebooks
│   ├── calcium/           # soma calcium extraction and plotting notebooks
│   ├── glutamate/         # glutamate extraction, QC, and analysis notebooks
│   ├── metadata/          # metadata construction and dataset organization
│   ├── morphology/        # morphology/tracing notebooks
│   ├── plotting/          # figure-generation notebooks
│   ├── qc/                # quality-control notebooks
│   └── voltage/           # voltage-imaging access and processing notebooks
├── src/vip_slap2_analysis/
│   ├── behavior/          # event-log validation, HARP/BonVision utilities
│   ├── calcium/           # soma calcium extraction and QC
│   ├── common/            # shared session/asset models
│   ├── glutamate/         # SLAP2 summary loading, alignment, analyses
│   ├── io/                # MATLAB v7.3/HDF5 helpers and session registry code
│   ├── metadata/          # dataset manifest and quality overview builders
│   ├── morphology/        # SNT/SWC loading, metrics, smoothing, plotting
│   ├── packaging/         # reusable packaging helpers for derived datasets
│   ├── plotting/          # movies, heatmaps, QC plots, and plot utilities
│   ├── utils/             # one-off reorganization and helper utilities
│   └── voltage/           # voltage-imaging processing scaffolding
└── pyproject.toml
```

Important glutamate modules:

- `vip_slap2_analysis.glutamate.summary`
  - lazy reader for SLAP2 `ExperimentSummary` / `SummaryLoCo` files;
  - handles MATLAB v7.3 HDF5 references and cell-array-like structures;
  - exposes source traces, user-ROI traces, summary images, selected-pixel masks,
    footprints, and processed soma calcium traces when present.
- `vip_slap2_analysis.common.alignment`
  - loads corrected behavior/event tables;
  - extracts image, change, and omission intervals;
  - reconstructs DMD session traces;
  - aligns traces to stimulus windows and summarizes event tensors.
- `vip_slap2_analysis.glutamate.analysis`
  - builds event response tables;
  - classifies activation/deactivation;
  - computes image tuning and variance-explained metrics;
  - analyzes sequence-position dynamics;
  - writes analysis tables and metadata.

## Installation

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

The current `pyproject.toml` intentionally keeps package metadata minimal. In the
active analysis environment, make sure the scientific Python dependencies used by
the modules and notebooks are available, including at least:

```bash
python -m pip install numpy pandas scipy h5py matplotlib seaborn statsmodels pyarrow
```

Additional optional dependencies may be needed for specific notebooks or plotting
workflows, such as TIFF/movie rendering, morphology plotting, or NWB packaging.

## Expected session assets

The exact session layout can vary, but the current workflows expect a session or
analysis directory containing some combination of:

```text
session_root/
├── slap2/
│   └── dynamic_data/
│       └── ExperimentSummary/
│           └── SummaryLoCo-*.mat
├── behavior/
│   └── VCO1_Behavior.harp/
│       ├── bonsai_event_log.csv
│       └── device.yml
├── extracted_files/
│   └── photodiode.pkl
├── qc/
│   ├── imaging_epochs.csv
│   ├── glutamate_qc.json
│   ├── calcium_qc.json
│   └── synapse_qc.csv
└── analysis/
    └── derived/
        └── glutamate/
            ├── glutamate_single_trial_df.npz
            ├── glutamate_mean_df.npz
            └── glutamate_sequence_df.npz
```

Not every workflow needs every file. For example, `GlutamateSummary` can inspect a
single `SummaryLoCo*.mat` file directly, while `run_glutamate_analysis` expects the
derived glutamate `.npz` products created by the extraction/alignment workflow.

## Shared epoch-duration QC

Behavior, glutamate, calcium, and voltage processing use a shared acquisition-
epoch rule: epochs lasting **at least 30 seconds** are eligible for analysis by
default. Raw extractors retain all epochs; Python QC rejects shorter fragments.
A mismatch between the raw number of SLAP2 and behavior epochs is therefore
allowed when short source fragments are discarded. The numbers of accepted
source and behavior epochs must still match when `strict_epoch_match=True`.

Accepted source epochs are paired chronologically with accepted HARP/DI3
intervals. Nominal sample spacing is preserved, overlong terminal data are
clipped rather than rescaled, and voltage/calcium F0 is estimated separately
within each retained epoch. Rejected source trials use analysis epoch label `0`
and are excluded from QC metrics and event-response outputs.

The default can be changed explicitly through `min_epoch_duration_sec` in
physiology functions or `min_epoch_duration` in behavior preprocessing. See
[`EPOCH_DURATION_QC_PATCH_NOTES.md`](EPOCH_DURATION_QC_PATCH_NOTES.md) for the
full policy and modality-specific behavior.

## Glutamate workflow

### Open a SLAP2 summary file

```python
from pathlib import Path

from vip_slap2_analysis.glutamate.summary import GlutamateSummary

summary_path = Path(r"path\to\SummaryLoCo-file.mat")

exp = GlutamateSummary(summary_path)

print(exp.n_dmds)
print(exp.n_trials)
print(exp.valid_trials)
print(exp.n_synapses)

# Public methods use 1-indexed DMD and trial numbers.
traces = exp.get_traces(
    dmd=1,
    trial=1,
    signal="dF",
    mode="ls",
)

mean_image = exp.get_summary_image(dmd=1, image_type="meanIM")
selected_pixels = exp.get_sel_pix(dmd=1)

exp.close()
```

`GlutamateSummary` is designed to read data lazily. Large arrays are not loaded
unless a method explicitly requests them.

### Run the derived glutamate response analysis

```python
from vip_slap2_analysis.glutamate.analysis import (
    GlutamateAnalysisConfig,
    run_glutamate_analysis,
)

session_root = r"path\to\session_root"

config = GlutamateAnalysisConfig(
    alpha=0.05,
    tuning_method="fve",
    tuning_fve_mode="trace",
    sequence_slope_method="binned_peak",
)

tables = run_glutamate_analysis(
    session_root,
    config=config,
)

activation_summary = tables["activation_summary_table"]
tuning_summary = tables["tuning_summary_table"]
sequence_summary = tables["sequence_summary_table"]
```

By default, `run_glutamate_analysis` resolves paths under:

```text
analysis/derived/glutamate/
analysis/derived/glutamate/glutamate_analysis/
```

The output directory can be overridden with `output_dir=...`.

### Run tuning analysis using a precomputed activation summary

```python
from vip_slap2_analysis.glutamate.analysis import run_glutamate_tuning_analysis

results = run_glutamate_tuning_analysis(
    session_root,
    activation_summary=r"path\to\activation_summary_table.parquet",
)
```

This is useful when activation labels have already been computed in a batch run
and tuning should be recomputed with a new configuration.

## Calcium workflow

The calcium modules support soma/user-ROI extraction and QC for sessions with a
second indicator channel, such as RCaMP-like soma calcium data.

Key modules:

- `vip_slap2_analysis.calcium.extraction`
  - reconstructs calcium session traces;
  - aligns calcium traces to image/change/omission windows;
  - packages mean, single-trial, and sequence outputs.
- `vip_slap2_analysis.calcium.qc`
  - checks whether a session contains a processable calcium indicator;
  - computes ROI-level trace metrics;
  - evaluates ROIs against configurable quality thresholds;
  - writes calcium QC results.

Example QC entry point:

```python
from vip_slap2_analysis.calcium.qc import CalciumQcThresholds, run_calcium_qc

thresholds = CalciumQcThresholds()
qc_result = run_calcium_qc(asset, thresholds=thresholds)
```

Here, `asset` is a `SessionAssets` object describing the session paths and
metadata.

## Behavior alignment and validation

Behavior-related utilities are used to keep stimulus events and physiology on a
common clock. Current workflows include:

- loading corrected Bonsai/BonVision event tables;
- validating event-log structure and expected columns;
- extracting image, change, and omission events;
- auditing whether events fall within imaging epochs;
- using HARP and photodiode-derived timing information when available.

Common event classes:

- image presentations, usually `.tiff` values in the event log;
- `Change` events;
- `Omission` events;
- photodiode state rows, which should not be treated as stimulus identities.

## Metadata and manifests

Session metadata and asset discovery are represented with shared models such as
`SessionAssets`:

```python
from pathlib import Path

from vip_slap2_analysis.common.session import SessionAssets

asset = SessionAssets(
    session_id="826033_2026-02-17_13-13-55",
    subject_id=826033,
    session_dir=Path(r"path\to\session"),
    summary_mat=Path(r"path\to\SummaryLoCo-file.mat"),
    qc_dir=Path(r"path\to\session\qc"),
    derived_dir=Path(r"path\to\session\analysis\derived"),
    metadata={
        "dmd1_depth": 25,
        "dmd2_depth": 200,
        "session_type": "familiar",
    },
)

asset.ensure_dirs()
```

Manifest utilities in `vip_slap2_analysis.metadata` build session-level and
mouse-level summaries from on-disk assets and QC files. These manifest tables are
intended to make batch analyses auditable by exposing which files were present,
which QC outputs were generated, and how sessions were categorized.

## Morphology utilities

Morphology utilities support reconstruction-derived context for selected VIP
neurons. Current inputs include SNT/SWC reconstructions and measurement exports.
The plotting utilities are intended to produce clean vector graphics that can be
edited in Illustrator.

Typical morphology outputs include:

- XY, XZ, and ZY projections;
- smoothed display traces for anisotropic z sampling;
- cable length, branch-point, tip, and branch-order metrics;
- Sholl-style summaries when exported measurement tables are available.

## Generated analysis outputs

`run_glutamate_analysis` writes CSV files for all output tables and attempts to
write Parquet copies when Parquet support is installed. It also writes a metadata
JSON file describing the analysis configuration and input paths.

Main glutamate analysis outputs:

```text
glutamate_analysis/
├── activation_event_table.csv
├── activation_event_table.parquet
├── activation_summary_table.csv
├── activation_summary_table.parquet
├── tuning_per_image_table.csv
├── tuning_per_image_table.parquet
├── tuning_summary_table.csv
├── tuning_summary_table.parquet
├── sequence_position_table.csv
├── sequence_position_table.parquet
├── sequence_per_image_table.csv
├── sequence_per_image_table.parquet
├── sequence_summary_table.csv
├── sequence_summary_table.parquet
└── glutamate_analysis_metadata.json
```

High-level table meanings:

- `activation_event_table`
  - event-level pre/post response metrics and test statistics;
- `activation_summary_table`
  - per-synapse activation class summaries;
- `tuning_per_image_table`
  - per-synapse/per-image response and selectivity metrics;
- `tuning_summary_table`
  - per-synapse image-tuning summaries;
- `sequence_position_table`
  - response metrics as a function of repeated-image sequence position;
- `sequence_per_image_table`
  - sequence metrics grouped by synapse and image identity;
- `sequence_summary_table`
  - per-synapse sequence/adaptation/facilitation summaries.

## Coding and documentation standards

This repository is being documented iteratively while preserving working analysis
behavior. The intended standard is:

- keep executable logic stable unless a change is explicitly requested;
- prefer small, reviewable updates over sweeping rewrites;
- document public functions, classes, dataclasses, and non-obvious private helpers;
- make assumptions about array shape, time base, and indexing explicit;
- preserve scientific provenance by writing input paths and configuration values to
  output metadata;
- keep code PEP8-compliant where possible without obscuring scientific intent;
- avoid silent changes to analysis definitions, thresholds, or statistical tests.

For review-facing code, especially modules used to open, process, and analyze
physiology data, docstrings should answer:

1. What data structure does this function expect?
2. What shape and units are the key arrays?
3. What clock or sampling rate is assumed?
4. What biological or statistical quantity is being computed?
5. What is returned, and how should downstream code interpret it?

## Current development notes

- The package is actively evolving and several workflows still live primarily in
  notebooks.
- The editable install works as a local analysis package, but dependency metadata
  in `pyproject.toml` is currently minimal.
- Some modules are mature analysis entry points, while others are scaffolding for
  future packaging or processing workflows.
- Derived-data schemas should be treated as part of the analysis contract. If a
  table column, `.npz` field, or event-label convention changes, update this
  README and the relevant docstrings together.
