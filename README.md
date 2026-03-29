# vip-slap2-analysis

Utilities for loading, processing, quality controlling, analyzing, and visualizing multimodal VIP SLAP2 datasets, including behavior, glutamate, calcium, voltage, and now morphology reconstructions.

## Contents

- [Experimental design with SLAP2](#experimental-design-with-slap2)
- [Pipeline](#pipeline)
- [Data modalities & formats](#data-modalities--formats)
- [Metadata](#metadata)
- [Analysis, figures, and findings](#analysis-figures-and-findings)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [Install (editable)](#install-editable)

## Experimental design with SLAP2

This repository is organized around VIP synaptic dynamics experiments collected with SLAP2 in mouse visual cortex. The codebase is built to support alignment between behavior and physiology, source-extracted glutamate and calcium traces, session-level QC, and figure generation for downstream biological interpretation.

The main experimental context currently includes:

- passive change-detection style visual stimulation
- glutamate imaging with iGluSnFR
- calcium imaging with RCaMP when present
- SLAP2 reference volumes and reconstruction-derived morphology for selected sessions
- session-level metadata pulled from spreadsheet registries and on-disk assets

## Pipeline

The package is developing toward a modality-aware but shared pipeline:

1. **Session discovery / registry**
   - resolve session directories and expected assets
2. **Behavior preprocessing**
   - load BonVision / Bonsai and HARP outputs
   - correct time bases and create event tables on a common clock
3. **Physiology extraction**
   - glutamate and calcium extraction aligned to behavior epochs
4. **QC generation**
   - metric tables and saved plots per session / modality
5. **Analysis and figure generation**
   - stimulus-aligned summaries, depth-aware comparisons, and publication-ready plots
6. **Morphology**
   - load SNT/SWC reconstructions, compute morphometrics, and export vector-friendly plots

Current morphology code lives in `src/vip_slap2_analysis/morphology/` and supporting documentation lives in `docs/morphology/`.

## Data modalities & formats

The repo currently works with several kinds of inputs:

### Behavior
- Bonsai / BonVision event logs (`bonsai_event_log.csv`)
- HARP extracts and photodiode traces

### Glutamate
- `SummaryLoCo*.mat` and related SLAP2 outputs
- source extraction products and aligned stimulus-response tables

### Calcium
- extracted ROI or user-ROI tables from SLAP2 summary outputs
- detrended / dF/F-like processed traces and QC summaries

### Voltage
- voltage imaging summary / preprocessing code and downstream trace analysis

### Morphology
- SNT exported `.swc` reconstructions
- `.traces` project files
- `QuickMeasurements.csv`
- `SNT_Measurements.csv`
- `Sholl_Table*.csv`
- prepared TIFF stacks used for reconstruction

## Metadata

Session-level metadata is handled through a mix of on-disk discovery and registry tables. Existing code in `vip_slap2_analysis.io.session_registry` and `vip_slap2_analysis.common.session` provides the current foundation for locating session assets and keeping derived outputs organized.

For morphology, the code is designed to keep a tight link between:

- session ID
- source image stack
- reconstruction outputs
- user notes / provenance
- exported metrics and figures

## Analysis, figures, and findings

The long-term goal of this repo is not just file IO, but interpretable analysis products. Existing and in-progress analysis targets include:

- stimulus-aligned glutamate and calcium response summaries
- session-level QC tables and saved figures
- depth-aware and DMD-aware summaries
- morphology-derived descriptors such as cable length, branch points, tips, branch order, and Sholl structure
- clean batch-rendered figures suitable for review, talks, and Illustrator polishing

Morphology plotting utilities currently support:

- XY / XZ / ZY projections
- display-only smoothing for anisotropic z sampling
- PDF / SVG / PNG export for downstream figure editing

## References

Primary external tools and concepts currently reflected in this repository include:

- Fiji / ImageJ
- SNT (Simple Neurite Tracer / SNT)
- Sholl analysis workflows
- SLAP2 preprocessing and summary outputs
- Allen Institute behavior and metadata tooling used alongside these analyses

## Acknowledgements

This codebase supports ongoing work on VIP synaptic dynamics and related multimodal physiology/morphology analysis. It reflects ongoing development across analysis notebooks, scripts, and reusable package code for the Podgorski / Svoboda context.

## Install (editable)

```bash
python -m pip install -e .
```

## Minimal morphology example

```python
from vip_slap2_analysis.morphology import (
    load_snt_bundle,
    compute_basic_metrics,
    plot_morphology_triptych,
)

bundle = load_snt_bundle(r"path\to\reconstruction_folder")
metrics = compute_basic_metrics(bundle.tree)
print(metrics)

fig, axes = plot_morphology_triptych(
    bundle.tree,
    smooth=True,
    save_path_stem=r"path\to\figures\cell01_triptych",
)
```
