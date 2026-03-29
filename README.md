# vip-slap2-analysis

`vip-slap2-analysis` is a notebook-friendly Python repository for loading, preprocessing, quality-controlling, and analyzing SLAP2-based datasets collected in the VIP Synaptic Dynamics project.

The codebase is organized around a session-centric workflow spanning several data modalities:

- **glutamate imaging** from SLAP2 experiment summaries
- **somatic Ca2+ traces** extracted from user-defined ROIs in `SummaryLoCo**.mat`
- **behavior / HARP alignment** for passive visual stimulation sessions
- **voltage imaging** utilities under active development
- **morphology** support for Fiji / SNT reconstructions of traced neurons

## Contents

- [Experimental design with SLAP2](#experimental-design-with-slap2)
- [Pipeline](#pipeline)
- [Data modalities & formats](#data-modalities--formats)
- [Metadata](#metadata)
- [Analysis, figures, and findings](#analysis-figures-and-findings)
- [Morphology](#morphology)
- [Repository layout](#repository-layout)
- [Install (editable)](#install-editable)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Experimental design with SLAP2

The repository is intended for experiments in which VIP interneuron physiology is measured during passive visual stimulation and related post hoc analyses are performed on the same session. A common starting point is a session directory containing:

- MATLAB-derived `SummaryLoCo**.mat` files from SLAP2 preprocessing
- Bonsai event logs for visual stimulus timing
- HARP registries containing acquisition clock and photodiode signals
- optional morphology reference volumes and traced neuron reconstructions

## Pipeline

At a high level, the repository supports the following staged workflow:

1. **Session resolution and loading**
   - locate summary files, behavior logs, HARP data, and derived outputs
2. **Physiology preprocessing / alignment**
   - align Bonsai event times to HARP time
   - identify valid epochs and acquisition intervals
3. **Quality control**
   - quantify session quality, synapse quality, and Ca2+ ROI quality
4. **Stimulus-aligned extraction**
   - generate stimulus-centered outputs for image, change, and omission events
5. **Analysis and plotting**
   - perform per-session and pooled analyses with shared plotting utilities
6. **Morphology integration**
   - ingest SNT / Fiji reconstructions, compute morphometry, and generate publication-quality graphics

## Data modalities & formats

### Glutamate

Primary source:
- `SummaryLoCo**.mat`

Typical contents accessed through `vip_slap2_analysis.glutamate.summary.GlutamateSummary`:
- trial-resolved synapse traces
- trial validity and summary images per DMD
- experiment metadata
- user-defined ROI traces

### Calcium

Current calcium support is centered on user-defined soma ROIs embedded in the glutamate experiment summary. Existing code covers extraction and QC for sessions that include an RCaMP channel.

### Behavior / HARP

Key session files:
- `bonsai_event_log.csv`
- `VCO1_Behavior.harp`
- derived `photodiode.pkl`
- optional saved `HARP_df.csv`

These support Bonsai-to-HARP correction and subsequent stimulus alignment.

### Morphology

Morphology inputs are expected to come from Fiji / SNT tracing workflows and may include:
- `.swc`
- `.traces`
- `SNT_Measurements.csv`
- `QuickMeasurements.csv`
- `Sholl_Table-*.csv`
- traced reference volume `.tif`

The morphology subpackage currently establishes the architectural foundation for analysis, rendering, and export.

## Metadata

The repository uses session-aware path handling and is designed to work with registry-style metadata tables for tracking subjects, sessions, and modality-specific assets.

Important metadata themes in the project include:
- subject / session identity
- DMD identity
- modality availability (glutamate, calcium, morphology, voltage)
- acquisition and processing provenance
- physical calibration information such as voxel size

## Analysis, figures, and findings

The repository is structured so that reusable analysis code lives in `src/vip_slap2_analysis`, while exploratory and session-specific work can be done from Jupyter notebooks in `notebooks/`.

Shared plotting utilities currently live under:
- `vip_slap2_analysis.plotting.plot_utils`
- `vip_slap2_analysis.plotting.qc_plots`

The intended direction is to keep figure generation reproducible and move stable logic out of notebooks into the library.

## Morphology

Morphology support is being added for traced VIP neurons reconstructed in Fiji with the SNT plugin.

Planned morphology capabilities:
- discover morphology assets from a reconstruction folder
- open and validate SWC + SNT tables
- compute morphometric summary tables
- generate XY / XZ / YZ / 3D morphology figures
- export clean SVG and PDF files for Illustrator refinement
- support display-only smoothing for visually stepped arbors caused by anisotropic z sampling

See:
- `docs/morphology_architecture.md`
- `docs/morphology_preprocessing_and_tracing.md`
- `notebooks/morphology/README.md`

## Repository layout

```text
src/vip_slap2_analysis/
    behavior/
    calcium/
    common/
    glutamate/
    io/
    morphology/
    plotting/
    utils/
    voltage/

notebooks/
    behavior/
    glutamate/
    metadata/
    morphology/
    qc/
    voltage/

docs/
    morphology_architecture.md
    morphology_preprocessing_and_tracing.md
```

## Install (editable)

```bash
python -m pip install -e .
```

## References

External tools and resources relevant to this repository include Fiji, SNT, HARP-based behavioral acquisition workflows, and the MATLAB preprocessing code used to generate SLAP2 experiment summaries.

## Acknowledgements

This repository supports the VIP Synaptic Dynamics project and related analysis workflows for SLAP2-based physiology and morphology datasets.
