# Multi-epoch physiology processing patch

## Why this patch was needed

SLAP2 line coordinates restart when an acquisition is stopped and restarted. The previous continuous-voltage extraction path opened only the first `MultiDataFiles` chain for each DMD and then attempted to slice every trial from that trace. In a multi-epoch session, later trial line ranges were therefore applied to the first epoch's samples. Downstream plotting could not recover the missing later acquisition.

The corrected processing contract is:

1. Behavior QC defines imaging epochs in HARP time (`qc/behavior/imaging_epochs.csv`).
2. Raw acquisition traces are extracted independently within each source epoch.
3. Trial-to-epoch labels remain explicit.
4. Downstream traces stay concatenated in acquisition-sample order, while their per-sample timebase jumps across imaging-off gaps.
5. Event windows crossing an acquisition restart are rejected.
6. Calcium and voltage baselines are estimated independently within each imaging epoch.

## MATLAB voltage extraction

The patched file is also included at `matlab/extractDendrites.m`.

Changes:

- Detects acquisition epochs from trial-table acquisition-prefix changes.
- Opens a separate `slap2.util.MultiDataFiles` chain for every DMD × epoch.
- Extracts every ROI independently for each epoch.
- Writes trial slices only from the trace belonging to that trial's epoch.
- Validates DMD geometry, parse-plan stability, ROI masks, and ROI order across epochs.
- Stores epoch source metadata in `summary.dmd(dmdIdx).epochs(epochIdx)`.
- Retains `/traces/trial_####` for event processing.
- Stores requested multi-epoch continuous output as:

  ```text
  /traces/epochs/epoch_0001/DMD1
  /traces/epochs/epoch_0001/DMD2
  /traces/epochs/epoch_0002/DMD1
  /traces/epochs/epoch_0002/DMD2
  ```

  Single-epoch continuous output remains under `/traces/continuous/DMD#`.

For multi-epoch voltage sessions, use `params.outputMode = 'trial'` or `'both'`. Event extraction intentionally requires trial datasets even when epoch-continuous datasets are also present.

## Shared epoch alignment

`common/epoch_alignment.py` and `common/alignment.py` now:

- validate finite, positive, ordered, non-overlapping HARP epochs;
- preserve imaging-off gaps in the explicit per-sample timebase;
- expose one-indexed `trial_epoch` and `sample_epoch` labels;
- report acquired duration, outer session span, and imaging-gap duration separately;
- support epoch-local duration scaling (`auto`, `always`, or `never`);
- reject event snippets that cross epoch labels or contain a timebase jump.

## Glutamate

- Session reconstruction now uses the behavior-derived epoch timebase.
- Event/sample lookup uses explicit HARP sample times rather than a single `(event - session_start) × rate` conversion.
- Event windows that touch an imaging restart are excluded.
- dFF provenance is recorded explicitly. Glutamate dFF remains a per-trial SummaryLoCo quantity: a stored `dFF/<mode>` dataset is used when present; otherwise it is safely derived as `dF/F0` using that trial's F0 dataset. No new baseline estimator spans epochs.
- Extraction QC reports trial-to-epoch assignment, per-epoch duration errors, acquired duration, session span, and imaging-gap duration.

## Calcium

- Calcium reconstruction starts from motion-corrected fluorescence (`ca_mc`) when `dff_scope='epoch'`.
- F0 is estimated independently within each acquisition epoch.
- dFF is computed after epochwise reconstruction as `(F - F0) / F0`.
- The legacy per-trial dFF route remains available with `dff_scope='trial'`.
- Calcium QC evaluates the same epochwise dFF definition used by extraction.
- QC/extraction metadata record baseline method, parameters, epoch labels, duration errors, and gap duration.

## Voltage (ASAP7 and ASAP8)

- Trial reconstruction requires extractor `trialEpoch` metadata for multi-epoch sessions.
- Static and robust F0 models are independent for every acquisition epoch by default (`f0_scope='epoch'`).
- ASAP7-like quenched indicators retain inverted polarity, `(F0 - F) / F0`.
- ASAP8-like brightening indicators retain standard polarity, `(F - F0) / F0`.
- Full-session H5 outputs now include `timebase_sec` and `sample_epoch`, in addition to `raw_f`, `f0`, and `dff`.
- Voltage QC compares extractor epoch count/labels with behavior QC and fails fast on disagreement.
- The reader recognizes epoch-local continuous datasets for inspection, but the event-processing path requires trial datasets for multi-epoch sessions.

## Updated deployment notebook

`notebooks/qc/VIP_SD_ProcessData_epochaware.ipynb` now:

- asserts that the imported package exposes all multi-epoch parameters;
- enables strict epoch matching for glutamate, calcium, and voltage;
- uses epoch-scoped calcium and voltage baselines;
- performs a cross-modality epoch integrity audit before physiology extraction;
- refuses multi-epoch voltage processing when patched trial datasets are absent;
- plots acquisition epochs as separate line segments on the HARP timebase;
- documents the required reprocessing order.

## Reprocessing order

1. Regenerate behavior QC and `qc/behavior/imaging_epochs.csv`.
2. Re-run `matlab/extractDendrites.m` from the raw voltage `.dat/.meta` files with `outputMode='trial'` or `'both'`.
3. Re-run voltage QC and verify the epoch-integrity section passes.
4. Re-run glutamate/calcium processing as needed.
5. Re-run voltage dFF, event-aligned means, and sequence extraction with the updated notebook.

Do not reuse the old multi-epoch voltage trace H5: the later raw epoch was not present in that extraction and cannot be repaired downstream.

## Validation performed here

- `python -m compileall -q src`
- `PYTHONPATH=src pytest -q`: **11 passed**
- Added tests for gap-preserving timebases, gap-crossing event rejection, epochwise calcium dFF, epochwise voltage F0, and epoch-continuous voltage H5 discovery.
- Updated notebook code cells were syntax-checked after removing Jupyter magics.

MATLAB/SLAP2 runtime execution was not available in this environment. The MATLAB patch was reviewed statically and should be tested on one known multi-epoch session before batch reprocessing.
