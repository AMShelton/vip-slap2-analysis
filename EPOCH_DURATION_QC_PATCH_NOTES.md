# Epoch-duration QC update

## Policy

Physiology and behavior acquisition fragments are now eligible for downstream
analysis only when their duration is **greater than or equal to 30 seconds** by
default. The threshold is exposed as `min_epoch_duration_sec` (or
`min_epoch_duration` in behavior preprocessing) so it can be overridden
explicitly for development or legacy reprocessing.

Raw extraction remains lossless: `extractDendrites.m` and SummaryLoCo source
files should retain every acquisition epoch. Rejection occurs in Python QC and
reconstruction, where short fragments are labeled as rejected and excluded from
baseline estimation, dF/F calculation, ROI QC, event alignment, and response
products.

## Reconciliation rules

1. Behavior DI3 pulse-train blocks are detected before duration filtering.
   Accepted and rejected candidates are written to
   `di3_pulse_train_epoch_qc.csv` and the JSON diagnostics. Only accepted blocks
   are written to `imaging_epochs.csv`.
2. Source acquisition epochs are classified independently. For patched voltage
   summaries, duration is taken from per-DMD acquisition metadata
   (`totalNumLines / lineRateHz`). Trial-covered duration is the fallback for
   older voltage summaries and SummaryLoCo-derived paths.
3. A raw source/behavior epoch-count mismatch is not itself an error. Accepted
   source epochs are paired chronologically with accepted behavior epochs.
4. With `strict_epoch_match=True`, the number of accepted source and behavior
   epochs must match. A mismatch after filtering remains fatal.
5. Short source epochs receive analysis epoch label `0`, retained length `0`, and
   are absent from processed traces and event-response products.
6. Nominal sample spacing is preserved. If an accepted source epoch extends past
   its paired behavior interval, its terminal samples are clipped; traces are not
   linearly rescaled.
7. F0/dF/F is estimated independently within retained analysis epochs for voltage
   and calcium. No baseline operation crosses an acquisition restart.

## Modality-specific behavior

### Behavior

`process_behavior_session` defaults to a 30-second minimum. Pulse-train
candidate diagnostics preserve rejected short blocks even though
`imaging_epochs.csv` contains accepted intervals only.

### Voltage (ASAP7 and ASAP8)

Voltage QC and session reconstruction use `summary.trialEpoch` plus per-DMD epoch
metadata when available. Rejected epochs do not contribute trial/ROI metrics,
F0, dF/F, event means, or sequences. The raw extractor/behavior epoch-count
comparison is diagnostic; accepted-count mismatch is the strict failure.

### Calcium

Calcium session reconstruction applies the same reconciliation before building
session traces. Source trial-epoch labels are used when exposed by SummaryLoCo;
otherwise the code records a behavior-duration fallback. Epoch-scoped baseline
estimation is the default analysis-safe path.

### Glutamate

Glutamate extraction filters behavior events to accepted intervals and attempts
to infer source acquisition labels from explicit trial-table epoch fields or
acquisition filename-prefix changes. Short source fragments are excluded from
session reconstruction and source-only synapse QC when labels are available.
Per-trial SummaryLoCo dF/F is retained as provided; no downstream concatenated
baseline spans epochs.

## Backward compatibility

Old single-epoch voltage and SummaryLoCo files remain readable. When a source
summary has no recoverable trial-to-epoch labels, the pipeline uses accepted
behavior durations to assign trials and records the fallback in output metadata.
This fallback cannot independently identify a short physiology restart fragment;
source labels are therefore strongly preferred for multi-epoch data.

## Outputs and provenance

Processed metadata now records, where applicable:

- raw and accepted epoch counts;
- source and analysis trial-epoch labels;
- source epoch duration and duration basis;
- accepted/rejected status and discard reason;
- retained and discarded samples;
- source-to-analysis epoch mapping;
- minimum duration threshold;
- nominal-rate clipping policy.

## Validation performed

The repository test suite includes duration-threshold boundary tests, rejection
of short source fragments, accepted-count mismatch failure, nominal-rate tail
clipping, acquisition-duration metadata precedence, gap-preserving timebases, and
epochwise calcium/voltage baseline tests.

## Validation against the supplied ASAP8 session

For session `852835_2026-08-01_14-58-35`, the provided HARP DI3 data produced
five candidate pulse-train blocks: 969.391 s (accepted) and 5.247, 9.348,
14.391, and 5.323 s (rejected). The patched voltage summary exposed three raw
SLAP2 source epochs. Using acquisition metadata, DMD1 durations were 975.733,
9.348, and 7.291 s (DMD2 differed by less than 0.01 s). The reconciliation
therefore accepted source epoch 1, rejected source epochs 2 and 3, clipped the
accepted source tail to the 969.391 s behavior interval, and excluded trials 106
and 107. This is the intended result under the shared 30-second policy.
