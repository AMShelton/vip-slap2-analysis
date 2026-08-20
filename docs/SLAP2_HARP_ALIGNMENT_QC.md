# SLAP2 ↔ HARP acquisition-clock alignment QC

## Alignment model

HARP is the experiment-time authority. Two independent hardware signals are used:

1. **Photodiode**: maps Bonsai stimulus/event timestamps onto HARP time.
2. **DI3**: records an acquisition-related SLAP2 hardware clock. Large DI3 gaps define imaging acquisition epochs.

Processed physiology is reconstructed independently within each accepted acquisition epoch. The current alignment policy preserves the nominal physiology sampling interval and clips unsupported acquisition tails; it does **not** stretch physiology to force agreement with HARP epoch duration.

## Hardware-clock audit results

### Modern voltage example — 863774_2026-08-07_13-29-18

- Raw HARP DI3 rising edges: **79,309**.
- Raw SLAP2 DMD1 cycles recorded across source epochs: **79,308**.
- Session difference: **+1 HARP edge** (~0.0013%).
- Stable accepted interior epochs agree to **0–1 cycle**.
- DMD1 uses 227 lines/cycle. Long-train HARP cadence corresponds to ~227.001 line periods/pulse (~5.25 ppm period error relative to DMD1 cycle cadence).
- Large first/last accepted-epoch differences are localized acquisition-boundary tails/fragments, not progressive clock drift. The reconciler clips these unsupported tails.

### Modern dual-color glutamate + calcium example — 834788_2026-03-02_10-18-42

- HARP DI3 rising edges: **147,970**.
- SLAP2 acquisition metadata requests **147,970 DMD1 cycles** for the 1,800 s acquisition.
- DMD1 uses 130 lines/cycle; long-train HARP cadence corresponds to ~130.001 line periods/pulse (~7.08 ppm period error).
- The uploaded bundle contains only the first raw `.dat` chunk, so the complete raw-data cycle count cannot be independently reconstructed from file sizes in this audit. Re-running the patched `summarize_LoCo.m` on the full session will record `MultiDataFiles.numCycles`, providing the same authoritative actual-cycle test used for voltage.

### Historical glutamate example — 803496_2025-07-25_13-02-10

- HARP DI3 rising edges: **88,128**.
- DMD1 acquisition parse plan: 210 lines/cycle.
- HARP long-train cadence corresponds to **~211.002 line periods/pulse**, i.e. a 211-line hardware cadence rather than the 210-line DMD1 cycle clock.
- Therefore direct `HARP pulses == DMD1 cycles` comparison is invalid for this historical session and should not be interpreted as dropped cycles.
- The supplied manifest contains 89 DMD1 chunks starting at cycles 0, 1000, ..., 88000. Because file sizes are not in the manifest, the exact final partial-chunk cycle count is not independently available from the manifest alone.

## Code changes

### MATLAB: `matlab/summarize_LoCo.m`

For SLAP2 sessions the summary now persists raw acquisition metadata before localization/extraction:

- `exptSummary.trialEpoch`
- `exptSummary.trialFilePrefix`
- `exptSummary.epochTable`
- `exptSummary.nEpochs`
- `exptSummary.multiEpochAcquisition`
- `exptSummary.acquisitionMetadataSchemaVersion`
- `exptSummary.dmd(dmd).metadata`
- `exptSummary.dmd(dmd).epochs(epoch)`
  - `nCycles`
  - `totalNumLines`
  - `linesPerCycle`
  - `durationSec`
  - `cycleRateHz`
  - first raw `.dat` path and acquisition metadata

Actual source-epoch duration is defined as `totalNumLines / lineRateHz`, matching the voltage extractor.

### Python summary/alignment

`GlutamateSummary` now reads the same raw per-DMD/per-epoch metadata pattern used by `VoltageSummary` and exposes:

- `get_dmd_metadata()`
- `get_dmd_epoch_metadata()`
- `get_dmd_epoch_durations_sec()`
- `get_line_rate_hz()`
- `get_dmd_cycle_rate_hz()`

Glutamate and calcium now pass raw source acquisition durations to the shared `reconcile_trial_epochs()` function. Multi-epoch sessions with strict matching require explicit source epoch labels.

### Shared clock QC

`common/clock_qc.py` compares HARP DI3 cadence/counts against SLAP2 acquisition metadata. It first infers the clock relationship instead of assuming that DI3 is always the DMD1 cycle clock:

- `direct_dmd_cycle_clock`
- `integer_line_cadence_not_dmd_cycle`
- `noninteger_or_unresolved_cadence`

Count-for-count QC is only enabled for a direct DMD-cycle relationship. Cadence is estimated over long contiguous pulse trains to average HARP timestamp quantization while excluding acquisition gaps and short fragments.

The same clock-QC block is now written into glutamate, calcium, and voltage extraction metadata/QC.

### Behavior preprocessing

When pulse-train epoch detection is used, `imaging_epochs.csv` now retains the HARP `n_pulses` and source pulse-train epoch index for each accepted epoch. The full detector diagnostics remain in `di3_pulse_train_detection.json`.

## Reprocessing / migration

Old derived glutamate/calcium files remain readable. To obtain the stronger hardware-level QC:

1. Re-run the patched `summarize_LoCo.m` on the full raw SLAP2 session to populate actual acquisition-cycle metadata.
2. Re-run behavior preprocessing with the pulse-train detector to populate modern DI3 diagnostics and per-epoch pulse counts.
3. Re-run glutamate/calcium extraction. The outputs will use the same nominal-rate, epoch-aware reconciliation policy as voltage and will include `slap2_harp_clock_qc`.

Historical sessions whose DI3 cadence is not a DMD cycle clock are retained and characterized rather than falsely failed.
