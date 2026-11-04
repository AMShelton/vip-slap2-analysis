# Retrospective SLAP2 ↔ HARP synchronization and modern summary loading

Branch `feature/retro-sync-giant-h5`, 2026-11-04. Companion to `SLAP2_HARP_ALIGNMENT_QC.md`
(hardware-clock QC) — this document covers the *absolute cycle origin*, the measured
timebase, and the GIAnT / Voltex HDF5 readers.

## 1. The problem

Every modality reconstructs its session trace by concatenating per-trial arrays and
placing the **first retained source sample of each imaging epoch at
`imaging_epochs.csv::start_time`**, i.e. at the first DI3 rising edge of the matching
HARP pulse-train epoch (`common.epoch_alignment.build_epoch_aware_timebase`, called from
`common.alignment.reconstruct_dmd_session_traces`,
`voltage.postprocess.reconstruct_voltage_dmd_session_traces` and
`calcium.extraction._reconstruct_ca_session_traces`). Samples then advance at the
nominal `.meta` rate and the tail is clipped to the behavior epoch.

That is only correct when the HARP recording covers the start of the SLAP2 acquisition.
Between 2026-07-23 and 2026-10-27 the workflow started SLAP2 first, so the first
*recorded* pulse is cycle `k > 0`. The processed products of those sessions are offset
by `k` cycles (≈ 20.7 ms per cycle for 227-line voltage acquisitions). No code path
certified `k`; `metadata/t03_*` audited it read-only.

A second, smaller effect: the HARP clock runs ≈ +6 ppm relative to the SLAP2 FPGA /
`.meta` line rate (measured on the 2026-10-27 reference: +5.6 to +6.5 ppm). A
nominal-rate ramp anchored at the first pulse is therefore ≈ 11 ms late at the end of
a 30-min session (half a voltage cycle, ≈ 2 glutamate frames).

## 2. Independent evidence available in the raw data

| Source | What it gives | Reader |
|---|---|---|
| raw `.dat` line headers (`slap2/dynamic_data/*_DMD<n>-TRIAL*-CYCLE-*.dat`) | 64-bit FPGA timestamp (200 MHz), `lineNumber`, `startOfAcq`/`startOfCycle` flags for every line; total cycle count `N` per DMD; source-side dropped cycles | `io/slap2_dat.py` (header-only, memmap gather) |
| `.meta` | `lineRateHz`, `linesPerCycle`, `acqDurationCycles`, `fpgaTimeReference` (SLAP2 PC wall clock) | `io/slap2_dat.read_meta_summary` |
| HARP `Behavior_32.bin` (DigitalInputState) | DI3 rise/fall times, initial state, **recording start** | `io/harp_bin.read_digital_input_edges` |
| HARP `Behavior_44.bin` (AnalogData) | recording start/end bounds, photodiode; first sample is behavior time zero | `io/harp_bin.read_harp_register` |

Verified on the reference session `test_subject_2026-10-27_09-00-55`:

* DMD1 7,178 cycles ↔ 7,178 DI3 rises (1:1); DI3 rise vs FPGA cycle-start residual 14 µs RMS
  (HARP tick 32 µs) ⇒ DI3 rises at the cycle start line (phase ≈ 0 for this wiring).
* DMD1 and DMD2 `.dat` files share `referenceTimestamp` and the first cycle timestamp
  ⇒ DMD2 cycle times are known exactly on the same FPGA clock (no separate DI line needed).
* HARP/FPGA rate ratio 1 + 6.5e-6.
* `.meta` `fpgaTimeReference`, launcher logs and HARP folder stamps agree only to seconds
  ⇒ wall clocks cannot pick an integer cycle.

## 3. Identifiability (what can and cannot be corrected)

Let `N` = source cycles (from `.dat` headers or extractor line counts), `n` = recorded DI3
pulses (+ inferred dropped pulses from internal gaps > 1.5 periods), `k` = cycle index of
the first recorded pulse.

| HARP coverage | Evidence | Result |
|---|---|---|
| head margin (DI3 low ≥ 10 cycles / 0.5 s before first pulse) | head anchor | `k = 0`, `N − n` terminal cycles unrecorded |
| tail margin (recording continues ≥ 10 cycles after last pulse) | tail anchor | `k = N − n` |
| both | consistency check | `k = 0` iff `n = N`, else `inconsistent` |
| neither | — | `ambiguous`: `k ∈ [0, N − n]`; **not identifiable from DI3** |
| `n > N` | — | `inconsistent` (DI3 is not this DMD's cycle clock, or wrong source epoch) |
| DI3 cadence ≠ one pulse per cycle (`integer_line_cadence_not_dmd_cycle`, e.g. 2025 glutamate at 211 lines) | — | counting undefined; head anchor gives `k = 0` only as `identified_with_assumptions` |

Reproduced on the reference pulses (`scripts/make_sync_qc_figures.py`,
`reference_truncation_benchmark.csv`): remove first 150 pulses → `k = 150` (tail anchor);
remove last 160 → `k = 0` (head anchor); remove both → 311 candidates, `ambiguous`;
internal dropped pulses are counted, not mistaken for an offset.

## 4. Implementation

* `common/cycle_registration.py` — `register_di3_to_source_cycles` → `CycleRegistration`
  (schema `vip_slap2_cycle_registration_v1`): start order + evidence, per-epoch
  `anchor`, `cycle_index_offset`, `candidate_offset_range`, `missing_initial/terminal_cycles`,
  `n_internal_missing_pulses`, HARP/source `rate_ratio` and `period_error_ppm`,
  `residual_rms_s`, `status ∈ {identified, identified_with_assumptions, ambiguous,
  inconsistent, unavailable}`, `safe_for_physiology`, `limitations`, `inputs`.
  `run_cycle_registration(asset, dmd=…, dynamic_data_dir=…)` reads the raw files and
  saves `analysis/qc/sync/cycle_registration_DMD<n>.json` (+ `_pulses.npz`).
  `EpochCycleRegistration.cycle_times_harp_s()` gives measured HARP time per cycle
  (labels 0 measured, 1/2 extrapolated head/tail, 3 interpolated gap);
  `source_lines_to_harp_time` maps 1-based source lines (within-cycle at the HARP-corrected
  line period).
* `common/epoch_alignment.py` — `reconcile_trial_epochs(source_epoch_leading_samples=…)`
  skips leading source samples per source epoch (reported in
  `trial_source_start_samples`); `build_epoch_aware_timebase(epoch_sample_times_sec=…,
  epoch_sample_time_labels=…)` accepts measured per-sample times and **raises** if their
  first sample is > 2 ms from the behavior epoch start (origin mismatch guard).
  Defaults are unchanged.
* `common/alignment.reconstruct_dmd_session_traces`,
  `voltage/postprocess.reconstruct_voltage_dmd_session_traces`,
  `calcium/extraction._reconstruct_ca_session_traces` — `cycle_registration=` applies an
  identified registration (needs per-trial source lines: `GlutamateSummary.get_trial_source_lines`
  = `E.frameLines`, `VoltageSummary.get_trial_source_lines` = trial line ranges,
  `GiantExperimentSummary`/`VoltexSummary` = `frame_line_idxs`). Refusal reasons are recorded in
  `metadata['cycle_registration']`; `timebase_method_by_epoch` says
  `di3_pulse_registered` vs `nominal_rate_anchored_first_pulse`.
* Extraction entry points (`glutamate`, `calcium`, `voltage`) gain
  `cycle_registration="auto"|None|CycleRegistration|{dmd: …}`,
  `require_identified_cycle_alignment` (fail closed) and `summary_preference`; QC JSON
  carries `summary_provenance`, `cycle_registration`, `trial_source_start_samples`,
  `sample_time_label` (voltage session traces).
* Corrected event log: `common.alignment.resolve_corrected_event_log` prefers
  `qc/behavior/bonsai_event_log_corrected.csv` (written when
  `save_corrected_in_place=False`) over the raw log.

### Modern summary products

* `io/giant_summary.GiantExperimentSummary` — GIAnT `experiment_summary.h5` (+ `trial_table.h5`),
  schema verified against `saveExperimentSummaryH5.m` / `saveStructToH5.m` (dims reversed in
  h5py, `/row_major = 0`, channel axis = full acquisition channel list in original order with
  `params/activityChannel` 1-based, `frame_line_idxs` 1-based global lines, no timebase stored).
  Exposes the `GlutamateSummary` API used by extraction/QC. Acquisition metadata comes from raw
  `.meta`/`.dat` headers when `dynamic_data_dir` is given; otherwise clock QC reports unavailable.
* `voltage/voltex_summary.VoltexSummary` — Voltex `VoltageSummary.h5` format 0.2.x (also written
  as `Voltage/experiment_summary.h5`); `raw_f`/`f0`/`dff` are exposed via `trace_kind`
  (default `raw_f` so dF/F is computed here exactly as for legacy products); `motion_censor`
  optional as discard mask.
* `io/summary_formats.sniff_summary_h5` — content-based classification (GIAnT vs Voltex) used by
  the registry; `io/summary_loader.load_glutamate_summary` / `load_voltage_summary` apply
  **modern (schema-validated) → legacy → explicit error** and return provenance.
* `io/session_registry.resolve_assets(…, summary_preference="modern"|"legacy")` records
  `summary_h5`, `summary_h5_candidates`, `trial_table_h5`, `summary_path`, `summary_format`,
  `summary_candidates`, and `slap2.dynamic_data_dir`. Legacy fields are unchanged.
* Calcium keeps the legacy reader (soma processing is implemented on `exptSummary/E`).
* `scripts/compare_modern_legacy_summary.py` is the numerical gate before trusting a modern file
  over a legacy one on real data (no real `experiment_summary.h5` was available in this sandbox).

## 5. Operating procedure (historical voltage session)

1. `python scripts/run_cycle_registration.py --summary-xlsx <VIP_SD_summary.xlsx> --session <id> --dmd 1 2 --out <status.csv>`
   on a VAST-connected machine (reads only HARP bins and `.dat` headers).
2. Inspect `qc/sync/cycle_registration_DMD*.json`: `status`, `acquisition_start_order`,
   `anchor`, `cycle_index_offset`, `candidate_offset_range`, `period_error_ppm`, `limitations`.
3. `VIP_SD_ProcessData_EpochAware.ipynb` with `REQUIRE_IDENTIFIED_CYCLE_ALIGNMENT = True`:
   `ambiguous`/`inconsistent` sessions are skipped and listed; identified ones are re-extracted
   with the registered timebase. Re-running with `cycle_registration=None` reproduces the
   historical products for before/after comparison.
4. `python scripts/build_sync_status_table.py … --registration-csv <status.csv>` for the
   session-level classification (corrected / verified_no_offset / identified_with_assumptions /
   unresolved / pending / not_registrable).

## 6. Known limits

* DI3 ↔ cycle phase (rise at cycle start) is verified on the 2026-10-27 wiring only; the
  2025 glutamate sessions expose a 211-line cadence that is not a cycle clock.
* `identified_with_assumptions` (count-consistent, no coverage margin) is accepted only because
  `k = 0` is the only count-consistent offset; review such sessions.
* Both-ends-missing sessions cannot be corrected from DI3. Additional information that would
  resolve them: any event visible on both clocks (e.g. a SLAP2 photodiode/stim artifact in the
  raw lines), or HARP-side logging of the SLAP2 `hDOAcquiring` level (`/vDAQ0/D0.3`).
