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

### Cross-acquisition check through the shared FPGA counter

Every acquisition's `.dat` header stores `referenceTimestamp`, the FPGA tick at its start,
and the line headers carry ticks on the same counter. If the counter runs continuously for
the SLAP2 session, the *absolute* FPGA time of every cycle of every acquisition is known,
and all acquisitions share one affine relation HARP = a·FPGA + b. `common/fpga_cross_epoch.py`
uses this as timing evidence independent of pulse *counting*:

* leave-one-out: fit the map on the pulses of the other identified acquisitions, predict the
  HARP time of this acquisition's cycle `k`, compare with its first observed pulse. A wrong
  count is off by a whole period (9–22 ms) against a fit residual of tens of µs, so the test is
  decisive; `all_identified_offsets_confirmed` is only `True` when the single fit also holds
  across acquisitions (`fpga_counter_continuous`).
* for `inconsistent`/`ambiguous` acquisitions in the same session, the fit says where the
  cycles without pulses fall (before the first pulse, inside the train, after the last pulse).

Single-acquisition sessions have nothing to fit against; the check does not help them.

Result on the voltage cohort (2026-11-06): all 12 multi-acquisition corrected sessions
confirmed to within 0.04 cycles (fit 17–50 µs RMS). The registration itself now uses the same
map as a *second pass* (`_resolve_epochs_via_shared_fpga`): an epoch that counting left
`inconsistent`/`ambiguous` is placed when its first pulse is within 0.25 cycles of a predicted
cycle start and the predicted cycles inside its train equal the pulses; such epochs carry
`anchor = "fpga_cross_epoch"` and are excluded from the leave-one-out fit. Pairing also rejects
DI3 trains whose cadence differs >2 % from the acquisition's cycle period (preview acquisitions
at 75 ms).

**Integer-line DI3 cadence (2025 glutamate, 826032_2026-02-02).** In these sessions DI3 fires every
`L` lines of the FPGA line clock (e.g. 211) rather than once per DMD cycle, so pulse counting cannot
give cycle offsets. They are still line-locked: with a head anchor, pulse `j` marks line `j·L` and the
first pulse marks line 0 = cycle 0. `_resolve_integer_line_epochs` fits the pulses to the FPGA times
of those lines (slope must be 1 within 200 ppm — a wrong `L` is off by ~1/L — and residual < 200 µs
within step-free segments), then maps every cycle start FPGA→HARP by interpolation between the
measured pulses and stores per-cycle times (`timing_source = di3_integer_line_cadence_via_fpga_lines`;
nominal line times when raw `.dat` headers are absent). The pulse phase within the first `L` lines
is an assumption, so the status is `identified_with_assumptions`; `scripts/run_cycle_registration.py`
makes it by default (`--no-head-assumption` leaves such epochs ambiguous). Without a head anchor the
epoch stays ambiguous. In every 2025 session seen so far `L` = DMD1's lines-per-cycle + 1 (211/210,
230/229, 147/146 …), i.e. the 2025 DI3 was a line counter slipping one line per DMD1 cycle, not the
cycle clock. One session (804730_2025-07-30) has a one-sided disturbance: most pulses sit on the grid, a
minority are delayed by up to ~1.8 ms (residual p5/p50/p95 −357/−311/+1506 µs about the all-pulse
fit). The resolver identifies the on-grid majority from the residual's median/MAD, drops the displaced
pulses and interpolates between the rest, so the timebase is not biased by the mean delay. Only when
no on-grid majority exists does it fall back to the affine fit over all pulses
(`timing_source = …affine_fit_jittery_pulses`, origin precise to jitter/√n only if the jitter is
zero-mean) with a wider anchor tolerance.

**Steps in the HARP-vs-FPGA relation.** Two early-2026 sessions (826031_02-02, 02-11) contain a
single −2.3/−2.5 ms jump of DI3-minus-FPGA inside one acquisition (`scripts/plot_harp_fpga_residuals.py`).
`detect_relation_steps` finds such jumps (> 200 µs between consecutive pulses); the DMD2 transfer maps
FPGA→HARP by interpolation between the measured DMD1 pulses (`piecewise_fpga_to_harp`, exact at every
pulse, affine extrapolation outside) and judges the fit per step-free segment; cross-epoch fits remove
the steps first and re-apply them to predictions. Steps are stored per epoch in `relation_steps`.
`scripts/check_fpga_cross_epoch.py` writes `qc/sync/fpga_cross_epoch_DMD<n>.json` per session
and a cohort CSV.

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
4. `python scripts/check_fpga_cross_epoch.py --summary-xlsx … --indicator ASAP7y ASAP8 --out <fpga.csv>`
   for every multi-acquisition session: a corrected offset counts as *independently confirmed*
   only when `all_identified_offsets_confirmed` is `True` here.
5. `python scripts/build_sync_status_table.py … --registration-csv <status.csv>` for the
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
