# T0.3: correction rollout, in order

**Scope:** Registry A0–B2; read-only start-order census first. Preserve legacy and modern photodiode correction independently of SLAP2 line-to-HARP timing.

## Stage 1 — audited evidence

1. For each registry session enumerate dynamic DMD `.meta` files, choose the acquisition associated with the BonVision workflow using timestamps and duration (flag multiple plausible candidates).
2. Parse MAT-file creation header and acquisition filename, record the separate behavior workflow-start time and the HARP recording-directory timestamp from `behavior/launcher_metadata/launcher.log`.
3. Where already available, add measured HARP DI3 pulse counts, pulse period, first-edge delay from HARP recording onset, and accepted epochs. Never use counts alone to assert dropped pulses.
4. Estimate start order with explicit bounds for **clock skew across PCs, MAT-file-to-hardware delay, and HARP-folder-to-first-register delay**. These are analyst assumptions, not experimentally established precision.
5. Calculate a *candidate set* of initial source-cycle indices and compare the total overhang with pulse counts. Report cases with missing/ambiguous evidence separately.
6. Review at least one `HARP first` and one `SLAP2 first` positive-control example against raw HARP/metadata, then freeze an audit CSV and a manually reviewed offset file.

## Stage 2 — code updates (separate PR after Stage 1 review)

- `common/epoch_alignment.py`: accept explicit, provenance-tagged `first_harp_di3_source_cycle` per source epoch. Trim correct HEAD and TAIL of source arrays; never rely only on behavior duration or always trim the tail. Match epochs using acquisition IDs and HARP periods rather than purely chronological positions when possible.
- `voltage/postprocess.py` and glutamate/calcium consumers: preserve/propagate source indices (e.g. `frame_line_idxs` when available). Build timestamps by interpolating measured DI3 times at the **correct source-line coordinates**, not by restarting a nominal-rate timebase at the first retained sample. Handle DMD paths independently and validate their phase relationship.
- `behavior/preprocess.py`: retain separately the BonVision→HARP alignment method, event residuals, and raw HARP epochs. The behavior photodiode correction must **not** be refit to hide source-to-HARP offsets.
- `common/clock_qc.py` and SLAP2 QC notebook: record acquisition order, estimated/reviewed source-cycle index, head/tail discarded samples, uncertainty bounds, DI3 cadence and source-index mapping evidence. Export before/after diagnostic plots.
- Metadata output: persist original timestamps, new corrected timestamps, mapping method, evidence files, review status, code version, and uncertainty; do not overwrite prior derived assets.
- Tests: HARP-first, SLAP2-first, crossing pulse while DI3 initially high, DMD1/DMD2 different cycle lengths, multiple source epochs, repeated trial endpoint, voltage vs GIAnT `frame_line_idxs`, missing launcher and partial/missing DI3.

## Stage 3 — selective re-postprocess

- Only sessions with **verified or explicitly reviewed offsets** are eligible for timing-corrected processing. Save corrected derivatives under versioned paths and keep originals.
- Save a per-session comparison: removed head and tail samples, HARP-to-source correspondence, stimuli within retained window, retained coverage, before/after time shifts, and timing uncertainty.
- Sessions lacking a unique offset are not certified for absolute stimulus latency. Estimated offsets can be used for exploratory timing only and must be labeled as such.
- Re-run behavioral alignment only if its own BonVision→HARP evidence fails, independent of source timing. Otherwise reuse its validated events.

## Highest-value additional information

1. Windows clock synchronization evidence between SLAP2 PC and behavior PC (NTP/PTP configuration, offset logs for recording dates). Without it, 1-second filenames cannot identify a unique ~20 ms cycle.
2. Raw/unchanged `.dat` chunk **original write-time** metadata with cycle chunk index; repeated chunk timestamps can estimate PC-wall-clock acquisition progression, if creation/mtime behavior is experimentally verified and unaffected by copying.
3. A single simultaneous physical event visible to both devices would anchor their clocks; it is not present in the established archival data.
4. Record a new calibration session with known shared trigger or source-output registered on HARP to establish hardware delay and filename-to-first-cycle timing for prospective experiments.
