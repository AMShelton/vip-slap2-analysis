# T0.3 methods and interpretation for the Kaspar discussion

## Decision criterion

**Original T0.3 acceptance (Oct 8):** show stimulus-event timing error relative to imaging acquisition below one stimulus display frame, per session, and exclude *measured failing* sessions from latency-sensitive conclusions. Independently verify the display-frame period instead of assuming 60 Hz. Report both the estimated **systematic offset** and temporal **jitter/drift**, with a clear convention for HARP and imaging clock origins.

A photodiode-BonVision alignment residual is **not** synonymous with imaging-to-stimulus error; timestamp series created from trial concatenation are not independent physical clocks.

## What we can currently quantify

| Quantity | Source | Formula / units | Defensible interpretation |
|---|---|---|---|
| BonVision→HARP fitted intercept | `qc/behavior/alignment_meta.json` | `intercept × 1000` ms | Relative clock coordinate origin of the fitted mapping; not optical response latency |
| BV↔HARP mapping residual, modern | BV `Photodiode-*` events + HARP `AnalogInput0` | p95 absolute error at 1-in-5 withheld rises | Predictive photodiode-clock mapping precision conditional on original edge detection, pairing and time-zero choice |
| BV↔HARP fitted residual, legacy | BV `Frame % 30 == phase` vs HARP PD edges | p95 nearest-edge residual from stored affine fit | Fitted consistency only; not a held-out validation, liable to modulo/cycle ambiguity |
| HARP PD→DI3 phase | HARP analog PD rises and HARP DI3 rises | `t_PD − nearest(t_DI3)` ms | Direct recorded stimulus transition relative to nearest SLAP2 acquisition-clock pulse, **not source-sample timing error** |
| DI3 cadence | HARP DI3 digital rises, recorded epochs | Within-epoch median interval; anomalous interval count | Acquisition clock regularity and clock-rate change; no claim from unequal full cycle counts |
| Source trial boundary repeats | Voltage `/traces/trial_*` attrs + literal endpoint samples | Matching counts | Possible repeated physical acquisition samples at trial concatenation; not independent HARP error |
| GIAnT line coordinates | `experiment_summary.h5 /Path*/frame_info/frame_line_idxs` | Count nonincreasing steps | Sample time coordinates in SLAP2 line domain; line conversion and HARP origin still required |
| Independent imaging-to-stimulus absolute lag/error | Verified per-session source line↔HARP DI3 correspondence + HARP PD and verified frame rate | lag, p95 absolute error, frame ms | **Only this can establish pass/fail** |

### Historic BonVision methods

- **Encoded photodiode:** `bv_photodiode_piecewise` or `bv_photodiode_affine`, with detectable `Photodiode-0/1` rows. Matched rises permit withheld anchor assessment, but periodic event trains can match at whole-toggle-period shifts. Audit reports the number of plausible shifts; selecting using an existing fitted intercept is **not** an independent absolute offset validation.
- **Older sessions:** `frame_modclass_affine_anchored_edges` reconstructs a BonVision flip sequence from frame modulo 30 and uses rising+falling HARP analog PD transitions, with dropout segmentation and a prior on clock offset. The audit's legacy nearest-edge scatter is descriptive consistency of the **existing fit** only. Report separately from modern held-out measurements.

### Modality-specific considerations

- **Legacy voltage:** `dendriticVoltageTraces-*.h5` source trial line ranges and ROI endpoint samples. `voltage_session_traces_dff_robust_f0_trial.h5` downstream `DMD*/timebase_sec` is generated from retained sample counts and nominal rate, not direct source-line timestamps.
- **GIAnT glutamate/Ca²⁺:** extraction can run at independent `analyzeHz`; authoritative timing is acquisition line-domain `frame_line_idxs`, not the downstream nominal rate. A dual-color experiment may have distinct channels / ROI grids while sharing underlying scanning clock.
- **DMD2:** even when stored DMD1/DMD2 timestamps match, DI3 may be linked to only one DMD's cycle signal; DMD2 requires an independently supported mapping.
- **Multi-epoch sessions:** align within epoch and preserve imaging-off gaps, not uniformly across inactive periods. Short rejected epochs are not accepted acquisitions.

### What to present before a fix

1. The **dataset denominator** from the live registry and each session's method/evidence coverage.
2. Per-session BonVision→HARP relative fit offset and (where measurable) predictive photodiode residual; fallback labelled separately.
3. Hardware HARP photodiode→DI3 phase distribution, acquisition-clock continuity, and any origin ambiguity.
4. The outstanding requirement to establish **physical SLAP2 sample-to-HARP mapping** rather than equating phase-within-cycle or clock-fit jitter with accuracy.
5. Optional prevalence of boundary repeat / nonincreasing `frame_line_idxs`, without attributing it prematurely to GIAnT or treating hypothetical cumulative ~18 ms displacement as observed behavioral lag.

Any session without independent end-to-end evidence stays **NOT_INDEPENDENTLY_VERIFIED**, not PASS or FAIL.


### Alignment-method provenance for reused corrected logs (v1.1)

`precorrected_without_method` is not a third alignment algorithm. When `behavior.preprocess` reuses a file that already has `corrected_timestamps`, it records `existing_corrected_timestamps` in `alignment_meta.json`, even if the corrected event log still records its original `alignment_method` (for example, `frame_modclass_affine_anchored_edges`). T0.3 reads the event-log method in that case and classifies the session as `legacy_frame_modulo`; it separately records the reuse operation and metadata origin. If the original method is genuinely absent or metadata disagree, the audit keeps the ambiguity visible. Recovering the method does not restore missing saved fit details (e.g. legacy phase/slope/intercept), so the event-level residual panel may still be unavailable unless that metadata is recovered from its original QC output.
