# T0.3 — synchronization audit (generated from local registry)

Session count: 137.  Independent status counts: {'NOT_INDEPENDENTLY_VERIFIED': 137}.

**Acceptance criterion:** independently established absolute imaging–stimulus alignment error below one independently measured frame period. A behavioral photodiode fit does **not** establish this end-to-end mapping.

**Measurements:** BonVision→HARP clock-offset model and, for encoded BV photodiode sessions, held-out predictive PD residual; legacy frame-modulo alignment reported separately; HARP photodiode-event phase relative to HARP DI3 cycle-clock; within-epoch DI3 cadence; optional source trial-boundary and frame_line_idxs checks.

**Open questions:** validated SLAP2 source-line-to-HARP epoch and phase correspondence; DMD2 timing relative to DI3; display period verified independently; legacy frame-modulo ambiguity and dropout handling; sample-rate differences across voltage, glutamate and somatic Ca²⁺.

**Excluded on demonstrated T0.3 failure:** 0.
**Not independently verified:** 137.

See `T03_Kaspar_session_disposition.csv`, `t03_session_results.csv`, and source-specific captions in `figures/*.txt`.
