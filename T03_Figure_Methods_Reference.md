# T0.3 figure-level interpretation reference

Each notebook plot has a companion `.txt` written by `savefig`, including exact asset class/dataset, calculation, scientific interpretation, and limitation. For review of the Kaspar presentation, use the six-figure order in `README.md`.

## Conventions and clocks

- **BV predicted event – HARP observed PD event (ms)**: signed, zero means the corrected BonVision event prediction and physically observed photodiode edge coincide in the HARP photodiode *relative* clock. A signed clock-origin intercept is distinct from a physiological response latency.
- **p95 absolute residual**: 95th percentile of `abs(predicted event − observed PD event)` over uniquely matched transitions. It includes systematic signed bias, unlike standard deviation or jitter alone.
- **Coverage**: uniquely matched BV event predictions divided by attempted BV event predictions. Under 75% coverage is considered insufficient for a plotted session-level summary. This coverage threshold is an audit display rule, NOT a physiological acceptance criterion.
- **Frame-period reference**: metadata-derived `1000 / display_rate_est_hz` if plausible (itself estimated using photodiode and frame-modulo 30); otherwise `1000 / DISPLAY_HZ_REFERENCE`. The official one-frame criterion requires separately verified display rate.
- **DI3 nearest-edge phase**: `t_photodiode − nearest_DI3_rising_edge`, not a synchronization error. Its bounded magnitude is a mathematical consequence of choosing the nearest rising edge.
- **Held-out modern PD alignment**: interpolation from retained matched anchors evaluated against every fifth withheld edge. This evaluates consistency/predictability of the mapping after the matching association has been chosen, and does not establish that a periodic absolute offset is resolved.
- **Legacy frame-modulo**: read original `alignment_method` from corrected CSV when metadata was overwritten by `existing_corrected_timestamps`. If a saved modulo phase is missing, evaluate 30 phases against HARP edges and label the resulting phase as **reconstructed**. Any low residual is explicitly *in-fit*.
- **Independent T0.3 PASS**: only from a reviewed, per-session source sample-to-HARP clock association and verified display-frame period. Good behavior fit and good DI3 cadence are supporting evidence, never automatic PASS.

## Remaining independent measurement for complete T0.3

Establish at least one verified per-epoch, per-DMD `(SLAP2 line index, HARP DI3 edge index/time)` anchor with the actual line period and exact cycle convention, and propagate it to the GIAnT `frame_line_idxs` or legacy voltage line coordinates. Demonstrate common HARP photodiode and DI3 acquisition timestamp origin (rather than just plausible start-time similarity), and compare source sample event coordinates against actual measured stimulus times over the full experiment. Use per-session signed bias, p95 absolute timing error, drift, and nonconforming sessions. The current toolkit will accept separately validated results through its documented evidence CSV while keeping preliminary diagnostics available.
