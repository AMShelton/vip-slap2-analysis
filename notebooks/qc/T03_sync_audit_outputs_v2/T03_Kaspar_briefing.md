# T0.3 — Results from the local session registry

- Sessions audited: **137**.
- Session-level behavioral event residuals with ≥75% matching coverage: **88**.
- Modern encoded-photodiode sessions with held-out BV→HARP predictions: **44**.
- Independent end-to-end T0.3 decisions: **{'NOT_INDEPENDENTLY_VERIFIED': 137}**.
- Absolute source line-index→HARP DI3 relationship and DMD/epoch mapping remains required for sessions without independent validation.

**Definitions:** signed per-event residual = corrected BonVision frame transition − HARP measured analog PD edge. This measures consistency after the saved alignment was already applied. HARP PD−DI3 phase measures physical stimulus onset relative to the acquisition cycle, *not* timestamp error. The one-frame visual reference is estimated or assumed and never substitutes for a verified display frame period.

**Slide figures:** `figures/*.png` and `figures/*.svg`; full per-figure provenance and interpretation in `figures/*.txt`. Missing measurements must be shown separately from scientific failures. No pipeline files were modified.
