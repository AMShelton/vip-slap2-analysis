# Changelog

Short, dated entries. Recover removed code from the `archive/*` git tags.

## 2026-11 consolidation (branch `chore/consolidate-2026-11`)

- Retired the T0.3 start-order audit and its tests; superseded by `common/cycle_registration.py` and `common/fpga_cross_epoch.py` (see `docs/RETROSPECTIVE_SYNC.md`). Tag: `archive/pre-consolidation-2026-11-10`.

## Earlier merged work (folded from root patch notes)

- Epoch-duration QC: acquisition fragments shorter than 30 s (`min_epoch_duration_sec`, `min_epoch_duration`) are rejected in Python QC; raw extraction stays lossless.
- Multi-epoch physiology processing: traces are extracted per source epoch (line coordinates restart on every acquisition), trial-to-epoch labels are explicit, and the timebase jumps across imaging-off gaps.
