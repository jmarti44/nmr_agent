# Spectral_Agent: implementer / reviewer hub

Shared channel between the implementer (Claude) and the reviewer. Source of truth for the design is `/SPEC.md`.

## Phase status
| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, synthetic Bruker round trip, real-data fixture, Docker/CI skeleton | Implemented; review items R0-1..R0-4, R0-6, R0-7 addressed, **R0-5 (git + green CI) waiting on the user**. See [[phase0-findings]], [[review-phase0]] |
| 1 | Schemas + verifier (V01-V11) | Not started |
| 2 | Peak-list vertical slice with LLM | Not started |
| 3 | Raw-data ingestion (Bruker zip, JCAMP-DX) | Not started |
| 4 | Hardening + deployment | Not started |
| 5 | Benchmark + report | Not started |

## Open items for the reviewer
- Agree the benchmark nuclei decision proposed in [[phase0-findings]] (both nuclei; ~40 named-solvent + ~10 unreported-solvent molecules).
- Confirm the "predicted spectrum = index present in Program/NMRProgram" rule is acceptable.
- R0-5: git repo and green CI. Blocked on which files may be pushed (the workspace also holds personal-context documents).

## Phase 3 entry criteria
- FID-only referencing: implement the SPEC §5.5 auto-referencing rule; `test_fid_only_without_procs_is_unreferenced` currently pins the
  uncorrected +0.0194 ppm offset and must change to assert the corrected axis.
- Weak-peak recall vs TopSpin (0.5 below 1% of maximum) investigated.
- SPEC §5.2 / §12 amended for the Hz-defined baseline window.

## Conventions
- One findings file per phase in this folder (`phaseN-findings.md`); reviewer comments go in `review-phaseN.md`.
