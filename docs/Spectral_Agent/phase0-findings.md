# Phase 0 findings

Revised 2026-10-08 after [[review-phase0]]. Changes from the first version are listed at the bottom.

## Done
- Repo scaffold per SPEC §3.3; uv lockfile; Python 3.12; pinned nmrglue 0.12 + NumPy 2.x (resolved scipy 1.18.1, numpy 2.5.3).
- Synthetic Bruker round trip (`tests/processing/test_bruker_roundtrip.py`) passes on NumPy 2. With `rev` removed, 4 tests fail
  (3 of 4 round-trip tests plus the real-data test), so the mirror guard is real.
- Real Bruker fixture (F1): Zenodo 1233301, exp 10 = 1H zg30, MeOD, 600 MHz, with `pdata/1/1r` (licence: see fixture README).
- `/healthz` + OpenAPI app, Dockerfile, CI workflow (CI not yet run: see R0-5 in [[review-phase0]]).
- 9 tests pass; `ruff check` and `ruff format --check` clean.

## Findings that change later phases

### Baseline window, not phasing, caused the negative lobes (R0-1)
My first version blamed ACME phasing. The reviewer's isolation, which I reproduced, shows the cause was `proc_bl.med` with the
SPEC §5.2 value `mw=24`. `mw` is in points: at 0.147 Hz/pt that is about 3.5 Hz, narrower than a multiplet, so the filter subtracts
real signal.

| Pipeline (exp 10, 65,536 pts) | min/max |
|---|---|
| ACME only, no baseline step | -0.002 |
| ACME + `med(mw=24)` (SPEC §5.2, old code) | -0.099 |
| ACME + `med(3.5 Hz)` / 10 / **15** / 20 / 30 Hz | -0.099 / -0.013 / **-0.008** / -0.006 / -0.005 |
| TopSpin 1r | -0.0005 |

**Decision:** the median window is now defined in Hz (`BASELINE_MW_HZ = 15`, `BASELINE_SF_HZ = 2.4`) and converted to points from
the spectrum resolution in `process_bruker_fid`. Result on the fixture: min/max = -0.0076. The `acme` vs `peak_minima` comparison
is dropped from the Phase 3 plan; SPEC §5.2 and §12 (auto-phasing row) should be amended to say the same. A regression test
asserts min/max > -0.01.

### Real-data agreement with TopSpin (R0-4, R0-6)
Compared on a common footing: both spectra normalised to their own maximum, peaks above a fixed fraction of that maximum
(`distance=10` points), match tolerance 0.01 ppm. The earlier "135 vs 234" compared peaks at 40x each spectrum's own sigma, which is
not like-for-like (sigma differs about 70x between ACME-only and baseline-corrected output).

| Peak threshold (fraction of max) | ours | TopSpin | TopSpin peaks matched by ours (recall) | our peaks matched by TopSpin (precision) |
|---|---|---|---|---|
| 0.001 | 72 | 180 | 0.49 | 1.00 |
| 0.003 | 33 | 75 | 0.48 | 1.00 |
| 0.01 | 15 | 17 | 0.88 | 1.00 |
| 0.03 | 5 | 6 | 0.67 | 1.00 |

Every peak we find is a real TopSpin peak. Recall is 0.88 at 1% of maximum but only about 0.5 below that, so weak-peak recall is the
open question for Phase 3 (the pipeline only sees the dominant solvent lines well). Median position difference is about 0.0005 ppm
(target 0.01). The test asserts precision >= 0.95 and recall >= 0.80 at the 1% threshold.

Limits of this test: both axes come from the same TopSpin processing file (`SF`, `OFFSET`), so it validates mirroring, phasing and
baseline, **not referencing**.

### FID-only (no `procs`/`pdata`) is unreferenced (R0-4)
With `procs` and `pdata` removed from the fixture, `referenced` is `False` and the ppm axis is offset by **+0.0194 ppm**
(HOD at 4.876 vs 4.856). This is the offset the SPEC §5.5 auto-referencing rule has to remove; the rule does not exist yet.
It is asserted in `test_fid_only_without_procs_is_unreferenced` and is a **Phase 3 entry criterion** (see [[README]]).

### NMRShiftDB2 census, full file (R0-2)
Script: `bench/nmrshiftdb2_census.py`; output: `bench/nmrshiftdb2_census_output.txt`. File: `nmrshiftdb2withsignals.sd`, 158,569,118 bytes,
58,187 records. A spectrum is "predicted" if its index has a non-empty value in the record's `Program` or `NMRProgram` tag; all others
are treated as experimental (assumption, not verified against NMRShiftDB2 documentation).

| Stage | Records |
|---|---|
| All | 58,187 |
| With any 1H / any 13C / both | 17,671 / 33,378 / 7,475 |
| Experimental 1H / 13C / both (predicted dropped) | 17,535 / 32,875 / 6,865 |
| After SPEC filters (<=30 heavy atoms, elements in C H N O S F Cl Br, MW <= 450): all | 36,729 |
| ... experimental 1H / 13C / both | 13,125 / 28,938 / **5,992** |

Solvent of experimental 1H spectra in the 5,992 filtered both-nuclei records (counted per spectrum, so can exceed records): not reported
4,987; CDCl3 654; D2O 378; DMSO-d6 72; CD3OD 68; acetone-d6 50; CCl4 48; DMSO 33.

**Reading.** The head-of-file sample (5 of 85 records with 1H) was unrepresentative: across the whole file 30% of records have 1H and
13% have both. A 50-molecule sample with both nuclei is feasible (5,992 candidates). Two caveats: (1) about 83% of the filtered both-nuclei spectra (4,987 of 5,992, counted per spectrum)
report no solvent, so solvent-based tagging (SPEC §5.5) cannot be exercised on them; (2) 1H signals carry no multiplicity.
**Proposal for reviewer agreement before Phase 5:** keep both nuclei; stratify by DoU as in SPEC §11.4 but draw about 40 of the 50 from
the roughly 1,000 records with a named solvent (the listed named-solvent spectrum counts sum to 1,303) and about 10 from "unreported" (run with `solvent=None`); report the two groups separately.

### NMRShiftDB2 SD layout (F2)
Per-record tags `Spectrum 1H N` / `Spectrum 13C N` (N = spectrum index); value `shift;intensity;atomIndex|...` (13C adds a multiplicity
letter, e.g. `17.6;0.0Q;10`). `Solvent`, `Field Strength [MHz]`, `Temperature [K]` are keyed `N:value`. Atom indices are 0-based and
repeat for split signals, so 1H integrals come from grouping by atom index. Records mix experimental and predicted spectra.

### Docker image (R0-3)
| | Uncompressed | Compressed (`docker save \| gzip`) |
|---|---|---|
| Before | 1.33 GB | not measured by me (reviewer: 265 MB) |
| After | **802 MB** | 163 MB |

Cause: `RUN useradd ... && chown -R appuser /app` rewrote the 473 MB virtualenv into a second layer. Now the virtualenv stays
root-owned and world-readable, and only the application code uses `COPY --chown=appuser`. The container still runs as
`uid=1000(appuser)`, serves `/healthz` and `/openapi.json`, and imports numpy, nmrglue and rdkit. Multi-stage was not needed.
SPEC §10.2's ~650 MB estimate was low: the virtualenv alone is 473 MB, so ~800 MB is the realistic figure.

### Memory (R0-7)
Idle: `docker stats --no-stream` (MemUsage) 8 s after `docker run` with the app serving only `/healthz`: 36-41 MiB across two runs.
Full stack: `resource.getrusage(RUSAGE_SELF).ru_maxrss` inside the container after importing numpy, scipy.ndimage, scipy.optimize,
nmrglue, rdkit.Chem, fastapi, pydantic, anthropic: **184 MiB** (SPEC §4 measured 164 MiB). This is import cost only, not a request under
load, so FFT/phasing/RDKit memory at the 512 MB Render limit is still unmeasured.

## Not done / caveats
- No git repo yet and CI has never run (R0-5). Pushing needs the user's decision on which files go in the repository.
- "Predicted spectrum" detection (above) is an inference from tag names.
- Zenodo licence is recorded from the API's licence id (`cc-by-4.0`); the record exposes no licence text.

## Changes from the first version of this note
1. Negative-lobe cause corrected from ACME phasing to the baseline window; fix implemented, not just recorded.
2. "135 vs 234 peaks" replaced by a common-footing comparison in both directions.
3. Removing `rev` fails 4 tests, not 3.
4. Image size now stated as uncompressed with the compressed figure; reduced from 1.33 GB to 802 MB.
5. Full-file NMRShiftDB2 census added; head-of-file counts withdrawn.
6. Memory measurement method stated; full-stack figure added.
