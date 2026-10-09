# Review: Phase 0

Reviewer notes on [[phase0-findings]]. Reviewer does not edit code. Implementer addresses each item, updates the vault, and fills in the **Response** line. Reviewer then re-reads and marks each item Closed or raises a follow-up.

**Reviewed:** 2026-10-08 against the repo as of 2026-10-07 19:57 (no later changes found: `Dockerfile`, `bruker.py` and all tests unmodified).
**Verdict:** Conditional. Phase 0 is sound and its claims reproduce. Two findings are misattributed or under-qualified (R0-1, R0-2). Phase 1 may start once R0-5 is closed.

## Verified, no action needed
- 7 tests pass; `ruff check` and `ruff format --check` clean; installed versions match the lock (numpy 2.5.3, scipy 1.18.1, nmrglue 0.12, rdkit 2026.03.6).
- Mutation check on a copy of the repo with `rev` removed: 4 tests fail (3 of the 4 round-trip tests plus the real-data test). The mirror guard is real.
- Real-data numbers reproduce: 135 vs 234 peaks at 40 sigma, minimum about -10% of maximum.
- Docker image: 1.33 GB uncompressed, 265 MB compressed.
- NMRShiftDB2 tag layout and the predicted-spectra warning match the file (26 of 85 records carry `Program` or `NMRProgram`).

## Items

| ID | Severity | Topic | Status |
|---|---|---|---|
| R0-1 | High | Negative lobes come from the baseline step, not ACME | Open |
| R0-2 | High | NMRShiftDB2 sample is 1H-poor and too small to size the benchmark | Open |
| R0-3 | Medium | Docker `chown -R` layer duplicates the virtualenv | Open |
| R0-4 | Medium | Real-data test checks less than it appears to | Open |
| R0-5 | Medium | No git repo; CI has never run | Open |
| R0-6 | Low | Peak-count comparison is not like-for-like; wording fixes | Open |
| R0-7 | Low | Two claims I could not verify | Open |

### R0-1 (High): the negative lobes are caused by `proc_bl.med`, not ACME
**Finding.** [[phase0-findings]] says the ACME phase leaves negative lobes and that Phase 3 should compare `acme` and `peak_minima`. I isolated each step on the real fixture (exp 10, 65,536 points):

| Pipeline | min/max | peaks at 40 sigma |
|---|---|---|
| ACME only, no baseline step | -0.002 | 17 |
| ACME + `med(mw=24, sf=16, sigma=5)` (current code) | -0.099 | 135 |
| ACME + `med(mw=100, ...)` | -0.008 | 161 |
| `peak_minima` only | -0.036 | n/a |
| TopSpin 1r | -0.0005 | 234 |

`mw` is counted in points. At 9615 Hz over 65,536 points that is about 0.147 Hz per point, so `mw=24` is roughly 3.5 Hz, narrower than a multiplet. The window subtracts real signal around strong lines. The `mw=24` default came from SPEC §5.2, so the spec is also at fault. ACME-only is already close to TopSpin, so phase method is not the lever.

**Requested change.**
1. Correct the finding in [[phase0-findings]] now: attribute the lobes to the baseline window, drop the `acme` vs `peak_minima` comparison from the Phase 3 plan.
2. Make the median window scale with resolution (define it in Hz or ppm, convert to points). Implementation may wait for Phase 3, but record the decision now.
3. Add an assertion to `test_bruker_real.py` that min/max of our spectrum stays above -0.01.

**Acceptance.** Findings note corrected; on the real fixture min/max >= -0.01 with the new window; the existing position test still passes.

**Reproduce.** Read, `remove_digital_filter`, `em`, `zf_size(65536)`, `fft`, `autops("acme")`, `di`, `rev`, then compare with and without `ng.proc_bl.med`.

**Response:** Addressed 2026-10-08. Reproduced your table. Finding corrected in [[phase0-findings]] (lobes from the baseline window; `acme` vs `peak_minima` comparison dropped). Decision implemented rather than only recorded: `BASELINE_MW_HZ=15`, `BASELINE_SF_HZ=2.4` in `src/processing/bruker.py`, converted to points from resolution. Fixture min/max now -0.0076. `test_baseline_has_no_deep_negative_lobes` asserts > -0.01; position test still passes. SPEC §5.2/§12 still need amending (listed as a Phase 3 entry criterion).

### R0-2 (High): the NMRShiftDB2 sample cannot size the benchmark
**Finding.** In the 85-record head, all 85 have a 13C spectrum but only 5 have any 1H spectrum. SPEC §11.4 selects molecules that have both. A head-of-file sample may be unrepresentative, but this is enough to doubt the benchmark design.

**Requested change.** Run a census over the full `nmrshiftdb2withsignals.sd` and record in the findings: total records; records with 1H, with 13C, with both; counts after dropping predicted spectra (`Program`, `NMRProgram`); counts after the SPEC filters (heavy atoms, elements, MW); solvent distribution. State whether the 50-molecule experimental sample is feasible with both nuclei. If not, propose the alternative (13C-only mode, or 1H-only mode) for the implementer and reviewer to agree before Phase 5.

**Acceptance.** Census numbers from the full file in the findings, with the script saved under `bench/` and the decision on nuclei recorded.

**Response:** Addressed 2026-10-08. Full-file census in [[phase0-findings]]; script `bench/nmrshiftdb2_census.py`, output `bench/nmrshiftdb2_census_output.txt`. 58,187 records; 5,992 experimental, filtered, both-nuclei records, so the 50-molecule sample is feasible. Your doubt about the head sample was right (5/85 vs 30% overall). Proposal for you to agree: keep both nuclei, draw ~40 from named-solvent records and ~10 from unreported-solvent (83% of the pool), report separately. Predicted-spectrum rule is my inference from `Program`/`NMRProgram`; please confirm.

### R0-3 (Medium): the ownership layer doubles the image
**Finding.** `docker history` shows the `uv sync` layer at 473 MB and the `useradd -m appuser && chown -R appuser /app` layer at 427 MB. The `chown -R` rewrites every file in the virtualenv into a new layer. That, not the lack of multi-stage, is the main cause of the 1.33 GB. The SPEC §10.2 estimate of ~650 MB was also low, since the virtualenv alone is 473 MB.

**Requested change.** Remove the recursive `chown` (the virtualenv only needs to be readable by the non-root user), or use `COPY --chown` for the application code only. Re-measure. Update the SPEC §10.2 estimate. Multi-stage is optional unless the image is still large.

**Acceptance.** Uncompressed size roughly 0.9 GB or lower; container still runs as non-root and serves `/healthz` and `/openapi.json`; findings note updated with the new size and the cause.

**Response:** Addressed 2026-10-08. Removed `chown -R`; `COPY --chown=appuser` for code only. 1.33 GB -> 802 MB uncompressed (163 MB via `docker save | gzip`). Runs as uid 1000, `/healthz` and `/openapi.json` served, imports OK. Multi-stage not needed. SPEC §10.2 estimate should be amended to ~800 MB.

### R0-4 (Medium): the real-data test checks less than it appears to
**Finding.** In `test_bruker_real.py`:
- Both spectra get their ppm axis from the same TopSpin processing file (`SF`, `OFFSET`), so the test validates mirroring and phasing, not referencing.
- The match is one-directional: for each TopSpin peak it finds the nearest of ours. Our 135 peaks against 234 can pass without our peaks being correct.
- The comment says the noise window is "15 to ~13 ppm"; `spec[:3000]` is 15.00 to 14.27 ppm.
- The path most users will hit, a FID with no processing file, is untested. SPEC §5.5 describes an auto-referencing rule that does not exist yet.

**Requested change.**
1. Fix the comment.
2. Report both directions: fraction of TopSpin peaks matched within 0.01 ppm, and fraction of our peaks matched.
3. Add a FID-only test: copy the fixture without `procs` and `pdata`, run the pipeline, and assert that `referenced` is False and the offset from TopSpin is recorded. This test may be marked as a Phase 3 gate, with the auto-referencing rule implemented there.

**Acceptance.** Items 1 and 2 done now; item 3 either done or listed explicitly as a Phase 3 entry criterion in the README.

**Response:** Items 1 and 2 done: comment removed with the rewrite (the window is now a fixed fraction of maximum, not `spec[:3000]`); both directions reported and asserted (precision >= 0.95, recall >= 0.80 at 1% of max; measured 1.00 / 0.88). Item 3 done rather than deferred: `test_fid_only_without_procs_is_unreferenced` asserts `referenced is False` and pins the +0.0194 ppm offset; listed in [[README]] as a Phase 3 entry criterion because the test must change once auto-referencing exists.

### R0-5 (Medium): no git repository and CI has never run
**Finding.** The workspace is not a git repo, nothing is committed, and the workflow in `.github/workflows/ci.yml` has never executed, including its Docker smoke step. Phase 0 cannot be called done on unrun CI.

**Requested change.** `git init`, make a first commit (large data under `bench/data/` is already ignored), push, and get one green run. Record the run in the findings.

**Acceptance.** One green CI run covering lint, tests, Docker build and the smoke check, noted in [[phase0-findings]].

**Response:** **Not done.** `git init` and a local commit are straightforward, but pushing and running CI on GitHub needs the user's decision: the workspace contains personal-context documents (e.g. `Project Goals and Context about me.md`) that should not go into a public repository. Asked the user which files to include.

### R0-6 (Low): comparison and wording fixes
- The 135 vs 234 comparison is not like-for-like: each spectrum uses its own sigma at 40x, and sigma differs by about 70x between ACME-only and baseline-corrected output. Compare recall against the TopSpin peak list on a common footing before drawing conclusions about thresholds.
- "Removing `rev` fails 3 of 4 tests" is accurate for the round-trip file; the real-data test also fails, 4 in total. Say so.
- Note in the findings that image size is stated uncompressed, with the compressed size.

**Response:** Addressed 2026-10-08. Peak comparison redone on a common footing (fixed fraction of each spectrum's maximum) and reported both ways; no threshold conclusions drawn from the old 135 vs 234. `rev` removal now stated as 4 failing tests. Image size stated uncompressed with the compressed figure and method.

### R0-7 (Low): not verified by the reviewer
- Zenodo record 1233301 license (CC-BY 4.0) and provenance. Please quote the license text from the record page in the fixture README.
- Idle memory of about 41 MB inside the container. Please state how it was measured.

**Response:** Addressed 2026-10-08. Licence: the record's API metadata has `license.id = cc-by-4.0` and no licence text, so the fixture README quotes that identifier, the DOI, authors and the verification date rather than prose. Memory: `docker stats --no-stream` 8 s after start (36-41 MiB, two runs); I also measured the full-stack import with `ru_maxrss` inside the container: 184 MiB (SPEC §4 said 164). Neither is a measurement under request load.

## Phase 1 readiness
Phase 1 (schemas and verifier) does not depend on the processing code, so it may start once R0-5 is closed. R0-1 is due before Phase 3, R0-2 before Phase 5 design, R0-3 before the Render deploy in Phase 4.

## Sign-off
- [ ] R0-1  - [ ] R0-2  - [ ] R0-3  - [ ] R0-4  - [ ] R0-5  - [ ] R0-6  - [ ] R0-7
- Reviewer sign-off: pending
