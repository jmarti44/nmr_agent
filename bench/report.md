# NMR-Agent benchmark report

**Question:** given a 1D ¹H and ¹³C peak list, the molecular formula and the solvent, how often does NMR-Agent return the correct structure?

**Answer:** on 50 real molecules from NMRShiftDB2, the correct structure was the service's first answer for **34 (68%)** and among its first three for **37 (74%)**. It solved every molecule with 10 or fewer heavy atoms and 3 of the 14 with more than 20. Almost all of the accuracy comes from the model: the deterministic verifier guarantees that returned structures are consistent with the formula and the spectrum, but it does not raise top-1 accuracy on this data.

Runs: 2026-10-19 and 2026-10-20, model `claude-opus-5-5` (effort `high`), prompt `2026.10.3+shifts.2026.10.2`, one analysis per molecule. Every number below is produced by `bench/benchmark.py report` from the saved results in `bench/results*/`.

## Setup

| | Main set | Unreported-solvent set | Pilot |
|---|---|---|---|
| Molecules | 50 | 20 | 20 |
| Solvent given | yes | no (database says "unreported") | yes |
| Solvent quotas | CDCl₃ 25, D₂O 8, DMSO-d₆ 8, CD₃OD 5, other 4 | — | CDCl₃ 8, DMSO-d₆ 4, D₂O 3, CD₃OD 3, other 2 |
| Drawn from | 727 unique structures | 4,468 unique structures | 794 unique structures |

- **Selection.** Each set was drawn with a fixed seed and written to `selection.json` before any model call; no set overlaps another by record or by structure. Records whose ¹³C list misses two or more of the structure's distinct carbons were excluded first (47 for the main set, 175 for the unreported set); the pilot predates that filter.
- **Input to the service.** Experimental ¹H and ¹³C shifts from one same-solvent spectrum pair, the molecular formula, and the solvent (except in the unreported set). Not given: multiplicities, coupling constants, exchangeable-proton flags, spectrometer frequency. ¹H integrals are given only when the database assigns every C–H carbon (otherwise they are withheld).
- **Scoring.** A molecule counts as correct when a candidate's InChIKey connectivity block matches the true structure (stereochemistry is not scored). A molecule whose first attempt failed (a timeout at the 120 s per-call limit) counts as a miss, even if a rerun later succeeded.

## Results

| | Service top-1 | Service top-3 | Model's own order, top-1 |
|---|---|---|---|
| **Main (50)** | **34/50 (68%)** | 37/50 (74%) | 35/50 |
| Unreported solvent (20) | 18/20 (90%) | 19/20 | 19/20 |
| Pilot (20) | 13/20 | 15/20 | 15/20 |

95% Wilson intervals: main top-1 54-79%, top-3 60-84%; unreported top-1 70-97%. The unreported set is an easier sample (no molecule above 15 heavy atoms), so its 90% is not comparable with the main set's 68%.

**Molecule size decides success** (main set, top-1):

| Heavy atoms | Top-1 | | Degree of unsaturation | Top-1 |
|---|---|---|---|---|
| 10 or fewer | 12/12 | | 0-1 | 7/8 |
| 11-20 | 19/24 | | 2-4 | 13/17 |
| more than 20 | 3/14 | | 5-7 | 9/11 |
| | | | 8 or more | 5/14 |

By solvent: CDCl₃ 16/25, D₂O 7/8, DMSO-d₆ 6/8, CD₃OD 3/5, other 2/4. The groups are too small to claim a solvent effect.

## Where the misses come from (main set, 16 misses)

| Cause | Count |
|---|---|
| The model never proposed the true structure | 9 |
| Timeout at the 120 s call limit, no answer | 3 |
| Timeout on the first attempt, answer only on a rerun (counted as a miss) | 1 |
| Proposed, but not ranked first | 3 |

The misses are large, ring-rich, natural-product-like molecules. For these, many same-formula isomers are consistent with a list of shifts, and the model's first choice is often one of them.

## What the verifier contributes

Every candidate is checked by eleven deterministic rules (RDKit for structure, literature shift tables for the spectrum): valid SMILES, formula, degree of unsaturation, mass, proton count, ¹³C and ¹H environment counts, required peaks for each functional group, unexplained peaks, duplicates. Candidates are then re-ranked by a confidence that is 85% verifier score and 15% model confidence.

- **It does not improve the first answer.** The re-ranking never moved a correct structure into first place. As run, it moved a correct first answer down once in each set, so the service was one below the model's own order (34 vs 35, 18 vs 19, 13 vs 15).
- **`accepted` means consistent, not correct.** Main set, service top-1 as run: accepted answers were correct 28 of 40 times; the 6 answers it flagged or rejected were all correct. Over every candidate the model proposed, 31 correct and 71 wrong structures were accepted.
- **What it does provide:** a guarantee that a returned structure has the right formula and can produce the observed number of signals, an audit trail of every rule, and a check of the model's functional-group claims against the structure.

**Rule fixes after the benchmark.** Six true structures the model proposed in the main set were flagged or rejected. Four causes were rule ranges that real data exceed (aryl aldehyde ¹H above 10.1 ppm, conjugated aldehyde ¹³C below 190 ppm, heteroaromatic C=N carbons above 160 ppm, missing exchangeable protons treated as failures). The ranges were re-derived from NMRShiftDB2 with all benchmark molecules left out (shift tables 2026.11.1). On 756 records outside the benchmark the fixes removed failures on true structures and on wrong isomers by about the same small amount, so they correct specific false rejections without making the rules sharper.

Re-verifying the saved answers with the fixed rules (no new model calls), **post hoc, not the headline**: main 35/50, unreported 19/20, pilot 14/20, equal to the model's own order in all three sets. The verifier no longer costs accuracy, and still does not add any. Ranking same-formula isomers would need each candidate's predicted spectrum compared with the observed one (for example a learned shift predictor or DP4-style scoring), which this version does not do.

## Cost and time

| | Main | Unreported | Pilot | All 90 |
|---|---|---|---|---|
| Cost (USD) | 4.43 | 1.25 | 1.80 | 7.48 |
| Mean model time per molecule | 42 s | 28 s | 38 s | |
| Repair rounds used | 7 of 50 | 1 of 20 | 3 of 20 | |

About $0.09 per molecule on average, reruns included. Costs are computed from token usage and list prices and have not been matched to a bill.

## Caveats

- **Possible training-data overlap.** NMRShiftDB2 is public and these molecules may be in the model's training data. The database has no record dates; splitting the main set by record id (a rough proxy for submission order) gives 18/25 for the lower half and 16/25 for the upper, which shows no sign of a difference but cannot rule overlap out.
- **An easier task than real use.** The formula is given, and the input is a clean database peak list without multiplicities or couplings. Processing raw spectra (Bruker or JCAMP-DX uploads) was not part of this benchmark.
- **No like-for-like external comparison.** The natural baseline here is the model without the verifier's re-ranking (35/50, above). Published systems report results on other datasets and inputs, for example 71% top-1 for a tool-using agent on the Alberts set ([arXiv 2607.19406](https://arxiv.org/abs/2607.19406)), 52.9% top-1 for NMR-Solver ([arXiv 2509.00640](https://arxiv.org/abs/2509.00640)) and 42.8% for NMRTrans ([arXiv 2602.10158](https://arxiv.org/abs/2602.10158)) on experimental spectra; these give context, not a ranking.
- **Small samples.** 50 molecules give a 25-point confidence interval on the headline.
- **Tuning.** The shift ranges were checked against the benchmark: recomputed without the benchmark molecules, none of the ranges the verifier uses moved beyond tolerance. Rule tolerances chosen earlier on NMRShiftDB2 records were not re-fitted.

## Reproduce

```bash
# needs the NMRShiftDB2 export nmrshiftdb2withsignals.sd at bench/data/nmrshiftdb2/ (not tracked)
PYTHONPATH=. uv run python bench/benchmark.py report --set main          # numbers above, from saved results
PYTHONPATH=. uv run python bench/benchmark.py rescore --set main         # re-verify saved answers, no model calls
set -a; . ./.env; set +a                                                 # ANTHROPIC_API_KEY
PYTHONPATH=. uv run python bench/benchmark.py run --set main --cap 5.5   # new model calls (resumes; stops at the cap)
```

Sets: `pilot`, `main`, `unreported`. Selections are never rewritten once they exist.

Data: NMRShiftDB2 (<https://nmrshiftdb.nmr.uni-koeln.de>), open-content database of assigned NMR spectra.
