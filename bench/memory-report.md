# NMR-Agent experience-memory report (H-large)

**Question:** for molecules with more than 20 heavy atoms, does showing the model 3 past cases from a 200-case memory bring its first answer closer to the true structure?

**Answer: no.** Under the pre-registered criteria (`docs/experience-memory-spec.md` §15.8 and §15.11), **H-large is falsified**. Memory moved the mean similarity by −0.006, a smaller change than the 0.032 gap between two identical no-memory runs. Retrieved cases did no better than random cases from the same memory (+0.006). Memory did no harm.

Run: 2026-11-16 and 2026-11-17, model `claude-opus-5-5`, prompt `2026.11.1+shifts.2026.11.1`, settings hash `2bb0d0ca…`, one attempt per analysis. Every number in the Results section comes from `bench/memory.py compare`, which reads `bench/memory/eval/` (`compare.json`, `compare.txt`). Exceptions are marked post hoc.

## Setup

- **Test set.** The 50 molecules of `bench/memory/mem_eval.json` with more than 20 heavy atoms. They were frozen at M3, and no earlier step used them for measurement.
- **Memory.** 200 approved cases in `memory/` (manifest SHA-256 `7c73c3b2…`, `memory_version` `20de57c79d4d`, commit `f3c33fb`): 120 with more than 20 heavy atoms, 60 with 11-20, 20 with 10 or fewer. The leakage check against all 100 MEM-EVAL molecules is clean.
- **Arms.**

  | Arm | Memory |
  |---|---|
  | M0-a, M0-b | none (run twice to measure noise) |
  | M200L | 3 cases retrieved by spectral similarity (parameters frozen at M4, `retrieval_params.json` `bf933bfc…`) |
  | R200L | 3 random cases from the same 200 (seed per molecule) |

  The four arms send the same request; only the examples block differs. Arms ran molecule by molecule, in a seeded shuffled order, so drift over the day hits every arm alike.
- **Measure.** Morgan (radius 2, 2048 bits) Tanimoto between the service's first candidate and the truth; 0 when there is no answer. S(arm) is the mean over the 50 molecules.

## Results

| Arm | S | Top-1 | Top-3 | Truth proposed | No answer (all timeouts) | Cost per analysis | Time per analysis |
|---|---|---|---|---|---|---|---|
| M0-a | 0.6524 | 25/50 | 30/50 | 31/50 | 7 | $0.1144 | 56.8 s |
| M0-b | 0.6843 | 28/50 | 32/50 | 32/50 | 8 | $0.1122 | 56.7 s |
| **M200L** | **0.6626** | 28/50 | 32/50 | 32/50 | 10 | $0.1233 | 56.4 s |
| R200L | 0.6567 | 27/50 | 33/50 | 33/50 | 8 | $0.1277 | 57.6 s |

Top-1 and top-3 use the InChIKey connectivity block. Cost and time are means over analyses that returned a value; a timeout records no cost.

**Verdict conditions (§15.8, as defined in §15.11):**

| Condition | Value | Holds |
|---|---|---|
| Supports: Δ ≥ +0.05 | Δ = −0.0057 | no |
| Supports: Δ > 2N | 2N = 0.0638 | no |
| Supports: Wilcoxon two-sided p < 0.05 | p = 0.8721 (19 non-zero d_i of 50) | no |
| Supports: S(M200L) − S(R200L) ≥ +0.03 | +0.0059 | no |
| Falsifies: Δ ≤ +0.02 | −0.0057 | **yes** |
| Falsifies: Δ ≤ N | N = 0.0319 | **yes** |
| Falsifies: S(R200L) ≥ S(M200L) | 0.6567 vs 0.6626 | no |
| Falsifies: gain confined to m_i ≥ 0.5 | Δ over the 48 molecules with m_i < 0.5 = −0.0124 | **yes** |
| Harm: timeouts M200L − mean(M0) > 3 | 10 − 7.5 = 2.5 | no |
| Harm: cost per molecule up more than 30% | $0.1233 vs $0.1133 (+8.8%) | no |

Δ = S(M200L) − mean(S(M0-a), S(M0-b)); N = |S(M0-a) − S(M0-b)|; d_i is the per-molecule Δ.

**Verdict: H-large falsified. Harm: none.**

The reviewer reproduced S for every arm, Δ, N, p and the non-zero count with a separate script that reads the result files directly (RDKit and scipy).

## What the result means

- **No moderate effect.** With 50 paired molecules, the test could detect a shift of about +0.05 (§15.8, power note). The result rules out an effect of that size. It does not rule out a smaller one.
- **Close analogues were rare.** m_i is the highest similarity between the truth and any case shown in M200L: mean 0.220, max 0.571, and only 2 of 50 molecules at or above 0.5. In most molecules the model saw no case that resembled the answer. This is the likely reason memory did not help; the run does not test it.
- **Per molecule (post hoc).** Memory changed the first answer's similarity for 19 molecules: 10 got closer and 9 moved further away. Three of the drops are M200L timeouts on molecules that an M0 run answered (−0.015 of Δ). Over the 40 molecules answered by M0-a, M0-b and M200L, the mean d_i is +0.011. That is still far below the +0.05 bar, so the timeouts do not explain the result.

## Limitations

- **Case approval is not expert review.** The owner approved the 200 cases and is not a chemist (`approved_by`: "Jose Martinez (owner, not a chemist)"). Correctness rests on the nmrshiftdb2 assignment, the automatic checks, and an AI screen of the missed cases. Two cases were rejected (083, a tautomer; 171, an incomplete ¹H list) and replaced. Solved cases carry the least risk, because the database and the agent reached the same structure independently. In the large stratum, 85 cases were solved, 17 missed and 18 had no answer.
- **Known service bug, left frozen.** `src/processing/normalize.py` gives a solvent- or water-position peak back to the analyte only when the integrals do not already fit the formula. Its window is wider than the one verifier rule V05 uses. On 14 of the 179 memory cases with a peak table, a real peak stays tagged as solvent. V05 can then hard-reject the true structure (5 approved missed cases), and the bug favours candidates with extra OH or NH. The bug behaves the same in every arm, so it does not bias the comparison, but it lowers every arm's score. It will be fixed after this result and the measurements repeated.
- **Spend is undercounted.** $23.88 is recorded for the four arms. Timeouts record no cost but are still billed (up to about $0.2 each, 33 timeouts). Collecting the 200 memory outcomes recorded $15.66.
- **One run per arm.** The noise estimate N comes from a single pair of M0 runs.

## Files

- `bench/memory/eval/run.json`: store, retrieval and configuration hashes, checked on every resume.
- `bench/memory/eval/<arm>/<id>.json`: one result per analysis, with the request hash, candidates, trace, injected case ids and their similarity to the truth, cost and timing.
- `bench/memory/eval/compare.json`, `compare.txt`: the output of `PYTHONPATH=. .venv/bin/python -I bench/memory.py compare`.
