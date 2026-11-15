# Spec: experience-memory layer for candidate generation

**Status:** draft, 2026-11-14; review decisions recorded the same day (§12). Spec only: no implementation exists yet. Work follows the order in §13.
**Scope:** add a store of approved, solved cases and inject the most relevant ones into the proposal prompt as worked examples. Nothing else in the pipeline changes.

Numbers marked *(census)* come from a read-only analysis of the NMRShiftDB2 pools run for this spec with the existing `bench/benchmark.py` filters (same-solvent ¹H/¹³C pairs, ¹³C-complete, one record per structure). Numbers marked *(simulation)* come from one seeded dry run of the proposed split (seed 20261114); the real selection will differ slightly.

---

## 1. Why, and what is being tested

Benchmark miss breakdown (`bench/report.md`, main set, 16 misses): 9 never proposed, 4 timeouts, 3 proposed but not first. Generation is the bottleneck, and the misses are concentrated in molecules above 20 heavy atoms (3/14 correct).

**Hypothesis H.** Showing the model a few solved cases with a similar formula and shift pattern, from scaffolds it is not being tested on, raises the rate at which it proposes the true structure, and therefore service top-1.

**What this is not.** Memory does not touch verification, scoring or ranking (constraint 1). It is not a lookup of the answer: the pool is scaffold-disjoint from every eval set (constraint 3).

---

## 2. Conflicts with the codebase and with the constraints

Flagged here rather than worked around. Each needs a decision at review.

| # | Conflict | Where | Proposed resolution |
|---|---|---|---|
| C1 | The system prompt is static, hashed per version and cached (`tests/prompting/prompt_manifest.json`, `cache_control` on the single system block). Per-query examples cannot live there. | `src/prompting/templates.py`, `builder.py` | Inject examples into the **user message**. The cached prefix stays byte-identical. |
| C2 | The model needs one new rule on how to treat examples. That changes the system prompt, so `PROMPT_VERSION` bumps and **every arm, including empty memory, runs on the new prompt**. Results are not comparable with the report's 34/50. | `templates.py` rule list | Accept; the empty-memory arm is the baseline, rerun under the new prompt. |
| C3 | `MAX_USER_TOKENS_ESTIMATE = 8000` covers the whole user message. Examples count toward it, so a long peak list plus examples could raise `PromptTooLarge` (HTTP 400) for a request that works today. | `builder.py` | Examples have their own cap (§5.4) and are **dropped first**; memory must never cause `PromptTooLarge`. |
| C4 | Constraint 2 lists formula, DoU and carbon count, but `molecular_formula` is optional in the API. Without it, DoU and carbon count are unknown. | `SampleContext` | Retrieval falls back to peak matching and solvent only (§5.2). The experiment always supplies the formula. |
| C5 | Constraint 2 lists solvent, but **90% of scaffold-eligible memory candidates have no reported solvent** (4,269 of 4,734 *(simulation)*; they come from the unreported-solvent pool). | NMRShiftDB2 | **Decided:** solvent is a soft feature: a match adds to the score, unknown is neutral, a mismatch is not penalised; unreported-solvent cases are allowed. Alternative: restrict memory to named-solvent records (465 candidates in total *(simulation)*; how many of those are large molecules was not counted, and large cases are the ones the eval set needs; see §7.2). |
| C6 | **Murcko scaffolds are undefined for acyclic molecules**: 175 of 747 named-pool molecules (23%) and 1,215 of 4,477 unreported-pool molecules (27%) have an empty scaffold *(census)*. A scaffold split treats them all as one group. | Constraint 3 | **Decided:** all acyclic molecules go to the memory side; the eval set contains none (§7.1). |
| C7 | **One scaffold dominates**: bare benzene is the scaffold of 156 named-pool (21%) and 1,360 unreported-pool (30%) molecules *(census)*. It also must sit wholly on one side. | Constraint 3 | **Decided:** benzene-scaffold molecules go to the memory side only. |
| C8 | The existing eval sets (pilot 20, main 50, unreported 20) are small, 24 of the 90 are acyclic, 12 of the main 50 are at the ceiling (10 or fewer heavy atoms, 12/12), and they were used to find the verifier range fixes. | `bench/results*/` | **Decided:** a **new held-out set** for this experiment (§7.1). The 90 old molecules are excluded from memory too, so the old sets stay usable. |
| C9 | The service is stateless and the image contains only `src/` (`Dockerfile`: `COPY src ./src`); the free host's disk is ephemeral. There is nowhere for the running service to write cases, and no feedback route. | `Dockerfile`, `src/api/` | The store is **read-only data shipped with the code**; cases are added offline through the approval command (§6). This also enforces constraint 4. Deployment needs one more `COPY`. |
| C10 | "Experience" implies the agent's own past analyses, but the service has no feedback path (C9). | — | **Decided:** each case is a database record with a known structure **plus the agent's own outcome on it**, collected offline by running the agent once on the memory molecules (§7.2a). Cases are marked solved or missed. |
| C11 | `Versions` and `AnalysisResponse` use `extra="forbid"`. Reporting which cases were injected changes the response schema. | `src/schemas.py` | Add `versions.memory_version` and `memory_cases`; bump `SCHEMA_VERSION` (§5.5). |
| C12 | Constraint 1 forbids routing memory into the verifier's confidence. `final_confidence = 0.85 × verifier + 0.15 × model confidence`, and examples can change the model's self-reported confidence, so memory can move the ranking indirectly. | `src/verify/verifier.py` | No code path from memory to the verifier. Report the model's own order next to service order so an indirect effect is visible. |
| C13 | Opus 5.5 accepts no `temperature`, so runs are not repeatable; the report's numbers are single runs. A learning curve needs a noise estimate. | `src/llm/orchestrator.py` | Run the empty arm twice (§7.3). |
| C14 | `bench/benchmark.py` has three hard-wired sets and builds settings globally; arms that differ only in memory need the store passed in. | `bench/benchmark.py`, `src/analysis.py` | Pass the store as an argument, like `client` (§5.5). |
| C15 | Cases would put NMRShiftDB2 peak lists and structures into a public repository. `bench/results*/` already does. | — | **Decided:** check the licence before committing any case. If redistribution is not allowed, the store lives outside git (§12). |

---

## 3. Design at a glance

```
                       offline, human in the loop
 NMRShiftDB2 record ──► propose ──► memory/pending/ ──► human approves ──► memory/cases/
                                                          (or rejects)        │  read-only
                                                                              ▼
 request ─► ingest/process ─► normalized PeakTable ─► retrieve k cases ─► build user message
                                                       (formula, DoU,        │  [worked examples]
                                                        shift regions,       │  [peak table]
                                                        solvent)             │  [context block]
                                                                             ▼
                                              Claude (unchanged call) ─► sanitize ─► verify ─► rank
                                                                             (no memory input)
```

---

## 4. Stored case: contents and on-disk format

### 4.1 Layout

```
memory/
  cases/<case_id>.json        approved cases; the only directory the service reads
  pending/<case_id>.json      proposed, not yet approved; never read by the service
  rejected/<case_id>.json     rejected, kept with the reason
  manifest.json               store version, list of case ids with SHA-256 of each file
  APPROVALS.log               append-only: date, case id, decision, approver, note
```

One JSON file per case, so a pull request shows each case as its own diff. Data lives in `memory/`, code in `src/memory/`.

### 4.2 Case schema (`MemoryCase`, Pydantic, `extra="forbid"`)

| Field | Type | Notes |
|---|---|---|
| `case_id` | str | `mem-` + 6 digits, assigned at proposal, never reused |
| `schema_version` | str | case-format version |
| **Provenance** | | constraint 5 |
| `provenance.source` | str | `nmrshiftdb2` in v1; later `user-confirmed` |
| `provenance.source_record_id` | str | NMRShiftDB2 id |
| `provenance.spectrum_ids` | dict | which ¹H and ¹³C spectra of the record |
| `provenance.truth_basis` | str | how the structure is known, e.g. "database-assigned structure" |
| `provenance.extracted_by` | str | script name and git commit that produced the file |
| `provenance.proposed_on` | date | |
| **Approval** | | constraint 4 |
| `approval.approved_by` | str | the human's name, typed at approval |
| `approval.date_added` | date | UTC date of approval; constraint 5 |
| `approval.note` | str \| null | never rendered into a prompt |
| **Query-side features** | | only what exists at proposal time; constraint 2 |
| `formula` | str | |
| `dou` | float | from the formula |
| `carbon_count`, `heavy_atoms` | int | from the formula |
| `solvent` | Solvent \| null | null when unreported |
| `h_peaks` | list of {shift_ppm, integral \| null} | analyte peaks as the service would see them after normalization |
| `c_peaks` | list of float | |
| **Answer** | | never used for retrieval |
| `smiles`, `inchikey` | str | RDKit-canonical |
| `murcko_scaffold`, `generic_scaffold` | str | stored for the leakage check, recomputed and compared at load |
| `assignments` | list of {nucleus, shift_ppm, environment} \| null | from the database's atom assignments; no invented reasoning |
| **Agent outcome** | | required in v1; see C10 and §7.2a |
| `agent_outcome.collected_on` | date | |
| `agent_outcome.model_id`, `prompt_version`, `settings_hash` | str | must equal the frozen eval configuration; the loader rejects a mismatch |
| `agent_outcome.status` | `solved` \| `missed` \| `no_answer` | `solved`: the agent's first candidate was the true structure; `missed`: it was not; `no_answer`: timeout or error |
| `agent_outcome.proposed_smiles` | list of str | the agent's candidates in its order (empty for `no_answer`) |

The loader rejects a case that fails validation, lacks approval fields, is missing from the manifest, or whose file hash or recomputed scaffold differs from what is stored. A rejected file stops startup in benchmark mode and is skipped with a logged error in service mode.

`memory_version` = SHA-256 over the sorted (case id, file hash) pairs, first 12 hex characters.

---

## 5. Retrieval and injection

### 5.1 Inputs

The normalized `PeakTable` (analyte-role peaks only) and the `SampleContext`. Nothing from any candidate structure, and nothing from the verifier.

### 5.2 Retrieval key

Only what exists at proposal time (constraint 2). The main signal is the spectrum itself, matched peak by peak; coarse region counts lose too much.

| Feature | Source | How it is compared | Without a formula |
|---|---|---|---|
| ¹³C shift list | ¹³C peaks | one-to-one peak matching (Hungarian assignment, `scipy`, already a dependency) within **2.0 ppm**; score = matched peaks weighted by closeness, divided by the larger list size | same |
| ¹H shift list | analyte ¹H peaks | the same matching within **0.15 ppm** | same |
| Element counts (C, H, N, O, S, halogens) | formula | normalized absolute difference | dropped |
| Degree of unsaturation | formula | normalized absolute difference | dropped |
| Solvent | context | soft: a match adds a small bonus; unknown or different is neutral | same |

Peak matching is what makes cases from different scaffolds still useful: two molecules that share a methyl ester, a prenyl group or a methoxyaryl unit share those peaks, whatever their core.

### 5.3 Scoring and selection

- Similarity is a fixed weighted sum of per-feature closeness (the two peak-matching scores, normalized absolute differences for element counts and DoU, a small bonus for a solvent match). No embedding model, no network call, no new dependency. With a few hundred cases a full scan, including peak matching, takes milliseconds.
- Deterministic. When two cases score within **0.02** of each other, a `missed` case is preferred over a `solved` one (the agent's weak spots first); remaining ties break on `case_id`.
- **k = 3** cases at most. At most one case per Murcko scaffold among those injected, so three near-identical examples cannot crowd the prompt.
- A case below a minimum similarity is not injected; a query can get fewer than three, or none.
- **Weights and the threshold are fixed before any eval model call.** They are tuned without model calls on the tuning half of the memory-side pool, then frozen; the check half decides go or no-go (§7.0). No eval molecule takes part.

### 5.4 Size limits (prompt-length control)

- Each rendered case: formula, DoU, solvent if known, ¹H shifts with integrals where known, ¹³C shifts, the correct SMILES, and one assignment line per signal where the database assigns it.
- A `missed` case is rendered **with the contrast**: the agent's earlier first candidate is shown and labelled wrong, then the correct structure with its assignments. No explanation of *why* it was wrong is generated; only stored facts are rendered, so nothing invented reaches the prompt.
- A `solved` or `no_answer` case is rendered as a plain worked example.
- Estimated 250-550 tokens per case for a 20-30 heavy-atom molecule (to be measured at implementation).
- Hard cap on the whole example block: **2,000 estimated tokens**. Cases are added best-first while they fit; a case that does not fit is skipped, not truncated.
- The example block is built after the peak table and context block are sized. If the three together would exceed `MAX_USER_TOKENS_ESTIMATE`, examples are dropped one by one until they fit. Memory never raises `PromptTooLarge` (C3).

### 5.5 The hook

| Step | Location today | Change |
|---|---|---|
| Retrieval | — | new `src/memory/` (loader, features, `retrieve(table, context, k) -> list[RetrievedCase]`) |
| Where it is called | `prepare_request()` in `src/prompting/builder.py`, after `normalize_peaklist` in both branches (peak list and upload) | call `retrieve` when a store is supplied |
| Where it is injected | `build_user_message(table, context, max_candidates)` | new optional argument `examples`; message order becomes **examples block, peak table, context block**. The task statement stays last. |
| How the store arrives | `run_analysis(request, settings, client, …)` in `src/analysis.py` | new optional argument `memory: MemoryStore \| None = None`, passed down to `prepare_request`. No global state, so benchmark arms can differ (C14). |
| Service wiring | `src/api/analyze_routes.py` dependencies | a `get_memory()` dependency that loads `memory/cases/` once if `NMR_AGENT_MEMORY_DIR` is set; unset means disabled and behaviour identical to today |
| System prompt | `src/prompting/templates.py` rule list | one new rule: the examples are other molecules, shown for the reasoning pattern; some show an earlier wrong proposal next to the correct answer; never copy a structure from them; the answer must fit the present peak table and formula. `PROMPT_VERSION` bumps (C2). |
| Repair round | `Conversation.repair()` | no change. History is append-only, so the examples remain in the first user turn; nothing is retrieved again. |
| Verifier, scoring, ranking | `src/verify/`, `_verify`, `final_confidence` | **no change, no new input** (constraint 1) |
| Response and trace | `Versions`, `AnalysisResponse`, `trace` dict in `run_analysis` | `versions.memory_version` ("none" when disabled); `memory_cases: [{case_id, similarity}]`; the same in the trace. `SCHEMA_VERSION` bumps (C11). |
| `/v1/process` | — | no change; it never builds the model prompt |

New settings: `NMR_AGENT_MEMORY_DIR` (unset by default), `NMR_AGENT_MEMORY_K` (3), `NMR_AGENT_MEMORY_MAX_TOKENS` (2000).

Prompt caching is unaffected: the system block is unchanged between requests, and examples add roughly 1-2k uncached input tokens (under $0.01 per analysis at list price).

---

## 6. Approval flow (constraint 4)

Nothing enters `memory/cases/` except through the approve command, run by a person.

1. **Propose.** `bench/memory.py propose --from <selection file>` builds case files from database records and the collected agent outcomes (§7.2a) into `memory/pending/` and writes a review sheet (Markdown): for each case the formula, solvent, peak lists, the structure, its scaffold, the agent's outcome and proposals, and the leakage-check result against every frozen eval selection. A case that fails the leakage check is not written to `pending/` at all.
2. **Review.** A person reads the sheet and checks, at least: the structure matches the formula; the peak list is plausible for it; for a `missed` case, that the agent's proposal really is wrong and the miss is not a database problem (a wrong structure, missing peaks, or a proposal that differs only in stereochemistry); nothing personal or confidential is in it. A doubtful `missed` case is rejected, not relabelled.
3. **Approve or reject.** `bench/memory.py approve <case_id>… --by "<name>"` asks for a typed confirmation, moves the file to `cases/`, fills `approval.*`, updates `manifest.json`, and appends to `APPROVALS.log`. `reject <case_id> --reason "…"` moves it to `rejected/`. There is no flag that skips the confirmation, and the service has no write path to the store.
4. **Commit.** The approved files, manifest and log are committed; the git history is the second record of who added what and when.

Rule for AI agents working in this repository: they may run `propose`; they must not run `approve`. This is a policy, not something the code can enforce, and belongs in the contributor notes.

For the experiment this means a person approves up to 200 cases (§7.2) before any eval model call. Batches of 25 with one review sheet each keep that manageable.

---

## 7. Evaluation plan

Existing methodology is kept: fixed seeds; every selection written to disk before any model call and never changed; a first-attempt timeout counts as a miss; every number comes from the report command.

The evaluation runs in three steps, each a gate for the next: an offline check with no model calls (§7.0), stage 1 (§7.3, four arms), and stage 2 (§7.3, the learning curve) only if stage 1 supports the hypothesis.

### 7.0 Offline go/no-go (no model calls, no eval data)

Question: does retrieval find cases whose *structures* are closer to the query's true structure than random cases are? If not, injecting them cannot plausibly help the model, and the experiment stops before any cost.

- **Data:** the memory-side candidate pool only (§7.2). It is split by Murcko scaffold, with a fixed seed, into a **tuning half** and a **check half**. No `MEM-EVAL` molecule is used.
- **Tuning:** the retrieval weights and the minimum-similarity threshold (§5.3) are set on the tuning half only, then frozen and hashed.
- **Check:** for each check-half molecule as a query, retrieve k = 3 cases from the rest of the check half, excluding its own scaffold, and draw 3 random cases (seeded) from the same set. Score each set two ways against the query's true structure:
  - **similarity:** the highest Morgan (radius 2, 2048 bits) Tanimoto similarity between a returned case and the truth;
  - **fragment coverage:** the share of the truth's functional groups (the SMARTS groups the verifier already defines in `src/verify/smarts.py`) that appear in at least one returned case. This is what an example can teach across scaffolds.
- **Go**, fixed now: retrieved sets beat random sets on both measures, by at least **0.10** in mean similarity and **0.15** in mean fragment coverage, and score higher than the random set on similarity for at least **60%** of queries. Otherwise **no-go**: stop and report the result.
- The check is cheap and deterministic; its output file is committed with the selections.

### 7.1 Held-out set (`MEM-EVAL`)

- 100 molecules from the named-solvent pool: 50 with 11-20 heavy atoms, 50 with more than 20. No molecules with 10 or fewer (they are at the ceiling).
- Excluded: the 90 molecules of the earlier sets and every molecule sharing a ring scaffold with them; all acyclic molecules and all benzene-scaffold molecules (C6, C7).
- At most 2 molecules per scaffold.
- Feasibility *(census, simulation)*: 265 eligible molecules (125 and 140 in the two size bins) over 206 scaffolds; a seeded draw filled 50 + 50 using 84 scaffolds.
- Input to the service as in the main set: shifts, formula, solvent; integrals only where the assignment is complete.

Consequence to state in the report: `MEM-EVAL` has no acyclic and no benzene-only molecules, and is harder than the main set by construction. From the main set's rates by size (19/24 and 3/14), the expected empty-memory top-1 is roughly 50%.

### 7.2 Memory pool and sizes

- Candidates: every pool molecule (named and unreported solvent) that is not in `MEM-EVAL`, not in the earlier 90, and shares no Murcko scaffold with `MEM-EVAL`. About 4,700 *(simulation)*, of which about 385 have more than 20 heavy atoms.
- From these, one seeded, **ordered** list of 200 cases, stratified 40% above 20 heavy atoms, 40% 11-20, 20% 10 or fewer, written before any model call.
- Nested sizes: the first 25, 50, 100 and 200 of that list. Each smaller memory is a subset of the larger ones.
- A case the reviewer rejects is replaced by the next unused molecule in the seeded order; the final list is frozen and hashed before the first eval call.

### 7.2a Collecting the agent's outcomes (error-driven memory)

- `bench/memory.py collect` runs the analysis once on each of the 200 ordered memory molecules (and on any replacements), **with memory disabled**, using exactly the frozen eval configuration: the same prompt version, model id and settings as the stage-1 arms. It never runs on a `MEM-EVAL` molecule.
- Each result is stored as the case's `agent_outcome` (§4.2): `solved` if the first candidate is the true structure, `missed` if not, `no_answer` for a timeout or error.
- Every case is kept, solved or missed; memory is not restricted to mistakes. Missed cases are distinguished by how they are rendered (§5.4) and by retrieval's tie-break (§5.3).
- A single run is a noisy label: a molecule missed once may be solved on another run. This is accepted and stated in the report.
- If the frozen configuration changes after collection, the outcomes are recollected; the loader rejects cases whose outcome configuration does not match.

### 7.3 Arms

**Stage 1** (runs only after a "go" in §7.0):

| Arm | Memory | Purpose |
|---|---|---|
| M0-a, M0-b | empty (twice) | baseline and run-to-run noise (C13) |
| M200 | all 200 approved cases, retrieved | the effect of memory |
| R200 | the same 200 cases, but k cases chosen **at random** (seeded) instead of retrieved | separates "relevant cases help" from "any examples or a longer prompt help" |

400 analyses. The verdict on the hypothesis (§8) comes from stage 1 alone.

**Stage 2** (runs only if stage 1 supports the hypothesis):

| Arm | Memory | Purpose |
|---|---|---|
| M25, M50, M100 | the first 25, 50, 100 of the ordered list, retrieved | the learning curve, with M0 and M200 from stage 1 as its ends |

300 analyses.

All arms in both stages: the same 100 molecules, the same prompt version, model and settings, k = 3, one attempt each. Stage 2 must run on the same prompt version, model id and settings as stage 1; if any of them has changed, stage 2 reruns M0-a and M200 as well.

### 7.4 Measures

- **Primary:** service top-1 on `MEM-EVAL`, per arm.
- **Secondary, pre-registered:** **best-candidate similarity**, the Morgan Tanimoto similarity between the service's first candidate and the true structure, averaged per arm and compared M200 against M0 with a paired Wilcoxon signed-rank test (a near-miss counts for something, so smaller effects become visible); true structure proposed in any round (the direct target of memory); top-3; the model's own order top-1 (C12); first-attempt timeouts; mean model time; cost per molecule; examples injected per query.
- **Curve (stage 2):** top-1 against memory size 0, 25, 50, 100, 200, with the M0-a/M0-b spread drawn as the noise band and R200 as a reference line.
- **Paired comparison:** per-molecule discordant counts between M200 and M0 (both replicates), exact McNemar test.
- **Leakage diagnostic:** for each eval molecule, the highest Tanimoto similarity between its true structure and any case injected for it; top-1 gain split at 0.5 similarity. If the gain sits only in the high-similarity half, the effect is near-duplicate retrieval.

### 7.5 Cost and time (estimate, to approve before the run)

The main set averaged $0.089 and 42 s per molecule; `MEM-EVAL` is larger molecules plus up to 2k more input tokens, so assume about $0.13 and 50 s.

| Step | Analyses | Estimate | Spend cap |
|---|---|---|---|
| §7.0 offline check | 0 | $0 | — |
| §7.2a outcome collection on memory molecules | about 200 | about $26 | $35 |
| Stage 1 | 400 | about $52, about 6 h of model time | $65 |
| Stage 2 (only if stage 1 supports H) | 300 | about $39 | $50 |

Totals: about **$0** if the offline check says no-go; about **$78** if stage 1 does not support the hypothesis; about **$117** with stage 2.

### 7.6 Power, stated before the run

With 100 paired molecules, a real gain below about 8-10 points will probably not reach significance: for example 12 molecules gained against 2 lost is significant, 8 against 3 is not. A flat result therefore means "no large effect", not "no effect".

---

## 8. Falsification criteria (fixed before the run)

Let Δ = top-1(M200) − mean top-1(M0-a, M0-b), in molecules out of 100, and N = |top-1(M0-a) − top-1(M0-b)|.

| Outcome | Condition |
|---|---|
| **H supported** | Δ ≥ +8, and Δ > 2N, and exact McNemar p < 0.05 for M200 against each M0 replicate, and top-1(M200) − top-1(R200) ≥ +4 |
| **H falsified** | Δ ≤ +3 or Δ ≤ N (no gain beyond noise); **or** top-1(R200) ≥ top-1(M200) (retrieval adds nothing over random examples); **or** the gain is confined to eval molecules whose injected cases exceed 0.5 Tanimoto to the truth (near-duplicate effect, not learning) |
| **Inconclusive** | anything else; report as such, with the power statement |
| **Harmful, stop** | first-attempt timeouts in M200 exceed the M0 mean by more than 5 molecules, or mean cost per molecule rises by more than 30% |

All of these are evaluated on stage 1. Stage 2 is descriptive: it shows how the gain grows with memory size, and it cannot change the stage-1 verdict. A curve that is flat or falls between 25 and 200 cases is reported as such.

No arm is rerun, no threshold is moved, and no retrieval weight is changed after the first eval call. A second experiment needs a new held-out set.

---

## 9. Leakage: how separation is verified (constraint 3)

A check command (`bench/memory.py check`) and a test that runs in CI over the committed selections and store. It fails unless all of these hold:

1. No memory case shares an InChIKey connectivity block with any molecule of `MEM-EVAL` or of the earlier three sets.
2. No memory case shares a canonical Murcko scaffold with any `MEM-EVAL` molecule. Scaffolds are recomputed from the stored SMILES with RDKit, not read from the file.
3. `MEM-EVAL` contains no molecule with an empty scaffold and none with the bare benzene scaffold.
4. No memory case has Morgan-fingerprint Tanimoto similarity of 0.70 or more to any `MEM-EVAL` molecule. In the simulation 2 of 100 eval molecules had such a neighbour after the scaffold split, so the scaffold rule alone is not quite enough.
5. No proposed SMILES stored in any case's `agent_outcome` matches a `MEM-EVAL` molecule by InChIKey connectivity block or Murcko scaffold, so a wrong earlier proposal cannot show the model a test answer.
6. At benchmark time, for each eval molecule, the injected cases are checked against its truth again (rules 1, 2, 4) and the result is saved in the trace; any violation aborts the run.

Reported, not enforced: the number of generic-scaffold (atom-type-agnostic) overlaps, and the distribution of nearest-neighbour similarity.

What the simulation says about the constraint itself: a record-level split left a median nearest-neighbour similarity of 0.39 (90th percentile 0.64, 6 eval molecules at 0.70 or above); the scaffold split gave 0.36 (0.58, 2 molecules). The scaffold split removes the closest pairs and little else. It is worth doing, and it also shows that most eval molecules have no close relative in memory at all, which is why the falsification criteria matter.

Not addressed by any of this: the model may have seen NMRShiftDB2 in training. That affects every arm equally, so it biases the level of the curve, not its slope.

---

## 10. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| **Leakage through near-duplicates** | An apparent learning curve that is really lookup | §9 checks; Tanimoto cap; the similarity-split diagnostic; R200 control |
| **Tuning on the eval set** | Optimistic result | Weights and threshold tuned on memory-side data only, without model calls; frozen before the first eval call |
| **Prompt-length growth** | More timeouts (already 8% of first attempts in the main set), cost, latency | 2,000-token cap; k = 3; examples dropped first; "harmful" stop rule; examples are uncached but small |
| **Anchoring** | The model copies an example's scaffold or substitution pattern; could lower accuracy on unlike molecules | New system rule; one case per scaffold; minimum-similarity threshold; R200 shows whether irrelevant examples hurt |
| **Ceiling of the approach** | 9 of 16 main-set misses were "never proposed" on large molecules; three examples may not change that | Stated power; falsification criteria; the secondary "proposed at all" measure |
| **Wrong structures in the database** | A wrong worked example teaches a wrong pattern | Human review; `truth_basis` recorded; cases can be removed, which changes `memory_version` |
| **Approval burden and error** | 200 cases reviewed by hand; rubber-stamping | Batches of 25 with review sheets; automatic checks run before a case reaches `pending/` |
| **Indirect effect on ranking** | Memory shifts model confidence, which is 15% of the final score | Report model-order top-1 beside service top-1 (C12) |
| **Unreported-solvent cases** | Solvent in the key is uninformative for most of memory (C5) | Soft solvent feature; report results with and without the solvent term if time allows |
| **Prompt injection through cases** | Text in a case steers the model | Cases render only structured fields; free-text `approval.note` is never rendered; cases are human-approved |
| **Run-to-run variance** | A real difference hidden, or noise read as a gain | Two empty-memory runs; criteria expressed against the measured spread |
| **Future user-derived cases** | Privacy and consent for stored spectra | Out of scope for v1: only public database records enter the store |
| **Deployment drift** | Store missing from the image, or stale | `memory_version` in every response; service starts with memory disabled if the directory is absent |

---

## 11. Out of scope for v1

- Any feedback route that lets the running service add cases.
- Using memory in the verifier, the scorer, or the repair report.
- Learned embeddings or a vector database.
- A memory of mistakes only. Considered and not chosen: narrower coverage, more exposure to database errors, about $75 to collect. It can be added later as one more stage-1 arm (about $13 plus collection).

---

## 12. Decisions (recorded 2026-11-14)

| # | Question | Decision | Reason |
|---|---|---|---|
| 1 | Acyclic and benzene-only molecules (C6, C7) | Excluded from `MEM-EVAL`, allowed in memory | They cannot be split by scaffold. The test targets larger ring systems, where the misses are. The report states that `MEM-EVAL` is harder by construction. |
| 2 | Solvent (C5) | Soft feature; unreported-solvent cases allowed | 90% of eligible memory has no reported solvent; excluding it would leave memory with few large molecules |
| 3 | Held-out set (C8) | A new set of 100 (§7.1) | The main 50 has 12 molecules at the ceiling and was used to find the verifier range fixes |
| 4 | Budget | Staged: offline check (free), outcome collection (about $26), stage 1 (400 analyses, about $52), stage 2 (300, about $39) only if stage 1 supports H | No money is spent on the curve if memory does not help |
| 5 | Thresholds | As in §7.0 and §8 | Strict enough that a "supported" result is believable at 100 molecules |
| 6 | Licence (C15) | Check the NMRShiftDB2 terms before committing any case. If redistribution is not allowed, keep `memory/cases/` out of git (gitignored), commit only `manifest.json` with the case ids and hashes, and rebuild the cases from the database with the propose command | The experiment can run either way |

| 7 | What memory stores | Every memory molecule with the agent's own outcome, marked `solved` or `missed`; missed cases shown with the contrast and preferred on near-ties (§5.3, §5.4, §7.2a) | Keeps memory broad, where coverage is the weakest link, while targeting the agent's weak spots |
| 8 | Retrieval | Peak-by-peak spectral matching plus formula features (§5.2) | Coarse region histograms lose the fragment-level signal that crosses scaffolds |
| 9 | Go/no-go | Similarity and fragment coverage, both required (§7.0) | Fragment coverage measures what an example can teach across scaffolds |
| 10 | Extra measure | Best-candidate similarity, pre-registered secondary (§7.4) | Detects smaller effects than top-1 alone |

Expectation, stated before any work: with plain database cases and coarse retrieval, a small or inconclusive effect was the likelier outcome (median nearest-memory similarity 0.36 *(simulation)*; the hard misses were never proposed; top-1 detects only gains of about 8 points). Decisions 7-10 target each of those limits, so a measurable gain is plausible, but not assured. A controlled negative result is still a usable outcome.

## 13. Order of work

```
1. Licence check (§12.6)                                   no cost
2. Build the offline parts: case schema, store loader,     no model calls
   spectral retrieval, leakage checks,
   propose/collect/approve/check CLI, tests (the leakage
   check runs in CI)
3. Write and freeze the selections: MEM-EVAL, the          no model calls
   ordered memory list, the tuning/check split
4. Offline go/no-go (§7.0)                                 free
     no-go ──► stop; report the retrieval result
5. Prompt change (one new system rule), version bump,      no model calls
   response fields; freeze model, prompt and settings
6. Collect the agent's outcomes on the memory              about $26
   molecules, memory disabled (§7.2a)
7. Approve the 200 cases (8 batches of 25, §6);            reviewer time
   hash the store
8. Stage 1: M0-a, M0-b, M200, R200                         about $52
     not supported ──► report the negative or inconclusive result
9. Stage 2: M25, M50, M100                                 about $39
10. Report: bench/memory-report.md, every number from
    the compare command
```

Steps 2-7 change no behaviour when memory is disabled: with `NMR_AGENT_MEMORY_DIR` unset, `/v1/analyze` is identical to today except for the new prompt version and the `memory_version: "none"` field.
