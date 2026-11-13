# NMR-Agent

A containerized FastAPI microservice that turns a 1D NMR dataset into a verified structural hypothesis.

The design is an LLM with deterministic guardrails:

1. **Ingest** a Bruker experiment, a JCAMP-DX spectrum, or a peak list.
2. **Process** raw data with [nmrglue](https://github.com/jjhelmus/nmrglue) into a compact peak table.
3. **Reason** over the peak table with Claude, using structured outputs validated by Pydantic.
4. **Verify** every claim with RDKit and literature chemical-shift rules.
5. **Return** ranked candidate structures with an audit trail and a confidence computed from the verifier, not from the model's self-report.

## Results

On 50 real molecules from NMRShiftDB2, given the ¹H and ¹³C shifts, the formula and the solvent, the correct structure was the first answer for **34 (68%)** and among the first three for **37 (74%)**. Every molecule with 10 or fewer heavy atoms was solved; 3 of 14 above 20 heavy atoms were. About $0.09 and 42 s per molecule. The verifier guarantees that returned structures are consistent with the formula and the spectrum, but on this data it does not raise top-1 accuracy over the model alone. Method, breakdowns and caveats: [bench/report.md](bench/report.md).

## What it does

| Route | In | Out |
|---|---|---|
| `POST /v1/analyze` | A peak list (JSON), or a Bruker zip or JCAMP-DX file (multipart: `file` + `metadata`) | Ranked candidate structures, each with its verifier rule ledger, plus warnings, token usage, cost and versions |
| `POST /v1/process` | The same inputs | The normalized peak table only (no model call) |
| `POST /v1/verify` | A peak table and candidate SMILES | Each candidate checked by rules V01-V11 (no model call) |
| `GET /v1/reference/shifts` | — | The chemical-shift, SMARTS and solvent tables with their sources |
| `GET /healthz` | — | Liveness |

How an analysis runs:

```
peak list / Bruker zip / JCAMP-DX
        │
        ▼
ingest + process (nmrglue) ──► normalized peak table (solvent, water, TMS tagged)
        │
        ▼
Claude, one structured-output call ──► candidate SMILES + assignments + claims
        │      (schema corrections, retries, refusal fallback)
        ▼
sanitize ──► verify every candidate (RDKit + literature shift rules, V01-V11)
        │
        ├── nothing accepted? one repair round with the failure report, then verify again
        ▼
rank by 0.85 × verifier score + 0.15 × model confidence ──► response
```

- `accepted` means consistent with the supplied data, not proven correct: same-formula isomers can also be accepted.
- `/v1/analyze` needs `ANTHROPIC_API_KEY`. Optional abuse controls: `NMR_AGENT_API_KEY` (header `X-API-Key`), `NMR_AGENT_PER_IP_PER_HOUR` (default 10) and `NMR_AGENT_DAILY_LLM_BUDGET` (model calls per day, default 100, reserved when a request is admitted). Counters are in memory and per process.
- Uploads are size-checked before parsing (body limits, zip and decompression limits, point limits).

Not built yet: demo mode, an async endpoint, client addresses from a trusted proxy, a rate limit on `/v1/process`, and a hosted deployment. The ¹³C path from a raw FID has not been validated on real data.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch it).

```bash
uv sync
uv run pytest -q
uv run uvicorn src.api.main:app --reload     # http://127.0.0.1:8000/docs
```

Docker:

```bash
docker build -t nmr-agent .
docker run --rm -e PORT=8080 -p 8080:8080 nmr-agent
curl localhost:8080/healthz
```

The image is about 800 MB uncompressed and runs as a non-root user.

## Layout

```
src/
  api/         FastAPI routes
  core/        settings and version constants
  ingest/      format detection and file-safety checks
  processing/  FID/spectrum -> peak table (pure NumPy functions)
  prompting/   prompt templates and peak-table rendering
  llm/         Anthropic client and structured-output calls
  verify/      RDKit checks, shift rules, scoring
tests/         unit tests, synthetic Bruker round trip, real-data fixture
bench/         benchmark harness, results and report (large downloads go in bench/data/, not tracked)
```

Of these, `api/`, `core/`, `ingest/`, `processing/`, `prompting/`, `llm/` and `verify/` contain code; `src/schemas.py` holds all Pydantic models.

## Testing notes

- `tests/processing/test_bruker_roundtrip.py` writes a synthetic Bruker experiment and checks that every programmed peak is recovered and not mirrored. It is the regression guard for running nmrglue 0.12 on NumPy 2.
- `tests/processing/test_bruker_real.py` compares the pipeline with the TopSpin-processed spectrum of a real dataset.
- Tests that call the live Anthropic API are marked `live` and are excluded by default.

## Data attribution

The real-data test fixture is experiment 10 of "Wadpressine" (Ben Bakrim et al., Zenodo [10.5281/zenodo.1233301](https://doi.org/10.5281/zenodo.1233301)), released under CC-BY 4.0. See [tests/fixtures/bruker/wadpressine_1h/README.md](tests/fixtures/bruker/wadpressine_1h/README.md).

## License

No license has been chosen yet.
