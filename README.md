# NMR-Agent

A containerized FastAPI microservice that turns a 1D NMR dataset into a verified structural hypothesis.

The design is an LLM with deterministic guardrails:

1. **Ingest** a Bruker experiment, a JCAMP-DX spectrum, or a peak list.
2. **Process** raw data with [nmrglue](https://github.com/jjhelmus/nmrglue) into a compact peak table.
3. **Reason** over the peak table with Claude, using structured outputs validated by Pydantic.
4. **Verify** every claim with RDKit and literature chemical-shift rules.
5. **Return** ranked candidate structures with an audit trail and a confidence computed from the verifier, not from the model's self-report.

## Status

Early development. Only the foundation exists; the service does not analyze spectra yet.

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, Bruker FID processing, tests, Docker, CI workflow | Done, review closed |
| 1 | Pydantic schemas and the RDKit verifier | Done, review closed |
| 2 | Peak-list analysis with the LLM | Not started |
| 3 | Raw-data ingestion (Bruker zip, JCAMP-DX) | Not started |
| 4 | Hardening and deployment | Not started |
| 5 | Benchmark on NMRShiftDB2 and report | Not started |

What works today:

- `process_bruker_fid` in [src/processing/bruker.py](src/processing/bruker.py): digital-filter removal, apodization, zero-fill, FFT, ACME auto-phasing, spectrum reversal, median baseline correction, ppm axis.
- `POST /v1/verify`: checks candidate SMILES against a peak table with rules V01-V11 (no LLM). `accepted` means consistent with the supplied data, not proven correct: isomers can also be accepted.
- `GET /v1/reference/shifts`: the chemical-shift, SMARTS and solvent tables with their sources.
- `GET /healthz` and the generated OpenAPI document.
- Tests against a synthetic Bruker experiment and a real, TopSpin-processed dataset.

The first GitHub Actions run (lint, tests, Docker build, container smoke check) passed.

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
bench/         benchmark scripts (large downloads go in bench/data/, not tracked)
```

Of these, `api/`, `core/`, `processing/` and `verify/` contain code so far; `src/schemas.py` holds all Pydantic models.

## Testing notes

- `tests/processing/test_bruker_roundtrip.py` writes a synthetic Bruker experiment and checks that every programmed peak is recovered and not mirrored. It is the regression guard for running nmrglue 0.12 on NumPy 2.
- `tests/processing/test_bruker_real.py` compares the pipeline with the TopSpin-processed spectrum of a real dataset.
- Tests that call the live Anthropic API are marked `live` and are excluded by default.

## Data attribution

The real-data test fixture is experiment 10 of "Wadpressine" (Ben Bakrim et al., Zenodo [10.5281/zenodo.1233301](https://doi.org/10.5281/zenodo.1233301)), released under CC-BY 4.0. See [tests/fixtures/bruker/wadpressine_1h/README.md](tests/fixtures/bruker/wadpressine_1h/README.md).

## License

No license has been chosen yet.
