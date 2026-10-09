FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
# libxrender1 only needed if molecule images are rendered; harmless to include (~2 MB)
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN pip install uv && uv sync --frozen --no-dev --no-install-project
# The virtualenv stays root-owned and world-readable; a recursive chown would copy it into a new layer.
RUN useradd -m appuser
COPY --chown=appuser src ./src
USER appuser
# Render injects PORT (default 10000); Cloud Run injects PORT=8080.
CMD ["sh", "-c", "/app/.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 75"]
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/healthz')"
