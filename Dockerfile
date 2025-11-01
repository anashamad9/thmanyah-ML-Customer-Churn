FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN pip install --upgrade pip && pip install uv

COPY pyproject.toml README.md ./

RUN uv pip install --system ".[dev]"

COPY src ./src
COPY api ./api
COPY scripts ./scripts
COPY configs ./configs
COPY monitoring ./monitoring
COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
