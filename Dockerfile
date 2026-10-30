FROM python:3.12.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

COPY requirements-lock.txt pyproject.toml setup.py README.md ./
COPY src ./src
COPY ml ./ml
COPY scripts ./scripts
RUN pip install -c requirements-lock.txt ".[worker]" \
    && useradd --system --uid 10001 --no-create-home app

USER app
EXPOSE 8000

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
