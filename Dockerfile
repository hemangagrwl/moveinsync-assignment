FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY deploy ./deploy

RUN useradd --system --uid 10001 app
USER app

ENV API_WORKERS=1     PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
EXPOSE 8080
# Each worker process runs its own recompute thread; jobs are claimed by compare-and-set, so
# several workers (or replicas) never process the same job. Requests are logged by the app.
CMD ["sh", "-c", "rm -rf $PROMETHEUS_MULTIPROC_DIR && mkdir -p $PROMETHEUS_MULTIPROC_DIR && exec uvicorn backfill.api.app:app_from_env --factory --host 0.0.0.0 --port 8080 --workers ${API_WORKERS} --no-access-log"]
