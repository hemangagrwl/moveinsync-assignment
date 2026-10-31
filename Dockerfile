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

EXPOSE 8080
CMD ["uvicorn", "backfill.api.app:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8080"]
