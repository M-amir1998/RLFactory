# syntax=docker/dockerfile:1
# RLFactory CLI image: CPU-only core (no torch/TRL), non-root, no Docker socket.
# Build: docker build --build-arg RLFACTORY_GIT_SHA="$(git rev-parse HEAD)" -t rlfactory:local .
# Run:   docker run --rm -v "$PWD/results:/data/results" rlfactory:local experiment run configs/experiments/ci-e2e.yaml
ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:0.11.21 AS uv

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim AS runtime
ARG RLFACTORY_GIT_SHA=unknown
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RLFACTORY_GIT_SHA=${RLFACTORY_GIT_SHA} \
    RLFACTORY_CONFIG_DIR=/app/configs \
    RLFACTORY_RESULTS_DIR=/data/results \
    HF_HUB_DISABLE_TELEMETRY=1
RUN groupadd --system --gid 10001 rlfactory \
    && useradd --system --uid 10001 --gid rlfactory --create-home --home-dir /home/rlfactory rlfactory \
    && mkdir -p /data/results \
    && chown rlfactory:rlfactory /data/results
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs /app/configs
USER rlfactory
# Inside this container generated code runs in the subprocess sandbox; the container itself is the
# outer boundary. Never mount /var/run/docker.sock here: it would grant root-equivalent host access.
ENTRYPOINT ["rlfactory"]
CMD ["--help"]
