# Minimal image for DockerSandbox executions of untrusted generated code.
# Build: docker build -f docker/sandbox.Dockerfile -t rlfactory-sandbox:local docker
# Use:   sandbox: {kind: docker, docker_image: rlfactory-sandbox:local}
# DockerSandbox adds the runtime controls: --network none, --user 65534:65534, --read-only,
# a noexec tmpfs, --cap-drop ALL, no-new-privileges, cgroup memory/CPU/PID limits, ulimits, --init.
# Pin the base image by digest for production use.
FROM python:3.12-slim
RUN rm -rf /root/.cache /var/lib/apt/lists/* \
    && python -c "import json, importlib.util, copy, math, subprocess"
USER 65534:65534
WORKDIR /sandbox
