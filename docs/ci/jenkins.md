# Jenkins CI

RLFactory uses Jenkins, not GitHub Actions (see [ADR-007](../adr/ADR-007-jenkins-instead-of-github-actions.md)).
The repository contains no `.github/workflows/`; `scripts/check_repo_hygiene.py` fails the build if
one appears.

## Requirements for the agent

| Requirement | Why |
|---|---|
| Linux agent labelled `linux` | POSIX resource limits in the subprocess sandbox; `sh` steps |
| `uv` on `PATH` | dependency resolution from `uv.lock` (`uv lock --check`, `uv sync --frozen`) |
| Python 3.12 available to uv (or network for uv to fetch it) | `UV_PYTHON=3.12` |
| Docker CLI + daemon (optional) | only for the `Docker build` stage and Docker-sandbox tests |
| ~2 GB free disk, 16 GB RAM is plenty | stages run sequentially |

The pipeline needs **no** GPU, CUDA, model download, Hugging Face login or paid API.
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `CUDA_VISIBLE_DEVICES=""` are set globally; every
rollout uses the deterministic mock policy and tiny suites.

## Pipeline

`options { disableConcurrentBuilds(); timestamps(); timeout(60 min) }` and all stages are sequential.

| Stage | Command (abridged) |
|---|---|
| Checkout | `checkout scm`; clean previous reports |
| Dependency validation | `uv lock --check`, `uv sync --frozen`, `uv pip check` |
| Formatting | `uv run ruff format --check .` |
| Ruff | `uv run ruff check .` (includes flake8-bandit `S` rules) |
| Strict mypy | `uv run mypy` (strict + pydantic plugin; the agent checks the Linux branches) |
| Unit tests | `pytest tests/unit -m "not sandbox and not verifier and not store and not docker"` |
| Property tests | `pytest tests/property` (Hypothesis) |
| Integration tests | `pytest tests/integration -m "not trl and not tiny_model"` |
| Environment sandbox tests | `pytest tests/unit -m "sandbox and not verifier"` (Docker tests self-skip without daemon/image) |
| Verifier tests | `pytest tests/unit -m verifier` |
| Trajectory-store tests | `pytest tests/unit -m store` |
| Mock rollout E2E | `pytest tests/e2e/test_rollout_e2e.py` (CLI run, inspect, query, resume) |
| Mock post-training E2E | `pytest tests/e2e/test_cpu_e2e.py` + `rlfactory experiment run configs/experiments/ci-e2e.yaml`; coverage XML; `coverage report --fail-under=85` |
| Security scan | `scripts/secret_scan.py`, `scripts/check_repo_hygiene.py`; optional `pip-audit` (parameter `RUN_PIP_AUDIT`, needs network) |
| Package build | `uv build`, then the hygiene check on `dist/` |
| Docker build | CLI image + sandbox image; `docker run ... doctor --no-docker` (parameter `RUN_DOCKER_BUILD`) |

Every pytest stage appends to one coverage database (`--cov-append`), so the final coverage figure
covers the whole suite.

## Artefacts

Archived: JUnit XML (`reports/junit/*.xml`), `reports/coverage.xml`, the wheel and sdist, and the
CI experiment's `report.md`, `plots/*.svg` and `manifest.json`.

Never archived: `results/*/rollouts/` (trajectories), datasets, checkpoints, model weights. The
`cleanup` block deletes `results/` and `dist/` and removes the CI images.

## Tests that never run in Jenkins

- `pytest -m trl`: TRL config-compatibility checks; need the optional `trl` extra.
- `pytest -m tiny_model` with `RLFACTORY_RUN_TRL_SMOKE=1`: a real 2-step TRL SFT run on
  `HuggingFaceTB/SmolLM2-135M-Instruct` from a local Hugging Face cache.

Run them locally on a machine that already has the extra and the cached model:

```bash
UV_PROJECT_ENVIRONMENT=.venv-trl uv sync --extra trl
RLFACTORY_RUN_TRL_SMOKE=1 UV_PROJECT_ENVIRONMENT=.venv-trl uv run pytest -m "trl or tiny_model" tests/integration
```

## Reproducing CI locally

```bash
uv sync --frozen
uv run ruff format --check . && uv run ruff check . && uv run mypy
uv run pytest -p no:cacheprovider --cov=rlfactory
uv run python scripts/secret_scan.py && uv run python scripts/check_repo_hygiene.py
uv build
```
