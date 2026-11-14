# Contributing to RLFactory

## Development setup

```bash
uv sync                       # core dependencies + the dev group (pytest, hypothesis, ruff, mypy, ...)
uv run rlfactory doctor
```

Python 3.11+ is required; `.python-version` pins 3.12 for local work. The optional training stack
lives in the `trl` extra. Because `uv sync` removes packages that are not part of the requested
extras, keep it in a separate environment instead of toggling extras in `.venv`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-trl uv sync --extra trl          # torch, transformers, trl, peft, ...
UV_PROJECT_ENVIRONMENT=.venv-trl uv run pytest -m trl         # TRL compatibility / smoke tests
```

verl, OpenRLHF, vLLM and SGLang are intentionally not extras (conflicting pins); they belong in
isolated GPU-host environments. See [docs/references/post-training-stack.md](docs/references/post-training-stack.md).

## Quality gates

Every change must pass the same gates Jenkins runs:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy                          # strict, checks the platform you are on
uv run mypy --platform linux         # also required on Windows: the Jenkins agent is Linux
uv run pytest
python scripts/secret_scan.py        # tokens, keys, private key blocks, local absolute paths
python scripts/check_repo_hygiene.py # no workflows dir, no large or binary artefacts, no weights
```

`mypy` runs with `strict = true` and the pydantic plugin. Platform-specific code must use
`sys.platform` checks (which mypy understands) rather than `os.name`, so both platforms type-check.

### Test markers

| Marker | Meaning |
|---|---|
| `unit` | fast, isolated |
| `property` | Hypothesis property-based tests |
| `integration` | several components together (e.g. task -> policy -> environment -> verifier -> store) |
| `e2e` | CPU-only end-to-end pipelines |
| `sandbox` | executes generated code in a sandbox |
| `verifier` | verifier known-value tests |
| `store` | trajectory-store tests |
| `docker` | needs a running Docker daemon and the sandbox image; skipped otherwise |
| `trl` | needs the optional TRL stack; skipped otherwise |
| `tiny_model` | needs a tiny model already in the local Hugging Face cache; never downloads |

Normal CI must not need a GPU, CUDA, model downloads, Hugging Face authentication or paid APIs.
Tests that would need any of these are marked and skip cleanly.

## Engineering conventions

- Domain records are frozen pydantic models with `extra="forbid"`; configuration is validated YAML
  (`yaml.safe_load` only). No magic dictionaries for core fields.
- Keep modules cohesive. No `utils.py`, no global mutable state, no unbounded concurrency.
- Every component that scores or transforms data carries a name and version; configuration hashes
  are part of verifier and reward versions so lineage changes when settings change.
- Never let an infrastructure failure become training data; never present simulated numbers as
  model results.
- Comments explain *why*; avoid restating the code.

## Commits

- Conventional prefixes: `feat(scope):`, `fix(scope):`, `test:`, `docs:`, `ci:`, `chore:`.
- Stage explicit paths; review `git diff --cached` before committing.
- Commits are authored by the human contributor. Do not add `Co-authored-by`, `Generated-by` or
  similar AI attribution trailers.
- Never commit `.env` files, tokens, keys, credentials, model weights, checkpoints, datasets,
  rollout dumps (`results/` is ignored) or machine-specific absolute paths.
- Do not force-push or rewrite published history.

## CI

The project uses Jenkins only (see [docs/ci/jenkins.md](docs/ci/jenkins.md)). Do not add GitHub
Actions workflows; `scripts/check_repo_hygiene.py` fails if `.github/workflows/` exists.
