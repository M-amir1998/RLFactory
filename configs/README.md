# Configuration library

| Directory | Loaded by | Contents |
|---|---|---|
| `tasks/` | task registry (auto-discovered) | task suites: generator sources, seeds, verifier overrides |
| `environments/` | copied into `sandbox:` sections; `environments check --sandbox` | sandbox profiles (subprocess, Docker) |
| `rollouts/` | `rlfactory rollout run` | rollout jobs (suites, policy, generation, retries) |
| `datasets/` | `rlfactory dataset build-*` | dataset jobs (source run, builder settings, held-out suites) |
| `training/` | `rlfactory train` | training jobs (backend, model, hyper-parameters) |
| `evaluation/` | `rlfactory evaluate` | evaluation jobs (suites, policy, gates, baseline) |
| `experiments/` | `rlfactory experiment plan/run` | full DAGs tying everything into one manifest |
| `remote/` | `rlfactory remote ...` | SSH host profiles; only `*.example.yaml` is committed |

Every file is validated against a strict pydantic schema (unknown keys are errors). Configs never
contain secrets: API keys and tokens are referenced by environment-variable *name* only.
GPU configs use `revision: null` on purpose; pin a commit SHA before any remote run.
