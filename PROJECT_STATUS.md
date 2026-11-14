# Project status

Snapshot of 2026-11-14. Measurements were taken on commit `dde46b0` (the code has not changed
since; later commits touch documentation only). Everything model-related below is simulated or a
wiring check: **no real model has been trained or evaluated, and no GPU has been used.**

## Overview

| Field | Status |
|---|---|
| Repository URL | <https://github.com/M-amir1998/RLFactory> (public) |
| Version | `0.1.0.dev0`, unreleased, no tag |
| Real GPU status | **Not run.** Real GPU post-training experiments are pending; no paid compute has been started |
| CPU E2E result | `cpu-e2e` (mock policy, mock trainer): all 12 stages in about 26 s; verified accuracy on 78 held-out trajectories, base 56.4% (95% CI 45.4-66.9%) -> candidate 64.1% (53.0-73.9%); gate passed, promotion `validated`; flywheel 56.4% -> 62.8% -> 71.8%. **Simulated**: the mock trainer's improvement is a configured knob, and even these intervals overlap. `ci-e2e`: 12 stages in about 11 s. The same experiment gives identical trajectories and datasets on Windows and Linux |
| Jenkins status | Jenkinsfile accepted by the declarative validator of Jenkins 2.580.2; every stage except Docker build replayed on Linux (non-root) from a fresh clone of the public repository: all 15 passed, and the marker-based stages select each unit test exactly once. Not yet run by a real Jenkins controller and agent |
| Docker status | CLI image (`python:3.12-slim`, non-root uid 10001, about 561 MB) and sandbox image (about 190 MB) build; the `ci-e2e` experiment ran inside the CLI image with outputs identical to the Windows run; Docker-sandbox tests pass on Docker 29.8.1 |

## Components

| Field | Status |
|---|---|
| Architecture | Clean architecture in 20 subpackages (153 modules): a framework-free, frozen pydantic domain (`rlfactory.domain`); application services (`tasks`, `rollouts`, `datasets`, `evaluation`, `experiments`, `flywheel`) working through protocols; adapters for subprocess/Docker sandboxes, OpenAI-compatible model servers, Parquet/DuckDB, TRL/verl/OpenRLHF and SSH. No TRL, Transformers or DuckDB type reaches the domain layer |
| Environments implemented | `arithmetic`, `structured_output`, `python` (generated functions run against hidden unit tests in a sandbox), `transformation`, `tool_use` (multi-step; `calculator`, `kv_lookup`, `json_transform`). 25 deterministic task template families feed them |
| Verifiers implemented | Kinds: exact match, numeric (exact rational, optional tolerance), JSON schema (+ expected value or constraints), unit test, regex, tool outcome, composite, optional LLM judge (never the primary signal; calibration utilities). Registered configs: `answer_tag_format`, `arithmetic_composite`, `exact_match`, `exact_strict`, `json_schema`, `numeric`, `numeric_lenient`, `python_block_format`, `python_composite`, `tool_outcome`, `unit_test`, `unit_test_binary` |
| Reward functions | `BinaryReward`, `WeightedVerifierReward`, `FormatReward`, `LengthPenalty` (DAPO-style soft over-length), `ToolCostPenalty`, `CompositeReward` (refuses over-shaped configs unless an ablation allows it). Registered: `binary`, `correctness_format`, `format_bonus`, `length_penalty`, `overshaped`, `overshaped_format`, `rlvr_default`, `tool_cost`, `weak_correctness`, `weighted_verifier`. Reward-hacking heuristics and reward calibration (correlation, AUC) |
| Policy adapters | Mock (deterministic, common random numbers, injected faults), scripted, OpenAI-compatible HTTP, vLLM, SGLang, Transformers (lazy import, `trl` extra) |
| Rollout system | `RolloutOrchestrator`: fixed worker pool over a bounded queue, per-generation and per-episode timeouts, exponential-backoff retries for infrastructure failures only, cancellation with flush, resume by deterministic trajectory ID, failure taxonomy, per-run Prometheus text metrics. About 95-100 trajectories/s with the mock policy on a laptop CPU |
| Trajectory store | `LocalParquetTrajectoryStore` (zstd Parquet shards, atomic writes, flat summary columns plus full payload, DuckDB queries that read the latest record per trajectory ID, retention policies); JSONL and in-memory stores. An object-store backend fits the same protocol but is not implemented |
| Dataset builders | SFT (`all_successful`, `highest_reward`, `shortest_correct`, `diverse_correct`), preference (`correct_vs_incorrect`, `reward_margin`, `short_correct_vs_verbose_incorrect`), RL prompts for GRPO/RLVR, reward model; contamination filter, validation and dataset cards; adapters for TRL 1.18, verl 0.9.1 and OpenRLHF 0.11.2 formats |
| Training backends | `mock`: complete (simulated). `trl`: complete, CPU smoke-tested. `verl`, `openrlhf`: prepared (run directory, reward shim, launch command), never launched |
| TRL support | Checked against TRL 1.18.0, Transformers 5.23.0 and PyTorch 2.15.0 (CPU): every key emitted for SFT, DPO, GRPO and reward-model configs exists and the configs construct without deprecation warnings; real 2-step CPU smoke runs of SFT, DPO and GRPO on `HuggingFaceTB/SmolLM2-135M-Instruct` (offline), GRPO logging `rewards/rlfactory_correctness/mean` and `rewards/rlfactory_format/mean`. LoRA wiring is exercised; QLoRA needs CUDA and is untested |
| Evaluation suites | Task suites `eval-heldout`, `ci-tiny-eval` and the verifier study `study-arithmetic-{strict,robust,composite}`; evaluation jobs `mock-base`, `vllm-candidate`, `verifier-study-{strict,robust,composite}`. Metrics: verified accuracy with Wilson 95% intervals, pass@1, unbiased pass@k, format compliance, tool success, sandbox and infrastructure failure rates, reward statistics, reward-correctness correlation, length, latency, failure categories |
| Regression gates | Rules `max_regression`, `no_regression`, `min_value`, `max_value` per scope (overall or domain); promotion states `candidate` -> `validated` or `rejected`; `rlfactory evaluate` exits with code 2 when a gate fails |
| Failure flywheel | Implemented: classify failures -> mutate failed tasks -> validate, deduplicate, decontaminate -> rejection-sampled SFT data -> train -> evaluate -> gate. Run only with the mock policy and mock trainer |
| Remote GPU support | SSH `check`, preflight, code and run sync, environment setup, 2-step smoke run, detached launch, `status`, `fetch`; confirmation prompt and `--dry-run`; GPU host scripts in `scripts/gpu/`. Never provisions or pays for machines. Not yet exercised against a real GPU host |

## Quality gates

| Field | Result |
|---|---|
| Test count | 329 tests in the default environment (unit 316, property 9, integration 2, end-to-end 2) plus 5 TRL tests that need the `trl` extra |
| Passed / failed / skipped | Windows 11: 328 passed, 0 failed, 2 skipped (the TRL module without its extra; one POSIX-only test). Linux, non-root, fresh clone: 327 passed, 0 failed, 3 skipped (the TRL module; two tests that need a Docker daemon). TRL environment: 5 passed, 0 failed, 0 skipped |
| Coverage | Branch coverage of the full suite: 90% on Windows, 89% on Linux (Jenkins gate: 85%) |
| Ruff | `ruff check .`: no findings; `ruff format --check .`: 220 files formatted |
| mypy | `--strict` with the pydantic plugin: no issues in 153 source files for both the win32 and linux platform targets |
| Package build | `uv build` produces the sdist and wheel; the artefact hygiene check passes (no results, weights or `.env`) |
| Secret scan | `scripts/secret_scan.py`: 281 files, 0 findings; `scripts/check_repo_hygiene.py`: 0 problems |
| Git authors | Every commit on `main` is authored by `M-amir1998`; no AI co-author or attribution metadata anywhere in the history |

## Exact first GPU commands

On a GPU host you already control (see [docs/training/gpu.md](docs/training/gpu.md)); pin the model
revision first (`revision:` in the training config):

```bash
cp configs/remote/gpu-host.example.yaml configs/remote/gpu-host.yaml   # edit host, workspace, code_dir
uv run rlfactory train configs/training/trl-sft-qwen3-0.6b.yaml --dry-run
uv run rlfactory remote check configs/remote/gpu-host.yaml
uv run rlfactory remote launch results/training/trl-sft-qwen3-0.6b --host configs/remote/gpu-host.yaml --dry-run
uv run rlfactory remote launch results/training/trl-sft-qwen3-0.6b --host configs/remote/gpu-host.yaml
uv run rlfactory remote status results/training/trl-sft-qwen3-0.6b --host configs/remote/gpu-host.yaml
uv run rlfactory remote fetch  results/training/trl-sft-qwen3-0.6b --host configs/remote/gpu-host.yaml
```

For a meaningful study, first replace the mock-generated training data with rollouts of the base
model (guide, section 2), then evaluate base and candidate with the same suite (section 6).

## Known limitations

- No real model has been trained or evaluated; every model-behaviour number in this repository is
  simulated, and the README comparison table is empty on purpose.
- The Jenkins pipeline has been validated and replayed, but not yet run by a real Jenkins agent;
  the remote GPU path has not yet run against a real host.
- verl and OpenRLHF backends prepare run directories only; they have never been launched.
- The subprocess sandbox is not a security boundary; the Docker sandbox is stronger but shares the
  host kernel.
- Preference pairs and GRPO prompts exclude multi-step tool-use episodes.
- Task families are small and synthetic; difficulty levels, reward-hacking flags and the GPU memory
  estimator are heuristics.
- Mock-policy token counts are whitespace approximations.
