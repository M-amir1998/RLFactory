# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) once it reaches 0.1.0.

## [Unreleased] - 0.1.0.dev0

### Added

- Framework-free domain model: tasks with typed payloads, trajectories with lineage, bounded
  verification results, decomposed rewards, training runs, run manifests, evaluation results and
  gate reports.
- Task layer: 25 deterministic template families across arithmetic, structured output, Python,
  text transformation and tool use; validation (reference must pass, known-wrong must fail);
  deterministic mutation; heuristic difficulty signals; deduplication; contamination control;
  grouped deterministic splits; curriculum samplers; YAML-discovered suite registry.
- Environments (`reset`/`step`/`close`), including a multi-step tool-use environment with
  calculator, key-value lookup and JSON transform tools.
- Sandboxes: subprocess (limits, watchdog, minimal environment), Docker (no network, non-root,
  read-only, dropped capabilities) and mock; two-process unit-test harness.
- Verifiers: exact match, numeric, JSON schema, unit test, regex, tool outcome, composite and an
  optional LLM judge with calibration utilities; config-driven registry with versioned configs.
- Rewards: binary, weighted verifier, format, DAPO-style length penalty, tool cost and composite
  with an over-shaping guard; reward-hacking heuristics and reward calibration.
- Policies: deterministic mock (common random numbers, simulated latency and faults), scripted,
  OpenAI-compatible HTTP, vLLM, SGLang and lazy Transformers policies.
- Rollout orchestrator with bounded concurrency, retries for infrastructure failures, timeouts,
  cancellation, deterministic trajectory IDs and resume; failure taxonomy.
- Parquet + DuckDB trajectory store (latest-record semantics, retention policies), JSONL and
  in-memory stores.
- SFT, preference, GRPO/RLVR and reward-model dataset builders, dataset validation, dataset cards
  and adapters for TRL 1.18, verl 0.9.1 and OpenRLHF 0.11.2 formats.
- Training backends: mock (simulated), TRL (SFT/DPO/GRPO/reward model, LoRA/QLoRA, verifier-backed
  GRPO reward functions), and prepared verl and OpenRLHF interfaces; GPU memory estimator.
- Evaluation with Wilson intervals, unbiased pass@k, failure analysis, verifier coverage,
  regression gates and promotion states; lm-evaluation-harness command adapter.
- Failure-driven data flywheel.
- Experiment DAG with partial execution, fingerprint-based stage caching and resume; Markdown and
  SVG reports.
- Remote GPU execution over SSH (check, preflight, sync, smoke, launch, status, fetch). The launch
  backgrounds only the job, with all standard streams detached from the SSH channel, so the
  session returns immediately; `--dry-run` never writes or overwrites `remote_handle.json`.
- Typer CLI, `doctor`, structured logging with secret redaction, metrics registry, Jenkins
  pipeline, Docker images, documentation and ADRs.
- Synthetic trajectory fixtures (`tests/fixtures/`) with a schema-compatibility test.

### Verified

- TRL 1.18.0 / Transformers 5.23.0 CPU smoke runs of SFT, DPO and GRPO (wiring checks only).
  Config construction raises no deprecation warnings: `gradient_checkpointing` is renamed to
  `activation_checkpointing` on the training host when the installed Transformers has it.
- Full test suite on Windows 11 and on Linux (non-root container); every Jenkinsfile stage
  except Docker build replayed on a Linux container; both Docker images built and the CI
  experiment run inside the CLI image with results identical to the Windows run.
- Jenkinsfile accepted by the declarative-pipeline validator of a local Jenkins 2.580.2.

### Not yet done

- Real GPU post-training experiments.
