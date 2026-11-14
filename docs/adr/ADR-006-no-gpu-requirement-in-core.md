# ADR-006: No GPU requirement in the core system

- Status: Accepted
- Date: 2026-11-14

## Context

The development machine is a CPU-only laptop with 16 GB RAM. GPU time is rented occasionally. If the
control plane (tasks, rollouts, storage, datasets, orchestration, evaluation, reporting) could only
be exercised on a GPU, most bugs would be found on expensive hardware, and CI would require GPUs.

## Decision

The core package and its tests run without a GPU, CUDA, model downloads or network access:

- heavy ML dependencies live in the optional `trl` / `gpu` extras and are imported lazily;
- a deterministic `MockPolicy` and a `MockTrainingBackend` let the complete pipeline run on CPU;
- simulated outputs are labelled `simulated` everywhere (manifests, evaluation results, reports);
- real training is prepared locally and executed on a remote GPU host (ADR-008);
- optional local checks (TRL CPU smoke runs with a tiny cached model) are marked and skipped when
  the stack or model is absent.

## Consequences

- The whole pipeline, including failure injection, retries, resume, gates and the flywheel, is
  tested on every CI build in about a minute.
- Simulation fidelity must never be mistaken for results; this is handled by labelling and by the
  report banner.
- Performance characteristics of real models (latency, memory) are only known after GPU runs.

## Alternatives considered

- **Tiny real model in CI**: slow, needs downloads or a cached model in CI, and still not
  representative; kept as an optional local smoke test.
- **GPU CI runners**: costly and unnecessary for control-plane correctness.
