# Evaluation (architecture)

Metric definitions and gate configuration are documented in
[../evaluation/README.md](../evaluation/README.md); the statistical reasoning in
[../methodology/evaluation.md](../methodology/evaluation.md). This page covers structure.

## Components

| Component | Module | Role |
|---|---|---|
| `Evaluator` | `evaluation/evaluator.py` | runs the suite through the rollout machinery and computes metrics |
| `summarize_trajectories`, `summarize_by_domain` | `evaluation/metrics.py` | `MetricsSummary` from trajectories |
| `analyze_failures`, `classify` | `evaluation/failures.py` | observed failure categories plus labelled hypotheses |
| `evaluate_gates` | `evaluation/gates.py` | baseline vs candidate rules -> `GateReport` |
| `PromotionRegistry` | `evaluation/promotion.py` | candidate -> validated / rejected, JSON history |
| `verifier_coverage` | `evaluation/coverage.py` | deterministic / heuristic / judge / unverifiable shares |
| `build_lm_eval_command`, `parse_lm_eval_results` | `evaluation/external.py` | optional lm-evaluation-harness adapter |

## Method independence

The evaluator takes a `Policy`, tasks and an `EvaluationConfig` (samples per task, generation
config, pass@k values, seed, concurrency, reward, verifier overrides, retry policy). It does not
know which training method produced the policy. Base, SFT, DPO and GRPO candidates are therefore
measured with identical prompts (`PromptBuilder`), environments, verifiers and metrics. Evaluation
reuses `run_rollouts`, so retries, timeouts and failure classification are identical to training
rollouts, and evaluation trajectories can be persisted (`evaluation/<label>/trajectories/`).

## Candidate policies

- Mock training: the candidate is a `MockPolicy` built from the checkpoint's simulated profile.
- Real checkpoints: the experiment needs `evaluation.candidate_policy`, typically a vLLM endpoint
  serving the trained LoRA adapter (`vllm serve <base> --enable-lora --lora-modules
  candidate=<adapter>`), addressed as model `candidate`.

## Gates and promotion

`evaluate_gates(baseline, candidate, config)` evaluates every rule on the requested scope
(`overall` or `domain:<name>`), fails closed when a metric is missing, and adds warnings for small
samples (below `min_samples_warning`, default 100 completed trajectories), different suites, and
simulated inputs. In an experiment the `gate` stage registers the candidate checkpoint as
`candidate` in `results/model_registry.json` and records `validated` or `rejected`; a decided
checkpoint cannot be decided again.

## External benchmarks

`evaluation/external.py` builds an `lm_eval` command (for `local-chat-completions`,
`local-completions` or `hf` backends) and parses its results file. It is a library API, not a CLI
command and not part of CI; verify flags against the installed harness before use.
