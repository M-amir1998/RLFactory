# Evaluation reference

```bash
uv run rlfactory evaluate configs/evaluation/mock-base.yaml --run-id mock-base-eval
```

`rlfactory evaluate` runs one policy on one or more suites and writes
`results/<run-id>/evaluation/<label>.json` (plus trajectories and `failures.parquet` under
`evaluation/<label>/`). With `baseline:` set, it also writes `gate_report.json`, prints every
check and exits with code 2 if the gate fails. Inside experiments the `evaluate_base`,
`evaluate_candidate` and `gate` stages do the same.

## Metrics

Computed per suite (`overall`) and per domain (`domain:<name>`) by `summarize_trajectories`:

| Metric | Definition |
|---|---|
| `verified_accuracy` | passed / completed; infrastructure-failed trajectories are excluded |
| `accuracy_ci95` | Wilson score interval (95%) for `verified_accuracy` |
| `pass_at_1` | mean over tasks of the per-task fraction of passing samples |
| `pass_at_k` | unbiased estimator `1 - C(n - c, k) / C(n, k)` averaged over tasks; only when temperature > 0 and every task has at least `k` completed samples; otherwise omitted |
| `format_compliance` | completed trajectories whose final observation has a parsed answer and no parse error |
| `tool_success_rate` | successful tool calls / all tool calls (tool-use tasks) |
| `sandbox_failure_rate` | Python trajectories that failed with `sandbox_error` / all Python trajectories |
| `infra_failure_rate` | infrastructure-failed trajectories / all trajectories |
| `mean_reward`, `reward_std` | over completed trajectories |
| `reward_correctness_correlation` | point-biserial correlation between reward and the verified indicator |
| `mean_completion_tokens`, `mean_completion_chars`, `total_completion_tokens` | completion length (token counts are estimated for the mock policy) |
| `latency_p50_s`, `latency_p95_s` | policy latency per trajectory |
| `failure_categories` | counts of observed failure categories |
| `num_tasks`, `num_trajectories`, `num_completed`, `num_infra_failures` | sizes |

`EvaluationResult` adds the label, policy ID, checkpoint, suite, samples per task, temperature,
`simulated` flag, creation time and code version.

## Failure categories

Observed, not inferred (see `evaluation/failures.py`):

| Category | Observed when |
|---|---|
| `reasoning_error` | well-formed answer that is wrong (`wrong_answer`, `value_mismatch`, `constraint_violation`, `tests_failed`) |
| `format_error` | no answer, invalid format or JSON, no code block, schema violation |
| `code_runtime_error` | syntax error, runtime error, resource limit, missing execution report |
| `timeout` | generated program exceeded its time limit |
| `tool_misuse` | step limit, required tool not used, failed tool calls |
| `verifier_failure` | verifier error or undecidable judge output |
| `infrastructure_failure` | the episode failed for infrastructure reasons |

`failures.parquet` stores one row per failed trajectory with an `observed` description and a
separate, explicitly `SPECULATIVE:` `hypothesis` column (e.g. a near-miss numeric answer, a correct
value present without the required tag, a hallucinated tool name). Hypotheses are never used for
gating.

## Regression gates

```yaml
gates:
  min_samples_warning: 100
  rules:
    - {name: overall accuracy may not drop more than 2 points, metric: verified_accuracy, kind: max_regression, threshold: 0.02}
    - {name: python pass rate may not regress, metric: verified_accuracy, scope: "domain:python", kind: no_regression}
    - {name: format compliance stays above 90%, metric: format_compliance, kind: min_value, threshold: 0.9}
    - {name: infrastructure failure rate below 5%, metric: infra_failure_rate, kind: max_value, threshold: 0.05}
```

| Kind | Passes when |
|---|---|
| `max_regression` | `candidate >= baseline - threshold` |
| `no_regression` | `candidate >= baseline - threshold` with `threshold` as a small tolerance (default 0) |
| `min_value` | `candidate >= threshold` |
| `max_value` | `candidate <= threshold` |

`metric` is any scalar field of the metrics table (or `pass_at_<k>`). A missing candidate metric,
or a missing baseline metric for a regression rule, fails the check. Rules with `required: false`
are reported but do not affect the overall result. Warnings are added for small samples, different
suites and simulated inputs.

## Promotion

`PromotionRegistry` (`results/model_registry.json` in experiments) stores small JSON records:
a checkpoint is registered as `candidate`, then moved once to `validated` (gate passed) or
`rejected` (with the failed checks as reasons). The append-only history keeps every transition.
Weights are never stored here.

## Verifier coverage

`verifier_coverage(tasks, verifiers)` counts tasks verified `deterministic`, `heuristic`,
`model_judge` or `unverifiable` (overall and per domain). The tasks stage of an experiment records
it in `tasks/summary.json` and the report.

## Judge calibration

For optional LLM judges (`rlfactory.verifiers.calibration`):

| Function | Measures |
|---|---|
| `binary_agreement(judge, reference)` | confusion matrix, accuracy, precision, recall, F1, Cohen's kappa, pass rates |
| `verbosity_flip_rate(judge, responses, padding)` | verdict changes caused by content-free padding |
| `position_flip_rate(pairwise_judge, pairs)` | preference changes when candidates are swapped |

Reference labels can come from deterministic verifiers on tasks where both apply, or from humans.
The statistics describe the judge on that label set only.

## External benchmarks

`rlfactory.evaluation.external` builds `lm_eval` commands (`local-chat-completions`,
`local-completions` or `hf` backends) and parses result files. It is not wired into the CLI or CI;
verify flags against the installed harness before use.
