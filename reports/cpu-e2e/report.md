# RLFactory report: cpu-e2e

> **SIMULATED RESULTS.** This run used the mock policy and/or the mock training backend. Every number below characterises the RLFactory infrastructure on synthetic tasks; none of it is a measurement of a real model, and no claim about SFT, DPO or GRPO effectiveness follows.

## Experiment

| Field | Value |
|---|---|
| Run ID | `cpu-e2e-example` |
| Experiment | cpu-e2e |
| Created (UTC) | 2026-11-14 11:59:14 |
| RLFactory | 0.1.0.dev0, git ec792009f599 |
| Seed | 7 |
| Config fingerprint | `e0301272b3c875cd` |
| Simulated | yes |

CPU-only end-to-end exercise of the RLFactory control plane with a deterministic mock policy and the mock training backend. Results characterise the infrastructure, not any model.

## Tasks

| Suite | Version | Tasks | Fingerprint |
|---|---|---|---|
| train-mixed | 1 | 75 | `5af39313496a8150` |
| eval-heldout | 1 | 39 | `a2dda31efe884b25` |

After validation and decontamination: **56** training, **19** development and **39** held-out evaluation tasks; 0 generated tasks failed validation.

| Domain | Train | Eval |
|---|---|---|
| arithmetic | 20 | 12 |
| python | 12 | 8 |
| structured_output | 12 | 8 |
| tool_use | 6 | 5 |
| transformation | 6 | 6 |

Verifier coverage: deterministic 95 (100%), heuristic 0 (0%), model_judge 0 (0%), unverifiable 0 (0%)

## Environments and verifiers

| Component | Name | Version |
|---|---|---|
| environment | arithmetic | 1.0.0 |
| environment | python | 1.0.0 |
| environment | structured_output | 1.0.0 |
| environment | tool_use | 1.0.0 |
| environment | transformation | 1.0.0 |
| verifier | exact_match | 1.0.0+d4a6649d |
| verifier | json_schema | 1.0.0+d4e50b34 |
| verifier | numeric | 1.0.0+81e0143d |
| verifier | tool_outcome | 1.0.0+77f5afd1 |
| verifier | unit_test | 1.0.0+f007667b |
| reward | binary | 1.0.0+73011565 |

Sandbox: `subprocess` (see docs/security/sandbox.md for its guarantees).

## Policy

Base policy `mock-base` (mock, model `rlfactory/mock-policy`).
Generation: temperature 0.8, top_p 1.0, max_tokens 512.

## Rollouts

| Metric | Value |
|---|---|
| Trajectories (completed / infra-failed) | 224 / 0 |
| Skipped (already completed, resume) | 0 |
| Verified pass rate | 51.3% |
| Mean reward | 0.513 |
| Retries (infrastructure only) | 15 |
| Infrastructure failure rate | 0.0% |
| Verification rate | 100.0% |
| Throughput | 105.9 trajectories/s |
| Completion tokens | 6448 (estimated) |
| Wall time | 2.1 s |

Failures by kind: `invalid_output` 24

| Domain | Completed | Failed | Pass rate | Mean reward |
|---|---|---|---|---|
| arithmetic | 80 | 0 | 52.5% | 0.525 |
| python | 48 | 0 | 58.3% | 0.583 |
| structured_output | 48 | 0 | 54.2% | 0.542 |
| tool_use | 24 | 0 | 33.3% | 0.333 |
| transformation | 24 | 0 | 45.8% | 0.458 |

## Reward

| Reward calibration | Value |
|---|---|
| Mean reward (passed / failed) | 1.000 / 0.000 |
| Point-biserial r(reward, verified) | 1.000 |
| AUC P(reward_pass > reward_fail) | 1.000 |
| Reward std | 0.500 |

Mean reward components: correctness +0.513, format +0.000, length +0.000, tool_cost +0.000, safety +0.000

Reward-hacking heuristics flagged: degenerate_repetition 9 (of 224 trajectories). Heuristic detectors: flags are leads for manual review, not proof of reward hacking, and an empty report does not prove its absence.

## Datasets

| Kind | Schema | Records | Excluded | Validation |
|---|---|---|---|---|
| dpo | `trl.dpo.conversational.preference_explicit_prompt` | 83 | multi_turn=24, no_valid_pair_in_group=6 | ok |
| grpo | `trl.grpo.conversational.prompt_only` | 44 | multi_step_task=6, saturated_all_fail=3, saturated_all_pass=3 | ok |
| sft | `trl.sft.conversational.prompt_completion` | 67 | not_passed=109, not_selected=63 | ok |

## Training

| Field | Value |
|---|---|
| Backend / method | mock / sft (simulated: no weights trained) |
| Base model | rlfactory/mock-base (unpinned) |
| Dataset | cpu-e2e-example-sft (67 records) |
| State / steps | succeeded / 20 |
| Final loss | 0.4550 |
| Wall time | 0.0 s |
| Tokens seen | n/a |
| Peak VRAM | n/a |
| Framework | n/a |
| Resumed from step | - |

## Evaluation

| Model | Method | Verified Acc. | Python Pass | Format | Reward | Train Time | VRAM |
|---|---|---|---|---|---|---|---|
| `mock-base` (simulated) | base | 56.4% | 62.5% | 85.9% | 0.564 | - | - |
| `mock-base-sft` (simulated) | sft | 64.1% | 68.8% | 91.0% | 0.641 | simulated | - |

Suite `eval-heldout`: 39 tasks x 2 samples, temperature 0.7. Infrastructure failures are excluded from accuracy and reported separately.

| Domain | base acc. (95% CI) | candidate acc. (95% CI) |
|---|---|---|
| arithmetic | 62.5% (42.7-78.8%) | 66.7% (46.7-82.0%) |
| python | 62.5% (38.6-81.5%) | 68.8% (44.4-85.8%) |
| structured_output | 56.2% (33.2-76.9%) | 81.2% (57.0-93.4%) |
| tool_use | 50.0% (23.7-76.3%) | 50.0% (23.7-76.3%) |
| transformation | 41.7% (19.3-68.0%) | 41.7% (19.3-68.0%) |
| **overall** | 56.4% (45.4-66.9%) | 64.1% (53.0-73.9%) |

Unbiased pass@k (stochastic sampling): base: pass@2 82.1%; candidate: pass@2 89.7%

| Policy | pass@1 | Tool success | Sandbox failures | Infra failures | Mean tokens | p50 latency (s) | r(reward, verified) |
|---|---|---|---|---|---|---|---|
| base | 56.4% | 65.4% | 0.0% | 0.0% | 26.8 | 0.000 | 1.000 |
| candidate | 64.1% | 65.4% | 0.0% | 0.0% | 26.7 | 0.000 | 1.000 |

## Regression gate

| Rule | Scope | Baseline | Candidate | Result |
|---|---|---|---|---|
| overall accuracy may not drop more than 2 points | overall | 0.5641 | 0.6410 | PASS |
| python pass rate may not regress | domain:python | 0.6250 | 0.6875 | PASS |
| format compliance stays above 80% | overall | 0.8590 | 0.9103 | PASS |
| infrastructure failure rate below 5% | overall | 0.0000 | 0.0000 | PASS |

**Gate: PASSED** (candidate vs base).
- only 78 completed candidate trajectories (95% CI 0.53-0.74); small differences are noise
- simulated evaluation (mock policy/backend): not evidence about real models

Promotion decision for `cpu-e2e-example-sft-step20`: **validated**.

## Failure analysis

| Source | Observed category | Count |
|---|---|---|
| evaluation (candidate) | format_error | 6 |
| evaluation (candidate) | reasoning_error | 18 |
| evaluation (candidate) | tool_misuse | 4 |
| training rollouts | format_error | 21 |
| training rollouts | reasoning_error | 82 |
| training rollouts | tool_misuse | 6 |

Categories describe what was observed (format, runtime error, timeout, wrong answer, tool misuse, verifier or infrastructure failure). Per-trajectory speculative root causes are kept in a separate, explicitly labelled `hypothesis` column of `failures.parquet`.

## Failure-driven data flywheel

| Iteration | Failure Tasks Added | Train Examples | Eval Accuracy | Gate |
|---|---|---|---|---|
| 0 | 0 | 0 | 56.4% (simulated) | - |
| 1 | 32 | 46 | 62.8% (simulated) | pass |
| 2 | 28 | 44 | 71.8% (simulated) | pass |

## Figures

![Verified accuracy by domain](plots/accuracy_by_domain.svg)
![Reward distribution](plots/reward_distribution.svg)
![Failure categories](plots/failure_categories.svg)
![Flywheel accuracy](plots/flywheel_accuracy.svg)

## Resource usage

| Stage | Status | Wall (s) | Peak RSS (GB) | Mean CPU % |
|---|---|---|---|---|
| analysis | completed | 0.1 | 0.10 | 87 |
| build_dpo | completed | 0.1 | 0.10 | 50 |
| build_grpo | completed | 0.1 | 0.10 | 74 |
| build_sft | completed | 0.2 | 0.10 | 36 |
| evaluate_base | completed | 1.8 | 0.10 | 39 |
| evaluate_candidate | completed | 1.9 | 0.10 | 42 |
| flywheel | completed | 14.6 | 0.11 | 53 |
| gate | completed | 0.0 | 0.10 | 52 |
| rollouts | completed | 2.2 | 0.08 | 76 |
| tasks | completed | 2.3 | 0.07 | 78 |
| train | completed | 0.0 | 0.10 | 25 |

Hardware: Windows 11 (AMD64), Python 3.12.13, 12 CPUs, 15.6 GB RAM, GPU: none.

## Limitations

- Model behaviour is simulated (mock policy / mock training); results describe the pipeline, not a model.
- Token counts are whitespace approximations reported by the mock policy.
- Generated Python ran in the subprocess sandbox, which is not a security boundary.
- Evaluation samples are small; read differences against the Wilson 95% intervals.
- Task difficulty levels are heuristic signals, not measures of cognitive difficulty.
- Reward-hacking detectors are heuristics; an empty report does not prove absence of hacking.

_Generated 2026-11-14 11:59 UTC by RLFactory from stored run artefacts._
