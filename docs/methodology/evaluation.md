# Evaluation methodology

## Same suite, same verifier, same infrastructure

Base, SFT, DPO and GRPO checkpoints are evaluated with one `Evaluator`: the same held-out tasks,
system prompt, environments, sandbox, verifiers, reward, metrics and failure taxonomy. Differences
between rows of a comparison table can then be attributed to the checkpoint rather than to the
measurement.

## What is measured

- **Verified accuracy** = passed / completed trajectories. Infrastructure failures are removed from
  the denominator and reported as `infra_failure_rate`, so a flaky server cannot masquerade as a
  model regression; a gate should bound that rate instead.
- **pass@1** = mean over tasks of the per-task pass fraction (equal weight per task).
- **pass@k** uses the unbiased estimator `1 - C(n - c, k) / C(n, k)` (Chen et al., 2021) and is
  reported only when sampling is stochastic (temperature > 0) and every task has at least `k`
  samples. With greedy decoding repeated samples are identical and pass@k would be meaningless.
- **Format compliance**, **tool success**, **sandbox failure rate**, completion length, latency
  percentiles, reward statistics and reward-vs-verified correlation complete the picture.

## Uncertainty

Accuracy is reported with a Wilson 95% interval. Evaluation suites in this repository are small,
so intervals are wide; gates add a warning below 100 completed trajectories. Gates are engineering
guardrails, not significance tests: a gate that allows a 2-point drop can pass a real regression
inside the noise, and a strict `no_regression` gate can fail on noise. Larger suites, more samples
per task, and paired comparisons on the same tasks reduce the problem; reading the intervals is
still required.

The mock policy uses common random numbers (outcome draws depend on the task and sample, not on
the policy), which removes most sampling noise from *simulated* comparisons. Real models do not
offer that, which is one more reason simulated numbers say nothing about real effect sizes.

## Contamination

Training data is filtered against the evaluation suite by task ID, mutation lineage, normalised
prompt and, optionally, template family and benchmark ID. Held-out *instances* of seen templates
measure in-distribution generalisation; held-out *families* (`group_by: family`,
`exclude_eval_families`) measure transfer to unseen templates. State which one a result uses.

## Reward is not correctness

Reward can be shaped, clipped or exploited; verified correctness cannot (up to verifier quality).
Reports always show both, plus their correlation and AUC, and the reward-hacking heuristics.

## Judges

LLM judges are optional, never primary, and must be calibrated against deterministic or human
labels (agreement, Cohen's kappa) and probed for verbosity and position bias before use. Their
model, prompt-template hash and temperature are stored with every verdict.

## External benchmarks

`evaluation/external.py` can build lm-evaluation-harness commands and parse their results for
benchmarks outside RLFactory's verifiable suites. These are kept out of CI and out of the core
comparison table unless run under the same conditions as the internal suites.
