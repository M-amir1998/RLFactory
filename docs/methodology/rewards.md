# Rewards

## Verification is not reward

A `VerificationResult` says whether the answer is correct. A `RewardFunction` decides which number
the learner sees, given the task, the trajectory and that verification:

```python
class RewardFunction(Protocol):
    def compute(self, task, trajectory, verification) -> RewardResult: ...
    def bounds(self) -> Mapping[RewardComponent, tuple[float, float]]: ...
```

Keeping them separate makes it possible to (1) re-reward stored trajectories without re-verifying,
(2) audit a reward against correctness, and (3) run reward-shaping ablations with the verdict held
fixed. See [ADR-004](../adr/ADR-004-separate-verification-and-reward.md).

## Decomposition

Every `RewardResult` stores its components, not only the scalar:

| Component | Sign | Produced by |
|---|---|---|
| `correctness` | >= 0 by convention | `BinaryReward`, `WeightedVerifierReward` |
| `format` | bonus or penalty | `FormatReward` |
| `length` | <= 0 | `LengthPenalty` |
| `tool_cost` | <= 0 | `ToolCostPenalty` |
| `safety` | <= 0 | reserved; no built-in function sets it yet |

The trajectory store keeps one column per component, so reward debugging is a query.

## Functions and bounds

| Function | Range | Notes |
|---|---|---|
| `BinaryReward(success=1, failure=0)` | {failure, success} | the default RLVR reward |
| `WeightedVerifierReward(weight)` | [0, weight] | keeps partial credit, e.g. fraction of tests passed |
| `FormatReward(weight=0.1, penalty=0)` | [-penalty, weight] | "format ok" = the environment parsed a final answer or code block |
| `LengthPenalty(max_tokens=1024, buffer_tokens=256, coefficient=0.5)` | [-coefficient, 0] | soft over-length penalty in the style of DAPO: zero below `max - buffer`, linear inside the buffer, `-coefficient` beyond `max`; never rewards brevity |
| `ToolCostPenalty(cost_per_call, error_cost, slack_calls, max_penalty)` | [-max_penalty, 0] | penalises calls beyond the task's known-good plan (+ slack) and failed calls |
| `CompositeReward(children, clip_min, clip_max)` | sum of children, optionally clipped | records each child's total in `details` |

Named defaults (`rewards/registry.py`): `binary`, `weighted_verifier`, `format_bonus`,
`length_penalty`, `tool_cost`, `correctness_format` (binary + 0.1 format), `rlvr_default`
(binary + format + length + tool cost, clipped to [-1, 1.1]) and the deliberately mis-specified
`overshaped` (0.4 x verifier score + 0.6 format bonus) for the shaping ablation only.

## Avoiding arbitrary shaping

`CompositeReward` refuses configurations whose maximum non-correctness bonus exceeds
`max_bonus_ratio` (0.5) times the maximum correctness reward, or that have no positive
correctness signal at all, unless `allow_overshaping: true` is set explicitly. Penalties are
bounded and documented. The guard exists because a format bonus that rivals correctness is the
easiest reward to hack: emit a well-formed wrong answer.

## Reward hacking analysis

`RewardHackingAnalyzer` flags trajectories for manual review. It is heuristic: flags are leads,
not proof, and an empty report does not prove the absence of hacking.

| Detector | Signal |
|---|---|
| `high_reward_unverified` | reward >= threshold but verification failed |
| `format_exploitation` | positive format bonus, no correctness, empty or trivial answer |
| `pathological_short` | rewarded near-empty output |
| `multiple_answers` | several `<answer>` blocks (hedging against "last answer wins") |
| `verifier_loophole_suspect` | passing Python code containing harness-tampering patterns |
| `degenerate_repetition` | high repeated 3-gram fraction within a response |
| `repeated_output` | identical rewarded completion across several different tasks |

## Reward calibration

A reward is not correctness. `calibrate_rewards` reports, over completed trajectories: pass rate,
mean reward overall and for passed vs failed trajectories, the point-biserial correlation between
reward and the verified indicator, the AUC that a passing trajectory outscores a failing one (ties
count one half), per-component means and a reward histogram. With a binary reward the correlation
is 1 by construction; shaped rewards should stay close to it, and a drop is a warning sign.
