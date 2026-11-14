# ADR-004: Separate verification from reward construction

- Status: Accepted
- Date: 2026-11-14

## Context

Many RL codebases compute a single "reward" that silently mixes correctness, formatting, length
and other shaping terms. When the reward is the only stored quantity, it is impossible to tell
whether a policy improved at the task or learned to exploit the shaping, and changing the reward
requires recomputing correctness.

## Decision

Two protocols with distinct outputs:

- `Verifier.verify(task, trajectory) -> VerificationResult` (bounded score, binary pass, reason
  code, verifier version);
- `RewardFunction.compute(task, trajectory, verification) -> RewardResult` (scalar plus components
  `correctness`, `format`, `length`, `tool_cost`, `safety`).

Both results are stored on every trajectory and as columns in the trajectory store. Accuracy
metrics and gates use verification; training uses reward.

## Consequences

- Reward ablations hold verification fixed and are directly comparable.
- Reward can be audited against correctness (point-biserial correlation, AUC, hacking heuristics).
- Stored trajectories can be re-rewarded without re-running environments or sandboxes.
- Two concepts to configure instead of one; defaults (`binary` reward) keep simple cases simple.

## Alternatives considered

- **Reward-only pipeline**: simpler, but hides reward hacking and couples verifier changes to
  reward changes.
- **Verification only (reward = pass/fail)**: loses partial credit, format and cost signals that
  some experiments legitimately need.
