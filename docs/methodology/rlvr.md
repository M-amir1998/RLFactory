# RLVR: reinforcement learning with verifiable rewards

RLVR replaces a learned or judged reward with a programmatic check of the answer: the number is
right, the unit tests pass, the JSON satisfies the schema and the constraints, the tool-derived
answer is correct. RLFactory's first focus is RLVR because the reward is objective, cheap and
auditable, and because it lets the infrastructure be tested end to end without a judge model.

## What makes a task usable for RLVR

A verifier that can be satisfied without solving the task produces a policy that learns the
shortcut. RLFactory therefore accepts a generated task only when:

1. its environment and verifier resolve;
2. the reference solution, played through the real environment (sandbox included), **passes**;
3. at least one known-wrong answer is **rejected** (the verifier discriminates).

Tasks are built from typed template parameters, so ground truth is recomputed for every task and
every mutation; nothing is hand-edited. Answers have explicit formats (`<answer>` tags, fenced
Python, JSON, tool calls) stated in the system prompt.

## Verifier design matters

The same completions can earn different rewards under different verifiers. With the canonical
reference answer `13.5`:

| Verifier | `<answer>27/2</answer>` | `The answer is 27/2.` (no tag) |
|---|---|---|
| `exact_strict` | fail | fail |
| `numeric` | pass | fail (`no_answer`) |
| `numeric_lenient` | pass | pass (last-number fallback) |
| `arithmetic_composite` (numeric 0.9 required + tag format 0.1) | pass, score 1.0 | fail, score 0.0 |

Strict verifiers punish correct answers in unexpected formats (sparser reward); lenient ones accept
answers the model never committed to (exploitable reward). The verifier-quality study configs
(`configs/evaluation/verifier-study-{strict,robust,composite}.yaml`) score identical arithmetic
tasks and policy samples under each verifier; with the mock policy they measure verifier
sensitivity, not learning.

## GRPO data in RLFactory

`RLPromptDatasetBuilder` writes TRL's conversational prompt-only format plus two extra columns that
TRL passes to reward functions: `rlfactory_task` (the serialised task) and `rlfactory_reward` (the
reward name). It drops:

- tasks the current policy always solves or never solves (when rollout statistics show at least
  `min_attempts_for_filter` attempts): their GRPO group advantages are zero, so they only cost
  compute;
- multi-step tool-use tasks (a single completion cannot be scored without the environment loop);
- contaminated tasks and, optionally, prompts above `max_prompt_chars` (TRL skips over-long prompts
  rather than truncating them).

During training, each configured reward component becomes a separate TRL reward function
(`rlfactory_correctness`, `rlfactory_format`, ...) backed by `CompletionScorer`, which replays the
completion through the same environment, verifier and reward used for rollouts and evaluation.
Training reward and evaluation verdict therefore cannot drift apart silently.

## TRL GRPO defaults to be aware of (TRL 1.18)

`beta=0.0` (no reference model, no KL term), `loss_type="dapo"`, `scale_rewards="group"`,
`num_generations=8` (must divide the generation batch), `max_completion_length=512`,
`mask_truncated_completions=False`, `vllm_mode="colocate"`. RLFactory's `GRPOParams` mirrors these
defaults explicitly so changes in TRL do not silently change experiments. See
[../references/post-training-stack.md](../references/post-training-stack.md).

## Planned studies (not yet run)

| Study | Question | Configs |
|---|---|---|
| Method comparison | base vs SFT vs DPO vs GRPO under identical tasks, verifier, lineage and evaluation | `configs/training/trl-{sft,dpo,grpo}-qwen3-0.6b.yaml` |
| Reward shaping | binary vs correctness+format vs deliberately over-shaped reward: accuracy, length, format compliance, hacking flags | `configs/experiments/gpu-qwen3-grpo-{binary,correctness_format,overshaped}.yaml` |
| Verifier quality | strict vs robust vs composite verifiers as RL signals | `configs/tasks/study-arithmetic-*.yaml` |
| Flywheel | does failure-driven data improve a small model across iterations? | see [failure-flywheel.md](failure-flywheel.md) |

All require a GPU host. No result is claimed until measured.
