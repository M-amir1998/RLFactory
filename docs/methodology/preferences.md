# Preference data

## Where preferences come from

RLFactory builds preference pairs from its own verified rollouts. In v0.1 a preference means
"this completion was verified correct (or earned more reward) and that one was not"; it is not a
human or stylistic preference.

Pairs are formed **only within one rollout group**: all samples of one task under one policy,
checkpoint, generation configuration, seed and verifier. The builder also checks that every member
of the group rendered the same prompt messages, and drops the group otherwise
(`prompt_mismatch`). Comparing completions of unrelated tasks would teach the model nothing about
the task and is never done.

## Strategies

| Strategy | Chosen | Rejected |
|---|---|---|
| `correct_vs_incorrect` (default) | a verified-correct completion | a verified-incorrect completion of the same group |
| `reward_margin` | higher reward | lower reward, with `reward(chosen) - reward(rejected) >= min_margin` |
| `short_correct_vs_verbose_incorrect` | shortest correct completions first | longest incorrect completions first, strictly longer than the chosen one |

Selection is deterministic (seeded shuffles), capped by `max_pairs_per_task`, and skips duplicate
pairs and pairs whose texts are identical.

## Validation

`validate_preference` rejects datasets with prompts that do not end in a user turn, empty chosen
or rejected completions, `chosen == rejected`, non-finite reward metadata, or evaluation tasks.
It warns when the chosen completion has a lower recorded reward than the rejected one.

## Format

DPO data uses TRL's conversational preference format with an **explicit prompt** (recommended by
TRL for DPO):

```json
{"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "chosen":   [{"role": "assistant", "content": "..."}],
 "rejected": [{"role": "assistant", "content": "..."}]}
```

Reward-model data uses the **implicit-prompt** format (prompt messages prepended to both chosen and
rejected), which TRL's dataset-format table recommends for `RewardTrainer`.
`RewardModelDatasetBuilder` keeps only pairs with a verified winner (chosen passed, rejected
failed). `RewardModelEvaluator` measures any scorer (a trained reward model, a judge, or a
deterministic verifier used as a sanity baseline) on held-out pairs: pairwise accuracy (ties count
one half) and mean margin.

## Limitations

- Multi-turn (tool-use) groups are excluded (`single_turn_only`): pairing full multi-turn episodes
  would train on tool-result tokens. Turn-level pairing is future work.
- Correctness-based pairs encode no preference among several correct answers, except through the
  length-based strategy.
- Pairs built from a mock policy are only useful for testing the pipeline.
