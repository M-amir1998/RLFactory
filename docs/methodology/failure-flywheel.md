# Failure-driven data flywheel

The flywheel turns observed model failures into new, verified training data and checks, each
iteration, that the held-out evaluation did not regress.

```
iteration i:
  1. evaluate the current policy on the development split
  2. classify failures (observed categories; infrastructure failures are ignored)
  3. take up to max_failure_tasks failed tasks
  4. mutate each one mutations_per_failure times (numbers, constraints, variable names,
     input lengths, formats, edge cases)
  5. validate, deduplicate and decontaminate the mutants against the evaluation suite
  6. roll out the current policy on the accepted mutants (samples_per_task samples each)
  7. keep verified-correct samples as SFT data (rejection sampling)
  8. train a new checkpoint
  9. evaluate on the held-out suite and gate against the previous iteration
```

Implemented in `rlfactory.flywheel.loop.FailureDrivenFlywheel`; training and checkpoint-to-policy
resolution are injected, so the loop itself is backend-agnostic.

## Leakage guarantees

| Risk | Control |
|---|---|
| mutants of evaluation tasks entering training | failures come from the development split; additionally every mutant's `parent_task_id` is checked against the evaluation task IDs |
| identical prompts across splits | normalised prompt fingerprints of evaluation tasks are excluded |
| same template in train and eval | instance-level splits (default) allow it by design; use `group_by: family` and/or `exclude_eval_families: true` for held-out-template evaluation |

Mutations rebuild tasks from perturbed template parameters, so every mutant has recomputed ground
truth and must pass the same validation as freshly generated tasks (reference passes, a wrong
answer fails).

## Output

`flywheel.json` records, per iteration: failures observed, failure tasks, failure categories,
mutation candidates, tasks added, rejected (invalid / contaminated), train examples, checkpoint,
held-out accuracy with Wilson interval, gate result and a `simulated` flag. The report renders:

| Iteration | Failure Tasks Added | Train Examples | Eval Accuracy | Gate |
|---|---|---|---|---|

Iteration 0 is the base policy.

## Status and limitations

- The automated `flywheel` experiment stage currently requires the mock policy and mock training
  backend, so its numbers are simulated. The mock policy's improvement comes from the documented
  profile adjustment of the mock trainer, which reacts to how many examples each domain received,
  not to which tasks failed; the loop's *mechanics* are real, its outcomes are not evidence.
- Running the loop with a real model needs a served policy per iteration (for example a vLLM
  endpoint reloaded with the new adapter) and remote training; this orchestration is future work.
- Failure categories are observations; any per-trajectory root-cause text is labelled speculative
  and kept in a separate `hypothesis` column.
