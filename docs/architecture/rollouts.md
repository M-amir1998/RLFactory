# Rollouts

Rollouts turn tasks into verified, rewarded trajectories:

```
Task -> Policy -> Environment -> Verifier -> Reward -> TrajectoryStore
```

## Jobs and idempotent IDs

`plan_jobs(tasks, config, policy_id, checkpoint)` creates one `RolloutJob` per (task, sample):

- `trajectory_id = "trj-" + hash(task_id, task fingerprint, policy_id, checkpoint, generation-config
  fingerprint, rollout seed, sample index, verifier name, reward name)`;
- `group_id = "grp-" + hash(task, policy, checkpoint, generation config, seed, verifier)`: all
  samples of one task under one configuration (a GRPO group and the only pool preference pairs
  are drawn from);
- per-sample seeds derive from the rollout seed, task and sample; per-turn request seeds from that.

Re-running an identical configuration maps onto the same IDs, so completed work is skipped;
anything that could change the outcome changes the ID.

## Worker

`RolloutWorker.run(job, attempt)` executes one episode and always returns a `Trajectory`:

1. builds the `Lineage` (task version and fingerprint, environment/verifier/reward/prompt template
   versions, policy, model, checkpoint, generation config, seed, code version);
2. alternates `policy.generate` (wrapped in `asyncio.wait_for` with the generation timeout) and
   `environment.step` (in a worker thread) until the episode ends or 64 turns;
3. builds a `PENDING` trajectory, runs the verifier (in a thread) and the reward function, and
   finalises it as `COMPLETED`;
4. records model-side failures (`invalid_output`, `sandbox_timeout`) on completed trajectories;
5. converts infrastructure failures into a `FAILED` trajectory that keeps partial messages for
   debugging and a `FailureRecord(kind, message, attempts, retryable)`.

## Failure taxonomy

| Kind | Origin | Retried by default | Typical cause |
|---|---|---|---|
| `policy_timeout` | infrastructure | yes | server overloaded, generation timeout |
| `policy_error` | infrastructure | if `retryable` | HTTP 429/5xx, transport error (retryable); HTTP 4xx (not retryable) |
| `sandbox_error` | infrastructure | yes | sandbox could not start |
| `infrastructure_error` | infrastructure | yes | unexpected exception, episode timeout |
| `verifier_error` | infrastructure | no | verifier bug; deterministic, retrying would not help |
| `invalid_output` | model | never | unparseable output, no answer within the step limit |
| `sandbox_timeout` | model | never | generated program ran too long |

Model failures are legitimate (negative) training signal and are rewarded like any wrong answer.
Infrastructure failures are excluded from every dataset and from accuracy denominators.

## Orchestrator

`RolloutOrchestrator.run(jobs, cancel)`:

- **bounded concurrency**: exactly `concurrency` consumer coroutines in an `asyncio.TaskGroup` pull
  from a queue of size `2 x concurrency`; a producer feeds it. No task is created per job;
- **retries**: `RetryPolicy(max_attempts=3, backoff 0.5 s x 2^n capped at 10 s, retry_on=...)`,
  only for failed trajectories whose failure is retryable and listed in `retry_on`;
- **timeouts**: generation timeout (`generation.timeout_s`) and a whole-episode timeout
  (`episode_timeout_s`, default 600 s);
- **cancellation**: an `asyncio.Event` stops the producer and lets in-flight work finish; Ctrl+C
  cancels the task group. Buffered trajectories are flushed in a `finally` block in both cases;
- **resume**: with `resume: true` (default) IDs whose latest stored record is `COMPLETED` are
  skipped, while previously failed IDs are attempted again;
- **progress**: a `ProgressReporter` protocol (`start`, `advance`, `finish`); the CLI uses Rich.

Trajectories are buffered and written in batches (`flush_every`, default 64). The store's
latest-record semantics make a retried trajectory replace its failed record in every query.

## Summary and metrics

`RolloutSummary` (written to `rollout_summary.json`) reports totals, skipped IDs, retries, pass
rate, mean reward, infrastructure failure rate, verification rate, trajectories/s, tokens/s (with
an `estimated` flag), failures by kind, reason codes and per-domain statistics.
`RolloutMetrics` keeps counters and histograms (trajectories, failures, retries, policy /
environment / verifier latency, reward distribution, sandbox failures) in an in-process registry
that can render Prometheus text format.

## Policies

| Policy | Use |
|---|---|
| `MockPolicy` | deterministic simulation for CI: per-domain correct probability, malformed, verbose, timeout and error probabilities, latency; outcome draws use common random numbers (independent of policy ID) |
| `ScriptedPolicy` | fixed responses per task/turn and scripted failures for tests |
| `OpenAICompatiblePolicy`, `VLLMPolicy`, `SGLangPolicy` | HTTP chat-completions with explicit sampling parameters; vLLM uses `max_completion_tokens`, removed `guided_*` fields are rejected |
| `TransformersPolicy` | local model, lazy import, safetensors only, `local_files_only=True` by default |

Real policies never see `task_id` beyond request metadata; only simulation policies use it.
