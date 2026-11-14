# Environments (architecture)

User-facing details (formats, tools, commands) are in
[../environments/README.md](../environments/README.md). This page covers how environments fit the
runtime.

## Construction and dependency injection

`EnvironmentRegistry` maps a name to an `EnvironmentSpec(name, version, domain, factory,
multi_step, requires_sandbox)`. Factories receive an `EnvironmentContext`, which carries the shared,
thread-safe dependencies:

| Field | Purpose |
|---|---|
| `sandbox` | where generated code runs (`SubprocessSandbox`, `DockerSandbox`, `MockSandbox`) |
| `tools` | `ToolRegistry` (calculator, kv_lookup, json_transform) |
| `python_limits` | `ResourceLimits` for the Python environment |

A new environment instance is created for every episode and closed afterwards, so environments may
keep per-episode state (the tool-use environment tracks steps and tool state) without locks.

## Episode lifecycle

```
env = registry.create(task.environment, context)
first = env.reset(task)                       # prompt observation
messages = prompts.initial_messages(task, first)
while not done and turn < 64:
    action = policy.generate(messages)        # async, bounded by the generation timeout
    result = env.step(action)                 # run in a worker thread (may execute a sandbox)
    record(action, result)                    # assistant message (+ observation if not done)
env.close()
```

`EpisodeRecorder` (`rollouts/episode.py`) accumulates messages, actions and observations and builds
the `Trajectory`. The same recorder is used synchronously by task validation and by the
`CompletionScorer` that external trainers call, so every code path produces identical
trajectories. `observations[0]` is the reset observation and `observations[i + 1]` follows
`actions[i]`; the domain model validates this alignment.

## Failure semantics

| Event | Classification |
|---|---|
| unparseable output, missing tags, invalid JSON | model outcome (`parse_error` on the observation; trajectory still verified) |
| generated program times out / exceeds limits | model outcome (`execution.status`; `sandbox_timeout` failure kind for timeouts) |
| sandbox could not start or crashed | `SandboxError` -> `sandbox_error` (infrastructure, retryable) |
| environment construction or reset failed | `infrastructure_error` (not retried; likely a configuration bug) |

## Determinism

Environments are deterministic functions of the task and the action text. All randomness lives in
task generation (seeded per template and index) and in policies (seeded per task and sample).
