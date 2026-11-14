# Environments

Environments execute model actions and return observations. They do **not** decide correctness;
verifiers do (see [../architecture/verifiers.md](../architecture/verifiers.md)). This separation
lets the same trajectories be re-scored under a different verifier.

```python
class Environment(Protocol):
    name: str
    version: str

    def reset(self, task: Task) -> Observation: ...
    def step(self, action: ModelAction) -> StepResult: ...
    def close(self) -> None: ...
```

An `Observation` has `content` (the only part shown to the policy) plus typed side channels for
verifiers and analysis: `parsed_answer`, `parse_error`, `execution` (hidden unit-test report) and
`tool` (tool-call record). `StepResult` adds `done` and `truncated`. Instances serve one episode
at a time and are created per episode by the `EnvironmentRegistry`.

| Name | Kind | Domain | Executes | Default verifier |
|---|---|---|---|---|
| `arithmetic` | single-turn | arithmetic | extracts `<answer>` | `numeric` |
| `structured_output` | single-turn | structured_output | extracts and JSON-parses `<answer>` | `json_schema` |
| `python` | single-turn | python | runs the code block against hidden tests in a sandbox | `unit_test` |
| `transformation` | single-turn | transformation | extracts `<answer>` | `exact_match` |
| `tool_use` | multi-step | tool_use | executes one tool call per turn until `<answer>` or the step limit | `tool_outcome` |

Check any environment end to end (reference solutions must pass, known-wrong answers must fail):

```bash
uv run rlfactory environments list
uv run rlfactory environments check python --samples 4 --sandbox subprocess
uv run rlfactory environments check python --sandbox docker     # needs Docker + python:3.12-slim
```

## Output-format conventions

Defined once in `rlfactory.environments.parsing` and shared by environments, verifiers, rewards,
the mock-policy oracle and the prompt builder:

| Convention | Rule |
|---|---|
| final answer | `<answer>...</answer>`; the **last** occurrence wins; content is stripped |
| Python solution | a fenced ```` ```python ```` block (the last one wins; an untagged fence is a fallback) |
| JSON answer | `<answer>` content (an inner ```` ```json ```` fence is tolerated) or a ```` ```json ```` block |
| tool call | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` |

The system prompt (`PromptBuilder`, template `rlfactory-chat` v1) states the convention for each
domain, and the same builder renders rollout, evaluation and training prompts.

## Single-turn environments

- **arithmetic**: reports `missing_answer_tag` or `empty_answer` as a parse error; numeric
  equivalence (`13.5` vs `27/2`) is the verifier's decision.
- **structured_output**: a missing tag or invalid JSON is a parse error (`invalid_json`), so format
  compliance can be measured separately from correctness.
- **transformation**: exact text answers; the verifier normalises trailing whitespace only.
- **python**: requires a fenced code block; execution uses the configured sandbox and the
  two-process harness described in [../security/sandbox.md](../security/sandbox.md). The
  observation shown to the policy contains only the execution status, never hidden test details.

## Multi-step tool use

Each turn the model either calls exactly one tool or gives a final answer. If a turn contains both,
the tool call wins and the answer is ignored. Invalid calls (malformed JSON, unknown or unavailable
tool, bad arguments) return an error observation instead of ending the episode; exceeding
`max_steps` truncates it. Tool observations are sent back as user turns prefixed with
`Tool result:` so they are accepted by every chat template and OpenAI-compatible server.

### Tool registry

| Tool | Arguments | Behaviour |
|---|---|---|
| `calculator` | `expression` | exact arithmetic via the restricted evaluator (`+ - * / // % **`, `abs`, `min`, `max`); returns a canonical number string |
| `kv_lookup` | `key` | exact-key lookup in the task's key-value store |
| `json_transform` | `op`, `path`, `field`, `where`, `descending` | `get`, `keys`, `count`, `sum`, `sort`, `filter` over the task's JSON document |

Tasks are designed so that the facts needed for the answer are only reachable through the tools,
and the `tool_outcome` verifier requires the task's `required_tools` to have been used successfully.

**There is no shell, network or filesystem tool.** A tool that runs commands would turn every
rollout into remote code execution on the rollout host. Generated code runs only through the
sandbox, never through a tool.

## Adding an environment

1. Implement the protocol (subclass `SingleTurnEnvironment` for one-shot tasks and implement
   `evaluate(task, action) -> Observation`).
2. Register an `EnvironmentSpec` (name, version, domain, factory, `multi_step`,
   `requires_sandbox`).
3. Add a task template with a reference solution so `TaskValidator` can prove every generated task
   is solvable and discriminative.
4. Add tests for reset, valid action, invalid action, termination, determinism and cleanup.
