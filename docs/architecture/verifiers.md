# Verifiers

A verifier answers one question: did this trajectory solve this task?

```python
class Verifier(Protocol):
    name: str
    version: str
    kind: VerifierKind  # deterministic | heuristic | model_judge

    def verify(self, task: Task, trajectory: Trajectory) -> VerificationResult: ...
```

`VerificationResult` is bounded and explicit:

| Field | Meaning |
|---|---|
| `passed` | binary decision used for accuracy (may be stricter than `score > 0`) |
| `score` | in `[0, 1]`; 1.0 = fully correct; a passing result must have a positive score |
| `reason_code` | `ok`, `wrong_answer`, `no_answer`, `invalid_format`, `invalid_json`, `schema_violation`, `value_mismatch`, `constraint_violation`, `no_code`, `syntax_error`, `runtime_error`, `timeout`, `resource_limit`, `tests_failed`, `required_tool_not_used`, `step_limit`, `pattern_mismatch`, `judge_rejected`, `judge_error`, ... |
| `verifier` | name and version (implementation version + config hash) |
| `components` | per-component results for composite verifiers |
| `metadata` | small JSON details (expected/got values, test counts, provenance) |

A verifier that cannot operate (e.g. a task without a reference answer) raises `VerifierError`,
which the rollout layer records as `verifier_error`. It never silently scores zero.

## Implementations

| Type | Default names | Semantics |
|---|---|---|
| `exact_match` | `exact_match`, `exact_strict` | text equality; the default normalises trailing whitespace; `exact_strict` does not normalise |
| `numeric` | `numeric`, `numeric_lenient` | exact rational comparison of parsed numbers (integers, decimals, fractions, `\frac`, thousands separators) with optional tolerance; `numeric_lenient` falls back to the last number in the completion when no `<answer>` tag exists |
| `json_schema` | `json_schema` | valid JSON, Draft 2020-12 schema, then the expected value (structural equality; booleans never equal numbers) and/or boolean constraints evaluated by the restricted evaluator |
| `unit_test` | `unit_test`, `unit_test_binary` | reads the hidden-test report from the Python environment; `passed` requires every test; `score` is the passed fraction when partial credit is on |
| `regex` | `answer_tag_format`, `python_block_format` | pattern on the completion or extracted answer; heuristic by default, deterministic when used as a pure format check |
| `tool_outcome` | `tool_outcome` | final answer (numeric if the reference is numeric) and, by default, successful use of the task's required tools |
| `composite` | `arithmetic_composite`, `python_composite` | weighted components (`correctness`, `format`, `constraint`); see below |
| `llm_judge` | none by default | optional judge over an OpenAI-compatible endpoint |

## Configuration

Verifiers are referenced by name from tasks (`Task.verifier`) and built lazily from typed specs.
Experiment and job configs can add or override specs under `verifiers:`; suites can switch the
verifier for a domain with `verifier_overrides` (used by the verifier-quality study). Each
verifier's version includes a hash of its spec, so changing a tolerance or weight changes the
lineage of every trajectory it scores. Composite references resolve recursively with cycle
detection.

## Composite verifiers

`score = sum(w_i * s_i) / sum(w_i)` and `passed` = every *required* component passed (if none is
marked required, all correctness components are required). By default correctness must carry at
least 50% of the total weight (`min_correctness_share`), so formatting can never dominate an RLVR
signal by accident; lower it only for explicit ablations.

## LLM judges

`LLMJudgeVerifier` is optional and never the primary mechanism. Each result records the judge
model, a hash of the prompt template and the temperature. Before trusting a judge, measure it with
`rlfactory.verifiers.calibration`:

- `binary_agreement(judge, reference)`: confusion matrix, accuracy, precision, recall, F1 and
  Cohen's kappa against deterministic or human labels;
- `verbosity_flip_rate(judge, responses, padding)`: how often content-free padding changes the
  verdict;
- `position_flip_rate(pairwise_judge, pairs)`: how often swapping candidates changes a pairwise
  preference.

## Coverage

`evaluation.coverage.verifier_coverage(tasks, registry)` reports how much of a suite is verified
deterministically, heuristically, by a judge, or not at all. Every built-in suite is 100%
deterministic.
