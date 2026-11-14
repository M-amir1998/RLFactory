"""Every environment: reset, valid action, invalid action, termination, determinism, cleanup."""

from __future__ import annotations

import json

import pytest

from rlfactory.domain.task import TaskDomain, ToolCall
from rlfactory.domain.trajectory import ModelAction, Role
from rlfactory.environments.base import EnvironmentContext, EpisodeStateError
from rlfactory.environments.parsing import render_tool_call
from rlfactory.environments.registry import EnvironmentSpec, default_environment_registry
from rlfactory.errors import RegistryError
from rlfactory.policies.oracle import ResponseMode
from rlfactory.sandbox.mock import MockSandbox
from tests.helpers import make_task, oracle_response

REGISTRY = default_environment_registry()
SINGLE_TURN = [
    ("arithmetic", "arithmetic.integer_chain"),
    ("structured_output", "structured_output.filter_sort"),
    ("transformation", "transformation.reverse_words"),
]


def _env(name: str) -> object:
    return REGISTRY.create(name, EnvironmentContext(sandbox=MockSandbox()))


@pytest.mark.parametrize(("name", "family"), SINGLE_TURN)
def test_single_turn_lifecycle(name: str, family: str) -> None:
    task = make_task(family)
    env = REGISTRY.create(name, EnvironmentContext(sandbox=MockSandbox()))
    with pytest.raises(EpisodeStateError):
        env.step(ModelAction(text="x"))
    first = env.reset(task)
    assert first.content == task.prompt
    result = env.step(ModelAction(text=oracle_response(task, ResponseMode.CORRECT)))
    assert result.done and result.observation.parse_error is None
    assert result.observation.parsed_answer is not None
    with pytest.raises(EpisodeStateError, match="finished"):
        env.step(ModelAction(text="again"))
    env.close()
    with pytest.raises(EpisodeStateError):
        env.step(ModelAction(text="after close"))


@pytest.mark.parametrize(("name", "family"), SINGLE_TURN)
def test_single_turn_invalid_action(name: str, family: str) -> None:
    env = REGISTRY.create(name, EnvironmentContext(sandbox=MockSandbox()))
    env.reset(make_task(family))
    result = env.step(ModelAction(text="I refuse to use the format."))
    assert result.done
    assert result.observation.parse_error is not None
    assert result.observation.parsed_answer is None


@pytest.mark.parametrize(("name", "family"), SINGLE_TURN)
def test_single_turn_deterministic(name: str, family: str) -> None:
    task = make_task(family)
    text = oracle_response(task, ResponseMode.CORRECT)
    outcomes = []
    for _ in range(2):
        env = REGISTRY.create(name, EnvironmentContext(sandbox=MockSandbox()))
        env.reset(task)
        outcomes.append(env.step(ModelAction(text=text)).observation)
    assert outcomes[0] == outcomes[1]


def test_environment_rejects_other_domains() -> None:
    env = REGISTRY.create("arithmetic", EnvironmentContext(sandbox=MockSandbox()))
    with pytest.raises(ValueError, match="cannot run"):
        env.reset(make_task("transformation.sort_lines"))


def test_structured_environment_flags_invalid_json() -> None:
    env = REGISTRY.create("structured_output", EnvironmentContext(sandbox=MockSandbox()))
    env.reset(make_task("structured_output.aggregate"))
    observation = env.step(ModelAction(text="<answer>{not json</answer>")).observation
    assert observation.parse_error == "invalid_json"


def test_python_environment_requires_code_block_and_uses_sandbox() -> None:
    sandbox = MockSandbox()
    env = REGISTRY.create("python", EnvironmentContext(sandbox=sandbox))
    task = make_task("python.sum_of_evens")
    env.reset(task)
    observation = env.step(ModelAction(text="def sum_of_evens(v): return 0")).observation
    assert observation.parse_error == "missing_code_block"
    assert sandbox.requests == []
    env.reset(task)
    observation = env.step(
        ModelAction(text="```python\ndef sum_of_evens(v):\n    return 0\n```")
    ).observation
    assert len(sandbox.requests) == 1
    assert observation.execution is not None
    request = sandbox.requests[0]
    assert set(request.files) == {"main.py", "runner.py", "solution.py"}
    spec = json.loads(request.stdin)
    assert "expected" in spec["cases"][0]
    # Hidden expectations never reach the untrusted files.
    assert str(spec["cases"][0]["expected"]) not in request.files["solution.py"]


def test_tool_use_multi_step_success_path() -> None:
    task = make_task("tool_use.inventory_value", seed=3)
    env = REGISTRY.create("tool_use", EnvironmentContext(sandbox=MockSandbox()))
    first = env.reset(task)
    assert "Available tools" in first.content and "kv_lookup" in first.content
    plan = task.payload.reference_plan  # type: ignore[union-attr]
    for call in plan:
        result = env.step(ModelAction(text=render_tool_call(call)))
        assert not result.done
        assert result.observation.role is Role.TOOL
        assert result.observation.tool is not None and result.observation.tool.succeeded
    final = env.step(ModelAction(text=f"<answer>{task.payload.expected_answer}</answer>"))  # type: ignore[union-attr]
    assert final.done and final.observation.parsed_answer == task.payload.expected_answer  # type: ignore[union-attr]


def test_tool_use_misuse_and_truncation() -> None:
    task = make_task("tool_use.order_total", seed=1)
    env = REGISTRY.create("tool_use", EnvironmentContext(sandbox=MockSandbox()))
    env.reset(task)
    unknown = env.step(
        ModelAction(text=render_tool_call(ToolCall(name="shell", arguments={"cmd": "ls"})))
    )
    assert unknown.observation.tool is not None and "not available" in (
        unknown.observation.tool.error or ""
    )
    malformed = env.step(ModelAction(text="<tool_call>{oops</tool_call>"))
    assert malformed.observation.tool is not None and malformed.observation.tool.call is None
    chatter = env.step(ModelAction(text="thinking..."))
    assert chatter.observation.parse_error == "no_tool_call_or_answer"
    last = env.step(ModelAction(text="still thinking"))
    assert last.done and last.truncated  # max_steps == 4 for order_total


def test_tool_call_takes_precedence_over_answer() -> None:
    task = make_task("tool_use.currency_conversion", seed=2)
    env = REGISTRY.create("tool_use", EnvironmentContext(sandbox=MockSandbox()))
    env.reset(task)
    text = render_tool_call(task.payload.reference_plan[0]) + "<answer>1</answer>"  # type: ignore[union-attr]
    assert not env.step(ModelAction(text=text)).done


def test_registry_specs_and_errors() -> None:
    assert REGISTRY.names() == [
        "arithmetic",
        "python",
        "structured_output",
        "tool_use",
        "transformation",
    ]
    assert REGISTRY.spec("tool_use").multi_step
    assert REGISTRY.spec("python").requires_sandbox
    assert {s.domain for s in REGISTRY.specs()} == set(TaskDomain)
    with pytest.raises(RegistryError, match="unknown environment"):
        REGISTRY.spec("browser")
    with pytest.raises(RegistryError, match="already"):
        REGISTRY.register(REGISTRY.spec("arithmetic"))
    assert isinstance(REGISTRY.spec("arithmetic"), EnvironmentSpec)
