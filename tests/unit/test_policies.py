from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from rlfactory.domain.model import GenerationConfig
from rlfactory.domain.task import TaskDomain
from rlfactory.domain.trajectory import Message, Role
from rlfactory.errors import PolicyError, PolicyTimeoutError
from rlfactory.policies.base import GenerationRequest, to_chat_messages
from rlfactory.policies.factory import HTTPPolicySpec, MockPolicySpec, build_policy, describe_policy
from rlfactory.policies.http import OpenAICompatiblePolicy, SGLangPolicy, VLLMPolicy
from rlfactory.policies.mock import MockPolicy, MockPolicyProfile
from rlfactory.policies.oracle import ResponseMode
from rlfactory.policies.scripted import ScriptedPolicy
from tests.helpers import make_task, tasks_for


def _request(
    task_id: str = "t", *, sample: int = 0, attempt: int = 1, config: GenerationConfig | None = None
) -> GenerationRequest:
    return GenerationRequest(
        request_id="r",
        messages=(Message(role=Role.SYSTEM, content="s"), Message(role=Role.USER, content="u")),
        config=config or GenerationConfig(temperature=0.7, max_tokens=64, stop=("</answer>",)),
        seed=123,
        task_id=task_id,
        sample_index=sample,
        attempt=attempt,
    )


def test_mock_policy_is_deterministic() -> None:
    task = make_task("arithmetic.integer_chain")
    policy = MockPolicy([task], seed=5)
    first = asyncio.run(policy.generate(_request(task.task_id)))
    second = asyncio.run(policy.generate(_request(task.task_id)))
    assert first.text == second.text
    assert first.usage.estimated and first.usage.completion_tokens > 0


def test_mock_policy_common_random_numbers_monotone() -> None:
    tasks = tasks_for(TaskDomain.ARITHMETIC, per_family=10)
    weak = MockPolicy(tasks, MockPolicyProfile(correct_probability=0.3), seed=1, policy_id="weak")
    strong = MockPolicy(
        tasks, MockPolicyProfile(correct_probability=0.8), seed=1, policy_id="strong"
    )
    for task in tasks:
        for sample in range(3):
            if weak.episode_mode(task, sample)[0] is ResponseMode.CORRECT:
                assert strong.episode_mode(task, sample)[0] is ResponseMode.CORRECT


def test_mock_policy_outcome_rates_follow_profile() -> None:
    tasks = tasks_for(TaskDomain.ARITHMETIC, per_family=50)
    policy = MockPolicy(
        tasks, MockPolicyProfile(correct_probability=0.5, malformed_probability=0.2), seed=3
    )
    modes = [policy.episode_mode(task, s)[0] for task in tasks for s in range(2)]
    correct = sum(m is ResponseMode.CORRECT for m in modes) / len(modes)
    malformed = sum(m is ResponseMode.MALFORMED for m in modes) / len(modes)
    assert 0.3 < correct < 0.5 and 0.1 < malformed < 0.3


def test_mock_policy_injected_faults_vary_by_attempt() -> None:
    task = make_task("arithmetic.percentage")
    policy = MockPolicy(
        [task], MockPolicyProfile(timeout_probability=0.5, error_probability=0.5), seed=0
    )
    outcomes = set()
    for attempt in range(1, 20):
        try:
            asyncio.run(policy.generate(_request(task.task_id, attempt=attempt)))
            outcomes.add("ok")
        except PolicyTimeoutError:
            outcomes.add("timeout")
        except PolicyError:
            outcomes.add("error")
    assert outcomes == {"ok", "timeout", "error"}


def test_mock_policy_unknown_task_is_non_retryable() -> None:
    with pytest.raises(PolicyError) as excinfo:
        asyncio.run(MockPolicy([]).generate(_request("missing")))
    assert excinfo.value.retryable is False


def test_scripted_policy_turns_failures_and_defaults() -> None:
    policy = ScriptedPolicy(
        {"t": ["one", "two"]}, failures={("t", 1): PolicyTimeoutError()}, default="fallback"
    )
    with pytest.raises(PolicyTimeoutError):
        asyncio.run(policy.generate(_request("t")))
    assert asyncio.run(policy.generate(_request("t", attempt=2))).text == "one"
    assert asyncio.run(policy.generate(_request("other", attempt=2))).text == "fallback"
    callable_policy = ScriptedPolicy(lambda request: f"turn {request.turn}")
    assert asyncio.run(callable_policy.generate(_request())).text == "turn 0"


def test_tool_messages_become_user_turns() -> None:
    converted = to_chat_messages([Message(role=Role.TOOL, content="42")])
    assert converted == [{"role": "user", "content": "Tool result:\n42"}]


def _transport(handler):  # type: ignore[no-untyped-def]
    return httpx.MockTransport(handler)


def test_http_policy_success_and_request_body() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [
                    {"message": {"content": "<answer>4</answer>"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    policy = VLLMPolicy("p", "http://server:8000/", "m", transport=_transport(handler))
    config = GenerationConfig(temperature=0.5, top_p=0.9, top_k=20, max_tokens=32)
    result = asyncio.run(policy.generate(_request(config=config)))
    assert (
        result.text == "<answer>4</answer>"
        and result.usage.completion_tokens == 3
        and not result.usage.estimated
    )
    assert captured["path"] == "/v1/chat/completions"
    assert captured["max_completion_tokens"] == 32 and "max_tokens" not in captured
    assert captured["top_k"] == 20 and captured["seed"] == 123 and captured["temperature"] == 0.5
    asyncio.run(policy.aclose())


@pytest.mark.parametrize(("status", "retryable"), [(503, True), (429, True), (400, False)])
def test_http_policy_error_classification(status: int, retryable: bool) -> None:
    policy = OpenAICompatiblePolicy(
        "p", "http://s", "m", transport=_transport(lambda r: httpx.Response(status, text="x"))
    )
    with pytest.raises(PolicyError) as excinfo:
        asyncio.run(policy.generate(_request()))
    assert excinfo.value.retryable is retryable


def test_http_policy_timeout_and_transport_errors() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(PolicyTimeoutError):
        asyncio.run(
            OpenAICompatiblePolicy("p", "http://s", "m", transport=_transport(timeout)).generate(
                _request()
            )
        )
    with pytest.raises(PolicyError) as excinfo:
        asyncio.run(
            OpenAICompatiblePolicy("p", "http://s", "m", transport=_transport(refused)).generate(
                _request()
            )
        )
    assert excinfo.value.retryable
    malformed = OpenAICompatiblePolicy(
        "p", "http://s", "m", transport=_transport(lambda r: httpx.Response(200, json={}))
    )
    with pytest.raises(PolicyError, match="malformed"):
        asyncio.run(malformed.generate(_request()))


def test_http_policy_guards() -> None:
    with pytest.raises(ValueError, match="guided_json"):
        OpenAICompatiblePolicy("p", "http://s", "m", extra_body={"guided_json": {}})
    plain = OpenAICompatiblePolicy(
        "p", "http://s", "m", transport=_transport(lambda r: httpx.Response(200))
    )
    with pytest.raises(PolicyError, match="top_k"):
        plain.build_body(_request(config=GenerationConfig(top_k=5)))
    with pytest.raises(PolicyError, match="not set"):
        OpenAICompatiblePolicy("p", "http://s", "m", api_key_env="RLFACTORY_TEST_UNSET_KEY")


def test_http_policy_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    monkeypatch.setenv("RLFACTORY_TEST_KEY", "secret-value")
    policy = SGLangPolicy(
        "p", "http://s", "m", api_key_env="RLFACTORY_TEST_KEY", transport=_transport(handler)
    )
    result = asyncio.run(policy.generate(_request()))
    assert seen == ["Bearer secret-value"]
    assert result.usage.estimated
    assert asyncio.run(policy.health()) is True


def test_factory_builds_and_describes() -> None:
    task = make_task("arithmetic.integer_chain")
    mock = build_policy(MockPolicySpec(policy_id="base"), [task])
    assert mock.simulated and mock.policy_id == "base"
    spec = HTTPPolicySpec(kind="vllm", policy_id="v", base_url="http://x", model="m")
    assert isinstance(build_policy(spec), VLLMPolicy)
    assert describe_policy(spec).kind == "vllm"
