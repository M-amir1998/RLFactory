"""Two-process unit-test harness executed inside the sandbox.

``main.py`` (trusted parent) receives the hidden test cases, including expected outputs and a
random report nonce, on stdin. It spawns ``runner.py`` (child), which imports the untrusted
``solution.py`` and receives **only the test inputs**. The child prints JSON-serialised return
values; the parent compares them with the expected outputs and prints nonce-tagged report lines.

Consequences:

* the generated code never has the expected outputs or the nonce in its address space, so it
  cannot read hidden answers or forge a passing report by introspecting the harness;
* results are compared after a JSON round-trip, which neutralises custom ``__eq__`` objects and
  ``int``/``str`` subclasses; ``bool`` and ``int`` are never conflated;
* if the child times out, results printed so far still earn partial credit.
"""

from __future__ import annotations

import json
import secrets

from rlfactory.domain.task import PythonFunctionPayload
from rlfactory.domain.trajectory import ExecutionReport, ExecutionStatus, UnitTestOutcome
from rlfactory.errors import SandboxError
from rlfactory.sandbox.base import ExecutionRequest, ExecutionResult, ResourceLimits, Sandbox

_PARENT = r"""
import json, math, subprocess, sys

def _equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b

def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value

def main():
    spec = json.loads(sys.stdin.read())
    marker = "@@RLFACTORY_" + spec.pop("nonce") + "@@"
    cases = spec["cases"]
    child_input = json.dumps({"entry_point": spec["entry_point"], "args": [c["args"] for c in cases]})
    timed_out = False
    command = [sys.executable, "-B", "-S", "-P", "-X", "utf8", "runner.py"]
    try:
        proc = subprocess.run(command, input=child_input.encode("utf-8"), capture_output=True,
                              timeout=spec["timeout"])
        out, err = _text(proc.stdout), _text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        out, err = _text(exc.stdout), _text(exc.stderr)
    results = {}
    load_error = None
    for line in out.splitlines():
        if not line.startswith("@@R@@"):
            continue
        try:
            item = json.loads(line[5:])
        except ValueError:
            continue
        if isinstance(item, list) and len(item) == 2 and item[0] == "load_error":
            load_error = str(item[1])
        elif isinstance(item, list) and len(item) == 3 and isinstance(item[0], int):
            results[item[0]] = (item[1], item[2])

    def emit(kind, payload):
        sys.stdout.write(marker + kind + ":" + json.dumps(payload) + "\n")

    emit("LOAD", {"ok": load_error is None and not (timed_out and not results), "error_type": load_error})
    for index, case in enumerate(cases):
        status, value = results.get(index, ("missing", None))
        if status == "ok":
            passed = _equal(value, case["expected"])
            error = None if passed else "wrong_output"
        elif status == "missing":
            passed, error = False, ("timeout" if timed_out else "no_result")
        else:
            passed, error = False, str(value)
        emit("TEST", [index, passed, error])
    emit("END", {"timed_out": timed_out, "stderr_tail": err[-400:]})
    sys.stdout.flush()

main()
"""

_RUNNER = r"""
import copy, importlib.util, json, sys

def emit(text):
    sys.__stdout__.write("@@R@@" + text + "\n")
    sys.__stdout__.flush()

def main():
    spec = json.loads(sys.stdin.read())
    try:
        module_spec = importlib.util.spec_from_file_location("solution", "solution.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        func = getattr(module, spec["entry_point"])
        if not callable(func):
            raise TypeError("entry point is not callable")
    except BaseException as exc:
        emit(json.dumps(["load_error", type(exc).__name__]))
        return
    for index, args in enumerate(spec["args"]):
        try:
            result = func(*copy.deepcopy(args))
            text = json.dumps([index, "ok", result], allow_nan=False)
        except BaseException as exc:
            text = json.dumps([index, "error", type(exc).__name__])
        emit(text)

main()
"""


def build_request(
    code: str, payload: PythonFunctionPayload, limits: ResourceLimits
) -> tuple[ExecutionRequest, str]:
    nonce = secrets.token_hex(16)
    child_timeout = max(0.5, limits.wall_time_s * 0.85 - 0.25)
    spec = {
        "nonce": nonce,
        "entry_point": payload.entry_point,
        "timeout": child_timeout,
        "cases": [{"args": list(case.args), "expected": case.expected} for case in payload.tests],
    }
    request = ExecutionRequest(
        files={"main.py": _PARENT, "runner.py": _RUNNER, "solution.py": code},
        entrypoint="main.py",
        stdin=json.dumps(spec),
        limits=limits,
    )
    return request, nonce


def parse_report(result: ExecutionResult, nonce: str, total: int) -> ExecutionReport:
    marker = f"@@RLFACTORY_{nonce}@@"
    loaded: bool | None = None
    error_type: str | None = None
    outcomes: dict[int, UnitTestOutcome] = {}
    finished = False
    timed_out = False
    child_stderr = ""
    for line in result.stdout.splitlines():
        if not line.startswith(marker):
            continue
        kind, _, raw = line[len(marker) :].partition(":")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if kind == "LOAD" and isinstance(data, dict):
            loaded = bool(data.get("ok"))
            error_type = data.get("error_type") or None
        elif kind == "TEST" and isinstance(data, list) and len(data) == 3:
            index = int(data[0])
            if 0 <= index < total:
                outcomes[index] = UnitTestOutcome(index=index, passed=bool(data[1]), error=data[2])
        elif kind == "END" and isinstance(data, dict):
            finished = True
            timed_out = bool(data.get("timed_out"))
            child_stderr = str(data.get("stderr_tail", ""))

    status = result.status
    if status is ExecutionStatus.OK:
        if timed_out:
            status = ExecutionStatus.TIMEOUT
        elif not finished or loaded is False:
            status = ExecutionStatus.RUNTIME_ERROR
    passed = sum(1 for outcome in outcomes.values() if outcome.passed)
    if error_type is None and status is not ExecutionStatus.OK and not outcomes:
        error_type = status.value
    return ExecutionReport(
        status=status,
        tests_passed=passed,
        tests_total=total,
        outcomes=tuple(outcomes[i] for i in sorted(outcomes)),
        error_type=error_type,
        report_found=finished,
        stderr_tail=(child_stderr or result.stderr)[-500:],
        duration_s=result.duration_s,
        sandbox=result.sandbox,
    )


def run_unit_tests(
    sandbox: Sandbox, code: str, payload: PythonFunctionPayload, limits: ResourceLimits
) -> ExecutionReport:
    """Execute ``code`` against the hidden tests. Raises :class:`SandboxError` on infra failure."""
    request, nonce = build_request(code, payload, limits)
    result = sandbox.execute(request)
    if result.status is ExecutionStatus.SANDBOX_ERROR:
        raise SandboxError(result.detail or "sandbox reported an internal error")
    return parse_report(result, nonce, len(payload.tests))
