# Sandbox security

Generated code is untrusted. RLFactory never executes it in its own process; it goes through a
`Sandbox` (`rlfactory.sandbox`). This page states precisely what each implementation enforces.

**Bottom line**

- `SubprocessSandbox` **is not a security boundary.** It contains accidents (runaway loops,
  memory blow-ups, output floods), not attackers.
- `DockerSandbox` provides **stronger but still limited** isolation: containers share the host
  kernel, so a kernel or container-runtime vulnerability can still lead to escape.
- `MockSandbox` executes nothing and exists for tests.

Choose the sandbox per run (`sandbox.kind` in experiment and job configs: `subprocess`, `docker`,
`auto`, `mock`). For models that may be adversarial, use Docker on a disposable machine.

## Common request model

An `ExecutionRequest` contains only text files with relative POSIX paths (absolute paths, `..`,
backslashes and drive letters are rejected), an entrypoint, stdin and `ResourceLimits`:

| Limit | Default | Field |
|---|---|---|
| wall-clock time | 10 s | `wall_time_s` |
| CPU time | 10 s | `cpu_time_s` |
| memory | 512 MB | `memory_mb` |
| captured output per stream | 64 KB | `max_output_bytes` |
| size of any written file | 1 MB | `max_file_bytes` |
| processes (Docker) | 32 | `max_processes` |

Results report one of `ok`, `runtime_error`, `timeout`, `memory_exceeded`, `output_limit`,
`file_limit` or `sandbox_error`. Only `sandbox_error` (the sandbox itself failed) is an
infrastructure failure; everything else is an outcome of the generated program.

## SubprocessSandbox

Each execution:

1. creates a fresh temporary directory and writes the request files into `work/`;
2. starts the interpreter with `-B -S -P -X utf8`:
   `-S` skips `site` (no site-packages: the program sees the standard library only, not RLFactory,
   torch or any other installed package), `-P` does not prepend the script directory to
   `sys.path`, `-B` writes no bytecode, `-X utf8` fixes I/O encoding;
3. uses a minimal environment built from scratch: `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`,
   `PYTHONIOENCODING=utf-8`, `HOME`/`TMP`/`TEMP`/`TMPDIR` inside the work directory, `LANG=C.UTF-8`
   and, on Windows only, `SYSTEMROOT`. No variable is inherited, so tokens and keys in the parent
   environment are not visible;
4. redirects stdout/stderr to files and runs a watchdog that polls every 10 ms.

### Enforcement by platform

| Limit | POSIX (Linux, macOS) | Windows |
|---|---|---|
| wall-clock time | watchdog kills the whole process group (`start_new_session` + `killpg`) | watchdog kills the process tree (psutil) |
| CPU time | `RLIMIT_CPU` (soft < hard, so exhaustion arrives as `SIGXCPU` and is reported as `timeout`) | not enforced separately (wall clock applies) |
| memory | `RLIMIT_AS` | RSS of the process tree polled every 10 ms; best effort, a fast allocation can overshoot briefly |
| output size | watchdog on captured stream size | same |
| file size | `RLIMIT_FSIZE` plus watchdog scan of the work directory | watchdog scan only |
| core dumps | `RLIMIT_CORE = 0` | n/a |

POSIX limits are applied by a tiny trusted bootstrap that lowers the limits and then `execv`s the
real interpreter, instead of `preexec_fn`, which is unsafe in multi-threaded parents (rollouts run
sandbox calls from worker threads).

### Not provided

No network isolation, no filesystem isolation beyond the working directory, no privilege drop, no
syscall filtering. A deliberately malicious program can read files your user can read and talk to
the network.

## DockerSandbox

Each execution runs `docker run --rm` in a fresh container:

| Flag | Purpose |
|---|---|
| `--network none` | no network |
| `--user 65534:65534` | unprivileged numeric user |
| `--read-only` | read-only root filesystem |
| `--tmpfs /tmp:rw,noexec,nosuid,size=64m` | small scratch space, no executables |
| `--mount type=bind,...,readonly` | the work directory is mounted read-only |
| `--memory` = `--memory-swap` | hard memory limit, no swap |
| `--cpus`, `--pids-limit` | CPU share and process count |
| `--ulimit fsize=...`, `--ulimit cpu=...` | file size and CPU time (soft < hard: `SIGXCPU` -> timeout) |
| `--cap-drop ALL`, `--security-opt no-new-privileges` | no capabilities, no privilege escalation |
| `--init` | proper signal handling and zombie reaping |
| `--env PYTHONHASHSEED=0` etc. | deterministic, no inherited environment |

The Docker socket is never mounted into a sandbox container, and no host secrets are passed. The
image defaults to `python:3.12-slim`; RLFactory never pulls it implicitly (`rlfactory doctor`
reports whether it is present). Container start-up time is added to the wall-clock deadline
(`startup_allowance_s`, 15 s by default). Exit code 137 (OOM kill) maps to `memory_exceeded`, 152
(`SIGXCPU`) to `timeout`, 153 (`SIGXFSZ`) to `file_limit`, and 125-127 (Docker could not run the
container) raise `SandboxError`.

**Residual risk:** shared kernel, container runtime and Docker daemon vulnerabilities; resource
exhaustion of the host through many concurrent containers (bound the rollout `concurrency`).

## Python unit-test harness

The Python environment runs hidden tests through a two-process harness inside the sandbox:

```
sandbox process
 +-- main.py (trusted parent)   reads the spec from stdin: entry point, test inputs,
 |                              expected outputs and a random report nonce
 +-- runner.py (child)          receives ONLY the test inputs; imports solution.py,
                                calls the function, prints JSON-serialised return values
```

- The generated code runs in the child, whose address space never contains the expected outputs
  or the nonce, so it cannot read hidden answers or forge a passing report by introspecting the
  harness.
- Return values are compared after a JSON round trip (`allow_nan=False`): objects with custom
  `__eq__`, int/str subclasses and other non-JSON values cannot fake equality, and `True` is never
  accepted for `1` (or `False` for `0`).
- The parent emits nonce-tagged report lines; lines without the current nonce are ignored.
- If the child times out, results printed so far still count (partial credit for the
  `unit_test` verifier when enabled); missing results count as failures.

**Residual risks:** the child sees the test *inputs* (unavoidable); a program could detect it is
being tested and behave differently; `os._exit`, closing stdout or crashing only ever cause
*failures*, never passes. The reward-hacking analyser additionally flags passing Python solutions
that contain harness-tampering patterns (`sys._getframe`, `f_back`, `inspect`, `gc.get_objects`,
`os._exit`, `__eq__`, `builtins`, `ctypes`, `open(`), as a heuristic for manual review.

## Measured behaviour

Measured on Windows with the subprocess sandbox (5 s wall clock, 256 MB memory) against a
three-test `sum_evens` task:

| Submitted code | Outcome |
|---|---|
| correct implementation | `ok`, 3/3 tests passed |
| wrong implementation (`sum(values)`) | `ok`, 1/3 passed |
| syntax error | `runtime_error`, `SyntaxError` reported, 0/3 |
| infinite loop | `timeout` after about 4 s, every test marked `timeout` (partial-result path) |
| 2 GB allocation | killed by the memory watchdog: `memory_exceeded` |
| returns an object whose `__eq__` always returns `True` | 0/3 (`TypeError` during JSON serialisation) |
| returns `True` where `0` is expected | that test fails |
| `os._exit(0)` inside the function | 0/3, `no_result` for every test |
| prints 10 MB of output | contained by the harness (output goes to the parent, not the sandbox stream) |

These behaviours are the regression contract for the sandbox tests.
