# ADR-005: Environment sandbox strategy

- Status: Accepted
- Date: 2026-11-14

## Context

The Python environment executes model-generated code. Executing it in-process would let any
completion read secrets, corrupt state or crash the run. The project must work on Windows, macOS
and Linux laptops, in CI without Docker privileges, and on GPU hosts where Docker is usually
available.

## Decision

All execution goes through a `Sandbox` protocol with three implementations:

1. `SubprocessSandbox` (default, portable): separate stdlib-only interpreter (`-S -P -B`), fresh
   temp directory, secret-free environment, wall-clock/output/file limits via a watchdog, POSIX
   rlimits via a trusted bootstrap, Windows memory via RSS polling. Documented as **not** a
   security boundary.
2. `DockerSandbox` (stronger): no network, non-root, read-only root, `noexec` tmpfs, dropped
   capabilities, `no-new-privileges`, cgroup limits, read-only work mount, Docker socket never
   mounted. Documented as defence in depth on a shared kernel.
3. `MockSandbox` for tests.

Unit tests use a two-process harness so generated code never sees expected outputs or the report
nonce.

## Consequences

- The pipeline runs everywhere; strong isolation is one config switch away.
- Security claims stay honest; documentation lists residual risks and measured behaviours.
- Docker adds about a second of start-up per execution; it is not the default for CPU CI.

## Alternatives considered

- **In-process `exec` with restricted builtins**: trivially escapable; rejected.
- **gVisor / Firecracker / VMs**: stronger, but not portable to Windows laptops; candidates for
  future GPU-host deployments.
- **seccomp/landlock in the subprocess sandbox**: Linux-only; possible future hardening.
