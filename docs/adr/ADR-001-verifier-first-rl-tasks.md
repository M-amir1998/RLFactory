# ADR-001: Verifier-first RL tasks

- Status: Accepted
- Date: 2026-11-14

## Context

Post-training signals can come from humans, reward models, LLM judges or programmatic checks. The
first three are expensive, noisy and themselves need validation; using them would make it
impossible to tell infrastructure bugs from signal problems. The project must also run its full
pipeline on a CPU-only laptop without paid APIs.

## Decision

The initial task domains are restricted to automatically verifiable tasks (arithmetic, structured
output, Python functions with hidden tests, deterministic text transformations, deterministic tool
use). Every task names a verifier. A generated task is accepted only if its reference solution
passes the verifier through the real environment and a known-wrong answer is rejected. LLM judges
exist as an optional verifier type and are never the primary signal.

## Consequences

- Rewards are objective, cheap and reproducible; RLVR experiments need no judge model.
- Verifier quality becomes a first-class concern (strict vs lenient parsing, partial credit), which
  is studied explicitly rather than hidden.
- Coverage is limited to verifiable skills; open-ended quality (helpfulness, style) is out of scope
  for v0.1.

## Alternatives considered

- **LLM-judge rewards first**: rejected; judge bias and cost would dominate early results.
- **Learned reward model first**: rejected; needs preference data and its own evaluation before it
  can be trusted.
- **External benchmarks as training tasks**: rejected; contamination risk and no control over
  verification.
