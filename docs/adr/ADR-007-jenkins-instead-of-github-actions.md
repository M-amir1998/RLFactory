# ADR-007: Jenkins instead of GitHub Actions

- Status: Accepted
- Date: 2026-11-14

## Context

The project owner does not want to consume GitHub Actions minutes, wants CI that can run on
self-hosted hardware next to GPU machines later, and wants explicit control over resource usage on
a 16 GB laptop.

## Decision

CI is a declarative `Jenkinsfile`. The repository contains no `.github/workflows/` directory, and a
hygiene script fails the build if one appears. The pipeline is CPU-only, uses
`disableConcurrentBuilds()` and `timestamps()`, runs heavyweight stages sequentially, uses tiny
deterministic fixtures, and archives only JUnit XML, coverage XML, small reports and the built
package (never checkpoints or rollout data).

## Consequences

- No hosted CI minutes are consumed; builds can run on any Jenkins agent with Python, uv and
  (optionally) Docker.
- Contributors without Jenkins run the same gates locally (see CONTRIBUTING.md).
- Pull requests on GitHub do not show CI status unless a Jenkins integration is configured.

## Alternatives considered

- **GitHub Actions**: excluded by requirement.
- **Self-hosted GitHub Actions runner**: still couples CI configuration to GitHub workflows, which
  the owner wants to avoid.
- **Make/tox only**: no history, artefacts or scheduling.
