# ADR-002: Trajectory storage uses Parquet with DuckDB queries

- Status: Accepted
- Date: 2026-11-14

## Context

Rollouts produce many nested records (messages, actions, observations, verification, rewards,
lineage) that must be filtered and aggregated (by task, environment, policy, checkpoint, verifier
result, reward, seed, run) and turned into datasets. Runs are interrupted and resumed, and retried
trajectories must not be double-counted. The development machine has limited disk.

## Decision

Store trajectories in append-only, zstd-compressed Parquet shards with flat summary columns plus
the full trajectory as a JSON payload column. Query with DuckDB (`read_parquet`), always selecting
the latest record per `trajectory_id`. Shards are written atomically (temp file + rename).
Retention policies may drop payloads but never summary rows. A JSONL store and an in-memory store
implement the same protocol.

## Consequences

- Columnar filters and aggregates are fast without loading payloads; DuckDB needs no server.
- Nested data stays manageable (one JSON column) while analytics use typed columns.
- Append-only shards make writes crash-safe and resumable; latest-record semantics make retries
  safe.
- Many small shards can accumulate; compaction is future work.
- Payload columns duplicate prompts across samples; zstd keeps this small in practice.

## Alternatives considered

- **JSONL only**: simple but slow to query and large on disk; kept as a debugging backend.
- **SQLite**: row-oriented, awkward for nested payloads and analytics over many runs.
- **Fully normalised Parquet schema (messages as nested lists)**: harder to evolve, little benefit
  over a JSON payload for v0.1.
- **A database server**: unnecessary operational weight for a laptop-first project.
