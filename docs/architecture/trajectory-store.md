# Trajectory store

```python
class TrajectoryStore(Protocol):
    def write(self, trajectories: Sequence[Trajectory]) -> None: ...  # rejects PENDING
    def flush(self) -> None: ...
    def completed_ids(self, run_id: str | None = None) -> set[str]: ...
    def query(self, query: TrajectoryQuery) -> list[TrajectorySummary]: ...
    def load(self, query: TrajectoryQuery) -> list[Trajectory]: ...
    def aggregate(self, group_by: Sequence[str], query: TrajectoryQuery) -> list[dict]: ...
```

Implementations: `LocalParquetTrajectoryStore` (default), `JSONLTrajectoryStore` (human-readable,
small runs) and `MemoryTrajectoryStore` (tests, ephemeral evaluations).

## Parquet layout

```
<run>/rollouts/part-<utc timestamp>-<8 hex>.parquet     zstd-compressed, written via temp file + rename
```

Each row holds flat summary columns plus the full trajectory as JSON:

| Group | Columns |
|---|---|
| identity | `trajectory_id`, `run_id`, `task_id`, `task_version`, `task_family`, `domain`, `group_id`, `sample_index`, `seed` |
| components | `environment(_version)`, `verifier(_version)`, `reward_fn`, `reward_version`, `prompt_template`, `policy_id`, `model`, `checkpoint`, `generation_fingerprint` |
| outcome | `status`, `failure_kind`, `failure_origin`, `attempts`, `passed`, `score`, `reason_code`, `truncated` |
| reward | `reward`, `reward_correctness`, `reward_format`, `reward_length`, `reward_tool_cost`, `reward_safety` |
| cost | `num_turns`, `prompt_tokens`, `completion_tokens`, `completion_chars`, `tokens_estimated`, `policy_latency_s`, `wall_time_s` |
| provenance | `git_sha`, `rlfactory_version`, `started_at`, `finished_at`, `written_at` |
| payload | `payload_retained`, `payload` (full trajectory JSON or null) |

Timestamps are stored as naive UTC and re-annotated as UTC on read (DuckDB would otherwise need
`pytz` to return timezone-aware values).

## Latest-record semantics

Records are append-only. A trajectory can be written more than once (an infrastructure failure,
then a successful retry or resume). Every query reads

```sql
SELECT ... FROM read_parquet(?, union_by_name=true)
QUALIFY row_number() OVER (PARTITION BY trajectory_id ORDER BY written_at DESC) = 1
```

so retried work is never double-counted. The shard glob and every filter value are bound
parameters; column lists come from the fixed schema and `GROUP BY` keys from a whitelist.

## Queries

`TrajectoryQuery` filters (combined with AND): run, task or task list, domain, environment,
policy, checkpoint, verifier, passed, status, failure kind, reward range, seed and limit.
`aggregate()` returns counts, completed/failed, pass rate, mean and std of reward, and mean
completion tokens per group.

```bash
uv run rlfactory trajectories inspect <run-id>
uv run rlfactory trajectories query <run-id> --domain python --failed --limit 20
uv run rlfactory trajectories query <run-id> --min-reward 0.5 --json
uv run rlfactory trajectories show <run-id> <trajectory-id>
```

## Retention

`RolloutConfig.retention` controls which full payloads are kept; the summary row is always kept:

| Policy | Payload kept for |
|---|---|
| `all` | every trajectory |
| `successful_only` | verified-correct trajectories |
| `failed_only` | everything else |
| `sampled` | a deterministic hash-based fraction (`retention_sample_rate`) |

Dataset builders only see trajectories whose payload was retained.

## Privacy

Trajectories can contain model outputs and prompts that must not be published. `results/` is
git-ignored; only synthetic fixtures under `tests/fixtures/` and reviewed example reports are
committed.

## Future backends

An object-store backend needs to provide the same append-only shard listing and atomic shard
publication; DuckDB can then read the shards through its `httpfs` extension. It is not implemented
in v0.1.
