# Test fixtures

Only synthetic content belongs here. Real rollouts stay in `results/`, which Git ignores.

## `trajectories/synthetic-v1.jsonl`

Fourteen trajectories, one JSON object per line, written by
[`scripts/make_fixture_trajectories.py`](../../scripts/make_fixture_trajectories.py):

- tasks come from RLFactory's own templates (all five domains, including a multi-step tool-use
  task);
- answers come from the mock policy with injected timeouts and server errors, so the file holds
  verified passes, model failures (`invalid_output`) and infrastructure failures
  (`policy_timeout`);
- Python answers were executed in the subprocess sandbox;
- the Git SHA in each record's lineage is a placeholder (`000...0`), and no model, dataset,
  network access or user data was involved.

[`tests/unit/test_fixture_trajectories.py`](../unit/test_fixture_trajectories.py) checks that every
committed fixture still loads with the current schema, keeps its outcome semantics, contains no
machine- or person-specific strings, and round-trips through the Parquet store.

When the trajectory schema changes on purpose, write a new version (`synthetic-v2.jsonl`) and keep
the old one, so that the test keeps proving older stored trajectories remain readable.
