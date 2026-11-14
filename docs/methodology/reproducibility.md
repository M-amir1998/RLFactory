# Reproducibility

## What every trajectory records

`Lineage` is stored on every trajectory (and as flat columns in the store):

| Field | Source |
|---|---|
| `run_id` | rollout or experiment run |
| `task_version`, `task_fingerprint` | task metadata and content hash (prompt, payload, version) |
| `environment` | name and version |
| `verifier` | name and version; the version embeds a hash of the verifier's config |
| `reward` | name and version (config hash included) |
| `prompt_template` | `rlfactory-chat` and its version |
| `policy_id`, `model`, `checkpoint` | policy identity |
| `generation` | full generation config (temperature, top-p, top-k, max tokens, stop, timeout) |
| `seed` | per-sample seed |
| `code` | RLFactory version, Git SHA and dirty flag (`RLFACTORY_GIT_SHA` overrides for images without `.git`) |

## Determinism

- Task generation is seeded per template and index; tasks are pure functions of template
  parameters, and task IDs hash those parameters.
- Splits hash a group key with a salt, independent of task order.
- Trajectory IDs hash content and configuration; resumes skip completed IDs.
- Request seeds are derived per sample and turn and sent to model servers (`seed` field).
- Sandboxed programs run with `PYTHONHASHSEED=0`.
- `stable_hash` uses SHA-256 over canonical JSON, never Python's randomised `hash()`.

## Run manifest

`manifest.json` captures run ID, experiment name, creation time, code version, seed, task suite
references (name, version, size, fingerprint), environment, verifier and reward versions, policy
descriptor, generation config, training backend, hardware (platform, Python, CPUs, RAM, GPU) and a
fingerprint of the full experiment config. `config.yaml` is the exact config used.

## Stage caching

A stage is reused only if its previous record completed and its fingerprint is unchanged. The
fingerprint hashes the stage's own config slice, the fingerprints of its dependencies and a hash of
every RLFactory source file, so any code change invalidates caches. The report stage always re-runs.
`--force STAGE` recomputes a stage; dependants recompute when the result changes their inputs.

## Datasets and checkpoints

Dataset cards record builder, builder config, build report (exclusion reasons), validation report,
source run IDs, contamination-filter sizes, code version and a content fingerprint. Checkpoint
metadata records base model and revision, method, dataset fingerprint, Git SHA, training config,
framework version and metrics.

## Cross-platform check

The `ci-e2e` experiment was run with the same config and run ID on Windows 11 and in the Linux CLI
container (`docker build -t rlfactory:local .`). All 140 trajectories across the rollout store, both
evaluation stores and the flywheel store matched on actions, observations (including sandboxed
unit-test results), verification, reward components and attempts; the five training datasets were
byte-identical; gate, flywheel, reward-analysis and training summaries were equal. Only the recorded
Git SHA differed (the image was built with `RLFACTORY_GIT_SHA=unknown`). Timings were excluded.
This covers the mock policy and mock trainer only; real-model runs remain subject to the limits
below.

## What is not reproducible bit-for-bit

- GPU kernels and server-side batching (vLLM, SGLang) are not deterministic even with a seed.
- Model servers and frameworks change behaviour across versions: record `vllm`, `trl`,
  `transformers` and driver versions with real runs (`rlfactory doctor` on the GPU host).
- Unpinned model revisions can change silently; remote training therefore requires a pinned
  revision.
- Wall-clock metrics (latency, throughput) depend on the machine.

Report these alongside results; RLFactory's lineage makes the rest exact.
