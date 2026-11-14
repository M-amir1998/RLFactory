# RLFactory

RLFactory is a production-oriented framework for generating verified LLM rollouts, constructing post-training datasets, running SFT/preference/RL workflows, and evaluating checkpoints through reproducible verifier-driven experiments.

**Status:** `0.1.0.dev0`. The CPU-only control plane (tasks, environments, sandbox, verifiers, rewards,
rollouts, trajectory store, dataset builders, training backends, evaluation, gates, flywheel,
reporting) is implemented and tested. **Real GPU post-training experiments are pending.**

---

## Why RLFactory

Most post-training repositories are `notebook -> trainer -> model`. The trainer is the easy part.
What decides whether an SFT, DPO or GRPO run means anything is the infrastructure around it:

- tasks whose answers can be **verified objectively** (numbers, unit tests, schemas, tool outcomes);
- environments that execute model actions, with generated code isolated in a **sandbox**;
- **verification kept separate from reward construction**, so a reward can be audited against
  correctness instead of being mistaken for it;
- rollouts that are **idempotent, resumable and lineage-tracked**, where infrastructure failures
  never silently become training data;
- dataset builders that **cannot leak evaluation tasks** into training;
- one evaluator for every method, **regression gates**, and a **failure-driven data flywheel**.

RLFactory implements that infrastructure. TRL is the primary training adapter; verl and OpenRLHF
are prepared behind the same framework-agnostic `TrainingBackend` protocol.

## Architecture

```
 configs/tasks/*.yaml
        |
        v
 TaskRegistry --> generators (25 templates) --> validation --> dedup --> contamination filter --> splits
        |
        v
 RolloutOrchestrator  (fixed worker pool over a bounded queue; retries; timeouts; cancel; resume)
        |  one episode per (task, sample)
        v
 Policy ------------> Environment (+ Sandbox | tools) -----> Verifier -----> RewardFunction
 mock | scripted |    arithmetic, structured_output,         exact, numeric,   binary, weighted,
 vLLM | SGLang |      python, transformation, tool_use       JSON schema,      format, length,
 OpenAI-compatible |                                         unit tests, ...   tool cost, composite
 Transformers                                                       |
        +-----------------------------------------------------------+
        v
 TrajectoryStore  (Parquet shards + DuckDB queries; lineage on every row; retention policies)
        |
        +--------------------+------------------------+
        v                    v                        v
 SFTDatasetBuilder   PreferenceDatasetBuilder   RLPromptDatasetBuilder (GRPO/RLVR)
        +-------- TRL / verl / OpenRLHF format adapters --------+
        v
 TrainingBackend:  mock | TRL | verl (prepared) | OpenRLHF (prepared)
        |
        v
 Evaluator (same suite, verifiers, metrics) --> regression gates --> promotion state
        |
        v
 failure analysis --> task mutation --> flywheel iteration --> report.md + figures
```

The code follows a clean architecture: the CLI calls application services (`rollouts`, `datasets`,
`evaluation`, `experiments`, `flywheel`), which operate on framework-free domain records
(`rlfactory.domain`) through protocols; adapters bind those protocols to subprocess/Docker,
HTTP model servers, Parquet/DuckDB, TRL and SSH. No TRL, Transformers or DuckDB type appears in the
domain layer. Details: [docs/architecture/overview.md](docs/architecture/overview.md).

| Package | Responsibility |
|---|---|
| `domain` | Tasks, trajectories, verification/reward results, training runs, manifests (pydantic, frozen) |
| `tasks` | Template generators, validation, mutation, difficulty signals, splits, contamination, sampling, registry |
| `environments`, `sandbox` | Gym-style environments, deterministic tools, subprocess/Docker/mock sandboxes |
| `verifiers`, `rewards` | Config-driven verifier and reward registries, reward-hacking and calibration analysis |
| `policies`, `rollouts` | Mock/scripted/HTTP/Transformers policies; episode worker and orchestrator |
| `trajectories` | Parquet+DuckDB, JSONL and in-memory stores |
| `datasets` | SFT, preference, GRPO and reward-model records, validation, framework adapters |
| `training` | Backend protocol, mock/TRL/verl/OpenRLHF backends, GPU memory estimation, reward bridges |
| `evaluation`, `flywheel` | Metrics, failure analysis, gates, promotion; failure-driven data loop |
| `experiments`, `reporting` | Stage DAG with caching and resume; Markdown/SVG reports |
| `remote`, `runtime`, `telemetry`, `cli` | SSH GPU execution, doctor/disk/HF cache, logs/metrics, Typer CLI |

## Tasks

Every task is a pure function of a typed template and its parameters, so generation, mutation and
re-validation are deterministic and ground truth is always recomputed, never edited.

| Domain | Templates | Default verifier |
|---|---|---|
| arithmetic | `integer_chain`, `fraction_sum`, `warehouse_word_problem`, `percentage` | `numeric` |
| structured_output | `filter_sort`, `aggregate`, `logic_constraints`, `extraction` | `json_schema` |
| python | `sum_of_evens`, `count_vowels`, `running_max`, `dedupe_preserve_order`, `second_largest`, `is_palindrome`, `rotate_left`, `char_frequency` | `unit_test` |
| transformation | `reverse_words`, `sort_lines`, `dedupe_lines`, `caesar_shift`, `snake_case`, `uppercase_vowels` | `exact_match` |
| tool_use | `inventory_value`, `order_total`, `currency_conversion` | `tool_outcome` |

The synthesis pipeline is `generate -> validate -> difficulty -> deduplicate -> decontaminate`.
A task is accepted only if its reference solution, played through the real environment (including
the sandbox for Python), passes its verifier **and** at least one known-wrong answer is rejected.
Mutations (`numbers`, `constraints`, `variable_names`, `input_length`, `format`, `edge_case`)
perturb template parameters and rebuild the task. Difficulty levels are coarse heuristics from
observable signals (operation counts, input size, program complexity, tool steps), not measures
of cognitive difficulty. Suites live in [`configs/tasks/`](configs/tasks/).

## Environments

Five environments implement `reset(task) -> Observation`, `step(action) -> StepResult`, `close()`:
four single-turn (`arithmetic`, `structured_output`, `python`, `transformation`) and one
multi-step (`tool_use`, with `calculator`, `kv_lookup` and `json_transform` tools). The Python
environment runs generated functions against hidden tests in a sandbox. There is deliberately no
shell, network or filesystem tool. See [docs/environments/README.md](docs/environments/README.md).

## Verifiers

`exact_match`, `numeric` (exact rational comparison with optional tolerance), `json_schema` (schema
plus expected value or constraints), `unit_test`, `regex`, `tool_outcome`, `composite` and an
optional `llm_judge`. Results are bounded (`score` in `[0, 1]`), carry a `reason_code`, and the
config hash is part of each verifier's version, so lineage records exactly which configuration
scored a trajectory. Composite verifiers require correctness to carry at least half the weight by
default. Judges are never the primary signal; calibration utilities compare them against
deterministic or human labels. See [docs/architecture/verifiers.md](docs/architecture/verifiers.md).

## Rewards

Rewards are computed from a `VerificationResult`, never instead of it, and are stored as
components (`correctness`, `format`, `length`, `tool_cost`, `safety`). Functions: `BinaryReward`,
`WeightedVerifierReward`, `FormatReward`, `LengthPenalty` (DAPO-style soft over-length penalty),
`ToolCostPenalty` (calls beyond a known-good plan) and `CompositeReward`, which refuses
configurations whose non-correctness bonus exceeds half the correctness reward unless
`allow_overshaping` is set for an ablation. Reward analysis reports reward-vs-verified correlation
and AUC, plus heuristic reward-hacking flags. See [docs/methodology/rewards.md](docs/methodology/rewards.md).

## Rollouts

`RolloutOrchestrator` runs exactly `concurrency` worker coroutines over a queue of size
`2 x concurrency` (never one task per job), with per-generation and per-episode timeouts,
exponential-backoff retries for **infrastructure** failures only, cancellation that flushes
completed work, and resume by deterministic trajectory ID. Failures are classified as
`policy_timeout`, `policy_error`, `sandbox_timeout`, `sandbox_error`, `verifier_error`,
`invalid_output` or `infrastructure_error`; model failures are verified and rewarded like any other
answer, infrastructure failures are kept out of datasets. See
[docs/architecture/rollouts.md](docs/architecture/rollouts.md).

## Trajectory Store

`LocalParquetTrajectoryStore` writes zstd-compressed Parquet shards atomically, with flat summary
columns (task, environment, verifier, policy, checkpoint, seed, reward components, failure kind,
tokens, latency, Git SHA, ...) plus the full trajectory JSON. DuckDB queries always read the
**latest record per trajectory ID**, so retried or resumed work is never double-counted. Retention
policies (`all`, `successful_only`, `failed_only`, `sampled`) drop payloads but always keep the
summary row. A JSONL store and an in-memory store implement the same protocol. See
[docs/architecture/trajectory-store.md](docs/architecture/trajectory-store.md).

## Dataset Builders

| Builder | Output | Policies |
|---|---|---|
| `SFTDatasetBuilder` | prompt/completion | `all_successful`, `highest_reward`, `shortest_correct`, `diverse_correct` |
| `PreferenceDatasetBuilder` | prompt/chosen/rejected | `correct_vs_incorrect`, `reward_margin`, `short_correct_vs_verbose_incorrect` |
| `RLPromptDatasetBuilder` | prompt-only + serialised task | drops saturated (all-pass/all-fail) and multi-step tasks |
| `RewardModelDatasetBuilder` | implicit-prompt preference | verified-winner pairs only |

Pairs are only formed inside one rollout group (same task, same rendered prompt). Every builder
applies the contamination filter (evaluation task IDs, their mutants, matching prompts, held-out
families, benchmark IDs), and every dataset is validated (roles, empty completions, duplicate
prompts, `chosen == rejected`, non-finite rewards, leakage) before it is written with a dataset
card. Adapters emit the TRL 1.18 conversational formats verified in
[docs/references/post-training-stack.md](docs/references/post-training-stack.md).

## Training Backends

| Backend | Status | What it does |
|---|---|---|
| `mock` | complete | Simulates steps, loss/LR curves, checkpoints, injected failures and resume; trains nothing |
| `trl` | complete | Writes a self-contained run directory (`run_config.json`, data, `launch.sh` using `accelerate launch -m rlfactory.training.trl.entrypoint`); SFT, DPO, GRPO, reward model; LoRA/QLoRA; GRPO rewards bridged to RLFactory verifiers with one named function per component |
| `verl` | prepared | verl 0.9.1 parquet data, `compute_score` shim, Hydra command; runs only in verl's own GPU-host environment |
| `openrlhf` | prepared | OpenRLHF 0.11.2 prompt data, `reward_func` shim, dotted-flag `train_ppo_ray` command |

Config keys emitted by the TRL adapter were checked against the installed TRL 1.18.0 dataclasses;
an integration test repeats that check whenever the `trl` extra is installed. See
[docs/architecture/training.md](docs/architecture/training.md).

## Evaluation

`Evaluator` runs the same suite, prompts, verifiers and metrics for base, SFT, DPO and GRPO
policies: verified accuracy with Wilson 95% intervals, pass@1, unbiased pass@k (only for stochastic
sampling with enough samples), format compliance, tool success, sandbox and infrastructure failure
rates, reward statistics and reward-vs-correctness correlation, completion length, latency and
observed failure categories. Config-driven gates (`max_regression`, `no_regression`, `min_value`,
`max_value`) decide whether a checkpoint becomes `validated` or `rejected`. See
[docs/evaluation/README.md](docs/evaluation/README.md).

## Failure-Driven Data Flywheel

```
evaluate on dev split -> classify failures -> mutate failed tasks -> validate / dedup / decontaminate
     ^                                                                        |
     |                                                                        v
 gate vs previous <- evaluate on held-out suite <- train <- verified rollouts (rejection-sampled SFT)
```

Mutants of evaluation tasks can never enter training. The loop is backend-agnostic; in the CPU
pipeline it runs with the mock policy and mock trainer, so its numbers are simulated. See
[docs/methodology/failure-flywheel.md](docs/methodology/failure-flywheel.md).

## Cloud GPU Experiments

RLFactory never rents, starts or pays for machines. Training runs on a GPU host you already
control, over SSH with your own agent/config:

```
rlfactory train configs/training/trl-sft-qwen3-0.6b.yaml --dry-run      # prepare + memory estimate
rlfactory remote check configs/remote/gpu-host.yaml                      # nvidia-smi over SSH
rlfactory remote launch results/training/trl-sft-qwen3-0.6b --host configs/remote/gpu-host.yaml
rlfactory remote status results/training/trl-sft-qwen3-0.6b --host configs/remote/gpu-host.yaml
rlfactory remote fetch  results/training/trl-sft-qwen3-0.6b --host configs/remote/gpu-host.yaml
```

`remote launch` asks for confirmation, runs a preflight (GPU visible, revision pinned, memory
estimate when available), syncs code and the run directory, prepares the environment, runs a
two-step smoke run (TRL) and only then launches. Only small result files are fetched; weights stay
on the host. Example GPU configs use `Qwen/Qwen3-0.6B` with `revision: null`: pin a commit SHA and
re-check the licence first. See [docs/training/gpu.md](docs/training/gpu.md).

## Jenkins

CI is Jenkins only (`Jenkinsfile`); the repository contains no GitHub Actions workflows. The
pipeline is CPU-only and offline after dependency installation: formatting, Ruff, strict mypy,
unit/property/integration/sandbox/verifier/store tests, mock rollout and post-training end-to-end
runs, security scan, package build and Docker build, with `disableConcurrentBuilds()`. See
[docs/ci/jenkins.md](docs/ci/jenkins.md).

## Quickstart

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/). No GPU, no network access after
installation, no API keys.

```bash
uv sync                                                    # core + dev tools
uv run rlfactory doctor                                    # environment report (no GPU is normal)
uv run rlfactory tasks list
uv run rlfactory environments check python                 # reference passes, wrong answers fail
uv run rlfactory experiment plan configs/experiments/cpu-e2e.yaml
uv run rlfactory experiment run  configs/experiments/cpu-e2e.yaml --run-id cpu-e2e-demo
uv run rlfactory experiment resume cpu-e2e-demo            # completed stages are served from cache
uv run rlfactory trajectories inspect cpu-e2e-demo
uv run rlfactory trajectories query cpu-e2e-demo --domain python --failed --limit 10
```

The run directory `results/cpu-e2e-demo/` then holds `manifest.json`, `config.yaml`,
`rollout_summary.json`, `training_summary.json`, `evaluation.json`, `failures.parquet`, `plots/`
and `report.md`. Step-by-step commands (`rollout run`, `dataset build-sft`, `train`, `evaluate`)
use the job configs in [`configs/`](configs/README.md). Also available: `tasks sample` (curriculum
sampling), `models plan` (download planning; never downloads), `evaluate --dry-run` and
`external lm-eval` (optional harness, never part of CI). `rlfactory --help` lists every command.

## Security

Generated code never runs in the RLFactory process. The subprocess sandbox is **not** a security
boundary; the Docker sandbox (no network, non-root, read-only root filesystem, dropped
capabilities, cgroup limits) is stronger but still shares the host kernel. Models load with
`trust_remote_code=False` and safetensors by default, YAML is parsed with `safe_load` only, logs
redact token-like strings, and configs reference secrets by environment-variable name only.
See [SECURITY.md](SECURITY.md) and [docs/security/sandbox.md](docs/security/sandbox.md).

## Results

**Real GPU post-training experiments pending.** No claim is made that SFT, DPO, GRPO or the
flywheel improves any model.

What has been measured so far concerns the infrastructure only:

| What | Measurement | Nature |
|---|---|---|
| CPU end-to-end pipeline (`ci-e2e`) | all 12 stages completed through the CLI in about 11 s on a 12-thread laptop CPU; a resume served every stage from cache | simulated (mock policy, mock trainer) |
| Cross-platform determinism (`ci-e2e`) | Windows 11 and the Linux CLI container produced identical trajectories (140: actions, observations, verification, rewards), byte-identical datasets and equal gate and flywheel results; only the recorded Git SHA differed | deterministic check (mock policy) |
| Rollout engine | about 95-100 trajectories/s with the mock policy, including sandboxed unit tests for Python tasks; injected 10% timeouts and 10% server errors were absorbed by 27 retries out of 100 jobs; resume skipped all completed IDs | infrastructure measurement |
| Task generation | every one of the 25 template families validated through the real environments and verifiers; 174 applicable mutations produced valid tasks | deterministic check |
| TRL 1.18 CPU smoke | SFT, DPO and GRPO each ran 2 optimizer steps on `HuggingFaceTB/SmolLM2-135M-Instruct`, offline, from datasets built by RLFactory; GRPO logged `rewards/rlfactory_correctness/mean` and `rewards/rlfactory_format/mean` | wiring check only; loss values are not results |

The committed example report [reports/cpu-e2e/report.md](reports/cpu-e2e/report.md) shows every
report section on a full simulated run. Its accuracy numbers come from a mock policy whose
"training" is a documented simulation knob, and it says so on the first line.

The comparison table below is filled only with measured real-model results:

| Model | Method | Verified Acc. | Python Pass | Format | Reward | Train Time | VRAM |
|---|---|---|---|---|---|---|---|
| pending | - | - | - | - | - | - | - |

## Limitations

- No real model has been trained or evaluated yet; all model-behaviour numbers are simulated.
- The subprocess sandbox is not a security boundary; use Docker for untrusted models, and treat
  even Docker as defence in depth rather than a VM boundary.
- verl and OpenRLHF backends prepare run directories but are not launched or tested end to end.
- Preference pairs and GRPO prompts exclude multi-step (tool-use) episodes in v0.1.
- Task families are small and synthetic; they test verifiable skills, not general capability.
- Difficulty levels and reward-hacking detectors are heuristics.
- The GPU memory estimator is a planning heuristic and needs the model's `config.json` locally
  (or a built-in shape); the smoke step is the real check.
- Mock-policy token counts are whitespace approximations.

## Roadmap

1. First real GPU study on a small open model: base vs SFT vs DPO vs GRPO on the same tasks,
   verifier, lineage and evaluation.
2. Failure-driven flywheel on a real model across several iterations.
3. RLVR reward-shaping study (binary vs correctness+format vs deliberately over-shaped) and the
   verifier-quality study (strict vs robust vs composite), configs already in `configs/`.
4. Launch and test the verl and OpenRLHF backends on a GPU host.
5. Multi-turn RL for tool use (e.g. through TRL's `environment_factory`).
6. Object-store trajectory backend and a live Prometheus scrape endpoint (rollout metrics are
   currently written per run as `metrics.prom` in the text exposition format).

## License

MIT, see [LICENSE](LICENSE).
