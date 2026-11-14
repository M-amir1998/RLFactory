# Architecture overview

RLFactory is organised as a clean architecture. Dependencies point inwards: adapters depend on
protocols and domain records, never the other way round.

```
 CLI (Typer)                                   rlfactory.cli
   |
 Application services                          tasks, rollouts, datasets, evaluation,
   |                                           flywheel, experiments, reporting
 Domain (framework-free pydantic records)      rlfactory.domain
   |
 Ports (typing.Protocol)                       Environment, Sandbox, Verifier, RewardFunction,
   |                                           Policy, TrajectoryStore, TrainingBackend,
   |                                           CommandRunner, JudgeClient
 Adapters                                      subprocess / Docker sandboxes, HTTP model servers,
                                               Transformers, Parquet + DuckDB, TRL, verl,
                                               OpenRLHF, SSH
```

`rlfactory.domain` imports no TRL, Transformers, DuckDB, HTTP or process APIs. Heavy optional
stacks (torch, transformers, trl, peft, datasets) are imported lazily inside the functions that
need them, so the core runs on a CPU-only laptop without them.

## End-to-end data flow

```
TaskSuiteConfig (YAML) --TaskRegistry--> Task[]
   --validate/dedup/decontaminate/split--> train | validation | evaluation tasks
train tasks --plan_jobs--> RolloutJob[] (deterministic trajectory IDs)
   --RolloutOrchestrator/RolloutWorker--> Trajectory[]  (Policy -> Environment -> Verifier -> Reward)
   --TrajectoryStore--> Parquet shards (+ DuckDB queries)
   --dataset builders--> SFT | preference | prompt-only RL records --adapters--> TRL/verl/OpenRLHF files
   --TrainingBackend.prepare/launch/inspect--> checkpoint (+ metadata)
evaluation tasks + base/candidate policies --Evaluator--> EvaluationResult x2
   --evaluate_gates--> GateReport --PromotionRegistry--> validated | rejected
dev failures --TaskMutator + SynthesisPipeline--> new tasks --> flywheel iteration
run directory --generate_report--> report.md + plots/*.svg
```

## Principles enforced in code

| Principle | Where |
|---|---|
| Verification is separate from reward | `verifiers/` produce `VerificationResult`; `rewards/` consume it |
| Infrastructure failures never become training data | `TrajectoryStatus.FAILED` trajectories are excluded by every builder |
| Idempotent work | trajectory IDs hash task, policy, checkpoint, generation config, seed, sample, verifier, reward |
| Lineage on every trajectory | `Lineage`: task version + fingerprint, environment/verifier/reward/prompt-template versions, policy, model, checkpoint, generation config, seed, code version |
| One prompt format everywhere | `tasks.prompts.PromptBuilder` renders rollout, evaluation and training prompts |
| No train/eval leakage | `tasks.contamination.ContaminationFilter` in synthesis, dataset builders and the flywheel |
| Bounded concurrency | fixed worker pool over a bounded `asyncio.Queue` |
| Conservative caching | stage fingerprints include config slice, upstream fingerprints and a hash of the package source |
| Honest simulation | mock policy and mock trainer set `simulated=True`; reports carry a banner and per-row labels |

## Run directory

Every experiment run writes one self-describing directory:

```
results/<run-id>/
  manifest.json            run ID, Git SHA, version, suites, component versions, policy, hardware, seed
  config.yaml              the exact experiment config
  stages/<stage>.json      status, fingerprint, timing, resources, outputs, errors
  tasks/                   train/validation/evaluation JSONL, contamination filter, summary
  rollouts/                Parquet trajectory shards              (git-ignored)
  rollout_summary.json
  datasets/{sft,dpo,grpo}/ records.jsonl, train.jsonl (framework format), dataset_card.json
  training/<train-run-id>/ run_config.json, launch.sh, metrics.jsonl, status.json, checkpoints
  training_summary.json
  evaluation/              base.json, candidate.json, per-label trajectories and failures.parquet
  evaluation.json          baseline + candidate + gate report + promotion decision
  gate_report.json
  failures.parquet         observed failure categories of the training rollouts
  reward_analysis.json     reward calibration and reward-hacking flags
  flywheel.json
  plots/*.svg
  report.md
```

`results/` is git-ignored as a whole. Small, reviewed example artefacts live under `reports/`.

## Extension points

| To add | Implement / register |
|---|---|
| a task family | `TaskTemplate` subclass + entry in `tasks/generators/catalog.py` |
| an environment | `Environment` + `EnvironmentSpec` in `environments/registry.py` |
| a verifier | `BaseVerifier` subclass + a spec type in `verifiers/registry.py` |
| a reward | `BaseReward` subclass + a spec type in `rewards/registry.py` |
| a model server | a `Policy` (HTTP subclasses only change body fields and health paths) |
| a storage backend | the `TrajectoryStore` protocol |
| a training framework | the `TrainingBackend` protocol (`prepare`, `launch`, `inspect`) |

Further reading: [environments](environments.md), [rollouts](rollouts.md),
[verifiers](verifiers.md), [trajectory store](trajectory-store.md), [training](training.md),
[evaluation](evaluation.md), [remote execution](remote-execution.md).
