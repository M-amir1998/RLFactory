# Training

RLFactory's core never depends on a training framework. Frameworks sit behind one protocol:

```python
class TrainingBackend(Protocol):
    name: str

    def prepare(self, run: TrainingRun) -> PreparedTrainingRun: ...
    def launch(self, run: PreparedTrainingRun) -> TrainingHandle: ...
    def inspect(self, handle: TrainingHandle) -> TrainingStatus: ...
```

`TrainingRun` is canonical: `method` (`sft`, `dpo`, `grpo`, `reward_model`), `model`
(`ModelReference`: name, revision, dtype, `trust_remote_code=False`), `dataset`
(`DatasetReference` with schema name and fingerprint), `config` (`TrainingConfig`: optimiser and
schedule settings, `LoRASettings`, typed method parameters, and an explicit `backend_overrides`
escape hatch), `seed` and `launch_mode` (`local` or `remote`). The method and its parameters are
validated to agree.

## Run-directory conventions

`prepare()` produces a self-contained directory that can be synced to a GPU host as-is. While
running, every backend writes the same files, so status inspection is uniform:

| File | Content |
|---|---|
| `run_config.json` | everything the launcher needs |
| `launch.sh` | POSIX launcher; tees output to `train.log`, records `exit_status.json` |
| `prepared.json` | the `PreparedTrainingRun` (used by remote tooling and resumes) |
| `status.json` | `TrainingStatus`: state, step, latest metrics, checkpoint events, error |
| `metrics.jsonl` | one `TrainingMetricsPoint` per logging step: loss, learning rate, tokens, wall time, peak VRAM, extra numeric logs |
| `training_summary.json` | `TrainingSummary` at the end (copied to the experiment run directory) |

`status_from_workdir()` falls back to `exit_status.json` for frameworks that cannot write
`status.json` themselves (verl, OpenRLHF).

## Backends

### Mock (complete)

Simulates training state, a decaying loss curve, a cosine schedule with warmup, checkpoint events
every `save_steps`, an optional injected failure (`simulate_failure_at_step`, fired once) and
resume from the last checkpoint. It trains **nothing**. So that evaluation and gates see a changed
policy, the final checkpoint carries a `simulated_policy_profile`: for each domain with `n`
training examples,

```
gain = max_gain * method_gain * min(1, log(1 + n) / log(1 + saturation_examples))
p_correct' = min(0.98, p_correct + gain * (1 - p_correct))
```

with `max_gain = 0.25`, `saturation_examples = 50` and `method_gain` 1.0 (SFT), 0.8 (DPO),
1.2 (GRPO), 0 (reward model). These constants are a simulation knob for exercising the control
plane, not a model of learning, and every artefact derived from them is labelled simulated.

### TRL (complete)

- `config_mapping.build_config_kwargs` maps the canonical run onto TRL 1.18 `SFTConfig`,
  `DPOConfig`, `GRPOConfig` or `RewardConfig` keyword arguments. Field names were verified against
  the installed dataclasses; keys removed in recent releases (`max_seq_length`,
  `max_prompt_length`, `tokenizer`, `warmup_ratio`, `use_liger_kernel`, `torch_dtype`) are rejected
  in `backend_overrides`. `warmup_ratio` is mapped to Transformers 5's float `warmup_steps`.
  `model_init_kwargs` always sets `dtype` (TRL otherwise loads string models in float32), the
  pinned `revision` and `trust_remote_code`.
- `prepare()` checks the dataset schema against the method, requires a pinned revision for remote
  launches, checks that `num_generations` divides the GRPO generation batch, copies the dataset
  into `data/`, and writes `launch.sh`:
  `accelerate launch --num_processes N --mixed_precision bf16 -m rlfactory.training.trl.entrypoint --run-config run_config.json`
  (`python -m ...` for the CPU smoke mode).
- `entrypoint.py` loads the JSONL dataset, builds `LoraConfig` (`task_type="CAUSAL_LM"`; for reward
  models `SEQ_CLS` with `modules_to_save=["score"]`) and, for QLoRA, `BitsAndBytesConfig`
  (NF4, double quantisation, bf16 compute), constructs the trainer from the model ID (TRL loads the
  model with `model_init_kwargs` and the tokenizer from the same pinned revision), streams metrics
  through a `TrainerCallback`, saves the final adapter or model, and writes
  `rlfactory_checkpoint.json` next to it plus a `TrainingSummary`.
- GRPO rewards (`training/trl/rewards.py`) delegate to `CompletionScorer`, which replays the
  completion through the task's real environment, verifier and reward. Each reward component is a
  separate function named `rlfactory_<component>` so TRL logs
  `rewards/rlfactory_correctness/mean`, `rewards/rlfactory_format/mean`, etc., and weights go to
  `reward_weights`. Multi-step tasks return `None` (TRL then ignores that sample for the function).
- `launch()` runs locally (blocking) only when TRL is importable; remote launches go through
  `rlfactory remote launch`.

The TRL path was exercised on CPU with `HuggingFaceTB/SmolLM2-135M-Instruct`: SFT, DPO and GRPO each
completed two optimiser steps from RLFactory-built datasets. That is a wiring check, not a result.

### verl (prepared interface)

GRPO only. `prepare()` writes `data/train.parquet` in verl's five-column schema (`data_source`,
`prompt`, `ability`, `reward_model.ground_truth` = serialised task, `extra_info`), a
`reward_fn.py` shim exposing `compute_score`, and a `launch.sh` with Hydra overrides using the
current `reward.custom_reward_function.*` keys and an explicit `actor_rollout_ref.rollout.name`.
`launch()` refuses to run locally: verl belongs in its own GPU-host environment.

### OpenRLHF (prepared interface)

GRPO only. `prepare()` writes `data/prompts.jsonl` (`input`, `label`), a `reward_func.py` shim
returning a dict of float tensors, and a `ray job submit ... python3 -m openrlhf.cli.train_ppo_ray`
command with the dotted flags of 0.11.2 (`--algo.advantage.estimator group_norm`,
`--reward.remote_url reward_func.py`, ...). `launch()` refuses to run locally.

For both prepared backends, dry-run the generated command on the target environment first
(`--cfg job` for verl's Hydra, `--help` for OpenRLHF): their READMEs lag their source.

## GPU memory estimation

`training/memory.py` estimates weights, trainable state (16 bytes per trainable parameter with
LoRA, 14 extra bytes per parameter for full fine-tuning), activations (with or without gradient
checkpointing), fp32 logits and an optional colocated vLLM fraction, plus overhead. Model shapes
come from a `config.json` (`ModelShape.from_hf_config`) or a small built-in table. It is a planning
heuristic; the remote smoke step is the real check.

## Checkpoints

`CheckpointMetadata` records the checkpoint reference (base model, adapter location, method,
training run, step, simulated flag), dataset name and fingerprint, Git SHA, training config,
framework and version, and metrics. Weights never enter Git.
