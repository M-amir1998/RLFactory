# Post-training stack reference

*Verified 2026-11-14. Re-verify before every real GPU run; these projects move fast.*

RLFactory's adapters target **current** framework APIs, not remembered ones. This page records
what was verified, how, and what the adapters do about it. Two kinds of evidence were used:

1. **Primary sources**: PyPI JSON metadata, official documentation, and framework source at the
   released tag (URLs below).
2. **Local introspection**: the `trl` extra was installed into an isolated venv (`.venv-trl`) and
   `dataclasses.fields(...)` / `inspect.signature(...)` were read from the installed packages.
   The test `tests/integration/test_trl_compat.py` re-checks every key the TRL adapter emits against
   the installed config dataclasses, so drift is caught automatically whenever the `trl` extra is
   present.

## Versions observed

| Package | Version | Notes |
|---|---|---|
| trl | 1.18.0 (2026-11-13) | Python >= 3.11; transformers >= 4.56.2, accelerate >= 1.11, datasets >= 4.7, torch >= 2.6 |
| transformers | 5.23.0 | `dtype=` replaces `torch_dtype=` (deprecated, still accepted with a warning) |
| peft | 0.21.3 | |
| accelerate | 1.15.0 | `accelerate launch -m/--module` supported |
| datasets | 5.2.0 | |
| torch | 2.15.0 | Windows wheels on PyPI are CPU-only |
| vllm | 0.33.0 | TRL 1.18 supports vLLM 0.23.0 - 0.33.x |
| verl | 0.9.1 | requires Python < 3.13 |
| openrlhf | 0.11.2 | |
| sglang | 0.5.23 | |

verl, OpenRLHF, vLLM and SGLang are **not** RLFactory extras: they pin mutually incompatible
torch / vLLM / flash-attn versions (flash-attn ships no wheels on PyPI, which also breaks universal
locking). They run in isolated GPU-host environments; see `scripts/gpu/` and ADR-003.

## Hugging Face TRL 1.18

**Namespace.** `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `RewardTrainer`, `KTOTrainer`,
`RLOOTrainer`, `DistillationTrainer` and `AsyncGRPOTrainer` (stable since 1.18) are exported from
the top-level `trl` package. CPO, ORPO, GKD and Online DPO remain under `trl.experimental`.

**API changes that matter for adapters** (all confirmed on the installed 1.18.0):

| Change | Since | Adapter consequence |
|---|---|---|
| `tokenizer=` -> `processing_class=` | v0.12 | entrypoint passes `processing_class` |
| SFT `max_seq_length` -> `max_length` | v0.16 | emit `max_length` |
| GRPO `max_prompt_length` removed (long prompts are skipped, not truncated) | deprecated v0.26, gone in 1.x | never emitted; prompts are length-filtered when the dataset is built |
| DPO `max_prompt_length` / `max_completion_length` removed | <= v1.0 | only `max_length` + `truncation_mode` |
| GRPO/RLOO `vllm_mode` default `"server"` -> `"colocate"` | v1.0 | `vllm_mode` is always emitted explicitly |
| `warmup_ratio` removed from `TrainingArguments`; `warmup_steps` is a float | transformers v5 | the adapter maps `warmup_ratio` -> `warmup_steps` |
| `trl vllm-serve` deprecated (removed in v2) | 1.x | server-mode scripts call `vllm serve` directly |
| `use_liger_kernel` deprecated for DPO/KTO/GRPO/RLOO | v1.15 | not emitted |

**Model loading.** When `model` is a string, TRL loads it in **float32** unless
`model_init_kwargs={"dtype": ...}` is set. The adapter always emits `model_init_kwargs` with
`dtype`, the pinned `revision`, and `trust_remote_code=False` (configs also expose a top-level
`trust_remote_code` field, default `False`).

**Shared defaults** (from `_BaseConfig(TrainingArguments)`): `logging_steps=10`,
`gradient_checkpointing=True`, `bf16` resolves to `True` unless `fp16` is set, `report_to="none"`,
`optim="adamw_torch_fused"`.

### Dataset formats

Conversational = lists of `{"role", "content"}` dicts; standard = plain strings.

| Type | Columns | Used by |
|---|---|---|
| Language modeling | `text` or `messages` | SFT |
| Prompt-only | `prompt` | GRPO, RLOO, AsyncGRPO |
| Prompt-completion | `prompt`, `completion` | SFT (completion-only loss by default) |
| Preference, explicit prompt | `prompt`, `chosen`, `rejected` | DPO (recommended), Reward |
| Preference, implicit prompt | `chosen`, `rejected` | Reward (recommended by the format table) |
| Unpaired preference | `prompt`, `completion`, `label` | KTO |

RLFactory emits **conversational prompt-completion** for SFT, **conversational explicit-prompt
preference** for DPO, **conversational prompt-only + extra columns** for GRPO, and
**conversational implicit-prompt preference** for the reward trainer.

### Config fields used by the adapter (defaults read from the installed dataclasses)

* **SFTConfig** (135 fields): `max_length=1024`, `packing=False`, `completion_only_loss=None`
  (completion-only for prompt-completion data), `assistant_only_loss=False` (needs
  `{% generation %}` chat-template markers), `learning_rate=2e-5`, `loss_type="nll"`.
* **DPOConfig** (139 fields): `beta=0.1`, `loss_type=["sigmoid"]` (a list; several losses can be
  mixed with `loss_weights`), `max_length=1024`, `truncation_mode="keep_start"`,
  `precompute_ref_log_probs=False`, `learning_rate=1e-6`. Tokenizer padding side must be left.
* **GRPOConfig** (191 fields): `num_generations=8` (must divide the effective batch size),
  `max_completion_length=512`, `temperature=1.0`, `top_p=1.0`, `top_k=0`, `beta=0.0` (no reference
  model loaded), `loss_type="dapo"` (also `grpo`, `dr_grpo`, `bnpo`, `cispo`, `sapo`, `luspo`,
  `vespo`), `scale_rewards="group"` (string: `group` / `batch` / `none`), `reward_weights=None`,
  `multi_objective_aggregation="sum_then_normalize"`, `num_iterations=1`, `epsilon=0.2`,
  `mask_truncated_completions=False`, `use_vllm=False`, `vllm_mode="colocate"`,
  `vllm_server_base_url=None`, `vllm_gpu_memory_utilization=0.3`, `log_completions=False`,
  `remove_unused_columns=False`. With vLLM importance-sampling correction on (default), `top_p<1`
  or `min_p>0` requires `top_k>0` (v1.17+).
* **RewardConfig** (125 fields): `max_length=1024` (longer pairs are filtered, not truncated),
  `center_rewards_coefficient=None` (docs suggest 0.01), `learning_rate=1e-4`. LoRA on a causal-LM
  base needs `modules_to_save=["score"]`.

### GRPO reward-function contract (read from `grpo_trainer.py` at 1.18.0)

```python
def reward_fn(prompts, completions, completion_ids, **kwargs) -> list[float | None]: ...
```

* `kwargs` contains every dataset column except `prompt`/`completion`/`completion_ids`, plus
  `trainer_state`, `log_extra(column, values)`, `log_metric(name, value)`, and `environments` when
  `environment_factory` is used.
* Conversational completions are message lists (`completion[0]["content"]`).
* Returning `None` drops that function for that sample (NaN, combined with `nansum` and
  `reward_weights`). Async reward functions are supported.
* Metrics are logged as `rewards/{function_name}/mean|std`, so RLFactory gives every bridged
  reward function a unique `__name__` (`rlfactory_correctness`, `rlfactory_format`, ...).
  This is how reward **components stay visible** during GRPO.
* `GRPOTrainer` also accepts `tools=`, `rollout_func=` and `environment_factory=` (objects with
  `reset`), an integration point for RLFactory's multi-step environments in a later release.

### vLLM integration

* Colocate mode (default): vLLM shares the training GPUs; tune `vllm_gpu_memory_utilization`.
* Server mode: run the server on **separate GPUs**:
  `CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve <model> --weight-transfer-config '{"backend": "nccl"}' --logprobs-mode processed_logprobs --max-logprobs -1`
  and set `vllm_mode="server"` plus `vllm_server_base_url` in GRPOConfig.

### PEFT and QLoRA

All four trainers take `peft_config: PeftConfig | None` and
`quantization_config: BitsAndBytesConfig | None` ("combine with `peft_config` for QLoRA").
`LoraConfig` defaults: `r=8`, `lora_alpha=8`, `lora_dropout=0.0`, `target_modules=None`
(`"all-linear"` = all linear layers except the output head), `task_type=None` (use `"CAUSAL_LM"`).
Never target `lm_head`; use `modules_to_save`. QLoRA:
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
bnb_4bit_compute_dtype=torch.bfloat16)`.

## Transformers 5.x

* `from_pretrained(..., revision=..., use_safetensors=..., token=..., weights_only=True)`;
  `dtype="auto"` reads the dtype from `config.json`. `torch_dtype=` logs a deprecation warning.
* `trust_remote_code` defaults to `False`; pin `revision=<commit sha>` whenever remote code is
  allowed. `use_safetensors=True` makes loading fail rather than fall back to pickle weights.
* `apply_chat_template(conversation, tools=None, ..., add_generation_prompt=False, tokenize=True,
  return_dict=True, ...)`: v5 returns a `BatchEncoding` when tokenizing; use `tokenize=False` for a
  string.

## Accelerate 1.15

`accelerate launch --num_processes N --mixed_precision bf16 -m rlfactory.training.trl.entrypoint
--run-config run_config.json` is what the TRL backend writes into `launch.sh`.

## Small open models for the first GPU study

| Model | License | Gated | Notes |
|---|---|---|---|
| Qwen/Qwen3-0.6B | apache-2.0 | no | TRL ships a training chat template |
| Qwen/Qwen3-1.7B | apache-2.0 | no | |
| Qwen/Qwen2.5-0.5B-Instruct | apache-2.0 | no | |
| Qwen/Qwen2.5-1.5B-Instruct | apache-2.0 | no | |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | apache-2.0 | no | not in TRL's assistant-only template list |
| HuggingFaceTB/SmolLM3-3B | apache-2.0 | no | template already has `{% generation %}` markers |
| meta-llama/Llama-3.2-1B-Instruct | llama3.2 | yes | needs HF token + accepted license |
| google/gemma-3-1b-it | gemma | yes | |

Qwen3.5-0.8B/2B (apache-2.0) are multimodal `ConditionalGeneration` models; do not load them with
`AutoModelForCausalLM`. The example GPU configs default to **Qwen/Qwen3-0.6B** with
`revision: null`, and the TRL backend refuses remote launches until the revision is pinned to a
commit SHA (read it from `https://huggingface.co/api/models/<id>` -> `sha`). Re-check the license
on the model card before running.

## verl 0.9.1 (canonical repo: github.com/verl-project/verl)

Python `>=3.10,<3.13`; v0.9.1 manages dependencies with uv and pins transformers 5.9.0, so verl
**cannot share an environment with TRL 1.18 / transformers 5.23**. Run it from its own checkout or
container on the GPU host.

* **GRPO launch**: `python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...` (Hydra
  config `ppo_trainer`; the V1 trainer is the default since 0.9.0). Essential overrides:
  `data.train_files`, `data.val_files` (quoted lists), `data.train_batch_size`,
  `data.max_prompt_length`, `data.max_response_length`, `data.filter_overlong_prompts=True`,
  `actor_rollout_ref.model.path`, `actor_rollout_ref.rollout.name` (**required**: hf / vllm /
  sglang / trtllm), `actor_rollout_ref.rollout.n`, `actor_rollout_ref.actor.ppo_mini_batch_size`,
  `actor_rollout_ref.actor.use_kl_loss`, `kl_loss_coef`, `kl_loss_type=low_var_kl`,
  `trainer.n_gpus_per_node`, `trainer.nnodes`, `trainer.total_epochs`, `trainer.logger`,
  `trainer.project_name`, `trainer.experiment_name`.
* **RL parquet schema**: `data_source` (routes the reward function), `prompt` (chat messages),
  `ability`, `reward_model={"style": "rule", "ground_truth": ...}`, `extra_info` (dict).
* **Custom reward**: `reward.custom_reward_function.path` / `.name` (default `compute_score`) and
  `reward.reward_manager.name` (`naive` default; also `dapo`, `remote`, `rate_limited`). The
  top-level legacy `custom_reward_function.*` keys only survive as null defaults; the config
  explanation page on readthedocs still shows the old layout.
  Signature: `compute_score(data_source, solution_str, ground_truth, extra_info=None)`, called with
  keyword arguments; `solution_str` is the decoded response only. Return a float or a dict with
  `"score"` (other keys are logged). `async def` is accepted.
* **LoRA (FSDP)**: `actor_rollout_ref.model.lora_rank` (>0 enables), `lora_alpha`,
  `target_modules=all-linear`, `actor_rollout_ref.rollout.load_format=safetensors`.
* **SFT**: `torchrun --standalone --nnodes=1 --nproc_per_node=N -m verl.trainer.sft_trainer
  data.train_files=... data.messages_key=messages engine=fsdp model.path=...` (parquet with a
  `messages` column; `fsdp_sft_trainer` no longer exists on main).
* **Multi-turn / tools**: async server-based agent loops (`single_turn_agent`, `tool_agent`,
  selected per sample by an `agent_name` column), under `verl/experimental/`.

## OpenRLHF 0.11.2

Flags are now **dotted and hierarchical, with no aliases for the old flat names** (`--pretrain`,
`--save_path`, `--advantage_estimator`, `--remote_rm_url` are not registered). The README is partly
stale; the CLI source and example scripts at the tag are authoritative.

* Modules: `openrlhf.cli.train_sft`, `train_dpo`, `train_rm` (launched with
  `deepspeed --module`), `train_ppo_ray` (`ray job submit ... -- python3 -m
  openrlhf.cli.train_ppo_ray`), `serve_rm`, `lora_combiner`.
* GRPO-style run: `--algo.advantage.estimator group_norm` (also `reinforce`,
  `reinforce_baseline`, `rloo`, `dr_grpo`, ...), `--rollout.n_samples_per_prompt`,
  `--algo.kl.init_coef`, `--algo.kl.use_loss`, `--data.prompt_dataset`, `--data.input_key`,
  `--data.label_key`, `--data.apply_chat_template`, `--data.max_len`, `--rollout.max_new_tokens`,
  `--actor.model_name_or_path`, `--ckpt.output_dir`, `--vllm.num_engines`,
  `--vllm.tensor_parallel_size`, `--vllm.gpu_memory_utilization`, hybrid engine via
  `--train.colocate_all --vllm.enable_sleep --ds.enable_sleep`, LoRA via `--ds.lora.rank`,
  `--ds.lora.alpha`, `--ds.lora.target_modules`.
* **Custom reward**: `--reward.remote_url /path/reward_func.py`; the module attribute
  `reward_func(queries, prompts, labels)` receives lists of strings (queries = prompt + response)
  and returns `{"rewards": tensor, "scores": tensor, "extra_logs": {...}}`. Whether plain floats are
  accepted is unverified, so RLFactory's shim returns float tensors.
* Agent mode: `--train.agent_func_path agent_func.py` with `AgentInstance.reset/step`.

## vLLM 0.33.0

* `vllm serve <model> --host --port 8000 --dtype auto --max-model-len --gpu-memory-utilization 0.92
  --served-model-name --api-key --seed --enable-lora --lora-modules --generation-config`.
  By default the server applies the model's `generation_config.json`; pass
  `--generation-config vllm` (or always send explicit sampling params, as RLFactory does).
* Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`,
  `/version`, `/tokenize`, `/detokenize`.
* Chat request extras (top-level JSON): `seed`, `logprobs`, `top_logprobs`, `n`,
  `max_completion_tokens` (`max_tokens` deprecated), `top_k`, `min_p`, `repetition_penalty`,
  `structured_outputs`, `response_format`, `return_token_ids`, `chat_template_kwargs`.
* Structured outputs: `"structured_outputs": {"json" | "regex" | "choice" | "grammar": ...}` or
  `response_format={"type": "json_schema", ...}`. **`guided_*` fields were removed in v0.12.0.**
* Metrics worth scraping: `vllm:num_requests_running|waiting`, `vllm:kv_cache_usage_perc`,
  `vllm:prompt_tokens`, `vllm:generation_tokens`, `vllm:e2e_request_latency_seconds`,
  `vllm:time_to_first_token_seconds`.

## SGLang 0.5.23 (docs moved to docs.sglang.io)

* `sglang serve <model> --host 0.0.0.0 --port 30000` (or
  `python -m sglang.launch_server --model-path ... --tp N`); `--mem-fraction-static`,
  `--context-length`, `--api-key`, `--random-seed`, `--enable-metrics` (off by default).
* Endpoints: OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`; native
  `/generate`; `/health`, `/health_generate` (generates one token), `/ready`; **`/model_info`**
  (`/get_model_info` is deprecated), `/server_info`.
* Chat extras: `seed`, `top_k`, `min_p`, `repetition_penalty`, `min_tokens`, `regex`, `ebnf`,
  `response_format` (json_schema), `separate_reasoning`, `chat_template_kwargs`.
  Sampling defaults come from the model's generation config unless overridden.

## What RLFactory does with these facts

* **HTTP policies** (`OpenAICompatiblePolicy`, `VLLMPolicy`, `SGLangPolicy`) always send explicit
  `temperature`, `top_p`, `seed` and a max-token budget; vLLM gets `max_completion_tokens`, engine
  extras go at the top level of the JSON body, and `guided_*` is never emitted. Readiness checks
  use `/health` (vLLM) and `/health` + `/model_info` (SGLang).
* **TRL backend** checks every emitted key against the installed config dataclasses (test-enforced
  when the `trl` extra is installed) and always sets `model_init_kwargs.dtype`, `revision` and
  `trust_remote_code`.
* **verl backend** writes the five-column parquet, a `compute_score` wrapper returning
  `{"score": ...}`, and a command using `reward.custom_reward_function.*` and an explicit
  `actor_rollout_ref.rollout.name`.
* **OpenRLHF backend** emits dotted flags only, maps GRPO to `group_norm`, and writes a
  `reward_func(queries, prompts, labels)` shim returning a dict of float tensors.
* Docs and READMEs lag behind source for both verl and OpenRLHF: dry-run every generated command
  on the target environment (`--help`, Hydra `--cfg job`) before a real run.

### Sources (verl / OpenRLHF / vLLM / SGLang)

* verl: https://pypi.org/pypi/verl/json, https://github.com/volcengine/verl/releases,
  https://raw.githubusercontent.com/verl-project/verl/main/examples/grpo_trainer/run_qwen3_8b_fsdp.sh,
  https://raw.githubusercontent.com/verl-project/verl/main/verl/trainer/config/ (ppo_trainer.yaml,
  reward/reward.yaml, rollout/rollout.yaml, actor/actor.yaml),
  https://verl.readthedocs.io/en/latest/preparation/prepare_data.html,
  https://verl.readthedocs.io/en/latest/preparation/reward_function.html,
  https://verl.readthedocs.io/en/latest/advance/ppo_lora.html,
  https://verl.readthedocs.io/en/latest/start/agentic_rl.html
* OpenRLHF: https://pypi.org/pypi/openrlhf/json, https://github.com/OpenRLHF/OpenRLHF,
  https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/v0.11.2/openrlhf/cli/train_ppo_ray.py,
  https://raw.githubusercontent.com/OpenRLHF/OpenRLHF/v0.11.2/openrlhf/utils/agent.py,
  https://github.com/OpenRLHF/OpenRLHF/tree/v0.11.2/examples/scripts
* vLLM: https://pypi.org/pypi/vllm/0.33.0/json, https://docs.vllm.ai/en/stable/cli/serve/,
  https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/,
  https://docs.vllm.ai/en/stable/features/structured_outputs/,
  https://docs.vllm.ai/en/stable/usage/metrics/
* SGLang: https://pypi.org/pypi/sglang/0.5.23/json, https://docs.sglang.io/docs/get-started/quickstart.md,
  https://docs.sglang.io/docs/advanced_features/server_arguments,
  https://raw.githubusercontent.com/sgl-project/sglang/v0.5.23/python/sglang/srt/entrypoints/http_server.py,
  https://docs.sglang.io/docs/basic_usage/sampling_params

## Sources (TRL / Hugging Face stack)

* https://pypi.org/pypi/trl/json, https://github.com/huggingface/trl/releases,
  https://raw.githubusercontent.com/huggingface/trl/v1.18.0/MIGRATION.md
* https://huggingface.co/docs/trl/dataset_formats
* https://huggingface.co/docs/trl/sft_trainer, https://huggingface.co/docs/trl/dpo_trainer,
  https://huggingface.co/docs/trl/grpo_trainer, https://huggingface.co/docs/trl/reward_trainer
* https://raw.githubusercontent.com/huggingface/trl/v1.18.0/trl/trainer/grpo_trainer.py
* https://huggingface.co/docs/trl/vllm_integration, https://huggingface.co/docs/trl/clis,
  https://huggingface.co/docs/trl/chat_templates
* https://huggingface.co/docs/transformers/main_classes/model,
  https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md
* https://huggingface.co/docs/peft/package_reference/lora,
  https://huggingface.co/docs/peft/developer_guides/quantization
* https://huggingface.co/docs/accelerate/package_reference/cli
* https://huggingface.co/api/models/{repo_id} for each model listed
