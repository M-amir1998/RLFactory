"""Optional external benchmark adapter (EleutherAI lm-evaluation-harness).

RLFactory only builds the command and parses the results file; the harness is never a core or CI
dependency. Verify flags against the installed harness (``lm_eval --help``) before relying on them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LMEvalHarnessConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: tuple[str, ...] = Field(default=("gsm8k",), min_length=1)
    model_backend: Literal["local-chat-completions", "local-completions", "hf"] = (
        "local-chat-completions"
    )
    model: str
    base_url: str | None = None
    """OpenAI-compatible server root, e.g. http://127.0.0.1:8000 (vLLM / SGLang)."""
    num_fewshot: int | None = None
    limit: int | None = Field(default=None, ge=1)
    apply_chat_template: bool = True
    num_concurrent: int = Field(default=4, ge=1)
    output_path: str = "lm_eval_results"


def build_lm_eval_command(config: LMEvalHarnessConfig) -> list[str]:
    if config.model_backend == "hf":
        model_args = f"pretrained={config.model}"
    else:
        if config.base_url is None:
            raise ValueError("server-backed lm-eval runs need base_url")
        endpoint = (
            "chat/completions"
            if config.model_backend == "local-chat-completions"
            else "completions"
        )
        model_args = (
            f"model={config.model},base_url={config.base_url.rstrip('/')}/v1/{endpoint},"
            f"num_concurrent={config.num_concurrent}"
        )
    command = [
        "lm_eval",
        "--model",
        config.model_backend,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(config.tasks),
        "--output_path",
        config.output_path,
    ]
    if config.apply_chat_template:
        command.append("--apply_chat_template")
    if config.num_fewshot is not None:
        command += ["--num_fewshot", str(config.num_fewshot)]
    if config.limit is not None:
        command += ["--limit", str(config.limit)]
    return command


def parse_lm_eval_results(path: Path) -> dict[str, dict[str, float]]:
    """Extract numeric metrics per task from an lm-eval ``results*.json`` file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    parsed: dict[str, dict[str, float]] = {}
    for task, metrics in data.get("results", {}).items():
        parsed[task] = {
            name: float(value)
            for name, value in metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    return parsed
