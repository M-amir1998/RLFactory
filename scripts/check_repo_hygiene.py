"""Repository hygiene gate.

Fails on: GitHub Actions workflows, model weights/checkpoints/credentials among commit candidates,
oversized files, tracked run outputs, and AI co-author/attribution trailers in commit messages.
With ``--dist DIR`` it also checks built artefacts contain no tests data, results or weights.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 512 * 1024
FORBIDDEN_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}
FORBIDDEN_NAMES = re.compile(
    r"(^|/)(\.env(\..+)?|id_rsa.*|id_ed25519.*|credentials\.json|\.hf_token)$"
)
ALLOWED_RESULTS = {"results/README.md", "results/.gitkeep"}
AI_TRAILER = re.compile(
    r"(?im)^\s*(co-authored-by|generated-by|ai-assisted-by)\s*:.*\b(claude|anthropic|openai|chatgpt|copilot|gpt)\b"
)


def git(*args: str) -> list[str]:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.splitlines() if completed.returncode == 0 else []


def check_tree() -> list[str]:
    problems: list[str] = []
    workflows = ROOT / ".github" / "workflows"
    if workflows.exists() and any(workflows.iterdir()):
        problems.append(".github/workflows exists: CI runs on Jenkins only (ADR-007)")
    for name in git("ls-files", "--cached", "--others", "--exclude-standard"):
        path = ROOT / name
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"{name}: weights/keys must never be committed")
        if FORBIDDEN_NAMES.search(name):
            problems.append(f"{name}: credential file")
        if name.startswith("results/") and name not in ALLOWED_RESULTS:
            problems.append(f"{name}: run outputs stay out of Git")
        if path.stat().st_size > MAX_FILE_BYTES and name != "uv.lock":
            problems.append(f"{name}: {path.stat().st_size} bytes exceeds {MAX_FILE_BYTES}")
    for message in "\n".join(git("log", "--format=%B", "-n", "200")).split("\n\n"):
        if AI_TRAILER.search(message):
            problems.append("commit history contains an AI co-author/attribution trailer")
            break
    return problems


def check_dist(directory: Path) -> list[str]:
    problems: list[str] = []
    for artefact in sorted(directory.glob("*")):
        if artefact.suffix == ".whl":
            names = zipfile.ZipFile(artefact).namelist()
        elif artefact.name.endswith(".tar.gz"):
            with tarfile.open(artefact) as archive:
                names = archive.getnames()
        else:
            continue
        problems.extend(
            f"{artefact.name}: contains forbidden member {name}"
            for name in names
            if (
                Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
                or "/results/" in f"/{name}"
                or Path(name).name == ".env"
            )
        )
        if artefact.suffix == ".whl" and not any(n.startswith("rlfactory/") for n in names):
            problems.append(f"{artefact.name}: package code missing")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, help="also inspect built artefacts in this directory")
    args = parser.parse_args()
    problems = check_tree()
    if args.dist:
        problems += check_dist(args.dist)
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"repository hygiene: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
