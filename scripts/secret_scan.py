"""Scan files that would be committed for credentials, private keys, personal paths and emails.

Usage: python scripts/secret_scan.py [--all]   (default: tracked + untracked-but-not-ignored files)
Exit code 1 when anything is found. A line containing ``secret-scan: allow`` is skipped; use it
only for deliberately fake test fixtures.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "results",
    "dist",
    ".uv-cache",
}
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".parquet",
    ".whl",
    ".gz",
    ".zip",
    ".pdf",
}

PATTERNS: dict[str, re.Pattern[str]] = {
    "huggingface token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "openai-style key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b"),
    "anthropic key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    "aws access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    "credential assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{16,}['\"]"
    ),
    "windows user path": re.compile(
        r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s'\"]+", re.IGNORECASE
    ),
    "posix home path": re.compile(
        r"(?<![\w.])/(?:Users|home)/(?!ubuntu\b|runner\b|rlfactory\b|<)[a-z][a-z0-9_\-]{1,30}/"
    ),
    "email address": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@(?!example\.(?:com|org)\b|users\.noreply\.github\.com\b)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
}


def candidate_files(scan_all: bool) -> list[Path]:
    if scan_all:
        return [
            p
            for p in ROOT.rglob("*")
            if p.is_file()
            and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)
            and not any(part.startswith(".venv") for part in p.relative_to(ROOT).parts)
        ]
    tracked = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [ROOT / name for name in tracked if (ROOT / name).is_file()]


def scan(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        if path.suffix.lower() in BINARY_SUFFIXES or path.name == "uv.lock":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if "secret-scan: allow" in line:
                continue
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {name}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="scan every file, not just commit candidates"
    )
    args = parser.parse_args()
    files = candidate_files(args.all)
    findings = scan(files)
    for finding in findings:
        print(f"FOUND {finding}")
    print(f"secret scan: {len(files)} files checked, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
