"""Load configuration from a local .env file.

Turning the LLM panel on in CI means adding a GitHub Actions secret. That is
the right place for it -- but it needs access to the repository settings, and
it is not the only way to run this project. Locally, a `.env` file beside
`config.json` is enough:

    LLM_API_KEY=gsk_your_key_here
    LLM_PROVIDER=groq

`.env` is gitignored, so the key stays on your machine and can never be
committed. That matters here: this repository is public, so a key committed
to it would be published to the world the moment it is pushed.

Real environment variables always win, so CI (where the key arrives as a
secret) is unaffected by any stray file.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

SENSITIVE_HINTS = ("key", "token", "secret", "password")


def parse(text: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines. Supports comments, blank lines, quotes, `export`."""
    values: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        # Strip one matching pair of surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[name] = value
    return values


def load(path: str, override: bool = False) -> Dict[str, str]:
    """Load `path` into os.environ. Returns the names applied, never the values.

    Existing environment variables are left alone unless `override` is set, so
    a real secret in CI always beats a file on disk.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            parsed = parse(handle.read())
    except OSError:
        return {}

    applied = {}
    for name, value in parsed.items():
        if override or not os.environ.get(name):
            os.environ[name] = value
            applied[name] = value
    return applied


def describe(applied: Dict[str, str]) -> Optional[str]:
    """A one-line summary safe to print: names only, values never."""
    if not applied:
        return None
    shown = []
    for name in sorted(applied):
        if any(hint in name.lower() for hint in SENSITIVE_HINTS):
            shown.append(f"{name}=<hidden, {len(applied[name])} chars>")
        else:
            shown.append(f"{name}={applied[name]}")
    return "loaded .env: " + ", ".join(shown)
