"""Model configuration: an explicit spec, plus environment-provided defaults.

Agents pick their own model by argument -- see `llm.get_model` -- so that one
agent can run on a fast small model and another on a slower, stronger one. The
`MAS_LLM_*` variables only supply the fallback used when an argument is omitted::

    MAS_LLM_PROVIDER=nvidia
    MAS_LLM_MODEL=openai/gpt-oss-20b
    MAS_LLM_TEMPERATURE=0.0
    MAS_LLM_TIMEOUT=120
    MAS_LLM_MAX_TOKENS=
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "LLMConfigError",
    "ModelSpec",
    "env_defaults",
]

DEFAULT_PROVIDER = "nvidia"
# Deterministic by default: reward is binary per task and pass^k counts a task
# only when every trial passes, so sampling variance is a direct score loss.
DEFAULT_TEMPERATURE = 0.0
# Bounded wait. Some free-tier endpoints stall indefinitely; failing one call
# beats hanging a simulation and, with it, a whole benchmark sweep.
DEFAULT_TIMEOUT = 120.0

_PREFIX = "MAS_LLM_"


class LLMConfigError(RuntimeError):
    """Raised when model configuration is missing, malformed, or unknown."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A fully resolved model choice. `None` on a setting means "do not send it"."""

    provider: str
    model: str
    temperature: float | None = DEFAULT_TEMPERATURE
    timeout: float | None = DEFAULT_TIMEOUT
    max_tokens: int | None = None

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def _read_number(env: Mapping[str, str], key: str, cast, default):
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{key} must be a number, got {raw!r}") from exc


def env_defaults(env: Mapping[str, str] | None = None) -> ModelSpec:
    """The fallback spec, read from `env` (defaults to `os.environ`).

    Does not load `.env`; entry points do that, matching tau2, which calls
    `load_dotenv()` itself at import time.
    """
    env = os.environ if env is None else env
    return ModelSpec(
        provider=env.get(f"{_PREFIX}PROVIDER", DEFAULT_PROVIDER).strip().lower(),
        model=env.get(f"{_PREFIX}MODEL", "").strip(),
        temperature=_read_number(env, f"{_PREFIX}TEMPERATURE", float, DEFAULT_TEMPERATURE),
        timeout=_read_number(env, f"{_PREFIX}TIMEOUT", float, DEFAULT_TIMEOUT),
        max_tokens=_read_number(env, f"{_PREFIX}MAX_TOKENS", int, None),
    )
