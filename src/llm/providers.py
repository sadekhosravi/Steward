"""Turning a `ModelSpec` into a live pydantic-ai model.

Every supported provider speaks the OpenAI wire protocol, so all are served by
`OpenAIChatModel`; only the provider object underneath differs. pydantic-ai has
no dedicated NVIDIA/NIM provider, so NVIDIA Build is reached with the generic
`OpenAIProvider` pointed at its base URL -- the documented way to use an
OpenAI-compatible endpoint, not a workaround. Groq is reached the same way.
OpenRouter has a first-class provider, which is why switching to it is a
one-line change.

Adding a provider is a dict entry and nothing else, which is the point: which
endpoint serves a model is a deployment fact, and the same model id appears on
several of them -- `openai/gpt-oss-120b` is served by NVIDIA Build and by Groq
under exactly that name, at different speeds and prices.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from llm.config import LLMConfigError, ModelSpec
from llm.harmony import harmonised

__all__ = [
    "PROVIDERS",
    "ProviderInfo",
    "api_key_for",
    "build_model",
    "get_provider",
    "litellm_name",
    "model_settings",
]


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    name: str
    base_url: str
    api_key_env: str
    # Prefix LiteLLM uses for the same provider. tau2's user simulator routes
    # through LiteLLM rather than pydantic-ai, so a model chosen here has to be
    # named differently when it is passed to `tau2 run --llm-user`.
    litellm_prefix: str
    catalog_url: str


PROVIDERS: dict[str, ProviderInfo] = {
    "nvidia": ProviderInfo(
        name="nvidia",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        litellm_prefix="nvidia_nim",
        catalog_url="https://build.nvidia.com/models",
    ),
    "gemini": ProviderInfo(
        name="gemini",
        # Google's OpenAI-compatibility layer rather than the native API, so
        # Gemini arrives as another `OpenAIChatModel` like everything else. The
        # trailing slash is required: the path is a prefix, not a host root.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        litellm_prefix="gemini",
        catalog_url="https://ai.google.dev/gemini-api/docs/models",
    ),
    "groq": ProviderInfo(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        litellm_prefix="groq",
        catalog_url="https://console.groq.com/docs/models",
    ),
    "openrouter": ProviderInfo(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        litellm_prefix="openrouter",
        catalog_url="https://openrouter.ai/models",
    ),
}

# A model wraps an AsyncOpenAI client whose httpx connection pool binds to the
# event loop that first drives it. pydantic-ai's `run_sync` creates one loop per
# thread and reuses it, and tau2 runs simulations on a ThreadPoolExecutor, so a
# process-wide cache would hand one pool to several loops. Per-thread caching
# gives each simulation its own client and keeps reuse within a thread.
_local = threading.local()


def get_provider(name: str) -> ProviderInfo:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise LLMConfigError(
            f"unknown provider {name!r}; supported providers are {', '.join(sorted(PROVIDERS))}"
        ) from None


def api_key_for(provider: str) -> str:
    info = get_provider(provider)
    key = os.environ.get(info.api_key_env, "").strip()
    if not key:
        raise LLMConfigError(
            f"{info.api_key_env} is not set, so the {provider!r} provider cannot be used. "
            f"Get a key at {info.catalog_url} and add it to .env."
        )
    return key


def litellm_name(spec: ModelSpec) -> str:
    """The same model named for LiteLLM, e.g. for `tau2 run --llm-user`."""
    return f"{get_provider(spec.provider).litellm_prefix}/{spec.model}"


def model_settings(spec: ModelSpec) -> OpenAIChatModelSettings:
    """The per-request settings in `spec`, omitting anything left as `None`.

    Both providers speak the OpenAI protocol, so reasoning effort travels under
    its `openai_` name even on NVIDIA Build.
    """
    settings = OpenAIChatModelSettings()
    if spec.temperature is not None:
        settings["temperature"] = spec.temperature
    if spec.timeout is not None:
        settings["timeout"] = spec.timeout
    if spec.max_tokens is not None:
        settings["max_tokens"] = spec.max_tokens
    if spec.reasoning_effort is not None:
        settings["openai_reasoning_effort"] = spec.reasoning_effort
    return settings


def _construct(spec: ModelSpec) -> Model:
    info = get_provider(spec.provider)
    if not spec.model:
        raise LLMConfigError(
            "no model given; pass model=... or set STEWARD_LLM_MODEL (or LLM_MODEL) in .env. "
            f"Available {info.name} models: {info.catalog_url}"
        )
    key = api_key_for(spec.provider)
    if info.name == "openrouter":
        provider = OpenRouterProvider(api_key=key)
    else:
        provider = OpenAIProvider(base_url=info.base_url, api_key=key)
    # Settings ride on the model, so a caller passes one object to `Agent`.
    return OpenAIChatModel(spec.model, provider=provider, settings=model_settings(spec))


def build_model(spec: ModelSpec) -> Model:
    """A pydantic-ai model for `spec`, cached per thread."""
    cache: dict[ModelSpec, Model] | None = getattr(_local, "models", None)
    if cache is None:
        cache = _local.models = {}
    if spec not in cache:
        # Wrapped on the way out rather than inside `_construct`, so every path
        # that builds a model gets the repair and none has to remember to ask.
        cache[spec] = harmonised(_construct(spec))
    return cache[spec]
