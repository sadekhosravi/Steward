"""LLM provider access for MAS agents.

One call gives an agent a fully configured model, settings included:

    from pydantic_ai import Agent
    import llm

    critic = Agent(
        model=llm.get_model("nvidia/nemotron-3.5-lightning-30b-a3b", temperature=0.0),
        output_type=Verdict,
    )

    investigator = Agent(model=llm.get_model("openai/gpt-oss-20b"))

Every argument is optional; anything omitted falls back to the `MAS_LLM_*`
environment defaults, so the model can be changed without touching code. To
suppress a setting entirely rather than default it, build a `ModelSpec` with
that field set to `None` and pass it to `build_model`.
"""

from llm.config import (
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    LLMConfigError,
    ModelSpec,
    env_defaults,
)
from llm.providers import (
    PROVIDERS,
    ProviderInfo,
    api_key_for,
    build_model,
    get_provider,
    litellm_name,
    model_settings,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
    "PROVIDERS",
    "LLMConfigError",
    "ModelSpec",
    "ProviderInfo",
    "api_key_for",
    "build_model",
    "env_defaults",
    "get_provider",
    "get_model",
    "litellm_name",
    "model_settings",
    "resolve",
]


def resolve(
    model: str | None = None,
    *,
    provider: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> ModelSpec:
    """Fill in whatever was not passed from the `MAS_LLM_*` environment defaults."""
    defaults = env_defaults()
    return ModelSpec(
        provider=(provider or defaults.provider).strip().lower(),
        model=(model or defaults.model).strip(),
        temperature=defaults.temperature if temperature is None else temperature,
        timeout=defaults.timeout if timeout is None else timeout,
        max_tokens=defaults.max_tokens if max_tokens is None else max_tokens,
    )


def get_model(
    model: str | None = None,
    *,
    provider: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_tokens: int | None = None,
):
    """A pydantic-ai model, with temperature/timeout/max_tokens already applied."""
    return build_model(
        resolve(
            model,
            provider=provider,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    )
