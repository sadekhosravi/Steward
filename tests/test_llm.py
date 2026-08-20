"""Config resolution and provider wiring. No network: nothing here sends a request."""

from __future__ import annotations

import threading

import pytest

import llm

KEY = "test-key-not-real"


@pytest.fixture
def env(monkeypatch):
    """A clean environment with a usable key and no MAS_LLM_* defaults."""
    for name in ("PROVIDER", "MODEL", "TEMPERATURE", "TIMEOUT", "MAX_TOKENS"):
        monkeypatch.delenv(f"MAS_LLM_{name}", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    return monkeypatch


def test_arguments_win_over_the_environment(env):
    env.setenv("MAS_LLM_MODEL", "from-env")
    env.setenv("MAS_LLM_TEMPERATURE", "0.9")
    spec = llm.resolve("openai/gpt-oss-20b", temperature=0.0)
    assert spec.model == "openai/gpt-oss-20b"
    assert spec.temperature == 0.0


def test_omitted_arguments_fall_back_to_the_environment(env):
    env.setenv("MAS_LLM_MODEL", "from-env")
    env.setenv("MAS_LLM_TIMEOUT", "30")
    spec = llm.resolve(temperature=0.7)
    assert spec.model == "from-env"
    assert spec.timeout == 30.0
    assert spec.temperature == 0.7


def test_two_agents_can_use_different_models_and_providers(env):
    env.setenv("MAS_LLM_MODEL", "openai/gpt-oss-20b")
    fast = llm.resolve()
    strong = llm.resolve("anthropic/claude-sonnet-4.5", provider="openrouter", temperature=0.2)
    assert (fast.provider, fast.model) == ("nvidia", "openai/gpt-oss-20b")
    assert (strong.provider, strong.model) == ("openrouter", "anthropic/claude-sonnet-4.5")
    assert fast.temperature == llm.DEFAULT_TEMPERATURE
    assert strong.temperature == 0.2


def test_defaults_apply_when_nothing_is_configured(env):
    spec = llm.resolve("some/model")
    assert spec.provider == llm.DEFAULT_PROVIDER
    assert spec.temperature == llm.DEFAULT_TEMPERATURE
    assert spec.timeout == llm.DEFAULT_TIMEOUT
    assert spec.max_tokens is None


def test_provider_name_is_normalised(env):
    assert llm.resolve("x/y", provider=" OpenRouter ").provider == "openrouter"


def test_non_numeric_temperature_is_a_config_error():
    with pytest.raises(llm.LLMConfigError, match="must be a number"):
        llm.env_defaults({"MAS_LLM_TEMPERATURE": "warm"})


def test_max_tokens_is_read_as_an_integer():
    assert llm.env_defaults({"MAS_LLM_MAX_TOKENS": "512"}).max_tokens == 512


def test_settings_omit_anything_left_unset():
    assert llm.model_settings(llm.ModelSpec("nvidia", "x", None, None, None)) == {}


def test_settings_carry_every_value_that_is_set():
    spec = llm.ModelSpec("nvidia", "x", temperature=0.2, timeout=5.0, max_tokens=64)
    assert llm.model_settings(spec) == {"temperature": 0.2, "timeout": 5.0, "max_tokens": 64}


def test_settings_ride_on_the_model_so_agents_pass_one_object(env):
    model = llm.get_model("openai/gpt-oss-20b", temperature=0.3, max_tokens=64)
    assert model.settings == {"temperature": 0.3, "timeout": llm.DEFAULT_TIMEOUT, "max_tokens": 64}


def test_litellm_names_differ_from_pydantic_ai_names():
    nvidia = llm.ModelSpec(provider="nvidia", model="openai/gpt-oss-20b")
    openrouter = llm.ModelSpec(provider="openrouter", model="anthropic/claude-sonnet-4.5")
    assert llm.litellm_name(nvidia) == "nvidia_nim/openai/gpt-oss-20b"
    assert llm.litellm_name(openrouter) == "openrouter/anthropic/claude-sonnet-4.5"


def test_unknown_provider_raises():
    with pytest.raises(llm.LLMConfigError, match="unknown provider"):
        llm.litellm_name(llm.ModelSpec(provider="nvida", model="x"))


def test_missing_api_key_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(llm.LLMConfigError, match="NVIDIA_API_KEY"):
        llm.api_key_for("nvidia")


def test_building_without_a_model_says_what_to_do(env):
    with pytest.raises(llm.LLMConfigError, match="MAS_LLM_MODEL"):
        llm.get_model()


def test_differing_settings_do_not_share_a_cached_model(env):
    """Settings live on the model, so they have to be part of the cache key."""
    cold = llm.get_model("openai/gpt-oss-20b", temperature=0.0)
    warm = llm.get_model("openai/gpt-oss-20b", temperature=0.9)
    assert cold is not warm
    assert cold is llm.get_model("openai/gpt-oss-20b", temperature=0.0)


def test_models_are_cached_per_thread(env):
    """A model holds an httpx pool bound to one event loop; threads must not share it."""
    spec = llm.ModelSpec(provider="nvidia", model="openai/gpt-oss-20b")
    assert llm.build_model(spec) is llm.build_model(spec)

    other: list[object] = []
    thread = threading.Thread(target=lambda: other.append(llm.build_model(spec)))
    thread.start()
    thread.join()
    assert other[0] is not llm.build_model(spec)
