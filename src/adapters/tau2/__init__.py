"""tau2-bench binding.

Holds the HalfDuplexAgent subclass and its registry registration. tau2 has no
plugin discovery, so `register()` must run in the same process as the runner,
before it starts -- see scripts/run_bench.py.
"""

import os

import llm
from adapters.tau2.agent import AgentState, StewardAgent

AGENT_NAME = "steward"

__all__ = ["AGENT_NAME", "GATE_MODEL_ENV", "AgentState", "StewardAgent", "create_agent", "register"]


# tau2 always sends `--agent-llm-args`, defaulting to {"temperature": 0.0}, so
# the settings it knows about have to be named here or the run silently ignores
# them. Anything else is a typo, and a typo that changes nothing is worse than
# an error at startup.
_LLM_ARGS = frozenset({"temperature", "timeout", "max_tokens", "reasoning_effort"})

# tau2's CLI has one `--agent-llm`, so a second model for the critic can only
# come from the environment. Unset means the critic runs on the actor's model,
# which is the configuration to beat before spending on a stronger one.
GATE_MODEL_ENV = "STEWARD_GATE_MODEL"


def create_agent(tools, domain_policy, **kwargs):
    """tau2 agent factory. `--agent-llm` and `--agent-llm-args` pick the model."""
    args = dict(kwargs.get("llm_args") or {})
    if unknown := sorted(set(args) - _LLM_ARGS):
        raise llm.LLMConfigError(
            f"--agent-llm-args does not support {', '.join(unknown)}; "
            f"supported keys are {', '.join(sorted(_LLM_ARGS))}"
        )
    model = llm.get_model(kwargs.get("llm") or None, **args)
    gate = os.environ.get(GATE_MODEL_ENV, "").strip()
    return StewardAgent(
        tools=tools,
        domain_policy=domain_policy,
        model=model,
        gate_model=llm.get_model(gate, **args) if gate else None,
    )


def register(name: str = AGENT_NAME) -> None:
    """Make the agent available as `--agent <name>`. Safe to call twice."""
    from tau2.registry import registry

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(create_agent, name)
