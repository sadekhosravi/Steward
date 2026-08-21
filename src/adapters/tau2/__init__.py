"""tau2-bench binding.

Holds the HalfDuplexAgent subclass and its registry registration. tau2 has no
plugin discovery, so `register()` must run in the same process as the runner,
before it starts -- see scripts/run_bench.py.
"""

from adapters.tau2.agent import AgentState, MASAgent

AGENT_NAME = "mas"

__all__ = ["AGENT_NAME", "AgentState", "MASAgent", "create_agent", "register"]


def create_agent(tools, domain_policy, **kwargs):
    """tau2 agent factory. `--agent-llm` picks the model, resolved by `llm`."""
    return MASAgent(tools=tools, domain_policy=domain_policy, model=kwargs.get("llm"))


def register(name: str = AGENT_NAME) -> None:
    """Make the agent available as `--agent <name>`. Safe to call twice."""
    from tau2.registry import registry

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(create_agent, name)
