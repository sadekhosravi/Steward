"""The one sub-agent there is so far: a plain assistant that follows the policy.

Environment tools are declared to it as *external* tools -- pydantic-ai's term
for a tool the model may call but someone else executes. That is exactly the
benchmark's contract, so the agent never holds a callable it could fire by
accident.
"""

from __future__ import annotations

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset

import llm

INSTRUCTIONS = """
You are a customer service agent. Help the user by following the policy below.
Each turn you either send a message to the user or make tool calls, never both.

<policy>
{policy}
</policy>
""".strip()


def build_assistant(
    tools: list[ToolDefinition], policy: str, model: str | Model | None = None
) -> Agent[None, str | DeferredToolRequests]:
    """An agent whose run ends with either a reply or the tool calls it wants made.

    `model` is a model id, or a ready-made model when the caller has one -- which
    is how tests hand it a scripted stand-in instead of a live endpoint.
    """
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS.format(policy=policy),
        toolsets=[ExternalToolset(tools)],
        output_type=[str, DeferredToolRequests],
    )
