"""A single pydantic-ai agent behind the tau2 seam.

tau2 hands us a message and expects back either text or tool calls. Emitting
tool calls returns control to the environment, which runs them and calls us
again with the results -- pydantic-ai calls tools with that shape *deferred*:
declared to the model, executed by someone else. Because the two sides agree,
this adapter is almost entirely translation between their message types.

This is the seam only. The multi-agent system replaces the one `Agent` below;
everything else here stays as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.tool import Tool

import llm

INSTRUCTIONS = """
You are a customer service agent. Help the user by following the policy below.
Each turn you either send a message to the user or make tool calls, never both.

<policy>
{domain_policy}
</policy>
""".strip()


@dataclass
class AgentState:
    """tau2 never serializes agent state, so pydantic-ai's own messages live here."""

    messages: list[ModelMessage] = field(default_factory=list)


def _tool_def(tool: Tool) -> ToolDefinition:
    """Describe a tau2 tool to the model without binding its callable.

    Calling a `Tool` in-process would mutate the scored database without the
    call ever appearing in the trajectory, so only the schema crosses over.
    """
    schema = tool.openai_schema["function"]
    return ToolDefinition(
        name=schema["name"],
        description=schema["description"],
        parameters_json_schema=schema["parameters"],
    )


def _to_pydantic_ai(
    message: ValidAgentInputMessage,
) -> tuple[str | None, DeferredToolResults | None]:
    """A tau2 input message as either a user prompt or results for pending calls."""
    if isinstance(message, MultiToolMessage):
        tool_messages = message.tool_messages
    elif isinstance(message, ToolMessage):
        tool_messages = [message]
    else:
        return message.content, None
    # tau2 preserves ToolCall.id as ToolMessage.id, which is the key pydantic-ai
    # matches results back to the calls it emitted last turn.
    return None, DeferredToolResults(calls={m.id: m.content or "" for m in tool_messages})


def _to_tau2(output: str | DeferredToolRequests) -> AssistantMessage:
    """A pydantic-ai run output as a tau2 assistant message: text XOR tool calls."""
    if isinstance(output, DeferredToolRequests):
        return AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id=call.tool_call_id, name=call.tool_name, arguments=call.args_as_dict())
                for call in output.calls
            ],
        )
    return AssistantMessage.text(output)


class MASAgent(HalfDuplexAgent[AgentState]):
    """The agent-under-test, as tau2 sees it."""

    def __init__(self, tools: list[Tool], domain_policy: str, model: str | None = None):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.agent = Agent(
            model=llm.get_model(model),
            instructions=INSTRUCTIONS.format(domain_policy=domain_policy),
            toolsets=[ExternalToolset([_tool_def(t) for t in tools])],
            output_type=[str, DeferredToolRequests],
        )

    def get_init_state(self, message_history: list[Message] | None = None) -> AgentState:
        return AgentState()

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        prompt, tool_results = _to_pydantic_ai(message)
        run = self.agent.run_sync(
            prompt,
            message_history=state.messages,
            deferred_tool_results=tool_results,
        )
        state.messages = run.all_messages()
        return _to_tau2(run.output), state
