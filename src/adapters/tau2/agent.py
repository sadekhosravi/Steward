"""The tau2 seam: translation between tau2's message types and the Kernel.

tau2 hands us a message and expects back either text or tool calls; emitting
tool calls returns control to the environment, which runs them and calls us
again with the results. The Kernel pauses on exactly that boundary, so this
module is only translation -- no decisions are taken here.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.tool import Tool
from tau2.environment.toolkit import MUTATES_STATE_ATTR

from adapters.tau2.descriptions import describe
from adapters.tau2.schemas import tighten
from core.kernel import Act, Kernel, Step


@dataclass
class AgentState:
    """tau2's per-conversation state: a handle into the Kernel's checkpointer."""

    thread: str


# Ending the conversation is irreversible, and tau2 does not label it that way:
# `transfer_to_human_agents` touches no table, so `mutates_state` is False and a
# gate keyed on writes alone never reviews the single most damaging action
# available. The diagnostic run made the cost plain -- 48 transfer calls across 50
# tasks, the second most-used tool in the run, correct for one of them. Named here
# rather than in the Kernel because it is a fact about tau2's toolset, and `core`
# does not import tau2. tau2 spells it the same way in every domain
# (`LLMSoloAgent.TRANSFER_TOOL_NAME`).
HANDOFF = frozenset({"transfer_to_human_agents"})


def _tool_def(tool: Tool) -> ToolDefinition:
    """Describe a tau2 tool without binding its callable.

    Calling a `Tool` in-process would mutate the scored database without the call
    ever appearing in the trajectory, so only the schema crosses over -- plus the
    one label the gate needs. `mutates_state` is set by tau2's `is_tool`
    decorator on the underlying function, which makes "is this a write?" a fact
    about the domain rather than something we have to guess or maintain a list of.
    What the gate routes on is the wider question -- can this be taken back? -- so
    the handoff is folded in here. `ToolDefinition.metadata` is not sent to the
    model, so carrying it there costs nothing in the prompt.

    The parameter schema is tightened on the way through, because tau2 declares a
    looser one than it enforces -- see `schemas`. The model is shown the tightened
    version and held to it, which is the whole point: a schema nobody checks is a
    suggestion. The description is widened for the opposite reason -- see
    `descriptions` -- because tau2's says less than the model needs.
    """
    schema = tool.openai_schema["function"]
    mutates = getattr(tool._func, MUTATES_STATE_ATTR, True)
    return ToolDefinition(
        name=schema["name"],
        description=describe(tool),
        parameters_json_schema=tighten(schema["parameters"]),
        metadata={"gated": mutates or schema["name"] in HANDOFF},
    )


def _tool_results(message: ValidAgentInputMessage) -> dict[str, str] | None:
    """Results for the calls we yielded last turn, or None if this is a user turn.

    tau2 preserves `ToolCall.id` as `ToolMessage.id`, which is the same key the
    Kernel resumes on, so routing needs no bookkeeping of its own.
    """
    if isinstance(message, MultiToolMessage):
        tool_messages = message.tool_messages
    elif isinstance(message, ToolMessage):
        tool_messages = [message]
    else:
        return None
    return {m.id: m.content or "" for m in tool_messages}


def _to_tau2(step: Step) -> AssistantMessage:
    """A Kernel step as a tau2 assistant message: text XOR tool calls."""
    if isinstance(step, Act):
        return AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id=call.id, name=call.name, arguments=call.arguments)
                for call in step.calls
            ],
        )
    return AssistantMessage.text(step.text)


class StewardAgent(HalfDuplexAgent[AgentState]):
    """The agent-under-test, as tau2 sees it."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        model: str | Model | None = None,
        gate_model: str | Model | None = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.kernel = Kernel([_tool_def(t) for t in tools], domain_policy, model, gate_model)

    def get_init_state(self, message_history: list[Message] | None = None) -> AgentState:
        return AgentState(thread=self.kernel.new_thread())

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        results = _tool_results(message)
        if results is None:
            step = self.kernel.send(state.thread, message.content)
        else:
            step = self.kernel.resume(state.thread, results)
        return _to_tau2(step), state
