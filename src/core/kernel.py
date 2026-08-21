"""The Kernel: the LangGraph graph that runs the multi-agent system.

The Kernel is deliberately LLM-free. Every choice it makes is a routing decision
over state; the model calls happen inside nodes, in `agents`. Reward is binary
per task and pass^k only counts a task when every trial passes, so sampling
variance in *control flow* is a direct score loss -- worth spending determinism
on even where a model would be more flexible.

The graph pauses the way the benchmark does. The `act` node calls `interrupt()`
with the tool calls the model wants, control returns to whoever is driving the
Kernel, they run those calls against the real environment, and `resume()` feeds
the results back in. Sub-agents added later inherit this for free: an interrupt
resumes inside the node that raised it, however deep in the graph it sits.

    kernel = Kernel(tools, policy)
    thread = kernel.new_thread()
    step = kernel.send(thread, "I'd like to cancel my flight")
    step = kernel.resume(thread, {call_id: result_text})
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from pydantic_core import to_jsonable_python

from agents.assistant import build_assistant
from core.state import MASState

# One user turn can cost many tool round trips. LangGraph's default of 25 would
# cut a long-but-legitimate investigation short and score it as a failure.
RECURSION_LIMIT = 100


class PendingCall(BaseModel):
    """A tool call the Kernel wants made, in the form the driver has to execute."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Say:
    """The turn ended with something to tell the user."""

    text: str


@dataclass(frozen=True)
class Act:
    """The turn is paused until these calls come back."""

    calls: list[PendingCall]


Step = Say | Act


def _think(state: MASState, assistant: Agent[None, str | DeferredToolRequests]) -> dict[str, Any]:
    """Ask the assistant what to do next, given everything that has happened.

    Also the one place new evidence enters, because it is the one node that sees
    both a user message and tool results before they are consumed.
    """
    seen = [t for t in [state.prompt, *state.tool_results.values()] if t]
    results = DeferredToolResults(calls=dict(state.tool_results)) if state.tool_results else None
    run = assistant.run_sync(
        state.prompt,
        message_history=ModelMessagesTypeAdapter.validate_python(state.messages),
        deferred_tool_results=results,
    )
    output = run.output
    calls = (
        [
            PendingCall(
                id=c.tool_call_id, name=c.tool_name, arguments=c.args_as_dict()
            ).model_dump()
            for c in output.calls
        ]
        if isinstance(output, DeferredToolRequests)
        else []
    )
    return {
        "messages": to_jsonable_python(run.all_messages()),
        "observed": state.observed + seen,
        "prompt": None,
        "tool_results": {},
        "calls": calls,
        "reply": "" if calls else output,
    }


def _act(state: MASState) -> dict[str, Any]:
    """The yield point, and nothing else.

    LangGraph re-runs a node from the top when it resumes, so anything with a
    side effect placed before `interrupt()` would happen twice. That is why this
    node does only this.
    """
    results: dict[str, str] = interrupt(state.calls)
    return {"tool_results": results, "calls": []}


def _route(state: MASState) -> Literal["act", "__end__"]:
    """Tool calls mean another round trip; anything else ends the turn."""
    return "act" if state.calls else END


def build_graph(assistant: Agent[None, str | DeferredToolRequests]) -> Any:
    graph = StateGraph(MASState)
    graph.add_node("think", partial(_think, assistant=assistant))
    graph.add_node("act", _act)
    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", _route, {"act": "act", END: END})
    graph.add_edge("act", "think")
    return graph.compile(checkpointer=InMemorySaver())


class Kernel:
    """Drives one compiled graph over many conversations, one thread each."""

    def __init__(self, tools: list[ToolDefinition], policy: str, model: str | Model | None = None):
        self.graph = build_graph(build_assistant(tools, policy, model))

    def new_thread(self) -> str:
        """A fresh conversation. State for it lives in the checkpointer, not here."""
        return uuid4().hex

    def send(self, thread: str, text: str) -> Step:
        """Deliver a user message and run until the Kernel needs something."""
        return self._run(thread, {"prompt": text})

    def resume(self, thread: str, results: dict[str, str]) -> Step:
        """Hand back the results of the calls from the last `Act` and carry on."""
        return self._run(thread, Command(resume=results))

    def _run(self, thread: str, payload: Any) -> Step:
        config = {
            "configurable": {"thread_id": thread},
            "recursion_limit": RECURSION_LIMIT,
        }
        out = self.graph.invoke(payload, config)
        paused = out.get("__interrupt__")
        if paused:
            return Act(calls=[PendingCall(**call) for call in paused[0].value])
        return Say(text=out["reply"])
