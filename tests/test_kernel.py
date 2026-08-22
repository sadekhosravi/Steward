"""The Kernel's control flow. No network: the model is a scripted stand-in."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from core.kernel import Act, Kernel, Say
from tests.tools import LOOKUP, PLANNER


def call_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Look something up the first time round, then report what came back."""
    seen = [part for m in messages for part in m.parts if isinstance(part, ToolCallPart)]
    if not seen:
        return ModelResponse(
            parts=[ToolCallPart("get_reservation", {"reservation_id": "HJK4RT"}, tool_call_id="c1")]
        )
    returned = [
        p for m in messages for p in m.parts if getattr(p, "part_kind", "") == "tool-return"
    ]
    return ModelResponse(parts=[TextPart(f"Done: {returned[-1].content}")])


def always_reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("Hello, how can I help?")])


def kernel(behaviour) -> Kernel:
    return Kernel(
        [LOOKUP], policy="Be helpful.", model=FunctionModel(behaviour), planner_model=PLANNER
    )


def test_a_turn_with_no_tool_calls_ends_immediately():
    k = kernel(always_reply)
    step = k.send(k.new_thread(), "hi")
    assert isinstance(step, Say)
    assert step.text == "Hello, how can I help?"


def test_tool_calls_pause_the_turn_and_results_resume_it():
    """The whole point of the graph: `act` yields, and `resume` picks up inside it."""
    k = kernel(call_then_report)
    thread = k.new_thread()

    paused = k.send(thread, "check HJK4RT")
    assert isinstance(paused, Act)
    assert [(c.name, c.arguments) for c in paused.calls] == [
        ("get_reservation", {"reservation_id": "HJK4RT"})
    ]

    done = k.resume(thread, {paused.calls[0].id: "found it"})
    assert isinstance(done, Say)
    assert done.text == "Done: found it"


def test_the_model_never_receives_a_callable_to_run_itself():
    """Environment tools are declared, not bound; executing one here would be invisible
    to the trajectory and would corrupt the scored database."""
    seen: list[str] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart("ok")])

    k = kernel(record)
    k.send(k.new_thread(), "hi")
    assert seen == ["get_reservation"]


def test_threads_do_not_share_history():
    """One Kernel serves many simulations; a leak between them would poison scoring."""
    k = kernel(call_then_report)
    first, second = k.new_thread(), k.new_thread()

    k.send(first, "check HJK4RT")
    k.resume(first, {"c1": "found it"})

    assert isinstance(k.send(second, "check HJK4RT"), Act)
