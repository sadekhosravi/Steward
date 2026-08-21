"""The Kernel's control flow. No network: the model is a scripted stand-in."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from core.kernel import Act, Kernel, Say

CANCEL = ToolDefinition(
    name="cancel_reservation",
    description="Cancel a reservation.",
    parameters_json_schema={
        "type": "object",
        "properties": {"reservation_id": {"type": "string"}},
        "required": ["reservation_id"],
    },
)


def call_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call the tool the first time round, then report whatever it returned."""
    seen = [part for m in messages for part in m.parts if isinstance(part, ToolCallPart)]
    if not seen:
        return ModelResponse(
            parts=[
                ToolCallPart("cancel_reservation", {"reservation_id": "HJK4RT"}, tool_call_id="c1")
            ]
        )
    returned = [
        p for m in messages for p in m.parts if getattr(p, "part_kind", "") == "tool-return"
    ]
    return ModelResponse(parts=[TextPart(f"Done: {returned[-1].content}")])


def always_reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("Hello, how can I help?")])


def kernel(behaviour) -> Kernel:
    return Kernel([CANCEL], policy="Be helpful.", model=FunctionModel(behaviour))


def test_a_turn_with_no_tool_calls_ends_immediately():
    k = kernel(always_reply)
    step = k.send(k.new_thread(), "hi")
    assert isinstance(step, Say)
    assert step.text == "Hello, how can I help?"


def test_tool_calls_pause_the_turn_and_results_resume_it():
    """The whole point of the graph: `act` yields, and `resume` picks up inside it."""
    k = kernel(call_then_report)
    thread = k.new_thread()

    paused = k.send(thread, "cancel HJK4RT")
    assert isinstance(paused, Act)
    assert [(c.name, c.arguments) for c in paused.calls] == [
        ("cancel_reservation", {"reservation_id": "HJK4RT"})
    ]

    done = k.resume(thread, {paused.calls[0].id: "cancelled"})
    assert isinstance(done, Say)
    assert done.text == "Done: cancelled"


def test_the_model_never_receives_a_callable_to_run_itself():
    """Environment tools are declared, not bound; executing one here would be invisible
    to the trajectory and would corrupt the scored database."""
    seen: list[str] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart("ok")])

    k = kernel(record)
    k.send(k.new_thread(), "hi")
    assert seen == ["cancel_reservation"]


def test_threads_do_not_share_history():
    """One Kernel serves many simulations; a leak between them would poison scoring."""
    k = kernel(call_then_report)
    first, second = k.new_thread(), k.new_thread()

    k.send(first, "cancel HJK4RT")
    k.resume(first, {"c1": "cancelled"})

    assert isinstance(k.send(second, "cancel HJK4RT"), Act)
