"""PLANNER: what it is told, what it may say, and how the actor is shown it.

No network. The planner is a scripted stand-in, which is what lets these assert
on the shape of a plan rather than on a model's opinion of one.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from agents.gate import transcript
from agents.planner import Plan, brief, build_planner, catalogue, render
from core.kernel import Act, Kernel, Say
from tests.tools import CANCEL, LOOKUP

POLICY = "Cancellations within 24 hours are free."

GOAL = "Cancel the reservation and refund it."


# --- scripted models --------------------------------------------------------


def _plans(payload: dict):
    """Answer with the planner's output type, whatever pydantic-ai named it."""

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, payload)])

    return planner


def _captures(seen: list[str]):
    """Record the instructions the planner was built with, then answer minimally.

    They arrive on the request itself rather than among its parts -- `instructions`
    is not a message, which is the whole difference between it and a system prompt.
    """

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(m.instructions for m in messages if getattr(m, "instructions", None))
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {"goal": "ok"})])

    return planner


# --- the catalogue ----------------------------------------------------------


def test_the_planner_is_told_which_tools_change_things():
    """It has to keep reads and writes in separate fields, so it has to tell them apart."""
    listing = catalogue([LOOKUP, CANCEL])
    assert "- get_reservation (read):" in listing
    assert "- cancel_reservation (write):" in listing


def test_an_unlabelled_tool_is_treated_as_a_write():
    """The same default the graph gates on: unknown means assume it is irreversible."""
    unlabelled = ToolDefinition(name="mystery", description="Who knows.", metadata=None)
    assert "(write)" in catalogue([unlabelled])


def test_a_multi_line_description_becomes_one_line():
    """The adapter's `Returns ...` line is worth having; fourteen of them stacked is a wall."""
    wordy = ToolDefinition(
        name="get_user",
        description="Look up a user.\nReturns an object with: user_id, payment_methods.",
        metadata={"gated": False},
    )
    entry = catalogue([wordy])
    assert entry.count("\n") == 0
    assert "payment_methods" in entry


def test_the_tools_and_the_policy_both_reach_the_model():
    seen: list[str] = []
    planner = build_planner([LOOKUP, CANCEL], POLICY, FunctionModel(_captures(seen)))
    planner.run_sync("plan this")
    assert "cancel_reservation (write)" in seen[0]
    assert POLICY in seen[0]


def test_the_planner_is_given_no_tools_to_call():
    """It plans; a call from here would reach the environment with nothing between."""
    planner = build_planner([LOOKUP, CANCEL], POLICY, FunctionModel(_plans({"goal": "ok"})))
    result = planner.run_sync("plan this")
    assert isinstance(result.output, Plan)
    called = [
        p.tool_name for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)
    ]
    assert "get_reservation" not in called
    assert "cancel_reservation" not in called


# --- the brief --------------------------------------------------------------


def test_the_brief_survives_an_empty_conversation():
    """The planner runs at the top of the first turn, when there is no history at all."""
    case = brief([], "I want to cancel.")
    assert "(nothing yet)" in case
    assert "I want to cancel." in case


def test_a_turn_with_no_new_message_says_so():
    """Re-planning mid-turn has no new customer message; a blank heading reads as a bug."""
    assert "continue from the conversation above" in brief([], None)


def test_the_brief_carries_what_was_looked_up():
    """A plan written without the results of the last lookups would re-plan the past."""
    history: list[ModelMessage] = [ModelResponse(parts=[TextPart("Let me check that for you.")])]
    assert "Let me check that for you." in brief(history, "and my bag?")


# --- what the actor is shown ------------------------------------------------


def test_a_full_plan_renders_every_section_in_order():
    text = render(
        Plan(
            goal="The reservation is cancelled.",
            lookups=["Find the reservation with get_reservation."],
            confirm="The 50 dollar penalty.",
            changes=["Cancel it with cancel_reservation."],
        )
    )
    assert text.index("Find out first") < text.index("Confirm before") < text.index("Then change")
    assert "  1. Find the reservation with get_reservation." in text
    assert "  1. Cancel it with cancel_reservation." in text


def test_a_plan_that_changes_nothing_shows_no_change_section():
    """An empty heading reads as an instruction to find something to put under it, and
    doing nothing is the correct answer to half the tasks in this benchmark."""
    text = render(Plan(goal="The policy does not allow this.", lookups=["Check the fare class."]))
    assert "Then change" not in text
    assert "Confirm before" not in text
    assert "The policy does not allow this." in text


def test_the_plan_tells_the_actor_it_may_deviate():
    """A plan the actor cannot overrule is worse than none: it was written before the lookups."""
    text = render(Plan(goal="Anything."))
    assert "not a script" in text


def test_a_plan_with_nothing_in_it_still_renders():
    """The model can return a goal and no lists; that must not raise on the way to the actor."""
    assert "Goal: Anything." in render(Plan(goal="Anything."))


# --- wiring -----------------------------------------------------------------


def _counted(behaviour):
    """A model that records how many times it was asked anything."""
    calls: list[int] = []

    def wrapped(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return behaviour(messages, info)

    return wrapped, calls


def _plans_the_goal(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"goal": GOAL})])


def _refuses_to_plan(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answers in prose when an output tool was required -- the documented 20B failure.

    pydantic-ai retries it, and when the retries run out the run raises. That is the
    path this exercises: what the graph does when the planner never answers.
    """
    return ModelResponse(parts=[TextPart("I would rather not.")])


def _looks_up_then_reports(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if not [p for m in messages for p in m.parts if isinstance(p, ToolCallPart)]:
        call = ToolCallPart("get_reservation", {"reservation_id": "HKD3PS"}, tool_call_id="c1")
        return ModelResponse(parts=[call])
    return ModelResponse(parts=[TextPart("Done.")])


def _records_instructions(seen: list[str]):
    def actor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(m.instructions for m in messages if getattr(m, "instructions", None))
        return ModelResponse(parts=[TextPart("Hello.")])

    return actor


def test_the_planner_runs_once_per_user_turn_not_once_per_tool_call():
    """The whole cost argument for putting `plan` at the entry rather than in the loop.

    A turn of many tool calls re-enters `think` every time and `plan` never, because
    `resume` returns into `act` and rejoins the graph below it.
    """
    planner, planned = _counted(_plans_the_goal)
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_looks_up_then_reports),
        planner_model=FunctionModel(planner),
    )
    thread = k.new_thread()

    paused = k.send(thread, "cancel HKD3PS")
    assert isinstance(paused, Act)
    k.resume(thread, {paused.calls[0].id: "HKD3PS: economy"})

    assert len(planned) == 1


def test_the_plan_reaches_the_actor_as_instructions():
    seen: list[str] = []
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_records_instructions(seen)),
        planner_model=FunctionModel(_plans_the_goal),
    )
    k.send(k.new_thread(), "cancel HKD3PS")

    assert any(GOAL in instructions for instructions in seen)


def test_the_plan_never_enters_the_conversation():
    """The reason it is instructions and not a prompt.

    `transcript` renders every user part as "Customer:", so a plan prepended to the
    message would reach the gate as something the customer said, and the gate would
    judge the actor against words nobody spoke.
    """
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_looks_up_then_reports),
        planner_model=FunctionModel(_plans_the_goal),
    )
    thread = k.new_thread()
    k.send(thread, "cancel HKD3PS")

    state = k.graph.get_state({"configurable": {"thread_id": thread}}).values
    history = ModelMessagesTypeAdapter.validate_python(state["messages"])

    assert GOAL in state["plan"]
    assert GOAL not in transcript(history)


def test_a_planner_that_never_answers_does_not_stop_the_turn():
    """It fails open where the gate fails closed. A verdict that never arrived
    authorises nothing; a plan that never arrived withholds only the advice."""
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_records_instructions([])),
        planner_model=FunctionModel(_refuses_to_plan),
    )
    step = k.send(k.new_thread(), "cancel HKD3PS")

    assert isinstance(step, Say)
    assert step.text == "Hello."
