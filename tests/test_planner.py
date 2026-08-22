"""PLANNER: what it is told, what it may say, and how the actor is shown it.

No network. The planner is a scripted stand-in, which is what lets these assert
on the shape of a plan rather than on a model's opinion of one.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from agents.planner import Plan, brief, build_planner, catalogue, render
from tests.tools import CANCEL, LOOKUP

POLICY = "Cancellations within 24 hours are free."


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
