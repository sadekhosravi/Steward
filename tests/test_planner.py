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
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from agents.gate import transcript
from agents.planner import Plan, brief, build_planner, catalogue, render
from core.kernel import REPLAN_LIMIT, Act, Kernel, Say
from core.state import Change
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


def test_the_planner_is_not_asked_to_rule_on_policy():
    """A `goal` that reads as a verdict empties `changes`, and an empty `changes` is
    nothing for the speaker to count -- which is how most of the gold writes in the
    diagnostic run were lost before the gate ever saw a proposal."""
    seen: list[str] = []
    build_planner([LOOKUP, CANCEL], POLICY, FunctionModel(_captures(seen))).run_sync("plan this")

    told = seen[0]

    assert "WHETHER IT IS ALLOWED IS NOT YOURS TO DECIDE" in told
    assert "WHEN THE ANSWER IS NO" not in told


def test_the_planner_is_told_something_else_does_the_refusing():
    """It stops writing verdicts only if it knows a reviewer is there to write them."""
    seen: list[str] = []
    build_planner([LOOKUP, CANCEL], POLICY, FunctionModel(_captures(seen))).run_sync("plan this")

    assert "checked against this policy before it runs" in seen[0]


def test_the_permission_asymmetry_is_fenced_off_from_scope():
    """The push to plan a change even when the policy is doubtful is deliberate, and
    it was being read as licence to plan changes nobody asked for. 44 of the 51
    surplus writes in the last full run were written down in a plan first; the actor
    freelanced once in 150 simulations."""
    seen: list[str] = []
    build_planner([LOOKUP, CANCEL], POLICY, FunctionModel(_captures(seen))).run_sync("plan this")

    told = " ".join(seen[0].split())

    assert "THAT IS ABOUT PERMISSION. IT IS NOT ABOUT SCOPE" in told
    assert "If you cannot find the sentence, delete the entry" in told
    assert "the policy permits almost every change nobody asked for" in told


def test_the_planner_is_told_a_question_is_finished_by_answering_it():
    """119 of the 308 unwanted changes in run 017 -- 39% -- were planned on tasks
    that wanted no write at all. The largest single thing this plan gets wrong."""
    seen: list[str] = []
    build_planner([LOOKUP, CANCEL], POLICY, FunctionModel(_captures(seen))).run_sync("plan this")

    told = " ".join(seen[0].split())

    assert "MANY REQUESTS ARE FINISHED BY ANSWERING THEM" in told
    assert "`changes` stays empty, and that is the plan being right" in told


def test_a_change_has_to_quote_the_customer():
    """The operational half. Stating the rule in the instructions was measured inert
    in run 018; the field the model actually fills has to carry it too."""
    described = Change.model_fields["what"].description or ""

    assert "in quotes" in described
    assert "If you cannot quote them, they did not ask" in described


def test_the_goal_is_the_shape_of_the_records_not_a_ruling():
    """The field description is the only place the model is told what `goal` is for."""
    described = Plan.model_fields["goal"].description or ""

    assert "Never a verdict" in described


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
            changes=[Change(tool="cancel_reservation", record="HKD3PS", what="cancel it")],
        )
    )
    assert text.index("Find out first") < text.index("Confirm before") < text.index("Then change")
    assert "  1. Find the reservation with get_reservation." in text
    assert "  1. cancel_reservation on HKD3PS: cancel it" in text


def test_a_plan_that_changes_nothing_shows_no_change_section():
    """An empty heading reads as an instruction to find something to put under it, and
    doing nothing is the correct answer to half the tasks in this benchmark."""
    text = render(Plan(goal="The fare class is known.", lookups=["Check the fare class."]))
    assert "Then change" not in text
    assert "Confirm before" not in text
    assert "The fare class is known." in text


def test_the_plan_tells_the_actor_it_may_deviate():
    """A plan the actor cannot overrule is worse than none: it was written before the lookups."""
    text = render(Plan(goal="Anything."))
    assert "not a script" in text


def test_a_plan_with_nothing_in_it_still_renders():
    """The model can return a goal and no lists; that must not raise on the way to the actor."""
    assert "Goal for this turn: Anything." in render(Plan(goal="Anything."))


# --- wiring -----------------------------------------------------------------


def _counted(behaviour):
    """A model that records the case it was handed, once per time it was asked.

    The case rather than a tally, so the same stand-in answers both questions a
    re-plan raises: how often the planner ran, and whether the results that
    prompted the run were in front of it.
    """
    asked: list[str] = []

    def wrapped(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        asked.append(
            "\n".join(
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
        )
        return behaviour(messages, info)

    return wrapped, asked


def _looks_up_every_time(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """An actor that never stops investigating. Only the budget ends the turn."""
    seen = sum(1 for m in messages for p in m.parts if isinstance(p, ToolCallPart))
    return ModelResponse(
        parts=[ToolCallPart("get_reservation", {"reservation_id": "HKD3PS"}, f"c{seen}")]
    )


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


def test_the_planner_runs_again_when_a_lookup_comes_back():
    """The turn's first plan is its worst-informed, and this is where that is fixed.

    `resume` returns into `act`, and `act` now rejoins the graph at `plan` rather
    than below it -- so one lookup buys one more plan, written with the answer in
    hand. It is the only edge that does this; a refusal or a held reply re-enters
    `think` directly, because neither of them is news about the world.
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

    assert len(planned) == 2


def test_a_re_plan_is_shown_what_just_came_back():
    """The results are on the state and not yet in the history -- the actor is what
    folds them in, and it has not run. Handed over separately or not at all."""
    planner, planned = _counted(_plans_the_goal)
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_looks_up_then_reports),
        planner_model=FunctionModel(planner),
    )
    thread = k.new_thread()

    paused = k.send(thread, "cancel HKD3PS")
    k.resume(thread, {paused.calls[0].id: "HKD3PS: basic economy, flown"})

    assert "HKD3PS: basic economy, flown" in planned[-1]


def test_re_planning_stops_at_the_budget():
    """Running out takes nothing away: the turn carries on under the plan it has."""
    planner, planned = _counted(_plans_the_goal)
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_looks_up_every_time),
        planner_model=FunctionModel(planner),
    )
    thread = k.new_thread()

    step = k.send(thread, "cancel HKD3PS")
    for _ in range(6):
        if not isinstance(step, Act):
            break
        step = k.resume(thread, {call.id: "HKD3PS: economy" for call in step.calls})

    assert len(planned) == 1 + REPLAN_LIMIT


# --- a re-plan may widen the turn and may not narrow it ----------------------

SECTIONED = """
# Toy Policy

Confirm before you change anything.

## Domain Basic

A reservation has a cabin.

## Cancel flight

Cancelling is free within 24 hours.

## Refunds

A refund takes five days.
"""


def _plans_in_turn(*payloads: dict):
    """A planner whose answer changes between calls, then holds at the last one."""
    remaining = list(payloads)

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

    return planner


def _state_after_one_lookup(planner_model, policy: str = POLICY) -> dict:
    k = Kernel(
        [LOOKUP],
        policy=policy,
        model=FunctionModel(_looks_up_then_reports),
        planner_model=planner_model,
    )
    thread = k.new_thread()
    paused = k.send(thread, "cancel HKD3PS")
    k.resume(thread, {paused.calls[0].id: "HKD3PS: economy"})
    return k.graph.get_state({"configurable": {"thread_id": thread}}).values


def test_a_re_plan_cannot_drop_a_change_the_turn_still_owes():
    """The failure this guards. `outstanding` counts `changes` against what the gate
    approved, so a planner that decides mid-turn the job is done would empty the
    list and switch the speaker off for the rest of the turn."""
    owed = [{"tool": "cancel_reservation", "record": "HKD3PS", "what": "cancel it"}]

    values = _state_after_one_lookup(
        FunctionModel(
            _plans_in_turn({"goal": GOAL, "changes": owed}, {"goal": GOAL, "changes": []})
        )
    )

    assert [c.tool for c in values["changes"]] == ["cancel_reservation"]


def test_a_re_plan_adds_a_change_it_has_only_now_realised_is_needed():
    values = _state_after_one_lookup(
        FunctionModel(
            _plans_in_turn(
                {"goal": GOAL, "changes": [{"tool": "cancel_reservation", "what": "cancel it"}]},
                {"goal": GOAL, "changes": [{"tool": "send_certificate", "what": "refund it"}]},
            )
        )
    )

    assert [c.tool for c in values["changes"]] == ["cancel_reservation", "send_certificate"]


def test_a_re_plan_cannot_take_away_a_policy_section_the_actor_is_working_from():
    """The original objection to moving a plan mid-turn, and the answer to it: the
    rules only ever widen, so nothing the actor is halfway through applying goes."""
    values = _state_after_one_lookup(
        FunctionModel(
            _plans_in_turn(
                {"goal": GOAL, "policy_sections": ["Cancel flight"]},
                {"goal": GOAL, "policy_sections": ["Refunds"]},
            )
        ),
        policy=SECTIONED,
    )

    assert "Cancelling is free within 24 hours." in values["policy"]
    assert "A refund takes five days." in values["policy"]


def _behaves_in_turn(*behaviours):
    """One stand-in per call, then the last one for good."""
    remaining = list(behaviours)

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return (remaining.pop(0) if len(remaining) > 1 else remaining[0])(messages, info)

    return planner


def test_a_re_plan_that_never_answers_leaves_the_previous_plan_standing():
    """Fails open, and more cheaply than the opening plan does: there is already a
    plan, so a re-plan nobody answered costs the correction and nothing else."""
    values = _state_after_one_lookup(
        FunctionModel(_behaves_in_turn(_plans_the_goal, _refuses_to_plan))
    )

    assert GOAL in values["plan"]


def test_a_re_plan_that_fails_still_spends_its_budget():
    """A planner that cannot answer is the one most likely to be asked again on the
    very next round trip, so a budget charged only for successes bounds nothing."""
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_looks_up_every_time),
        planner_model=FunctionModel(_behaves_in_turn(_plans_the_goal, _refuses_to_plan)),
    )
    thread = k.new_thread()

    step = k.send(thread, "cancel HKD3PS")
    for _ in range(6):
        if not isinstance(step, Act):
            break
        step = k.resume(thread, {call.id: "HKD3PS: economy" for call in step.calls})

    assert k.graph.get_state({"configurable": {"thread_id": thread}}).values["replans"] == (
        REPLAN_LIMIT
    )


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


# --- the request, and what may rewrite it ------------------------------------

WIDE = "Change the cabin on all four of Omar Davis's upcoming reservations."
NARROW = "Change the cabin on reservation JG7FMM."


def test_a_lookup_cannot_narrow_the_request():
    """The failure this guards, verbatim from the last full run: a plan said "each
    of Omar Davis's reservations" and then, one lookup later, "reservation
    JG7FMM". The other four were never mentioned by anybody again."""
    values = _state_after_one_lookup(
        FunctionModel(
            _plans_in_turn(
                {"request": WIDE, "goal": GOAL},
                {"request": NARROW, "goal": GOAL},
            )
        )
    )

    assert values["request"] == WIDE


def test_the_customer_speaking_is_what_rewrites_the_request():
    """The other direction. A scope that could never change would be worse than one
    that drifts: the customer is allowed to ask for something else."""
    k = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_just_replies),
        planner_model=FunctionModel(
            _plans_in_turn({"request": WIDE, "goal": GOAL}, {"request": NARROW, "goal": GOAL})
        ),
    )
    k.send("t", "change the cabin on all of them")
    k.send("t", "actually, only do JG7FMM")

    values = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert values["request"] == NARROW


def test_the_actor_is_shown_the_request_and_the_turn_separately():
    text = render(Plan(request=WIDE, goal="JG7FMM is in economy."))
    assert f"What the customer asked for: {WIDE}" in text
    assert "Goal for this turn: JG7FMM is in economy." in text


def test_the_standing_request_is_put_to_the_planner():
    assert WIDE in brief([], None, standing=WIDE)
    assert "keep both" in brief([], None, standing=WIDE)


def _just_replies(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("Right you are.")])


def test_a_second_request_is_added_to_the_first_not_swapped_for_it():
    """The defect the first version of this had. Told to keep the standing request
    or replace it, the planner did neither with a customer who asked for something
    *as well*: across four samples it kept the old one verbatim and dropped the new
    ask entirely. Task 7 is exactly that shape -- a cancellation, and then "what do
    my other flights cost" three messages later."""
    standing = "Cancel Daiki Muller's two upcoming reservations."
    case = brief([], "Also, what do my other upcoming flights cost in total?", standing=standing)

    assert "keep both" in case
    assert "no longer want it" in case
