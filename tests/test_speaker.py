"""SPEAKER: when it fires, what it costs when it does not, and where it stops.

No network. The actor, the planner and the speaker are separate scripted
stand-ins, because the thing under test is whether one of them can send another
back to work -- which a single model playing every part could not show.

Several of these tests assert that the speaker was *not* consulted. Those are the
ones that matter most: this node sits on the exit every turn takes, and a check
that fires on turns it has nothing to say about is a tax on the twenty-three
no-write tasks that already pass.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.speaker import NOTHING_OWED, UNCHECKED, hold, outstanding
from core.kernel import DEFERRAL_LIMIT, Act, Kernel, Say
from tests.tools import CANCEL, LOOKUP

SEEN_ID = "HKD3PS"
CHANGE = "cancel_reservation to cancel the booking"


# --- scripted models --------------------------------------------------------


def _output(info: AgentInfo, **payload: object) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


def planning(*changes: str):
    """A planner that names these changes and nothing else."""

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _output(info, goal="Do what they asked.", changes=list(changes))

    return planner


def allows(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _output(info, allowed=True, reason="Asking them to agree is the policy working.")


def holds_once():
    """Hold the first message, allow the second -- the intended correction loop."""
    seen: list[int] = []

    def speaker(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(1)
        if len(seen) > 1:
            return _output(info, allowed=True, reason="The change was made.")
        return _output(
            info,
            allowed=False,
            reason="The customer already agreed and the cancellation has not been made.",
            remediation="Call cancel_reservation with the reservation id from the lookup.",
        )

    return speaker


def always_holds(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _output(
        info,
        allowed=False,
        reason="The cancellation has not been made.",
        remediation="Call cancel_reservation now.",
    )


def fumbles(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answer in prose, which is never the output tool. Exhausts every retry."""
    return ModelResponse(parts=[TextPart("I think it is probably fine.")])


# --- scripted actors --------------------------------------------------------


def just_talks(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("Let me know if there is anything else.")])


def talks_then_acts():
    """Speak first; after a correction, make the call. The behaviour we want back."""

    def actor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, TextPart) for m in messages for part in m.parts):
            call = ToolCallPart(
                "cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id=uuid4().hex[:8]
            )
            return ModelResponse(parts=[call])
        return ModelResponse(parts=[TextPart("Shall I go ahead and cancel it?")])

    return actor


def cancels_then_talks(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if any(part.part_kind == "tool-return" for m in messages for part in m.parts):
        return ModelResponse(parts=[TextPart("Done -- that reservation is cancelled.")])
    call = ToolCallPart(
        "cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id=uuid4().hex[:8]
    )
    return ModelResponse(parts=[call])


def approves_writes(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _output(info, allowed=True, reason="The policy permits this.")


def counted(behaviour):
    """A model that records how many times it was consulted."""
    calls: list[int] = []

    def wrapped(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return behaviour(messages, info)

    return wrapped, calls


def kernel(actor, speaker, planner, gate=approves_writes) -> Kernel:
    return Kernel(
        [LOOKUP, CANCEL],
        policy="Cancel only after the customer agrees.",
        model=FunctionModel(actor),
        gate_model=FunctionModel(gate),
        planner_model=FunctionModel(planner),
        speaker_model=FunctionModel(speaker),
    )


# --- the deterministic half -------------------------------------------------


def test_a_turn_that_plans_no_changes_owes_nothing():
    assert outstanding([], []) == []


def test_a_change_whose_tool_was_approved_is_done():
    assert outstanding([CHANGE], ["cancel_reservation"]) == []


def test_a_change_whose_tool_was_never_approved_is_outstanding():
    assert outstanding([CHANGE], ["update_reservation_baggages"]) == [CHANGE]


def test_only_the_unmet_changes_are_outstanding():
    """A task needing three writes is not finished by one, and the speaker is told
    which two are left rather than that something is missing."""
    changes = [
        "update_reservation_baggages to add a bag",
        "update_reservation_passengers to change the passenger",
        "update_reservation_flights to upgrade the cabin",
    ]
    assert outstanding(changes, ["update_reservation_baggages"]) == changes[1:]


def test_a_change_naming_no_tool_stays_outstanding():
    """The planner is told to name the tool on every line. When it does not, the
    line cannot be matched -- and the cost of asking about a finished turn is one
    model call, where the cost of missing an unfinished one is the task."""
    assert outstanding(["cancel their booking"], ["cancel_reservation"]) == ["cancel their booking"]


def test_the_case_says_plainly_when_nothing_is_owed():
    assert NOTHING_OWED in hold([], "All done.", [])


# --- what it costs when it has nothing to say -------------------------------


def test_a_turn_with_no_planned_changes_never_consults_the_speaker():
    """The economy the whole node depends on. Twenty-three of the forty-nine tasks
    need no writes at all and already pass at 0.695; none of them should pay for
    this check even once."""
    speaker, consulted = counted(allows)
    step = kernel(just_talks, speaker, planning()).send("t", "what is your baggage policy?")

    assert isinstance(step, Say)
    assert consulted == []


def test_a_turn_that_made_its_change_never_consults_the_speaker():
    speaker, consulted = counted(allows)
    k = kernel(cancels_then_talks, speaker, planning(CHANGE))

    step = k.send("t", f"cancel {SEEN_ID}")
    assert isinstance(step, Act)
    step = k.resume("t", {step.calls[0].id: "Cancelled."})

    assert isinstance(step, Say)
    assert consulted == []


def test_a_turn_still_owing_a_change_does_consult_the_speaker():
    speaker, consulted = counted(allows)
    kernel(just_talks, speaker, planning(CHANGE)).send("t", f"cancel {SEEN_ID}")

    assert len(consulted) == 1


# --- what it does with a ruling ---------------------------------------------


def test_an_allowed_message_is_delivered_unchanged():
    """Most messages the speaker sees are correct -- asking the customer to agree is
    the policy working -- so allowing has to be the cheap, ordinary path."""
    step = kernel(just_talks, allows, planning(CHANGE)).send("t", f"cancel {SEEN_ID}")

    assert step == Say(text="Let me know if there is anything else.")


def test_a_held_message_sends_the_actor_back_and_the_work_gets_done():
    """The whole point: the turn ended by talking, and it should have ended by
    writing. The actor is returned to work and the call it was avoiding comes out."""
    step = kernel(talks_then_acts(), holds_once(), planning(CHANGE)).send("t", f"cancel {SEEN_ID}")

    assert isinstance(step, Act)
    assert [c.name for c in step.calls] == ["cancel_reservation"]


def test_the_correction_reaches_the_actor_as_an_instruction():
    """Not as a user message. `transcript` renders a UserPromptPart as "Customer:",
    so a correction delivered that way would put words in the customer's mouth that
    the gate and the speaker would then judge the actor against."""
    seen: list[str] = []

    def actor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions or "")
        if len(seen) > 1:
            call = ToolCallPart(
                "cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id=uuid4().hex[:8]
            )
            return ModelResponse(parts=[call])
        return ModelResponse(parts=[TextPart("Anything else?")])

    kernel(actor, always_holds, planning(CHANGE)).send("t", f"cancel {SEEN_ID}")

    assert "Call cancel_reservation now." in seen[1]
    assert "Call cancel_reservation now." not in seen[0]


def test_the_actor_is_not_held_twice_in_one_turn():
    """An actor that comes back with the same message after being told once has
    nothing else to give, and the customer is waiting through every round."""
    speaker, consulted = counted(always_holds)
    step = kernel(just_talks, speaker, planning(CHANGE)).send("t", f"cancel {SEEN_ID}")

    assert len(consulted) == DEFERRAL_LIMIT
    assert isinstance(step, Say)


def test_a_check_that_never_answers_lets_the_message_through():
    """Fails open, unlike the gate. A refusal here cannot produce the write it is
    asking for -- it only stops the customer being answered."""
    step = kernel(just_talks, fumbles, planning(CHANGE)).send("t", f"cancel {SEEN_ID}")

    assert isinstance(step, Say)
    assert UNCHECKED.allowed


# --- where it does not sit --------------------------------------------------


def test_an_escalated_turn_bypasses_the_speaker():
    """Escalation exists to end a turn the gate would not let continue. A check whose
    only power is to send the actor back to work has nothing to say about it, and
    holding that reply would loop the turn between the two things that stop it."""

    def blocks(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _output(info, allowed=False, reason="Not permitted.", remediation="Do not do this.")

    def keeps_trying(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Once escalation has stripped the tools, proposing another call would
        # raise and take the whole run down -- so this stand-in reads the same cue
        # the real actor does and stops.
        latest = [p for p in messages[-1].parts if p.part_kind == "retry-prompt"]
        if latest and "tell the customer" in str(latest[-1].content):
            return ModelResponse(parts=[TextPart("I cannot do that. Can we look at options?")])
        call = ToolCallPart(
            "cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id=uuid4().hex[:8]
        )
        return ModelResponse(parts=[call])

    speaker, consulted = counted(always_holds)
    step = kernel(keeps_trying, speaker, planning(CHANGE), gate=blocks).send(
        "t", f"cancel {SEEN_ID}"
    )

    assert isinstance(step, Say)
    assert consulted == []


# --- the measurement instrument ---------------------------------------------


def test_an_allowed_message_still_records_that_the_check_ran():
    """The hole the gate had. A verdict nobody can count is a verdict nobody reads:
    last time, an allowed action and an action never proposed both left no trace,
    and the real block rate had to be recovered from Langfuse afterwards."""
    k = kernel(just_talks, allows, planning(CHANGE))
    k.send("t", f"cancel {SEEN_ID}")

    state = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert state["consulted"] == 1
    assert state.get("holds", 0) == 0


def test_a_held_message_is_counted_separately_from_a_ruling():
    k = kernel(just_talks, always_holds, planning(CHANGE))
    k.send("t", f"cancel {SEEN_ID}")

    state = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert state["consulted"] == 1
    assert state["holds"] == 1


def test_a_turn_the_speaker_never_saw_counts_nothing():
    k = kernel(just_talks, allows, planning())
    k.send("t", "what is your baggage policy?")

    state = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert state.get("consulted", 0) == 0
