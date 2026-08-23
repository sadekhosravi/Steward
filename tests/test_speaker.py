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
from core.state import Change, Written
from tests.tools import CANCEL, LOOKUP

SEEN_ID = "HKD3PS"
CHANGE = Change(tool="cancel_reservation", record=SEEN_ID, what="cancel the booking")


# --- scripted models --------------------------------------------------------


def _output(info: AgentInfo, **payload: object) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


def planning(*changes: Change):
    """A planner that names these changes and nothing else."""

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _output(
            info,
            goal="Do what they asked.",
            changes=[c.model_dump() for c in changes],
        )

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


def test_a_change_whose_tool_and_record_were_approved_is_done():
    assert outstanding([CHANGE], [Written(tool="cancel_reservation", records=[SEEN_ID])]) == []


def test_a_change_whose_tool_was_never_approved_is_outstanding():
    assert outstanding([CHANGE], [Written(tool="update_reservation_baggages")]) == [CHANGE]


def test_the_same_tool_on_another_record_does_not_discharge_it():
    """The defect this function existed to have and did not. A request covering
    six reservations was satisfied, as far as the old substring test could tell,
    by writing to one of them -- and across four runs no task needing writes to
    more than one record was ever completed."""
    approved = [Written(tool="cancel_reservation", records=["4WQ150"])]
    assert outstanding([CHANGE], approved) == [CHANGE]


def test_each_record_is_discharged_by_its_own_call():
    changes = [
        Change(tool="cancel_reservation", record="AAA111", what="cancel it"),
        Change(tool="cancel_reservation", record="BBB222", what="cancel it"),
        Change(tool="cancel_reservation", record="CCC333", what="cancel it"),
    ]
    approved = [Written(tool="cancel_reservation", records=["AAA111"])]
    assert outstanding(changes, approved) == changes[1:]


def test_only_the_unmet_changes_are_outstanding():
    """A task needing three writes is not finished by one, and the speaker is told
    which two are left rather than that something is missing."""
    changes = [
        Change(tool="update_reservation_baggages", record=SEEN_ID, what="add a bag"),
        Change(tool="update_reservation_passengers", record=SEEN_ID, what="change who flies"),
        Change(tool="update_reservation_flights", record=SEEN_ID, what="upgrade the cabin"),
    ]
    approved = [Written(tool="update_reservation_baggages", records=[SEEN_ID])]
    assert outstanding(changes, approved) == changes[1:]


def test_a_change_with_no_record_falls_back_to_the_tool_alone():
    """A booking creates the record it is about, so there is nothing to match on.
    The tool name is the only answer available, and it is the old behaviour."""
    booking = Change(tool="book_reservation", what="book the new trip")
    assert outstanding([booking], [Written(tool="book_reservation", records=["NEW999"])]) == []
    assert outstanding([booking], [Written(tool="cancel_reservation")]) == [booking]


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


# --- a refusal the assistant could have fixed itself ------------------------


def blocks_once_over_something_fixable():
    """Refuse the first attempt, naming a fix that needs nobody, then allow it.

    `recoverable` is the whole of the difference. Without it the actor reads any
    remediation as an instruction and carries it all the way to the customer: 146
    of the 203 refusals in the 50-task run ended the turn in talk rather than in a
    second attempt, and the write never happened.
    """
    seen: list[int] = []

    def critic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(1)
        if len(seen) > 1:
            return _output(info, allowed=True, reason="The arguments are right now.")
        return _output(
            info,
            allowed=False,
            reason="The itinerary lists the same flight twice.",
            remediation="Call cancel_reservation with the id from the lookup.",
            recoverable=True,
        )

    return critic


def blocks_pending_the_customer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _output(
        info,
        allowed=False,
        reason="The customer has not agreed to the cancellation fee.",
        remediation="Tell them the fee is 50 dollars and ask them to confirm.",
        recoverable=False,
    )


def gives_up_then_acts():
    """Reply after being refused -- the behaviour being corrected -- then call."""
    seen: list[int] = []

    def actor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(1)
        if len(seen) == 2:
            return ModelResponse(parts=[TextPart("Sorry, I could not make that change.")])
        call = ToolCallPart(
            "cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id=uuid4().hex[:8]
        )
        return ModelResponse(parts=[call])

    return actor


def test_a_reply_after_a_fixable_refusal_is_held_without_asking_the_speaker():
    """The economy argument, applied to a case the gate has already ruled on. There
    is no judgement left to buy: the gate said the assistant was not waiting on
    anybody, so a reply here is it walking away from work it was told it could do."""
    speaker, consulted = counted(allows)
    k = kernel(
        gives_up_then_acts(), speaker, planning(CHANGE), gate=blocks_once_over_something_fixable()
    )

    step = k.send("t", f"cancel {SEEN_ID}")

    assert isinstance(step, Act)
    assert [c.name for c in step.calls] == ["cancel_reservation"]
    assert consulted == []


def test_the_held_reply_is_counted_like_any_other_hold():
    """It bypasses the model, not the instruments. A hold nobody can count is the
    hole the gate had, and skipping the counters would dig it again."""
    k = kernel(
        gives_up_then_acts(), allows, planning(CHANGE), gate=blocks_once_over_something_fixable()
    )
    k.send("t", f"cancel {SEEN_ID}")

    state = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert state["holds"] == 1
    assert state["fixable"] == ""


def test_a_refusal_waiting_on_the_customer_does_not_hold_the_reply():
    """The other half, and the one that must not regress. When the gate's fix needs
    the customer, ending the turn is the turn ending correctly -- holding it would
    loop the actor against a condition only the customer can clear."""
    speaker, consulted = counted(allows)
    k = kernel(just_talks, speaker, planning(CHANGE), gate=blocks_pending_the_customer)

    step = k.send("t", f"cancel {SEEN_ID}")

    assert isinstance(step, Say)
    assert len(consulted) == 1


def test_an_approved_action_clears_a_fix_left_over_from_an_earlier_refusal():
    """Otherwise the next reply of the turn is held over a refusal already answered."""
    k = kernel(cancels_then_talks, allows, planning(CHANGE), gate=approves_writes)
    step = k.send("t", f"cancel {SEEN_ID}")
    k.resume("t", {step.calls[0].id: "Cancelled."})

    state = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert state["fixable"] == ""
    assert state.get("holds", 0) == 0


# --- the ledger across a customer turn ---------------------------------------


def _forgets_after_the_first_turn():
    """A planner that names both records once and then, on the customer's next
    message, re-plans as though only the record it just read were in scope.

    Not a strawman: this is task 18 out of the last full run, where `changes`
    went from "update_reservation_flights (once per reservation to change cabin
    to economy)" to `["update_reservation_flights"]` between two consecutive
    plans, with nothing in between but the customer saying yes.
    """
    seen: list[int] = []

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(1)
        both = [
            Change(tool="cancel_reservation", record="AAA111", what="cancel it").model_dump(),
            Change(tool="cancel_reservation", record="BBB222", what="cancel it").model_dump(),
        ]
        return _output(
            info,
            goal="Both bookings are cancelled.",
            changes=both if len(seen) == 1 else both[:1],
        )

    return planner


def test_a_commitment_survives_the_customer_speaking_again():
    """The defect Part 2 exists for. The plan is rewritten from scratch every time
    the customer speaks, so a request covering two records became a request
    covering one -- and the second was owed by nobody for the rest of the task."""
    k = kernel(just_talks, allows, _forgets_after_the_first_turn())

    k.send("t", "cancel both my bookings")
    k.send("t", "yes please, go ahead")

    values = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert [c.record for c in values["changes"]] == ["AAA111", "BBB222"]


def test_an_approved_write_is_still_remembered_next_turn():
    """The other half. A ledger of what is owed is worth nothing beside a ledger
    of what is done that resets underneath it -- the second turn would re-owe the
    write the first one had already made."""
    k = kernel(cancels_then_talks, allows, planning(CHANGE))

    step = k.send("t", f"cancel {SEEN_ID}")
    assert isinstance(step, Act)
    k.resume("t", {step.calls[0].id: "cancelled"})
    k.send("t", "thanks")

    values = k.graph.get_state({"configurable": {"thread_id": "t"}}).values
    assert [w.tool for w in values["written"]] == ["cancel_reservation"]
    assert outstanding(values["changes"], values["written"]) == []


def test_a_record_named_in_prose_is_still_matched():
    """Told to name the record, the planner does not always hand back a bare id:
    across five samples it answered `JG7FMM`, `@JG7FMM_reservation_id` and "the
    reservation id from get_reservation_details for JG7FMM". All three name the
    same record, and an exact match would have discharged none of them."""
    approved = [Written(tool="cancel_reservation", records=["JG7FMM"])]
    for spelling in (
        "JG7FMM",
        "@JG7FMM_reservation_id",
        "the reservation id from get_reservation_details for JG7FMM",
    ):
        change = Change(tool="cancel_reservation", record=spelling, what="cancel it")
        assert outstanding([change], approved) == []


def test_a_record_named_in_prose_does_not_match_a_different_one():
    approved = [Written(tool="cancel_reservation", records=["4WQ150"])]
    change = Change(tool="cancel_reservation", record="@JG7FMM_reservation_id", what="cancel it")
    assert outstanding([change], approved) == [change]


def test_a_commitment_the_customer_never_withdrew_cannot_wedge_the_turn():
    """The risk carrying the ledger introduces. A change nobody retires stays
    outstanding for the rest of the conversation, so the bound on what that can
    cost has to be the deferral budget and nothing else: one hold per turn, then
    the message goes."""
    speaker, consulted = counted(always_holds)
    k = kernel(just_talks, speaker, planning(CHANGE))

    for _ in range(3):
        step = k.send("t", f"cancel {SEEN_ID}")
        # Every turn still ends by talking to the customer. The commitment is
        # carried, not enforced.
        assert isinstance(step, Say)

    # One consult per turn and no more: the second attempt is over the deferral
    # budget and leaves without asking. So an unretired commitment costs one model
    # call a turn, for as long as it stands, and can never stop the conversation.
    assert len(consulted) == 3 * DEFERRAL_LIMIT
