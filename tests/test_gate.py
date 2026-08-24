"""GATE: what it fires on, what it does with a refusal, and where it stops.

No network. The actor and the critic are separate scripted stand-ins, which is
the point of several of these tests -- a single model playing both roles could
not show that the gate is a real second opinion.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.gate import (
    CAVEAT,
    FALLBACK,
    INSTRUCTIONS,
    NARROW,
    NO_DEMANDS,
    NO_FINDINGS,
    NO_PROVENANCE,
    NOTHING_OWED,
    OUTPUT_RETRIES,
    UNAVAILABLE,
    build_gate,
    decide,
    demands,
    findings,
    provenance,
    review,
    transcript,
)
from core.kernel import REVISION_LIMIT, Act, Kernel, Say
from core.state import Change, Demand, PendingCall, _windows, mispriced
from tests.tools import CANCEL, LOOKUP, PLANNER

SEEN_ID = "HKD3PS"
INVENTED_ID = "H0000X"


# --- scripted models --------------------------------------------------------


def _verdict(info: AgentInfo, **payload: object) -> ModelResponse:
    """Answer with the gate's output type, whatever pydantic-ai named it."""
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


def approves(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _verdict(info, allowed=True, reason="The policy permits this.")


def blocks_once():
    """Refuse the first proposal, accept the second -- the intended correction loop.

    A closure because the gate starts each review with no history of its own: it
    cannot tell a first look from a second by reading its messages, which is
    itself the point of the design.
    """
    seen: list[int] = []

    def critic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(1)
        if len(seen) > 1:
            return _verdict(info, allowed=True, reason="The reservation was looked up.")
        return _verdict(
            info,
            allowed=False,
            reason="The reservation was never looked up.",
            remediation="Look it up first.",
        )

    return critic


def always_blocks(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _verdict(
        info,
        allowed=False,
        reason="Basic economy cannot be modified.",
        remediation="Do not cancel this.",
    )


def blocks_without_a_fix(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A refusal with the remediation left empty -- unrepresentable while the
    output type was a union, and merely discouraged now that it is one model."""
    return _verdict(info, allowed=False, reason="Not allowed.")


def fumbles(times: int):
    """Answers in prose `times` times before filling the output tool in.

    The observed failure: a small model sometimes just talks instead of calling
    the tool it was asked to call.
    """
    seen: list[int] = []

    def critic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(1)
        if len(seen) <= times:
            return ModelResponse(parts=[TextPart("Looks fine to me.")])
        return approves(messages, info)

    return critic


def never_answers(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("Hmm.")])


def _retries(messages: list[ModelMessage]) -> list[RetryPromptPart]:
    return [p for m in messages for p in m.parts if isinstance(p, RetryPromptPart)]


def proposes_a_cancellation(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Ask to cancel; if corrected, ask again with an id that was actually seen."""
    parts = [p for m in messages for p in m.parts]
    returned = [p for p in parts if isinstance(p, ToolReturnPart)]
    if returned:
        return ModelResponse(parts=[TextPart(f"Done: {returned[-1].content}")])
    retries = _retries(messages)
    call = ToolCallPart(
        "cancel_reservation",
        {"reservation_id": SEEN_ID if retries else INVENTED_ID},
        tool_call_id=f"c{len(retries)}",
    )
    return ModelResponse(parts=[call])


def cites_a_reason_nobody_gave(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A grounded identifier with an ungrounded justification attached.

    The toolset would refuse an invented id outright, so this is what an
    ungrounded value looks like by the time the gate is the one deciding.
    """
    if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
        return ModelResponse(parts=[TextPart("Done.")])
    call = ToolCallPart(
        "cancel_reservation",
        {"reservation_id": SEEN_ID, "reason": "storm damage"},
        tool_call_id="c0",
    )
    return ModelResponse(parts=[call])


def proposes_a_lookup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
        return ModelResponse(parts=[TextPart("Found it.")])
    call = ToolCallPart("get_reservation", {"reservation_id": SEEN_ID}, tool_call_id="r1")
    return ModelResponse(parts=[call])


def refuses_to_give_up(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Keeps proposing the same forbidden write until it is told to stop talking to
    the database and start talking to the customer."""
    latest = [p for p in messages[-1].parts if isinstance(p, RetryPromptPart)]
    if latest and "tell the customer" in str(latest[-1].content):
        return ModelResponse(parts=[TextPart("I cannot cancel that one. Can we look at options?")])
    call = ToolCallPart(
        "cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id=uuid4().hex[:8]
    )
    return ModelResponse(parts=[call])


def kernel(actor, critic) -> Kernel:
    return Kernel(
        [LOOKUP, CANCEL],
        policy="Cancel only after looking the reservation up.",
        model=FunctionModel(actor),
        gate_model=FunctionModel(critic),
        planner_model=PLANNER,
    )


def counted(behaviour):
    """A critic that records how many times it was consulted."""
    calls: list[int] = []

    def wrapped(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(1)
        return behaviour(messages, info)

    return wrapped, calls


# --- what the gate fires on -------------------------------------------------


def test_a_read_only_step_never_reaches_the_gate():
    """Reads cannot damage the scored database, so paying a critic to bless one is
    pure cost -- and cost is the argument against gating at all."""
    critic, consulted = counted(approves)
    step = kernel(proposes_a_lookup, critic).send("t", "check HKD3PS")

    assert isinstance(step, Act)
    assert consulted == []


def test_a_write_reaches_the_gate():
    critic, consulted = counted(approves)
    kernel(proposes_a_cancellation, critic).send("t", "cancel HKD3PS")

    assert len(consulted) == 1


def test_an_unlabelled_tool_is_treated_as_a_write():
    """A domain we have not met may hand over a tool with no label. Gating a read by
    mistake costs one model call; missing a write costs the task."""
    critic, consulted = counted(approves)
    unlabelled = CANCEL.__class__(
        name="cancel_reservation",
        description=CANCEL.description,
        parameters_json_schema=CANCEL.parameters_json_schema,
    )
    k = Kernel(
        [unlabelled],
        policy="Be careful.",
        model=FunctionModel(proposes_a_cancellation),
        gate_model=FunctionModel(critic),
        planner_model=PLANNER,
    )
    k.send("t", f"cancel {SEEN_ID}")

    assert len(consulted) == 1


# --- what it does with a verdict --------------------------------------------


def test_an_approved_write_is_emitted_exactly_as_proposed():
    """The proposal is held apart from the pending call so that nothing between
    approval and emission can rewrite it. Approving one action and performing
    another is not an approval."""
    step = kernel(proposes_a_cancellation, approves).send("t", f"cancel {SEEN_ID}")

    assert isinstance(step, Act)
    assert [(c.name, c.arguments) for c in step.calls] == [
        ("cancel_reservation", {"reservation_id": SEEN_ID})
    ]


def test_a_blocked_write_is_not_emitted_and_the_corrected_one_is():
    k = kernel(proposes_a_cancellation, blocks_once())
    step = k.send("t", "cancel HKD3PS")

    assert isinstance(step, Act)
    assert step.calls[0].arguments == {"reservation_id": SEEN_ID}


def test_the_actor_is_told_why_and_what_to_do_instead():
    """A refusal with no remediation just produces the same proposal again."""
    k = kernel(proposes_a_cancellation, blocks_once())
    k.send("t", "cancel HKD3PS")

    history = k.graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"]
    rejection = str(history)
    assert "The reservation was never looked up." in rejection
    assert "Look it up first." in rejection


def test_the_whole_step_is_blocked_not_just_the_write():
    """The gate judges a plan. Letting the harmless half through would leave the
    actor re-planning from a position it never chose."""

    def proposes_both(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _retries(messages):
            return ModelResponse(parts=[TextPart("Understood.")])
        return ModelResponse(
            parts=[
                ToolCallPart("get_reservation", {"reservation_id": SEEN_ID}, tool_call_id="r1"),
                ToolCallPart("cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id="c1"),
            ]
        )

    step = kernel(proposes_both, always_blocks).send("t", "cancel HKD3PS")

    assert isinstance(step, Say)


# --- where it stops ---------------------------------------------------------


def test_a_turn_that_keeps_being_blocked_ends_by_talking_to_the_user():
    """The baseline produced four simulations that never terminated. An unbounded
    correction loop is the obvious way to produce more, so the loop is capped and
    the cap fails toward the customer rather than toward the database."""
    critic, consulted = counted(always_blocks)
    step = kernel(refuses_to_give_up, critic).send("t", "cancel HKD3PS")

    assert isinstance(step, Say)
    assert step.text.startswith("I cannot cancel that one")
    assert len(consulted) == REVISION_LIMIT + 1


def test_a_refused_call_never_enters_the_provenance_ledger():
    """It was never executed, so it showed us nothing. Letting a denial count as
    evidence would let the actor ground its next argument in its own invention."""
    k = kernel(refuses_to_give_up, always_blocks)
    k.send("t", "cancel HKD3PS")

    observed = k.graph.get_state({"configurable": {"thread_id": "t"}}).values["observed"]
    assert observed == ["cancel HKD3PS"]


def test_the_revision_budget_is_per_turn_not_per_conversation():
    """A customer who says something new has earned a fresh attempt."""
    critic, consulted = counted(always_blocks)
    k = kernel(refuses_to_give_up, critic)
    k.send("t", "cancel HKD3PS")
    k.send("t", "please try again")

    assert len(consulted) == 2 * (REVISION_LIMIT + 1)


# --- PRE-GATE ---------------------------------------------------------------


def test_pre_gate_flags_a_value_that_was_never_shown():
    call = PendingCall(id="c1", name="cancel_reservation", arguments={"reservation_id": "H0000X"})
    report = findings([call], ["your reservation is HKD3PS"])

    assert "`reservation_id`" in report
    assert "lead to check, not as proof" in report


def test_pre_gate_stays_quiet_when_everything_was_seen():
    call = PendingCall(id="c1", name="cancel_reservation", arguments={"reservation_id": "HKD3PS"})

    assert findings([call], ["your reservation is HKD3PS"]) == NO_FINDINGS


def test_pre_gate_findings_reach_the_gate():
    """Evidence, not a verdict: the gate is told, and the gate decides."""
    seen: list[str] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(str(messages[-1].parts[-1].content))
        return approves(messages, info)

    kernel(cites_a_reason_nobody_gave, record).send("t", f"cancel {SEEN_ID}")

    assert "appears nowhere in what the assistant has been shown" in seen[0]
    assert "reason='storm damage'" in seen[0]


def test_the_transcript_shows_the_lookups_not_just_the_talking():
    """The gate is usually checking whether a prerequisite was met, and
    prerequisites are met by looking things up."""
    k = kernel(proposes_a_lookup, approves)
    paused = k.send("t", "check HKD3PS")
    k.resume("t", {paused.calls[0].id: "economy, 1 bag"})

    messages = k.graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"]
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    rendered = transcript(ModelMessagesTypeAdapter.validate_python(messages))
    assert "Customer: check HKD3PS" in rendered
    assert "Assistant looks up: get_reservation(reservation_id='HKD3PS')" in rendered
    assert "Result: economy, 1 bag" in rendered


# --- when the gate itself fails ---------------------------------------------


def test_a_gate_that_fumbles_its_answer_is_asked_again():
    """One malformed reply used to be fatal, because the default budget is a
    single retry and a raise inside the gate ends the simulation."""
    step = kernel(proposes_a_cancellation, fumbles(OUTPUT_RETRIES)).send("t", f"cancel {SEEN_ID}")

    assert isinstance(step, Act)


def test_a_gate_whose_whole_call_fails_is_called_again():
    """The measured failure this retry exists for. Across the two 15-task runs the
    gate raised on 48% of the verdicts it was asked for, and most of those were the
    provider returning a completion with a null `id` that the client refuses to
    parse -- so a legitimate write was blocked by a bad packet. It arrives on the
    first call and costs nothing to retry.

    `fumbles(OUTPUT_RETRIES + 1)` exhausts the in-run retry budget once, which is
    what makes `run_sync` raise; the answer on the far side of that raise is the
    thing being tested.
    """
    step = kernel(proposes_a_cancellation, fumbles(OUTPUT_RETRIES + 1)).send(
        "t", f"cancel {SEEN_ID}"
    )

    assert isinstance(step, Act)


def test_a_refusal_that_names_no_fix_is_given_one():
    """`remediation` reaches the actor verbatim and is all it gets, so an empty one
    spends a revision on nothing. The union output type made this unrepresentable;
    with one output type it has to be repaired instead."""
    k = kernel(refuses_to_give_up, blocks_without_a_fix)
    k.send("t", "cancel HKD3PS")

    history = str(k.graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"])
    assert FALLBACK in history


def test_a_gate_that_never_answers_blocks_instead_of_raising():
    """Refusing is the only honest verdict on an action nobody checked -- and it
    costs one action, where the exception cost the whole task."""
    k = kernel(refuses_to_give_up, never_answers)
    step = k.send("t", "cancel HKD3PS")

    assert isinstance(step, Say)
    history = str(k.graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"])
    assert UNAVAILABLE.reason in history


def test_a_gate_that_never_answers_emits_no_write():
    k = kernel(refuses_to_give_up, never_answers)
    k.send("t", "cancel HKD3PS")

    assert k.graph.get_state({"configurable": {"thread_id": "t"}}).values["observed"] == [
        "cancel HKD3PS"
    ]


# --- the narrowed fallback --------------------------------------------------


def answers_only_the_narrow_question(allowed: bool):
    """Never fills the four-field verdict in, but can say yes or no.

    The measured failure this exists for: 38 of the 166 write refusals in the
    50-task run carried the words "the policy check did not complete", which is a
    20B model unable to compose the output object -- not a judgement about the
    action. Every one of them blocked something nobody had ruled on.
    """

    def critic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        asked = " ".join(
            str(p.content) for m in messages for p in m.parts if isinstance(p, UserPromptPart)
        )
        if NARROW in asked:
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"response": allowed})]
            )
        return ModelResponse(parts=[TextPart("I am not sure how to fill that in.")])

    return critic


def test_a_verdict_that_never_parses_is_asked_again_in_one_bit():
    gate = build_gate("Cancel freely.", FunctionModel(answers_only_the_narrow_question(True)))

    verdict = decide(gate, "cancel_reservation(reservation_id='HKD3PS')")

    assert verdict.allowed


def test_the_narrowed_check_can_still_refuse():
    """It is a fallback, not an amnesty. The point is that an answer is reached,
    not that the answer is yes."""
    gate = build_gate("Never cancel.", FunctionModel(answers_only_the_narrow_question(False)))

    verdict = decide(gate, "cancel_reservation(reservation_id='HKD3PS')")

    assert not verdict.allowed
    assert verdict.remediation.strip()


def test_an_action_stands_refused_when_neither_check_answers():
    """The floor is unchanged: refusing is still the only honest thing to say about
    an action nobody managed to look at."""
    gate = build_gate("Be careful.", FunctionModel(never_answers))

    assert decide(gate, "cancel_reservation(reservation_id='HKD3PS')") is UNAVAILABLE


# --- what the gate has already required -------------------------------------

CANCELLATION = [PendingCall(id="c1", name="cancel_reservation", arguments={})]

CONFIRM = Demand(
    action="cancel_reservation",
    reason="The policy requires the customer to confirm, and they have not.",
    turn=1,
)


def test_a_condition_the_customer_has_had_a_chance_to_answer_is_shown_back():
    """The gate has no memory of its own, and re-deriving 'have they confirmed?'
    from prose is what it gets wrong most: 70 of 166 write refusals in the 50-task
    run were the same demand made a second time."""
    rendered = demands(CANCELLATION, [CONFIRM], turn=2)

    assert "cancel_reservation" in rendered
    assert "confirm" in rendered
    assert "once" in rendered


def test_a_condition_imposed_on_the_turn_still_running_is_not_shown():
    """Nothing has happened since it was imposed. Showing it would be the gate
    arguing with itself, and inviting it to drop a condition it just set."""
    assert demands(CANCELLATION, [CONFIRM], turn=1) == NO_DEMANDS


def test_only_conditions_on_the_actions_being_proposed_are_shown():
    baggage = [PendingCall(id="c2", name="update_reservation_baggages", arguments={})]

    assert demands(baggage, [CONFIRM], turn=3) == NO_DEMANDS


def test_a_gate_that_has_required_nothing_says_so():
    assert demands(CANCELLATION, [], turn=2) == NO_DEMANDS


# --- where each identifier came from ----------------------------------------


def test_the_gate_is_shown_the_text_each_identifier_was_read_from():
    """The failure `findings` cannot see. Both reservations are in the ledger, so
    neither is invented -- the question is which record is about to be changed,
    and the only way to raise it is to quote where the value was read."""
    proposal = [
        PendingCall(id="c1", name="cancel_reservation", arguments={"reservation_id": "UM3OG5"})
    ]
    shown = ["Your bookings: FQ8APE (EWR to ORD, economy), UM3OG5 (LAS to DEN, basic economy)"]

    rendered = provenance(proposal, shown)

    assert "UM3OG5" in rendered
    assert "LAS to DEN" in rendered


def test_an_identifier_with_no_source_says_so():
    proposal = [
        PendingCall(id="c1", name="cancel_reservation", arguments={"reservation_id": INVENTED_ID})
    ]

    assert "nothing in the conversation" in provenance(proposal, ["unrelated text"])


def test_a_proposal_carrying_no_identifiers_says_so():
    proposal = [PendingCall(id="c1", name="send_certificate", arguments={"amount": 100})]

    assert provenance(proposal, ["anything"]) == NO_PROVENANCE


# --- entries written twice --------------------------------------------------


def test_an_entry_written_twice_reaches_the_gate_as_a_finding():
    """The toolset refuses these first, so this is the second line rather than the
    first. It is here because `findings` is the gate's whole view of what the
    deterministic pass saw, and a check missing from it reads as a check that
    passed."""
    proposal = [
        PendingCall(
            id="c1",
            name="update_reservation_passengers",
            arguments={
                "reservation_id": SEEN_ID,
                "passengers": [
                    {"first_name": "Omar", "dob": "1970-06-06"},
                    {"first_name": "Omar", "dob": "1970-06-06"},
                ],
            },
        )
    ]

    assert "same entry twice" in findings(proposal, [SEEN_ID, "Omar", "1970-06-06"])


# --- the demand ledger, across turns ----------------------------------------


def records_then_approves(cases: list[str]):
    """Refuse for want of confirmation once, then approve, keeping every case seen."""

    def critic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        cases.append(
            " ".join(
                str(p.content) for m in messages for p in m.parts if isinstance(p, UserPromptPart)
            )
        )
        if len(cases) > 1:
            return _verdict(info, allowed=True, reason="They have now confirmed.")
        return _verdict(
            info,
            allowed=False,
            reason="The policy requires the customer to confirm, and they have not.",
            remediation="Ask the customer to confirm the cancellation.",
        )

    return critic


def proposes_then_talks(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Propose the write; on a refusal just received, take it to the customer.

    The refusal has to be read off the *last* message, not the whole history: a
    turn later it is still in there, and an actor that never proposes again is one
    that cannot show the ledger reaching a second review.
    """
    if [p for p in messages[-1].parts if isinstance(p, RetryPromptPart)]:
        return ModelResponse(parts=[TextPart("Shall I go ahead and cancel it?")])
    call = ToolCallPart(
        "cancel_reservation", {"reservation_id": SEEN_ID}, tool_call_id=uuid4().hex[:8]
    )
    return ModelResponse(parts=[call])


def test_a_condition_imposed_last_turn_is_put_back_in_front_of_the_gate():
    """The gate is handed a transcript and asked to rule, so a condition it set one
    turn ago has to be re-derived from prose every time -- and 70 of the 166 write
    refusals in the 50-task run were it failing to and demanding the same thing
    again. Recording the demand turns that into something it is told."""
    cases: list[str] = []
    k = kernel(proposes_then_talks, records_then_approves(cases))

    k.send("t", f"cancel {SEEN_ID}")
    k.send("t", "yes please, go ahead")

    assert len(cases) == 2
    assert NO_DEMANDS in cases[0]
    assert "You refused cancel_reservation earlier" in cases[1]
    assert "replied once since" in cases[1]


def test_the_first_review_of_a_conversation_has_nothing_to_report():
    cases: list[str] = []
    kernel(proposes_a_cancellation, records_then_approves(cases)).send("t", f"cancel {SEEN_ID}")

    assert NO_DEMANDS in cases[0]


# --- what the turn still owes ------------------------------------------------


OWED = Change(tool="cancel_reservation", record="HKD3PS", what="cancel the reservation")


def _case(owed: list[str] | None) -> str:
    return review([], [PendingCall(id="1", name=CANCEL.name, arguments={})], [], owed=owed)


def test_the_gate_is_shown_what_the_turn_still_owes():
    """The run had the plan, the ledger and the speaker all correct and lost the
    task anyway: the handoff leaves through the gate, and the gate could see none
    of it."""
    assert "cancel_reservation on HKD3PS: cancel the reservation" in _case([OWED])


def test_a_turn_owing_nothing_says_so_rather_than_showing_a_blank():
    assert NOTHING_OWED in _case(None)
    assert NOTHING_OWED in _case([])


def test_the_gate_is_told_to_refuse_a_handoff_that_leaves_work_behind():
    """The ledger is only worth showing it if it knows what the ledger decides."""
    told = " ".join(INSTRUCTIONS.split())

    assert "A handoff proposed while a change is outstanding is refused." in told


def test_the_kernel_hands_the_gate_the_ledger_the_speaker_counts():
    """Wiring: the same `outstanding` call, not a second list that can drift."""
    cases: list[str] = []
    k = Kernel(
        [LOOKUP, CANCEL],
        policy="Cancel only after looking the reservation up.",
        model=FunctionModel(proposes_a_cancellation),
        gate_model=FunctionModel(records_then_approves(cases)),
        planner_model=FunctionModel(_plans_a_cancellation),
    )

    k.send("t", f"cancel {SEEN_ID}")

    assert f"cancel_reservation on {SEEN_ID}: cancel it" in cases[0]


def _plans_a_cancellation(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "goal": "The reservation is cancelled.",
                    "changes": [
                        {
                            "tool": "cancel_reservation",
                            "record": SEEN_ID,
                            "what": "cancel it",
                        }
                    ],
                },
            )
        ]
    )


# --- money attached to the wrong row -----------------------------------------


SEARCH = (
    '{"flight_number": "HAT139", "origin": "JFK", "destination": "SEA", '
    '"prices": {"basic_economy": 87, "economy": 114, "business": 401}}\n'
    '{"flight_number": "HAT271", "origin": "JFK", "destination": "SEA", '
    '"prices": {"basic_economy": 92, "economy": 174, "business": 445}}'
)


def test_a_price_taken_from_the_neighbouring_flight_is_flagged():
    """The booking the run lost: 114 is real, HAT271 is real, and 114 is not HAT271's."""
    call = {"flights": [{"flight_number": "HAT271", "price": 114}]}

    assert mispriced(call, [SEARCH]) == ["flights[0].price"]


def test_a_price_shown_beside_its_own_flight_passes():
    call = {"flights": [{"flight_number": "HAT271", "price": 174}]}

    assert mispriced(call, [SEARCH]) == []


def test_an_identifier_that_was_never_shown_is_left_to_the_other_check():
    """`invented` owns that finding; reporting it here counts one fault twice."""
    call = {"flights": [{"flight_number": "HAT999", "price": 114}]}

    assert mispriced(call, [SEARCH]) == []


def test_an_entry_with_no_money_in_it_is_not_examined():
    assert (
        mispriced({"flights": [{"flight_number": "HAT271", "date": "2024-05-22"}]}, [SEARCH]) == []
    )


def test_a_whole_number_is_matched_the_way_the_corpus_spells_it():
    """The model returns 174.0 where the search said 174."""
    call = {"flights": [{"flight_number": "HAT271", "price": 174.0}]}

    assert mispriced(call, [SEARCH]) == []


def test_each_leg_is_judged_against_its_own_flight():
    call = {
        "flights": [
            {"flight_number": "HAT139", "price": 114},
            {"flight_number": "HAT271", "price": 114},
        ]
    }

    assert mispriced(call, [SEARCH]) == ["flights[1].price"]


PROFILE = (
    '{"user_id": "mohamed_silva_9265", "payment_methods": {'
    '"gift_card_8020792": {"source": "gift_card", "id": "gift_card_8020792", "amount": 198.0}, '
    '"credit_card_2198526": {"source": "credit_card", "id": "credit_card_2198526", '
    '"brand": "mastercard", "last_four": "9363"}}}'
)


def test_what_is_charged_to_a_card_is_not_judged_as_a_price():
    """A payment amount is a remainder worked out from the rest of the basket, so
    it is never shown beside the instrument it is charged to. A card carries a
    brand and a last_four and no balance at all, so asking whether the sum appears
    beside it can only ever fail.

    Replayed against the benchmark's own gold actions this was the only thing the
    check ever flagged: three correct payment splits on task 23, for 44, 621 and
    621 -- which add up to 1286, the single figure that task is scored on."""
    call = {
        "reservation_id": "K1NW8N",
        "payment_methods": [
            {"payment_id": "certificate_3765853", "amount": 500},
            {"payment_id": "credit_card_2198526", "amount": 44},
        ],
    }

    assert mispriced(call, [PROFILE]) == []


def test_a_past_charge_to_the_same_card_does_not_make_the_next_one_wrong():
    """The ledger holds every earlier settlement, and none of them is evidence
    about this one."""
    history = '{"payment_history": [{"payment_id": "credit_card_2198526", "amount": 2628}]}'
    call = {"payment_methods": [{"payment_id": "credit_card_2198526", "amount": 44}]}

    assert mispriced(call, [PROFILE, history]) == []


def test_an_identifier_is_read_against_the_record_it_names():
    """`payment_methods` hangs each record off its own id, so walking outwards
    from the key lands on the object holding all of them -- and every balance in
    the block would count as shown beside every card. The window is the record the
    identifier names."""
    assert "last_four" in _windows("credit_card_2198526", [PROFILE])[0]
    assert "gift_card_8020792" not in _windows("credit_card_2198526", [PROFILE])[0]


def test_the_gate_is_shown_the_mispricing_as_evidence():
    call = PendingCall(
        id="1",
        name="book_reservation",
        arguments={"flights": [{"flight_number": "HAT271", "price": 114}]},
    )

    reported = findings([call], [SEARCH])

    assert "flights[0].price" in reported
    assert CAVEAT in reported


def test_the_critic_can_be_taken_out_without_taking_the_ledger_with_it(monkeypatch):
    """`STEWARD_GATE=off` is a measurement arm. It has to remove the critic's
    judgement and nothing else: if it also emptied the written ledger, the
    speaker would hold on every turn and the run would measure two changes at
    once, telling us about neither."""
    from core import kernel
    from core.state import StewardState

    call = {"id": "1", "name": "book_reservation", "arguments": {"reservation_id": "K1NW8N"}}
    state = StewardState(calls=[call])

    def never(*args, **kwargs):
        raise AssertionError("the critic was asked despite being turned off")

    monkeypatch.setattr(kernel, "REVIEWING", False)
    monkeypatch.setattr(kernel, "decide", never)

    out = kernel._gate(state, None, frozenset({"book_reservation"}))

    assert out["approved"] == [call]
    assert [w.tool for w in out["written"]] == ["book_reservation"]
    assert [w.records for w in out["written"]] == [["K1NW8N"]]


def test_a_read_only_step_still_writes_nothing_to_the_ledger_either_way(monkeypatch):
    """The bypass approves writes without asking; it must not turn a lookup into
    something the ledger thinks discharged a planned change."""
    from core import kernel
    from core.state import StewardState

    state = StewardState(calls=[{"id": "1", "name": "get_user_details", "arguments": {}}])
    monkeypatch.setattr(kernel, "REVIEWING", False)

    out = kernel._gate(state, None, frozenset({"book_reservation"}))

    assert out["approved"] == state.calls
    assert "written" not in out


def test_the_switch_is_off_only_when_asked(monkeypatch):
    """Read through a function rather than re-imported, because reloading the
    module rebuilds every class in it and silently breaks whatever already holds
    a reference to one."""
    from core import kernel

    for value, expected in (
        ("off", False),
        ("OFF", False),
        (" off ", False),
        ("on", True),
        ("", True),
    ):
        monkeypatch.setenv("STEWARD_GATE", value)
        assert kernel._reviewing() is expected
    monkeypatch.delenv("STEWARD_GATE")
    assert kernel._reviewing() is True
