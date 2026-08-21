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
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.gate import NO_FINDINGS, OUTPUT_RETRIES, UNAVAILABLE, findings, transcript
from core.kernel import REVISION_LIMIT, Act, Kernel, Say
from core.state import PendingCall
from tests.tools import CANCEL, LOOKUP

SEEN_ID = "HKD3PS"
INVENTED_ID = "H0000X"


# --- scripted models --------------------------------------------------------


def _verdict(info: AgentInfo, kind: str, payload: dict) -> ModelResponse:
    """Answer with one of the gate's output types, whatever pydantic-ai named it."""
    tool = next(t for t in info.output_tools if t.name.endswith(kind))
    return ModelResponse(parts=[ToolCallPart(tool.name, payload)])


def approves(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _verdict(info, "Approved", {"reason": "The policy permits this."})


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
            return _verdict(info, "Approved", {"reason": "The reservation was looked up."})
        return _verdict(
            info,
            "Blocked",
            {
                "violation": "The reservation was never looked up.",
                "remediation": "Look it up first.",
            },
        )

    return critic


def always_blocks(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return _verdict(
        info,
        "Blocked",
        {"violation": "Basic economy cannot be modified.", "remediation": "Do not cancel this."},
    )


def fumbles(times: int):
    """Answers in prose `times` times before choosing an output tool.

    The observed failure: a union output type gives the model two output tools to
    pick between, and a small one sometimes just talks instead.
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
    )
    k.send("t", "cancel it")

    assert len(consulted) == 1


# --- what it does with a verdict --------------------------------------------


def test_an_approved_write_is_emitted_exactly_as_proposed():
    """The proposal is held apart from the pending call so that nothing between
    approval and emission can rewrite it. Approving one action and performing
    another is not an approval."""
    step = kernel(proposes_a_cancellation, approves).send("t", "cancel it")

    assert isinstance(step, Act)
    assert [(c.name, c.arguments) for c in step.calls] == [
        ("cancel_reservation", {"reservation_id": INVENTED_ID})
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

    kernel(proposes_a_cancellation, record).send("t", "cancel my flight")

    assert "appears nowhere in what the assistant has been shown" in seen[0]
    assert "cancel_reservation(reservation_id='H0000X')" in seen[0]


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
    step = kernel(proposes_a_cancellation, fumbles(OUTPUT_RETRIES)).send("t", "cancel it")

    assert isinstance(step, Act)


def test_a_gate_that_never_answers_blocks_instead_of_raising():
    """Refusing is the only honest verdict on an action nobody checked -- and it
    costs one action, where the exception cost the whole task."""
    k = kernel(refuses_to_give_up, never_answers)
    step = k.send("t", "cancel HKD3PS")

    assert isinstance(step, Say)
    history = str(k.graph.get_state({"configurable": {"thread_id": "t"}}).values["messages"])
    assert UNAVAILABLE.violation in history


def test_a_gate_that_never_answers_emits_no_write():
    k = kernel(refuses_to_give_up, never_answers)
    k.send("t", "cancel HKD3PS")

    assert k.graph.get_state({"configurable": {"thread_id": "t"}}).values["observed"] == [
        "cancel HKD3PS"
    ]
