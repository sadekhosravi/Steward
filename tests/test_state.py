"""The three ledgers. Pure functions and one graph-level check; no network."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from core.kernel import Kernel
from core.state import Obligation, StewardState, ungrounded, unmet
from tests.tools import LOOKUP, PLANNER

SEEN = ["My user id is mia_li_3668", '{"reservations": ["HKD3PS", "X4RTG9"]}']


def test_an_argument_the_system_was_shown_is_grounded():
    assert ungrounded({"reservation_id": "HKD3PS"}, SEEN) == []


def test_an_invented_argument_is_caught():
    """The failure this ledger exists for: a plausible id that came from nowhere."""
    assert ungrounded({"reservation_id": "HKD3P5"}, SEEN) == ["reservation_id"]


def test_grounding_reaches_inside_nested_arguments():
    """Write payloads nest, and an invented id one level down is just as fatal."""
    payload = {"passengers": [{"first_name": "Mia", "last_name": "Li"}]}
    assert ungrounded(payload, ["Mia Li flies tomorrow"]) == []
    assert ungrounded(payload, ["Mia flies tomorrow"]) == ["passengers"]


def test_numbers_and_flags_are_not_subject_to_provenance():
    """They carry no identity, so demanding they be 'seen' would reject everything."""
    assert ungrounded({"total_baggages": 2, "insurance": True}, SEEN) == []


def test_nothing_observed_means_nothing_is_grounded():
    assert ungrounded({"reservation_id": "HKD3PS"}, []) == ["reservation_id"]


TRANSFER = Obligation(
    id="transfer-notice",
    description="Tell the user they are being transferred, in the exact words.",
    must_contain="YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
)


def test_an_obligation_is_discharged_by_the_exact_wording():
    assert unmet([TRANSFER], TRANSFER.must_contain) == []


def test_a_paraphrase_does_not_discharge_an_obligation():
    """Policies that dictate wording mean it; 'roughly that' is a scored failure."""
    assert unmet([TRANSFER], "You are being transferred to a human agent, hold on.") == [TRANSFER]


def test_an_obligation_with_no_wording_cannot_be_cleared_by_replying():
    owed = Obligation(id="x", description="Read the reservation before cancelling.")
    assert unmet([owed], "anything at all") == [owed]


def test_the_ledgers_start_empty():
    """A fresh conversation owes nothing and has seen nothing."""
    state = StewardState()
    assert (state.approved, state.observed, state.obligations) == ([], [], [])


def call_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if not [p for m in messages for p in m.parts if isinstance(p, ToolCallPart)]:
        call = ToolCallPart("get_reservation", {"reservation_id": "HKD3PS"}, tool_call_id="c1")
        return ModelResponse(parts=[call])
    return ModelResponse(parts=[TextPart("Done.")])


def test_user_messages_and_tool_results_both_enter_the_ledger():
    """Provenance is only as good as its intake: both sources have to land."""
    k = Kernel(
        [LOOKUP],
        policy="Be helpful.",
        model=FunctionModel(call_then_report),
        planner_model=PLANNER,
    )
    thread = k.new_thread()

    paused = k.send(thread, "check HKD3PS please")
    k.resume(thread, {paused.calls[0].id: "HKD3PS: economy, 1 bag"})

    observed = k.graph.get_state({"configurable": {"thread_id": thread}}).values["observed"]
    assert observed == ["check HKD3PS please", "HKD3PS: economy, 1 bag"]
