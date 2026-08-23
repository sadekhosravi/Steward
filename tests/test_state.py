"""The three ledgers. Pure functions and one graph-level check; no network."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from core.kernel import Kernel
from core.state import Obligation, StewardState, duplicated, pruned, sources, ungrounded, unmet
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


# --- entries written twice --------------------------------------------------


def test_the_same_passenger_listed_twice_is_a_repeat():
    """A real loss: a booking whose passenger list named one person two times."""
    assert duplicated(
        {
            "passengers": [
                {"first_name": "Omar", "last_name": "Rossi", "dob": "1970-06-06"},
                {"first_name": "Omar", "last_name": "Rossi", "dob": "1970-06-06"},
            ]
        }
    ) == ["passengers[0] and passengers[1]"]


def test_a_leg_standing_in_for_its_own_return_is_a_repeat():
    """The harder half, and the reason entries are compared on their identifiers
    rather than whole. This itinerary's two legs differ in date and in price and
    are still the same flight, in the same direction, twice."""
    assert duplicated(
        {
            "flights": [
                {"flight_number": "HAT169", "date": "2024-05-17", "price": 171},
                {"flight_number": "HAT169", "date": "2024-05-19", "price": 124},
            ]
        }
    ) == ["flights[0] and flights[1]"]


def test_two_different_flights_are_not_a_repeat():
    assert (
        duplicated(
            {
                "flights": [
                    {"flight_number": "HAT169", "date": "2024-05-17"},
                    {"flight_number": "HAT033", "date": "2024-05-19"},
                ]
            }
        )
        == []
    )


def test_a_list_of_plain_values_is_left_alone():
    """Two identical strings in a bare list can be meant, and guessing costs a retry."""
    assert duplicated({"tags": ["window", "window"], "total_baggages": 2}) == []


# --- where a value came from ------------------------------------------------


def test_an_identifier_is_quoted_with_the_text_it_came_from():
    """The check `invented` cannot make. Both ids are in the ledger, so both pass
    it -- the question is which record the assistant is about to change, and the
    only way to raise it is to show where the value was read."""
    shown = ["Your reservations: FQ8APE (EWR to ORD, economy), UM3OG5 (LAS to DEN, basic)"]

    (path, value, snippet) = sources({"reservation_id": "UM3OG5"}, shown)[0]

    assert (path, value) == ("reservation_id", "UM3OG5")
    assert "LAS to DEN" in snippet


def test_an_identifier_that_was_never_shown_has_no_source():
    assert sources({"reservation_id": "H0000X"}, ["nothing relevant"]) == [
        ("reservation_id", "H0000X", "")
    ]


def test_only_identifiers_are_traced():
    """Same narrowing as `invented`: a date or a name can be legitimately rewritten,
    so quoting where it 'came from' would be quoting a coincidence."""
    assert sources({"first_name": "Sophia", "total_baggages": 2}, ["Sophia Silva"]) == []


# The shape tau2 produces for a nested model, after `schemas.tighten` has
# collapsed its "or any object at all" branch: a `$ref` into `$defs`.
ITINERARY = {
    "$defs": {
        "FlightInfo": {
            "type": "object",
            "properties": {
                "flight_number": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["flight_number", "date"],
        }
    },
    "type": "object",
    "properties": {
        "reservation_id": {"type": "string"},
        "cabin": {"type": "string"},
        "flights": {"type": "array", "items": {"$ref": "#/$defs/FlightInfo"}},
    },
    "required": ["reservation_id", "cabin", "flights"],
}


def test_pruned_drops_undeclared_keys_inside_a_referenced_item():
    """The failure this exists for: a correct itinerary, decorated."""
    call = {
        "reservation_id": "JG7FMM",
        "cabin": "economy",
        "flights": [
            {
                "flight_number": "HAT056",
                "date": "2024-05-27",
                "origin": "LGA",
                "destination": "ORD",
                "price": 146,
            }
        ],
    }
    assert pruned(call, ITINERARY) == {
        "reservation_id": "JG7FMM",
        "cabin": "economy",
        "flights": [{"flight_number": "HAT056", "date": "2024-05-27"}],
    }


def test_pruned_leaves_a_call_that_was_already_right_alone():
    call = {
        "reservation_id": "JG7FMM",
        "cabin": "economy",
        "flights": [{"flight_number": "HAT056", "date": "2024-05-27"}],
    }
    assert pruned(call, ITINERARY) == call


def test_pruned_drops_an_undeclared_top_level_argument():
    call = {"reservation_id": "JG7FMM", "cabin": "economy", "flights": [], "total": 146}
    assert "total" not in pruned(call, ITINERARY)


def test_pruned_keeps_everything_when_there_is_no_schema():
    """An unknown tool is not an excuse to throw arguments away."""
    call = {"anything": {"at": "all"}}
    assert pruned(call, {}) == call


def test_pruned_keeps_everything_under_a_choice_of_shapes():
    """Picking a branch could delete a key that belonged to the other one."""
    schema = {
        "type": "object",
        "properties": {
            "payment": {
                "anyOf": [
                    {"type": "object", "properties": {"card": {"type": "string"}}},
                    {"type": "object", "properties": {"certificate": {"type": "string"}}},
                ]
            }
        },
    }
    call = {"payment": {"certificate": "certificate_7815826"}}
    assert pruned(call, schema) == call


def test_pruned_keeps_everything_where_the_schema_allows_extras():
    schema = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "additionalProperties": True,
    }
    call = {"note": "hello", "extra": 1}
    assert pruned(call, schema) == call
