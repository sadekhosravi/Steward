"""The three ledgers. Pure functions and one graph-level check; no network."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from core.kernel import Kernel
from core.state import (
    ANY,
    Change,
    Demand,
    Obligation,
    StewardState,
    anchored,
    answered,
    duplicated,
    misfiled,
    performable,
    pruned,
    sources,
    ungrounded,
    unmet,
)
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


# --- the customer answering a condition -------------------------------------


def _standing(turn: int = 1) -> list[Demand]:
    return [Demand(action="cancel_reservation", reason="the customer must confirm", turn=turn)]


def test_a_plain_yes_retires_the_demand_into_consent():
    standing, given = answered(_standing(), "yes, go ahead", turn=2)

    assert standing == []
    assert [c.action for c in given] == ["cancel_reservation"]
    assert given[0].words == "yes, go ahead"
    assert given[0].turn == 2


def test_the_customer_s_words_are_kept_verbatim():
    """A paraphrase of a consent is the one thing nobody should be asked to trust."""
    _, given = answered(_standing(), "Okay. Please do it.", turn=2)

    assert given[0].words == "Okay. Please do it."


def test_a_qualified_yes_is_not_an_answer():
    """Every one of these carries a yes and none of them is agreement to this."""
    for reply in [
        "yes, but not that one",
        "yes -- actually, wait",
        "ok, hold on",
        "sure, though not yet",
        "yes, cancel the other one instead",
    ]:
        standing, given = answered(_standing(), reply, turn=2)

        assert given == [], reply
        assert standing == _standing(), reply


def test_a_refusal_is_not_an_answer():
    for reply in ["no thanks", "please don't", "no, leave it"]:
        _, given = answered(_standing(), reply, turn=2)

        assert given == [], reply


def test_agreement_is_matched_on_whole_words():
    """`ok` must not be found inside `book`, and `no` must not be found in `now`."""
    _, given = answered(_standing(), "book it now", turn=2)

    assert given == []


def test_a_demand_made_this_turn_is_not_answered_by_the_message_that_provoked_it():
    """It has not been put to the customer yet, so a yes in that message agreed to
    something else."""
    standing, given = answered(_standing(turn=2), "yes please", turn=2)

    assert given == []
    assert standing == _standing(turn=2)


# --- the customer answering the assistant, which no demand ever covered ------


ASKED = "The fare difference is $340. Shall I go ahead and change the flights?"


def test_a_yes_to_the_assistant_s_own_question_is_recorded():
    """The defect this fixes: consent only ever entered through a demand the gate
    had made, so the first confirmation of a conversation was always invisible and
    the gate refused for want of an agreement it had been given."""
    _, given = answered([], "yes, go ahead", turn=2, asked=ASKED)

    assert [c.action for c in given] == [ANY]
    assert given[0].reason == ASKED
    assert given[0].words == "yes, go ahead"
    assert given[0].turn == 2


def test_the_assistant_s_question_is_kept_verbatim_beside_the_answer():
    """Both halves, or the gate cannot tell what the yes covered."""
    _, given = answered([], "sure", turn=2, asked=ASKED)

    assert given[0].reason == ASKED
    assert given[0].words == "sure"


def test_a_hesitant_reply_records_nothing_however_it_was_asked():
    for reply in ["yes, but not the return leg", "ok, hold on", "no thanks"]:
        _, given = answered([], reply, turn=2, asked=ASKED)

        assert given == [], reply


def test_nothing_is_recorded_when_the_assistant_asked_nothing():
    """An opening message the customer volunteers a yes into agreed to nothing."""
    for asked in ["", "   "]:
        _, given = answered([], "yes please", turn=2, asked=asked)

        assert given == []


def test_an_answered_demand_and_an_answered_question_are_both_kept():
    """They are different evidence: one closes a condition this gate set, the
    other is the customer's own words about a question nobody here framed."""
    _, given = answered(_standing(), "yes, go ahead", turn=2, asked=ASKED)

    assert [c.action for c in given] == ["cancel_reservation", ANY]


# ---------------------------------------------------------------- anchored


def test_a_placeholder_record_is_reduced_to_nothing():
    """The defect: the planner writes where the id will come from, not the id.

    Each phrasing is a different `Change.key`, so widening files each one as a
    new commitment and none of them can ever be discharged.
    """
    changes = [
        Change(tool="cancel_reservation", record="the same reservation id", what="cancel it"),
        Change(tool="cancel_reservation", record="reservation_id_from_get_reservation_details"),
        Change(tool="cancel_reservation", record="the reservation id from get_reservation_details"),
    ]
    assert {c.record for c in anchored(changes, ["nothing useful here"])} == {None}
    assert len({c.key for c in anchored(changes, ["nothing useful here"])}) == 1


def test_an_identifier_the_conversation_has_seen_is_kept():
    seen = ['{"reservation_id": "OBUT9V", "cabin": "economy"}']
    (change,) = anchored([Change(tool="cancel_reservation", record="OBUT9V")], seen)
    assert change.record == "OBUT9V"


def test_an_identifier_wrapped_in_where_it_came_from_is_unwrapped():
    """The shape `_covers` grew its containment match for, settled here instead."""
    seen = ['{"reservation_id": "JG7FMM"}']
    (change,) = anchored(
        [Change(tool="cancel_reservation", record="the reservation id from the lookup for JG7FMM")],
        seen,
    )
    assert change.record == "JG7FMM"


def test_a_bare_identifier_nobody_has_read_yet_is_still_kept():
    """Two bookings named in one breath have been read by nobody.

    Dropping them both would merge two commitments into one, which is the
    multi-record failure the ledger exists to stop. An invented identifier is the
    safe way to be wrong here: it stays owed rather than being discharged.
    """
    changes = anchored(
        [
            Change(tool="cancel_reservation", record="AAA111"),
            Change(tool="cancel_reservation", record="BBB222"),
        ],
        ["nobody has looked anything up"],
    )
    assert [c.record for c in changes] == ["AAA111", "BBB222"]
    assert len({c.key for c in changes}) == 2


def test_a_field_name_is_not_an_identifier_even_though_it_is_in_every_record():
    """`reservation_id` occurs in the text of every reservation ever read."""
    seen = ['{"reservation_id": "OBUT9V"}']
    for prose in ("reservation_id", "get_reservation_details", "new_reservation_id"):
        (change,) = anchored([Change(tool="cancel_reservation", record=prose)], seen)
        assert change.record is None, prose


def test_a_date_in_the_description_is_not_mistaken_for_the_record():
    """The one that forces the letter-and-digit rule: 2024 is in every record."""
    seen = ['{"reservation_id": "OBUT9V", "date": "2024-05-27"}']
    (change,) = anchored(
        [Change(tool="update_reservation_flights", record="the reservation for 2024-05-27")], seen
    )
    assert change.record is None


def test_two_records_named_plainly_stay_two_commitments():
    """Anchoring must not collapse a multi-record request into one change."""
    seen = ['{"reservation_id": "8C8K4E"}', '{"reservation_id": "UDMOP1"}']
    changes = anchored(
        [
            Change(tool="cancel_reservation", record="8C8K4E"),
            Change(tool="cancel_reservation", record="UDMOP1"),
        ],
        seen,
    )
    assert len({c.key for c in changes}) == 2


def test_the_original_changes_are_not_mutated():
    original = Change(tool="cancel_reservation", record="the same reservation id")
    anchored([original], [])
    assert original.record == "the same reservation id"


# --- a change has to name a call somebody can make -------------------------


WRITES = frozenset({"cancel_reservation", "update_reservation_flights", "book_reservation"})


def test_a_planned_write_survives():
    changes = [Change(tool="cancel_reservation", record="ABC123", what="cancel it")]

    assert performable(changes, WRITES) == changes


def test_a_lookup_filed_as_a_change_is_dropped():
    """Run 017 filed 23 of these -- get_reservation_details, search_direct_flight,
    calculate, transfer_to_human_agents. A read is never discharged by an approved
    write, so the ledger keeps it owed and the speaker holds the reply forever."""
    changes = [
        Change(tool="get_reservation_details", record="ABC123", what="read it"),
        Change(tool="cancel_reservation", record="ABC123", what="cancel it"),
    ]

    kept = performable(changes, WRITES)

    assert [c.tool for c in kept] == ["cancel_reservation"]


def test_a_tool_that_does_not_exist_is_dropped():
    """`update_reservation_cabin`, six times in run 017. The policy has a "Change
    cabin" heading and the model reasons from it to a call to match; cabin is an
    argument to update_reservation_flights."""
    changes = [Change(tool="update_reservation_cabin", record="ABC123", what="to business")]

    assert performable(changes, WRITES) == []


def test_nothing_is_repaired_into_a_write_nobody_asked_for():
    """Dropping, not guessing. `update_reservation_cabin` could mean a cabin change
    or a flight change, and inventing the difference here would file a write the
    customer never requested."""
    changes = [Change(tool="update_reservation_cabin", record="ABC123", what="to business")]

    assert all(c.tool != "update_reservation_flights" for c in performable(changes, WRITES))


def test_an_empty_write_set_admits_nothing():
    """This function knows only what it is given: with no writes named, no change
    names one. The kernel is what decides not to ask -- it skips the filter when
    the graph was built without the tools it routes on, so a kernel with no
    catalogue behaves exactly as it did before this existed."""
    changes = [Change(tool="cancel_reservation", record=None, what="x")]

    assert performable(changes, frozenset()) == []


KNOWN = WRITES | frozenset({"get_reservation_details", "search_direct_flight"})


def test_a_read_filed_as_a_change_becomes_a_lookup():
    """Dropping it would lose the one thing the entry got right -- that the actor
    has to make this call. `lookups` is the field that carries that, and it is
    rendered to the actor as "Find out first"."""
    changes = [Change(tool="get_reservation_details", record="ABC123", what="read it")]

    assert misfiled(changes, WRITES, KNOWN) == ["get_reservation_details: read it"]


def test_a_tool_that_does_not_exist_is_not_rescued_into_a_lookup():
    """`update_reservation_cabin` is not a misplaced read. Sending the actor to
    find out with a call nobody can make is worse than saying nothing."""
    changes = [Change(tool="update_reservation_cabin", record="ABC123", what="to business")]

    assert misfiled(changes, WRITES, KNOWN) == []


def test_a_write_is_never_rescued_into_a_lookup():
    changes = [Change(tool="cancel_reservation", record="ABC123", what="cancel it")]

    assert misfiled(changes, WRITES, KNOWN) == []
