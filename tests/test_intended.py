"""The record-selection check, and the four ways it is required to stay quiet.

`intended` is the first check in this codebase that acts on something a model
produced, so it is tested from both ends: that the comparison fires when a stated
number contradicts a record, and -- at greater length -- that nothing a model can
return is able to block a write the customer's own words do not rule out.

The refusals-to-fire are the half that matters. A criterion the customer never
uttered, a paraphrase dressed up as a quote, a reservation nobody has read, a
cabin the call is itself changing: each one of those, allowed through, blocks a
write gold makes. The tests below are named for the case each one protects.
"""

from __future__ import annotations

import json

from adapters.tau2.intended import grounded, intended, named, read_first, said
from adapters.tau2.modifications import passenger_count_fixed
from adapters.tau2.verifiers import SELECTION
from core.state import PendingCall
from core.verifiers import Evidence, first


def call(name, **arguments):
    return PendingCall(id="p", name=name, arguments=arguments)


def reservation(**overrides):
    """Two passengers, economy, LAS->ATL, no insurance, round trip."""
    record = {
        "reservation_id": "UM3OG5",
        "user_id": "omar_rossi_1241",
        "origin": "LAS",
        "destination": "ATL",
        "flight_type": "round_trip",
        "cabin": "economy",
        "flights": [{"flight_number": "HAT005", "date": "2024-05-20", "price": 65}],
        "passengers": [
            {"first_name": "Omar", "last_name": "Rossi", "dob": "1990-01-01"},
            {"first_name": "Mia", "last_name": "Rossi", "dob": "1992-02-02"},
        ],
        "created_at": "2024-05-01T05:17:41",
        "total_baggages": 0,
        "nonfree_baggages": 0,
        "insurance": "no",
    }
    return json.dumps({**record, **overrides})


def evidence(*records, dialogue="", stated=None):
    return Evidence.of(list(records), dialogue, [], [], stated or {})


SPOKE = "Customer: please change the passenger to myself and add three bags"


def spoken(dialogue):
    """The customer's turn as dumped pydantic-ai history.

    The Kernel builds `Evidence.dialogue` from the message history rather than
    being handed it, so a state with no history has nothing the check can ground
    a quote against -- and every criterion would be discarded for the wrong
    reason.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart

    text = dialogue.removeprefix("Customer: ")
    return ModelMessagesTypeAdapter.dump_python(
        [ModelRequest(parts=[UserPromptPart(content=text)])]
    )


# ------------------------------------------------------- reading the dialogue


def test_only_the_customer_s_own_lines_count_as_something_they_said():
    spoken = said(
        "Assistant: Hi! How can I help you today?\n"
        "Customer: cancel the one with one passenger\n"
        "Assistant looks up: get_reservation_details(...)\n"
        "Result: {}\n"
        "Assistant: I will cancel UM3OG5 for you."
    )

    assert spoken == "cancel the one with one passenger"


def test_a_turn_that_runs_to_several_lines_is_not_cut_off_at_the_first():
    """The answer key's scenarios are all multi-line, and taking only labelled
    lines would hide almost all of the source a false block would come from."""
    spoken = said("Customer: cancel my trip\nbut only the one with one passenger\nAssistant: Sure.")

    assert "only the one with one passenger" in spoken
    assert "Sure." not in spoken


def test_the_assistant_announcing_its_own_write_is_not_the_customer_speaking():
    """`Assistant calls:` is a label only `gate_bench` writes. Left out of the
    speaker list it reads as a continuation of the customer's previous turn, and
    the assistant's account of what it is doing becomes the customer's words."""
    spoken = said("Customer: go ahead\nAssistant calls: cancel_reservation(...)")

    assert spoken == "go ahead"


# ------------------------------------------------------------------ grounding


def test_a_quote_the_customer_never_typed_discards_the_whole_extraction():
    assert not grounded("the one with a single passenger", "cancel the one with one passenger")


def test_a_quote_survives_a_line_break_and_a_capital_letter():
    assert grounded("One\nPassenger", "cancel the one   passenger booking")


def test_a_quote_too_short_to_be_evidence_of_reading_is_refused():
    assert not grounded("the", "cancel the one with one passenger")
    assert not grounded("", "cancel the one with one passenger")
    assert not grounded(None, "cancel the one with one passenger")


def test_an_identifier_is_not_found_inside_a_longer_token():
    """Flight numbers and reservation ids share an alphabet here -- `HAT110` looks
    exactly like an id -- so an unbounded match would let any six characters
    clear the check that exists to protect a customer who named their booking."""
    assert named("UM3OG5", "please cancel um3og5")
    assert not named("HAT110", "my flight is HAT1102")


# ------------------------------------------------------------ the comparisons


def test_a_reservation_with_two_passengers_is_not_the_one_they_described():
    finding = intended(
        call("update_reservation_baggages", reservation_id="UM3OG5", total_baggages=3),
        evidence(reservation(), dialogue=SPOKE, stated={"words": "the passenger", "passengers": 1}),
    )

    assert finding is not None
    assert finding.check == "intended:passengers"
    assert finding.recoverable
    assert "the passenger" in finding.reason
    assert "UM3OG5" in finding.remediation


def test_the_record_that_does_match_what_they_described_goes_through():
    finding = intended(
        call("update_reservation_baggages", reservation_id="UM3OG5", total_baggages=3),
        evidence(
            reservation(passengers=[{"first_name": "Omar", "last_name": "Rossi", "dob": "1990"}]),
            dialogue=SPOKE,
            stated={"words": "the passenger", "passengers": 1},
        ),
    )

    assert finding is None


def test_a_customer_who_named_the_reservation_outranks_every_description():
    """Task 44's customer names two reservations and gold writes three others, so
    a named identifier may clear this check and may never fail it."""
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(
            reservation(),
            dialogue="Customer: cancel UM3OG5, it has one passenger",
            stated={"words": "one passenger", "passengers": 1},
        ),
    )

    assert finding is None


def test_a_reservation_nobody_has_read_leaves_nothing_to_compare():
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(dialogue=SPOKE, stated={"words": "the passenger", "passengers": 1}),
    )

    assert finding is None


def test_an_extraction_that_came_back_empty_is_silence_and_not_a_refusal():
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(reservation(), dialogue=SPOKE),
    )

    assert finding is None


def test_a_call_that_creates_its_own_record_is_not_routed_here_at_all():
    assert SELECTION.for_tool("book_reservation") == []
    assert SELECTION.for_tool("send_certificate") == []
    assert SELECTION.for_tool("cancel_reservation") == [intended]


# ------------------------------------------------------------------ the cabin


def test_the_cabin_the_call_is_changing_is_never_read_as_the_cabin_it_is_in():
    """Gold's task 7 upgrades a basic economy reservation to business, and the
    customer's own sentence names business. On that call the stated cabin
    describes where the record is going, and a check that cannot tell those apart
    refuses gold's own write."""
    finding = intended(
        call(
            "update_reservation_flights",
            reservation_id="UM3OG5",
            cabin="business",
            flights=[{"flight_number": "HAT005", "date": "2024-05-20"}],
        ),
        evidence(
            reservation(cabin="basic_economy"),
            dialogue="Customer: upgrade it to business and then cancel it",
            stated={"words": "upgrade it to business", "cabin": "business"},
        ),
    )

    assert finding is None


def test_a_cabin_they_described_is_compared_when_the_call_leaves_it_alone():
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(
            reservation(cabin="economy"),
            dialogue="Customer: cancel my basic economy booking",
            stated={"words": "my basic economy booking", "cabin": "Basic Economy"},
        ),
    )

    assert finding is not None
    assert finding.check == "intended:cabin"


def test_a_cabin_this_domain_does_not_have_is_dropped_rather_than_matched():
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(
            reservation(),
            dialogue="Customer: cancel my first class booking",
            stated={"words": "my first class booking", "cabin": "first"},
        ),
    )

    assert finding is None


# ------------------------------------------------------- the rest of the set


def test_a_city_name_is_not_matched_against_three_letters_that_are_not_it():
    """`New York` and `JFK` are the same place and resolving that needs a table
    this does not have -- and New York has three airports, so a guess would be a
    false block rather than a near miss."""
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(
            reservation(),
            dialogue="Customer: cancel my trip from New York",
            stated={"words": "my trip from New York", "origin": "New York"},
        ),
    )

    assert finding is None


def test_an_airport_code_they_gave_is_compared_against_the_record():
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(
            reservation(),
            dialogue="Customer: cancel my trip out of jfk",
            stated={"words": "my trip out of jfk", "origin": "jfk"},
        ),
    )

    assert finding is not None
    assert finding.check == "intended:origin"


def test_a_one_way_trip_they_described_against_a_round_trip_record():
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(
            reservation(),
            dialogue="Customer: cancel my one way flight",
            stated={"words": "my one way flight", "flight_type": "one way"},
        ),
    )

    assert finding is not None
    assert finding.check == "intended:flight_type"


def test_insurance_they_said_the_booking_has_against_one_that_has_none():
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(
            reservation(insurance="no"),
            dialogue="Customer: cancel the one I bought insurance on",
            stated={"words": "the one I bought insurance on", "insurance": True},
        ),
    )

    assert finding is not None
    assert finding.check == "intended:insurance"


def test_the_panel_reaches_this_check_the_same_way_it_reaches_the_free_ones():
    finding = first(
        call("update_reservation_baggages", reservation_id="UM3OG5", total_baggages=3),
        evidence(reservation(), dialogue=SPOKE, stated={"words": "the passenger", "passengers": 1}),
        SELECTION,
    )

    assert finding is not None
    assert finding.check == "intended:passengers"


# --------------------------------------------------- the stage inside the gate


def test_the_extractor_is_not_paid_for_on_a_call_a_free_check_already_refused():
    """Order is cost. Every check in `PANEL` is a pure function of things already
    on disk; this stage costs a model call, so it is reached only by proposals the
    free ones have cleared -- on the labelled corpus, four in five."""
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    asked = []

    def describe(_call, _evidence):
        asked.append(_call)
        return {"words": "the passenger", "passengers": 1}

    proposed = {
        "id": "1",
        "name": "update_reservation_passengers",
        "arguments": {"reservation_id": "UM3OG5", "passengers": [{"a": 1}]},
    }
    free = Panel(verifiers={"update_reservation_passengers": [passenger_count_fixed]})
    state = StewardState(calls=[proposed], observed=[reservation()])

    out = kernel._gate(
        state,
        None,
        frozenset({"update_reservation_passengers"}),
        free,
        SELECTION,
        describe,
    )

    assert out["approved"] == []
    assert "number of passengers" in out["denied"]["1"]
    assert asked == []


def test_a_selection_refusal_is_counted_as_a_block_and_not_as_an_argument(monkeypatch):
    """It ends in a verifier reading a record, so it spends the budget verifiers
    spend. The extraction in front of it does not change what stopped the call."""
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    monkeypatch.setattr(kernel, "REVIEWING", False)
    proposed = {
        "id": "1",
        "name": "update_reservation_baggages",
        "arguments": {"reservation_id": "UM3OG5", "total_baggages": 3},
    }
    state = StewardState(calls=[proposed], observed=[reservation()], messages=spoken(SPOKE))

    out = kernel._gate(
        state,
        None,
        frozenset({"update_reservation_baggages"}),
        Panel(),
        SELECTION,
        lambda _c, _e: {"words": "the passenger", "passengers": 1},
    )

    assert out["approved"] == []
    assert out["blocked"] == 1
    assert "revisions" not in out
    assert "UM3OG5" in out["denied"]["1"]


def test_an_extractor_that_returns_nothing_lets_the_call_straight_through(monkeypatch):
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    monkeypatch.setattr(kernel, "REVIEWING", False)
    proposed = {
        "id": "1",
        "name": "update_reservation_baggages",
        "arguments": {"reservation_id": "UM3OG5", "total_baggages": 3},
    }
    state = StewardState(calls=[proposed], observed=[reservation()], messages=spoken(SPOKE))

    out = kernel._gate(
        state,
        None,
        frozenset({"update_reservation_baggages"}),
        Panel(),
        SELECTION,
        lambda _c, _e: {},
    )

    assert out["approved"] == state.calls


def test_a_passenger_count_below_one_is_a_malformed_answer_and_not_a_claim():
    """Every reservation has at least one passenger, so a stray zero left in
    would block every record it was ever compared against."""
    finding = intended(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(reservation(), dialogue=SPOKE, stated={"words": "the passenger", "passengers": 0}),
    )

    assert finding is None


# ------------------------------------------------- a record nobody has read


def test_a_write_on_a_reservation_nobody_looked_at_is_stopped():
    """Task 41's run cancelled seven reservations and had read six of them not at
    all. Nothing to compare a description against, and nothing to check the
    cancellation rules against either -- `cancellable` next door falls silent for
    exactly this reason, and this is what that silence should have been saying."""
    finding = read_first(
        call("cancel_reservation", reservation_id="UDMOP1"),
        evidence(reservation(), dialogue="Customer: cancel all my upcoming flights"),
    )

    assert finding is not None
    assert finding.check == "read_first"
    assert finding.recoverable
    assert "get_reservation_details" in finding.remediation


def test_a_customer_who_named_the_booking_has_settled_which_record_it_is():
    """Gold's task 42 cancels two reservations it never reads, because the
    customer names both. Without this escape the check blocks them and scores
    82%; with it, 8 surplus and no gold at all."""
    finding = read_first(
        call("cancel_reservation", reservation_id="HSR97W"),
        evidence(dialogue="Customer: please cancel HSR97W and FDZ0T5"),
    )

    assert finding is None


def test_a_reservation_that_was_read_is_this_check_s_business_no_longer():
    finding = read_first(
        call("cancel_reservation", reservation_id="UM3OG5"),
        evidence(reservation(), dialogue="Customer: cancel it"),
    )

    assert finding is None


def test_a_call_that_names_no_reservation_at_all_is_left_alone():
    assert read_first(call("book_reservation", user_id="omar_rossi_1241"), evidence()) is None


def test_the_free_panel_asks_this_before_anything_that_needs_the_record():
    """Order is cost, and a check that cannot run without the record has nothing
    to say about a call that never read one."""
    from adapters.tau2.verifiers import PANEL

    assert PANEL.for_tool("cancel_reservation")[0] is read_first
    finding = first(
        call("cancel_reservation", reservation_id="UDMOP1"),
        evidence(reservation(), dialogue="Customer: cancel all my upcoming flights"),
        PANEL,
    )

    assert finding is not None
    assert finding.check == "read_first"
