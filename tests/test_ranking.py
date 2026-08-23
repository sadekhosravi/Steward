"""The comparison appended to a result that offers a choice.

Every fixture here is the shape tau2 actually returns -- JSON, one dict per
flight, itineraries as a list of legs -- because the whole value of this module
is that it reads what the environment really writes.
"""

from __future__ import annotations

import json

from adapters.tau2.ranking import ranked


def flight(number, economy, business=999, seats=None, departs="09:00:00"):
    return {
        "flight_number": number,
        "origin": "JFK",
        "destination": "SEA",
        "status": "available",
        "scheduled_departure_time_est": departs,
        "scheduled_arrival_time_est": "15:00:00",
        "date": None,
        "available_seats": seats or {"economy": 5, "business": 5},
        "prices": {"economy": economy, "business": business},
    }


def test_the_cheapest_is_named_first_whatever_order_it_arrived_in():
    """Task 23, as it happened: the actor took the $453 business fare printed
    first and the gold answer was the $259 one two rows below."""
    content = json.dumps([flight("HAT021", 108, 453), flight("HAT100", 108, 259)])

    text = ranked(content)

    assert "by business price: HAT100 $259, HAT021 $453" in text


def test_the_raw_rows_are_left_exactly_as_they_arrived():
    """The provenance ledger is built from this text, so every value the actor is
    allowed to use has to still be in it, spelled the way the environment spelled
    it."""
    content = json.dumps([flight("HAT021", 108), flight("HAT100", 259)])

    assert ranked(content).startswith(content)


def test_a_sold_out_cabin_is_not_offered_as_an_option():
    """A seat that does not exist is not a slower option, it is not an option --
    and booking into one spends the call on an error."""
    content = json.dumps(
        [
            flight("HAT021", 108, seats={"economy": 0, "business": 4}),
            flight("HAT100", 259, seats={"economy": 6, "business": 4}),
        ]
    )

    text = ranked(content)

    assert "by economy price" not in text
    assert "by business price" in text


def test_an_itinerary_is_costed_across_all_its_legs():
    """`search_onestop_flight` returns a list of legs per result, and what the
    customer pays is the pair, not either half."""
    content = json.dumps(
        [
            [flight("HAT057", 141), flight("HAT039", 103)],
            [flight("HAT001", 100), flight("HAT002", 100)],
        ]
    )

    text = ranked(content)

    assert "HAT001+HAT002 $200, HAT057+HAT039 $244" in text


def test_an_itinerary_takes_the_scarcest_seat_count_on_any_leg():
    """A cabin is only bookable if it is bookable end to end."""
    content = json.dumps(
        [
            [
                flight("HAT057", 141, seats={"economy": 0, "business": 9}),
                flight("HAT039", 103, seats={"economy": 8, "business": 9}),
            ],
            [flight("HAT001", 100), flight("HAT002", 100)],
        ]
    )

    text = ranked(content)

    assert "by economy price" not in text


def test_departure_time_is_offered_as_well_as_price():
    """The customer who asked for a morning flight is not asking about money."""
    content = json.dumps(
        [flight("HAT021", 108, departs="19:00:00"), flight("HAT100", 259, departs="06:00:00")]
    )

    assert "by departure: HAT100 06:00:00, HAT021 19:00:00" in ranked(content)


def test_a_single_result_gets_no_comparison():
    """There is no choice to present, and a heading over one row is noise in a
    prompt that is already long."""
    content = json.dumps([flight("HAT021", 108)])

    assert ranked(content) == content


def test_anything_it_cannot_read_is_left_alone():
    """A lookup, an error, a shape from a domain nobody has looked at yet."""
    for content in (
        "Error: Reservation not found",
        "",
        json.dumps({"reservation_id": "HKD3PS", "cabin": "economy"}),
        json.dumps([{"reservation_id": "HKD3PS"}, {"reservation_id": "4WQ150"}]),
        "not json at all",
    ):
        assert ranked(content) == content


def test_the_comparison_reaches_the_kernel_through_the_adapter():
    """Wiring, and worth a test of its own because it is invisible from outside:
    tau2 logs the `ToolMessage` it built, not what we handed the Kernel, so a
    transcript will never show this block whether it happened or not."""
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    from adapters.tau2.agent import _tool_results

    content = json.dumps([flight("HAT021", 108, 453), flight("HAT100", 108, 259)])
    message = MultiToolMessage(
        role="tool",
        tool_messages=[ToolMessage(id="1", role="tool", content=content, requestor="assistant")],
    )

    assert "by business price: HAT100 $259" in (_tool_results(message) or {})["1"]


def test_a_failed_call_is_passed_through_untouched():
    """An error is not a set of options, and appending a comparison to one would
    bury the thing the actor has to read to fix its call."""
    from tau2.data_model.message import ToolMessage

    from adapters.tau2.agent import _tool_results

    message = ToolMessage(
        id="1",
        role="tool",
        content="Error: Reservation not found",
        error=True,
        requestor="assistant",
    )

    assert (_tool_results(message) or {})["1"] == "Error: Reservation not found"
