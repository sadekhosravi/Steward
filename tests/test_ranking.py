"""The comparison appended to a result that offers a choice.

Every fixture here is the shape tau2 actually returns -- JSON, one dict per
flight, itineraries as a list of legs -- because the whole value of this module
is that it reads what the environment really writes.
"""

from __future__ import annotations

import json

from adapters.tau2.ranking import ranked


def flight(number, economy, business=999, seats=None, departs="09:00:00", arrives="15:00:00"):
    return {
        "flight_number": number,
        "origin": "JFK",
        "destination": "SEA",
        "status": "available",
        "scheduled_departure_time_est": departs,
        "scheduled_arrival_time_est": arrives,
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


# ------------------------------------------------------------------ duration


def test_the_quickest_is_named_first():
    """Task 21's ask, in one line: "the fastest possible on May 27"."""
    rows = [
        flight("SLOW", 100, departs="04:00:00", arrives="20:00:00"),
        flight("QUICK", 300, departs="14:00:00", arrives="20:00:00"),
        flight("MIDDLING", 200, departs="04:00:00", arrives="16:00:00"),
    ]
    line = next(row for row in ranked(json.dumps(rows)).splitlines() if "economy duration" in row)
    assert line.index("QUICK") < line.index("MIDDLING") < line.index("SLOW")
    assert "QUICK 6h" in line


def test_the_quickest_option_with_no_seats_is_not_the_answer():
    """Task 21 exactly. Its two quickest itineraries have no economy seats, and
    gold is the quickest that can actually be booked. Without the seat filter
    this recommends a flight the actor cannot buy."""
    rows = [
        flight(
            "FASTEST",
            90,
            seats={"economy": 0, "business": 5},
            departs="11:00:00",
            arrives="16:00:00",
        ),
        flight("GOLD", 300, departs="14:00:00", arrives="20:00:00"),
        flight("SLOW", 100, departs="04:00:00", arrives="20:00:00"),
    ]
    line = next(row for row in ranked(json.dumps(rows)).splitlines() if "economy duration" in row)
    assert "FASTEST" not in line
    assert line.index("GOLD") < line.index("SLOW")


def test_an_itinerary_is_timed_from_first_departure_to_last_arrival():
    """ "Including layovers", which is what task 44's customer asked for."""
    first = flight("LEG1", 50, departs="08:00:00", arrives="10:00:00")
    second = flight("LEG2", 50, departs="14:00:00", arrives="16:00:00")
    other = [
        flight("LEG3", 50, departs="08:00:00", arrives="09:00:00"),
        flight("LEG4", 50, departs="09:30:00", arrives="11:00:00"),
    ]
    line = next(
        row
        for row in ranked(json.dumps([[first, second], other])).splitlines()
        if "economy duration" in row
    )
    assert "LEG1+LEG2 8h" in line
    assert "LEG3+LEG4 3h" in line
    assert line.index("LEG3") < line.index("LEG1")


def test_a_landing_before_takeoff_is_read_as_crossing_midnight():
    rows = [
        flight("REDEYE", 100, departs="22:00:00", arrives="02:00:00"),
        flight("DAY", 100, departs="09:00:00", arrives="15:00:00"),
    ]
    line = next(row for row in ranked(json.dumps(rows)).splitlines() if "economy duration" in row)
    assert "REDEYE 4h" in line and "DAY 6h" in line


def test_dates_are_used_when_the_rows_carry_them():
    """An overnight connection is a real itinerary, not clock arithmetic."""
    first = dict(flight("OUT", 50, departs="20:00:00", arrives="23:00:00"), date="2024-05-20")
    second = dict(flight("BACK", 50, departs="06:00:00", arrives="09:00:00"), date="2024-05-21")
    same = [
        dict(flight("A", 50, departs="08:00:00", arrives="09:00:00"), date="2024-05-20"),
        dict(flight("B", 50, departs="10:00:00", arrives="12:00:00"), date="2024-05-20"),
    ]
    line = next(
        row
        for row in ranked(json.dumps([[first, second], same])).splitlines()
        if "economy duration" in row
    )
    assert "OUT+BACK 13h" in line
    assert "A+B 4h" in line


def test_a_row_with_no_times_is_left_out_rather_than_guessed_at():
    rows = [
        flight("TIMED", 100, departs="09:00:00", arrives="12:00:00"),
        flight("ALSOTIMED", 100, departs="09:00:00", arrives="11:00:00"),
        {k: v for k, v in flight("UNTIMED", 100).items() if k != "scheduled_arrival_time_est"},
    ]
    line = next(row for row in ranked(json.dumps(rows)).splitlines() if "economy duration" in row)
    assert "UNTIMED" not in line
    assert "ALSOTIMED" in line and "TIMED" in line


def test_price_and_duration_disagree_and_both_are_offered():
    """The point of computing rather than recommending: the cheapest is not the
    quickest, and which one the customer wants is in the dialogue."""
    rows = [
        flight("CHEAP_SLOW", 100, departs="04:00:00", arrives="20:00:00"),
        flight("DEAR_QUICK", 400, departs="14:00:00", arrives="20:00:00"),
    ]
    out = ranked(json.dumps(rows))
    price = next(row for row in out.splitlines() if "economy price" in row)
    duration = next(row for row in out.splitlines() if "economy duration" in row)
    assert price.index("CHEAP_SLOW") < price.index("DEAR_QUICK")
    assert duration.index("DEAR_QUICK") < duration.index("CHEAP_SLOW")
