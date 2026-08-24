"""The arithmetic appended to a reservation record.

Every fixture is the shape tau2 really returns -- one JSON object, `flights` as
priced legs, `passengers` as a list, `payment_history` as amounts -- because the
whole value of this module is that it reads what the environment actually writes.
The numbers are the ones from the tasks that lost on them.
"""

from __future__ import annotations

import json

from adapters.tau2.totals import totals


def reservation(prices, passengers=1, paid=None, **extra):
    record = {
        "reservation_id": "K1NW8N",
        "cabin": "basic_economy",
        "flights": [{"flight_number": f"HAT{i:03}", "price": p} for i, p in enumerate(prices)],
        "passengers": [{"first_name": "A", "last_name": str(i)} for i in range(passengers)],
        "payment_history": [{"payment_id": "gift_card_1", "amount": paid}] if paid else [],
    }
    return json.dumps({**record, **extra})


def test_the_fare_and_the_booking_total_are_told_apart():
    """Task 23, as it happened: three legs at 53/71/65 for three passengers, and
    the reply called `$189.00` the total for the reservation. It was $567."""
    text = totals(reservation([53, 71, 65], passengers=3, paid=567))

    assert "fare per passenger: $189  ($53 + $71 + $65)" in text
    assert "for 3 passengers: $189 x 3 = $567" in text


def test_the_total_is_offered_for_a_reservation_that_was_being_invented():
    """Task 18 priced LQ940Q at `$4,200` in a table. The record says $503, and the
    saving the task turned on could not survive the difference."""
    text = totals(reservation([133, 166, 103, 101], passengers=1, paid=533))

    assert "fare per passenger: $503" in text
    assert "for 1 passenger: $503 x 1 = $503" in text


def test_what_was_paid_is_shown_separately_from_what_the_fare_comes_to():
    """They differ by the insurance, and guessing at the difference would be the
    same fabrication this module exists to remove -- so both are shown, labelled."""
    text = totals(reservation([133, 166, 103, 101], passengers=1, paid=533))

    assert "paid so far, from payment_history: $533" in text


def test_a_refund_nets_off_what_was_paid():
    """`payment_history` records a refund as a negative amount, so the sum is the
    balance of the booking rather than everything ever charged."""
    content = json.dumps(
        {
            "flights": [{"price": 100}],
            "passengers": [{"first_name": "A"}],
            "payment_history": [{"amount": 100}, {"amount": -30}],
        }
    )

    assert "paid so far, from payment_history: $70" in totals(content)


def test_a_single_leg_is_not_given_a_sum_to_show():
    """`$154 ($154)` is noise in a prompt that is already long."""
    assert "fare per passenger: $154\n" in totals(reservation([154]))


def test_the_record_is_left_exactly_as_it_arrived():
    """The provenance ledger is built from this text, so every value the actor may
    use has to still be in it, spelled the way the environment spelled it."""
    content = reservation([53, 71, 65], passengers=3, paid=567)

    assert totals(content).startswith(content)


def test_a_record_missing_a_price_gets_no_arithmetic_at_all():
    """A sum over some of the legs is worse than no sum: it is wrong and it looks
    authoritative."""
    content = json.dumps(
        {
            "flights": [{"price": 100}, {"flight_number": "HAT002"}],
            "passengers": [{"first_name": "A"}],
        }
    )

    assert totals(content) == content


def test_anything_that_is_not_a_booking_is_left_alone():
    """A profile, a search, an error, an empty result, a shape nobody has looked
    at yet."""
    for content in (
        "Error: Reservation not found",
        "",
        "not json at all",
        json.dumps({"user_id": "mohamed_silva_9265", "reservations": ["K1NW8N"]}),
        json.dumps([{"flight_number": "HAT021", "prices": {"economy": 108}}]),
        json.dumps({"flights": [], "passengers": []}),
    ):
        assert totals(content) == content


def test_it_does_not_touch_what_the_comparison_reads_or_the_other_way_round():
    """The two run one after the other on every result, so the guarantee that
    keeps them safe is that no result is a shape both of them recognise."""
    from adapters.tau2.ranking import ranked

    booking = reservation([53, 71, 65], passengers=3, paid=567)
    assert ranked(booking) == booking

    search = json.dumps(
        [
            {
                "flight_number": "HAT021",
                "available_seats": {"economy": 5},
                "prices": {"economy": 108},
            },
            {
                "flight_number": "HAT100",
                "available_seats": {"economy": 5},
                "prices": {"economy": 88},
            },
        ]
    )
    assert totals(search) == search


def test_the_arithmetic_reaches_the_kernel_through_the_adapter():
    """Wiring, and worth a test of its own because it is invisible from outside:
    tau2 logs the `ToolMessage` it built, not what we handed the Kernel, so a
    transcript will never show this block whether it happened or not."""
    from tau2.data_model.message import ToolMessage

    from adapters.tau2.agent import _tool_results

    message = ToolMessage(
        id="1",
        role="tool",
        content=reservation([561, 1339], passengers=3, paid=5700),
        requestor="assistant",
    )

    assert "for 3 passengers: $1,900 x 3 = $5,700" in (_tool_results(message) or {})["1"]
