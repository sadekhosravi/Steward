"""The free checked-bag allowance appended to a record.

Every fixture is the shape tau2 really returns. The numbers are the ones the run
got wrong.
"""

from __future__ import annotations

import json

from adapters.tau2.baggage import bags


def reservation(**overrides):
    record = {
        "reservation_id": "FQ8APE",
        "user_id": "omar_rossi_1241",
        "cabin": "economy",
        "flights": [{"flight_number": "HAT056", "date": "2024-05-25", "price": 100}],
        "passengers": [{"first_name": "Omar", "last_name": "Rossi"}],
        "payment_history": [],
        "created_at": "2024-05-10T00:00:00",
        "total_baggages": 3,
        "nonfree_baggages": 0,
        "insurance": "no",
    }
    return json.dumps({**record, **overrides})


def user(**overrides):
    record = {
        "user_id": "omar_rossi_1241",
        "name": {"first_name": "Omar", "last_name": "Rossi"},
        "membership": "gold",
        "payment_methods": {},
        "saved_passengers": [],
        "reservations": ["FQ8APE"],
    }
    return json.dumps({**record, **overrides})


def test_the_row_task_17_turned_on_is_worked_out_for_the_reader():
    """Gold, economy, one passenger, three bags. Gold sets `nonfree_baggages` to
    0; the gate insisted on 1 and made the actor write it."""
    text = bags(reservation())

    assert "gold    3 free -> nonfree_baggages 0" in text


def test_task_21_gets_the_same_treatment():
    """Silver, economy, one passenger, two bags. Two free, nothing to pay. The
    run wrote 1."""
    text = bags(reservation(total_baggages=2))

    assert "silver  2 free -> nonfree_baggages 0" in text


def test_the_cabin_is_read_off_the_record_as_it_stands_now():
    """The whole defect. Task 17's reservation was *booked* basic economy and the
    turn moves it to economy before the bags are added; every write tool in this
    domain returns the updated reservation, so the card that lands just before the
    baggage call has already seen the change."""
    booked = bags(reservation(cabin="basic_economy"))
    changed = bags(reservation(cabin="economy"))

    assert "gold    2 free -> nonfree_baggages 1" in booked
    assert "gold    3 free -> nonfree_baggages 0" in changed
    assert "if the cabin changes these change with it" in changed


def test_every_tier_is_offered_because_the_record_does_not_name_one():
    """A reservation carries `user_id`, never the membership. Guessing the tier
    would be inventing the one fact this does not have."""
    text = bags(reservation())

    for tier in ("regular", "silver", "gold"):
        assert f"{tier:<8}" in text
    assert "get_user_details gives the membership" in text


def test_more_passengers_multiply_the_allowance():
    text = bags(
        reservation(
            passengers=[{"first_name": "A"}, {"first_name": "B"}],
            total_baggages=5,
            cabin="business",
        )
    )

    assert "for 2 passengers" in text
    assert "regular 4 free -> nonfree_baggages 1" in text
    assert "gold    8 free -> nonfree_baggages 0" in text


def test_the_allowance_never_goes_negative():
    """Fewer bags than the allowance is nothing to pay, not a credit."""
    text = bags(reservation(total_baggages=0))

    assert "regular 1 free -> nonfree_baggages 0" in text


def test_only_the_increase_over_what_is_already_paid_for_is_charged():
    """`update_reservation_baggages` bills `50 * max(0, new - old)`."""
    text = bags(reservation(nonfree_baggages=2))

    assert "only the increase over the 2 already paid for is charged" in text


def test_a_user_record_is_told_its_own_row():
    text = bags(user())

    assert "gold membership: 2 in basic economy, 3 in economy, 4 in business" in text
    assert "Each bag past that is $50." in text


def test_each_tier_gets_the_policy_table_verbatim():
    assert "regular membership: 0 in basic economy, 1 in economy, 2 in business" in bags(
        user(membership="regular")
    )
    assert "silver membership: 1 in basic economy, 2 in economy, 3 in business" in bags(
        user(membership="silver")
    )


def test_zero_bags_paid_for_is_an_answer_and_not_a_missing_field():
    """`0` is the commonest value in the database and has to survive the check
    that the field is there at all."""
    assert "CHECKED BAGS ON THIS RESERVATION" in bags(reservation(nonfree_baggages=0))


def test_the_record_is_left_exactly_as_it_arrived():
    for content in (reservation(), user()):
        assert bags(content).startswith(content)


def test_anything_else_is_left_alone():
    for content in (
        "Error: Reservation not found",
        "",
        "not json at all",
        json.dumps([{"flight_number": "HAT021", "prices": {"economy": 108}}]),
        json.dumps({"cabin": "economy"}),
        json.dumps({"membership": "platinum"}),
        json.dumps({"cabin": "economy", "passengers": [{"first_name": "A"}]}),
        json.dumps(
            {"cabin": "economy", "passengers": [{"first_name": "A"}], "total_baggages": "two"}
        ),
    ):
        assert bags(content) == content


def test_the_four_notes_stack_without_reading_each_other():
    """All four parse the result as JSON to decide whether they have anything to
    say, so each is handed the raw text and only its own tail is kept."""
    from adapters.tau2.agent import _noted

    text = _noted(reservation())

    assert text.startswith(reservation())
    assert "WHAT THIS RESERVATION COSTS" in text
    assert "WHETHER THIS RESERVATION CAN BE CANCELLED" in text
    assert "CHECKED BAGS ON THIS RESERVATION" in text


def test_the_user_record_gets_its_row_and_nothing_meant_for_a_reservation():
    from adapters.tau2.agent import _noted

    text = _noted(user())

    assert "CHECKED BAGS THIS USER GETS FREE" in text
    assert "WHETHER THIS RESERVATION CAN BE CANCELLED" not in text
    assert "WHAT THIS RESERVATION COSTS" not in text


def test_the_allowance_reaches_the_kernel_through_the_adapter():
    """tau2 logs the `ToolMessage` it built, not what we handed the Kernel, so a
    transcript will never show this block whether it happened or not."""
    from tau2.data_model.message import ToolMessage

    from adapters.tau2.agent import _tool_results

    message = ToolMessage(id="1", role="tool", content=reservation(), requestor="assistant")

    assert "gold    3 free -> nonfree_baggages 0" in (_tool_results(message) or {})["1"]
