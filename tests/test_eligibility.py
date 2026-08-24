"""The policy conditions appended to a reservation record.

Every fixture is the shape tau2 really returns -- one JSON object with `cabin`,
`created_at`, `insurance` and `passengers` on it -- because the whole value of
this module is that it reads what the environment actually writes. The records
are the ones the gate got wrong in the 50-task run.
"""

from __future__ import annotations

import json

from adapters.tau2.eligibility import NOW, eligibility


def reservation(**overrides):
    record = {
        "reservation_id": "HSR97W",
        "user_id": "sophia_martin_4574",
        "origin": "ORD",
        "destination": "SFO",
        "flight_type": "one_way",
        "cabin": "business",
        "flights": [{"flight_number": "HAT289", "date": "2024-05-22", "price": 705}],
        "passengers": [{"first_name": "Sophia", "last_name": "Martin"}],
        "payment_history": [{"payment_id": "credit_card_1402274", "amount": 1227}],
        "created_at": "2024-05-11T10:09:09",
        "total_baggages": 0,
        "nonfree_baggages": 0,
        "insurance": "yes",
    }
    return json.dumps({**record, **overrides})


def test_a_business_cabin_reservation_is_reported_as_meeting_the_test():
    """Task 42, as it happened. The gate quoted the rule correctly -- "it is a
    business cabin reservation" -- and then refused HSR97W for being one."""
    text = eligibility(reservation())

    assert "business cabin: yes -- cabin is business" in text
    assert "=> one already holds" in text


def test_a_booking_made_within_the_day_is_reported_as_fresh():
    """K1NW8N was created at 16:03 on the 14th, which is 23 hours before the
    current time -- inside the window, and by less than an hour."""
    text = eligibility(reservation(cabin="basic_economy", created_at="2024-05-14T16:03:16"))

    assert "booked in the last 24 hours: yes" in text
    assert "=> one already holds" in text


def test_a_booking_made_just_outside_the_day_is_not():
    text = eligibility(reservation(cabin="economy", created_at="2024-05-14T14:59:00"))

    assert "booked in the last 24 hours: no" in text


def test_insurance_alone_is_handed_back_as_a_question_about_the_reason():
    """Four of the eleven reservations gold cancels rest on insurance, and none of
    them is settled by the record: the policy covers health and weather reasons
    only, and the reason lives in the conversation."""
    text = eligibility(reservation(cabin="economy", created_at="2024-05-05T09:00:00"))

    assert "travel insurance on file: yes" in text
    assert "the reason decides it" in text
    assert "=> one already holds" not in text


def test_a_record_that_settles_nothing_says_so_without_forbidding_anything():
    """XEHM4B -- basic economy, booked on the 1st, no insurance. Gold cancels it,
    on the one ground this cannot see. Saying "no" here would cost that task."""
    text = eligibility(
        reservation(cabin="basic_economy", created_at="2024-05-01T05:17:41", insurance="no")
    )

    assert "=> none holds on this record" in text
    assert "cancelled by the airline: not on this record -- get_flight_status" in text


def test_the_condition_that_is_not_on_the_record_is_always_named():
    for record in (reservation(), reservation(insurance="no")):
        assert "get_flight_status" in eligibility(record)


def test_basic_economy_is_told_that_a_cabin_change_is_the_way_through():
    """Gold calls `update_reservation_flights` on a basic economy reservation eight
    times and changes the cabin every time. The gate cited basic economy in
    twenty-two refusals."""
    text = eligibility(reservation(cabin="basic_economy"))

    assert "Basic economy flights cannot be changed while it stays basic economy" in text
    assert "update_reservation_flights is the call that does it" in text


def test_a_cabin_that_can_be_re_flighted_is_not_warned_off():
    """The basic economy line is the only modification rule stated, so it has to
    stay off every record it does not apply to."""
    assert "Basic economy" not in eligibility(reservation(cabin="economy"))
    assert "Basic economy" not in eligibility(reservation(cabin="business"))


def test_the_rest_of_the_modification_rules_are_left_where_the_model_has_them():
    """Length is paid on every reservation read, and a task that reads six of them
    pays it six times. Restating the policy here bought nothing and cost that."""
    text = eligibility(reservation())

    assert "passengers:" not in text
    assert "baggage:" not in text
    assert len(text) - len(reservation()) < 800


def test_a_certificate_is_named_as_no_way_to_pay_for_a_change():
    """policy.md:131, and the environment enforces it: the run has
    `Error: Certificate cannot be used to update reservation`."""
    assert "a certificate cannot" in eligibility(reservation())


def test_a_leg_already_behind_the_current_time_is_flagged():
    text = eligibility(reservation(flights=[{"flight_number": "HAT289", "date": "2024-05-12"}]))

    assert "dated 2024-05-12, already past" in text


def test_a_future_reservation_carries_no_such_note():
    assert "already past" not in eligibility(reservation())


def test_an_already_cancelled_reservation_is_not_offered_four_conditions():
    text = eligibility(reservation(status="cancelled"))

    assert "It is already cancelled." in text
    assert "business cabin:" not in text


def test_the_record_is_left_exactly_as_it_arrived():
    """The provenance ledger is built from this text, so every value the actor may
    use has to still be in it, spelled the way the environment spelled it."""
    content = reservation()

    assert eligibility(content).startswith(content)


def test_anything_that_is_not_a_reservation_is_left_alone():
    for content in (
        "Error: Reservation not found",
        "",
        "not json at all",
        json.dumps({"user_id": "mohamed_silva_9265", "reservations": ["K1NW8N"]}),
        json.dumps([{"flight_number": "HAT021", "prices": {"economy": 108}}]),
        json.dumps({"cabin": "economy"}),
        json.dumps({"reservation_id": "X", "cabin": "sleeper", "created_at": "2024-05-01"}),
    ):
        assert eligibility(content) == content


def test_an_unreadable_timestamp_is_admitted_rather_than_read_as_a_no():
    text = eligibility(reservation(cabin="economy", created_at="last tuesday"))

    assert "booked in the last 24 hours: cannot be read from this record" in text


def test_the_current_time_still_matches_the_policy_it_was_copied_from():
    """A constant about someone else's fixture, so it is checked against the
    fixture rather than trusted."""
    from tau2.domains.airline.utils import AIRLINE_POLICY_PATH

    policy = open(AIRLINE_POLICY_PATH, encoding="utf-8").read()

    assert f"The current time is {NOW:%Y-%m-%d %H:%M:%S} EST." in policy


def test_it_does_not_touch_what_the_comparison_reads():
    """The three run on every result, so what keeps them apart is that a search is
    a list and a reservation is an object."""
    from adapters.tau2.ranking import ranked

    booking = reservation()
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
    assert eligibility(search) == search


def test_a_reservation_gets_the_cost_and_the_conditions_and_keeps_its_record():
    """The two notes that share this shape. Chained, the second would parse a
    record with a block of English on the end of it and fall silent."""
    from adapters.tau2.agent import _noted

    text = _noted(reservation())

    assert text.startswith(reservation())
    assert "WHAT THIS RESERVATION COSTS" in text
    assert "WHETHER THIS RESERVATION CAN BE CANCELLED" in text


def test_the_conditions_reach_the_kernel_through_the_adapter():
    """Wiring, and worth a test of its own because it is invisible from outside:
    tau2 logs the `ToolMessage` it built, not what we handed the Kernel, so a
    transcript will never show this block whether it happened or not."""
    from tau2.data_model.message import ToolMessage

    from adapters.tau2.agent import _tool_results

    message = ToolMessage(id="1", role="tool", content=reservation(), requestor="assistant")

    assert "business cabin: yes" in (_tool_results(message) or {})["1"]


def test_a_failed_lookup_is_passed_through_untouched():
    from tau2.data_model.message import ToolMessage

    from adapters.tau2.agent import _tool_results

    message = ToolMessage(
        id="1", role="tool", content=reservation(), error=True, requestor="assistant"
    )

    assert (_tool_results(message) or {})["1"] == reservation()
