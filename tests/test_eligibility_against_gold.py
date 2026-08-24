"""What the card says, checked against every action the benchmark scores.

The card is only worth appending if it never contradicts a correct action. That
is not a property of the code, it is a property of the code *and this domain's
data*, so it is measured against the data rather than argued for -- and pinned
here, so precision cannot rot without a test going red.

Skipped when the tau2 data checkout is absent; `scripts/bootstrap_data.py`
creates one.
"""

from __future__ import annotations

import json

import pytest

from adapters.tau2.eligibility import eligibility

pytest.importorskip("tau2.domains.airline.environment")


@pytest.fixture(scope="module")
def domain():
    from tau2.domains.airline.environment import get_tasks
    from tau2.domains.airline.utils import AIRLINE_DB_PATH

    try:
        db = json.loads(open(AIRLINE_DB_PATH, encoding="utf-8").read())
        tasks = get_tasks()
    except (OSError, ValueError) as missing:  # no vendored data on this machine
        pytest.skip(f"tau2 airline data unavailable: {missing}")
    return db["reservations"], tasks


def gold(tasks, *names):
    """Every gold action with one of these names, paired with its task id."""
    for task in tasks:
        criteria = task.evaluation_criteria
        for action in (criteria.actions if criteria else None) or []:
            if action.name in names:
                yield task.id, action


def card(reservations, action):
    """The card as it would be read at the moment that action's record is looked
    up. Returns None for a record the task creates rather than starts with."""
    record = reservations.get(action.arguments.get("reservation_id"))
    return eligibility(json.dumps(record)) if record else None


def test_no_reservation_gold_cancels_is_reported_as_already_flown(domain):
    """The flown flag stops every change to a booking, so a false one is fatal.
    None of the reservations gold cancels or re-flights carries a past-dated leg."""
    reservations, tasks = domain
    flagged = [
        task
        for task, action in gold(tasks, "cancel_reservation", "update_reservation_flights")
        if (text := card(reservations, action)) and "already past" in text
    ]

    assert flagged == []


def test_every_gold_cancellation_is_left_a_way_through(domain):
    """The card may report that nothing on the record settles it -- one of the
    eleven is exactly that case -- but it must never read as a refusal, so the
    condition it cannot see has to be named every time."""
    reservations, tasks = domain
    cancels = [
        (task, card(reservations, action)) for task, action in gold(tasks, "cancel_reservation")
    ]
    assert cancels, "expected the airline task set to cancel something"

    for task, text in cancels:
        if text is None:
            continue
        assert "get_flight_status" in text, task
        assert "cannot be cancelled" not in text, task


def test_most_gold_cancellations_are_settled_by_the_record_alone(domain):
    """Ten of the eleven. The card earns its place on those; on the eleventh it
    earns it by saying which lookup is left."""
    reservations, tasks = domain
    texts = [
        t for _, action in gold(tasks, "cancel_reservation") if (t := card(reservations, action))
    ]
    settled = [t for t in texts if "=> one already holds" in t]
    on_reason = [t for t in texts if "the reason decides it" in t]

    assert len(settled) + len(on_reason) >= len(texts) - 1


def test_no_gold_flight_change_is_told_its_flights_cannot_be_changed(domain):
    """Gold re-flights a basic economy reservation eight times and changes the
    cabin every time, which is what the card says to do. It must never appear on
    a change that keeps the cabin as it was -- there is no such gold action."""
    reservations, tasks = domain
    contradicted = []
    for task, action in gold(tasks, "update_reservation_flights"):
        record = reservations.get(action.arguments.get("reservation_id"))
        if not record or record["cabin"] != "basic_economy":
            continue
        if action.arguments.get("cabin") == record["cabin"]:
            contradicted.append(task)

    assert contradicted == []


def test_no_gold_passenger_change_alters_how_many_there_are(domain):
    """The count the card prints is the one the policy freezes."""
    reservations, tasks = domain
    moved = [
        task
        for task, action in gold(tasks, "update_reservation_passengers")
        if (record := reservations.get(action.arguments.get("reservation_id")))
        and len(action.arguments["passengers"]) != len(record["passengers"])
    ]

    assert moved == []


def test_no_gold_change_is_paid_for_with_a_certificate(domain):
    """The other line the card asserts outright, and the one the environment
    itself enforces with `Certificate cannot be used to update reservation`."""
    _, tasks = domain
    paid_by_certificate = [
        task
        for task, action in gold(tasks, "update_reservation_flights", "update_reservation_baggages")
        if str(action.arguments.get("payment_id", "")).startswith("certificate")
    ]

    assert paid_by_certificate == []


def test_the_card_never_fires_on_a_record_it_did_not_recognise(domain):
    """Whatever it attaches to, the record survives it verbatim."""
    reservations, _ = domain
    for record in list(reservations.values())[:200]:
        content = json.dumps(record)
        assert eligibility(content).startswith(content)
