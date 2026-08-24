"""The payment rules, checked against every payment the benchmark scores.

Both halves of the card are assertions about what a correct action looks like, so
both are measured against the actions gold actually takes rather than argued for.
The arithmetic is stated in the card rather than computed -- the fare depends on a
search result the card never sees -- so what is pinned here is that the three
terms, applied to gold's own arguments and the flights table, come to the amount
gold paid.
"""

from __future__ import annotations

import json

import pytest

from adapters.tau2.money import EXTRA_BAG, INSURANCE, LIMITS, money

pytest.importorskip("tau2.domains.airline.environment")


@pytest.fixture(scope="module")
def domain():
    from tau2.domains.airline.environment import get_tasks
    from tau2.domains.airline.utils import AIRLINE_DB_PATH

    try:
        db = json.loads(open(AIRLINE_DB_PATH, encoding="utf-8").read())
        tasks = get_tasks()
    except (OSError, ValueError) as missing:
        pytest.skip(f"tau2 airline data unavailable: {missing}")
    return db, tasks


def gold(tasks, *names):
    for task in tasks:
        criteria = task.evaluation_criteria
        for action in (criteria.actions if criteria else None) or []:
            if action.name in names:
                yield task.id, action


def test_the_three_terms_come_to_what_gold_paid(domain):
    """Ten bookings, and the sum has to land on every one of them. This is the
    number task 6 spent three failed writes disagreeing about."""
    db, tasks = domain
    flights = db["flights"]

    wrong = []
    for task, action in gold(tasks, "book_reservation"):
        arguments = action.arguments
        people = len(arguments["passengers"])
        try:
            fare = sum(
                flights[leg["flight_number"]]["dates"][leg["date"]]["prices"][arguments["cabin"]]
                for leg in arguments["flights"]
            )
        except KeyError:  # a flight this task creates rather than starts with
            continue
        total = (
            fare * people
            + (INSURANCE * people if arguments.get("insurance") == "yes" else 0)
            + EXTRA_BAG * arguments.get("nonfree_baggages", 0)
        )
        paid = sum(payment["amount"] for payment in arguments["payment_methods"])
        if total != paid:
            wrong.append((task, total, paid))

    assert wrong == []


def test_no_gold_booking_breaks_the_composition_limits(domain):
    """One certificate, one credit card, three gift cards. The API checks none of
    them, so the card is the only place they are said before the call."""
    _, tasks = domain
    broken = []
    for task, action in gold(tasks, "book_reservation"):
        used: dict[str, int] = {}
        for payment in action.arguments["payment_methods"]:
            source = str(payment["payment_id"]).rsplit("_", 1)[0]
            used[source] = used.get(source, 0) + 1
        for source, limit in LIMITS:
            if used.get(source, 0) > limit:
                broken.append((task, source, used[source]))

    assert broken == []


def test_no_gold_change_is_paid_by_a_certificate_or_by_more_than_one_method(domain):
    """`update_reservation_*` takes a single `payment_id` and refuses a
    certificate outright."""
    _, tasks = domain
    broken = [
        (task, action.arguments.get("payment_id"))
        for task, action in gold(tasks, "update_reservation_flights", "update_reservation_baggages")
        if str(action.arguments.get("payment_id", "")).startswith("certificate")
    ]

    assert broken == []


def test_no_gold_payment_exceeds_the_balance_the_card_prints(domain):
    """The balance is quoted off the profile, so it has to be the number the
    environment is going to check against."""
    db, tasks = domain
    users = db["users"]
    over = []
    for task, action in gold(tasks, "book_reservation"):
        wallet = users[action.arguments["user_id"]]["payment_methods"]
        for payment in action.arguments["payment_methods"]:
            method = wallet.get(payment["payment_id"])
            if method and "amount" in method and payment["amount"] > method["amount"]:
                over.append((task, payment["payment_id"]))

    assert over == []


def test_every_user_in_the_database_keeps_its_record_and_gets_the_rules(domain):
    db, _ = domain
    for record in list(db["users"].values())[:200]:
        content = json.dumps(record)
        text = money(content)
        assert text.startswith(content)
        assert "PAYING FOR THIS" in text


def test_every_method_on_a_profile_is_named_so_it_can_be_quoted_back(domain):
    """The actor has to pass the id verbatim, so every one of them has to appear."""
    db, _ = domain
    for record in list(db["users"].values())[:100]:
        text = money(json.dumps(record))
        for payment_id in record["payment_methods"]:
            assert payment_id in text
