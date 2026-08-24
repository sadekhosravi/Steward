"""The allowance, checked against every bag figure the benchmark scores.

The card is only worth appending if the row it works out is the row gold wrote.
That is a property of the code *and* this domain's data, so it is measured
against the data and pinned here rather than argued for.

The cabin matters and is the whole reason this exists, so gold's own actions are
replayed in order: a task that changes a reservation's cabin before touching its
bags is scored against the cabin it changed it to, which is what the card sees at
that moment because every write tool returns the updated reservation.
"""

from __future__ import annotations

import json

import pytest

from adapters.tau2.baggage import FREE, bags

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


def written(db, tasks):
    """Every gold write that sets `nonfree_baggages`, with the membership, cabin
    and passenger count in force when gold made it."""
    reservations, users = db["reservations"], db["users"]
    for task in tasks:
        criteria = task.evaluation_criteria
        cabins: dict[str, str] = {}
        for action in (criteria.actions if criteria else None) or []:
            arguments = action.arguments
            if action.name == "update_reservation_flights" and arguments.get("cabin"):
                cabins[arguments["reservation_id"]] = arguments["cabin"]
            if action.name == "book_reservation":
                user_id, cabin = arguments["user_id"], arguments["cabin"]
                people = len(arguments["passengers"])
            elif action.name == "update_reservation_baggages":
                record = reservations.get(arguments["reservation_id"])
                if record is None:
                    continue
                user_id = record["user_id"]
                cabin = cabins.get(arguments["reservation_id"], record["cabin"])
                people = len(record["passengers"])
            else:
                continue
            yield task.id, users[user_id]["membership"], cabin, people, arguments


def test_the_row_the_card_works_out_is_the_one_gold_wrote(domain):
    """Fifteen of them, and the card has to agree with every one. This is the
    number the whole part rests on."""
    db, tasks = domain
    checked = list(written(*domain))
    assert checked, "expected the airline task set to write some bag figures"

    wrong = []
    for task, tier, cabin, people, arguments in checked:
        free = FREE[tier][cabin] * people
        computed = max(0, arguments.get("total_baggages", 0) - free)
        if computed != arguments.get("nonfree_baggages", 0):
            wrong.append((task, tier, cabin, people, arguments))

    assert wrong == []


def test_the_stale_cabin_is_what_would_break_it(domain):
    """The guard on the guard. Scored against the cabin each reservation was
    booked in rather than the one in force, the same formula disagrees with gold
    -- so the test above is really testing the thing this module claims."""
    db, tasks = domain
    reservations, users = db["reservations"], db["users"]

    wrong = 0
    for task in tasks:
        criteria = task.evaluation_criteria
        for action in (criteria.actions if criteria else None) or []:
            if action.name != "update_reservation_baggages":
                continue
            record = reservations.get(action.arguments["reservation_id"])
            if record is None:
                continue
            free = FREE[users[record["user_id"]]["membership"]][record["cabin"]] * len(
                record["passengers"]
            )
            stale = max(0, action.arguments.get("total_baggages", 0) - free)
            wrong += stale != action.arguments.get("nonfree_baggages", 0)

    assert wrong > 0


def test_every_reservation_in_the_database_keeps_its_record(domain):
    db, _ = domain
    for record in list(db["reservations"].values())[:200]:
        content = json.dumps(record)
        assert bags(content).startswith(content)


def test_every_user_in_the_database_is_given_a_row(domain):
    """Three tiers, and the card has to recognise all of them."""
    db, _ = domain
    for record in list(db["users"].values())[:200]:
        content = json.dumps(record)
        text = bags(content)
        assert text.startswith(content)
        assert f"{record['membership']} membership:" in text
