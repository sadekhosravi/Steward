"""No verifier may refuse a write the benchmark's own answer key makes.

The bar for letting code stop an action, and it is zero rather than a rate. There
are 49 gold writes across the fifty airline tasks and every one of them is
correct by definition, so a single block here is a defect with a task id attached
-- something to read and fix, not a false-positive rate to trade against recall.

This is the same shape as `test_money_against_gold.py`, which pins the payment
card at 38/38, and it exists for the same reason: precision measured once decays
silently. `scripts/gate_bench.py` reports the number against a much larger corpus
including surplus writes; this pins the half that must never move.
"""

from __future__ import annotations

import json

import pytest

from adapters.tau2.verifiers import PANEL
from core.state import PendingCall
from core.verifiers import Evidence, first

pytest.importorskip("tau2.domains.airline.environment")

WRITES = {
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "cancel_reservation",
    "send_certificate",
}


def text_of(result):
    """A tool result as tau2 serialises it -- pydantic's JSON, not a repr."""
    if isinstance(result, str):
        return result
    try:
        from pydantic_core import to_json

        return to_json(result).decode()
    except Exception:
        return json.dumps(result, default=str)


@pytest.fixture(scope="module")
def replayed():
    """Gold's writes, each with the evidence a run would have had at that point.

    Walked in order, executing gold's own actions as it goes, because several of
    these checks read the record *as it now stands*: task 7 upgrades a basic
    economy reservation to business and then cancels it, and a ledger built from
    the reads alone would still say basic economy.
    """
    from tau2.domains.airline.environment import get_environment, get_tasks

    try:
        tasks = get_tasks()
    except (OSError, ValueError) as missing:
        pytest.skip(f"tau2 airline data unavailable: {missing}")

    proposals = []
    for task in tasks:
        criteria = task.evaluation_criteria
        actions = (criteria.actions if criteria else None) or []
        if not any(action.name in WRITES for action in actions):
            continue
        environment = get_environment()
        observed = [str(task.user_scenario.instructions)]
        looked_up = []
        committed = []
        for action in actions:
            arguments = dict(action.arguments or {})
            if action.name in WRITES:
                proposals.append(
                    (
                        task.id,
                        PendingCall(id="p", name=action.name, arguments=arguments),
                        Evidence.of(list(observed), "", list(committed), list(looked_up)),
                    )
                )
            try:
                result = environment.make_tool_call(
                    tool_name=action.name, requestor=action.requestor, **arguments
                )
            except Exception:  # gold this build cannot replay
                continue
            text = text_of(result)
            observed.append(text)
            if action.name in WRITES:
                committed.append(action.name)
            else:
                looked_up.append((action.name, arguments, text))
    return proposals


def test_every_gold_write_is_worth_replaying(replayed):
    """A guard on the fixture. If the walk silently stopped finding gold's
    writes, every assertion below would pass by having nothing to check."""
    assert len(replayed) >= 45


def test_no_verifier_blocks_a_write_gold_makes(replayed):
    refused = [
        (task, call.name, finding.check)
        for task, call, evidence in replayed
        if (finding := first(call, evidence, PANEL)) is not None
    ]

    assert refused == []
