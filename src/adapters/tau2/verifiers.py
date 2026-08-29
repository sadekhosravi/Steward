"""Which deterministic checks apply to which of this domain's writes.

The routing table, and the only place that knows both halves. `core.verifiers`
knows what a check is and nothing about airlines; the modules beside this one
know one rule each and nothing about when they are asked. This says which tool
gets which, and that is all it says -- a router with an opinion is a second
critic, which is the thing being taken apart.

Routing is by tool name, in code. It is tempting to have a model pick the
relevant checks, and it would undo the entire point: the failure being fixed is a
20B model asked to hold every rule at once and decide which of them bears on the
case in front of it.
"""

from __future__ import annotations

from core.state import PendingCall
from core.verifiers import Panel

from .cancellable import cancellable
from .compensation import compensation
from .flown import not_yet_flown
from .intended import read_first
from .modifications import (
    baggage_only_grows,
    flights_changeable,
    passenger_count_fixed,
)
from .payment import payment_composition, payment_for_change

__all__ = ["PANEL", "planned"]

PANEL = Panel(
    verifiers={
        "book_reservation": [payment_composition],
        "send_certificate": [compensation],
        "cancel_reservation": [read_first, not_yet_flown, cancellable],
        "update_reservation_flights": [
            read_first,
            not_yet_flown,
            flights_changeable,
            payment_for_change,
        ],
        "update_reservation_passengers": [read_first, not_yet_flown, passenger_count_fixed],
        "update_reservation_baggages": [
            read_first,
            not_yet_flown,
            baggage_only_grows,
            payment_for_change,
        ],
    }
)


def planned(tool: str, record: str) -> PendingCall:
    """A change nobody has proposed yet, as the call it would be.

    Every write in this domain that lands on an existing record names it in
    `reservation_id`, so a planned change carries enough for the checks that read
    the record and nothing else -- `cancellable`, `not_yet_flown`, `read_first`.
    The ones that compare *proposed* values, like `flights_changeable` and
    `baggage_only_grows`, see no values and fall silent, which is the right answer:
    a change with no arguments yet has not decided anything they could object to.

    That is the whole of the trick. A cancellation is completely described by the
    record it is on, so asking "could this ever be allowed?" of a plan is the same
    question as asking it of the call.
    """
    return PendingCall(id="", name=tool, arguments={"reservation_id": record})
