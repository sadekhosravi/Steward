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

from core.verifiers import Panel

from .cancellable import cancellable
from .compensation import compensation
from .flown import not_yet_flown
from .handoff import work_still_owed
from .intended import intended, read_first
from .modifications import (
    baggage_only_grows,
    flights_changeable,
    passenger_count_fixed,
)
from .payment import payment_composition, payment_for_change

__all__ = ["PANEL", "SELECTION"]

PANEL = Panel(
    verifiers={
        # Not a write, and gated all the same -- `agent.HANDOFF` folds it in
        # because the question the gate asks is whether an action can be taken
        # back, and a transfer ends the conversation. It is the only entry here
        # whose rule is about the turn rather than about the record.
        "transfer_to_human_agents": [work_still_owed],
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

# Kept apart from `PANEL` because of what it costs, not what it decides. Every
# verifier above is a pure function of things already on disk; `intended` needs
# `Evidence.stated`, which one model call has to produce first. Running it in the
# same table would pay for that extraction on proposals a free check was about to
# refuse anyway -- 62 of the 198 in the labelled corpus never reach a model at all.
#
# It is also the only stage here that is off by default. Measured, it blocks no
# gold write in either source and catches 2 surplus ones for 136 model calls;
# `agents.selector` records why that is kept rather than deleted.
#
# `book_reservation` is absent and always will be: it creates the record it is
# about, so there is no existing reservation to have described.
SELECTION = Panel(
    verifiers={
        "cancel_reservation": [intended],
        "update_reservation_flights": [intended],
        "update_reservation_passengers": [intended],
        "update_reservation_baggages": [intended],
    }
)
