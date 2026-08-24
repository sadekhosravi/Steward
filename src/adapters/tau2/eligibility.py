"""Which policy conditions a reservation meets, worked out when it is read.

The third of the siblings, after `ranking` and `totals`, and the same bargain: a
fact both the actor and the critic keep getting wrong is computed here from the
record itself and appended to the result, leaving the rows exactly as the
environment wrote them.

The evidence is the gate, from the 50-task run. It refused a write that was
byte-identical to a gold action fourteen times across nine tasks, and its own
words say why:

  Task 42 -- "cancellation is permitted only if it was booked within the last 24
  hours, the airline has cancelled it, it is a business cabin reservation, or the
  user has travel insurance covering the reason; HSR97W is a business-class
  reservation booked long ago without insurance, so cancellation is disallowed."
  HSR97W is business cabin. The rule is quoted correctly and applied backwards in
  the same sentence.

  Task 39 -- "only reservation 8C8K4E meets this rule" -- while refusing the call
  that cancels 8C8K4E.

  Task 30 -- "missing required fields (origin, destination, price)". Those fields
  are not in that tool's schema. Ten refusals, two escalations, task lost.

A four-way disjunction over four fields on the record is not a judgement, and it
is being got wrong at a rate that costs whole tasks. So it stops being a
judgement. Both the actor and the critic read the same tool result, which is why
appending here fixes two readers for the price of one.

The same holds on the other side. The policy says basic economy flights cannot be
modified, and it also says any reservation may change cabin; the gate cited basic
economy in twenty-two refusals. Gold calls `update_reservation_flights` on a basic
economy reservation eight times, and every one of them changes the cabin at the
same time. That is the shape of the rule, and it is stated here rather than left
to be rediscovered.

WHAT THIS DELIBERATELY DOES NOT DO

It never says whether to cancel. It says which of the four conditions the record
already settles, and names the two it cannot: whether the airline cancelled a
flight, which lives in the flights table and is reachable through
`get_flight_status`, and whether the reason the customer gave is one insurance
covers, which lives in the conversation. A block that answered either from here
would be inventing what it does not have -- and the whole reason the gate is
worth correcting is that it did exactly that.

It is also not a check. Nothing here can refuse a call or hold a reply. On every
gold action in the run it reports facts that agree with what gold did: none of the
thirty-one reservations gold cancels or re-flights has a leg dated before the
current time, none of the twenty flight updates changes a basic economy itinerary
without also changing its cabin, and ten of the eleven reservations gold cancels
already meet a condition on the record alone. The eleventh is reported as needing
the lookup, not as forbidden.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

__all__ = ["eligibility"]

# policy.md line 3: "The current time is 2024-05-15 15:00:00 EST." A fact about
# this domain's fixture, in the same class as the handoff tool's name -- the
# airline database is static and every task is written against this instant.
# `tests/test_eligibility.py` reads the policy and fails if the two ever part.
NOW = datetime(2024, 5, 15, 15, 0, 0)

DAY = timedelta(hours=24)

# The cabins this domain has. Used only to recognise a reservation by its shape,
# so a result carrying some other object's `cabin` is left alone.
CABINS = ("basic_economy", "economy", "business")

CANCELLED = (
    "WHETHER THIS RESERVATION CAN BE CANCELLED\n"
    "  It is already cancelled. There is nothing left to cancel on it."
)

HEADING = (
    "WHETHER THIS RESERVATION CAN BE CANCELLED\n"
    "The policy permits it when any one of four conditions holds. Checked against "
    f"the record above, at the current time of {NOW:%Y-%m-%d %H:%M:%S} EST:"
)

SETTLED = "  => one already holds, so the policy's test is met on that ground alone."

ON_THE_REASON = "  => none holds outright; with insurance on file the reason decides it."

UNSETTLED = "  => none holds on this record. Only the lookup above could settle it."

FLOWN = (
    "  Note: a leg here is dated {dates}, already past. If any portion has been "
    "flown the policy stops every change to it, cancellation included."
)

# The one modification rule worth stating, because it reads as a contradiction
# and the run turned on it: basic economy flights cannot be changed, and every
# reservation may change cabin. Gold re-flights a basic economy reservation eight
# times and changes the cabin every time. The rest of the modification rules are
# in the policy the model already has, and repeating them here costs a multiple of
# this block on every record read -- see the note on length below.
BASIC = (
    "  Basic economy flights cannot be changed while it stays basic economy. Any "
    "reservation may change cabin, and update_reservation_flights is the call that "
    "does it -- pass the new cabin with the flights."
)

PAYMENT = "  A change is paid for by one gift card or credit card; a certificate cannot."


def eligibility(content: str) -> str:
    """The same tool result, with the conditions it settles appended.

    Returns `content` untouched for anything that is not a single reservation --
    a search, a user profile, an error, a shape from a domain nobody has looked
    at yet.
    """
    record = _reservation(content)
    if record is None:
        return content
    if record.get("status") == "cancelled":
        return f"{content}\n\n{CANCELLED}"
    return f"{content}\n\n{_cancelling(record)}\n\n{_changing(record)}"


def _reservation(content: str) -> dict | None:
    """The record, or None if this result is not one reservation.

    Recognised by the fields the two cards actually read, rather than by which
    tool was called, so it holds for the lookup, the booking and each of the
    updates alike -- they all return the same object -- and stays quiet on
    everything else. A record missing any of them is left alone: a card with a
    blank where a condition should be is worse than no card, because the blank
    reads as a "no".
    """
    try:
        record = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict):
        return None
    if not isinstance(record.get("cabin"), str) or record["cabin"] not in CABINS:
        return None
    if not isinstance(record.get("reservation_id"), str):
        return None
    if not isinstance(record.get("created_at"), str):
        return None
    if not isinstance(record.get("passengers"), list) or not record["passengers"]:
        return None
    return record


def _cancelling(record: dict) -> str:
    """The four conditions, each answered from the record or handed back."""
    fresh = _booked_within_a_day(record["created_at"])
    business = record["cabin"] == "business"
    insured = record.get("insurance") == "yes"
    lines = [
        f"  booked in the last 24 hours: {_answer(fresh)} -- created {record['created_at']}",
        f"  business cabin: {_answer(business)} -- cabin is {record['cabin']}",
        f"  travel insurance on file: {_answer(insured)}"
        + (" -- it covers health and weather reasons, so the reason decides it" if insured else ""),
        "  cancelled by the airline: not on this record -- get_flight_status will say.",
    ]
    if fresh or business:
        lines.append(SETTLED)
    elif insured:
        lines.append(ON_THE_REASON)
    else:
        lines.append(UNSETTLED)
    past = _past_legs(record)
    if past:
        lines.append(FLOWN.format(dates=", ".join(past)))
    return f"{HEADING}\n" + "\n".join(lines)


def _changing(record: dict) -> str:
    """The modification rules this record makes non-obvious, and nothing else.

    Length is a cost paid on every reservation the actor reads, and a task that
    reads six of them pays it six times. So this states the rule that reads as a
    contradiction and the one the environment itself enforces, and leaves the
    rest of the policy where the model already has it.
    """
    rules = [BASIC] if record["cabin"] == "basic_economy" else []
    return "\n".join([*rules, PAYMENT])


def _booked_within_a_day(created_at: str) -> bool | None:
    """Whether the booking is less than 24 hours old, or None if the stamp is not
    a timestamp this can read. Unreadable is reported as unreadable, never as no."""
    try:
        booked = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        return None
    return NOW - booked.replace(tzinfo=None) < DAY


def _past_legs(record: dict) -> list[str]:
    """Dates on this reservation that are already behind the current time.

    Only a flag. Whether a leg was flown is a fact about the flight and not about
    the booking, so this points at the lookup rather than answering for it. It
    fires on no reservation that gold cancels or re-flights.
    """
    flights = record.get("flights")
    if not isinstance(flights, list):
        return []
    dates = {
        leg["date"] for leg in flights if isinstance(leg, dict) and isinstance(leg.get("date"), str)
    }
    return sorted(date for date in dates if date < f"{NOW:%Y-%m-%d}")


def _answer(held: bool | None) -> str:
    """`yes`, `no`, or an admission. Three answers, because two would be a guess."""
    if held is None:
        return "cannot be read from this record"
    return "yes" if held else "no"
