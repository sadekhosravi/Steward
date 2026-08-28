"""A trip that has already started cannot be changed or cancelled.

`policy.md:141`, and it is the flattest rule in the document:

    If any portion of the flight has already been flown, the agent cannot help
    and transfer is needed.

`policy.md:116` says the same thing again for cabin changes. Neither is checked
by the API.

Task 41 is what this is for. The customer asks to cancel every reservation with a
single passenger on it; the two that qualify are UDMOP1, which meets no
cancellation ground, and 4XGCCM, whose legs are `landed` and `cancelled` -- it
has flown. Gold cancels nothing at all. The run cancelled seven reservations,
including 4XGCCM and I6M8JQ, both already flown, on nine occasions across the
saved runs.

Decided from the dates on the record rather than from a flight-status lookup.
Status would be better evidence and is almost never fetched -- requiring it would
make this check silent, which is how the cancellation ground for an
airline-cancelled flight ended up unusable. A leg dated before the benchmark's
present has flown, and that needs nothing but the record the assistant already
read.

Strictly before, not on or before. The present is 15 May at 15:00 and a leg dated
15 May may be a morning flight already gone or an evening one still to come; the
record does not say which. Blocking the ambiguous case would be guessing, and
guessing against the action is the failure this whole tier exists to stop.

THE REMEDIATION IS ABOUT ONE RESERVATION, NOT THE CONVERSATION

It said "transfer them to a human agent" full stop, and the actor obeyed exactly:
task 37's customer names three reservations, only one of which has flown, and the
run transferred before doing the single upgrade that was the whole answer key.
That is the shape this domain keeps producing -- a customer with several records,
one of them untouchable -- and the only run of task 37 that ever scored did the
allowed work first and transferred afterwards. Transfer ends the conversation, so
it has to be the last thing, and the wording has to say so.
"""

from __future__ import annotations

from datetime import datetime

from core.verifiers import Evidence, Finding

from .records import reservations

__all__ = ["not_yet_flown"]

CHECK = "not_yet_flown"

# The benchmark's fixed present, to the day. Restated rather than imported so
# this module does not depend on a card.
TODAY = datetime(2024, 5, 15).date()


def not_yet_flown(call, evidence: Evidence) -> Finding | None:
    """Whether any leg of this reservation is already behind us."""
    arguments = getattr(call, "arguments", {}) or {}
    record = reservations(evidence.observed).get(arguments.get("reservation_id"))
    if record is None:
        return None

    flown = [leg for leg in (record.get("flights") or []) if _past(leg.get("date"))]
    if not flown:
        return None
    when = ", ".join(sorted({str(leg.get("date")) for leg in flown}))
    return Finding(
        check=CHECK,
        reason=(
            f"This reservation has already been partly flown -- it has a flight dated "
            f"{when}. The policy says that when any portion has been flown the agent "
            "cannot help and the customer must be transferred."
        ),
        remediation=(
            "Do not change or cancel this reservation. Tell the customer that part of "
            "this trip has already been flown, so a human agent has to handle this one. "
            "First finish everything else they asked for that is allowed -- their other "
            "reservations are unaffected. Only once nothing else is left to do, call "
            "transfer_to_human_agents."
        ),
    )


def _past(date: object) -> bool:
    """Strictly before today. An unreadable date is not a flown one."""
    if not isinstance(date, str):
        return False
    try:
        return datetime.fromisoformat(date).date() < TODAY
    except ValueError:
        return False
