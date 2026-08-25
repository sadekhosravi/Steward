"""Whether the policy permits cancelling this reservation, decided from the record.

`policy.md:143-147` is a four-way disjunction and the API checks none of it:

    Otherwise, flight can be cancelled if any of the following is true:
    - The booking was made within the last 24 hrs
    - The flight is cancelled by airline
    - It is a business flight
    - The user has travel insurance and the reason for cancellation is covered

    The API does not check that cancellation rules are met, so the agent must
    make sure the rules apply before calling the API!

`eligibility.py` already computes the first three and prints them on the record
when it is read, which fixed nothing on its own -- the run reads the card, agrees
with it, and cancels anyway when the customer presses. Twelve of the thirty-seven
single writes that lost an otherwise-won task were cancellations, the largest
group of all.

WHEN THIS BLOCKS, AND WHEN IT KEEPS QUIET

Only when the record settles every ground *against* the cancellation. An unknown
is not a no: the fourth ground turns on a reason the customer gives and the
second on a flight status this conversation may never have looked up, so either
one being unestablished means this check has nothing to say and the judges get
the case instead. That asymmetry is the whole discipline here. The monolithic
critic refuses on unknowns -- "no flight status was provided", "the reservation
id was never established" -- and it is a large part of why it refuses 41% of
gold's own writes while refusing only 46% of the surplus ones.

THE CASE THAT SHAPES THE CODE

Gold's task 7 cancels XEHM4B, which is basic economy, booked on 1 May, uninsured,
with both legs available. All four grounds read false and the cancellation is
still correct -- because gold *upgrades it to business first* and then cancels.
So the record this reads has to be the record as it now stands, not as it was
first read, which is why `records.reservations` keeps the newest version of each
and why a write's own result belongs in the evidence ledger. Computed against the
first read of each reservation, this check refuses gold. Against the current one,
it does not.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from core.verifiers import Evidence, Finding

from .records import reservations, statuses

__all__ = ["cancellable"]

CHECK = "cancellable"

TOOL = "cancel_reservation"

# The benchmark's fixed present. `eligibility.NOW` is the same instant; it is
# restated rather than imported so this module does not depend on a card.
NOW = datetime(2024, 5, 15, 15, 0, 0)

DAY = timedelta(hours=24)

FORBIDDEN = Finding(
    check=CHECK,
    reason=(
        "The policy allows a cancellation only if the booking was made in the last 24 "
        "hours, the airline cancelled a flight on it, it is a business cabin "
        "reservation, or insurance covers the reason. This reservation was booked more "
        "than 24 hours ago, is not business cabin, has no insurance, and no flight on "
        "it is cancelled."
    ),
    remediation=(
        "Do not cancel this reservation. Tell the customer that the policy does not "
        "allow it and why. If they want to cancel a business cabin reservation, they "
        "may upgrade the cabin first and then cancel."
    ),
)


def cancellable(call, evidence: Evidence) -> Finding | None:
    """Whether every ground for cancelling is settled against this reservation."""
    if getattr(call, "name", "") != TOOL:
        return None
    arguments = getattr(call, "arguments", {}) or {}
    record = reservations(evidence.observed).get(arguments.get("reservation_id"))
    if record is None:
        # Never read. That is a provenance question and a different check's; a
        # cancellation this one cannot see the record for is not one it can rule on.
        return None

    if str(record.get("cabin", "")).lower() == "business":
        return None
    if str(record.get("insurance", "")).lower() != "no":
        return None  # insured, or unstated -- the reason decides and this cannot read it
    if _within_a_day(record.get("created_at")) is not False:
        return None

    known = statuses(evidence.looked_up)
    legs = record.get("flights") or []
    for leg in legs:
        if known.get(leg.get("flight_number")) == "cancelled":
            return None
    return FORBIDDEN if legs else None


def _within_a_day(created_at: object) -> bool | None:
    """True, False, or None when the timestamp is missing or unreadable.

    Three-valued on purpose: a `created_at` this cannot parse must not be read as
    an old booking, because that is the reading that blocks.
    """
    if not isinstance(created_at, str):
        return None
    try:
        return datetime.fromisoformat(created_at) >= NOW - DAY
    except ValueError:
        return None
