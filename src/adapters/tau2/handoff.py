"""When a transfer to a human is abandonment rather than help.

Handing off is usually right in this domain, and the numbers are lopsided enough
that it is worth stating before the rule: across the 150 simulations of the arm C
run, 51 transferred and 42 of those scored 1.00. Most of this task set is made of
requests the policy does not permit, and refusing them and handing over is the
correct ending. Nothing here may disturb that.

The other nine are one shape. Splitting the transfers by whether the task needed
a write at all:

    transfers on tasks needing no write   41, of which 41 scored 1.00
    transfers on tasks needing a write    10, of which  9 scored 0.00

So the discriminator is not the transfer, and it is not whether the customer
asked for a person -- that was true in only 12 of the 51 and separates nothing.
It is whether work was still owed. Task 24 is the type, and it fails this way in
all three trials: the customer asks for a passenger to be removed and, separately,
for a round trip to be booked. The removal is not permitted, the assistant is
right to refuse it, and it then transfers -- carrying the booking, which is the
only write gold makes, out of the conversation with it.

`agents.gate` already states this rule in prose, and states it well: "a handoff
proposed while a change is outstanding is refused". It let ten through anyway.
That is the finding `core.verifiers` was built on -- a rule a 20B model has to
recognise in a paragraph is a rule applied at roughly its base rate -- and the
answer is the same here as everywhere else in this package: make it arithmetic.

WHAT THIS DOES NOT DO

It does not ask whether the transfer is justified, whether the customer would
prefer a person, or whether the refusal that preceded it was correct. Those are
judgments and they stay with the critic. This compares two lists.
"""

from __future__ import annotations

from core.verifiers import Evidence, Finding

from .modifications import BASIC
from .records import reservations

__all__ = ["work_still_owed"]


def work_still_owed(call, evidence: Evidence) -> Finding | None:
    """Refuse a handoff while the plan still has a change nobody has made."""
    owed = tuple(pair for pair in evidence.owed if not _barred(*pair, evidence))
    if not owed:
        return None
    return Finding(
        check="work_still_owed",
        reason=(
            "This hands the customer to a human while the turn still owes "
            f"{_listed(owed)}. Transferring ends the conversation, so that "
            "work is not postponed by it -- it is abandoned."
        ),
        remediation=(
            f"Do not transfer. Make the outstanding call instead: {_listed(owed)}. "
            "If part of what the customer asked for is not permitted, refuse that part in "
            "your own words and carry out the rest -- both belong to the same turn."
        ),
        # The assistant can do this alone and now: the change is one it planned,
        # on a record it has already read. Nothing here waits on the customer.
        recoverable=True,
    )


def _barred(tool: str, record: str | None, evidence: Evidence) -> bool:
    """Is this change one the record itself puts out of reach?

    The reason this check was reverted once, and the reason it is back with a
    filter rather than without one. The planner writes a change down whenever the
    customer *asks* for it, including where the policy forbids it -- and a change
    that can never be carried out is owed for the rest of the conversation, so a
    rule that blocks on "work is owed" wedges the exit shut on exactly the tasks
    whose right ending is a handoff.

    Task 13 is the whole of it: a one-stop ATL-LAX basic economy reservation, and
    the customer wants it re-routed to LAS. The policy forbids that twice over,
    gold's own action is a transfer, and both trials score 1.00 today. Without
    this filter the check fires there and takes them.

    Measured over the transfers proposed in the 15x2 of 2026-08-29, with the
    ledger anchored: unfiltered it fires 27 times where gold does not transfer and
    2 where it does -- both of those being task 13. Filtered it fires 27 and 0.
    The bar costs nothing and removes the whole of the known error.

    Only what the record settles on its own. `modifications.flights_changeable`
    knows this rule already and cannot be reused here, because it compares
    *proposed* legs against the record and a planned change carries none -- the
    call it would judge was never made. So the same fact is read the only way it
    can be at this point: from the cabin the reservation is in.
    """
    if tool != "update_reservation_flights" or not record:
        return False
    reservation = reservations(evidence.observed).get(record)
    return reservation is not None and str(reservation.get("cabin", "")).lower() == BASIC


def _listed(owed: tuple[tuple[str, str | None], ...]) -> str:
    """`book_reservation` and `cancel_reservation on HATXYZ`, in the assistant's terms."""
    return ", ".join(f"{tool} on {record}" if record else tool for tool, record in owed)
