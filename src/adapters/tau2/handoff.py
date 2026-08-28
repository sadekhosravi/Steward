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

__all__ = ["work_still_owed"]


def work_still_owed(call, evidence: Evidence) -> Finding | None:
    """Refuse a handoff while the plan still has a change nobody has made."""
    if not evidence.owed:
        return None
    return Finding(
        check="work_still_owed",
        reason=(
            "This hands the customer to a human while the turn still owes "
            f"{_listed(evidence.owed)}. Transferring ends the conversation, so that "
            "work is not postponed by it -- it is abandoned."
        ),
        remediation=(
            f"Do not transfer. Make the outstanding call instead: {_listed(evidence.owed)}. "
            "If part of what the customer asked for is not permitted, refuse that part in "
            "your own words and carry out the rest -- both belong to the same turn."
        ),
        # The assistant can do this alone and now: the change is one it planned,
        # on a record it has already read. Nothing here waits on the customer.
        recoverable=True,
    )


def _listed(owed: tuple[tuple[str, str | None], ...]) -> str:
    """`book_reservation` and `cancel_reservation on HATXYZ`, in the assistant's terms."""
    return ", ".join(f"{tool} on {record}" if record else tool for tool, record in owed)
