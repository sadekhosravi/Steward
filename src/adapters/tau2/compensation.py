"""When a certificate may be sent, and for how much.

The single most costly thing this agent does, per call made. Across 220 saved
simulations `send_certificate` was emitted seven times and the answer key never
calls it once -- not on any of the fifty tasks. Every one of those seven cost the
task outright, because a surplus write and a missing gold write lose the database
component identically and there is nothing else in the score to recover it.

Nothing in the run is confused about the tool. It is a sympathetic reflex: the
customer is angry about a delay, asks to be compensated, and the assistant obliges.
The policy is unusually explicit that this is exactly the wrong instinct
(`policy.md:154-167`) --

    Do not proactively offer a compensation unless the user explicitly asks.
    Do not compensate if the user is regular member and has no travel insurance
      and flies (basic) economy.
    Always confirms the facts before offering compensation.
    Only compensate if the user is a silver/gold member or has travel insurance
      or flies business.
    - cancelled flights ... a certificate ... $100 times the number of passengers
    - delayed flights ... *and wants to change or cancel the reservation* ... after
      confirming the facts *and changing or cancelling the reservation* ... $50
      times the number of passengers
    Do not offer compensation for any other reason than the ones listed above.

-- and every clause of it is arithmetic over records the conversation has already
read. That is why this is a verifier and not a question for a model: asked in
prose, a 20B model agrees with the upset customer.

WHAT THIS BLOCKS, ON THE FIVE DISTINCT CALLS THE RUNS MADE

  task 27  $150 and $300, on a delayed flight the customer refuses to change or
           cancel. $300 is the cancellation rate and no flight of theirs is
           cancelled; $150 is the delay rate and the delay clause pays only
           *after* the reservation has been changed or cancelled, which never
           happened. Both refused, on separate clauses.
  task 38  $200 and $100, on a delayed flight. $100 is the cancellation rate with
           nothing cancelled; $200 matches no rate at all.
  task  5  $200, which is the delay rate for four passengers -- allowed only if
           the reservation was changed or cancelled first.

The last clause is doing most of the work, and it is the one that cannot be
answered from the text: whether a change actually went through is a fact about
what this conversation committed, which is why `Evidence` carries it.

WHAT IT DELIBERATELY DOES NOT DO

Block the tool. Sending a certificate is a thing this agent is supposed to be
able to do, and a task set where the answer key never does it is not a licence to
delete the capability -- the rules above are the reason these seven were wrong,
and a certificate that satisfies them passes untouched.

Decide whether the customer asked. "Do not proactively offer" is about who raised
it, which is a question about the conversation and belongs to a judge, not here.
"""

from __future__ import annotations

from core.verifiers import Evidence, Finding

from .records import reservations, statuses, users

__all__ = ["compensation"]

CHECK = "compensation"

TOOL = "send_certificate"

# policy.md:163-165. Per passenger, and which one applies is decided by what the
# customer is complaining about, not by which is larger.
FOR_CANCELLED = 100
FOR_DELAYED = 50

ENTITLED = ("silver", "gold")

# The writes that satisfy "and changing or cancelling the reservation".
SETTLING = ("update_reservation_flights", "cancel_reservation")

UNCONFIRMED = Finding(
    check=CHECK,
    reason=(
        "The policy requires the facts to be confirmed before compensation is offered, "
        "and this conversation has not read the customer's profile or any reservation "
        "of theirs."
    ),
    remediation=(
        "Call get_user_details, then get_reservation_details on the reservation the "
        "customer is complaining about, before offering any certificate."
    ),
)

NOT_ENTITLED = Finding(
    check=CHECK,
    reason=(
        "The policy compensates only a silver or gold member, a customer with travel "
        "insurance, or one flying business, and this customer is none of those."
    ),
    remediation=(
        "Do not send a certificate. Tell the customer that compensation is not "
        "available for their reservation, and ask what else you can help with."
    ),
)


def compensation(call, evidence: Evidence) -> Finding | None:
    """Whether this certificate is one the policy allows, and for the right amount."""
    if getattr(call, "name", "") != TOOL:
        return None
    arguments = getattr(call, "arguments", {}) or {}
    amount = arguments.get("amount")
    if not isinstance(amount, int | float) or isinstance(amount, bool):
        return None  # a malformed argument is the schema's business, not the policy's

    owner = arguments.get("user_id")
    profile = users(evidence.observed).get(owner)
    held = [
        record
        for record in reservations(evidence.observed).values()
        if record.get("user_id") == owner
    ]
    if profile is None and not held:
        return UNCONFIRMED

    member = str((profile or {}).get("membership", "")).lower()
    known = statuses(evidence.looked_up)
    settled = any(name in SETTLING for name in evidence.committed)

    allowed: set[float] = set()
    entitled = False
    for record in held:
        if not _entitled(member, record):
            continue
        entitled = True
        people = len(record.get("passengers") or [])
        for leg in record.get("flights") or []:
            status = known.get(leg.get("flight_number"), "")
            if status == "cancelled":
                allowed.add(FOR_CANCELLED * people)
            elif status == "delayed" and settled:
                allowed.add(FOR_DELAYED * people)

    if held and not entitled:
        return NOT_ENTITLED
    if amount in allowed:
        return None
    return _wrong(amount, allowed, known, settled)


def _entitled(member: str, record: dict) -> bool:
    """policy.md:161. Any one of the three is enough; the record settles two."""
    return (
        member in ENTITLED
        or str(record.get("insurance", "")).lower() == "yes"
        or str(record.get("cabin", "")).lower() == "business"
    )


def _wrong(amount: float, allowed: set[float], known: dict[str, str], settled: bool) -> Finding:
    """A refusal that names the clause it failed, because the actor only gets this.

    Three different mistakes end up here and they need three different fixes, so
    the reason distinguishes them rather than reporting the amount back as though
    the number were the problem.
    """
    delayed = any(status == "delayed" for status in known.values())
    cancelled = any(status == "cancelled" for status in known.values())
    if not delayed and not cancelled:
        return Finding(
            check=CHECK,
            reason=(
                "The policy compensates only for a cancelled or a delayed flight, and "
                "no flight on this customer's reservations has been shown to be either."
            ),
            remediation=(
                "Call get_flight_status on the flight the customer is complaining about "
                "before offering a certificate. If it is neither cancelled nor delayed, "
                "do not send one."
            ),
        )
    if delayed and not settled:
        return Finding(
            check=CHECK,
            reason=(
                "For a delayed flight the policy allows a certificate only after the "
                "reservation has actually been changed or cancelled, and it has not been."
            ),
            remediation=(
                "Do not send a certificate yet. Ask the customer whether they want to "
                "change or cancel the reservation, and only offer one after that is done."
            ),
        )
    return Finding(
        check=CHECK,
        reason=(
            f"The policy sets the certificate at ${FOR_CANCELLED} per passenger for a "
            f"cancelled flight and ${FOR_DELAYED} per passenger for a delayed one; "
            f"${amount:g} is not "
            + (
                "either of those for this reservation."
                if not allowed
                else "the "
                + " or ".join(f"${value:g}" for value in sorted(allowed))
                + " it comes to."
            )
        ),
        remediation=(
            "Count the passengers on the reservation and send "
            + (
                f"${sorted(allowed)[0]:g}"
                if allowed
                else "the amount the policy sets for the number of passengers"
            )
            + ", or do not send a certificate at all."
        ),
    )
