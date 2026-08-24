"""What a reservation costs, worked out at the moment the record is read.

The sibling of `ranking`, and the same bargain: a fact the actor keeps getting
wrong on its own is computed here from the environment's own numbers and appended
to the result, without touching the rows it was appended to.

The evidence is the four tasks that lost COMMUNICATE in the 30-simulation
selection run. Five of the seven runs that got the *database* exactly right were
scored zero anyway, on a number said in a sentence:

  Task 23 needed 1286. The reply laid out its own operands -- $44, $621, $621, in
  a table -- and the total line underneath them read $1,186. Earlier in the same
  conversation it called a three-passenger, three-leg itinerary "a total of
  $189.00", which is the per-passenger fare; the reservation cost $567.

  Task 11 needed 5244. That is 5700 - 456, and both operands are in the reply.
  It said the refund was $1,010.

  Task 18 needed 23553, the saving across six reservations. The reply priced them
  in a table -- "LQ940Q  $4,200", "2FBBAH  $3,900" -- and every one of those fares
  is invented. The record says 503 and 4390. It later admitted it had "used the
  average business fare".

Those are three different faults and only one of them is arithmetic. Task 23
confuses the per-passenger fare with what the booking cost; task 18 does not read
the fares at all. Both are answered by putting the multiplication in front of the
actor instead of leaving it to be done in prose, which is why this computes the
total and the fare it came from and shows the working.

WHAT IT DELIBERATELY DOES NOT DO

It does not price the segments in any other cabin. That is the other half of both
downgrade tasks -- task 11's 456, task 18's 2323 -- and it is not in the record:
it would have to be read out of the flights table directly. Doing that would put
numbers in front of the actor that no lookup of its own ever returned, which is
the one thing the provenance ledger exists to prevent, and it would be reading
scored environment state from outside the trajectory. Those prices are already
reachable the honest way, through a search, where `ranking` meets them.

It also does not reconcile the total against what was paid. The two differ
routinely and for good reasons -- insurance at $30 a passenger, a fare that has
moved since booking -- so both are shown, separately and labelled, and which one
the question is about is left to the actor. A note that guessed at the difference
would be the same fabrication this module exists to remove.
"""

from __future__ import annotations

import json

__all__ = ["totals"]

HEADING = (
    "WHAT THIS RESERVATION COSTS\n"
    "Worked out from the record above. Use these figures as they stand rather "
    "than re-deriving them, and note that the fare is per passenger while the "
    "total is for the whole booking."
)


def totals(content: str) -> str:
    """The same tool result, with its arithmetic appended when it has any.

    Returns `content` untouched for anything that is not a single reservation
    with priced flights and passengers on it -- a search, a user profile, an
    error, a shape from a domain nobody has looked at yet.
    """
    record = _reservation(content)
    if record is None:
        return content
    fares, people = record
    fare = sum(fares)
    working = f"  ({' + '.join(_money(f) for f in fares)})" if len(fares) > 1 else ""
    lines = [
        f"  fare per passenger: {_money(fare)}{working}",
        f"  for {people} passenger{'s' * (people != 1)}:"
        f" {_money(fare)} x {people} = {_money(fare * people)}",
    ]
    paid = _paid(content)
    if paid is not None:
        lines.append(f"  paid so far, from payment_history: {_money(paid)}")
    return f"{content}\n\n{HEADING}\n" + "\n".join(lines)


def _reservation(content: str) -> tuple[list[float], int] | None:
    """The leg prices and the passenger count, or None if this is not a booking.

    Recognised by shape rather than by which tool was called, so it holds for
    every result that carries a reservation back -- the lookup, the booking, and
    each of the updates -- and stays quiet on everything else. A record missing
    any single leg price is left alone: a sum over some of the legs is worse than
    no sum, because it is wrong and looks authoritative.
    """
    try:
        record = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict):
        return None
    flights, passengers = record.get("flights"), record.get("passengers")
    if not isinstance(flights, list) or not flights:
        return None
    if not isinstance(passengers, list) or not passengers:
        return None
    fares = [f.get("price") if isinstance(f, dict) else None for f in flights]
    if any(not isinstance(f, (int, float)) or isinstance(f, bool) for f in fares):
        return None
    return [float(f) for f in fares], len(passengers)  # type: ignore[arg-type]


def _paid(content: str) -> float | None:
    """What `payment_history` adds up to, or None when it is absent or unreadable.

    A refund is recorded there as a negative amount, so this is the running
    balance of the booking rather than the sum of what was ever charged.
    """
    record = json.loads(content)
    history = record.get("payment_history")
    if not isinstance(history, list) or not history:
        return None
    amounts = [h.get("amount") if isinstance(h, dict) else None for h in history]
    if any(not isinstance(a, (int, float)) or isinstance(a, bool) for a in amounts):
        return None
    return float(sum(amounts))  # type: ignore[arg-type]


def _money(amount: float) -> str:
    """`$5,700` for a whole number of dollars, `$5,700.50` when there are cents."""
    return f"${amount:,.2f}".removesuffix(".00")
