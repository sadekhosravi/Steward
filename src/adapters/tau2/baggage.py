"""How many checked bags are free, worked out when the record is read.

The fourth sibling, after `ranking`, `totals` and `eligibility`, and the same
bargain: a number computed here from the environment's own record rather than
left to be re-derived, appended below the rows without touching them.

The policy states the allowance as a table (policy.md:82-95) -- free bags per
passenger, by the booking user's membership and the cabin -- and then
`nonfree_baggages` is what is left over. That is a lookup and a subtraction, and
both go wrong:

  Task 17. The reservation is basic economy, gold member, one passenger, and the
  turn changes its cabin to economy before adding a third bag. Gold sets
  `nonfree_baggages` to 0, because economy for a gold member is three free. The
  gate refused that call -- "for a gold member in basic economy a single
  passenger has 2 free bags, so adding three bags requires 1 nonfree bag" -- and
  made the actor write 1 instead. The tier was right. The cabin was the one the
  reservation had been booked in, not the one it was in by then.

  Task 21. Silver, economy, one passenger, two bags. Two free, so nothing to pay.
  The run wrote 1.

That is the whole failure: the subtraction is done against a stale cabin. So the
cabin is read off the record as it stands at that moment, which is what makes
this worth computing here rather than anywhere earlier -- every write tool in
this domain returns the updated reservation, so the card that lands immediately
before an `update_reservation_baggages` call has already seen the cabin change.
Against the fifty-task set, replaying gold's own actions in order so the cabin in
force is the one gold put there, the formula reproduces every one of the fifteen
`nonfree_baggages` gold writes.

WHAT IT DELIBERATELY DOES NOT DO

It does not pick a row. A reservation record carries `user_id` but not the
membership that goes with it, and a note that guessed at the tier would be
inventing the one fact it does not have. So all three rows are printed with the
answer already worked out on each, and the reader has only to know which line is
theirs -- which the note on the user record, below, tells them.

It does not say how many bags the customer should have. "Do not add checked bags
that the user does not need" is the sentence next to the table in the policy, and
it is about the conversation, not the record. Two of the run's wrong bookings set
`total_baggages` to bags nobody asked for; nothing here can see that, and a card
that pretended to would be answering a question it cannot hear.
"""

from __future__ import annotations

import json

__all__ = ["bags"]

# policy.md:82-95, free checked bags per passenger. The two axes the policy names,
# in the order it names them.
FREE: dict[str, dict[str, int]] = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}

CABINS = ("basic_economy", "economy", "business")

EXTRA = 50

HEADING = "CHECKED BAGS ON THIS RESERVATION"

STANDING = (
    "  it has {total} in total, {paid} of them paid for, in {cabin} for {people} passenger{s}"
)

DEPENDS = (
    "  free bags depend on the booking user's membership and on the cabin as it\n"
    "  stands now, so if the cabin changes these change with it:"
)

ROW = "    {tier:<8}{free} free -> nonfree_baggages {nonfree}"

WHICH_ROW = (
    "  get_user_details gives the membership. Each bag past the allowance is "
    f"${EXTRA}, and only the increase over the {{paid}} already paid for is charged."
)

ALLOWANCE = (
    "CHECKED BAGS THIS USER GETS FREE\n"
    "  {tier} membership: {rows}, per passenger.\n"
    f"  Each bag past that is ${EXTRA}."
)


def bags(content: str) -> str:
    """The same tool result, with the free allowance appended when it applies.

    Fires on the two records that carry half the answer each -- a reservation,
    which knows the cabin and the passengers, and a user, who knows the tier.
    Returns `content` untouched for anything else.
    """
    record = _record(content)
    if record is None:
        return content
    if isinstance(record.get("membership"), str):
        return f"{content}\n\n{_allowance(record)}"
    return f"{content}\n\n{_standing(record)}"


def _record(content: str) -> dict | None:
    """A user record or a reservation record, or None for anything else.

    A user is recognised by a membership this domain has; a reservation by the
    three fields the arithmetic needs. Anything missing one of them is left
    alone -- a row with a blank in it reads as a zero, and a zero here is a bill.
    """
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("membership") in FREE:
        return parsed
    if parsed.get("cabin") not in CABINS:
        return None
    if not isinstance(parsed.get("passengers"), list) or not parsed["passengers"]:
        return None
    if not _count(parsed, "total_baggages") or not _count(parsed, "nonfree_baggages"):
        return None
    return parsed


def _count(record: dict, field: str) -> bool:
    """Whether a bag count is present and is a count. `0` is a real answer."""
    value = record.get(field)
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _standing(record: dict) -> str:
    """Where this reservation's bags stand, and what each tier would owe on them."""
    people = len(record["passengers"])
    total, paid = record["total_baggages"], record["nonfree_baggages"]
    rows = [
        ROW.format(
            tier=tier,
            free=FREE[tier][record["cabin"]] * people,
            nonfree=max(0, total - FREE[tier][record["cabin"]] * people),
        )
        for tier in FREE
    ]
    return "\n".join(
        [
            HEADING,
            STANDING.format(
                total=total, paid=paid, cabin=record["cabin"], people=people, s="s" * (people != 1)
            ),
            DEPENDS,
            *rows,
            WHICH_ROW.format(paid=paid),
        ]
    )


def _allowance(record: dict) -> str:
    """This user's row of the table, so the reservation card collapses to one line."""
    tier = record["membership"]
    rows = ", ".join(f"{FREE[tier][cabin]} in {cabin.replace('_', ' ')}" for cabin in CABINS)
    return ALLOWANCE.format(tier=tier, rows=rows)
