"""What a payment has to add up to, and what is allowed to pay it.

The fifth sibling. It goes on the *user* record rather than the reservation,
which is where the payment methods live and also the cheaper place to put it:
the fifty-task set reads a user fourteen times and a reservation fifty-seven, so
a block here is paid for once a task instead of once a lookup.

Two facts, and the run gets both wrong.

THE TOTAL

A quarter of every write emitted came back an error, and nine of those fourteen
were the same one:

  Error: Payment amount does not add up, total price is 168, but paid 158

Task 6 saw that three times and never recovered, because the gate was insisting
on the wrong figure -- "the booking amount must equal the flight fare plus the
insurance fee, which is $128 + $30 = $158, but the proposed call provides $168"
-- while the environment said 168. Neither of them was going to win that, and a
turn was spent on it either way. The environment's own arithmetic is three terms
(`tools.py`, `book_reservation`):

  fare for one passenger, summed over the legs, times the passengers
  + $30 a passenger if insurance was taken
  + $50 for each bag past the free allowance -- flat, not per passenger

Stated here rather than computed, because the fare depends on the cabin the
customer has not chosen yet and on flights that live in a search result this
does not have. What can be given is the shape of the sum and which term is per
passenger, which is exactly what the two wrong figures above disagreed about.
Against the fifty-task set it reproduces the amount gold paid on all ten of its
bookings.

WHAT MAY PAY IT

The composition rules are policy, and the API enforces only one of them, so the
policy says in as many words: "the agent must make sure the rules apply before
calling the API". A booking may use at most one travel certificate, one credit
card and three gift cards (policy.md:78). A change may be paid only by a single
gift card or credit card (policy.md:131) -- `_payment_for_update` raises
`Certificate cannot be used to update reservation`, which the run hit.

Task 14 paid for a booking with three certificates. Task 20 used two. Neither is
a close call and neither was caught. Across all thirty-eight gold write actions
in the task set there is not one violation of these rules, so stating them
contradicts nothing gold does.

WHAT IT DELIBERATELY DOES NOT DO

It does not choose the methods or split the amount between them. Which card the
customer wants to use is theirs to say, and the run has tasks that turn on being
asked. It only says what the sum has to come to and which combinations the API
will refuse.

It does not check a proposal. Nothing here can refuse a call -- these are notes
on a lookup, and a booking that ignores them fails at the environment, exactly as
it does today.
"""

from __future__ import annotations

import json

__all__ = ["money"]

# policy.md:78, for a new booking. The API does not check these.
LIMITS = (("certificate", 1), ("credit_card", 1), ("gift_card", 3))

# Sources that carry a balance; a credit card has none to run out of.
BALANCED = ("gift_card", "certificate")

INSURANCE = 30

EXTRA_BAG = 50

HEADING = "PAYING FOR THIS"

ON_FILE = "  on file: {methods}"

NOTHING_ON_FILE = (
    "  on file: none. Every payment method has to be on the profile already, so "
    "there is nothing here that can pay for anything."
)

TOTAL = (
    "  a booking's payments must add up to the total exactly:\n"
    "    fare for one passenger, summed over the legs, x passengers\n"
    f"    + ${INSURANCE} x passengers if insurance is taken\n"
    f"    + ${EXTRA_BAG} for each bag past the free allowance (flat, not per passenger)"
)

COMPOSITION = "  a booking may use at most one certificate, one credit card and three gift cards"

CHANGE = (
    "  a change to an existing reservation is paid by ONE gift card or credit card; "
    "a certificate cannot pay for one"
)

BALANCE = "  a gift card or certificate cannot be charged more than the balance shown"


def money(content: str) -> str:
    """The same tool result, with the payment rules appended to a user record.

    Returns `content` untouched for anything that is not a user profile -- a
    reservation, a search, an error, a shape nobody has looked at yet.
    """
    methods = _methods(content)
    if methods is None:
        return content
    lines = [
        HEADING,
        ON_FILE.format(methods="; ".join(methods)) if methods else NOTHING_ON_FILE,
        TOTAL,
        COMPOSITION,
        CHANGE,
        BALANCE,
    ]
    return f"{content}\n\n" + "\n".join(lines)


def _methods(content: str) -> list[str] | None:
    """One line per payment method, or None if this result is not a user profile.

    Recognised by `payment_methods` being the mapping this domain writes, so it
    holds for the profile lookup and stays quiet on everything else. A user with
    an empty wallet is still a user -- the empty list is an answer, and `None` is
    the only way to say "not a profile".
    """
    try:
        record = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("user_id"), str):
        return None
    wallet = record.get("payment_methods")
    if not isinstance(wallet, dict):
        return None
    return [_describe(method) for method in wallet.values() if isinstance(method, dict)]


def _describe(method: dict) -> str:
    """A method as the actor has to name it, with the balance when it has one."""
    name = str(method.get("id", "?"))
    if method.get("source") not in BALANCED:
        return name
    return f"{name} ({_balance(method.get('amount'))})"


def _balance(amount: object) -> str:
    """`$250 left`, or an admission. A balance nobody can read is not a zero."""
    if not isinstance(amount, int | float) or isinstance(amount, bool):
        return "balance not shown"
    return f"${amount:,.2f}".removesuffix(".00") + " left"
