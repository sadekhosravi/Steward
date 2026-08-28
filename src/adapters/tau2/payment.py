"""What may pay for a booking or a change, and what may not.

`money.py` already states these rules on the user's profile when it is read, and
states them correctly -- replayed against all thirty-eight gold write actions
there is not one violation. What it cannot do is stop a call, and the run breaks
them anyway: task 14 paid for a booking with three travel certificates, task 20
with two, and the environment accepts both because the API checks none of it.

    A booking may use at most one travel certificate, one credit card and three
    gift cards.                                                   (policy.md:78)
    A change is paid by a single gift card or credit card; a certificate cannot
    pay for one.                                                 (policy.md:131)

The second the environment does enforce, by raising `Certificate cannot be used
to update reservation` -- which the run hit. Catching it here costs a turn less
than catching it there.

Deliberately absent: the arithmetic of the total. It is the most common write
error in the logs, nine of fourteen, but the fare depends on a price that lives
in a search result the proposal does not carry, and a check that reconstructs it
from whichever search happens to be in the ledger is a check that is sometimes
guessing. Replayed against gold and against every surplus write in the corpus, a
version that did reconstruct it fired on neither. It buys nothing and could cost
something, so `money.py` goes on supplying the sum and nothing here enforces it.
"""

from __future__ import annotations

from collections import Counter

from core.verifiers import Evidence, Finding

from .money import LIMITS

__all__ = ["payment_composition", "payment_for_change"]

CERTIFICATE = "certificate"


def payment_composition(call, evidence: Evidence) -> Finding | None:
    """At most one certificate, one credit card, three gift cards, on a booking."""
    methods = (getattr(call, "arguments", {}) or {}).get("payment_methods")
    if not isinstance(methods, list):
        return None
    used = Counter(_source(method) for method in methods if isinstance(method, dict))
    for source, limit in LIMITS:
        if used.get(source, 0) > limit:
            kind = source.replace("_", " ")
            return Finding(
                check="payment_composition",
                reason=(
                    f"This booking pays with {used[source]} {kind}s. The policy allows at "
                    f"most {limit}, and the API does not check it."
                ),
                remediation=(
                    f"Use at most {limit} {kind} on this booking. Ask the customer which "
                    "of their payment methods to use for the rest."
                ),
            )
    return None


def payment_for_change(call, evidence: Evidence) -> Finding | None:
    """A certificate cannot pay for a change to an existing reservation."""
    payment = (getattr(call, "arguments", {}) or {}).get("payment_id")
    if not isinstance(payment, str) or not payment.startswith(CERTIFICATE):
        return None
    return Finding(
        check="payment_for_change",
        reason=(
            "This change is paid with a travel certificate. The policy allows a change "
            "to be paid only by a single gift card or credit card, and the environment "
            "rejects a certificate outright."
        ),
        remediation=(
            "Ask the customer for a gift card or credit card already on their profile, "
            "and use that as the payment_id instead."
        ),
    )


def _source(method: dict) -> str:
    """`gift_card_7773485` is a gift card. The id carries its own kind."""
    return str(method.get("payment_id", "")).rsplit("_", 1)[0]
