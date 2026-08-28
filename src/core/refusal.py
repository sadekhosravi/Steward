"""Whether a refusal survives being checked against the conversation.

The critic is a model, and at this size it is a poor one. Put every write gold
makes to it -- with the reads already done, the action listed to the customer and
the customer answering yes -- and it refuses 41 of 49 across 22 of the 26 write
tasks. Gold's writes are correct by definition, so all 41 are wrong.

What it refuses for is the useful part, because it is not judgement:

    28  "the customer has not confirmed"   -- said of a transcript ending
                                              `Customer: Yes, please go ahead.`
     7  "the reservation id was not given" -- said of an id quoted from a lookup
                                              in the same prompt
     5  "no cancellation reason"
     1  "basic economy forbids this"

The first two are not opinions that happen to be wrong. They are statements about
the text of the conversation, and the conversation is right there. So they can be
checked, and a refusal that fails its own check is discarded.

DIRECTION

Every check here can only ever *discard* a refusal. Nothing in this module can
turn an approval into a block, or block anything on its own. That asymmetry is
the safety argument: the failure mode being repaired is a correct write refused,
so a repair that could itself refuse would be reintroducing the disease. If a
check is wrong, the cost is one write the critic wanted stopped going through --
and the critic wants 84% of correct writes stopped.

WHAT IT DOES NOT TOUCH

Refusals on any other ground are left standing. "No cancellation reason" is not
checked because a reason is a thing said in words, and matching words to the
policy's three categories is the judgement this module exists to avoid making.
"""

from __future__ import annotations

import re

from core.state import PendingCall

__all__ = ["survives"]

# What the critic says when it thinks nobody agreed to the action.
CONFIRMATION = re.compile(
    r"\b(confirm\w*)\b.{0,80}\b(not|never|no|without|lack\w*|missing|absent|yet)\b"
    r"|\b(not|never|no|without|lack\w*|missing|absent|yet)\b.{0,80}\b(confirm\w*)\b",
    re.IGNORECASE | re.DOTALL,
)

# What it says when it thinks an identifier was invented.
UNPROVENANCED = re.compile(
    r"\b(reservation|payment|user|order|flight)[ _]?(id|number)\b.{0,120}"
    r"\b(not|never|no|without|missing|absent)\b"
    r"|\b(not|never|no|without|missing|absent)\b.{0,120}"
    r"\b(provided|supplied|given|looked up|retrieved)\b",
    re.IGNORECASE | re.DOTALL,
)

# The customer agreeing. Deliberately narrow: this decides whether a refusal is
# thrown away, so a loose match here is a write nobody agreed to going through.
AGREED = re.compile(
    r"^\W*(yes|yeah|yep|sure|correct|confirmed?|ok(ay)?|go ahead|please do|do it|"
    r"that works|sounds good|proceed)\b",
    re.IGNORECASE,
)


def survives(reason: str, proposal: list[PendingCall], dialogue: str, observed: list[str]) -> bool:
    """Whether this refusal is still standing after its own claim is checked.

    `dialogue` is the rendered conversation the critic was shown -- the same text,
    so a check here cannot appeal to evidence the critic never had.
    """
    if CONFIRMATION.search(reason) and _confirmed(dialogue):
        return False
    if UNPROVENANCED.search(reason) and _all_identifiers_shown(proposal, observed):
        return False
    return True


def _confirmed(dialogue: str) -> bool:
    """Whether the customer's last word was agreement to something just proposed.

    Read backwards, and both halves are required. The last thing said has to be
    the customer agreeing -- a yes followed by more lookups was agreement to
    those, and the policy asks for the action to be listed and agreed to
    immediately before it is taken. And something has to have been listed: an
    agreement with no assistant turn in front of it is the customer answering a
    question nobody asked.

    Deliberately strict. What it cannot tell is whether the customer agreed to
    *this* action rather than some other one -- that is a judgement about meaning,
    and this module does not make those. Being strict is how that gap is paid
    for: the cost of refusing to discard is one wrong refusal left standing,
    which is where we already are.
    """
    lines = [line for line in dialogue.splitlines() if line.strip()]
    while (
        lines and lines[-1].startswith("Assistant:") and not lines[-1][len("Assistant:") :].strip()
    ):
        lines.pop()
    if not lines or not lines[-1].startswith("Customer:"):
        return False
    if not AGREED.match(lines[-1][len("Customer:") :].strip()):
        return False
    return any(
        line.startswith("Assistant:") and line[len("Assistant:") :].strip() for line in lines[:-1]
    )


def _all_identifiers_shown(proposal: list[PendingCall], observed: list[str]) -> bool:
    """Whether every identifier in the proposal appears in something already read.

    The same test `provenance` reports on, applied as an answer rather than as
    evidence. An empty proposal, or one naming no identifiers, is not a claim
    about provenance and cannot settle one -- so it does not discard the refusal.
    """
    from core.state import _identifiers

    found = False
    for call in proposal:
        for _, value in _identifiers(call.arguments):
            found = True
            if not any(value in text for text in observed):
                return False
    return found
