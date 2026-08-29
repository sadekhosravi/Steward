"""Never write to a record nobody has looked at.

One check, and what is left of a family that had two. The wrong-record problem
looked like a reading-comprehension problem -- the customer describes a booking
in their own words and the agent picks the wrong one -- and most of it turned out
to be an incuriosity problem: the agent was writing to records it had never read
at all. Task 41's run cancelled seven reservations having read six of them not at
all.

WHAT WAS HERE AND IS NOT

`intended` compared what the customer said the record looks like -- one
passenger, basic economy, a round trip out of SFO -- against the record the call
names, using a description a model extracted first. It never blocked a gold write
and it caught 2 surplus ones in an offline replay, for a model call on every
write proposal that reached it, so it lived behind `STEWARD_SELECT` rather than
in `PANEL`.

It has now been given every chance and taken none of them. In the 15x2 run of
2026-08-29, with the record visible for the first time (`cf5262f`) and the flag
on, it fired **zero times against 112 write proposals**. Removed, with
`agents.selector`, `adapters.tau2.describing` and the `SELECTION` panel. The
offline number was real; it does not survive contact with a conversation.

WHY THIS ONE STAYS

`read_first` needs nothing a model has to produce, so it costs nothing and runs
always. Two conditions, and the second is what makes it exact: the record was
never read, *and* the customer never typed the identifier -- a customer who names
their booking has settled which record this is, and gold's task 42 cancels two
reservations it never reads because the customer named both.

A customer naming one reservation does not mean they are talking about only that
one, either. Task 44's customer names two and gold writes three others. So a
named identifier can clear this check and can never fail it; the asymmetry is the
point.

THE REMEDIATION IS "READ IT FIRST", NOT "STOP"

Every other verifier here ends in "tell the customer this is not allowed". This
one does not: the action is allowed and the fix is entirely in the arguments, so
it is `recoverable` and the actor is sent back to look the record up.

That property made this the most expensive check in the package for four days.
While `records` could not parse a tool result with a note appended to it, this
fired on every write, told the actor to read a record it had just read, and
looped -- 56 refusals, all false, and every simulation it touched scored 0.00.
The defect was in `records`; what it shows about this check is that a recoverable
refusal is only as safe as the fact underneath it.
"""

from __future__ import annotations

import re

from core.verifiers import Evidence, Finding

from .context import ASSISTANT, CUSTOMER
from .records import reservations

__all__ = ["grounded", "named", "read_first", "said"]

# Every label that can head a line in a rendered dialogue -- `gate.transcript`
# writes the first four, `gate_bench` adds the fifth when it replays a saved run.
# A label missing from this list is worse than harmless: its line would be read as
# a continuation of whoever spoke before it, which is how the assistant's own
# account of what it is doing ends up counted as the customer's words.
SPEAKERS = (
    CUSTOMER,
    ASSISTANT,
    "Assistant looks up: ",
    "Assistant calls: ",
    "Result: ",
    "Gate: ",
)


# Short quotes match by accident. "my trip" occurs in almost any conversation
# about an airline, so a span below this length is not evidence that the model
# read anything.
QUOTE = 4


def read_first(call, evidence: Evidence) -> Finding | None:
    """A write on a reservation nobody in this conversation has looked at.

    Free, exact, and it lives in `PANEL` rather than beside `intended` because it
    needs nothing a model has to produce. It is here because it was found while
    reading why `intended` stays silent so often: task 41's run cancelled seven
    reservations and had read six of them not at all, so there was no record to
    compare a description against. The description was never the problem.

    Two conditions, and the second is what makes it exact. The record was never
    read, *and* the customer never typed the identifier -- a customer who names
    their booking has settled which record this is, and gold's task 42 cancels two
    reservations it never reads because the customer named both. Without that
    escape this check blocks those two and scores 82%; with it, 8 surplus and no
    gold at all.

    The policy backs it directly. `policy.md:143-147` makes cancellation
    conditional on four facts that live on the record, and says the API checks
    none of them, so the agent must. An agent that has not read the record cannot
    have checked, and `cancellable` next door falls silent for exactly that
    reason -- this is what that silence should have been saying.
    """
    arguments = getattr(call, "arguments", {}) or {}
    identifier = arguments.get("reservation_id")
    if not isinstance(identifier, str) or not identifier:
        return None
    if reservations(evidence.observed).get(identifier) is not None:
        return None
    if named(identifier, said(evidence.dialogue)):
        return None
    return Finding(
        check="read_first",
        recoverable=True,
        reason=(
            f"Nothing in this conversation has read reservation {identifier}, and the "
            "customer never named it. Its cabin, its passengers and its flights are "
            "all unknown, so none of the conditions the policy puts on this action "
            "have been checked."
        ),
        remediation=(
            f"Call get_reservation_details for {identifier} first and read what comes "
            "back. Then decide whether it is the reservation the customer means and "
            "whether the policy allows this, and only then make the change."
        ),
    )


def said(dialogue: str) -> str:
    """Only what the customer typed, joined, continuation lines included.

    The assistant's turns are dropped here even though `exchange` keeps them for
    the judge. The two uses are opposite: a judge needs the offer so an agreement
    means something, and this needs the *source* of a description -- and the
    assistant announcing which reservation it chose is the very thing under test.

    A turn that ran to several lines only carries the speaker's label on its
    first, so the label sets a speaker and every line after it belongs to that
    speaker until another label appears. Taking the labelled lines alone would
    silently truncate every multi-line turn to its opening sentence, and the
    answer key's scenarios are all multi-line -- so the check would look clean on
    the one source where a false block matters most, by seeing almost none of it.
    """
    lines: list[str] = []
    speaking = False
    for line in dialogue.splitlines():
        marker = next((prefix for prefix in SPEAKERS if line.startswith(prefix)), None)
        if marker is not None:
            speaking = marker == CUSTOMER
            line = line[len(marker) :]
        if speaking:
            lines.append(line.strip())
    return "\n".join(lines)


def named(identifier: str, spoken: str) -> bool:
    """Did the customer type this reservation id themselves?

    Bounded so that an id is not found inside a longer token. Reservation ids and
    flight numbers share an alphabet in this domain -- `HAT110` looks exactly like
    one -- and matching without boundaries would let any six characters clear the
    check.
    """
    return re.search(rf"\b{re.escape(identifier)}\b", spoken, re.IGNORECASE) is not None


def grounded(words: object, spoken: str) -> bool:
    """Is the quoted phrasing really in what the customer said?

    Whitespace and case are normalised because a model copying a span across a
    line break will not reproduce the break. Nothing else is: a quote that has
    been reworded is not a quote, and accepting a paraphrase here would give back
    the freedom this whole design takes away.
    """
    if not isinstance(words, str) or len(words.strip()) < QUOTE:
        return False
    return _flat(words) in _flat(spoken)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
