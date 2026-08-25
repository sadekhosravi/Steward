"""Is this the record the customer described? Answered by comparison, not judgement.

`agents.selector` reads the customer's words and reports what they said the
reservation looks like -- one passenger, basic economy, a round trip out of SFO.
This counts the passengers on the record the call actually names and compares.
Nothing here reaches a model, and nothing here interprets anything: every rule is
one stated value against one field.

WHAT THIS FAMILY COSTS TODAY

Of the 98 surplus writes the deterministic tier lets through, 17 are aimed at the
wrong reservation outright and a further share of the 29 unasked ones are the same
mistake wearing a different label. None of the 98 is a policy violation. They are
legal, well-formed, correctly-priced writes on a record nobody asked about, and no
rule in `policy.md` can reach them because the customer's own sentence is the only
place the answer is written down.

FOUR WAYS THIS REFUSES TO FIRE

Each one exists because a version without it blocks a write gold makes.

  1. The customer named the identifier. If the call's `reservation_id` appears in
     what the customer typed, this says nothing at all -- an explicit id outranks
     every description, and it is checked in code without asking anyone.
  2. The quote is not in the conversation. Every criterion has to be supported by
     `words`, the customer's literal phrasing, and `grounded` checks that phrasing
     really occurs in the customer's turns. A criterion that cannot point at a
     line is discarded whole, along with the rest of the extraction.
  3. The record has not been read. There is nothing to compare against, and
     guessing from the identifier is not comparing.
  4. The stated value is not one this domain uses. A cabin that is not one of the
     three, an airport that is not three letters, a trip type that is neither --
     all dropped rather than matched loosely. Loose matching is how a check starts
     firing on things it has not understood.

WHY A MISMATCHED IDENTIFIER IS NOT ITSELF A REFUSAL

A customer naming one reservation does not mean they are talking about only that
one. Task 44's customer names two and gold writes three others. So a named
identifier can clear this check and can never fail it -- the asymmetry is the
point, and it is the same one that keeps `cancellable` off gold's own writes.

WHAT THE TWO CHECKS HERE MEASURED

Against `scripts/gate_bench.py`, over 198 real proposals and the 49-write answer
key: neither blocks a single gold write in either source. `read_first` catches 8
surplus and costs nothing, so it sits in `PANEL` and runs always. `intended`
catches 2 more and costs a model call each, so it sits in `SELECTION` behind
`STEWARD_SELECT`.

The ordering of those two numbers is the finding. The wrong-record family looked
like a reading-comprehension problem and most of it was an incuriosity problem:
the agent was writing to records it had never looked at.

THE REMEDIATION IS "PICK THE RIGHT ONE", NOT "STOP"

Every other verifier here ends in "tell the customer this is not allowed". This
one does not: the action is allowed, and the fix is entirely in the arguments.
So it is `recoverable`, and the actor is sent back to find the record that
matches, or to ask which one they mean.
"""

from __future__ import annotations

import re
from typing import Any

from core.verifiers import Evidence, Finding

from .context import ASSISTANT, CUSTOMER
from .records import reservations

__all__ = ["CABINS", "grounded", "intended", "named", "read_first", "said"]

CHECK = "intended"

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

CABINS = ("basic_economy", "economy", "business")

TRIPS = ("one_way", "round_trip")

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


def intended(call, evidence: Evidence) -> Finding | None:
    """The first stated criterion this record contradicts, if any."""
    arguments = getattr(call, "arguments", {}) or {}
    identifier = arguments.get("reservation_id")
    if not isinstance(identifier, str) or not identifier:
        return None

    spoken = said(evidence.dialogue)
    if named(identifier, spoken):
        return None

    stated = dict(evidence.stated or {})
    if not grounded(stated.get("words"), spoken):
        return None

    record = reservations(evidence.observed).get(identifier)
    if record is None:
        return None

    quote = str(stated["words"]).strip()
    for field, mismatch in (
        ("passengers", _passengers),
        ("cabin", _cabin),
        ("origin", _airport),
        ("destination", _airport),
        ("flight_type", _trip),
        ("insurance", _insurance),
    ):
        if field not in stated:
            continue
        told = mismatch(field, stated[field], record, arguments)
        if told is not None:
            return _finding(field, identifier, quote, told)
    return None


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


def _passengers(_field: str, value: Any, record: dict, _arguments: dict) -> tuple[str, str] | None:
    """How many people are on the record, against how many they said.

    The single most load-bearing comparison in this module. Task 17's customer
    says "change *the* passenger to myself" and holds one booking with one
    passenger and three with two; task 41's says "cancel every reservation that
    has only one passenger" and the run cancels five with two or three on them.
    Both are settled by `len()`.
    """
    # A count below one is not a description of anything a customer can hold, so
    # it is a malformed extraction rather than a claim about the record. Every
    # reservation has at least one passenger on it, so left in, a stray zero would
    # block every record it was compared against.
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        return None
    people = record.get("passengers")
    if not isinstance(people, list) or len(people) == value:
        return None
    return (f"{value} passenger" + ("s" if value != 1 else ""), f"{len(people)} on it")


def _cabin(_field: str, value: Any, record: dict, arguments: dict) -> tuple[str, str] | None:
    """The cabin the record is in, against the cabin they said it is in.

    Skipped entirely when the call sets a cabin. Gold's task 7 upgrades a basic
    economy reservation to business and the customer's own sentence names
    business, so on that call the stated cabin describes where the record is
    going, not where it is -- and a check that cannot tell those apart refuses
    gold. Where the call leaves the cabin alone the ambiguity does not arise.
    """
    if "cabin" in arguments:
        return None
    stated, current = _cabined(value), _cabined(record.get("cabin"))
    if stated is None or current is None or stated == current:
        return None
    return (stated.replace("_", " "), f"it is {current.replace('_', ' ')}")


def _cabined(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    flat = re.sub(r"[\s-]+", "_", value.strip().lower()).removesuffix("_class")
    return flat if flat in CABINS else None


def _airport(field: str, value: Any, record: dict, _arguments: dict) -> tuple[str, str] | None:
    """Where the trip flies from or to, when the customer gave a code.

    Codes only. A customer saying "New York" and a record saying `JFK` are the
    same place, and resolving that needs a table this does not have -- so a city
    name is dropped rather than matched against three letters that do not look
    like it.
    """
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{3}", value.strip()):
        return None
    stated, current = value.strip().upper(), record.get(field)
    if not isinstance(current, str) or stated == current.upper():
        return None
    return (f"{field} {stated}", f"it is {current}")


def _trip(_field: str, value: Any, record: dict, _arguments: dict) -> tuple[str, str] | None:
    stated = re.sub(r"[\s-]+", "_", str(value).strip().lower()) if value is not None else ""
    current = record.get("flight_type")
    if stated not in TRIPS or not isinstance(current, str) or stated == current:
        return None
    return (stated.replace("_", " "), f"it is a {current.replace('_', ' ')}")


def _insurance(_field: str, value: Any, record: dict, _arguments: dict) -> tuple[str, str] | None:
    current = record.get("insurance")
    if not isinstance(value, bool) or current not in ("yes", "no"):
        return None
    if value == (current == "yes"):
        return None
    return (
        "travel insurance" if value else "no travel insurance",
        "it has insurance" if current == "yes" else "it has none",
    )


def _finding(field: str, identifier: str, quote: str, told: tuple[str, str]) -> Finding:
    described, actual = told
    return Finding(
        check=f"{CHECK}:{field}",
        recoverable=True,
        reason=(
            f"The customer described the reservation as having {described} -- they said "
            f'"{quote}". Reservation {identifier} does not match: {actual}.'
        ),
        remediation=(
            f"Do not change {identifier}. Find the reservation that matches what the "
            f"customer described ({described}) and make the change there. If none of "
            "the reservations you have read matches, or more than one does, ask the "
            "customer which one they mean before changing anything."
        ),
    )
