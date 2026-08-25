"""What the `requested` judge is shown about one proposed write.

The judge is domain-free and deliberately small: it compares what the customer
said against an action. Turning an airline proposal into the strings that make
that comparison answerable is knowledge about an airline, so it lives here.

The first version of this module gave the judge 46% precision -- it blocked 14
gold writes against 12 surplus ones, worse than the monolithic critic it was
built to replace. Reading back what it had actually been handed, the question was
often unanswerable from the page, and twice the design contradicted itself.

WHAT IT DOES, NOT JUST WHAT IT IS CALLED

    THE ACTION ABOUT TO BE TAKEN
    update_reservation_flights(reservation_id='XEHM4B', cabin='business', ...)

Task 7's customer says "upgrade XEHM4B to business, then cancel it". Connecting
those two lines requires knowing that this domain has no `upgrade_cabin` tool and
that a cabin upgrade *is* `update_reservation_flights` carrying the same flights
and a new cabin. Nobody told the judge that. The assistant is given a described
tool catalogue; the judge got a bare function name it had never seen defined, and
answering "no, they asked for an upgrade, not a flight change" was the reasonable
reading of what was on the page.

So `means()` says what the call does in the words a customer would use, and
where the record is known it says what actually changes:

    cabin: basic_economy -> business
    flights: unchanged

which turns the comparison into something close to arithmetic.

THE OFFER HAS TO BE VISIBLE FOR THE AGREEMENT TO MEAN ANYTHING

The instructions told the judge to answer yes when the customer "agreed to it
when it was offered to them" -- and `said()` kept only `Customer:` lines, so
every offer was deleted before the judge saw it. A bare "Yes." with nothing to
point at reads as a customer who asked for nothing.

The reason for stripping the assistant was real: its account of what was asked
for is the thing under test, so it cannot also be the evidence. The fix is not to
hide it but to label it. `exchange()` keeps both sides in order and marks whose
words are whose, and the instructions say plainly that the assistant's turns are
there to give "yes" an antecedent and are not themselves evidence of a request.

Tool calls and their results stay out. Those are the assistant investigating,
they are the bulk of the transcript, and none of them is anybody asking for
anything.

WHAT HAS ALREADY BEEN DONE

A customer asks once and the actor sometimes acts twice. Without the ledger of
writes already committed, the second attempt looks exactly like the first, and
the judge has no way to tell a request from a repeat of one already satisfied.
"""

from __future__ import annotations

from .records import reservations, users

__all__ = ["already", "exchange", "facts", "means", "spelled"]

NOTHING = "No record for this has been read."

CUSTOMER = "Customer: "
ASSISTANT = "Assistant: "

# What each write does, in the words a customer would use for it. Deliberately
# not tau2's own descriptions: those are written for the actor choosing a tool and
# say what to pass, where this reader needs to know what the action means to the
# person who asked for it.
MEANING = {
    "book_reservation": "Creates a brand new reservation and charges for it.",
    "cancel_reservation": (
        "Cancels an entire existing reservation and refunds it. All passengers, all flights on it."
    ),
    "update_reservation_flights": (
        "Changes which flights are on a reservation, the cabin it is booked in, or "
        "both. This domain has no separate upgrade tool, so upgrading or downgrading "
        "a cabin is done with this call, keeping the same flights and naming the new "
        "cabin."
    ),
    "update_reservation_passengers": (
        "Changes the passenger details on a reservation. It cannot add or remove "
        "passengers, only correct who they are."
    ),
    "update_reservation_baggages": "Changes how many checked bags a reservation carries.",
    "send_certificate": (
        "Sends the customer a travel certificate -- money off future flights. It is "
        "compensation, not a refund."
    ),
    "transfer_to_human_agents": "Hands the customer to a human and ends this conversation.",
}


def exchange(dialogue: str, turns: int = 16) -> str:
    """The conversation, both sides, most recent last.

    Both sides, because half of what counts as asking is agreeing to something
    offered, and an agreement without the offer beside it says nothing. Marked by
    speaker, because the assistant's account of the request is the thing being
    checked and must never be mistaken for the request.

    Lookups and their results are dropped. They are most of the transcript and
    none of them is anyone asking for anything.
    """
    lines = []
    for line in dialogue.splitlines():
        for prefix, who in ((CUSTOMER, "customer"), (ASSISTANT, "assistant")):
            body = line[len(prefix) :].strip() if line.startswith(prefix) else ""
            if body:
                lines.append(f"{who}: {body}")
                break
    return "\n".join(lines[-turns:]) or "customer: (nothing said yet)"


def spelled(name: str, arguments: dict) -> str:
    """The call as one readable line, for a reader rather than a parser."""
    listed = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name}({listed})"


def means(name: str, arguments: dict, observed: tuple[str, ...] | list[str]) -> str:
    """What the call does, and where it can be worked out, what it changes.

    The difference is the answerable part. "Change the flights on XEHM4B" and
    "upgrade XEHM4B to business" are the same call, and only the diff against the
    record tells the judge which of them the customer is looking at.
    """
    meaning = MEANING.get(name, "")
    record = reservations(observed).get(arguments.get("reservation_id"))
    changes = _changes(name, arguments, record) if record else []
    if not changes:
        return meaning or "No description available for this action."
    return "\n".join([meaning, "", "What changes:", *(f"  {line}" for line in changes)])


def _changes(name: str, arguments: dict, record: dict) -> list[str]:
    """Field by field, old against new, for the updates that carry a diff."""
    lines = []
    if name == "update_reservation_flights":
        lines += _moved("cabin", record.get("cabin"), arguments.get("cabin"))
        before = _legs(record.get("flights") or [])
        after = _legs(arguments.get("flights") or [])
        lines.append(f"flights: {before} -> {after}" if before != after else "flights: unchanged")
    elif name == "update_reservation_baggages":
        lines += _moved(
            "checked bags", record.get("total_baggages"), arguments.get("total_baggages")
        )
    elif name == "update_reservation_passengers":
        lines += _moved(
            "passengers",
            len(record.get("passengers") or []),
            len(arguments.get("passengers") or []),
        )
    elif name == "cancel_reservation":
        lines.append(
            f"the whole reservation, {len(record.get('flights') or [])} flight(s), is voided"
        )
    return lines


def _moved(label: str, before: object, after: object) -> list[str]:
    if after is None or before == after:
        return [f"{label}: unchanged"]
    return [f"{label}: {before} -> {after}"]


def already(committed: tuple[str, ...] | list[str]) -> str:
    """The writes that have already gone through in this conversation.

    Without it a second attempt at a satisfied request is indistinguishable from
    the first, and the judge is asked a question it has no way to answer.
    """
    if not committed:
        return "Nothing has been changed yet in this conversation."
    return "\n".join(f"- {tool}" for tool in dict.fromkeys(committed))


def facts(name: str, arguments: dict, observed: tuple[str, ...] | list[str]) -> str:
    """The record this call points at, in the terms customers set conditions in.

    Task 41's customer asks to cancel every reservation with a single passenger on
    it, and the run cancels five that have two or three. Shown the call alone, no
    judge on earth can tell those apart. Shown "passengers: 2", the comparison is
    nearly arithmetic.

    A booking names no record because it is creating one, so what is described is
    the booking itself. Everything else names a reservation, and if that
    reservation has never been read there is nothing honest to say about it.
    """
    if name == "book_reservation":
        return _proposed(arguments, observed)
    record = reservations(observed).get(arguments.get("reservation_id"))
    if record is None:
        return NOTHING
    return _existing(record)


def _existing(record: dict) -> str:
    legs = record.get("flights") or []
    lines = [
        f"reservation {record.get('reservation_id')}",
        f"  passengers: {len(record.get('passengers') or [])}",
        f"  cabin: {record.get('cabin')}",
        f"  route: {record.get('origin')} to {record.get('destination')}"
        f" ({record.get('flight_type')})",
        f"  flights: {_legs(legs)}",
        f"  checked bags: {record.get('total_baggages')}",
        f"  travel insurance: {record.get('insurance')}",
        f"  booked on: {str(record.get('created_at'))[:10]}",
    ]
    return "\n".join(lines)


def _proposed(arguments: dict, observed: tuple[str, ...] | list[str]) -> str:
    """A booking has no record yet, so the proposal is described instead."""
    profile = users(observed).get(arguments.get("user_id")) or {}
    held = profile.get("reservations")
    lines = [
        "a new booking",
        f"  passengers: {len(arguments.get('passengers') or [])}",
        f"  cabin: {arguments.get('cabin')}",
        f"  route: {arguments.get('origin')} to {arguments.get('destination')}"
        f" ({arguments.get('flight_type')})",
        f"  flights: {_legs(arguments.get('flights') or [])}",
        f"  checked bags: {arguments.get('total_baggages')}",
        f"  travel insurance: {arguments.get('insurance')}",
    ]
    if isinstance(held, list):
        # A booking that duplicates a trip the customer already holds is the shape
        # of task 6 -- cancel and rebook to add insurance, which nobody asked for.
        lines.append(f"  this customer already holds {len(held)} reservations")
    return "\n".join(lines)


def _legs(legs: list) -> str:
    if not legs:
        return "none given"
    return ", ".join(
        f"{leg.get('flight_number')} on {leg.get('date')}" for leg in legs if isinstance(leg, dict)
    )
