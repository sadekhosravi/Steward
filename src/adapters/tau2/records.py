"""The records a conversation has already read, parsed back out of the ledger.

The cards each keep a private parser -- `eligibility._reservation`,
`baggage._record`, `money._methods` -- because each one is handed a single tool
result and only has to decide whether that one result is the shape it cares
about. A verifier is asked a different question: not "is this a reservation?" but
"which reservations has anyone seen?", across everything observed so far. That is
one lookup over the whole ledger rather than a test on one string, so it lives
here rather than being a fourth copy of the same recogniser.

Recognition is by shape and nothing else, so a result nobody has looked at yet is
skipped rather than mangled. Order matters in one place: a reservation carries a
`user_id` too, so it has to be tested for before a profile is.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["reservations", "statuses", "users"]


def reservations(observed: tuple[str, ...] | list[str]) -> dict[str, dict]:
    """Every reservation record seen, newest wins, keyed by its id.

    Newest wins because a record read again after a write is the one that is now
    true -- the cabin a change moved it to, the bag count an update left behind.
    A verifier reasoning from the first version would be reasoning about a
    reservation that no longer exists.
    """
    found: dict[str, dict] = {}
    for record in _records(observed):
        key = record.get("reservation_id")
        if isinstance(key, str):
            found[key] = record
    return found


def users(observed: tuple[str, ...] | list[str]) -> dict[str, dict]:
    """Every user profile seen, keyed by user id.

    A profile is a record with a `user_id` and a `payment_methods` mapping. The
    mapping is what separates it from a reservation, which also names its owner.
    """
    found: dict[str, dict] = {}
    for record in _records(observed):
        key = record.get("user_id")
        if isinstance(key, str) and isinstance(record.get("payment_methods"), dict):
            found[key] = record
    return found


def statuses(looked_up: tuple[tuple[str, dict, str], ...]) -> dict[str, str]:
    """What the conversation established about each flight's status.

    Two sources, and one of them is unreadable on its own.
    `get_flight_status` answers with the bare word `"delayed"` and no flight
    attached, so the flight number has to come from the arguments that asked --
    which is the whole reason `Evidence` carries the calls and not only their
    results. A search result carries its own `flight_number` and needs no help.

    Keyed by flight number alone. A search does not report the date it was asked
    about, so a richer key would silently drop every status a search established.
    """
    found: dict[str, str] = {}
    for name, arguments, result in looked_up:
        if name == "get_flight_status":
            number = arguments.get("flight_number")
            if isinstance(number, str) and isinstance(result, str):
                found[number] = _bare(result)
            continue
        for row in _rows(result):
            number, status = row.get("flight_number"), row.get("status")
            if isinstance(number, str) and isinstance(status, str):
                found.setdefault(number, status)
    return found


def _bare(result: str) -> str:
    """`"delayed"` and `delayed` are the same answer; a JSON string is still one."""
    text = result.strip()
    try:
        loaded = json.loads(text)
    except ValueError:
        return text.strip('"')
    return loaded if isinstance(loaded, str) else text


def _records(observed: tuple[str, ...] | list[str]) -> list[dict]:
    out: list[dict] = []
    for text in observed:
        loaded = _loaded(text)
        if isinstance(loaded, dict):
            out.append(loaded)
    return out


def _rows(result: Any) -> list[dict]:
    loaded = _loaded(result) if isinstance(result, str) else result
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    return []


# What separates a record with a note stapled to it from a result that merely
# starts with digits. Every note in `agent.NOTES` appends a blank line and then an
# upper-case heading -- `WHAT THIS RESERVATION COSTS`, `PAYING FOR THIS` -- so
# that is the only tail this accepts. `2024-05-15` would otherwise parse as the
# number 2024 with a remainder nobody looked at, and two records run together
# would parse as the first with the second discarded silently.
NOTED = re.compile(r"\n\n[A-Z]")


def _loaded(text: Any) -> Any:
    """The JSON record a tool result carries, notes and all, or None.

    `json.loads` is not enough and never was. The results reaching `observed` are
    the ones `adapters.tau2.agent._noted` has already appended English to, so
    every reservation and every profile in a real conversation arrives as valid
    JSON followed by a blank line and a paragraph -- which `loads` rejects whole.
    That failure was silent, and it made every verifier reading this module blind
    to every record: measured over a 28-simulation run, 0 of the 72 reservations
    the agent had read were visible here, and `intended.read_first` refused 56
    writes on records sitting in the transcript in front of it.

    So the leading value is parsed and the tail is checked rather than ignored.
    Accepting any tail would trade one silent failure for another.
    """
    if not isinstance(text, str):
        return text
    body = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(body)
    except ValueError:
        return None
    rest = body[end:]
    if rest.strip() and not NOTED.match(rest):
        return None
    return value
