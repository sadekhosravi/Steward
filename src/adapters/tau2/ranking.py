"""Turning a list of candidates into a comparison, before the actor picks one.

A flight search returns its rows in whatever order the environment holds them.
That order means nothing, and the actor treats it as if it meant everything: in
the 50-task run, four of the five wrong flight choices were the row printed
first. Task 30 took a $155 nonstop over the $146 one below it; task 33 took $158
over $111; task 23 took $453 over $259. Each time the cheaper flight was in the
same result, two lines further down.

This is a known and measured property of the models rather than a quirk of ours.
Shown a list of options, the average model picks whatever is displayed first
about 64% of the time, and the same model changes its answer on 41% of pairs when
the order is swapped. It is not a reasoning failure that a better prompt fixes;
it is the ordering doing work the ordering was never entitled to do.

So the deciding facts are computed here and appended to the result. Not
*replacing* it -- the raw rows stay exactly as the environment wrote them, so the
provenance ledger still holds every value the actor is allowed to use, and
nothing is hidden from it. What is added is the one thing it was getting wrong on
its own: which of these is actually cheapest, per cabin, with the full cabins left
out because a seat that does not exist is not an option.

Deliberately not a recommendation. Cheapest is what the customer wants often
enough to be worth computing and not always -- they ask for a morning departure,
or the one that gets in before a meeting -- so this ranks by each criterion it
can compute and says plainly that the choice belongs to the conversation.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["ranked"]

# Below this there is no choice to present, and a "comparison" of one row is
# noise in a prompt that is already long.
MINIMUM = 2

HEADING = (
    "COMPARING THE {count} OPTIONS ABOVE\n"
    "They are listed in no meaningful order, so do not take the first one because "
    "it is first. Choose on what the customer asked for, and say which it was."
)

NOTHING_COMPARABLE = "  (no cabin has seats on more than one of these)"


def ranked(content: str) -> str:
    """The same tool result, with a comparison appended when there is one to make.

    Returns `content` untouched for anything that is not a list of two or more
    flight options -- a lookup, an error, a single match. A result this cannot
    read is a result it leaves alone.
    """
    options = _options(content)
    if len(options) < MINIMUM:
        return content
    lines = [_by_cabin(options, cabin) for cabin in _cabins(options)]
    lines = [line for line in lines if line] or [NOTHING_COMPARABLE]
    departures = _by_departure(options)
    if departures:
        lines.append(departures)
    return f"{content}\n\n{HEADING.format(count=len(options))}\n" + "\n".join(lines)


def _options(content: str) -> list[dict[str, Any]]:
    """Each option as one row: what to call it, what it costs, when it leaves.

    An option is a single flight or a whole itinerary -- `search_onestop_flight`
    returns a list of legs per result -- and an itinerary is summarised as its
    legs cost together and its scarcest cabin, because that is what booking it
    would actually require.
    """
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    options = []
    for item in parsed:
        legs = item if isinstance(item, list) else [item]
        if not legs or not all(isinstance(leg, dict) and "prices" in leg for leg in legs):
            return []
        options.append(
            {
                "name": "+".join(str(leg.get("flight_number", "?")) for leg in legs),
                "prices": _totals(legs),
                "seats": _scarcest(legs),
                "departs": legs[0].get("scheduled_departure_time_est"),
            }
        )
    return options


def _totals(legs: list[dict[str, Any]]) -> dict[str, float]:
    """What each cabin costs across every leg, for cabins every leg prices."""
    priced = [leg.get("prices") or {} for leg in legs]
    shared = set(priced[0]).intersection(*priced) if priced else set()
    return {
        cabin: sum(float(p[cabin]) for p in priced)
        for cabin in shared
        if all(isinstance(p[cabin], int | float) and not isinstance(p[cabin], bool) for p in priced)
    }


def _scarcest(legs: list[dict[str, Any]]) -> dict[str, int]:
    """The tightest seat count on any leg. A cabin is only bookable end to end."""
    counts = [leg.get("available_seats") or {} for leg in legs]
    shared = set(counts[0]).intersection(*counts) if counts else set()
    return {cabin: min(int(c[cabin]) for c in counts) for cabin in shared}


def _cabins(options: list[dict[str, Any]]) -> list[str]:
    """Every cabin any option prices, cheapest class first for a stable reading."""
    seen: dict[str, float] = {}
    for option in options:
        for cabin, price in option["prices"].items():
            seen[cabin] = min(seen.get(cabin, price), price)
    return sorted(seen, key=lambda cabin: seen[cabin])


def _by_cabin(options: list[dict[str, Any]], cabin: str) -> str:
    """One cabin's options, cheapest first, with the sold-out ones dropped.

    Dropped rather than marked: a cabin with no seats is not a slower option, it
    is not an option, and the run has the actor booking into one and spending the
    call on an error.
    """
    available = [
        option
        for option in options
        if cabin in option["prices"] and option["seats"].get(cabin, 1) > 0
    ]
    if len(available) < MINIMUM:
        return ""
    ordered = sorted(available, key=lambda option: option["prices"][cabin])
    priced = ", ".join(f"{o['name']} ${_money(o['prices'][cabin])}" for o in ordered)
    return f"  by {cabin} price: {priced}"


def _by_departure(options: list[dict[str, Any]]) -> str:
    """Earliest departure first, for the customer who asked about time not money."""
    timed = [option for option in options if option["departs"]]
    if len(timed) < MINIMUM:
        return ""
    ordered = sorted(timed, key=lambda option: str(option["departs"]))
    return "  by departure: " + ", ".join(f"{o['name']} {o['departs']}" for o in ordered)


def _money(value: float) -> str:
    """No trailing `.0` on a whole number of dollars, which is all of them here."""
    return str(int(value)) if float(value).is_integer() else str(value)
