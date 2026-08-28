"""What an update is allowed to change about a reservation it already has.

Four rules from `policy.md:110-129`, each one a comparison between the record as
it now stands and the arguments about to be sent, and each one ending in the same
sentence: *"The API does not check these for the agent, so the agent must make
sure the rules apply before calling the API!"*

    - Basic economy flights cannot be modified.
    - Other reservations can be modified without changing the origin, destination,
      and trip type.
    - Cabin cannot be changed if any flight in the reservation has already been flown.
    - The user can add but not remove checked bags.
    - The user can modify passengers but cannot modify the number of passengers.
      Even a human agent cannot modify the number of passengers.

The first is the one the critic gets backwards, and it does so because the policy
draws a distinction one line apart: basic economy *flights* cannot be modified,
and *"all reservations, including basic economy, can change cabin without
changing the flights"*. Gold calls `update_reservation_flights` on a basic economy
reservation eight times and every one of them is a cabin change that leaves the
legs alone. The critic cited basic economy in twenty-two refusals. Stated as a
comparison between two lists of legs, the distinction stops being subtle.

The last is the flattest rule in the whole policy -- the passenger count may not
change, and it says even a human cannot do it -- and it is still a rule that can
only be checked by counting two lists, which is not a thing to ask a 20B model to
do in prose.

None of these fires on an unknown. Every one needs the current record, and
without it they have nothing to compare against and say nothing.
"""

from __future__ import annotations

from core.verifiers import Evidence, Finding

from .records import reservations

__all__ = ["baggage_only_grows", "flights_changeable", "passenger_count_fixed"]

BASIC = "basic_economy"


def flights_changeable(call, evidence: Evidence) -> Finding | None:
    """Basic economy legs may not move, and the trip may not be re-routed."""
    record = _record(call, evidence)
    if record is None:
        return None
    arguments = getattr(call, "arguments", {}) or {}

    before = _legs(record.get("flights"))
    after = _legs(arguments.get("flights"))
    if after is None or before is None:
        return None

    if before != after and str(record.get("cabin", "")).lower() == BASIC:
        # The cabin argument decides which rule this call falls under. Moving the
        # legs of a reservation that stays in basic economy is the modification
        # the policy forbids; moving them as part of leaving basic economy is
        # what gold does eight times.
        if str(arguments.get("cabin", record.get("cabin"))).lower() == BASIC:
            return Finding(
                check="flights_changeable",
                reason=(
                    "The policy says basic economy flights cannot be modified, and this "
                    "call changes the flights on a basic economy reservation without "
                    "changing its cabin."
                ),
                remediation=(
                    "Do not change the flights. Tell the customer that basic economy "
                    "flights cannot be modified. They may change the cabin instead, or "
                    "cancel if the reservation qualifies."
                ),
            )

    # The policy also forbids changing the origin, destination or trip type, and
    # that one is deliberately absent: a proposed leg carries a flight number and
    # a date and nothing else, so where it flies between is only knowable by
    # looking the flight up in a table this does not have. A check that guessed
    # it from the first and last flight numbers would be a check that fires on
    # itineraries it has not understood.
    return None


def passenger_count_fixed(call, evidence: Evidence) -> Finding | None:
    """The one rule the policy says even a human agent cannot bend."""
    record = _record(call, evidence)
    if record is None:
        return None
    proposed = (getattr(call, "arguments", {}) or {}).get("passengers")
    current = record.get("passengers")
    if not isinstance(proposed, list) or not isinstance(current, list):
        return None
    if len(proposed) == len(current):
        return None
    return Finding(
        check="passenger_count_fixed",
        recoverable=True,
        reason=(
            f"The reservation has {len(current)} passengers and this call would leave it "
            f"with {len(proposed)}. The policy says the number of passengers cannot be "
            "modified, and that even a human agent cannot do it."
        ),
        remediation=(
            f"Send exactly {len(current)} passengers. Tell the customer the number of "
            "passengers cannot be changed, and that a separate booking is needed to "
            "travel with a different number of people."
        ),
    )


def baggage_only_grows(call, evidence: Evidence) -> Finding | None:
    """Bags can be added and not taken away."""
    record = _record(call, evidence)
    if record is None:
        return None
    proposed = (getattr(call, "arguments", {}) or {}).get("total_baggages")
    current = record.get("total_baggages")
    if not _counted(proposed) or not _counted(current) or proposed >= current:
        return None
    return Finding(
        check="baggage_only_grows",
        recoverable=True,
        reason=(
            f"The reservation has {current} checked bags and this call would reduce it to "
            f"{proposed}. The policy allows bags to be added but not removed."
        ),
        remediation=(
            f"Do not reduce the bag count below {current}. Tell the customer that checked "
            "bags cannot be removed once they are on a reservation."
        ),
    )


def _record(call, evidence: Evidence) -> dict | None:
    arguments = getattr(call, "arguments", {}) or {}
    return reservations(evidence.observed).get(arguments.get("reservation_id"))


def _legs(flights: object) -> list[tuple[str, str]] | None:
    """A flight list as the pairs that identify it, or None if it is not one.

    Only the flight number and the date. A reservation's legs carry a price and a
    proposal's do not, so comparing whole entries would report every call as a
    change to the itinerary.
    """
    if not isinstance(flights, list):
        return None
    legs = []
    for leg in flights:
        if not isinstance(leg, dict):
            return None
        legs.append((str(leg.get("flight_number", "")), str(leg.get("date", ""))))
    return legs


def _counted(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
