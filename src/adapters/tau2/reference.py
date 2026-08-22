"""Facts about the world the domain assumes and never states.

A tau2 policy describes a business. A tool schema describes a call. Neither says
what an airport code is, and the airline domain is written as though everyone
already knows -- so the model supplies the gap from its own head, and a 20B model
supplies it wrongly.

That is not a hypothetical. In task 17 of the 15-task run the actor had already
read the right reservation -- `FQ8APE`, EWR to ORD -- did not recognise it as the
"New York to Chicago" trip the customer described, searched for flights between
`NYC` and `CHI`, got two empty lists back, and transferred the conversation to a
human with the answer in its hands. Task 20 made the same call, `NYC` to `SEA`,
and only recovered because it thought to try `JFK` afterwards.

This is a system prompt block rather than a lookup tool on purpose. A tool is
answered only when the model knows it has a question, and the whole failure here
is that it does not: it is not hesitating over New York, it is confident that
`NYC` is an airport. Twenty airports also cost less as a list than a tool costs
as a schema, which the model is shown on every request whether it calls it or not.

Kept in `adapters` because it is a fact about one tau2 domain, and keyed off the
tools present rather than a domain name, so the retail and telecom environments
-- which have no airports and no flight search -- are handed nothing.
"""

from __future__ import annotations

from pydantic_ai.tools import ToolDefinition

__all__ = ["AIRPORTS", "CODES", "reference"]

# Every airport in the airline database. `tests/test_reference.py` checks this
# against the domain itself, so a data refresh that adds a destination fails a
# test rather than quietly leaving the actor a list it will trust.
CODES = frozenset(
    {
        "ATL",
        "BOS",
        "CLT",
        "DEN",
        "DFW",
        "DTW",
        "EWR",
        "IAH",
        "JFK",
        "LAS",
        "LAX",
        "LGA",
        "MCO",
        "MIA",
        "MSP",
        "ORD",
        "PHL",
        "PHX",
        "SEA",
        "SFO",
    }
)

AIRPORTS = """
AIRPORTS

These twenty are the only airports in the system, and the codes below are the
only ones that exist. A search or a booking using anything else comes back empty,
and an empty result means you used a code that is not real -- not that the flight
is unavailable. `NYC` and `CHI` are not airports.

  Atlanta ATL        Detroit DTW           New York JFK, LGA, EWR
  Boston BOS         Houston IAH           Orlando MCO
  Charlotte CLT      Las Vegas LAS         Philadelphia PHL
  Chicago ORD        Los Angeles LAX       Phoenix PHX
  Dallas DFW         Miami MIA             San Francisco SFO
  Denver DEN         Minneapolis MSP       Seattle SEA

New York is served by three, and EWR is Newark, so a customer who says "New York"
may hold a reservation on any of the three.

Read this list in both directions. Going out, it turns the city a customer names
into the code a search needs. Coming back, it turns the code on a reservation
into the city they would call it: a reservation from EWR to ORD is the one they
mean when they say they are flying from New York to Chicago. If a customer
describes a trip you cannot see, translate their cities before you conclude the
reservation is missing.
""".strip()


# The two tools no other tau2 domain has. Presence of both is what identifies the
# airline environment here -- a name would have to be passed down from the caller
# and kept in step with tau2's registry, and this cannot fall out of step.
_FLIGHT_SEARCH = frozenset({"search_direct_flight", "search_onestop_flight"})


def reference(tools: list[ToolDefinition]) -> str:
    """Reference facts for the environment these tools belong to, or nothing.

    Nothing is the right answer for every domain but one, and it has to be
    literally empty rather than a heading with no list under it: an actor shown
    an empty section reads it as something it has failed to be given.
    """
    return AIRPORTS if _FLIGHT_SEARCH <= {tool.name for tool in tools} else ""
