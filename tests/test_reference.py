"""REFERENCE: the facts the domain assumes, and which environments get them.

The drift test reads the airline database, so it skips where the benchmark data
has not been fetched. Everything else here is offline: a scripted actor, and
assertions about which text reached it.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from adapters.tau2.reference import AIRPORTS, CODES, reference
from core.kernel import Kernel
from tests.tools import LOOKUP

POLICY = "Cancellations within 24 hours are free."

SEARCH = [
    ToolDefinition(
        name=name,
        description="Search flights.",
        parameters_json_schema={"type": "object", "properties": {}},
        metadata={"gated": False},
    )
    for name in ("search_direct_flight", "search_onestop_flight")
]


# --- which environments are given anything ----------------------------------


def test_the_airline_environment_is_given_the_airports():
    assert reference([LOOKUP, *SEARCH]) == AIRPORTS


def test_an_environment_without_flight_search_is_given_nothing():
    """Retail and telecom have no airports. An empty section reads to the actor as
    something it was supposed to have been given, so it must be literally empty."""
    assert reference([LOOKUP]) == ""


def test_half_the_flight_tools_is_not_the_airline_environment():
    """Both are required, so a domain that happens to name one tool the same way
    does not inherit a list of airports that mean nothing in it."""
    assert reference([LOOKUP, SEARCH[0]]) == ""


# --- the content ------------------------------------------------------------


def test_every_code_appears_in_the_block_the_actor_reads():
    """The list and the block are written out separately, so they can disagree."""
    for code in CODES:
        assert code in AIRPORTS


def test_the_codes_that_cost_task_17_are_named_as_wrong():
    """`NYC` and `CHI` are what the actor invented. Naming them beats describing
    them: the wrong answer it reaches for is the one worth contradicting."""
    assert "NYC" in AIRPORTS
    assert "CHI" in AIRPORTS


def test_new_york_carries_all_three_of_its_airports():
    """The task 17 failure in one line: EWR is New York and the actor did not know."""
    newyork = next(line for line in AIRPORTS.splitlines() if "New York" in line)

    assert "JFK" in newyork
    assert "LGA" in newyork
    assert "EWR" in newyork


@pytest.mark.skipif(not os.environ.get("TAU2_DATA_DIR"), reason="benchmark data not fetched")
def test_the_list_still_matches_the_airline_database():
    """Guards the one thing a hand-written table gets wrong: going stale. A data
    refresh that adds a destination fails here rather than leaving the actor a
    list it has been told is exhaustive."""
    db = pathlib.Path(os.environ["TAU2_DATA_DIR"]) / "tau2/domains/airline/db.json"
    if not db.exists():
        pytest.skip("airline database not present")
    flights = json.loads(db.read_text(encoding="utf-8"))["flights"]
    flights = flights.values() if isinstance(flights, dict) else flights
    actual = {airport for f in flights for airport in (f["origin"], f["destination"])}

    assert actual == set(CODES)


# --- through the Kernel -----------------------------------------------------


def _records_instructions(seen: list[str]):
    def actor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(m.instructions for m in messages if getattr(m, "instructions", None))
        return ModelResponse(parts=[TextPart("Hello.")])

    return actor


def _plans(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"goal": "Help them."})])


def _instructions_with(reference_block: str) -> str:
    seen: list[str] = []
    kernel = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_records_instructions(seen)),
        planner_model=FunctionModel(_plans),
        reference=reference_block,
    )
    kernel.send(kernel.new_thread(), "I fly from New York to Chicago.")
    return "\n".join(seen)


def test_the_reference_reaches_the_actor():
    assert "EWR" in _instructions_with(AIRPORTS)


def test_an_empty_reference_adds_nothing_for_the_actor_to_read_into():
    told = _instructions_with("")

    assert "AIRPORTS" not in told
    assert "EWR" not in told
