"""The tool surface, and the two things it refuses.

Every case here is a call the assistant actually made during the diagnostic run
and the environment rejected. Caught at this layer they cost a retry inside one
agent run; uncaught they cost a step of the benchmark budget, a turn of the
conversation, and -- when the tool was a write -- the task.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.tools import ToolDefinition

from adapters.tau2.schemas import tighten
from agents.toolset import MAX_RETRIES, ValidatedToolset, _grounded, _SchemaValidator
from core.state import Deps

# tau2 declares every nested model as "the real thing, or any object at all", so
# an incomplete passenger validates. This is that shape, not an invention.
BOOK = tighten(
    {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "passengers": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "first_name": {"type": "string"},
                                "dob": {"type": "string"},
                            },
                            "required": ["first_name", "dob"],
                        },
                        {"type": "object", "additionalProperties": True},
                    ]
                },
            },
        },
        "required": ["user_id", "passengers"],
    }
)

WHOLE = {"user_id": "sophia_silva_7557", "passengers": [{"first_name": "S", "dob": "1990-01-01"}]}


def refusal(arguments: dict) -> str:
    with pytest.raises(ModelRetry) as raised:
        _SchemaValidator(BOOK).validate_python(arguments)
    return str(raised.value)


def test_a_complete_call_passes_through_unchanged():
    assert _SchemaValidator(BOOK).validate_python(WHOLE) == WHOLE


def test_a_missing_nested_field_is_refused():
    """tau2's own schema accepts this: the passenger matches its permissive branch.
    Only the tightened one does not."""
    assert "'dob' is a required property" in refusal(
        {**WHOLE, "passengers": [{"first_name": "Sophia"}]}
    )


def test_the_model_is_told_which_field_and_where():
    """A collapsed `anyOf` reports the reason; an uncollapsed one reports only that
    the value 'is not valid under any of the given schemas', which is unactionable."""
    message = refusal({**WHOLE, "passengers": [{"first_name": "Sophia"}]})
    assert "passengers.0" in message
    assert "any of the given schemas" not in message


def test_a_required_collection_may_not_be_empty():
    """The worst write of the run: a reservation booked with no flights, no
    passengers and no payment, described to the customer as a draft. There is no
    draft, and the row is in the scored database for good."""
    assert "should be non-empty" in refusal({**WHOLE, "passengers": []})


def test_a_half_streamed_call_is_not_judged():
    """Partial arguments are a call still arriving, not a wrong one."""
    partial = {"user_id": "sophia_silva_7557"}
    assert _SchemaValidator(BOOK).validate_python(partial, allow_partial=True) == partial


# --- grounding --------------------------------------------------------------


class _Deps:
    """A `RunContext` stand-in: the grounding check reads `ctx.deps.observed` and
    nothing else, so nothing else has to exist for it."""

    def __init__(self, observed: list[str]):
        self.deps = Deps(observed=observed)


def test_an_identifier_that_was_never_shown_is_refused():
    with pytest.raises(ModelRetry) as raised:
        _grounded(_Deps(["your reservation HKD3PS"]), reservation_id="H0000X")
    assert "`reservation_id`" in str(raised.value)


def test_an_identifier_that_was_shown_is_allowed():
    assert _grounded(_Deps(["your reservation HKD3PS"]), reservation_id="HKD3PS") is None


def test_a_nested_identifier_is_found():
    """The ones that matter are nested: a payment id lives inside a payment method."""
    with pytest.raises(ModelRetry) as raised:
        _grounded(_Deps([]), payment_methods=[{"payment_id": "cert_123", "amount": 100}])
    assert "`payment_methods[0].payment_id`" in str(raised.value)


def test_a_reformatted_value_is_not_an_invented_one():
    """The reason this check is narrow. A date the model rewrote from 'May 21' looks
    invented to a substring test, so only identifiers are grounds for refusal."""
    assert _grounded(_Deps([]), date="2024-05-21", first_name="Sophia") is None


# --- wiring -----------------------------------------------------------------


def test_every_tool_carries_the_validator_and_a_retry_budget():
    """The regression: `ExternalToolset` hardcodes an any-schema validator and zero
    retries, so without this override nothing is checked and nothing is corrected.

    `asyncio.run` rather than an async test, so the suite needs no plugin we do not
    declare -- `get_tools` is the only coroutine in reach and it awaits nothing.
    """
    toolset = ValidatedToolset(
        [ToolDefinition(name="book_reservation", description="Book.", parameters_json_schema=BOOK)]
    )
    tools = asyncio.run(toolset.get_tools(None))
    tool = tools["book_reservation"]

    assert isinstance(tool.args_validator, _SchemaValidator)
    assert tool.args_validator_func is _grounded
    assert tool.max_retries == MAX_RETRIES


# --- entries written twice --------------------------------------------------


def test_a_passenger_named_twice_is_refused():
    """Refused here rather than at the gate because it needs no judgement, and a
    `ModelRetry` is corrected inside the actor's own run -- it spends none of the
    two revisions the gate allows and never reaches the environment."""
    with pytest.raises(ModelRetry, match="passengers"):
        _grounded(
            _Deps([]),
            passengers=[
                {"first_name": "Omar", "last_name": "Rossi", "dob": "1970-06-06"},
                {"first_name": "Omar", "last_name": "Rossi", "dob": "1970-06-06"},
            ],
        )


def test_a_flight_repeated_in_place_of_the_return_is_refused():
    with pytest.raises(ModelRetry, match="flights"):
        _grounded(
            _Deps(["HAT169 DTW to JFK"]),
            flights=[
                {"flight_number": "HAT169", "date": "2024-05-17"},
                {"flight_number": "HAT169", "date": "2024-05-19"},
            ],
        )


def test_a_genuine_itinerary_passes():
    assert (
        _grounded(
            _Deps(["HAT169 DTW to JFK", "HAT033 JFK to DTW"]),
            flights=[
                {"flight_number": "HAT169", "date": "2024-05-17"},
                {"flight_number": "HAT033", "date": "2024-05-19"},
            ],
        )
        is None
    )


def test_an_invented_identifier_is_reported_before_a_repeat():
    """Order matters for the message the actor reads: a value it never saw is a
    different mistake from one it wrote out twice, and hearing both at once is
    how a correction turns into a guess."""
    with pytest.raises(ModelRetry, match="do not appear anywhere"):
        _grounded(
            _Deps([]),
            flights=[
                {"flight_number": "HAT169", "date": "2024-05-17"},
                {"flight_number": "HAT169", "date": "2024-05-19"},
            ],
        )
