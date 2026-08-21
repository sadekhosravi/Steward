"""Tool definitions the tests share.

`gated` is stated on both, never defaulted: it is the label the gate routes on,
so a test that leaves it off is not testing the path it looks like it is testing.
The adapter sets it from tau2's `mutates_state`, widened to cover the handoff.
"""

from __future__ import annotations

from pydantic_ai.tools import ToolDefinition

_RESERVATION_ID = {
    "type": "object",
    "properties": {"reservation_id": {"type": "string"}},
    "required": ["reservation_id"],
}

LOOKUP = ToolDefinition(
    name="get_reservation",
    description="Look up a reservation.",
    parameters_json_schema=_RESERVATION_ID,
    metadata={"gated": False},
)

CANCEL = ToolDefinition(
    name="cancel_reservation",
    description="Cancel a reservation.",
    # `reason` is free text, and that is the point of it being here: the toolset
    # refuses an invented *identifier* before the gate ever runs, so a test of the
    # pre-gate's broader "was this ever shown?" evidence needs an argument that is
    # not an identifier to carry it.
    parameters_json_schema={
        "type": "object",
        "properties": {"reservation_id": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["reservation_id"],
    },
    metadata={"gated": True},
)
