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
    parameters_json_schema=_RESERVATION_ID,
    metadata={"gated": True},
)
