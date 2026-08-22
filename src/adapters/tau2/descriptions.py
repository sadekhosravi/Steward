"""What a tool returns, added to what tau2 says it does.

tau2 describes a tool in one line and says nothing about the shape of what comes
back. That is a gap in substance, not in style: `get_user_details` is described
as "Get the details of a user, including their reservations", and the record it
returns also carries `payment_methods`, with the balances on them. An agent asked
for a gift card balance reads that description, finds nothing that offers one,
and tells the customer it has no such tool -- which is exactly what happened in
task 14 of the smoke run. The information it needed was in the return value, and
it could not see the return value without first making the call it had already
talked itself out of.

The repair is a transform over the signature rather than a table of hand-written
blurbs: every tau2 tool annotates its return type, and in all three domains that
type is a pydantic model. Listing its fields therefore stays correct when a
domain changes, and there is nothing to maintain per tool.

Overrides are the exception, and there is one. They are for descriptions that are
wrong for us rather than merely thin -- see `_OVERRIDES`.
"""

from __future__ import annotations

import types
import typing
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel
from tau2.environment.tool import Tool

__all__ = ["describe"]


# tau2's own text for the handoff tells the model to transfer whenever it
# "cannot solve the user's issue", which is an invitation to leave rather than a
# condition to check -- and the model accepted it 48 times in 50 tasks. A tool
# description sits next to the tool the model is about to reach for, so it argues
# with the system prompt from closer range and tends to win. Ours states the same
# rule the gate enforces, so the two stop contradicting each other.
_OVERRIDES = {
    "transfer_to_human_agents": (
        "Hand the conversation to a human agent. This ends the conversation, and "
        "everything still outstanding is left undone, so it is a last resort. Use it "
        "only where the policy directs a transfer, or where the customer asks for a "
        "person. Not being sure how to proceed is not a reason to use it: look it up "
        "or ask the customer."
    ),
}


def describe(tool: Tool) -> str:
    """The description the model is shown: tau2's, plus what comes back."""
    if tool.name in _OVERRIDES:
        return _OVERRIDES[tool.name]
    stated = (tool.openai_schema["function"].get("description") or "").strip()
    returns = _returns(_return_type(tool))
    return f"{stated}\n{returns}" if returns else stated


def _return_type(tool: Tool) -> Any:
    """The tool's annotated return type, or None if it has not got one.

    `get_type_hints` rather than the raw annotation because it resolves string
    annotations and strips `Annotated`, which tau2 uses. It raises on a forward
    reference it cannot resolve, and a missing description is not worth an
    exception at startup.
    """
    try:
        return typing.get_type_hints(tool._func).get("return")
    except Exception:
        return None


def _returns(annotation: Any) -> str:
    """The one line naming the fields, or "" when there is nothing useful to say.

    Only the top level. Nesting is where the field count explodes, and the
    question this answers is "is what I need in here?" -- for which a name is
    enough to justify making the call and reading the rest.
    """
    model, many = _model(annotation)
    if model is None:
        return ""
    fields = ", ".join(model.model_json_schema().get("properties") or {})
    if not fields:
        return ""
    subject = "a list, each entry with" if many else "an object with"
    return f"Returns {subject}: {fields}."


def _model(annotation: Any) -> tuple[type[BaseModel] | None, bool]:
    """The pydantic model inside an annotation, and whether it comes as a list.

    Unwraps the wrappers tau2 actually uses -- `list[X]`, `X | None`, and the
    `Annotated` that survives when a hint could not be resolved. A plain `str` or
    `dict` return yields nothing, which is correct: neither has fields to name.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _model(get_args(annotation)[0])
    if origin in (list, set, tuple):
        arguments = get_args(annotation)
        model, _ = _model(arguments[0]) if arguments else (None, False)
        return model, True
    if origin in (Union, types.UnionType):
        for argument in get_args(annotation):
            if argument is not type(None):
                model, many = _model(argument)
                if model is not None:
                    return model, many
        return None, False
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation, False
    return None, False
