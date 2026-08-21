"""Making tau2's declared schemas mean what tau2 actually enforces.

tau2 builds each tool's JSON Schema from the function's pydantic signature, and
the conversion leaves a hole. Every nested model is declared as a choice between
the real thing and anything at all::

    "items": {"anyOf": [{"$ref": "#/$defs/Passenger"},
                        {"additionalProperties": true, "type": "object"}]}

So `{"first_name": "Sophia"}` is a valid `Passenger` as far as the schema is
concerned, and only fails once tau2 validates it properly inside the function --
by which point the call has been spent and the turn has been wasted. Six of the
errors in the diagnostic run were exactly this, `dob` and `amount` and
`payment_id` missing from objects the schema said were fine.

Both transforms here are general. Neither knows an airline from a phone company,
which is what makes them safe to apply to a domain nobody has looked at yet.
"""

from __future__ import annotations

from typing import Any

__all__ = ["tighten"]


def tighten(schema: dict[str, Any]) -> dict[str, Any]:
    """The same schema, saying what the environment will really accept."""
    return _no_empty_arrays(_no_escape_hatch(schema))


def _no_escape_hatch(node: Any) -> Any:
    """Drop the "or any object at all" branch from every `anyOf`.

    A value matching only that branch is one tau2 will reject anyway, so removing
    it does not forbid anything that would have worked -- it moves the rejection
    to where it is still cheap. If an `anyOf` is *only* permissive branches it is
    left alone, since there would be nothing left to validate against.

    A single surviving branch replaces the `anyOf` outright rather than becoming a
    choice of one. That is not tidiness: a failed `anyOf` reports itself as "is not
    valid under any of the given schemas" and swallows the reason, so the model
    would be told to fix something without being told what. Collapsed, the same
    call comes back as `passengers.0: 'dob' is a required property`.
    """
    if isinstance(node, dict):
        node = {name: _no_escape_hatch(value) for name, value in node.items()}
        options = node.get("anyOf")
        if isinstance(options, list):
            strict = [option for option in options if not _permissive(option)]
            if len(strict) == 1:
                node = {**{k: v for k, v in node.items() if k != "anyOf"}, **strict[0]}
            elif strict:
                node["anyOf"] = strict
        return node
    if isinstance(node, list):
        return [_no_escape_hatch(item) for item in node]
    return node


def _permissive(option: Any) -> bool:
    """An object schema that constrains nothing: any keys, any values."""
    return (
        isinstance(option, dict)
        and option.get("type") == "object"
        and option.get("additionalProperties") is True
        and not option.get("properties")
        and not option.get("required")
    )


def _no_empty_arrays(schema: dict[str, Any]) -> dict[str, Any]:
    """Require at least one item in an array the tool requires at all.

    An argument that is mandatory and empty is a contradiction the model reaches
    for under pressure: the diagnostic run has it booking a reservation with no
    flights, no passengers and no payment, then telling the customer it had
    created a draft. There is no draft. That row is in the scored database for
    good, and the task is lost from that moment.

    Stated as a rule rather than a list of tool names because a list would be
    wrong in the first domain nobody has looked at yet. It says only that a
    caller who must supply a collection must supply something in it, which is a
    property of the word "required", not of airlines.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema
    tightened = dict(properties)
    for name in schema.get("required") or []:
        field = tightened.get(name)
        if isinstance(field, dict) and field.get("type") == "array":
            tightened[name] = {**field, "minItems": 1}
    return {**schema, "properties": tightened}
