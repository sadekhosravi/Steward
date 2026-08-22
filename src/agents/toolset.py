"""The tool surface the assistant is given, with the schema actually enforced.

pydantic-ai's `ExternalToolset` is the right shape for this benchmark -- a tool
the model may call but someone else executes -- and it declares every tool's JSON
Schema to the model. It does not check the answer against it:

    args_validator=TOOL_SCHEMA_VALIDATOR   # SchemaValidator(any_schema())
    max_retries=0

So whatever the model emits goes out to the environment untouched, and the first
thing to notice a missing field is tau2, by which point the call has been spent.
That was 21% of the errors in the diagnostic run -- `dob` missing from a
passenger, `amount` missing from a payment -- and a further 41% were identifiers
the model invented, which no schema can catch but the provenance ledger can.

Both are caught here instead, before the call leaves the agent. A rejection is a
`ModelRetry`, so the correction happens inside the assistant's own run: it sees
what was wrong with the arguments it just wrote and writes them again. Nothing
about the graph changes, and no failed call reaches the environment or spends a
step of the benchmark's budget.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from jsonschema import Draft202012Validator
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from core.state import Deps, invented

__all__ = ["MAX_RETRIES", "ValidatedToolset"]

# Attempts a tool call gets to come out valid before the run gives up and the
# error is reported as an ordinary failure. A schema complaint is a typo the model
# fixes on sight, so this is generous where the gate's revision budget is not --
# but it is finite, because an unbounded correction loop is exactly how Run 001
# produced four simulations that never terminated.
MAX_RETRIES = 3

MALFORMED = "Your arguments do not match the tool's schema. {problems}\nFix them and call it again."

INVENTED = (
    "These values do not appear anywhere you have been shown, so they cannot be "
    "right: {paths}. Identifiers come from the environment -- look them up with a "
    "read tool and use what it returns. Do not guess and do not reuse an example."
)


class _SchemaValidator:
    """A JSON Schema, in the shape pydantic-ai asks a validator to be.

    `ToolsetTool.args_validator` is typed against a protocol -- anything with
    `validate_python` and `validate_json` -- rather than a concrete pydantic-core
    class, which is what makes this possible at all. `ModelRetry` is raised rather
    than a `ValidationError` because both are caught in the same place and this one
    lets the message be written for the reader.
    """

    def __init__(self, schema: dict[str, Any]):
        self._validator = Draft202012Validator(schema)

    def validate_python(self, input: Any, *, allow_partial: Any = False, **kwargs: Any) -> Any:
        arguments = input or {}
        # Partial input is a half-streamed call, not a wrong one. Judging it would
        # reject every tool call that has not finished arriving.
        if allow_partial not in (False, "off"):
            return arguments
        problems = [
            f"{'.'.join(str(p) for p in error.absolute_path) or 'arguments'}: {error.message}"
            for error in sorted(self._validator.iter_errors(arguments), key=lambda e: e.path)
        ]
        if problems:
            raise ModelRetry(MALFORMED.format(problems="\n".join(problems)))
        return arguments

    def validate_json(self, input: str | bytes | bytearray, **kwargs: Any) -> Any:
        return self.validate_python(json.loads(input or "{}"), **kwargs)


def _grounded(ctx: RunContext[Deps], **arguments: Any) -> None:
    """Refuse identifiers the system was never shown.

    Runs after the schema passes, on the validated arguments. The ledger arrives
    on `ctx.deps` because it grows with the conversation while the toolset is built
    once -- so it is passed per run, not captured here.
    """
    paths = invented(arguments, ctx.deps.observed if ctx.deps else [])
    if paths:
        raise ModelRetry(INVENTED.format(paths=", ".join(f"`{p}`" for p in paths)))


class ValidatedToolset(ExternalToolset[Deps]):
    """`ExternalToolset`, with the declared schema treated as binding."""

    async def get_tools(self, ctx: RunContext[Deps]) -> dict[str, ToolsetTool[Deps]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=replace(tool_def, kind="external"),
                max_retries=MAX_RETRIES,
                args_validator=_SchemaValidator(tool_def.parameters_json_schema),
                args_validator_func=_grounded,
            )
            for tool_def in self.tool_defs
        }


def declared(tools: list[ToolDefinition]) -> ValidatedToolset:
    """The assistant's toolset. A function so the call site reads as intent."""
    return ValidatedToolset(tools)
