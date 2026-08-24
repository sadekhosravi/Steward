"""Repairing the one thing this endpoint gets wrong about its own output format.

gpt-oss speaks the "harmony" format, in which a reply is split into channels and
each channel is opened by a control token -- `<|channel|>commentary`, and its
siblings. The tokens are supposed to be consumed by the parser and never reach
us. Occasionally one does not, and the residue lands inside the name of the
function the model was calling:

    Tool 'book_reservation<|channel|>commentary' exceeded max retries count of 1

Nothing downstream can survive that. The name matches no declared tool, so the
lookup raises; because no tool of that name exists there is no per-tool retry
budget to fall back on, so the agent default of one applies; and the second
malformed name ends the run as `UnexpectedModelBehavior`. tau2 records the whole
simulation as an infrastructure error with no reward at all, which the scorer
then counts as zero. Two of the three simulations lost this way across every run
on record were this, once on `book_reservation` and once on
`transfer_to_human_agents`.

It is repaired here, at the model boundary, because that is what it is: an
artefact of one provider's wire format, not a decision any agent made. Nothing
above this line should have to know the format exists.

The repair is deliberately the narrowest one that works. A control token cannot
occur in a name the model was legitimately trying to call -- no tool anywhere is
named with a `<|` in it -- so the text from the marker onwards is dropped and
nothing else is touched. A name that does not contain one is returned as it came,
and a name that is *only* a marker is left alone as well: there is no call to
recover there, and truncating it to nothing would turn a bad name into a
confusing one.
"""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

__all__ = ["MARKER", "Harmonised", "repaired"]

# The opening of every harmony control token. Not the whole token, because which
# channel leaked is not something to keep a list of.
MARKER = "<|"


class Harmonised(WrapperModel):
    """A model whose tool calls are named the way the tools actually are."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return repaired(await super().request(messages, model_settings, model_request_parameters))


def repaired(response: ModelResponse) -> ModelResponse:
    """The same response, with any control token cut off the end of a tool name.

    Returns the response itself when there is nothing to fix, so the overwhelming
    majority of calls -- every one from a provider that does not do this -- pay a
    scan of the parts and no allocation.
    """
    if not any(_leaking(part) for part in response.parts):
        return response
    return replace(
        response,
        parts=[
            replace(part, tool_name=part.tool_name.split(MARKER, 1)[0]) if _leaking(part) else part
            for part in response.parts
        ],
    )


def _leaking(part: object) -> bool:
    """Whether this part is a tool call whose name has a control token in it.

    A name that begins with the marker is not counted: cutting it would leave an
    empty name, which is a worse thing to report than the name that arrived.
    """
    return (
        isinstance(part, ToolCallPart)
        and MARKER in part.tool_name
        and not part.tool_name.startswith(MARKER)
    )


def harmonised(model: Model) -> Model:
    """`model`, wrapped so that a leaked control token cannot end the run."""
    return Harmonised(model)
