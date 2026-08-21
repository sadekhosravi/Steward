"""The graph schema, and the three ledgers the design turns on.

Everything the multi-agent system knows lives here, because tau2's orchestrator
carries no memory for us: it hands the agent a message and a `State` object and
expects the whole world to be in that object.

Three of the fields are not conversation, they are evidence:

- `approved` -- the **proposal**. A call the gate has passed, held apart from
  `calls` so that nothing between approval and emission can rewrite it. An
  approval that does not bind the thing executed is not an approval.
- `observed` -- the **provenance ledger**. Every text the system has actually
  been shown. An argument that appears nowhere in it was invented.
- `obligations` -- the **obligation ledger**. Things now owed, entered when an
  action incurs them and cleared only by discharging them.

The checks over them are plain functions, not nodes, so they can be tested
without a model and reused by whichever node ends up calling them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["StewardState", "Obligation", "PendingCall", "invented", "ungrounded", "unmet"]


class PendingCall(BaseModel):
    """A tool call the system wants made, in the form the driver has to execute.

    Lives here rather than with the graph because the gate reasons about one
    before the graph is allowed to emit it.
    """

    id: str
    name: str
    arguments: dict[str, Any]


class Obligation(BaseModel):
    """Something the system owes before the turn may end.

    `must_contain` is what makes the check deterministic: policies that impose
    an obligation on the *wording* say so literally ("send the message '...'"),
    so the discharge condition is a substring test, not a judgment.
    """

    id: str
    description: str
    """Stated as the remediation the actor will be handed, not as a rule."""

    must_contain: str | None = None
    """Literal text the reply must carry. `None` means no reply can clear it."""


class StewardState(BaseModel):
    """The graph schema, and the only place conversation state lives.

    Kept JSON-native because the checkpointer serializes it: pydantic-ai's
    message objects are stored dumped and rehydrated in the node that uses them,
    so the state stays portable if the checkpointer ever moves off memory.
    """

    prompt: str | None = None
    """A new user message, consumed by the next `think`."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    """Dumped pydantic-ai history. Overwritten, not appended: a run returns the
    whole history, so a reducer would duplicate it."""

    tool_results: dict[str, str] = Field(default_factory=dict)
    """Results for the calls we last yielded, keyed by call id."""

    reply: str = ""

    calls: list[dict[str, Any]] = Field(default_factory=list)
    """Pending calls, as plain `PendingCall` dumps -- see `messages` above."""

    approved: list[dict[str, Any]] = Field(default_factory=list)
    """Calls the gate has passed, verbatim. The only thing ever emitted."""

    denied: dict[str, str] = Field(default_factory=dict)
    """Remediations for calls the gate refused, keyed by call id.

    Kept apart from `tool_results` because a refused call was never executed:
    nothing came back, so nothing may enter `observed` on its account.
    """

    revisions: int = 0
    """Blocked attempts so far this user turn. Bounds the correction loop."""

    observed: list[str] = Field(default_factory=list)
    """Every text the system has been shown: user messages and tool results."""

    obligations: list[Obligation] = Field(default_factory=list)
    """Unmet obligations. A turn with any of these left is not finished."""


def ungrounded(arguments: dict[str, Any], observed: list[str]) -> list[str]:
    """Argument names whose value appears in nothing the system was ever shown.

    Deliberately a substring test over raw text, not a typed lookup. The rule has
    to be answerable before the domain is known -- "did we see this?" generalizes,
    "is this a valid reservation id?" does not. It is a floor, not a ceiling:
    short and numeric values match by accident, but invented identifiers, which
    are the failure that actually costs reward, do not.
    """
    corpus = "\n".join(observed)
    missing = []
    for name, value in arguments.items():
        for leaf in _leaves(value):
            if leaf not in corpus:
                missing.append(name)
                break
    return missing


# Argument names whose value is an identifier: something the environment issued
# and the system can only have been told. Everything else in a tool call can be
# legitimately rewritten -- a date reformatted, a name recased, a number computed
# -- and looks invented to a substring test when it is not. Identifiers cannot:
# there is no correct way to derive `HKD3PS` from anything.
_IDENTIFIER = ("_id", "_number")


def invented(arguments: dict[str, Any], observed: list[str]) -> list[str]:
    """Identifier arguments whose value was never shown, as `a.b[0].c` paths.

    The narrow, blocking counterpart to `ungrounded`. That one asks the broad
    question and is only ever evidence, because it cannot tell an invented value
    from a correctly reformatted one. This asks it where the answer is safe to act
    on, which makes it the check a tool call can be refused over: every "not found"
    error in the diagnostic run was an identifier, and they were 41% of all the
    errors the environment returned.
    """
    corpus = "\n".join(observed)
    return [path for path, value in _identifiers(arguments) if value not in corpus]


def _identifiers(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Every identifier-named string inside a JSON value, with the path to it.

    Recursive because the ones that matter are nested: a payment id lives inside
    `payment_methods[0]`, a flight number inside `flights[2]`.
    """
    if isinstance(value, dict):
        return [
            found
            for name, item in value.items()
            for found in (
                [(f"{path}{name}", item)]
                if isinstance(item, str) and name.endswith(_IDENTIFIER)
                else _identifiers(item, f"{path}{name}.")
            )
        ]
    if isinstance(value, list):
        return [
            found
            for index, item in enumerate(value)
            for found in _identifiers(item, f"{path.rstrip('.')}[{index}].")
        ]
    return []


def _leaves(value: Any) -> list[str]:
    """The scalar strings inside a JSON value, which are what can be invented.

    Booleans and numbers are skipped: they carry no identity, so their presence
    in the corpus says nothing, and requiring it would reject every `True`.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _leaves(item)]
    return []


def unmet(obligations: list[Obligation], reply: str) -> list[Obligation]:
    """Obligations this reply does not discharge."""
    return [o for o in obligations if o.must_contain is None or o.must_contain not in reply]
