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

import json
import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

__all__ = [
    "Demand",
    "Deps",
    "StewardState",
    "Obligation",
    "PendingCall",
    "duplicated",
    "invented",
    "sources",
    "ungrounded",
    "unmet",
]


class Deps(BaseModel):
    """What one run of the actor is handed besides its messages.

    All three change with the conversation while the agent is built once, which is
    why they arrive per run rather than baked into instructions. They travel
    together because pydantic-ai allows a run exactly one dependency object, and
    they are wanted in the same two places: the toolset checks an argument against
    `observed`, and the instructions carry `plan` and `policy`.
    """

    observed: list[str] = Field(default_factory=list)
    """The provenance ledger, as of this run."""

    plan: str = ""
    """The planner's route for this turn, rendered. Empty when there is none."""

    policy: str = ""
    """The policy sections this turn was routed to. Empty when it was routed to
    none -- the escalation run, which is not doing any more work and needs no
    procedure to do it by."""

    correction: str = ""
    """A held message sent back for another attempt. Empty on every normal run.

    Carried as a dependency rather than as a prompt for the same reason the plan
    is: `transcript` renders a `UserPromptPart` as "Customer:", and a correction
    delivered that way would put words in the customer's mouth that the gate and
    the speaker would then judge the actor against."""


class PendingCall(BaseModel):
    """A tool call the system wants made, in the form the driver has to execute.

    Lives here rather than with the graph because the gate reasons about one
    before the graph is allowed to emit it.
    """

    id: str
    name: str
    arguments: dict[str, Any]


class Demand(BaseModel):
    """Something the gate required before it would allow an action.

    Kept because the gate has no memory of its own. It is handed a transcript and
    asked to rule, so a condition it imposed one turn ago -- almost always "the
    customer has not confirmed" -- has to be re-derived from prose every time, and
    it frequently is not: 70 of the 166 write refusals in the 50-task run were the
    same demand made again after the customer had already answered it. Recording
    the demand turns that from something the gate must reconstruct into something
    it is told.
    """

    action: str
    """The tool that was refused."""

    reason: str
    """What the gate said it required, verbatim, so the answer can be matched to
    the question that was actually asked."""

    turn: int
    """Which user turn imposed it. The gate compares this against the current turn
    to know whether the customer has had a chance to answer."""


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

    plan: str = ""
    """The planner's route for this turn, rewritten each time a lookup comes back.

    This used to be written once per user turn and left alone, on the reasoning
    that mid-turn results are answers to the plan rather than reasons to rewrite
    it. Measurement reversed it. The first plan of a turn is the one with the
    least evidence behind it, and in the last full run 59 of 207 plans had a
    refusal for a goal -- 23 of them written before the record that decides the
    question had been read. Task 39 refused three permitted cancellations across
    five turns and never called `get_reservation_details` once, because every
    later plan re-derived the first one instead of the facts.

    What the old reasoning got right is kept in how it is rewritten: `sections`
    and `changes` only ever widen, so nothing the actor is working from is
    removed underneath it. See `_plan`.
    """

    policy: str = ""
    """The policy sections the planner has routed this turn to, rendered.

    Rewritten with `plan`, and only ever wider -- a re-plan can bring a section
    in and cannot take one away. That is what keeps the original objection to
    mid-turn re-planning answered: the rules the actor is midway through applying
    stay in front of it, whatever the new plan concluded.
    """

    sections: list[str] = Field(default_factory=list)
    """Section names chosen so far this turn, kept so the next plan can widen them.

    Stored rather than recovered from `policy`, for the same reason `changes` is
    stored rather than parsed back out of `plan`: a set recovered from rendered
    prose is a set waiting to drift."""

    replans: int = 0
    """Mid-turn plans written so far. Bounds the cost of asking again."""

    changes: list[str] = Field(default_factory=list)
    """The writes the planner said this turn needs, one line each, as it wrote them.

    Kept beside the rendered `plan` rather than parsed back out of it, because the
    speaker counts these against `written` and a count taken off rendered prose is
    a count waiting to drift."""

    written: list[str] = Field(default_factory=list)
    """Names of the gated calls the gate has approved this turn.

    Approved rather than executed: what the speaker is asking is whether the actor
    walked away from work, and a call that was approved and then failed in the
    environment was not walked away from. Reset with the plan, since that is the
    span the changes belong to."""

    correction: str = ""
    """A held reply, as the instruction the actor gets for its next attempt."""

    deferrals: int = 0
    """Messages held so far this user turn. Bounds the speaker's loop."""

    consulted: int = 0
    """Times the speaker has been asked to rule, over the whole conversation.

    Counted, and counted separately from `holds`, because the last time a check
    was added its failures were invisible: an allowed message and a check that
    never ran both return nothing, and telling them apart afterwards took a
    Langfuse archaeology session. Cumulative rather than per-turn so the totals
    survive to the end of the conversation."""

    holds: int = 0
    """Times the speaker has sent a reply back, over the whole conversation."""

    turns: Annotated[int, operator.add] = 0
    """User messages received so far. A reducer rather than a plain field because
    `send` sets it without reading the state first, and the count is the only thing
    that tells a demand made this turn from one the customer has already answered."""

    demanded: list[Demand] = Field(default_factory=list)
    """What the gate has required and not yet been shown, latest per action.

    Kept for the whole conversation, not the turn: the point of it is that the
    customer answers on a *later* turn than the one that asked."""

    fixable: str = ""
    """The remediation from the last refusal the assistant could carry out itself.

    Set only when the gate says the fix needs nothing from the customer, and it is
    what turns a refusal into another attempt instead of the end of the turn. In
    the 50-task run 146 of 203 refusals ended in talk, because the remediation is
    written as an instruction and the actor followed it to the customer. Cleared
    the moment an action is approved, or the message is held over it."""

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


def duplicated(arguments: dict[str, Any]) -> list[str]:
    """Entries repeated inside one list argument, as `name[i] and name[j]` paths.

    Free, domain-free, and it catches a mistake no schema can: a booking whose
    passenger list names the same person twice, an itinerary whose return leg is
    the outbound flight again. Both were losses in the 50-task run, and both are
    the model copying the previous line instead of composing the next one.

    Two entries are the same when their identifiers match, or -- for entries that
    carry none -- when they are identical outright. Comparing identifiers rather
    than whole entries is what makes the itinerary case visible: the repeated leg
    differed in its date and its price and was still the same flight.
    """
    repeats = []
    for name, value in arguments.items():
        if not isinstance(value, list) or len(value) < 2:
            continue
        seen: dict[str, int] = {}
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            ids = _identifiers(item)
            key = json.dumps(sorted(ids) if ids else item, sort_keys=True)
            if key in seen:
                repeats.append(f"{name}[{seen[key]}] and {name}[{index}]")
            else:
                seen[key] = index
    return repeats


def sources(arguments: dict[str, Any], observed: list[str]) -> list[tuple[str, str, str]]:
    """For each identifier in a call: its path, its value, and the text it came from.

    The counterpart to `invented`, and the answer to what that check cannot see.
    `invented` asks whether a value was ever shown; this asks *where*. An
    identifier belonging to the customer's other reservation passes `invented`
    untouched -- it is in the ledger, just on the wrong record -- and that is
    exactly how two tasks were lost, one modifying a booking the customer had not
    mentioned and one paying with the wrong gift card.

    Quoting the line each value came from is the whole of the check. Deciding
    whether that line is the right one is a judgment, so it is left to the reader.
    """
    found = []
    for path, value in _identifiers(arguments):
        line = next((text for text in reversed(observed) if value in text), "")
        found.append((path, value, _around(line, value)))
    return found


def _around(text: str, value: str, width: int = 70) -> str:
    """The text either side of `value`, collapsed onto one line."""
    if not text:
        return ""
    at = text.find(value)
    start, end = max(0, at - width), min(len(text), at + len(value) + width)
    clip = " ".join(text[start:end].split())
    return f"{'...' if start else ''}{clip}{'...' if end < len(text) else ''}"


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
