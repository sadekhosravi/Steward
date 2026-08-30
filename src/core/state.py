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
import re
from typing import Annotated, Any

from pydantic import BaseModel, Field

__all__ = [
    "ANY",
    "Change",
    "Consent",
    "Demand",
    "Deps",
    "StewardState",
    "Obligation",
    "PendingCall",
    "Written",
    "answered",
    "duplicated",
    "invented",
    "misfiled",
    "mispriced",
    "performable",
    "pruned",
    "quoted",
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


class Consent(BaseModel):
    """The customer's own agreement to something the gate required, kept verbatim.

    The gate has no memory, so a condition it imposed one turn ago has to be
    re-derived from prose every time -- and 70 of the 166 write refusals in the
    50-task run were the same demand made again after the customer had already
    answered it. `Demand` fixed half of that by reminding the gate what it had
    asked. This is the other half: recording that the question came back answered,
    so the gate is reading a fact rather than re-reading a transcript.

    Evidence, never an approval. Nothing here lets a call through -- the gate still
    rules on every proposal, and all this changes is what it knows while ruling.
    The customer grants the permission; we only write it down.
    """

    action: str
    """The tool the demand was about, and so the tool this agreement covers."""

    reason: str
    """What the gate had required, verbatim, so the agreement stays attached to the
    question it answers."""

    words: str
    """What the customer actually said. Kept literally because a paraphrase of a
    consent is the one thing nobody should be asked to trust."""

    turn: int
    """The user turn that gave it."""


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


class Change(BaseModel):
    """One write the request needs, and the record it has to land on.

    Structured rather than prose because the speaker has to decide whether it has
    happened, and against prose it could only ask whether the tool's *name*
    appeared in an approved call. Where a request touches several records that is
    the wrong question by construction: "update_reservation_flights (once per
    reservation)" is a line one call satisfies and six calls' worth of work, and
    the run discharged it on the first every time.
    """

    # Described through `Field` rather than as attribute docstrings, because
    # pydantic only carries the latter into a JSON Schema when
    # `use_attribute_docstrings` is set -- so the planner was being handed three
    # unlabelled fields and filling in the two whose names spoke for themselves.
    tool: str = Field(description="The tool that makes this change.")

    record: str | None = Field(
        default=None,
        description=(
            "The identifier this change lands on -- the reservation, the order, the "
            "line. Fill it in whenever a lookup has already returned it. Null only "
            "when nothing has been looked up yet, or when the write creates the "
            "record it is about. Never guess one."
        ),
    )

    what: str = Field(
        default="",
        description=(
            "What this call has to change about that record, in a few words. Begin "
            "with the customer's own words asking for it, in quotes, then the "
            'instruction: \'"cancel my Boston flight" -- cancel it and refund to the '
            "original card'. If you cannot quote them, they did not ask, and the "
            "entry does not belong here. The assistant reads the rest as its "
            "instruction, so write it as one. Do not put the identifier here -- it "
            "belongs in `record`, and an identifier written only here leaves this "
            "change indistinguishable from every other change to the same tool."
        ),
    )

    @property
    def key(self) -> str:
        """What discharges this, as a line the planner can be shown."""
        return f"{self.tool} on {self.record}" if self.record else self.tool


# What a record identifier looks like anywhere, stated as loosely as it can be
# and still exclude English. Long enough that a year or a day of the month cannot
# match, and mixed enough that a field name cannot: `OBUT9V` and
# `sophia_silva_7557` qualify, `reservation_id`, `get_reservation_details`,
# `2024` and `the` do not. This is a heuristic and is only ever used to *discard*
# a record the ledger could not have matched on anyway -- see `anchored`.
TOKEN = re.compile(r"[A-Za-z0-9_]{5,}")


def performable(changes: list[Change], gated: frozenset[str]) -> list[Change]:
    """Changes whose `tool` is a write this domain actually has.

    `changes` is the list of writes the request needs, and the planner files three
    other things in it. Over run 017's 148 simulations, of 404 entries: 23 named a
    lookup or a handoff -- `get_reservation_details`, `search_direct_flight`,
    `calculate`, `transfer_to_human_agents` -- and 7 named a tool that does not
    exist, six of them `update_reservation_cabin`, which is the model reasoning
    from the policy's "Change cabin" heading to a call to match it. There is no
    such call; cabin is an argument to `update_reservation_flights`.

    Both kinds are worse than noise. A change is a debt: `outstanding` keeps it
    owed until an approved call discharges it, and neither a read nor a tool
    nobody can call ever will. The speaker then holds a reply against work that
    cannot be done, and the planner is asked again and re-files it. One entry
    naming `calculate` is a turn that cannot end.

    Dropping rather than repairing, because the repair is a guess. A planner that
    wrote `update_reservation_cabin` may have meant a cabin change, or a flight
    change, or both, and inventing the difference here would put a write into the
    ledger that nobody asked for -- which is the failure this whole seam exists to
    stop. What is dropped is not lost: the planner is asked again on the next
    lookup, holding the same request.
    """
    return [change for change in changes if change.tool in gated]


def misfiled(changes: list[Change], gated: frozenset[str], known: frozenset[str]) -> list[str]:
    """The lookups a plan filed as changes, as lines for `Plan.lookups`.

    `performable` drops them, and dropping alone would lose the one thing the entry
    got right: that the actor has to call this tool. Reads have their own field --
    `lookups` is rendered to the actor as "Find out first" -- so the fix is to move
    the line rather than delete it, and the plan says the same thing in the field
    that can carry it.

    Only tools this domain actually has. A change naming `update_reservation_cabin`
    is not a misplaced lookup, it is a call nobody can make, and putting it under
    "Find out first" would send the actor after a tool that does not exist.
    """
    return [
        f"{change.tool}: {change.what}".rstrip(": ")
        for change in changes
        if change.tool not in gated and change.tool in known
    ]


def anchored(changes: list[Change], seen: list[str]) -> list[Change]:
    """Changes with the record reduced to one the conversation has actually seen.

    The planner is asked for the identifier a change lands on, and a good deal of
    the time it answers with where the identifier is going to come from: "the same
    reservation id", "reservation_id_from_get_reservation_details", "the
    reservation ID that matches the Houston-to-Denver return flight on
    2024-05-27". Every one of those is a different `Change.key`, so a re-plan
    files each re-phrasing as a *new* commitment rather than recognising the one
    already held -- and none of them can ever be discharged, because `outstanding`
    matches an approved call's identifiers against this text and a placeholder
    contains none.

    Measured over the 15x2 run of 2026-08-29: 38% of the 5,136 ledger entries
    carried prose here, task 21 ended a conversation owing ten changes that were
    four, and the ledger only ever grew. That permanent debt is what kept
    `outstanding` non-empty on every write task, which is what the speaker held
    122 replies against.

    So a record is kept only where the conversation has seen it -- the same test
    `invented` puts on a tool call's arguments, for the same reason -- and reduced
    to `None` otherwise. `None` is not a loss: `outstanding` already falls back to
    matching on the tool alone, which is the most that can honestly be said about
    a change whose record nobody knows yet. The last match wins, because a record
    named after the phrase that introduces it is the shape the planner writes:
    "the reservation id from get_reservation_details for JG7FMM".
    """
    corpus = "\n".join(seen)
    return [
        change.model_copy(update={"record": _anchor(change.record, corpus)}) for change in changes
    ]


def _anchor(record: str | None, corpus: str) -> str | None:
    """The identifier in `record`, or None if there is not one in there.

    A record that is already a bare identifier is kept whether or not anyone has
    read it yet. Two bookings the customer names in one breath have been read by
    nobody, and dropping them both would merge two commitments into one -- the
    exact multi-record failure the ledger was built to stop. Nothing is risked by
    keeping one: `outstanding` still needs an approved call that names it, so an
    identifier the planner invented simply stays owed, which is the safe way for
    this to be wrong.

    Where the record is prose, the conversation has to settle which of its words
    is the identifier, because "the reservation for 2024-05-27" and "the
    reservation_id from get_reservation_details" both contain tokens that would
    otherwise pass for one.
    """
    if not record:
        return None
    if _identifierish(record.strip()):
        return record.strip()
    found = [token for token in TOKEN.findall(record) if _identifierish(token) and token in corpus]
    return found[-1] if found else None


def _identifierish(token: str) -> bool:
    """Long enough that a year cannot match, mixed enough that a field name cannot."""
    return (
        len(token) >= 5
        and TOKEN.fullmatch(token) is not None
        and any(c.isdigit() for c in token)
        and any(c.isalpha() for c in token)
    )


class Written(BaseModel):
    """A gated call the gate approved, as the ledger remembers it.

    Approved rather than executed: what the speaker is asking is whether the actor
    walked away from work, and a call that was approved and then failed in the
    environment was not walked away from.
    """

    tool: str
    records: list[str] = Field(default_factory=list)
    """Every identifier the call named. Which of them is "the" record is a
    question about a domain, and `core` does not have one -- so the ledger keeps
    them all and lets the plan's own record decide the match."""

    @classmethod
    def of(cls, name: str, arguments: dict[str, Any]) -> Written:
        return cls(tool=name, records=[value for _, value in _identifiers(arguments)])


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
    """Attempts the critic has blocked this user turn. Bounds the argument.

    The critic's refusals and the verifiers' are counted apart because they mean
    different things. A model that refused an action will refuse it again, so a
    small budget is the right one: two rounds of arguing with it is generous. A
    verifier states a fact about the record, and the actor's correct response to
    it is usually not to retry at all but to go and do something else -- which
    must not be paid for out of the argument budget."""

    blocked: int = 0
    """Attempts a verifier has stopped this user turn. Bounds the sieve.

    One turn can legitimately spend many of these. Task 37's customer asks for
    three reservations at once, two of which the policy forbids touching; task 41
    names seven. Every forbidden one is a block, and the turn is going correctly
    while they happen -- the actor is meant to work through them and complete the
    requests that *are* allowed."""

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

    before: str = ""
    """The goal and the lookups of the plan written last, for the next one to read.

    The planner is a fresh call every time, so it cannot otherwise tell a first
    attempt at a question from a fourth, and run 020 measured what that costs: 42%
    of plans named no change at all, goals opening with "determine" ran at 0.295
    per plan, and one task wrote six consecutive plans meaning collect, calculate,
    determine while every fact it had asked for was already in hand.

    Held as rendered text rather than as the `Plan`, because the only consumer is
    a prompt and a stored object would be a second copy of `changes` waiting to
    disagree with the first."""

    request: str = ""
    """What the customer asked for, at the scope they asked for it.

    Held apart from `plan` and rewritten on a different clock. The plan is
    reworked every time a lookup comes back, and the run shows what that does to
    the scope: one task's goals went "a reservation", "reservation(s)", "each of
    Omar Davis's reservations", "reservation JG7FMM", and then JG7FMM three more
    times -- the request shrinking to the record that happened to have been read
    most recently, with the other four never mentioned by anybody again.

    So this is settable only when the customer has spoken. A mid-turn re-plan
    carries it unchanged, which makes the narrowing above unrepresentable rather
    than merely discouraged -- the lookup that caused it is exactly the thing that
    can no longer rewrite it. See `_plan`."""

    opened: str = ""
    """The first request of the conversation, never rewritten.

    The rule above stops a *lookup* narrowing the scope. It cannot stop the next
    customer turn doing it, because the planner re-derives the request from a
    conversation the assistant's own framing now dominates, and by then the
    framing may be wrong. Task 41 asked to cancel "all of my upcoming flights that
    only have me on the reservation"; the assistant proposed four reservations
    with two and three passengers on them, the customer agreed to what it offered,
    and the next request read "cancel all four reservations and process the $90
    refund" -- the criterion that made three of those four wrong, gone.

    Kept beside `request` rather than instead of it. A customer really may change
    their mind, and task 7 is a task about exactly that, so neither one is the
    truth on its own. Both are shown to the gate and the difference between them
    is the thing worth seeing."""

    changes: list[Change] = Field(default_factory=list)
    """The writes the request needs, as the planner has described them so far.

    Kept beside the rendered `plan` rather than parsed back out of it, because the
    speaker counts these against `written` and a count taken off rendered prose is
    a count waiting to drift.

    Not reset when the customer speaks again. It used to be, and that is the
    single defect the run turned on: a request covering six reservations was
    planned once, correctly, and then re-planned from scratch the moment the
    customer replied -- so the five reservations still owed stopped being owed by
    anybody. Across four runs, tasks needing writes on more than one record were
    completed zero times out of three. A commitment now leaves this list one way
    only, by being carried out."""

    written: list[Written] = Field(default_factory=list)
    """The gated calls the gate has approved, for as long as the conversation
    lasts. Carried across user turns with `changes`, for the same reason: a ledger
    of what is owed is worth nothing beside a ledger of what is done that resets
    underneath it."""

    ruled_out: list[Written] = Field(default_factory=list)
    """Changes a verifier has settled as not permitted, in the same shape as the
    ones that happened.

    The second way a commitment may leave `changes`. "A commitment leaves only by
    being carried out" is right about the writes it was written about, and wrong
    about the ones the policy forbids: the customer asks for two things, one of
    them is not allowed, and without this the refused one is owed for the rest of
    the conversation. Everything reading `outstanding` is then reasoning from a
    debt that can never be paid -- the planner is told the turn is unfinished, the
    critic is handed it as context for every later proposal, and the speaker is
    asked about it on every exit.

    It was built for a stronger reason than that, and the stronger reason did not
    survive. A verifier that refused a handoff while work was owed would have made
    an unpayable debt wedge the exit shut, which made this a correctness
    requirement rather than a tidiness one. That verifier is gone -- measured over
    a 15-task run it fired nine times where transferring was correct and nine
    times where it was not, which is no signal at all, and it fired on the one
    task whose gold action *is* a transfer. What that run also showed is that
    nothing ever reached this ledger: 0 writes in 242 gate decisions, because the
    refusals in question came from the critic in prose and the rule below bars the
    critic from settling anything. So this is kept for what it does to the three
    readers above, and it is honest to say it has not yet been observed to fire.

    Only a *deterministic* refusal writes here, and only one that is not
    recoverable. Both halves are load-bearing. Arithmetic over the record is the
    only ruling in this system that can be replayed and checked, and the whole
    reason `core.verifiers` exists is that the critic's opinion at 20B is close to
    a coin -- a model that could retire an obligation by calling it impossible
    would be able to talk itself out of the work, which is the failure this ledger
    was built to stop. `recoverable` refusals are excluded because they are not
    rulings at all: they say "send this differently", and the change stands.

    Matched against `changes` by tool *and* record, so refusing one reservation
    leaves the other five owed -- the multi-record defect that `changes` records
    is not reintroduced here."""

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

    consented: list[Consent] = Field(default_factory=list)
    """Demands the customer has since answered, latest per action.

    A demand leaves `demanded` only by arriving here, so the two lists never both
    describe the same action and the gate is never shown a standing condition and
    its answer at the same time."""

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


# What counts as the customer saying yes. Deliberately short and deliberately
# literal: the cost of missing an agreement is one wasted turn, and the cost of
# inventing one is a write nobody asked for, so anything that has to be
# interpreted is left to the gate rather than decided here.
AGREED = (
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "confirm",
    "confirmed",
    "agree",
    "agreed",
    "proceed",
    "please do",
    "go ahead",
    "do it",
    "sounds good",
    "that works",
    "correct",
)

# The action a consent covers when the customer was answering the assistant's own
# question rather than a condition this gate imposed. A tool name would be a
# guess -- the assistant asked in prose and the answer is to the prose -- so the
# consent is filed against every action and the gate reads the question to decide
# what it actually covers.
ANY = "*"

# Any of these and the message is not treated as agreement, whatever else is in
# it. "Yes, but not that one" and "yes -- wait" are the sentences this exists for.
REFUSED = (
    "no",
    "not",
    "don't",
    "do not",
    "dont",
    "wait",
    "hold on",
    "hold off",
    "stop",
    "instead",
    "actually",
    "but",
    "however",
    "cancel that",
    "never mind",
    "nevermind",
)


def answered(
    demanded: list[Demand], reply: str, turn: int, asked: str = ""
) -> tuple[list[Demand], list[Consent]]:
    """Move every demand this message answers out of `demanded` and into consent.

    Returns both lists so the caller writes them together: a demand that has been
    answered and a demand that is still standing are the same object in different
    places, and updating one without the other is how the gate ends up shown both.

    Only demands from an *earlier* turn are eligible. One made during the turn in
    progress has not been put to the customer yet, so a "yes" in the message that
    provoked it is agreement to something else.

    `asked` is the assistant's own last message, and it is the half this was
    missing. Consent only ever entered the ledger through a demand *this gate*
    had made, so the first confirmation in a conversation was always invisible:
    the assistant says "the difference is $340, shall I go ahead?", the customer
    says yes, no demand exists, nothing is recorded, and the gate refuses for
    want of an agreement it had already been given. Measured on the 381 gate
    decisions of the 50x3, 17 of the 23 gold writes it refused were exactly that.
    The ledger was one refusal late.

    What is recorded is evidence, not permission -- the question and the answer,
    both verbatim, under `ANY` because the assistant's question is not about a
    tool. Whether a yes to "shall I look that up?" covers a booking is a
    judgement, and it stays with the gate, which is shown both halves and can see
    the difference.

    The test is a word match, not a judgement. It is one-sided on purpose: an
    unrecognised agreement leaves the demand standing and costs a turn, while a
    misread one would record a permission the customer never gave. So a message
    carrying any hesitation at all is treated as no answer, and the gate goes on
    reading the transcript itself.
    """
    if not _agrees(reply):
        return list(demanded), []
    standing = [demand for demand in demanded if demand.turn >= turn]
    given = [
        Consent(action=demand.action, reason=demand.reason, words=reply.strip(), turn=turn)
        for demand in demanded
        if demand.turn < turn
    ]
    if asked.strip():
        given.append(Consent(action=ANY, reason=asked.strip(), words=reply.strip(), turn=turn))
    return standing, given


def _agrees(reply: str) -> bool:
    """Whether this message is an unqualified yes."""
    words = re.findall(r"[a-z']+", reply.lower())
    if not words:
        return False
    text = " ".join(words)
    if any(_says(text, words, phrase) for phrase in REFUSED):
        return False
    return any(_says(text, words, phrase) for phrase in AGREED)


def _says(text: str, words: list[str], phrase: str) -> bool:
    """Whole words only. "ok" must not match "book", and "no" must not match "now"."""
    if " " in phrase:
        return phrase in text
    return phrase in words


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


def quoted(value: str, texts: list[str]) -> str:
    """The most recent of `texts` holding `value`, clipped around it. "" if none.

    `sources` searches everything the system has been shown, which answers "was
    this value established" and cannot answer "is this the record the customer
    meant" -- a reservation the customer owns but never mentioned is quoted from
    the lookup that returned it and looks exactly as grounded as one they named.
    Given only the customer's own turns this answers the second question, and the
    two together are what the gate needs: task 41 lost on three cancellations and
    task 1 on one, every one of them an identifier the customer never typed.
    """
    return _around(next((text for text in reversed(texts) if value in text), ""), value)


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


# How far either side of an identifier a figure counts as quoted with it, used
# only where the surrounding text is not JSON and there is no object to read
# instead. Generous on purpose: this is the fallback that must not invent a
# finding, and a missed mispricing costs less than a wrongly flagged fare.
WINDOW = 400

# Argument names that carry money. Narrower than "any number": a date, a seat
# count or a passenger age has no reason to appear beside a flight number, and
# flagging those would bury the one finding that matters.
_MONEY = ("price", "amount", "total", "cost", "fee")

# What marks an entry as settling a bill rather than quoting a price.
_SETTLEMENT = "payment"


def mispriced(arguments: dict[str, Any], observed: list[str]) -> list[str]:
    """Money quoted against an identifier it was never shown beside, as paths.

    The third deterministic check, and the one that answers what the other two
    cannot see. `invented` asks whether a value was shown at all and `duplicated`
    whether an entry is a copy; a price is neither invented nor duplicated when it
    is simply the wrong flight's. That is how the run lost a booking -- it charged
    twice the economy fare of a flight the customer was not taking, a figure that
    was genuinely in the transcript, attached to a flight number that was also
    genuinely in the transcript, just not to each other.

    Evidence and not a verdict, like `findings` around it: a fare the assistant
    correctly worked out -- a difference, a total across two legs -- appears
    nowhere beside anything and is not wrong for that. What it catches is the
    figure copied off the neighbouring row.
    """
    flags = []
    for path, entry in _entries(arguments):
        # What is paid is settled, not quoted. An amount put against a payment
        # method is a remainder worked out from everything else in the basket, so
        # it is never shown beside the instrument it is charged to and cannot be
        # judged by whether it was: a card carries a brand and a last_four and no
        # balance at all. Replayed against the benchmark's own gold actions this
        # was the only thing the check ever flagged -- three correct payment
        # splits on task 23, for 44, 621 and 621, which add up to the one figure
        # that task is scored on.
        if any(_SETTLEMENT in name.lower() for name in entry):
            continue
        money = {
            name: value
            for name, value in entry.items()
            if name.lower() in _MONEY
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        }
        if not money:
            continue
        windows = [
            window
            for _, identifier in _identifiers(entry)
            for window in _windows(identifier, observed)
        ]
        # No window means the identifier itself was never shown, which is
        # `invented`'s finding and not this one's. Reporting it here as well would
        # be the same fault counted twice.
        if not windows:
            continue
        # A window with nothing priced in it cannot be where a figure was copied
        # from, so an amount missing from one is not evidence of anything.
        if not any(_priced(window) for window in windows):
            continue
        flags += [
            f"{path}{name}"
            for name, value in money.items()
            if not any(_figure(value) in window for window in windows)
        ]
    return flags


def _priced(window: str) -> bool:
    """Whether this window has any money in it at all.

    The case this exists for is a credit card. Its record carries a brand and a
    `last_four` and no balance, so the sum charged to one is always a remainder
    worked out somewhere else and can never appear beside the card -- which made
    every correctly split payment look like a figure taken off the wrong row.
    Replayed against the benchmark's own gold actions, that was the only thing
    this check ever flagged: three `book_reservation` calls on task 23, for 44,
    621 and 621, which add up to the single figure that task is scored on.

    The original catch is untouched. A flight row is full of prices, so the fare
    read off the neighbouring flight still has a priced window to be missing from.
    """
    return any(name in window.lower() for name in _MONEY)


def _entries(value: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Every dict inside a JSON value, with the path to it. The call's own
    arguments count: a top-level `total_price` beside a top-level id is the same
    mistake one level up."""
    if isinstance(value, dict):
        return [(path, value)] + [
            found for name, item in value.items() for found in _entries(item, f"{path}{name}.")
        ]
    if isinstance(value, list):
        return [
            found
            for index, item in enumerate(value)
            for found in _entries(item, f"{path.rstrip('.')}[{index}].")
        ]
    return []


def _windows(identifier: str, observed: list[str]) -> list[str]:
    """The object around each place an identifier was shown.

    The object and not a span of characters. A flight search returns one entry per
    flight, and every entry has a price in it, so any window wide enough to hold
    one is wide enough to hold its neighbour -- which is the confusion this whole
    check exists to catch. Matching braces is what makes "shown beside" mean the
    same row rather than the same screenful. Tool results are JSON; text that is
    not falls back to the span.
    """
    return [
        _object_at(text, at, len(identifier))
        for text in observed
        for at in _occurrences(text, identifier)
    ]


def _object_at(text: str, at: int, length: int) -> str:
    """The record an identifier names, falling back to the one it sits inside.

    An identifier is not always a field within a record; often it is the key the
    record hangs off, as `payment_methods` hangs every card off its own id. Walking
    outwards from a key lands on the object holding *all* of them, so every balance
    in the block counts as shown beside every card -- which is the same
    same-screenful confusion `_windows` matches braces to avoid, one level up.
    """
    named = _value_object(text, at + length)
    return named if named is not None else _object_around(text, at)


def _value_object(text: str, after: int) -> str | None:
    """The object an identifier introduces as a key, as in `"X": { ... }`."""
    index = after
    while index < len(text) and (text[index] in '" :' or text[index].isspace()):
        index += 1
    if index < len(text) and text[index] == "{":
        return text[index : _closing(text, index)]
    return None


def _object_around(text: str, at: int) -> str:
    """The innermost JSON object enclosing position `at`, or a span if there is none."""
    start = _opening(text, at)
    if start is None:
        return text[max(0, at - WINDOW) : at + WINDOW]
    return text[start : _closing(text, start)]


def _opening(text: str, at: int) -> int | None:
    depth = 0
    for index in range(at - 1, -1, -1):
        if text[index] == "}":
            depth += 1
        elif text[index] == "{":
            if depth == 0:
                return index
            depth -= 1
    return None


def _closing(text: str, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def _occurrences(text: str, needle: str) -> list[int]:
    found, at = [], text.find(needle)
    while at != -1:
        found.append(at)
        at = text.find(needle, at + 1)
    return found


def _figure(value: float) -> str:
    """A number as the corpus would spell it: `114`, not `114.0`."""
    return str(int(value)) if float(value).is_integer() else str(value)


def pruned(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """The same arguments with every key the schema does not declare removed.

    tau2 scores an action by comparing the whole argument dict it received against
    the gold one, over the keys *we* sent -- `tool_args == action_args` in
    `tasks.Action.compare`. So an extra key nothing reads is not ignored, it is a
    miss. The environment discards it, the database ends up identical, and the run
    is still marked wrong.

    That is not hypothetical: six of the write misses in the 50-task run were calls
    the model got right and then decorated, putting `origin`, `destination` and
    `price` inside `flights` entries whose schema declares `flight_number` and
    `date`. The model is not confused about the itinerary; it is padding an object
    with what it happens to know.

    Removing rather than refusing, because this is decidable outright: the schema
    the model was shown is the schema the environment accepts, and a key outside it
    can have no effect on anything. A `ModelRetry` would spend a round trip to
    reach the same dict, and the published gate ablations are consistent about what
    a check that makes the model think again costs when the deterministic answer
    was already in hand.
    """
    return _prune(arguments, schema, schema)


def _prune(value: Any, node: Any, root: dict[str, Any]) -> Any:
    """One value against one schema node, following `$ref` back to `root`."""
    node = _resolved(node, root)
    if not isinstance(node, dict):
        return value
    # A choice of shapes is a choice of key sets, and picking the wrong one would
    # delete a key that was right. `tighten` collapses the only such branch tau2
    # produces, so this is the guard for a schema nobody has looked at yet.
    if any(key in node for key in ("anyOf", "oneOf", "allOf")):
        return value
    if isinstance(value, dict):
        properties = node.get("properties")
        if not isinstance(properties, dict) or node.get("additionalProperties"):
            return value
        return {
            name: _prune(item, properties[name], root)
            for name, item in value.items()
            if name in properties
        }
    if isinstance(value, list) and isinstance(node.get("items"), dict):
        return [_prune(item, node["items"], root) for item in value]
    return value


def _resolved(node: Any, root: dict[str, Any]) -> Any:
    """A `#/$defs/Name` reference, followed. Anything else, unchanged."""
    if isinstance(node, dict) and isinstance(node.get("$ref"), str):
        name = node["$ref"].removeprefix("#/$defs/")
        return root.get("$defs", {}).get(name, node)
    return node
