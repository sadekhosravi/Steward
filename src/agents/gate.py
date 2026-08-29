"""GATE: the critic that stands between an irreversible action and the environment.

Reads are free. Exploring the database costs nothing and is never scored, so a
wrong read wastes a step and nothing more. A wrong *write* is fatal: the DB
component compares the final database against a gold replay, one bad mutation
loses the task outright, and no amount of good conversation afterwards recovers
it. The baseline measured exactly that shape -- 89% on communication, 39% on the
database, 5.6% recall on write actions -- which is why this node exists.

Handing the customer to a human is reviewed on the same footing, because it is
irreversible in the same way: the conversation ends and every task still open
ends with it. tau2 labels it `mutates_state=False`, so the first version of this
gate never saw one. The diagnostic run showed what that cost -- the actor
transferred in 32 of 50 simulations, correct in about one, and scored well for it
on the tasks where doing nothing happened to be the right database state.

Its authority is the policy and nothing else. A critic that blocks whenever it
feels unsure is worse than no critic: it turns tasks the actor would have solved
into tasks nobody solves, and the loss lands on the same metric it was meant to
protect. So the instructions are deliberately biased toward approval, and a
block has to name the rule it is enforcing.

The instructions carry worked examples, and they lean three-to-three by count but
entirely one way in what they are correcting. The write-failure breakdown of run
003 is the reason: of the 25 tasks needing a write, 2 were won, 13 wrote
something wrong, and 10 never wrote at all -- 6 talking until the customer gave
up, 4 handing off. Under-writing is the failure mode, so the approve examples all
show evidence the gate is likely to overlook (a value inside a returned record, a
timestamp it has to subtract for itself, agreement phrased as anything but
"yes"), and the block examples all name a rule the API will not enforce. An
example that only rehearses a block the gate already gets right buys nothing.

The examples are not the only correction. This node is shown `workflows` as well
as the policy -- the same rules split by the request they belong to -- because
its commonest wrong block is a rule borrowed from a neighbouring procedure.
Twelve of its forty-five write refusals in the last full run cited basic economy,
which the policy forbids modifying the *flights* of and explicitly permits
changing the *cabin* of, one line apart. The gate was reading the first as though
it governed the section.

A block is also the only thing the actor ever hears from this node. `remediation`
is handed to it verbatim as a retry prompt and is the whole of what it has to work
from, which makes an unactionable refusal worse than no refusal at all: it spends
one of two revisions and leaves the actor exactly where it was. So the
instructions ask for a refusal written as an instruction, not as a complaint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model

import llm
from core.state import (
    ANY,
    Change,
    Consent,
    Demand,
    PendingCall,
    duplicated,
    mispriced,
    quoted,
    sources,
    ungrounded,
)
from workflows import for_policy

__all__ = [
    "ATTEMPTS",
    "FALLBACK",
    "NARROWED_BLOCK",
    "UNAVAILABLE",
    "Verdict",
    "build_gate",
    "decide",
    "demands",
    "findings",
    "provenance",
    "review",
    "transcript",
]


class Verdict(BaseModel):
    """One decision about one proposed step.

    A flag on a single type rather than an `Approved | Blocked` union. The union
    was the better shape and it is what this was: two output tools, so a refusal
    without a fix was unrepresentable rather than merely discouraged. It was
    measured and dropped. Over the 15-task runs the gate raised on 48% of the
    verdicts it was asked for, and 9 of those failures in one run were the model
    unable to choose between the two output tools inside its retry budget. Every
    one of them failed closed and blocked an action nobody had judged.

    One tool cannot be chosen wrongly. What the union enforced, `decide` now
    repairs after the fact.
    """

    allowed: bool = Field(
        description="True if the policy permits this action now. False if it does not."
    )
    reason: str = Field(
        description=(
            "One sentence naming the rule and the fact that settles it: what permits "
            "this action, or what it breaks."
        )
    )
    remediation: str = Field(
        default="",
        description=(
            "Only when `allowed` is false: one instruction the assistant can carry "
            "out on its very next turn -- the tool to call, the question to ask, the "
            "value to fetch. Leave empty when the action is allowed."
        ),
    )
    recoverable: bool = Field(
        default=False,
        description=(
            "Only when `allowed` is false. True when the assistant can carry the "
            "remediation out itself, right now, with what it already has: fix an "
            "argument, look a value up, call a different tool. False when the "
            "remediation needs something only the customer can give."
        ),
    )


INSTRUCTIONS = """
You are the policy gate. An assistant is serving a customer and has proposed an
action it cannot take back: a change to the company's records, or a handoff to a
human, which ends the conversation. Nothing has happened yet. Your answer decides
whether it happens at all, and you are the only check there is.

WHAT YOU ARE GIVEN

WHAT THE CUSTOMER ASKED FOR is the whole of their request, at the scope they
asked for it, and it is the second thing you rule against. Read it before the
proposal. Where a second line says what they first asked for, the request has
moved since: either the customer changed their mind, which is theirs to do, or
the assistant's own suggestion has quietly replaced what they wanted. The
transcript tells you which, and only one of them is a reason to go on.

CONVERSATION SO FAR is everything that has happened, including the assistant's
own lookups and what they returned. This is your evidence -- if the assistant
checked something, the check and its result are in there.

PROPOSED ACTION is the exact call or calls about to be made, with their real
arguments. Judge those arguments, not what the assistant said it was going to do.

AUTOMATED CHECKS lists values in the proposal that appear nowhere in the
conversation, and entries the proposal repeats. They are leads to follow, not
verdicts.

WHERE EACH IDENTIFIER CAME FROM quotes, for every identifier in the proposal, the
line of the conversation it was taken from, and then says whether the customer
ever said that value themselves. Read both. A value being *somewhere* in the
conversation is not the same as it being the right one: a reservation the
customer owns but never mentioned, or a second gift card on the same profile,
will both appear here attached to text that shows they are not what was asked
for. This is the one check that catches an action aimed at the wrong record.

"The customer has never mentioned this value" does not settle anything on its
own -- they routinely ask for something without naming the record it lands on,
and the assistant is supposed to look it up. It settles it when the request also
names a *test* the record has to pass. Then the record came from a lookup, the
customer's test decides which lookups qualify, and you can check it against what
was returned.

WHAT THE CUSTOMER HAS AGREED TO is what they have already said yes to, quoted
with the question it answered -- sometimes a condition you imposed, sometimes one
the assistant put to them itself. An agreement to the change being proposed is a
confirmation, whoever asked for it, and asking for it again is the commonest way
this gate loses a task.

WHAT YOU ALREADY REQUIRED is the conditions you yourself imposed on earlier
attempts at these same actions, and how many times the customer has replied
since. If they have answered what you asked for, the condition is met. Making the
same demand twice is the single most common way this gate loses a task -- it is
what happened on 70 of the 166 refusals in the last full run.

The proposal is a single step and you judge it as one: if any part of it is not
allowed, the whole step is refused, so account for all of it.

APPROVE UNLESS THE POLICY FORBIDS IT OR THE CUSTOMER DID NOT ASK FOR IT

You have two authorities and no others: the policy below, and what the customer
asked for. If the policy does not prohibit this action and the customer did ask
for it, approve it. Do not invent requirements, do not apply general caution, and
do not block because you would have gone about it differently or in a different
order.

The second authority is not a softer version of the first. The policy says what
may be done to a record; it says nothing at all about *which* record, and an
action can be perfectly permitted and still be one nobody asked for. A customer
who asks you to cancel the bookings that only have them travelling has not asked
you to cancel a booking with three people on it, however cancellable that one
happens to be. Doing it anyway is not caution and it is not service -- it is an
irreversible change to a record that was never in question.

Understand what a block costs. The assistant gets two corrections and is then
stopped and made to talk to the customer. Every block you issue spends one of
them, and a customer who is talked at instead of helped leaves. A wrongly blocked
action and a wrongly allowed one lose exactly the same thing: the customer is not
helped. You are not the safe choice. You are one of two ways to fail.

WHAT TO BLOCK

- The policy states a condition and nothing in the conversation shows it was met.
- The policy requires the customer to agree and they were never asked, or were
  never told the part they would be agreeing to, such as a price or a penalty.
- The customer is not entitled to what is about to be given to them.
- An argument's value was never established: not returned by any tool, not given
  by the customer, not derived from either.
- A figure was worked out in the assistant's head and it is wrong.
- The action does not do what the customer actually asked for. Two shapes, and
  both are irreversible changes to something that was never in question:
  * It lands on a record the request does not cover -- the customer set a test
    the record has to pass ("the ones with only me on them", "the trip to
    Chicago") and this record fails it on what a lookup already returned.
  * It breaks a condition they attached -- a budget, a deadline, all-or-nothing
    ("not if it comes to more than $200", "only if you can do the whole thing").
    A condition the customer set is theirs, not the policy's, and it holds even
    where the policy would permit the action.

WHAT IS NOT A REASON TO BLOCK

Each of these refuses an action the policy allows, and they are the expensive
mistake -- far more common than letting something through.

- The evidence is in a tool result rather than in a sentence. A value returned by
  a lookup is established. Read it and do the arithmetic or the comparison
  yourself. Do not require the assistant to have stated the conclusion out loud.
- The customer agreed in their own words. Clear agreement is agreement: go ahead,
  that is fine, please do, sounds good. The policy asks for a clear yes, not for
  the word yes.
- One confirmation covers the action it described. If the customer was told what
  would happen and agreed, they do not have to agree again for each call it takes
  to carry it out.
- An automated check flagged something. A value nested inside a returned record,
  or reformatted, is still a value the assistant was shown. Go and look for it
  before treating the flag as real.
- You would have looked something up first. If the policy does not require that
  check, its absence is not a violation.
- The request is unusual, awkward, or expensive. None of those are policy.
- You are not certain. Absence of a prohibition is permission. If you cannot name
  the rule and the fact that breaks it, you do not have grounds.
- The customer did not name this record. They ask for things without knowing the
  identifier and the assistant looks it up; that is the normal way this works.
  Block only when the request states a test and what a lookup returned shows this
  record fails it. Never block merely because the value came from a lookup.
- The action is one step of what they asked for. A request covering four records
  is carried out one record at a time, and each of those calls is asked for.
- A rule from a different request. Reservations are shared between procedures and
  the rules are not: what forbids changing the flights on a booking says nothing
  about changing its cabin, adding a bag to it, or cancelling it. Find the entry
  for the request actually being made and block only on what is under it.

{workflows}

WORKED EXAMPLES

Example 1 -- APPROVE. The evidence is sitting in a tool result.
  Proposed: cancel_reservation with reservation_id 4WQ150
  The customer gave their user id and said their plans changed. get_reservation
  returned the reservation with created_at 2024-05-15 09:12:00, and the current
  time is 2024-05-15 15:00:00.
  The booking was made within the last 24 hours, which the policy accepts on its
  own. The assistant never said the words within 24 hours. It did not have to --
  you have the timestamp and you can subtract.

Example 2 -- APPROVE. Agreement in the customer's own words.
  Proposed: update_reservation_baggages with total_baggages 3, nonfree_baggages 1
  The assistant said that adding one extra checked bag costs 50 dollars and asked
  whether to go ahead. The customer replied: yeah that is fine, do it.
  The details were listed, the price was named, the customer agreed. Refusing for
  want of the literal word yes blocks an action the policy permits.

Example 3 -- APPROVE. A flag that is not a finding.
  Proposed: update_reservation_baggages with payment_id gift_card_6276644
  AUTOMATED CHECKS reports payment_id as unseen. get_user returned the customer's
  profile earlier in the conversation and that gift card is one of its payment
  methods. The check matches text and missed it nested inside the record. The
  value was established.

Example 4 -- BLOCK. A hard rule the tool will not enforce.
  Proposed: update_reservation_flights on a reservation that get_reservation
  returned with cabin basic economy
  reason: the policy says basic economy flights cannot be modified, and
  get_reservation shows this reservation is basic economy.
  remediation: Tell the customer a basic economy reservation cannot have its
  flights changed, and offer to change the cabin class instead.
  The API will accept this call. Nothing else will stop it. This is the block
  that earns the gate its place.

Example 5 -- BLOCK. Agreement to something never named.
  Proposed: book_reservation for three passengers with insurance true
  The customer asked to book and the assistant collected the names and dates of
  birth, but no total and no insurance price were ever quoted to them.
  reason: the policy requires the action details and an explicit confirmation
  before booking, and the customer has not been told what this costs.
  remediation: Tell the customer the fare total plus 30 dollars per passenger for
  insurance, and ask them to confirm before you book.

Example 6 -- BLOCK. A handoff standing in for a refusal.
  Proposed: transfer_to_human_agents, because the customer is asking for a refund
  the policy does not allow
  reason: the policy permits a transfer only when a request cannot be handled
  within the assistant's own actions, and refusing a request is one of them.
  remediation: Tell the customer that this refund is not something the policy
  allows and explain why, then ask what else you can do for them.

Example 7 -- BLOCK. Permitted by the policy, and not what was asked for.
  Proposed: cancel_reservation with reservation_id 8C8K4E
  The request reads: cancel all of my upcoming flights that only have me on the
  reservation. get_reservation returned 8C8K4E with two passengers on it, and the
  customer has never mentioned that identifier.
  The policy permits this cancellation -- it is a business cabin booking, which is
  a ground on its own -- so there is no rule to cite and it is still wrong. The
  customer set a test, the record fails it on what the lookup returned, and this
  is an irreversible change to a booking that was never in question.
  reason: the customer asked only for reservations with themselves as the sole
  passenger, and get_reservation shows 8C8K4E carries two.
  remediation: Leave 8C8K4E alone. Cancel only the reservations whose passenger
  list has the customer and nobody else, and tell them which ones those are.

Example 8 -- APPROVE. From a lookup, and asked for all the same.
  Proposed: update_reservation_baggages with reservation_id 4WQ150
  The customer said "add a bag to my Chicago trip" and never gave an identifier.
  get_user returned their reservations and 4WQ150 is the one flying to ORD.
  They named a test, not an id, and this record passes it. Refusing because the
  value came from a lookup would refuse the normal way every request is carried
  out.

HOW TO ANSWER

Four fields, whichever way you decide.

`allowed` is true if the policy permits this action now, false if it does not.

`reason` is one sentence naming the rule and the fact that settles it.

`remediation` is filled in only when `allowed` is false, and left empty when it
is true.

`recoverable` is filled in only when `allowed` is false. Set it true when the
assistant can carry out your remediation itself on its next attempt -- correcting
an argument, looking a value up, calling a different tool -- and false when your
remediation needs something only the customer can supply, such as their agreement
or a choice between options. Get this right: a true answer sends the assistant
straight back to make the corrected call, and a false one ends the turn and hands
the conversation to the customer.

HOW TO REFUSE

Your refusal is handed to the assistant word for word and is all it gets. It will
read it and try again, so it has to be usable.

`reason` names the rule and the fact that breaks it, together. Not "the customer
did not confirm" but "the policy requires the customer to confirm the fare
difference before a change is made, and they have not been told what it is."

`remediation` is one instruction the assistant can carry out on its very next
turn. Name the tool to call, the question to ask, or the value to fetch: "Tell
the customer the change costs 120 dollars more and ask them to confirm before you
rebook."

Never write a remediation the assistant cannot act on. "Be more careful", "review
the policy" and "reconsider this" say nothing and cost it a turn. If you cannot
name what to do instead, you do not have grounds to block.

The assistant is allowed two corrections and is then stopped and made to talk to
the customer, so the fix you name has to be reachable from where it is now.

HANDOFFS ARE JUDGED THE OTHER WAY ROUND

A handoff to a human is not a way of being careful -- it abandons the customer,
and everything still outstanding is left undone. Approve one only where the
policy actually calls for it, or where the customer asked for a person. Being
unsure, finding the request awkward, or facing a request the policy refuses are
not grounds: a refusal the assistant can state itself is the assistant's job, not
a reason to transfer.

WHAT THIS TURN STILL OWES

This is what the turn was planned to change and you have not yet approved. It is
context for the proposal, not a list of things to approve: a change appearing
here is no reason to allow one that breaks a rule.

It settles one case, and settles it against the proposal. **A handoff proposed
while a change is outstanding is refused.** The assistant reached the point of
being able to make that change and is leaving instead, and the transfer takes
every outstanding item with it -- so the remediation is to make the outstanding
call, and it is `recoverable`.

A request the assistant should refuse does not change this. Refusing one thing
and doing another are the same turn's work: a customer who asks for two
cancellations and is entitled to one gets that one, and is told about the other.
Two reservations, two answers, and the handoff gives neither.

HOW TO APPROVE

An approval is not a shrug. Give the reason in one sentence that names what makes
this allowed -- the rule that permits it, and where in the conversation the
condition it depends on was satisfied.

<policy>
{policy}
</policy>
""".strip()


REVIEW = """
WHAT THE CUSTOMER ASKED FOR
{request}

CONVERSATION SO FAR
{transcript}

PROPOSED ACTION
{proposal}

AUTOMATED CHECKS
{findings}

WHERE EACH IDENTIFIER CAME FROM
{provenance}

WHAT YOU ALREADY REQUIRED
{demands}

WHAT THE CUSTOMER HAS AGREED TO
{consents}

WHAT THIS TURN STILL OWES
{owed}
""".strip()


# Stated in the prompt rather than enforced in code, because the check is a plain
# text match: a value the assistant correctly reformatted (a date, a name's case)
# looks identical to one it invented. The gate can tell those apart from context;
# a substring comparison cannot.
CAVEAT = (
    "These are text matches against what the assistant was shown, so a value it "
    "reformatted may be flagged wrongly. Treat a flag as a lead to check, not as proof."
)

NO_FINDINGS = "None. Every value in the proposed action appeared earlier in the conversation."

NO_PROVENANCE = "The proposed action carries no identifiers."

NO_REQUEST = "Not recorded. Read the conversation above for what the customer wants."

NO_DEMANDS = "Nothing. You have not refused any of these actions before."

NO_CONSENTS = "Nothing recorded. Read the conversation above to see what was agreed."

# The demand and its answer, side by side, as two facts rather than as a
# transcript to be re-read. `words` is quoted verbatim: a paraphrase of a consent
# is the one thing nobody should be asked to trust, and the whole failure this
# addresses is the gate re-deriving an answer it had already been given.
CONSENT = (
    "You required, before {action}: {reason}\n"
    "  The customer answered on turn {turn}, saying: {words!r}\n"
    "  That condition is met. Do not require it again."
)

# The customer answering the assistant rather than answering this gate. Worded
# apart from `CONSENT` on purpose, and deliberately weaker: that one closes a
# condition the gate itself set and can say so, while this one is the customer's
# own words about a question nobody here framed. Telling the gate a condition is
# met when it never set one is how a yes to "shall I look that up?" would come to
# license a booking. So it is put as evidence and the reading is left open.
ANSWERED = (
    "The assistant asked the customer this, and they replied.\n"
    "  Asked: {reason}\n"
    "  Answered on turn {turn}: {words!r}\n"
    "  Both quoted verbatim. What that answer covers is yours to decide: compare "
    "what was described against what is now proposed."
)

NOTHING_OWED = "Nothing. Every change this turn was planned to make has been approved."

# Written as a reminder of the gate's own words rather than as a summary of them,
# because the summary is the thing it gets wrong: asked "has the customer
# confirmed?" it re-reads the transcript and answers no. Asked "you required X and
# they have replied twice since" it has the question and the answer side by side.
DEMAND = (
    "You refused {action} earlier, requiring: {reason}\n"
    "  The customer has replied {replies} since. If what they said answers what you "
    "required, that condition is now met -- do not require it a second time."
)

REPEATED = "- {call}: `{paths}` are the same entry twice."

MISPRICED = (
    "- {call}: the figure at `{path}` was never shown beside the identifier it is "
    "attached to here. Check it is that one's and not the neighbouring row's."
)

FROM = "- {path} = {value!r}\n    taken from: {snippet}"

NOWHERE = "- {path} = {value!r}\n    taken from: nothing in the conversation."

# The half the snippet cannot show. `sources` quotes the most recent text holding
# the value, which for a reservation id is a slice of the record a lookup
# returned -- true, and silent on the only question that matters, which is
# whether this is the record the customer meant. Both surplus cancellations that
# lost task 41 and the one that lost task 1 are on identifiers the customer never
# typed; every one of them was quoted from a lookup and approved.
NAMED = "\n    the customer named this themselves: {snippet}"

UNNAMED = (
    "\n    the customer has never mentioned this value. It comes only from a "
    "lookup, so nothing here says it is the record they asked about."
)


# Malformed answers from the model, inside one `run_sync`. A 20B model sometimes
# answers in prose instead of filling the output tool in, and pydantic-ai's
# default budget of one retry makes a single fumble raise.
OUTPUT_RETRIES = 3

# Whole `run_sync` calls, when the one above is exhausted or never gets going.
# The two runs on the 15-task set raised on 48% of the verdicts asked for, and
# most of those were not the model at all: NVIDIA NIM intermittently returns a
# completion with a null `id`, pydantic-ai rejects it as malformed, and one bad
# packet became a blocked write. That failure arrives on the first call and costs
# nothing to retry, which is the whole argument for this constant.
ATTEMPTS = 3

# The verdict used when no verdict was reached. Refusing is not a judgement about
# the action -- it is the only honest thing to say about an action nobody checked,
# and it is recoverable: the actor proposes again, the gate is asked again, and
# the revision cap ends the turn by talking to the customer if it never answers.
UNAVAILABLE = Verdict(
    allowed=False,
    reason="The policy check did not complete, so this action was never authorised.",
    remediation="Propose the same action again.",
)

# The same question with the answer collapsed to one bit, asked when the
# structured verdict never arrives. The failure it answers is not judgment: 38 of
# the 166 write refusals in the 50-task run were this path, a 20B model unable to
# fill in a four-field object inside its retry budget, and every one of them
# blocked an action nobody had ruled on. One boolean is the smallest thing that
# can still be an answer, and it is the same collapse that made the gate
# answerable when it stopped being a two-tool union.
NARROW = (
    "Answer with true or false and nothing else. Does the policy permit the "
    "proposed action, exactly as written, right now? Answer true if it does. "
    "Answer false if it does not."
)

NARROWED_ALLOW = Verdict(
    allowed=True,
    reason="The full policy check did not return, and the narrowed check permitted this action.",
)

NARROWED_BLOCK = Verdict(
    allowed=False,
    reason="The full policy check did not return, and the narrowed check refused this action.",
    remediation=(
        "Tell the customer what you are trying to do and what you need from them "
        "to do it. Do not repeat this action unchanged."
    ),
)

# What a refusal that named no fix is turned into. The union output type used to
# make this unrepresentable; with one output type the model can leave the field
# empty, so the repair happens here instead. Ending the turn is the right default:
# a refusal nobody can act on spends a revision and leaves the actor where it was.
FALLBACK = "Do not repeat this action. Tell the customer what you need from them."


def build_gate(policy: str, model: str | Model | None = None) -> Agent[None, Verdict]:
    """A gate bound to one domain's policy. The policy is static, the case is not."""
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS.format(policy=policy, workflows=for_policy(policy)),
        output_type=Verdict,
        retries={"output": OUTPUT_RETRIES},
    )


def decide(gate: Agent[None, Verdict], case: str, attempts: int = ATTEMPTS) -> Verdict:
    """Get a verdict, or refuse. Never raises.

    A raise here used to mean one thing and now means another. It used to be the
    model failing to answer, which is a judgement of sorts. Measurement showed it
    is mostly the provider handing back a response the client will not parse --
    an action refused because of a bad packet, with the customer told to try
    again. So the call is retried before the refusal stands.

    A block that named no fix is repaired rather than passed on: `remediation` is
    handed to the actor verbatim and is all it gets, and an empty one costs it a
    revision to read nothing.
    """
    for remaining in range(attempts, 0, -1):
        try:
            verdict = gate.run_sync(case).output
        except UnexpectedModelBehavior:
            if remaining == 1:
                return _narrowed(gate, case)
            continue
        if not verdict.allowed and not verdict.remediation.strip():
            return verdict.model_copy(update={"remediation": FALLBACK})
        return verdict
    return UNAVAILABLE


def _narrowed(gate: Agent[None, Verdict], case: str) -> Verdict:
    """The last ask before refusing unread: the same case, one bit of answer.

    Reached only when the four-field verdict never parsed, which measurement says
    is a shape problem and not a judgement -- so asking the same question in a
    shape that cannot be got wrong is worth one more call. `output_type` is
    overridden per run rather than by building a second agent, which keeps the
    instructions, the policy and the retry budget identical to the check that
    just failed. Only if this fails too does the action stand refused unread.
    """
    try:
        allowed = gate.run_sync(f"{case}\n\n{NARROW}", output_type=bool).output
    except UnexpectedModelBehavior:
        return UNAVAILABLE
    return NARROWED_ALLOW if allowed else NARROWED_BLOCK


def review(
    messages: list[ModelMessage],
    proposal: list[PendingCall],
    observed: list[str],
    demanded: list[Demand] | None = None,
    turn: int = 0,
    consented: list[Consent] | None = None,
    owed: list[Change] | None = None,
    request: str = "",
    opened: str = "",
) -> str:
    """The case put to the gate: what happened, what is proposed, what looks off,
    where each identifier came from, what this gate has already required, and what
    the turn has left to do.

    `owed` is the same ledger the speaker counts against, and it is here because
    the run showed both guards being right and the turn being lost between them:
    the plan said cancel one reservation and refuse the other, the speaker held the
    reply and named the call to make, the actor proposed a handoff instead, and
    this node approved it with no idea a confirmed cancellation was outstanding. A
    ledger only one guard can see does not cover the exit the other one guards.

    `request` is the planner's own statement of what the customer wants, at the
    scope they asked for it, and this node was ruling without it. Task 41's said
    "cancel all of my upcoming flights that only have me on the reservation" while
    this gate approved three cancellations on reservations carrying two and three
    passengers -- correctly, on the only authority it had, because the policy
    permits cancelling a business-cabin booking and nothing here was asking the
    other question.
    """
    return REVIEW.format(
        request=asked_for(request, opened),
        transcript=transcript(messages) or "(nothing yet)",
        proposal="\n".join(f"{c.name}({_arguments(c.arguments)})" for c in proposal),
        findings=findings(proposal, observed),
        provenance=provenance(proposal, observed, messages),
        demands=demands(proposal, demanded or [], turn),
        consents=consents(proposal, consented or []),
        owed="\n".join(f"- {c.key}: {c.what}".rstrip(": ") for c in owed) if owed else NOTHING_OWED,
    )


def asked_for(request: str, opened: str) -> str:
    """What the customer wants, and what they wanted to begin with.

    Both, when they have come apart, and without saying which is right. The
    planner may only rewrite this when the customer has spoken, so a difference
    here is a real turn of the conversation -- and it is either the customer
    changing their mind or the assistant's own framing having replaced theirs.
    Task 7 is the first and task 41 is the second, they are indistinguishable
    from the strings alone, and the gate has the transcript to tell them apart.
    """
    now = " ".join(request.split())
    first = " ".join(opened.split())
    if not now and not first:
        return NO_REQUEST
    if not first or first == now:
        return now or first
    return f"{now}\n  When they first asked, it was: {first}"


def findings(proposal: list[PendingCall], observed: list[str]) -> str:
    """PRE-GATE: the deterministic pass, reported as evidence rather than a verdict.

    Three checks, all free and none a judgement: values the assistant was never
    shown, entries it wrote down twice -- a passenger repeated on a booking, a leg
    of an itinerary standing in for its own return -- and money quoted against an
    identifier it was never shown beside.

    The third is the one the arithmetic did not explain. `calculate` was right
    every time it was used across the diagnostic run; what went wrong was the
    numbers going into it, and a booking was lost to twice the economy fare of a
    flight the customer was not taking. Neither of the other two checks can see
    that: the figure was shown, the entry is not a copy, and only the pairing is
    wrong.
    """
    lines = [
        f"- {call.name}: the value given for `{name}` appears nowhere in what the "
        f"assistant has been shown."
        for call in proposal
        for name in ungrounded(call.arguments, observed)
    ]
    lines += [
        REPEATED.format(call=call.name, paths=path)
        for call in proposal
        for path in duplicated(call.arguments)
    ]
    lines += [
        MISPRICED.format(call=call.name, path=path)
        for call in proposal
        for path in mispriced(call.arguments, observed)
    ]
    return "\n".join([*lines, "", CAVEAT]) if lines else NO_FINDINGS


def provenance(
    proposal: list[PendingCall],
    observed: list[str],
    messages: list[ModelMessage] | None = None,
) -> str:
    """Every identifier in the proposal, quoted with the text it was taken from.

    Evidence, in the same spirit as `findings`, for a failure that one cannot see:
    an identifier is checked for having been shown, never for having been shown
    *about the thing being changed*. A reservation the customer owns but never
    mentioned passes untouched, and two tasks were lost that way -- one modifying
    the wrong booking, one paying with the wrong gift card. Quoting the
    surrounding text puts the question in front of something that can answer it,
    and costs no model call and no knowledge of the domain.
    """
    spoken = _spoken(messages or [])
    lines = [
        (FROM if snippet else NOWHERE).format(
            path=f"{call.name}.{path}", value=value, snippet=snippet
        )
        + _by_the_customer(value, spoken)
        for call in proposal
        for path, value, snippet in sources(call.arguments, observed)
    ]
    return "\n".join(lines) if lines else NO_PROVENANCE


def _spoken(messages: list[ModelMessage]) -> list[str]:
    """Only what the customer typed. Not the lookups, not the assistant."""
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


def _by_the_customer(value: str, spoken: list[str]) -> str:
    """Whether the customer ever said this value, and where.

    The non-drifting half of the pair. `request` is the planner's restatement and
    can be re-derived wrongly; this is what the customer actually typed, and no
    later turn can rewrite it.
    """
    snippet = quoted(value, spoken)
    return NAMED.format(snippet=snippet) if snippet else UNNAMED


def demands(proposal: list[PendingCall], demanded: list[Demand], turn: int) -> str:
    """What this gate has already required of these same actions, and since when.

    Only demands the customer has had a chance to answer are shown: one made on
    the turn still in progress is the gate arguing with itself.
    """
    proposed = {call.name for call in proposal}
    replied = {1: "once", 2: "twice"}
    lines = [
        DEMAND.format(
            action=demand.action,
            reason=" ".join(demand.reason.split()),
            replies=replied.get(turn - demand.turn, f"{turn - demand.turn} times"),
        )
        for demand in demanded
        if demand.action in proposed and turn > demand.turn
    ]
    return "\n".join(f"- {line}" for line in lines) if lines else NO_DEMANDS


def consents(proposal: list[PendingCall], consented: list[Consent]) -> str:
    """What the customer has already agreed to, for the actions being proposed.

    The other half of `demands`. That one stops the gate forgetting it asked; this
    one stops it forgetting it was answered -- 70 of the 166 write refusals in the
    50-task run were a condition re-imposed after the customer had met it, and the
    gate had no way to know except by re-reading a transcript it reads badly.

    Filtered to the proposal for the same reason `demands` is: an agreement about
    a different action is not evidence about this one, and a list of everything
    the customer has ever said yes to is a list the gate will misapply.
    """
    proposed = {call.name for call in proposal}
    lines = [
        (ANSWERED if consent.action == ANY else CONSENT).format(
            action=consent.action,
            reason=" ".join(consent.reason.split()),
            turn=consent.turn,
            words=" ".join(consent.words.split()),
        )
        for consent in consented
        if consent.action in proposed or consent.action == ANY
    ]
    return "\n".join(f"- {line}" for line in lines) if lines else NO_CONSENTS


def transcript(messages: list[ModelMessage]) -> str:
    """The dialogue as prose, including the assistant's own tool use.

    The gate is checking whether a prerequisite was met, and prerequisites are
    usually met by a lookup, so the reads have to be visible -- a transcript of
    the customer-facing turns alone would hide the evidence being judged.
    """
    lines: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                lines.append(f"Customer: {part.content}")
            elif isinstance(part, TextPart):
                lines.append(f"Assistant: {part.content}")
            elif isinstance(part, ToolCallPart):
                call = f"{part.tool_name}({_arguments(part.args_as_dict())})"
                lines.append(f"Assistant looks up: {call}")
            elif isinstance(part, ToolReturnPart):
                lines.append(f"Result: {part.content}")
            elif isinstance(part, RetryPromptPart):
                lines.append(f"Gate: {part.content}")
    return "\n".join(lines)


def _arguments(arguments: dict) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in arguments.items())
