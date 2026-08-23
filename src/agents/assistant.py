"""The one sub-agent there is so far: a plain assistant that follows the policy.

Environment tools are declared to it as *external* tools -- pydantic-ai's term
for a tool the model may call but someone else executes. That is exactly the
benchmark's contract, so the agent never holds a callable it could fire by
accident. `toolset` makes the declared schema binding rather than decorative;
see there for why that is not the default.

The agent's dependency is `Deps`: the provenance ledger, which is how the toolset
knows what the system has actually been shown, and the planner's route for this
turn. Both are passed per run rather than bound at build time, because both grow
with the conversation and the agent is built once.

The plan reaches the model as a second, *dynamic* instructions block rather than
as part of the prompt. That is not a style choice. A plan prepended to the user
message would be indistinguishable from something the customer said -- `transcript`
renders every `UserPromptPart` as "Customer:" -- and the gate would then be
judging the actor against words it put in the customer's mouth.

The policy arrives in two pieces. What holds on every turn -- the preamble, and
the section defining what a reservation and a cabin class are -- is built into
the instructions. The procedure sections are not: the planner names the ones the
turn is about and only those are shown, as a third instructions block. `core.policy`
says why, and the short version is that the actor rebuilds its instructions on
every request of a turn while the planner pays once.

The instructions below are not general advice. Every section of them answers a
failure counted in the 50-task diagnostic run: invented identifiers (41% of all
environment errors), arithmetic done in the model's head (31%, with the
calculator tool called once in the entire run), required fields left out (21%),
and -- the largest bucket of lost tasks -- bailing out to a human rather than
doing the work. A tau2 domain policy describes the business; it says nothing
about how to operate, so what a policy leaves unsaid is what has to be said here.
"""

from __future__ import annotations

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition

import llm
from agents.toolset import declared
from core.policy import contents, standing
from core.state import Deps

__all__ = [
    "Assistant",
    "build_assistant",
    "correction_for_this_turn",
    "plan_for_this_turn",
    "rules_for_this_turn",
]

Assistant = Agent[Deps, "str | DeferredToolRequests"]
"""The actor. Deps are the provenance ledger and this turn's plan."""

INSTRUCTIONS = """
You are a customer service agent for the company whose policy appears at the end
of these instructions. You are speaking to a real customer and acting on the
company's real records. Your job is to resolve what they came for, in full, using
the tools you have been given.

The policy is your only authority. It says what you may do, what must be true
before you do it, and what you must refuse. When the policy does not allow
something, say so plainly and offer what you can instead.

HOW A TURN WORKS

Each turn you do exactly one of two things: send a message to the customer, or
make tool calls. Never both in one turn, and never neither. Tool calls do not end
the conversation, they pause it -- the results come back to you and you carry on
from there. So take as many tool-calling turns as the work needs before you next
speak.

WHAT YOU DO, IN ORDER

1. Establish who you are talking to and what they actually have. Look it up. Act
   on the record, never on the customer's description of the record.
2. Check the policy for the action they want: is it permitted, and what has to be
   true first?
3. Fill in what you are missing -- from a tool where the system knows it, from
   the customer where only they know it.
4. Tell the customer exactly what you are about to do and what it will cost them,
   and wait for them to agree.
5. Do it, then confirm what was done.

CALLING A TOOL

Every value in a tool call has to have come from somewhere: a tool result, the
customer's own words, or the policy. Identifiers -- reservation ids, order ids,
payment ids, flight numbers -- can only come from a tool result or the customer.
There is no way to work one out and no such thing as a typical one. If you do not
have it, look it up.

Send every required field, filled in. A partial object is rejected outright, not
completed for you: if a tool asks for a passenger it wants that passenger's first
name, last name and date of birth, and you need to have read all three somewhere
before you call it.

Do not do arithmetic yourself. Where you have a calculator tool, use it for every
total, difference and fee, including the ones that look easy.

Worked example. It is from a different company than yours -- follow its shape,
not its tool names, and never reuse its values, which belong to nobody. A
customer wants to add a checked bag. You do not know their booking reference, so
you start by looking them up:

    get_user_details(user_id="sara_doe_496")

That lists their bookings, so now you can read the one they mean:

    get_reservation_details(reservation_id="HKD3PS")

That shows two passengers, one bag already paid for, and the cards on file. The
policy prices an extra bag at 50 dollars, so you work the charge out with the
tool rather than in your head:

    calculate(expression="50 * 1")

You now have everything the change needs, and once the customer has agreed to the
50 dollars you make it:

    update_reservation_baggages(reservation_id="HKD3PS", total_baggages=2,
                                nonfree_baggages=1,
                                payment_id="credit_card_7815826")

Every one of those values was read out of an earlier result. None was assumed.

WHEN SOMETHING COMES BACK WRONG

A rejected call is information, not a dead end: read what it says, fix the thing
it names, and call it again. Anything that changes the records is also reviewed
before it runs, and a refusal tells you which rule it breaks and what to do
instead. Do that, and propose again.

TRANSFERRING TO A HUMAN

A transfer ends the conversation and leaves everything unfinished. Do it only
where the policy tells you to, or where the customer asks for a person. Not
because the request is awkward, not because the answer is no -- a refusal is
yours to give -- and not because you are unsure. Unsure means look it up or ask.

<policy>
{standing}
</policy>

The policy has a further section for each thing you can do:

{contents}

The ones this turn is about are reproduced below, under RULES FOR THIS TURN. The
others still hold: everything you propose to change is reviewed against the whole
policy before it runs, so a rule you were not shown can still stop you. If what
the customer wants turns out to belong to a section you have not been given, say
what you can establish and ask them for what you need -- do not guess at the rule
and do not transfer.
""".strip()


RULES = """
RULES FOR THIS TURN
{sections}
""".strip()


def plan_for_this_turn(ctx: RunContext[Deps]) -> str:
    """The planner's route, as instructions rather than as a message.

    Returns nothing at all when there is no plan -- an empty run, a turn the
    planner could not answer for -- so the actor is never shown an empty heading
    to read meaning into.
    """
    return ctx.deps.plan if ctx.deps else ""


def correction_for_this_turn(ctx: RunContext[Deps]) -> str:
    """A message that was held, as the instruction to fix it.

    Last of the instruction blocks, so it is the final thing the actor reads
    before it answers. Empty on every run but the one after a hold, and empty is
    the whole of its cost the rest of the time.
    """
    return ctx.deps.correction if ctx.deps else ""


def rules_for_this_turn(ctx: RunContext[Deps]) -> str:
    """The policy sections the planner routed this turn to.

    Dynamic for the same reason the plan is: it differs per run, and pydantic-ai
    re-evaluates the callable on every request, so the rules stay in front of the
    actor across a whole turn of tool calls rather than only the first.

    Returns nothing when the turn was routed nowhere. That is the escalation run,
    which has had its tools taken away and exists only to end the turn by talking
    to the customer -- a procedure for making a change is the one thing it must
    not be handed.
    """
    return RULES.format(sections=ctx.deps.policy) if ctx.deps and ctx.deps.policy else ""


def build_assistant(
    tools: list[ToolDefinition],
    policy: str,
    model: str | Model | None = None,
    reference: str = "",
) -> Assistant:
    """An agent whose run ends with either a reply or the tool calls it wants made.

    `model` is a model id, or a ready-made model when the caller has one -- which
    is how tests hand it a scripted stand-in instead of a live endpoint.

    `policy` is the whole domain policy. What is built in here is the part that
    holds whatever the turn is about, plus the list of section names; the sections
    themselves arrive per run on `Deps`.

    `reference` is what the domain assumes and never states -- for the airline,
    which cities the airport codes belong to. It is a separate element rather than
    a slot inside the instructions so that a domain with nothing to say adds
    nothing at all, not a blank line where a heading would go. It is static: it
    describes the environment, which does not change between turns.

    Instructions are a sequence: the standing ones, then this turn's rules, then
    the plan. The last two are functions because they differ per run, and
    pydantic-ai re-evaluates a callable on every request of a run, so both stay in
    front of the actor across a whole turn of tool calls rather than only on the
    first one. Rules before plan, because the plan is a route through them.
    """
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=[
            INSTRUCTIONS.format(standing=standing(policy), contents=contents(policy)),
            *([reference] if reference else []),
            rules_for_this_turn,
            plan_for_this_turn,
            correction_for_this_turn,
        ],
        deps_type=Deps,
        toolsets=[declared(tools)],
        output_type=[str, DeferredToolRequests],
    )
