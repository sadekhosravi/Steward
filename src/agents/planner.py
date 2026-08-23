"""PLANNER: the route from the request to it being done, written before it starts.

The actor can read a policy. What it loses hold of is the shape of the whole job
while it is busy composing one tool call, and the 50-task diagnostic run shows
the cost twice over: 57% of the gold read actions were matched against the
unscaffolded baseline's 66%, and of 25 tasks that required a write, exactly two
were won. Only two simulations in the whole run performed every write their task
needed. Those are not failures of policy comprehension -- they are a plan being
held in the same breath as an argument list, and losing.

So this node runs once at the top of a user turn and writes the plan down, and
the actor is handed it as something to check itself against. Nothing here is
enforced: a plan is advice, and the gate is the thing with authority.

The output shape is the read/write split the environment already scores on --
find out, agree, change -- because that ordering is the one the actor breaks. It
is four flat fields, three of them lists of plain strings, deliberately: the
gate's own notes record a 20B model answering a two-member union in prose often
enough to need three output retries, and nested models fail the same way.

It also decides which parts of the policy the actor gets to see. That is not a
second job bolted on: choosing the route and choosing the rules the route runs
under is one act, and this node is already the only one that reads the whole
policy once per turn rather than once per tool call. `core.policy` has the
reasoning. What matters here is that the field is a list of headings copied from
a list, not a judgment -- and that naming too few is the way it hurts, so the
instructions push the same direction they push on `lookups`.

It is also shown the policy twice: once as the document, and once as `workflows`
-- the same rules as branches, each carrying the line it was copied from. That is
not redundancy for its own sake. This node was not reasoning freely about policy
and getting it wrong; it was running a procedure it already held, and the
procedure was wrong. Of 207 plans in the last full run, 59 had a refusal for a
goal, and 23 of those were written before the reservation that decides the
question had ever been read. `workflows` names the fact each branch turns on, so
a refusal has somewhere to come from other than memory.

The one thing this node must not do is write down an identifier. It plans before
the lookups have run, so it holds none, and any it produced would be invented and
copied forward by the actor into exactly the tool call the provenance ledger
exists to catch. The instructions spend a paragraph on that for a reason.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition

import llm

# The conversation reads the same way for a planner as for a critic, and one
# rendering shared beats two that drift apart. It lives in `gate` because that is
# where it was first needed, not because it belongs to the gate.
from agents.gate import transcript
from core.policy import contents
from workflows import for_policy

__all__ = ["OUTPUT_RETRIES", "Plan", "brief", "build_planner", "catalogue", "render"]


class Plan(BaseModel):
    """The route from what the customer is asking for to it being done."""

    goal: str = Field(
        description=(
            "What the customer wants, in one sentence, as the records would have to "
            "end up for it to be true. If the policy does not permit it, say that "
            "here instead."
        )
    )
    lookups: list[str] = Field(
        default_factory=list,
        description=(
            "What has to be found out before anything is decided, one line each, "
            "naming the tool that answers it. A price, a fee or a difference is "
            "something to find out like any other fact -- name the tool that returns "
            "it. Refer to identifiers by where they will come from, never by a value."
        ),
    )
    confirm: str | None = Field(
        default=None,
        description=(
            "What the customer must be told and must agree to before the records are "
            "touched -- the price, the difference, the penalty. Name the figure and "
            "where it comes from: which tool returns the numbers, and what is taken "
            "from what. Both, never the figure alone -- a number with nowhere to come "
            "from is one the assistant ends up asking the customer for. Null when "
            "nothing will be changed."
        ),
    )
    changes: list[str] = Field(
        default_factory=list,
        description=(
            "Every write the request needs, in the order they must happen, one line "
            "each, naming the tool. Empty when the request needs none."
        ),
    )
    policy_sections: list[str] = Field(
        default_factory=list,
        description=(
            "Which sections of the policy this turn is governed by, copied exactly "
            "from the list of section names in your instructions. Name every area "
            "the request touches, not just the main one. Leave empty only if you "
            "cannot tell."
        ),
    )


INSTRUCTIONS = """
You plan the work. Another agent does it.

A customer service assistant is part-way through a conversation. You are given the
company's policy, the tools it can use, and everything that has happened so far.
You write the route from where things stand now to the customer's request being
done. You never speak to the customer and you never call a tool: what you produce
is read by the assistant, which does both.

WHAT MAKES A PLAN WORTH HAVING

The assistant can already read the policy. What it loses hold of is the shape of
the whole job while it is busy writing one tool call, so it looks up too little
and stops part-way through a change that needed three steps. Your plan is what it
checks itself against. Write down the steps it would otherwise forget.

FINDING OUT

`lookups` is what has to be known before anything is decided: one line each,
naming the tool that answers it. Reads cost nothing and are never held against
the assistant, so the mistake to avoid is listing too few, not too many. Anything
the request depends on -- who the customer is, what they hold, what it currently
costs, what is still available -- is a lookup, and stays one even when the
customer has already told you. The record is the authority, not their description
of it.

A price is a lookup. What a reservation cost is on the reservation; what it would
cost now is in the flight search, which reports a price for every cabin on every
flight it returns. So the difference between two cabins, or between two
itineraries, is two reads and a subtraction, and it is never something to ask the
customer for -- they do not know it, and the one number they must agree to is the
one thing you cannot take their word for.

You do not have any identifiers and you must not write one down. You are planning
before the lookups have run, so no reservation id, order id, payment id or flight
number is known to you yet. Name one by where it will come from -- "the
reservation id from get_user_details" -- never by a value. Any value you write
here is one you made up, and the assistant may copy it into a real call.

AGREEING

`confirm` is what the customer has to be told, and has to accept, before anything
is written: the price, the fare difference, the cancellation penalty, the fact
that it cannot be undone. Say which figure has to be quoted and where it comes
from -- the tool that returns the numbers and the subtraction that turns them into
the amount. Naming the figure alone leaves the assistant owing the customer a
number it has no way to reach, and asking them for it is how the turn ends without
the change being made. If the plan changes nothing, `confirm` is null.

CHANGING

`changes` is every write the request needs, in the order they have to happen, one
line each, naming the tool. All of them. A change left half-done is worse than one
never started, because the records end up in a state nobody asked for. If the
request needs no change at all -- a question, a lookup, something the policy does
not allow -- leave it empty.

WHEN THE ANSWER IS NO

If the policy does not permit what is being asked, that is a finished plan and not
a failure. Say so in `goal`, list the lookups that establish it, and leave
`changes` empty. Do not plan a handoff to a human: a refusal is the assistant's
own to give, and a handoff ends the conversation with everything else still
undone.

A refusal is a finding, and a finding needs the fact it rests on. The workflow for
the request names what decides it -- the cabin, the purchase time, whether a
segment has flown, whether there is insurance -- and every one of those is on a
record you have not read yet. So the plan that establishes a refusal is a plan
with `lookups` in it. If you have not got the fact, the honest plan is to go and
get it, and the answer is no once it comes back and not before.

{workflows}

WHICH RULES APPLY

`policy_sections` is the other half of your job. The assistant is not shown the
whole policy -- it is shown the rules that hold on every turn, plus the sections
you name here. A section you leave out is one it will be working without, so name
every area the request touches: a cancellation that ends in money going back
needs the cancelling rules and the refund rules both, and a customer who asks
what a change would cost is asking about modifying whether or not they go through
with it. Reading a section that turned out not to matter costs nothing.

A policy states a fee once, in the section for the action that first incurs it,
and it goes on applying everywhere else. So when a turn involves money, name the
section that sets the amount as well as the section that permits the action: what
a bag costs is written where bags are first bought, not where they are later added.

Copy the names exactly as they appear here:

<sections>
{contents}
</sections>

If you genuinely cannot tell, leave it empty and the assistant is shown all of
them -- which is safe, and worse than choosing, so do not use it to avoid the
question.

Keep every line short enough to act on. You will be asked again as the
conversation goes on, so plan from what is true now, not from what you are hoping
for.

<tools>
{tools}
</tools>

<policy>
{policy}
</policy>
""".strip()


BRIEF = """
CONVERSATION SO FAR
{transcript}

WHAT THE CUSTOMER HAS JUST ASKED FOR
{request}
""".strip()


PLAN = """
PLAN FOR THIS TURN
{body}

This is a plan, not a script. If a lookup shows something it did not assume,
follow what you found.
""".strip()


# A single output type, so there is nothing for the model to choose between, but
# the retries stay: the failure the gate documented was answering in prose rather
# than calling the output tool at all, and one output type does not rule that out.
OUTPUT_RETRIES = 3


def build_planner(
    tools: list[ToolDefinition], policy: str, model: str | Model | None = None
) -> Agent[None, Plan]:
    """A planner bound to one domain. It is told the tools; it is given none.

    Declaring them would let it emit a call, and a call from here would reach the
    environment before anything had reviewed it. It needs to know what exists, not
    to be able to reach it.

    It is given the policy whole, and unlike the actor it keeps it: this runs once
    per user turn, so the saving would be one copy against the actor's twenty, and
    a router that cannot see what it is choosing between is not a router.
    """
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS.format(
            tools=catalogue(tools),
            policy=policy,
            contents=contents(policy),
            workflows=for_policy(policy),
        ),
        output_type=Plan,
        retries={"output": OUTPUT_RETRIES},
    )


def catalogue(tools: list[ToolDefinition]) -> str:
    """The tools, each marked read or write.

    The mark comes from the same `gated` label the graph routes on, so the planner
    and the critic agree about what is irreversible. A planner that cannot tell a
    lookup from a write cannot be asked to keep them in separate fields.
    """
    return "\n".join(f"- {tool.name} ({_kind(tool)}): {_description(tool)}" for tool in tools)


def brief(messages: list[ModelMessage], request: str | None) -> str:
    """The case put to the planner: what has happened, and what was just asked."""
    return BRIEF.format(
        transcript=transcript(messages) or "(nothing yet)",
        request=request or "(nothing new -- continue from the conversation above)",
    )


def render(plan: Plan) -> str:
    """The plan as the actor is shown it.

    Empty sections are left out rather than printed empty. A heading with nothing
    under it reads as an instruction to find something to put there, and for a task
    that genuinely needs no write that is the wrong nudge entirely.
    """
    body = [f"Goal: {plan.goal}"]
    if plan.lookups:
        body += ["", "Find out first:", *_numbered(plan.lookups)]
    if plan.confirm:
        body += ["", f"Confirm before changing anything: {plan.confirm}"]
    if plan.changes:
        body += ["", "Then change, in this order:", *_numbered(plan.changes)]
    return PLAN.format(body="\n".join(body))


def _numbered(lines: list[str]) -> list[str]:
    return [f"  {index}. {line}" for index, line in enumerate(lines, start=1)]


def _kind(tool: ToolDefinition) -> str:
    return "write" if (tool.metadata or {}).get("gated", True) else "read"


def _description(tool: ToolDefinition) -> str:
    """One entry of the catalogue, kept to a single line per tool.

    The adapter widens a tool's description with the fields it returns, which is
    what lets the catalogue answer "which tool gives me the payment id" -- but it
    arrives over several lines, and several lines each turns a list of fourteen
    tools into a wall.
    """
    return " ".join((tool.description or "").split())
