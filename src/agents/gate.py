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

A block is also the only thing the actor ever hears from this node. `remediation`
is handed to it verbatim as a retry prompt and is the whole of what it has to work
from, which makes an unactionable refusal worse than no refusal at all: it spends
one of two revisions and leaves the actor exactly where it was. So the
instructions ask for a refusal written as an instruction, not as a complaint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent
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
from core.state import PendingCall, ungrounded

__all__ = [
    "UNAVAILABLE",
    "Approved",
    "Blocked",
    "Verdict",
    "build_gate",
    "findings",
    "review",
    "transcript",
]


class Approved(BaseModel):
    """The action is permitted. It will be executed exactly as proposed."""

    reason: str = Field(description="One sentence: why the policy allows this now.")


class Blocked(BaseModel):
    """The action is not permitted, and will not be executed."""

    violation: str = Field(description="The policy rule this breaks, in one sentence.")
    remediation: str = Field(
        description=(
            "What to do instead, written as a direct instruction to the assistant "
            "that it can act on immediately."
        )
    )


Verdict = Approved | Blocked
"""Two output types rather than one with a flag, so the shape carries the rule:
there is no way to express a block without a fix, or a pass with a complaint."""


INSTRUCTIONS = """
You are the policy gate. An assistant is serving a customer and has proposed an
action it cannot take back: a change to the company's records, or a handoff to a
human, which ends the conversation. Nothing has happened yet. Your answer decides
whether it happens at all, and you are the only check there is.

WHAT YOU ARE GIVEN

CONVERSATION SO FAR is everything that has happened, including the assistant's
own lookups and what they returned. This is your evidence -- if the assistant
checked something, the check and its result are in there.

PROPOSED ACTION is the exact call or calls about to be made, with their real
arguments. Judge those arguments, not what the assistant said it was going to do.

AUTOMATED CHECKS lists values in the proposal that appear nowhere in the
conversation. They are leads to follow, not verdicts.

The proposal is a single step and you judge it as one: if any part of it is not
allowed, the whole step is refused, so account for all of it.

APPROVE UNLESS THE POLICY FORBIDS IT

The policy below is your only authority. If it does not prohibit this action,
approve it. Do not invent requirements, do not apply general caution, and do not
block because you would have gone about it differently or in a different order.
Blocking a permitted action costs exactly what allowing a forbidden one costs --
either way the customer is not helped.

WHAT TO BLOCK

- The policy states a condition and nothing in the conversation shows it was met.
- The policy requires the customer to agree and they were never asked, or were
  never told the part they would be agreeing to, such as a price or a penalty.
- The customer is not entitled to what is about to be given to them.
- An argument's value was never established: not returned by any tool, not given
  by the customer, not derived from either.
- A figure was worked out in the assistant's head and it is wrong.
- The action does not do what the customer actually asked for.

HOW TO REFUSE

Your refusal is handed to the assistant word for word and is all it gets. It will
read it and try again, so it has to be usable.

`violation` names the rule and the fact that breaks it, together. Not "the
customer did not confirm" but "the policy requires the customer to confirm the
fare difference before a change is made, and they have not been told what it is."

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

HOW TO APPROVE

An approval is not a shrug. Give the reason in one sentence that names what makes
this allowed -- the rule that permits it, and where in the conversation the
condition it depends on was satisfied.

<policy>
{policy}
</policy>
""".strip()


REVIEW = """
CONVERSATION SO FAR
{transcript}

PROPOSED ACTION
{proposal}

AUTOMATED CHECKS
{findings}
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


# A union output type gives the model two output tools to choose between, and a
# 20B model sometimes answers in prose instead of choosing. pydantic-ai's default
# budget of one retry makes a single fumble raise, and a raise inside the gate
# takes the whole simulation down: Run 002 lost one that way, and replaying real
# writes through the gate offline reproduced it in 3 cases out of 8. Retries are
# cheap; the failure they prevent costs a task outright.
OUTPUT_RETRIES = 3

# The verdict used when no verdict was reached. Refusing is not a judgement about
# the action -- it is the only honest thing to say about an action nobody checked,
# and it is recoverable: the actor proposes again, the gate is asked again, and
# the revision cap ends the turn by talking to the customer if it never answers.
UNAVAILABLE = Blocked(
    violation="The policy check did not complete, so this action was never authorised.",
    remediation="Propose the same action again.",
)


def build_gate(policy: str, model: str | Model | None = None) -> Agent[None, Verdict]:
    """A gate bound to one domain's policy. The policy is static, the case is not."""
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS.format(policy=policy),
        output_type=[Approved, Blocked],
        retries={"output": OUTPUT_RETRIES},
    )


def review(messages: list[ModelMessage], proposal: list[PendingCall], observed: list[str]) -> str:
    """The case put to the gate: what happened, what is proposed, what looks off."""
    return REVIEW.format(
        transcript=transcript(messages) or "(nothing yet)",
        proposal="\n".join(f"{c.name}({_arguments(c.arguments)})" for c in proposal),
        findings=findings(proposal, observed),
    )


def findings(proposal: list[PendingCall], observed: list[str]) -> str:
    """PRE-GATE: the deterministic pass, reported as evidence rather than a verdict."""
    lines = [
        f"- {call.name}: the value given for `{name}` appears nowhere in what the "
        f"assistant has been shown."
        for call in proposal
        for name in ungrounded(call.arguments, observed)
    ]
    return "\n".join([*lines, "", CAVEAT]) if lines else NO_FINDINGS


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
