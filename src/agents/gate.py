"""GATE: the critic that stands between a proposed write and the environment.

Reads are free. Exploring the database costs nothing and is never scored, so a
wrong read wastes a step and nothing more. A wrong *write* is fatal: the DB
component compares the final database against a gold replay, one bad mutation
loses the task outright, and no amount of good conversation afterwards recovers
it. The baseline measured exactly that shape -- 89% on communication, 39% on the
database, 5.6% recall on write actions -- which is why this node exists and why
it fires on writes only.

Its authority is the policy and nothing else. A critic that blocks whenever it
feels unsure is worse than no critic: it turns tasks the actor would have solved
into tasks nobody solves, and the loss lands on the same metric it was meant to
protect. So the instructions are deliberately biased toward approval, and a
block has to name the rule it is enforcing.
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
You are a policy gate. An assistant is serving a customer and wants to perform an
action that changes the company's records. Nothing has happened yet -- your answer
decides whether it happens at all.

Approve unless the policy forbids it. The policy below is your only authority. If
it does not prohibit this action, approve it. Do not invent requirements, do not
apply general caution, and do not block because you would have gone about it
differently. Blocking a permitted action costs exactly as much as allowing a
forbidden one.

Block when the policy states a condition that has not been met: a check the
assistant skipped, a confirmation the customer never gave, an option the customer
is not entitled to, or a value the assistant does not actually have. Say which
rule, and say what to do next.

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


def build_gate(policy: str, model: str | Model | None = None) -> Agent[None, Verdict]:
    """A gate bound to one domain's policy. The policy is static, the case is not."""
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS.format(policy=policy),
        output_type=[Approved, Blocked],
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
