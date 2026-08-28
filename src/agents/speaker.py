"""SPEAKER: the check on the other way out of a turn.

The gate stands between the actor and the database. Nothing stood between the
actor and the customer, and that is the edge every lost task leaves through.

Across the two 15-task runs at the current commit, fourteen tasks were lost and
every one of them failed on DB -- the database was wrong at the end. Only two
also failed COMMUNICATE, so what the assistant *said* was very rarely the
problem. But eight of those fourteen made **no writes at all**, on tasks
requiring one to three, and finished with a reply. The turn ended by talking when
it should have ended by writing, the customer left, and the records were never
touched.

So this node asks one question, at the moment the actor tries to speak: the plan
named changes -- have they happened?

WHAT THIS IS NOT

It is not a critic of the message. Judging whether a reply is well-phrased, or
whether a question is worth asking, is exactly the open-ended judgment the gate
was failing at before it was narrowed, and a wrongly suppressed question is worse
than an unnecessary one: it pushes the actor into acting on information it does
not have. COMMUNICATE is already passing 13 of 15. There is nothing to win there.

The question here is a *count*, and the expensive half of it needs no model at
all. `outstanding` compares the writes the planner asked for against the writes
the gate has approved this turn, and when nothing is owed this node returns
without a model call -- which is what keeps it free on the twenty-three no-write
tasks that already pass at 0.695 and must not be disturbed.

THE FAILURE MODE TO AVOID

The policy *requires* the actor to stop and talk: quote the price, name the
penalty, wait for the customer to agree. A turn that ends by asking for
confirmation has done the right thing and has changes still outstanding, and it
will look identical to this check's deterministic half. That case is the reason
the model is asked at all, and it is why the instructions spend more room on when
to let the reply stand than on when to hold it.

WHAT THIS NODE NO LONGER HAS TO CATCH

That exemption for confirmation requests is broad, and for a while it was too
broad: the gate's most common refusal is a demand for confirmation, the actor
turns that demand into a message, and this check was told to allow every message
of exactly that shape. The two guards were pointed at each other, and it converted
none of the four holds it managed on the 50-task run.

So the case is taken off it. A refusal the gate marked as needing nothing from
the customer holds the reply in the Kernel, deterministically, before this node
is reached -- the ruling was already made and there is no second opinion left to
buy. What arrives here is what it was always for: a turn nobody has refused
anything on, where the plan named a change and the actor is leaving without it.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model

import llm
from agents.gate import UNAVAILABLE, Verdict, decide, transcript
from core.state import Change, Written

__all__ = [
    "HELD",
    "HOLD",
    "NOTHING_OWED",
    "OUTPUT_RETRIES",
    "UNCHECKED",
    "build_speaker",
    "hold",
    "outstanding",
    "permit",
]


INSTRUCTIONS = """
You decide whether an assistant may stop working and speak.

A customer service assistant is part-way through a turn. A planner wrote down
what this turn has to change in the company's records; some of those changes have
not been made; and the assistant is now about to send the customer a message,
which hands the turn back to them. If that message is the wrong thing to send,
the customer answers it, the conversation drifts, and the work is never done.

Your job is not to judge the wording. It is to answer one question: is stopping
here the right move, or is the assistant walking away from work it can do right
now?

WHAT YOU ARE GIVEN

CONVERSATION SO FAR is everything that has happened, including the assistant's
own lookups and what they returned. If it looked something up, the result is in
there, and that means the assistant already knows it.

STILL TO DO is what the planner said this turn must change and the gate has not
yet approved. It is a plan, not an instruction, and it can be wrong: a planner
that named a change the customer never asked for is not a reason to hold anything.

THE MESSAGE is what the assistant wants to send.

LET IT SPEAK UNLESS THE WORK IS AVAILABLE NOW

Allow the message unless the assistant could carry out one of the outstanding
changes on this turn, with what it already has, and is choosing to talk instead.

Understand what holding it costs. The assistant is sent back to work, and if it
had nothing it could do it comes straight back with the same message, one round
of budget poorer and the customer waiting. Talking is very often correct. Most
messages you see will be.

WHEN TO HOLD THE MESSAGE

- The customer already agreed to this change, and the assistant is saying
  goodbye, summarising, or offering further help instead of making it.
- The message asks the customer for a value that is already in the conversation
  -- a reservation id, a date of birth, a flight number, a payment method that a
  lookup already returned.
- The message says something cannot be found, and it is in the conversation.
  Cities and airport codes are the same thing under different names: a
  reservation from EWR to ORD is the trip a customer calls New York to Chicago.
- The assistant says it will do something and then does not do it in this turn.
- The message hands off to a human, or ends the conversation, with an outstanding
  change the assistant was never blocked from making.

WHEN TO LET IT SPEAK -- these are the expensive mistakes

- **The message is asking the customer to agree.** The policy requires the
  customer to be told the price, the fare difference or the penalty and to accept
  it before the records are touched. A turn that ends by quoting a figure and
  asking whether to go ahead is the policy working. Always allow it.
- **The message asks for something only the customer knows.** Which of three
  reservations they meant, whether they want the refund to a card or a
  certificate, what date they would rather fly. Nobody can look those up.
- **The work is done.** The outstanding list can lag: if the conversation shows
  the changes were made, or the customer changed their mind, allow the message.
- **The policy forbids the change** and the assistant is explaining why. Refusing
  a customer is part of the job and it is not walking away.
- **The assistant is missing a value and has said what it needs.** If it cannot
  act, stopping is right.
- **You are not certain.** If you cannot name the change it could make right now
  and the values it would use, you do not have grounds. Let it speak.

HOW TO ANSWER

`allowed` is true to send the message, false to hold it.

`reason` is one sentence: why stopping here is right, or which change is
available right now and what makes it available.

`remediation` matters only when you hold the message. It is read by the assistant
as its next instruction, so write it as one: name the tool to call and where each
value comes from. Do not tell it to think again or to be more careful. If you
cannot write that sentence, allow the message.

<policy>
{policy}
</policy>
""".strip()


HOLD = """
CONVERSATION SO FAR
{transcript}

STILL TO DO
{outstanding}

THE MESSAGE THE ASSISTANT WANTS TO SEND
{reply}
""".strip()


# What the actor is handed when a message is held. Phrased as a statement of
# where the turn stands rather than as a rebuke, because it arrives as an
# instruction and the actor acts on it directly.
HELD = "You have not finished this turn. {reason} {remediation}"

NOTHING_OWED = "Nothing. Every change the planner named has been approved."

# Malformed answers inside one `run_sync`, matching the gate. Same model, same
# output type, same failure: a 20B model answering in prose instead of filling
# the output tool in.
OUTPUT_RETRIES = 3


def build_speaker(policy: str, model: str | Model | None = None) -> Agent[None, Verdict]:
    """A speaker bound to one domain's policy.

    Returns the gate's `Verdict` rather than a type of its own. The shape is the
    same question -- may this happen, why not, what instead -- and sharing it
    means `decide` and its retry budget are shared too, which is the machinery
    that made the gate answerable at all.
    """
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS.format(policy=policy),
        output_type=Verdict,
        retries={"output": OUTPUT_RETRIES},
    )


def permit(speaker: Agent[None, Verdict], case: str) -> Verdict:
    """A ruling on the message, or permission to send it. Never raises.

    `decide` is shared with the gate and fails **closed**, which is right there and
    wrong here. The gate's refusal stops an irreversible action nobody checked; a
    refusal here stops a customer from being answered, and cannot itself produce
    the write it is asking for -- the actor comes back, has nothing new, and says
    the same thing a round of budget later. So a check that did not complete is
    turned back into permission at this one boundary.
    """
    verdict = decide(speaker, case)
    return UNCHECKED if verdict is UNAVAILABLE else verdict


# What an unanswered check means here: nothing, and the message goes.
UNCHECKED = Verdict(
    allowed=True,
    reason="The check on this message did not complete, so it stands as written.",
)


def outstanding(
    changes: list[Change], written: list[Written], ruled_out: list[Written] | None = None
) -> list[Change]:
    """Planned changes that neither an approved call nor a ruling has covered yet.

    The deterministic half, and the part that decides whether the model is asked
    at all. A change is done when the gate has approved a call to its tool that
    named its record.

    Both halves of that matter. This used to ask only whether the tool's name
    appeared somewhere in the change line, which meant one approved call
    discharged every change that mentioned the same tool -- so a request covering
    six reservations was satisfied, as far as this function could tell, by writing
    to one of them. Matching the record is what makes the sixth still owed while
    the first is done.

    A change whose record the plan never knew falls back to the tool alone, which
    is the old behaviour and the only answer available: there is nothing to match
    on. The bias stays where it was -- a change that cannot be shown to have
    happened stays outstanding, because the cost of asking about a turn that was
    actually finished is one model call, and the cost of missing one is the task.

    `ruled_out` discharges a change the same way an approved call does, and by the
    same comparison, because the question here is only whether the change is still
    open. A verifier has settled that it will never happen; leaving it owed would
    make the assistant liable for work the policy forbids. See
    `StewardState.ruled_out` for why only arithmetic is allowed to put anything
    there.
    """
    settled = list(written) + list(ruled_out or [])
    return [change for change in changes if not any(_covers(done, change) for done in settled)]


def _covers(done: Written, change: Change) -> bool:
    """Whether one approved call carries out one planned change."""
    if done.tool != change.tool:
        return False
    if change.record is None:
        return True
    # `in` rather than `==` because the planner does not always hand back a bare
    # identifier. Told to name the record, a fifth of the time it answers "the
    # reservation id from get_reservation_details for JG7FMM" -- the right record,
    # wrapped in where it came from. Containment reads that correctly and cannot
    # confuse two ids of the same length for each other.
    return any(record in change.record for record in done.records)


def hold(messages: list[ModelMessage], reply: str, owed: list[Change]) -> str:
    """The case put to the speaker: what happened, what is owed, what it wants to say."""
    return HOLD.format(
        transcript=transcript(messages) or "(nothing yet)",
        outstanding="\n".join(f"- {change.key}: {change.what}".rstrip(": ") for change in owed)
        if owed
        else NOTHING_OWED,
        reply=reply,
    )
