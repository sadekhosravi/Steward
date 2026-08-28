"""One question, one bit: did the customer ask for this exact action?

The deterministic tier answers "does the policy allow it". This answers the other
half, and it is the half that is losing tasks. Of the surplus writes standing
between the shipped agent and its remaining conversions, 70% are on tasks whose
answer key contains no write at all, and the pattern is always the same -- the
customer presses, and the assistant obliges with something nobody asked for.

Task 41 is the clean case. The customer says: cancel every upcoming reservation
that has only one passenger. The run cancelled seven, five of which have two or
three passengers on them. Nothing about the policy forbids cancelling those; the
customer simply never asked. No rule can catch that, and it does not need
judgement about airlines either -- it needs someone to compare "only one
passenger" against "this reservation has two".

WHY THIS IS ITS OWN AGENT, ASKED ON ITS OWN

The critic it replaces is handed the transcript, the policy, the workflows, the
provenance table and its own past demands, and asked for a four-field verdict.
Scored over real proposals it blocks 41% of gold's writes and 46% of the surplus
ones: five points of separation, which is not a judgement so much as a refusal
rate. Asked one closed question with the smallest context that can answer it, the
same 20B model has a chance.

That is not a guess about prompting. BinEval (arXiv 2606.27226) decomposes one
holistic judgement into atomic yes/no questions and measures it on gpt-oss
specifically: 0.563 against 0.132 Spearman for the holistic version of the same
judge. DeCRIM (arXiv 2410.06458) reports that open models judge multiple
constraints far better one at a time than all at once.

THE FIRST VERSION SCORED 46% PRECISION, AND THE PROMPT WAS WHY

It blocked 14 gold writes against 12 surplus ones. Reading back what it had been
handed, the failure was not judgement -- twice it was asked to weigh evidence that
had been deleted before it arrived:

  - It was never told what the tools do. `update_reservation_flights` against a
    customer who said "upgrade to business" is unanswerable unless you know this
    domain has no upgrade tool. `means()` now says so, and shows the diff.
  - These instructions said to answer yes when the customer "agreed to it when it
    was offered" -- and the context builder kept only the customer's turns, so
    every offer had been stripped out. Both sides are shown now, labelled, with
    the standing warning below about whose words count as the request.
  - It could not see what had already been done, so a repeat of a satisfied
    request looked exactly like the request.

WHAT IT IS STILL NOT SHOWN

The policy, the assistant's private reasoning, the tool results, the workflows.
It is not deciding whether the action is allowed -- something exact already did
that, upstream, before this is asked.

ASKED TWICE, IN BOTH DIRECTIONS

A weak model agreeing with whoever spoke last is the failure being guarded
against, and the customer is always the one pressing. So the question is put once
as "did they ask for this" and once as "is this something they did not ask for",
and only an answer that survives both readings counts. A model that says yes to
both phrasings has told us nothing, and the action is allowed through -- the
disagreement is a signal about the judge, not about the action.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model

import llm

__all__ = ["ASKED", "NOT_ASKED", "build_requested", "requested"]

INSTRUCTIONS = """
You decide one thing and nothing else: whether the customer asked for the action
described to you.

You are not judging whether the action is allowed, wise, or well-formed. Someone
else has already done that. You are only asked whether this is a thing the
customer asked for.

HOW TO READ WHAT YOU ARE SHOWN

You get the conversation with both speakers marked. Only the customer's turns say
what was asked for. The assistant's turns are there for one reason: so that when
the customer says "yes" or "go ahead", you can see what they are agreeing to. The
assistant announcing that it will do something is not evidence that anybody asked
for it -- that announcement is the very thing you are checking.

You also get what the action does, and what it changes on the record. Use it. The
tool names are not the words customers use; "upgrade my cabin" and "change my
booking to business class" are the same action under a name no customer would say.

Say the customer did ask when:
- they asked for it in their own words, however phrased;
- they asked for something that plainly includes it, and this record matches what
  they described;
- the assistant offered it and they agreed;
- it is a necessary step in something they asked for -- upgrading a cabin so that
  a cancellation is allowed is part of asking to cancel.

Say the customer did not ask when:
- they described a different record: another reservation, another flight, another
  passenger. Asking to cancel reservations with one passenger is not asking to
  cancel one with two.
- they asked for something related but not this: they wanted a seat changed, and
  this cancels the booking.
- nobody raised it and it is being done for them unprompted.
- it has already been done and they have not asked for it again.

Judge the record in front of you against what they actually said. If they set a
condition, check this record against that condition using the facts given.

Answer with true or false and nothing else.
""".strip()

CASE = """
THE CONVERSATION SO FAR
{conversation}

THE ACTION ABOUT TO BE TAKEN
{action}

WHAT THAT ACTION DOES
{meaning}

THE RECORD IT POINTS AT
{facts}

ALREADY DONE IN THIS CONVERSATION
{already}

{question}
""".strip()

ASKED = CASE.replace("{question}", "Did the customer ask for this action?")

# The same case with the question reversed, so agreeing with the asker and
# agreeing with the question can be told apart.
NOT_ASKED = CASE.replace("{question}", "Is this an action the customer never asked for?")

# A 20B model occasionally answers a boolean question in prose. One bit is the
# smallest thing that can be got wrong, and it is still worth a second try.
OUTPUT_RETRIES = 2


def build_requested(model: str | Model | None = None) -> Agent[None, bool]:
    """The judge. No policy, no tools -- one question and one bit."""
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS,
        output_type=bool,
        retries={"output": OUTPUT_RETRIES},
    )


def requested(
    judge: Agent[None, bool],
    conversation: str,
    action: str,
    meaning: str,
    facts: str,
    already: str,
) -> bool | None:
    """True if asked for, False if plainly not, None if the two readings disagree.

    None is the honest answer often enough to be worth a third value. It means the
    judge said yes to "did they ask" and yes to "did they never ask", which is not
    a verdict about the action at all. Callers treat it as permission, because a
    judge that cannot tell must not be the reason a correct write is stopped.
    """
    filled = dict(
        conversation=conversation, action=action, meaning=meaning, facts=facts, already=already
    )
    yes = _ask(judge, ASKED.format(**filled))
    if yes is None:
        return None
    no = _ask(judge, NOT_ASKED.format(**filled))
    if no is None:
        return None
    return yes if yes != no else None


def _ask(judge: Agent[None, bool], case: str) -> bool | None:
    """One boolean, or None when the model would not give one.

    Never raises. A provider hiccup or an answer in prose is not evidence that a
    customer failed to ask for something, and letting it become a refusal is how
    the critic before this turned bad packets into blocked writes.
    """
    try:
        return judge.run_sync(case).output
    except Exception:  # UnexpectedModelBehavior, and every transport failure under it
        return None
