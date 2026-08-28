"""What the customer said about *which* record, copied out rather than judged.

Tier 1 answers "does the policy allow this". The judge next door answers "did
they ask for it". Neither answers the one that is left, and it is the largest
family in what is still getting through: the action is legal, well formed and
correctly priced, and it is pointed at the wrong reservation.

Task 17 is the whole shape of it. The customer says "add 3 checked bags, change
the passenger to myself, and upgrade to economy class" and never names a booking.
They hold four. Gold puts all three writes on FQ8APE, the only one with a single
passenger on it. The run gets FQ8APE right and then repeats all three on QKRY03
and UM3OG5, which have two passengers each. Eight of that task's eleven surplus
writes are one record, missed in all four runs.

WHY THIS IS AN EXTRACTOR AND NOT A JUDGE

We have measured a 20B model asked to rule on a record twice: the monolithic
critic at 53% precision, and the narrowed `requested` judge at 60% with almost no
recall. We have measured the same model producing a fact and letting code decide
six times, and every one of those is at 100%. The gap is not about the question's
wording. Asking "is this the right reservation?" requires holding the customer's
words, the record and a comparison at once and returning a verdict; asking "what
did the customer say the reservation looks like?" requires reading.

So this returns "one passenger", and `adapters.tau2.intended` counts the
passengers on the record. The comparison is arithmetic and never reaches a model.

IT IS NEVER SHOWN THE RECORD

Deliberately, and it is the load-bearing decision here. A model shown both the
customer's words and the record it is about to be compared against will describe
the record -- every criterion then matches by construction and the check is an
expensive way to approve everything. It sees the conversation and the action. It
does not see the reservation, the ledger, or any tool result.

EVERY CRITERION HAS TO BE QUOTED

`words` carries the customer's own phrasing, verbatim, and
`intended.grounded` checks that the quote really occurs in what the customer
typed before any criterion is allowed to block. A criterion that cannot point at
a line of the conversation is discarded whole. That is a hallucination guard that
costs nothing and runs in code, which is the only kind worth having here.

SILENCE IS THE COMMON ANSWER

Most customers describe nothing -- they name the reservation, or they hold one.
Every field defaults to null and null means "they did not say", never "no". This
is the same discipline `cancellable` runs on: an unknown is not a refusal, and it
is the reason that check refuses none of gold's own cancellations.

WHAT IT MEASURED, AND WHY IT IS OFF BY DEFAULT

Safe and small. Against `scripts/gate_bench.py` it blocks 0 of the 74 gold writes
in the run corpus and 0 of the 49 in the answer key -- the extraction is grounded
enough that nothing it returned was able to stop a correct action. It catches 2
surplus writes for 136 model calls.

That is not a trade worth making by default, so `STEWARD_SELECT` gates it. What
reading its output paid for instead was `intended.read_first`: the reason the
comparison stayed silent so often was not the description but the record, which
in the worst case had never been read at all. That check is free, exact, and
catches four times as many.

The extraction itself improved under measurement and the record is worth keeping.
Its first version quoted whole sentences and filled almost no fields -- it had
been told to copy and not infer, and it copied and stopped. Adding worked
examples and the rule that an identifier is not a description moved the criteria
it filled from 34 proposals to 42, and what it quotes from the sentence to the
span.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

import llm

__all__ = ["CASE", "Criteria", "build_selector", "described"]


class Criteria(BaseModel):
    """How the customer described the record, in fields that can be compared.

    Every field is optional and every field means "the customer said so". There
    is no value here for "they did not say" other than null, and no field is ever
    filled from the assistant's turns or from a guess about what is likely.
    """

    words: str = Field(
        default="",
        description=(
            "The customer's own words describing which record this action applies "
            "to, copied exactly as they typed them. Not a paraphrase and not your "
            "own summary -- the literal span. Leave empty if they never described "
            "it, and if it is empty every other field must be null."
        ),
    )
    passengers: int | None = Field(
        default=None,
        description=(
            "How many people the customer said are on this reservation. 'The "
            "passenger' and 'just me' are 1. 'My wife and I' is 2. Null unless "
            "they said or plainly implied a number."
        ),
    )
    cabin: str | None = Field(
        default=None,
        description=(
            "The cabin the customer said the reservation is booked in *now*, "
            "before this action: basic_economy, economy or business. Null if they "
            "only said what they want it changed to -- that is the action, not a "
            "description of the record."
        ),
    )
    origin: str | None = Field(
        default=None,
        description=(
            "The three-letter airport code the customer said this trip departs "
            "from, only if they gave a code. Null if they named a city, or said "
            "nothing."
        ),
    )
    destination: str | None = Field(
        default=None,
        description=(
            "The three-letter airport code the customer said this trip is going "
            "to, only if they gave a code. Null if they named a city, or said "
            "nothing."
        ),
    )
    flight_type: str | None = Field(
        default=None,
        description=(
            "one_way or round_trip, if the customer said which this reservation is. Null otherwise."
        ),
    )
    insurance: bool | None = Field(
        default=None,
        description=(
            "True if the customer said this reservation has travel insurance, "
            "false if they said it does not. Null if they did not say."
        ),
    )


INSTRUCTIONS = """
You copy out how a customer described a record. You do not decide anything.

You are shown a conversation and one action an assistant is about to take. Your
only job is to report what the customer said about *which* record that action
should land on -- the reservation itself, not what they want done to it.

WHAT COUNTS

Only the customer's turns. The assistant's turns are shown so that a "yes" has
something to point at; the assistant saying which reservation it picked is not
the customer describing one, and must never fill a field.

QUOTE THE SHORT SPAN, THEN FILL THE FIELD IT SUPPORTS

`words` is the customer's literal phrasing -- their exact characters, so it can be
found again in what they typed. Quote the *shortest* span that carries the
description, not the whole sentence around it.

A quote on its own is not an answer. Having quoted, say what the quote means in
the fields. "The passenger" is a quote; `passengers: 1` is the answer.

AN IDENTIFIER IS NOT A DESCRIPTION

If all the customer did was name the reservation -- "my reservation VA5SGQ",
"cancel HXDUBJ" -- they have described nothing. Leave `words` empty and every
field null. Someone else already handles a customer who named their booking, and
quoting the identifier back tells them nothing they did not have.

THE ACTION IS NOT THE DESCRIPTION EITHER

"Upgrade me to business" says what they want done. It does not say the
reservation is in business -- it says the opposite, and you still leave `cabin`
null, because "not business" is not a value.

WORKED EXAMPLES

  "I'd like to change the passenger to myself"
      words: "the passenger"      passengers: 1
      -- "the passenger", singular, says one person is on this reservation.

  "cancel all my upcoming flights that only have one passenger"
      words: "only have one passenger"      passengers: 1
      -- a condition over every reservation is a description of this one too.

  "my wife and I are booked on it"
      words: "my wife and I"      passengers: 2

  "add a bag to my basic economy booking"
      words: "my basic economy booking"      cabin: "basic_economy"

  "please cancel reservation HXDUBJ"
      words: ""      everything null
      -- an identifier, not a description.

  "upgrade me to business class"
      words: ""      everything null
      -- that is the action being taken, not what the record looks like now.

  "I want to change my flight"
      words: ""      everything null
      -- nothing was said about which record.

WHEN THEY DESCRIBE SEVERAL

A customer may ask for several things about several records. Report only the
description that belongs to the action you were shown.

Null is still the normal answer, and an empty one costs nothing. But a
description the customer *did* give, left unreported, is the thing this job
exists to catch.
""".strip()

CASE = """
THE CONVERSATION SO FAR
{conversation}

THE ACTION ABOUT TO BE TAKEN
{action}

WHAT THAT ACTION DOES
{meaning}

How did the customer describe the record this action should be applied to?
""".strip()

# One structured object out of a 20B model, which is the thing it is worst at.
# Three tries, matching the planner, because a dropped extraction here is a
# silent approval and the retry is cheap.
OUTPUT_RETRIES = 3


def build_selector(model: str | Model | None = None) -> Agent[None, Criteria]:
    """The extractor. No policy, no tools, no record -- a conversation and a call."""
    return Agent(
        model=model if isinstance(model, Model) else llm.get_model(model),
        instructions=INSTRUCTIONS,
        output_type=Criteria,
        retries={"output": OUTPUT_RETRIES},
    )


def described(
    selector: Agent[None, Criteria], conversation: str, action: str, meaning: str
) -> dict:
    """The criteria as a plain mapping, or an empty one when nothing was said.

    A mapping rather than the model, because `Evidence` is carried by `core` and
    `core` must not learn what a cabin is. The adapter that reads these keys is
    the same one that knows what they mean.

    Never raises. A provider hiccup is not evidence that a customer described a
    record, and a failure here has to mean "nothing to say" -- the opposite
    convention would turn every dropped packet into a blocked write, which is
    exactly how the critic this replaces earned its refusal rate.
    """
    try:
        criteria = selector.run_sync(
            CASE.format(conversation=conversation, action=action, meaning=meaning)
        ).output
    except Exception:  # UnexpectedModelBehavior, and every transport failure under it
        return {}
    stated = {key: value for key, value in criteria.model_dump().items() if value not in (None, "")}
    return stated if stated.get("words") else {}
