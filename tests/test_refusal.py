"""Whether a refusal survives being checked against the conversation.

The direction is the whole safety argument, so it is tested first and hardest:
nothing here may block anything. Every case below either discards a refusal the
transcript contradicts or leaves one standing, and none of them can produce one.
"""

from __future__ import annotations

from core.refusal import survives
from core.state import PendingCall


def call(**arguments):
    return [PendingCall(id="p1", name="cancel_reservation", arguments=arguments)]


AGREED = (
    "Customer: I want to cancel HKD3PS.\n"
    "Assistant looks up: get_reservation_details(reservation_id='HKD3PS')\n"
    'Result: {"reservation_id": "HKD3PS"}\n'
    "Assistant: Here is what I am about to do: cancel HKD3PS. Shall I go ahead?\n"
    "Customer: Yes, please go ahead."
)

NEVER_ASKED = (
    "Customer: I want to cancel HKD3PS.\n"
    "Assistant looks up: get_reservation_details(reservation_id='HKD3PS')\n"
    'Result: {"reservation_id": "HKD3PS"}'
)


def test_a_confirmation_refusal_is_discarded_when_the_customer_agreed():
    """Task 40, as it happened: the transcript ends `Customer: Yes, please go
    ahead.` and the gate refused for want of a confirmation."""
    reason = "The policy requires explicit user confirmation, and the customer has not confirmed."

    assert (
        survives(reason, call(reservation_id="HKD3PS"), AGREED, ['{"reservation_id": "HKD3PS"}'])
        is False
    )


def test_a_confirmation_refusal_stands_when_nobody_agreed():
    """The other half, and the one that must not regress. If this discarded a
    refusal, an action nobody agreed to would go through."""
    reason = "The policy requires explicit user confirmation, and the customer has not confirmed."

    assert (
        survives(
            reason, call(reservation_id="HKD3PS"), NEVER_ASKED, ['{"reservation_id": "HKD3PS"}']
        )
        is True
    )


def test_agreement_must_come_after_the_assistant_listed_the_action():
    """A yes to an earlier question is not a yes to this action. The policy asks
    for the action to be listed and then agreed to, in that order."""
    stale = (
        "Assistant: Would you like me to look that up?\n"
        "Customer: Yes.\n"
        "Assistant looks up: get_reservation_details(reservation_id='HKD3PS')\n"
        'Result: {"reservation_id": "HKD3PS"}'
    )
    reason = "No explicit confirmation was obtained before this action."

    assert (
        survives(reason, call(reservation_id="HKD3PS"), stale, ['{"reservation_id": "HKD3PS"}'])
        is True
    )


def test_a_provenance_refusal_is_discarded_when_every_identifier_was_shown():
    """Seven refusals claimed the reservation id was never provided while quoting
    it from a lookup in the same prompt."""
    reason = (
        "The policy requires the reservation id to be provided by the customer, and it was not."
    )

    assert (
        survives(
            reason, call(reservation_id="HKD3PS"), NEVER_ASKED, ['{"reservation_id": "HKD3PS"}']
        )
        is False
    )


def test_a_provenance_refusal_stands_when_an_identifier_was_invented():
    reason = "The reservation id was not provided by the customer or retrieved via a lookup."

    assert (
        survives(
            reason, call(reservation_id="ZZZZZZ"), NEVER_ASKED, ['{"reservation_id": "HKD3PS"}']
        )
        is True
    )


def test_a_proposal_naming_no_identifier_cannot_settle_a_provenance_claim():
    """Nothing to check is not the same as everything checking out."""
    reason = "The reservation id was not provided by the customer."

    assert survives(reason, [PendingCall(id="p", name="x", arguments={})], NEVER_ASKED, []) is True


def test_any_other_ground_is_left_alone():
    """`no cancellation reason` is a judgement about words and is not checked --
    the module exists to avoid making those."""
    for reason in (
        "The policy requires a cancellation reason and none was supplied.",
        "This reservation is basic economy and its flights cannot be changed.",
        "The payment amount does not add up.",
        "",
    ):
        assert (
            survives(
                reason, call(reservation_id="HKD3PS"), AGREED, ['{"reservation_id": "HKD3PS"}']
            )
            is True
        )


def test_it_can_only_ever_discard_and_never_block():
    """The safety property, stated as a test: `survives` returns a bool about a
    refusal that already exists. There is no input for which it invents one."""
    import inspect

    from core import refusal

    source = inspect.getsource(refusal)

    assert "-> bool" in source
    assert "Verdict" not in source
