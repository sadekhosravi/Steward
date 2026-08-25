"""The `requested` judge: its two-sided asking, and the context it is given.

The model itself is not tested here -- `scripts/gate_bench.py` scores that against
the labelled corpus. What is tested is everything around it that decides whether
the question it gets is answerable and whether its answer is used correctly,
because that is where the first version lost its precision. It scored 46%, and
reading back what it had been handed, two of the three causes were the prompt
asking about evidence the context builder had already deleted:

  - the instructions said an offer the customer agreed to counts as asking, and
    every offer had been stripped out with the rest of the assistant's turns;
  - the action arrived as a bare tool name with no statement of what it does, so
    "upgrade to business" could not be matched to `update_reservation_flights`.

Both are pinned below, because both would pass silently if they came back.
"""

from __future__ import annotations

import json

from adapters.tau2.context import already, exchange, facts, means, spelled
from agents.requested import requested


class Stub:
    """A judge that answers from a script, so the two-sided logic can be pinned."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def run_sync(self, case, **_kwargs):
        self.asked.append(case)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return type("Run", (), {"output": answer})()


def ask(judge, **overrides):
    case = {
        "conversation": "customer: c",
        "action": "a",
        "meaning": "m",
        "facts": "f",
        "already": "n",
    }
    return requested(judge, **{**case, **overrides})


def test_both_readings_must_agree_for_a_refusal():
    """Asked "did they ask?" -> no, and "did they never ask?" -> yes."""
    judge = Stub(False, True)

    assert ask(judge) is False


def test_both_readings_must_agree_for_permission():
    judge = Stub(True, False)

    assert ask(judge) is True


def test_a_judge_that_says_yes_to_both_phrasings_has_told_us_nothing():
    """The sycophancy guard. Agreeing with the question rather than reading the
    case shows up as agreeing with both of them."""
    judge = Stub(True, True)

    assert ask(judge) is None


def test_a_judge_that_says_no_to_both_is_equally_useless():
    judge = Stub(False, False)

    assert ask(judge) is None


def test_the_second_question_is_not_asked_when_the_first_fails():
    judge = Stub(RuntimeError("bad packet"))

    assert ask(judge) is None
    assert len(judge.asked) == 1


def test_a_failure_on_the_second_question_is_not_a_refusal():
    judge = Stub(False, RuntimeError("bad packet"))

    assert ask(judge) is None


def test_the_two_questions_are_actually_different():
    judge = Stub(True, False)
    ask(judge)

    assert judge.asked[0] != judge.asked[1]
    assert "Did the customer ask for this action?" in judge.asked[0]
    assert "never asked" in judge.asked[1]


def test_every_piece_of_context_reaches_the_model():
    """A block added to the template and never filled would leave the judge
    exactly as blind as before while looking fixed."""
    judge = Stub(True, False)
    ask(
        judge,
        conversation="customer: CONV",
        action="ACTION",
        meaning="MEANING",
        facts="FACTS",
        already="ALREADY",
    )

    for piece in ("CONV", "ACTION", "MEANING", "FACTS", "ALREADY"):
        assert piece in judge.asked[0]
        assert piece in judge.asked[1]


# ------------------------------------------------------------------- context


def reservation(**overrides):
    record = {
        "reservation_id": "LU15PA",
        "user_id": "amelia_davis_8890",
        "origin": "SFO",
        "destination": "JFK",
        "flight_type": "round_trip",
        "cabin": "basic_economy",
        "flights": [{"flight_number": "HAT001", "date": "2024-05-20", "price": 100}],
        "passengers": [{"first_name": "A"}, {"first_name": "B"}],
        "created_at": "2024-05-01T00:00:00",
        "total_baggages": 1,
        "nonfree_baggages": 0,
        "insurance": "no",
    }
    return json.dumps({**record, **overrides})


def test_the_passenger_count_reaches_the_judge():
    """Task 41 turns on it. The customer asks to cancel reservations with one
    passenger; without this line the call says nothing about how many there are."""
    text = facts("cancel_reservation", {"reservation_id": "LU15PA"}, [reservation()])

    assert "passengers: 2" in text


def test_a_reservation_nobody_read_is_admitted_rather_than_invented():
    text = facts("cancel_reservation", {"reservation_id": "NOPE"}, [reservation()])

    assert "No record" in text


def test_a_booking_describes_itself_because_there_is_no_record_yet():
    text = facts(
        "book_reservation",
        {
            "user_id": "u",
            "origin": "SFO",
            "destination": "JFK",
            "cabin": "economy",
            "passengers": [{"first_name": "A"}],
            "flights": [{"flight_number": "X", "date": "d"}],
        },
        [],
    )

    assert "a new booking" in text
    assert "passengers: 1" in text


def test_the_call_is_spelled_out_for_a_reader():
    line = spelled("cancel_reservation", {"reservation_id": "LU15PA"})

    assert line == "cancel_reservation(reservation_id='LU15PA')"


# ------------------------------------------------- what the action means


def test_an_upgrade_is_recognisable_as_an_upgrade():
    """Task 7's customer says "upgrade XEHM4B to business". The call is
    `update_reservation_flights`, and without this the two cannot be matched."""
    text = means(
        "update_reservation_flights",
        {
            "reservation_id": "LU15PA",
            "cabin": "business",
            "flights": [{"flight_number": "HAT001", "date": "2024-05-20"}],
        },
        [reservation()],
    )

    assert "no separate upgrade tool" in text
    assert "cabin: basic_economy -> business" in text
    assert "flights: unchanged" in text


def test_a_real_flight_change_says_the_flights_moved():
    text = means(
        "update_reservation_flights",
        {
            "reservation_id": "LU15PA",
            "cabin": "basic_economy",
            "flights": [{"flight_number": "HAT999", "date": "2024-06-01"}],
        },
        [reservation()],
    )

    assert "cabin: unchanged" in text
    assert "HAT999" in text


def test_an_action_against_a_record_nobody_read_still_says_what_it_does():
    """No diff is available, but the meaning never depended on the record."""
    text = means("cancel_reservation", {"reservation_id": "NOPE"}, [])

    assert "Cancels an entire existing reservation" in text


def test_a_certificate_is_named_as_compensation_rather_than_a_refund():
    """The confusion is in the transcripts: customers ask for their money back
    and the run sends a certificate, which is not the same thing."""
    text = means("send_certificate", {"user_id": "u", "amount": 100}, [])

    assert "compensation, not a refund" in text


# ---------------------------------------------------------- the exchange


def test_both_speakers_are_shown_and_marked():
    """The offer has to be visible or the agreement to it means nothing -- and it
    has to be marked, or the assistant's account of the request becomes the
    evidence for it."""
    dialogue = "\n".join(
        [
            "Customer: I want to cancel LU15PA",
            "Assistant: That is basic economy. I can upgrade it first, shall I?",
            "Customer: Yes",
        ]
    )
    text = exchange(dialogue)

    assert "customer: I want to cancel LU15PA" in text
    assert "assistant: That is basic economy. I can upgrade it first, shall I?" in text
    assert text.strip().endswith("customer: Yes")


def test_lookups_and_their_results_stay_out():
    """They are the bulk of the transcript and none of them is anyone asking for
    anything."""
    dialogue = "\n".join(
        [
            "Customer: cancel the ones with one passenger",
            "Assistant looks up: get_user_details(user_id='u')",
            "Result: {...}",
        ]
    )
    text = exchange(dialogue)

    assert "cancel the ones with one passenger" in text
    assert "get_user_details" not in text
    assert "{...}" not in text


def test_the_most_recent_turns_are_the_ones_kept():
    dialogue = "\n".join(f"Customer: turn {index}" for index in range(20))
    text = exchange(dialogue, turns=3)

    assert "turn 19" in text
    assert "turn 16" not in text


def test_an_empty_conversation_says_so_rather_than_going_blank():
    assert "nothing said yet" in exchange("")


# ------------------------------------------------------------- what is done


def test_what_has_already_been_written_is_shown():
    """Without it, a second attempt at a satisfied request looks like the first."""
    assert "cancel_reservation" in already(["cancel_reservation"])


def test_a_conversation_that_has_changed_nothing_says_so():
    assert "Nothing has been changed yet" in already([])


def test_the_same_tool_twice_is_listed_once():
    assert already(["cancel_reservation", "cancel_reservation"]).count("cancel_reservation") == 1
