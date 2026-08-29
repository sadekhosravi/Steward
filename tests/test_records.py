"""The seam between the notes and the verifiers, which is where the run was lost.

`adapters.tau2.agent._noted` appends English to a tool result on its way in, and
everything in `records` reads those results back out. Both halves were tested
thoroughly and apart, and the join between them was tested nowhere -- so a
`json.loads` that rejects a record with a paragraph stapled to it survived into
a benchmark run and made every verifier blind to every reservation.

`tests/test_verifiers_against_gold.py` and `scripts/gate_bench.py` both fed the
*raw* environment output, which is a shape production never produces. These tests
use `_noted` itself, so a note whose format changes breaks a test here rather
than a run three weeks later.
"""

from __future__ import annotations

import json

import pytest

from adapters.tau2.agent import _noted
from adapters.tau2.records import reservations, statuses, users

RESERVATION = {
    "reservation_id": "Z7GOZK",
    "user_id": "olivia_gonzalez_2305",
    "origin": "EWR",
    "destination": "IAH",
    "flight_type": "round_trip",
    "cabin": "basic_economy",
    "flights": [
        {
            "flight_number": "HAT188",
            "origin": "EWR",
            "destination": "IAH",
            "date": "2024-05-28",
            "price": 52,
        }
    ],
    "passengers": [{"first_name": "Olivia", "last_name": "Gonzalez", "dob": "1988-06-13"}],
    "payment_history": [{"payment_id": "gift_card_2200803", "amount": 169}],
    "created_at": "2024-05-13T19:41:32",
    "total_baggages": 0,
    "nonfree_baggages": 0,
    "insurance": "yes",
    "status": None,
}

PROFILE = {
    "user_id": "olivia_gonzalez_2305",
    "name": {"first_name": "Olivia", "last_name": "Gonzalez"},
    "payment_methods": {
        "gift_card_2200803": {"source": "gift_card", "id": "gift_card_2200803", "amount": 123.0}
    },
    "membership": "regular",
    "reservations": ["Z7GOZK"],
}

ROWS = [
    {
        "flight_number": "HAT188",
        "origin": "EWR",
        "destination": "IAH",
        "status": "available",
        "prices": {"basic_economy": 52, "economy": 100, "business": 300},
    },
    {
        "flight_number": "HAT207",
        "origin": "EWR",
        "destination": "IAH",
        "status": "available",
        "prices": {"basic_economy": 61, "economy": 120, "business": 340},
    },
]


def test_a_reservation_with_its_notes_appended_is_still_a_reservation():
    """The defect, stated once. This failed before `_loaded` learned to read a tail."""
    noted = _noted(json.dumps(RESERVATION))
    assert noted != json.dumps(RESERVATION), "the fixture must exercise a real note"
    assert "Z7GOZK" in reservations([noted])


def test_a_profile_with_its_notes_appended_is_still_a_profile():
    noted = _noted(json.dumps(PROFILE))
    assert noted != json.dumps(PROFILE)
    assert "olivia_gonzalez_2305" in users([noted])


def test_a_bare_record_with_no_notes_still_reads():
    """Not every result attracts a note, and the plain shape must not regress."""
    assert "Z7GOZK" in reservations([json.dumps(RESERVATION)])


def test_leading_whitespace_is_not_a_reason_to_lose_a_record():
    assert "Z7GOZK" in reservations(["\n  " + _noted(json.dumps(RESERVATION))])


def test_a_search_result_keeps_its_rows_under_the_comparison_note():
    """`ranked` appends to a *list*, and `statuses` reads those rows back."""
    noted = _noted(json.dumps(ROWS))
    assert "COMPARING" in noted
    assert statuses([("search_direct_flight", {}, noted)])["HAT188"] == "available"


@pytest.mark.parametrize(
    "text",
    [
        "available",
        '"delayed"',
        "Error: Reservation AB1234 not found",
        "Transfer successful",
        "",
        "{not json at all",
        '{"reservation_id": "Z7GOZK"',
    ],
    ids=[
        "bare_word",
        "quoted_word",
        "error_text",
        "handoff_text",
        "empty",
        "malformed",
        "truncated",
    ],
)
def test_results_that_are_not_records_yield_nothing(text):
    """Every non-record shape a real run produces, and two that are simply broken."""
    assert reservations([text]) == {}
    assert users([text]) == {}


def test_a_number_is_not_a_record():
    """`calculate` answers with a bare number. It parses, and it is not a record."""
    assert reservations(["580.0"]) == {}


@pytest.mark.parametrize(
    "text",
    ['{"reservation_id": "Z7GOZK"}{"reservation_id": "OBUT9V"}', "2024-05-15", "580 dollars"],
    ids=["two_records_run_together", "a_date", "a_number_with_words"],
)
def test_a_tail_that_is_not_a_note_is_refused_rather_than_half_read(text):
    """The reason this does not simply call `raw_decode` and take what it gets.

    Each of these parses as a leading JSON value with something left over, and in
    every case taking the value alone would be reading something nobody wrote. A
    date is the number 2024; two records run together are the first one.
    """
    assert reservations([text]) == {}


def test_a_record_read_twice_keeps_the_newer_one():
    """Notes must not break the rule that a re-read record supersedes the first."""
    later = dict(RESERVATION, cabin="business")
    found = reservations([_noted(json.dumps(RESERVATION)), _noted(json.dumps(later))])
    assert found["Z7GOZK"]["cabin"] == "business"
