"""The extractor's contract with the code that compares what it returns.

The model is not tested here -- `scripts/gate_bench.py --gate selection` scores
that against the labelled corpus. What is tested is the boundary: that an empty
or failed extraction becomes silence rather than a refusal, that an unquoted
answer is discarded whole, and that the record never reaches the question.

That last one is the load-bearing property of the whole design. A model shown
both the customer's words and the record its answer will be compared against
describes the record, every comparison matches, and the check becomes an
expensive way to approve everything.
"""

from __future__ import annotations

from adapters.tau2.describing import asking, describing
from agents.selector import Criteria, described
from core.state import PendingCall
from core.verifiers import Evidence


class Stub:
    """A selector that answers from a script, so the plumbing can be pinned."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def run_sync(self, case, **_kwargs):
        self.asked.append(case)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return type("Run", (), {"output": answer})()


def test_a_quoted_description_comes_back_as_facts_to_compare():
    stated = described(
        Stub(Criteria(words="only one passenger", passengers=1)), "customer: c", "a", "m"
    )

    assert stated == {"words": "only one passenger", "passengers": 1}


def test_a_customer_who_described_nothing_produces_nothing():
    """The common answer. Most customers name a booking or hold one."""
    assert described(Stub(Criteria()), "customer: c", "a", "m") == {}


def test_a_criterion_with_no_quote_behind_it_is_discarded_along_with_the_rest():
    """`words` is the only thing standing between a filled field and a guess, so
    a field arriving without one takes the whole extraction down with it."""
    assert described(Stub(Criteria(passengers=1)), "customer: c", "a", "m") == {}


def test_a_provider_failure_is_not_evidence_that_a_customer_described_anything():
    """The convention the critic this replaces got backwards: it answered an
    unreachable model with a refusal, so every dropped packet blocked a write."""
    assert described(Stub(RuntimeError("502")), "customer: c", "a", "m") == {}


RECORD = (
    '{"reservation_id": "UM3OG5", "cabin": "economy", "origin": "LAS", '
    '"destination": "ATL", "flight_type": "round_trip", "total_baggages": 0, '
    '"flights": [{"flight_number": "HAT005", "date": "2024-05-20"}], '
    '"passengers": [{"first_name": "Omar"}, {"first_name": "Mia"}]}'
)


def bags():
    return PendingCall(
        id="p", name="update_reservation_baggages", arguments={"reservation_id": "UM3OG5"}
    )


def test_the_extractor_is_never_shown_the_record_it_will_be_checked_against():
    """The load-bearing property. `means()` prints a diff against the reservation
    wherever one has been read, and here it must not: the answer is about to be
    compared against that same record, and a model that has read it describes it
    back -- every comparison then matches and the check approves everything."""
    _conversation, _action, meaning = asking(
        bags(), Evidence.of([RECORD], "Customer: add three bags")
    )

    assert "What changes" not in meaning
    assert "UM3OG5" not in meaning
    assert "economy" not in meaning


def test_the_conversation_and_the_call_itself_do_reach_the_question():
    conversation, action, meaning = asking(
        bags(), Evidence.of([RECORD], "Customer: add three bags\nAssistant: Sure.")
    )

    assert conversation == "customer: add three bags\nassistant: Sure."
    assert action == "update_reservation_baggages(reservation_id='UM3OG5')"
    assert "checked bags" in meaning


def test_a_stubbed_selector_reaches_the_kernel_s_side_of_the_seam_unchanged():
    selector = Stub(Criteria(words="add three bags", passengers=1))
    describe = describing(selector=selector)

    stated = describe(bags(), Evidence.of([RECORD], "Customer: add three bags"))

    assert stated == {"words": "add three bags", "passengers": 1}
    assert "add three bags" in selector.asked[0]
    assert "economy" not in selector.asked[0]
