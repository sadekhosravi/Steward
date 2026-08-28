"""The deterministic checks, and the unknowns they are required to stay quiet on.

Every verifier here can stop a write, so each one is tested twice: once that it
fires on the violation it exists for, and once that it says nothing when the
evidence does not settle the matter. The second half is the one that matters. The
critic these replace blocks 41% of gold's writes and 46% of the surplus ones --
five points of separation -- and read by the predicate each refusal turns on
rather than its wording, most of them are unknowns treated as noes: "no flight
status was provided", "the reservation id was never established". A check that
blocks on what it has not been told reproduces exactly that failure with fewer
words.
"""

from __future__ import annotations

import json

from adapters.tau2.cancellable import cancellable
from adapters.tau2.compensation import compensation
from adapters.tau2.handoff import work_still_owed
from adapters.tau2.modifications import (
    baggage_only_grows,
    flights_changeable,
    passenger_count_fixed,
)
from adapters.tau2.payment import payment_composition, payment_for_change
from adapters.tau2.verifiers import PANEL
from agents.gate import Verdict as _Verdict
from core.state import PendingCall
from core.verifiers import Evidence, Panel, first


def call(name, **arguments):
    return PendingCall(id="p", name=name, arguments=arguments)


def reservation(**overrides):
    record = {
        "reservation_id": "XEHM4B",
        "user_id": "daiki_muller_1116",
        "origin": "LAS",
        "destination": "ATL",
        "flight_type": "round_trip",
        "cabin": "economy",
        "flights": [
            {"flight_number": "HAT005", "date": "2024-05-20", "price": 65},
            {"flight_number": "HAT178", "date": "2024-05-30", "price": 83},
        ],
        "passengers": [{"first_name": "Daiki", "last_name": "Muller", "dob": "1990-01-01"}],
        "created_at": "2024-05-01T05:17:41",
        "total_baggages": 1,
        "nonfree_baggages": 0,
        "insurance": "no",
        "status": None,
    }
    return json.dumps({**record, **overrides})


def profile(**overrides):
    record = {
        "user_id": "daiki_muller_1116",
        "name": {"first_name": "Daiki", "last_name": "Muller"},
        "membership": "regular",
        "payment_methods": {"credit_card_1": {"source": "credit_card", "id": "credit_card_1"}},
        "reservations": ["XEHM4B"],
    }
    return json.dumps({**record, **overrides})


def seen(*records, looked_up=None, committed=None):
    return Evidence.of(list(records), "", committed or [], looked_up or [])


# --------------------------------------------------------------- cancellation


def test_a_cancellation_with_no_ground_left_is_refused():
    finding = cancellable(call("cancel_reservation", reservation_id="XEHM4B"), seen(reservation()))

    assert finding is not None
    assert finding.check == "cancellable"


def test_a_business_reservation_may_always_be_cancelled():
    evidence = seen(reservation(cabin="business"))

    assert cancellable(call("cancel_reservation", reservation_id="XEHM4B"), evidence) is None


def test_task_7_upgrades_to_business_and_then_cancels():
    """Gold's own sequence, and the case that decides how the record is read.

    XEHM4B is basic economy, booked on 1 May, uninsured, both legs available --
    every ground reads false on the record as first seen. Gold cancels it anyway,
    because it upgrades the cabin first. The evidence ledger holds both versions
    and the later one has to win, or this refuses a gold write.
    """
    evidence = seen(reservation(cabin="basic_economy"), reservation(cabin="business"))

    assert cancellable(call("cancel_reservation", reservation_id="XEHM4B"), evidence) is None


def test_insurance_leaves_the_reason_to_decide_and_this_does_not_guess():
    evidence = seen(reservation(insurance="yes"))

    assert cancellable(call("cancel_reservation", reservation_id="XEHM4B"), evidence) is None


def test_a_booking_made_in_the_last_day_is_left_alone():
    evidence = seen(reservation(created_at="2024-05-15T09:00:00"))

    assert cancellable(call("cancel_reservation", reservation_id="XEHM4B"), evidence) is None


def test_an_airline_cancellation_established_by_a_lookup_is_a_ground():
    evidence = seen(
        reservation(), looked_up=[("get_flight_status", {"flight_number": "HAT005"}, "cancelled")]
    )

    assert cancellable(call("cancel_reservation", reservation_id="XEHM4B"), evidence) is None


def test_a_reservation_never_read_is_not_this_check_s_business():
    assert cancellable(call("cancel_reservation", reservation_id="XEHM4B"), seen()) is None


def test_an_unreadable_timestamp_is_not_read_as_an_old_booking():
    evidence = seen(reservation(created_at="not a date"))

    assert cancellable(call("cancel_reservation", reservation_id="XEHM4B"), evidence) is None


# -------------------------------------------------------------- compensation


def test_a_certificate_for_a_delay_before_anything_was_changed_is_refused():
    """Task 27: the customer is angry, asks to be paid, and refuses to change or
    cancel. The policy pays the delay rate only after the change happens."""
    evidence = seen(
        profile(membership="gold"),  # entitled, so the delay clause is what decides
        reservation(),
        looked_up=[("get_flight_status", {"flight_number": "HAT005"}, "delayed")],
    )
    finding = compensation(
        call("send_certificate", user_id="daiki_muller_1116", amount=50), evidence
    )

    assert finding is not None
    assert "changed or cancelled" in finding.reason


def test_the_delay_rate_is_allowed_once_the_change_has_gone_through():
    evidence = seen(
        profile(membership="gold"),
        reservation(),
        looked_up=[("get_flight_status", {"flight_number": "HAT005"}, "delayed")],
        committed=["update_reservation_flights"],
    )

    assert (
        compensation(call("send_certificate", user_id="daiki_muller_1116", amount=50), evidence)
        is None
    )


def test_the_cancellation_rate_needs_a_flight_that_was_actually_cancelled():
    evidence = seen(
        profile(membership="gold"),
        reservation(),
        looked_up=[("get_flight_status", {"flight_number": "HAT005"}, "delayed")],
    )
    finding = compensation(
        call("send_certificate", user_id="daiki_muller_1116", amount=100), evidence
    )

    assert finding is not None


def test_a_regular_member_in_economy_without_insurance_gets_nothing():
    evidence = seen(profile(), reservation())
    finding = compensation(
        call("send_certificate", user_id="daiki_muller_1116", amount=100), evidence
    )

    assert finding is not None
    assert "silver or gold" in finding.reason


def test_business_cabin_is_entitlement_on_its_own():
    """policy.md:161 -- silver/gold OR insurance OR business, and any one is enough."""
    evidence = seen(
        profile(),
        reservation(cabin="business"),
        looked_up=[("get_flight_status", {"flight_number": "HAT005"}, "cancelled")],
    )

    assert (
        compensation(call("send_certificate", user_id="daiki_muller_1116", amount=100), evidence)
        is None
    )


def test_nothing_read_at_all_means_the_facts_were_never_confirmed():
    finding = compensation(
        call("send_certificate", user_id="daiki_muller_1116", amount=100), seen()
    )

    assert finding is not None
    assert "confirmed" in finding.reason


# ------------------------------------------------------------- modifications


def test_moving_basic_economy_legs_is_refused():
    evidence = seen(reservation(cabin="basic_economy"))
    finding = flights_changeable(
        call(
            "update_reservation_flights",
            reservation_id="XEHM4B",
            cabin="basic_economy",
            flights=[{"flight_number": "HAT999", "date": "2024-05-21"}],
        ),
        evidence,
    )

    assert finding is not None
    assert "basic economy" in finding.reason


def test_leaving_basic_economy_while_moving_the_legs_is_what_gold_does():
    """Eight times. The policy forbids modifying basic economy *flights* and
    permits any reservation changing *cabin*, one line apart."""
    evidence = seen(reservation(cabin="basic_economy"))
    finding = flights_changeable(
        call(
            "update_reservation_flights",
            reservation_id="XEHM4B",
            cabin="business",
            flights=[{"flight_number": "HAT999", "date": "2024-05-21"}],
        ),
        evidence,
    )

    assert finding is None


def test_a_cabin_change_that_keeps_the_legs_is_allowed_on_basic_economy():
    evidence = seen(reservation(cabin="basic_economy"))
    finding = flights_changeable(
        call(
            "update_reservation_flights",
            reservation_id="XEHM4B",
            cabin="economy",
            flights=[
                {"flight_number": "HAT005", "date": "2024-05-20"},
                {"flight_number": "HAT178", "date": "2024-05-30"},
            ],
        ),
        evidence,
    )

    assert finding is None


def test_the_passenger_count_cannot_move():
    evidence = seen(reservation())
    finding = passenger_count_fixed(
        call(
            "update_reservation_passengers",
            reservation_id="XEHM4B",
            passengers=[{"first_name": "A"}, {"first_name": "B"}],
        ),
        evidence,
    )

    assert finding is not None
    assert "even a human agent" in finding.reason


def test_renaming_the_same_number_of_passengers_is_fine():
    evidence = seen(reservation())
    finding = passenger_count_fixed(
        call("update_reservation_passengers", reservation_id="XEHM4B", passengers=[{"a": 1}]),
        evidence,
    )

    assert finding is None


def test_bags_cannot_be_taken_away():
    evidence = seen(reservation(total_baggages=2))
    finding = baggage_only_grows(
        call("update_reservation_baggages", reservation_id="XEHM4B", total_baggages=0), evidence
    )

    assert finding is not None


def test_bags_can_be_added():
    evidence = seen(reservation(total_baggages=1))
    finding = baggage_only_grows(
        call("update_reservation_baggages", reservation_id="XEHM4B", total_baggages=3), evidence
    )

    assert finding is None


# ------------------------------------------------------------------- payment


def test_a_booking_paid_by_three_certificates_is_refused():
    """Task 14 did exactly this and the API took it."""
    finding = payment_composition(
        call(
            "book_reservation",
            payment_methods=[
                {"payment_id": "certificate_1", "amount": 100},
                {"payment_id": "certificate_2", "amount": 100},
                {"payment_id": "certificate_3", "amount": 100},
            ],
        ),
        seen(),
    )

    assert finding is not None
    assert "certificate" in finding.reason


def test_three_gift_cards_are_within_the_limit():
    finding = payment_composition(
        call(
            "book_reservation",
            payment_methods=[
                {"payment_id": "gift_card_1", "amount": 1},
                {"payment_id": "gift_card_2", "amount": 1},
                {"payment_id": "gift_card_3", "amount": 1},
            ],
        ),
        seen(),
    )

    assert finding is None


def test_a_certificate_cannot_pay_for_a_change():
    finding = payment_for_change(
        call("update_reservation_flights", reservation_id="X", payment_id="certificate_7504069"),
        seen(),
    )

    assert finding is not None


def test_a_gift_card_can():
    finding = payment_for_change(
        call("update_reservation_flights", reservation_id="X", payment_id="gift_card_1"), seen()
    )

    assert finding is None


# ----------------------------------------------------------------- the panel


def test_the_panel_asks_only_the_checks_that_apply_to_the_tool():
    asked = []

    def noisy(check_call, _evidence):
        asked.append(check_call.name)
        return None

    panel = Panel(verifiers={"cancel_reservation": [noisy]})
    first(call("book_reservation"), seen(), panel)

    assert asked == []


def test_the_panel_stops_at_the_first_thing_that_fires():
    from core.verifiers import Finding

    reached = []

    def fires(_call, _evidence):
        return Finding(check="first", reason="r", remediation="m")

    def later(_call, _evidence):
        reached.append(True)
        return None

    panel = Panel(verifiers={"cancel_reservation": [fires, later]})
    finding = first(call("cancel_reservation"), seen(), panel)

    assert finding.check == "first"
    assert reached == []


# ------------------------------------------------- the seam into the Kernel


def test_a_verifier_blocks_even_with_the_critic_switched_off(monkeypatch):
    """The deterministic tier is not part of the critic and does not share its
    switch. `STEWARD_GATE=off` is a measurement arm about one model call; a
    check that costs nothing and refuses none of gold's writes has no business
    being turned off with it."""
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    def never(*args, **kwargs):
        raise AssertionError("the critic was asked despite being turned off")

    monkeypatch.setattr(kernel, "REVIEWING", False)
    monkeypatch.setattr(kernel, "decide", never)

    proposed = {
        "id": "1",
        "name": "update_reservation_passengers",
        "arguments": {"reservation_id": "XEHM4B", "passengers": [{"a": 1}, {"b": 2}]},
    }
    state = StewardState(calls=[proposed], observed=[reservation()])
    out = kernel._gate(
        state,
        None,
        frozenset({"update_reservation_passengers"}),
        Panel(verifiers={"update_reservation_passengers": [passenger_count_fixed]}),
    )

    assert out["approved"] == []
    # Charged to the sieve's budget, not the critic's -- the critic never ruled.
    assert out["blocked"] == 1
    assert "revisions" not in out
    assert "1" in out["denied"]
    assert "number of passengers" in out["denied"]["1"]
    # An argument-only fix is one the assistant can carry out on its next turn.
    assert out["fixable"]


def test_a_refusal_the_customer_has_to_answer_does_not_send_the_actor_back(monkeypatch):
    """Most findings end in "tell the customer no", which is the turn ending
    correctly rather than something to retry."""
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    monkeypatch.setattr(kernel, "REVIEWING", False)
    proposed = {"id": "1", "name": "cancel_reservation", "arguments": {"reservation_id": "XEHM4B"}}
    state = StewardState(calls=[proposed], observed=[reservation()])
    out = kernel._gate(
        state,
        None,
        frozenset({"cancel_reservation"}),
        Panel(verifiers={"cancel_reservation": [cancellable]}),
    )

    assert out["approved"] == []
    assert out["fixable"] == ""


def test_an_empty_panel_changes_nothing(monkeypatch):
    """`core` ships without a domain, so the default has to be inert."""
    from core import kernel
    from core.state import StewardState

    monkeypatch.setattr(kernel, "REVIEWING", False)
    proposed = {"id": "1", "name": "cancel_reservation", "arguments": {"reservation_id": "XEHM4B"}}
    state = StewardState(calls=[proposed], observed=[reservation()])

    out = kernel._gate(state, None, frozenset({"cancel_reservation"}))

    assert out["approved"] == [proposed]


def test_the_airline_panel_is_the_one_the_adapter_hands_over():
    """A panel built here and never wired in would pass every test above and do
    nothing in a run."""
    import inspect

    from adapters.tau2 import agent
    from adapters.tau2.verifiers import PANEL

    source = inspect.getsource(agent)

    assert "panel=PANEL" in source
    assert PANEL.for_tool("cancel_reservation")


# ------------------------------------------------------- the two budgets


def test_a_verifier_block_does_not_spend_the_critic_s_argument_budget(monkeypatch):
    """The failure this split exists to stop.

    Task 37's customer names three reservations and the policy forbids touching
    two of them. Under one shared budget the second refusal put the turn one step
    from `escalate`, where the actor is re-run without tools -- so the third
    reservation, the only one gold actually changes, could never be reached."""
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    monkeypatch.setattr(kernel, "REVIEWING", False)
    proposed = {
        "id": "1",
        "name": "update_reservation_passengers",
        "arguments": {"reservation_id": "XEHM4B", "passengers": [{"a": 1}, {"b": 2}]},
    }
    panel = Panel(verifiers={"update_reservation_passengers": [passenger_count_fixed]})

    blocked = 0
    for _ in range(kernel.REVISION_LIMIT + 2):
        state = StewardState(calls=[proposed], observed=[reservation()], blocked=blocked)
        out = kernel._gate(state, None, frozenset({"update_reservation_passengers"}), panel)
        blocked = out["blocked"]
        assert kernel._route_gate(StewardState(**{**out, "calls": []})) == "think"


def test_the_sieve_still_ends_a_turn_that_will_not_stop(monkeypatch):
    """Larger is not unbounded. Run 001 produced four simulations that never
    terminated and every budget here exists because of them."""
    from core import kernel
    from core.state import StewardState

    state = StewardState(blocked=kernel.BLOCK_LIMIT + 1)

    assert kernel._route_gate(state) == "escalate"


def test_the_critic_keeps_its_own_small_budget():
    """Arguing with a model that has already said no is worth two rounds, not
    eight, and raising the sieve's ceiling must not raise that."""
    from core import kernel
    from core.state import StewardState

    assert kernel._route_gate(StewardState(revisions=kernel.REVISION_LIMIT)) == "think"
    assert kernel._route_gate(StewardState(revisions=kernel.REVISION_LIMIT + 1)) == "escalate"
    assert kernel.BLOCK_LIMIT > kernel.REVISION_LIMIT


def test_both_counters_reset_when_a_new_user_turn_arrives():
    """A budget that carried over would make a long conversation progressively
    harder to serve, which is the bug one turn up from the one being fixed."""
    import inspect

    from core import kernel

    source = inspect.getsource(kernel.Kernel.send)

    assert '"revisions": 0' in source
    assert '"blocked": 0' in source


def test_a_flown_leg_does_not_tell_the_actor_to_abandon_the_other_reservations():
    """Transfer ends the conversation, so it has to be the last thing done.

    Task 37 lost its one gold write to this wording: the customer named three
    reservations, one had flown, and the run transferred before doing the upgrade
    the answer key asks for."""
    from adapters.tau2.flown import not_yet_flown

    flown = reservation(flights=[{"flight_number": "HAT001", "date": "2024-05-13"}])
    finding = not_yet_flown(call("cancel_reservation", reservation_id="XEHM4B"), seen(flown))

    assert finding is not None
    assert "finish everything else" in finding.remediation
    assert "Only once nothing else is left" in finding.remediation


# ------------------------------------------------- the critic rules first


def test_a_proposal_the_critic_refuses_never_reaches_the_verifiers(monkeypatch):
    """The ordering that the 2x2 bought.

    Verifiers-first and critic-first block the same proposals; what differs is
    which component is denied the chance to see one. Running the free checks
    first means every proposal they answer is a proposal the critic never rules
    on, and the arms measured 0.487 alone, 0.493 alone and only 0.500 together --
    an interaction of -0.060. The critic now goes first, and these run over what
    it allowed.
    """
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    monkeypatch.setattr(kernel, "REVIEWING", True)
    monkeypatch.setattr(
        kernel, "decide", lambda *_: _Verdict(allowed=False, reason="no", remediation="ask")
    )

    def never(*_args, **_kwargs):
        raise AssertionError("the verifiers ran on a proposal the critic had already refused")

    proposed = {"id": "1", "name": "cancel_reservation", "arguments": {"reservation_id": "XEHM4B"}}
    state = StewardState(calls=[proposed], observed=[reservation()])

    out = kernel._gate(
        state,
        None,
        frozenset({"cancel_reservation"}),
        Panel(verifiers={"cancel_reservation": [never]}),
    )

    assert out["approved"] == []
    assert out["revisions"] == 1


def test_a_verifier_still_vetoes_what_the_critic_allowed(monkeypatch):
    """The half of the reversal that has to keep working: the critic's word is not
    final, it is only first."""
    from core import kernel
    from core.state import StewardState
    from core.verifiers import Panel

    monkeypatch.setattr(kernel, "REVIEWING", True)
    monkeypatch.setattr(
        kernel, "decide", lambda *_: _Verdict(allowed=True, reason="fine", remediation="")
    )

    proposed = {"id": "1", "name": "cancel_reservation", "arguments": {"reservation_id": "XEHM4B"}}
    state = StewardState(calls=[proposed], observed=[reservation()])

    out = kernel._gate(
        state,
        None,
        frozenset({"cancel_reservation"}),
        Panel(verifiers={"cancel_reservation": [cancellable]}),
    )

    assert out["approved"] == []
    assert out["blocked"] == 1
    assert "revisions" not in out  # the deterministic budget, not the critic's


# --- the handoff ---------------------------------------------------------
#
# The one check here whose subject is the turn rather than the record, and the
# one with the most to lose by firing wrongly: 42 of the 51 transfers in the arm
# C run scored 1.00, because most of this task set is requests the policy refuses
# and handing over is the right ending. So the tests below spend most of their
# room on silence.


def test_a_handoff_is_refused_while_the_plan_still_owes_a_change():
    finding = work_still_owed(
        call("transfer_to_human_agents", summary="cannot remove the passenger"),
        Evidence.of([], owed=[("book_reservation", None)]),
    )
    assert finding is not None
    assert finding.check == "work_still_owed"
    # The remediation is the actor's entire retry prompt, so it has to name the call.
    assert "book_reservation" in finding.remediation
    assert finding.recoverable is True


def test_the_refusal_names_the_record_the_change_was_going_to_land_on():
    finding = work_still_owed(
        call("transfer_to_human_agents"),
        Evidence.of([], owed=[("cancel_reservation", "H9ZU1C")]),
    )
    assert "cancel_reservation on H9ZU1C" in finding.remediation


def test_a_handoff_with_nothing_outstanding_is_left_alone():
    """The common case, and the one that carries the reward: refusing a request
    the policy does not permit and handing over is a task passed, not a failure."""
    assert work_still_owed(call("transfer_to_human_agents"), Evidence.of([])) is None


def test_a_handoff_is_not_judged_by_whether_the_customer_asked_for_a_person():
    """Measured over the 51 transfers, that was true in 12 and separates nothing.

    A verifier that read it would block correct handoffs and miss the abandonment
    it exists for, so the dialogue is deliberately not consulted.
    """
    asked = Evidence.of([], dialogue="Customer: just put me through to a real person")
    unasked = Evidence.of([], dialogue="Customer: fine, whatever")
    assert work_still_owed(call("transfer_to_human_agents"), asked) is None
    assert work_still_owed(call("transfer_to_human_agents"), unasked) is None
    begged = Evidence.of(
        [], dialogue="Customer: put me through to a person", owed=[("book_reservation", None)]
    )
    assert work_still_owed(call("transfer_to_human_agents"), begged) is not None


def test_the_panel_asks_this_of_a_handoff_and_of_nothing_else():
    owing = Evidence.of([], owed=[("book_reservation", None)])
    assert first(call("transfer_to_human_agents"), owing, PANEL) is not None
    # A write proposed while another change is outstanding is ordinary work. It
    # has its own checks and may well fail one of them -- what it must never do
    # is fail this one.
    finding = first(call("cancel_reservation", reservation_id="H9ZU1C"), owing, PANEL)
    assert finding is None or finding.check != "work_still_owed"
