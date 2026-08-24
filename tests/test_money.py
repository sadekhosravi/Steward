"""The payment rules appended to a user profile.

Every fixture is the shape tau2 really returns -- `payment_methods` as a mapping
of id to a record carrying `source`, and an `amount` on the two kinds that have a
balance. The cases are the ones the run got wrong.
"""

from __future__ import annotations

import json

from adapters.tau2.money import money


def user(methods=None, **overrides):
    if methods is None:
        methods = {
            "credit_card_4421486": {
                "source": "credit_card",
                "id": "credit_card_4421486",
                "brand": "visa",
                "last_four": "7447",
            },
            "certificate_7504069": {
                "source": "certificate",
                "id": "certificate_7504069",
                "amount": 250.0,
            },
            "gift_card_7773485": {
                "source": "gift_card",
                "id": "gift_card_7773485",
                "amount": 67.0,
            },
        }
    record = {
        "user_id": "mia_li_3668",
        "name": {"first_name": "Mia", "last_name": "Li"},
        "membership": "gold",
        "payment_methods": methods,
        "saved_passengers": [],
        "reservations": ["FQ8APE"],
    }
    return json.dumps({**record, **overrides})


def test_the_rule_task_14_and_20_broke_is_stated():
    """Task 14 paid for a booking with three certificates, task 20 with two. The
    API does not check this, which is why the policy says the agent must."""
    text = money(user())

    assert "at most one certificate, one credit card and three gift cards" in text


def test_a_change_is_told_a_certificate_cannot_pay_for_it():
    """`_payment_for_update` raises `Certificate cannot be used to update
    reservation`, and the run hit it."""
    text = money(user())

    assert "paid by ONE gift card or credit card" in text
    assert "a certificate cannot pay for one" in text


def test_the_three_terms_of_the_total_are_spelled_out():
    """Nine of the fourteen failed writes were `Payment amount does not add up`.
    On task 6 the gate insisted on $158 while the environment said $168, and the
    disagreement was over which term is charged per passenger."""
    text = money(user())

    assert "x passengers" in text
    assert "$30 x passengers if insurance is taken" in text
    assert "$50 for each bag past the free allowance (flat, not per passenger)" in text


def test_a_balance_is_shown_for_what_can_run_out():
    text = money(user())

    assert "certificate_7504069 ($250 left)" in text
    assert "gift_card_7773485 ($67 left)" in text
    assert "a gift card or certificate cannot be charged more than the balance shown" in text


def test_a_credit_card_is_named_without_a_balance_it_does_not_have():
    text = money(user())

    assert "credit_card_4421486" in text
    assert "credit_card_4421486 (" not in text


def test_cents_survive_and_whole_dollars_do_not_grow_a_tail():
    text = money(
        user(
            methods={
                "gift_card_1": {"source": "gift_card", "id": "gift_card_1", "amount": 67.5},
                "gift_card_2": {"source": "gift_card", "id": "gift_card_2", "amount": 40},
            }
        )
    )

    assert "gift_card_1 ($67.50 left)" in text
    assert "gift_card_2 ($40 left)" in text


def test_an_empty_wallet_is_an_answer_and_not_a_shape_it_did_not_recognise():
    """Every method has to be on the profile already, so an empty wallet is worth
    saying out loud rather than falling silent on."""
    text = money(user(methods={}))

    assert "on file: none" in text
    assert "nothing here that can pay for anything" in text


def test_a_balance_that_is_not_a_number_is_admitted_rather_than_printed():
    text = money(
        user(methods={"gift_card_1": {"source": "gift_card", "id": "gift_card_1", "amount": None}})
    )

    assert "gift_card_1 (balance not shown)" in text


def test_the_record_is_left_exactly_as_it_arrived():
    content = user()

    assert money(content).startswith(content)


def test_anything_that_is_not_a_profile_is_left_alone():
    for content in (
        "Error: User not found",
        "",
        "not json at all",
        json.dumps([{"flight_number": "HAT021", "prices": {"economy": 108}}]),
        json.dumps({"reservation_id": "FQ8APE", "cabin": "economy"}),
        json.dumps({"user_id": "mia_li_3668"}),
        json.dumps({"user_id": "mia_li_3668", "payment_methods": []}),
        json.dumps({"payment_methods": {"gift_card_1": {"source": "gift_card"}}}),
    ):
        assert money(content) == content


def test_the_five_notes_stack_and_a_profile_gets_only_its_own():
    from adapters.tau2.agent import _noted

    text = _noted(user())

    assert text.startswith(user())
    assert "PAYING FOR THIS" in text
    assert "CHECKED BAGS THIS USER GETS FREE" in text
    assert "WHETHER THIS RESERVATION CAN BE CANCELLED" not in text
    assert "WHAT THIS RESERVATION COSTS" not in text


def test_a_reservation_does_not_get_the_payment_block():
    """It would be paid for on fifty-seven lookups instead of fourteen, and the
    methods are not on that record anyway."""
    from adapters.tau2.agent import _noted

    reservation = json.dumps(
        {
            "reservation_id": "FQ8APE",
            "user_id": "mia_li_3668",
            "cabin": "economy",
            "flights": [{"flight_number": "HAT056", "date": "2024-05-25", "price": 100}],
            "passengers": [{"first_name": "Mia"}],
            "created_at": "2024-05-10T00:00:00",
            "total_baggages": 1,
            "nonfree_baggages": 0,
            "insurance": "no",
        }
    )

    assert "PAYING FOR THIS" not in _noted(reservation)


def test_the_payment_rules_reach_the_kernel_through_the_adapter():
    """tau2 logs the `ToolMessage` it built, not what we handed the Kernel, so a
    transcript will never show this block whether it happened or not."""
    from tau2.data_model.message import ToolMessage

    from adapters.tau2.agent import _tool_results

    message = ToolMessage(id="1", role="tool", content=user(), requestor="assistant")

    assert "at most one certificate" in (_tool_results(message) or {})["1"]
