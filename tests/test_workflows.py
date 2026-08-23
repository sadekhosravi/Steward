"""WORKFLOWS: that every rule can show its source, and that two of them say
the thing the run got wrong.

The transcription is only worth having if it cannot drift from the policy, so
the load-bearing test here is the dull one: every quote, verbatim, in the real
document. It runs against the vendored policy rather than a fixture, because a
fixture would only prove the transcription matches itself.

The last two tests are regression locks on the specific misreadings that cost
the run -- a cabin change refused for basic economy, and a cancellation refused
for it. Both read as though they were tidying an inconsistency.
"""

from __future__ import annotations

import os
import pathlib

import pytest
from dotenv import find_dotenv, load_dotenv

from workflows import CUSTOMER, Fact, Rule, Workflow, applicable, flat, unquoted
from workflows.airline import (
    AIRLINE,
    CANCEL,
    CHANGE_CABIN,
    CHANGE_FLIGHTS,
    STANDING,
)

POLICY = """
# Toy Policy

Confirm before you change anything.

## Domain Basic

A reservation has a cabin.

## Cancel flight

A reservation can be cancelled if it was booked in the last 24 hrs.
"""

CANCEL_TOY = Workflow(
    name="Cancel",
    section="Cancel flight",
    facts=(Fact(name="the reservation id", source=CUSTOMER),),
    permits=(
        Rule(
            statement="Booked recently enough.",
            quote="booked in the last 24 hrs",
        ),
    ),
)


# --- the check itself --------------------------------------------------------


def test_a_quote_the_policy_contains_is_grounded():
    assert unquoted(CANCEL_TOY, POLICY) == []


def test_a_quote_the_policy_does_not_contain_is_reported():
    """The failure the module exists to prevent: a rule written from memory."""
    invented = Rule(statement="Basic economy cannot be cancelled.", quote="basic economy")
    made_up = Workflow(name="Cancel", section="Cancel flight", blocks=(invented,))

    assert unquoted(made_up, POLICY) == ["basic economy"]


def test_line_wrapping_and_trailing_spaces_do_not_break_a_quote():
    """tau2 policies wrap their bullet lists and leave spaces after headings."""
    wrapped = Rule(statement="Booked recently.", quote="booked in the\n   last 24 hrs")

    assert unquoted(Workflow(name="c", section="Cancel flight", rules=(wrapped,)), POLICY) == []


def test_flat_reduces_any_whitespace_to_single_spaces():
    assert flat("  two\n\n  lines\t") == "two lines"


# --- what survives selection -------------------------------------------------


def test_a_grounded_workflow_is_applicable():
    assert applicable([CANCEL_TOY], POLICY) == [CANCEL_TOY]


def test_a_workflow_whose_rule_is_not_in_the_policy_is_dropped():
    """Fails closed. There is no safe fallback for a confident wrong instruction."""
    invented = Rule(statement="Anything.", quote="a sentence this policy does not have")
    made_up = Workflow(name="Cancel", section="Cancel flight", blocks=(invented,))

    assert applicable([made_up], POLICY) == []


def test_a_workflow_naming_a_section_the_policy_lacks_is_dropped():
    """How a domain's workflows are kept out of another domain's policy."""
    elsewhere = Workflow(
        name="Book",
        section="Book flight",
        rules=(Rule(statement="Confirm first.", quote="Confirm before you change anything."),),
    )

    assert applicable([elsewhere], POLICY) == []


def test_selection_keeps_the_order_it_was_given():
    assert [w.name for w in applicable([CANCEL_TOY, CANCEL_TOY], POLICY)] == ["Cancel", "Cancel"]


# --- the transcription, against the document it was copied from --------------


# Where the vendored policy is. Read here rather than left to whichever test
# module happened to import tau2 first and pull the environment in with it.
load_dotenv(find_dotenv(usecwd=True))


def airline_policy() -> str:
    root = pathlib.Path(os.environ["TAU2_DATA_DIR"])
    return (root / "tau2/domains/airline/policy.md").read_text(encoding="utf-8")


needs_data = pytest.mark.skipif(
    not os.environ.get("TAU2_DATA_DIR"), reason="benchmark data not fetched"
)


@needs_data
@pytest.mark.parametrize("workflow", AIRLINE, ids=lambda w: w.name)
def test_every_rule_quotes_the_airline_policy_verbatim(workflow):
    assert unquoted(workflow, airline_policy()) == []


@needs_data
def test_the_standing_rules_quote_the_airline_policy_verbatim():
    standing = Workflow(name="standing", section="Book flight", rules=STANDING)

    assert unquoted(standing, airline_policy()) == []


@needs_data
def test_the_whole_airline_set_is_applicable_to_its_own_policy():
    assert len(applicable(AIRLINE, airline_policy())) == len(AIRLINE)


@needs_data
def test_no_airline_workflow_applies_to_another_domains_policy():
    """The grounding check is also what keeps retail's policy from getting these."""
    root = pathlib.Path(os.environ["TAU2_DATA_DIR"])
    retail = (root / "tau2/domains/retail/policy.md").read_text(encoding="utf-8")

    assert applicable(AIRLINE, retail) == []


@needs_data
def test_every_tool_named_as_a_source_exists_in_the_domain():
    """A fact whose source is a misspelled tool sends the planner nowhere."""
    from tau2.domains.airline.environment import get_environment

    known = {tool.name for tool in get_environment().get_tools()}
    named = {
        word.strip(" ,.")
        for workflow in AIRLINE
        for fact in workflow.facts
        for word in fact.source.replace(" or ", " ").split()
        if "_" in word
    }

    assert named <= known, f"not tools in this domain: {sorted(named - known)}"


def test_every_fact_says_where_it_comes_from():
    for workflow in AIRLINE:
        for fact in workflow.facts:
            assert fact.source.strip(), f"{workflow.name}: {fact.name} has no source"


# --- the two rules the run read backwards ------------------------------------


def test_a_cabin_change_is_permitted_on_basic_economy():
    """Task 7. The agent refused the upgrade and the cancellation that followed
    from it; gold does both. `Basic economy flights cannot be modified` belongs
    to the flights procedure, and only to it."""
    permitted = " ".join(rule.quote for rule in CHANGE_CABIN.permits)

    assert "including basic economy" in permitted
    assert not any("Basic economy" in rule.quote for rule in CHANGE_CABIN.blocks)


def test_basic_economy_blocks_a_flight_change_and_nothing_else():
    blocking = [rule for rule in CHANGE_FLIGHTS.blocks if "Basic economy" in rule.quote]

    assert len(blocking) == 1


def test_basic_economy_is_not_a_reason_to_refuse_a_cancellation():
    """Task 39. Three cancellations refused, all three permitted, and the
    reservation record was never read."""
    assert not any("economy" in rule.quote.lower() for rule in CANCEL.blocks)
    assert not any("economy" in rule.quote.lower() for rule in CANCEL.permits)


def test_the_four_cancellation_conditions_are_carried_as_one_any_of_rule():
    """Read as a conjunction they permit almost nothing, which is the shape of
    every wrong refusal in the run."""
    assert len(CANCEL.permits) == 1
    for condition in ("last 24 hrs", "cancelled by airline", "business flight", "travel insurance"):
        assert condition in CANCEL.permits[0].quote


def test_the_only_hard_stop_on_a_cancellation_is_a_flown_segment():
    assert len(CANCEL.blocks) == 1
    assert "already been flown" in CANCEL.blocks[0].quote
