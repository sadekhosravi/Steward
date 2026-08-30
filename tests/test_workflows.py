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
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agents.assistant import build_assistant
from agents.gate import build_gate
from agents.planner import build_planner
from core.state import Deps
from tests.tools import CANCEL as CANCEL_TOOL
from tests.tools import PLANNER
from workflows import CUSTOMER, Fact, Rule, Workflow, applicable, flat, for_policy, render, unquoted
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


def test_the_four_cancellation_conditions_are_four_separate_alternatives():
    """One rule each, so the rendering can put them under a heading that says any
    one is enough. Carried as a single rule they flatten onto one line and stop
    looking like a choice, which is the shape of every wrong refusal in the run."""
    quotes = " ".join(rule.quote for rule in CANCEL.permits)

    assert len(CANCEL.permits) == 4
    for condition in ("last 24 hrs", "cancelled by airline", "business flight", "travel insurance"):
        assert condition in quotes


def test_the_only_hard_stop_on_a_cancellation_is_a_flown_segment():
    assert len(CANCEL.blocks) == 1
    assert "already been flown" in CANCEL.blocks[0].quote


# --- how they are rendered ---------------------------------------------------


def test_nothing_applicable_renders_as_nothing():
    """The prompts interpolate this, so "" has to leave them as they were."""
    assert render([]) == ""


def test_a_rendered_workflow_names_its_section_and_its_facts():
    text = render([CANCEL_TOY])

    assert "Cancel   (policy section: Cancel flight)" in text
    assert "the reservation id  <-  the customer" in text


def test_permits_are_rendered_as_alternatives():
    """The heading is the whole point of the `permits` list existing."""
    assert "ALLOWED WHEN any single one of these holds" in render([CANCEL_TOY])


def test_a_deciding_rule_is_rendered_with_the_policy_wording_under_it():
    assert "policy: booked in the last 24 hrs" in render([CANCEL_TOY])


def test_a_bulleted_quote_keeps_its_bullets():
    """Flattened onto one line, four alternatives read as one long condition."""
    listed = Workflow(
        name="Cancel",
        section="Cancel flight",
        permits=(
            Rule(
                statement="Either will do.",
                quote="A reservation can be cancelled\nif it was booked in the last 24 hrs.",
            ),
        ),
    )

    text = render([listed])

    assert "A reservation can be cancelled\n" in text
    assert "        if it was booked in the last 24 hrs." in text


def test_a_rendering_of_a_grounded_policy_covers_every_workflow():
    airline = for_policy(POLICY)

    assert airline == ""


@needs_data
def test_the_airline_rendering_carries_all_seven_and_the_standing_rules():
    text = for_policy(airline_policy())

    for workflow in AIRLINE:
        assert f"### {workflow.name}" in text
    assert "ON EVERY REQUEST, WHATEVER IT IS" in text


@needs_data
def test_the_rendering_says_basic_economy_does_not_block_a_cabin_change():
    """The one sentence this whole part exists to put in front of both agents."""
    text = for_policy(airline_policy())

    assert "all reservations, including basic economy, can change cabin" in text


# --- and where they end up ---------------------------------------------------


def instructions_of(agent, deps: Deps | None = None) -> str:
    """What an agent is told, taken from the request rather than from the agent.

    Instructions arrive on the `ModelRequest` itself, which is the same seam
    `test_policy` reads them through.

    `deps` only for the actor: its instructions are partly callables over a
    dependency object, and the other three agents take none at all.
    """
    seen: list[str] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(m.instructions for m in messages if getattr(m, "instructions", None))
        # The actor may answer in prose; the planner and the critics may not. Both
        # shapes have to be satisfiable here, because what is being read is the
        # request rather than whatever comes back.
        if not info.output_tools:
            return ModelResponse(parts=[TextPart("ok")])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, _blank(info))])

    with agent.override(model=FunctionModel(record)):
        agent.run_sync("anything", **({"deps": deps} if deps is not None else {}))
    return "\n".join(seen)


def _blank(info: AgentInfo) -> dict[str, object]:
    """The least an output tool will accept, whichever agent is being asked."""
    required = info.output_tools[0].parameters_json_schema.get("required", [])
    return {name: "" if name != "allowed" else True for name in required}


@needs_data
def test_the_planner_is_shown_the_workflows():
    planner = build_planner([CANCEL_TOOL], airline_policy(), PLANNER)

    told = instructions_of(planner)

    assert "### Cancel a reservation" in told
    assert "ALLOWED WHEN any single one of these holds" in told


@needs_data
def test_the_gate_is_shown_the_workflows():
    gate = build_gate(airline_policy(), PLANNER)

    told = instructions_of(gate)

    assert "### Change the cabin on a reservation" in told
    assert "all reservations, including basic economy, can change cabin" in told


@needs_data
def test_the_actor_is_shown_the_workflows():
    """The node that ends up giving the refusal, and the last one to be given them."""
    actor = build_assistant([CANCEL_TOOL], airline_policy(), PLANNER)

    told = instructions_of(actor)

    assert "### Cancel a reservation" in told
    assert "ALLOWED WHEN any single one of these holds" in told


@needs_data
def test_the_actor_is_told_a_refusal_waits_for_the_record():
    """It refused seven reservations it had never read, from the conditions alone."""
    told = instructions_of(build_assistant([CANCEL_TOOL], airline_policy(), PLANNER))

    assert "BEFORE YOU SAY NO" in told
    assert "Listing the conditions back to the customer is not the same as checking them" in told


def test_a_policy_with_no_workflows_still_builds_every_agent():
    """The interpolation has to survive a domain nothing was transcribed for."""
    assert "### " not in instructions_of(build_planner([CANCEL_TOOL], POLICY, PLANNER))
    assert "### " not in instructions_of(build_gate(POLICY, PLANNER))
    told = instructions_of(build_assistant([CANCEL_TOOL], POLICY, PLANNER))
    assert "### " not in told
    # The heading goes with them. An empty one reads as an instruction to find
    # something to put under it, which is the whole reason this is its own block.
    assert "WHAT EACH REQUEST NEEDS" not in told


def test_every_write_the_domain_can_make_is_named_by_the_workflow_that_makes_it():
    """A tool the workflows never name is a tool the planner has no reason to
    believe exists. Naming the block without naming the route through it is what
    produced `update_reservation_cabin` -- a call this domain does not have,
    reasoned from the policy's "Change cabin" heading, planned 0.069 times per plan
    in run 017 and zero times in run 020 once the cabin workflow named
    `update_reservation_flights`. In the same run the actor told a customer "the
    system can't remove a passenger from an existing booking" and offered to cancel
    instead, on a task whose gold writes included `update_reservation_passengers`.
    """
    from tau2.domains.airline.environment import get_environment

    # The six calls that move the scored database. `transfer_to_human_agents` is
    # gated too but writes nothing, so its absence here would prove nothing.
    writes = {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
    real = {tool.name for tool in get_environment().get_tools()}
    assert writes <= real, sorted(writes - real)
    blob = " ".join(rule.statement for w in AIRLINE for rule in w.cited)
    assert not [name for name in sorted(writes) if name not in blob]
