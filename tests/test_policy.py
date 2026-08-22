"""POLICY: what the actor is shown of it, and what it is spared.

No network. The planner and the actor are both scripted stand-ins, so these
assert on which text reached the model rather than on its opinion of that text.

The fixture below is shaped like a tau2 policy rather than like a minimal
example, on purpose: a preamble carrying a rule with no procedure attached, a
first section that is a data dictionary with a `###` inside it, and three
procedures for a turn to pick between. Each of those shapes is load-bearing
somewhere here.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from core.kernel import Kernel
from core.policy import contents, excerpt, preamble, sections, standing, titles
from tests.tools import LOOKUP

POLICY = """
# Airline Agent Policy

Before you change anything you must obtain explicit confirmation.

## Domain Basic

### Reservation
Each reservation has a reservation id and a cabin class.

## Book flight

Each extra baggage is 47 dollars.

## Cancel flight

Flights can be cancelled within 24 hours of booking.

## Refunds and Compensation

Compensation is 100 dollars per passenger.
""".strip()

CONFIRMATION = "obtain explicit confirmation"
DICTIONARY = "reservation id and a cabin class"
# Deliberately not a round number, and deliberately not 50: the actor's standing
# instructions carry a worked example that prices a bag at 50 dollars, so a
# sentinel of "50 dollars" is found in every prompt whatever the turn was routed
# to, and the test that matters here passes for the wrong reason.
BAGGAGE = "47 dollars"
CANCELLING = "within 24 hours"
COMPENSATION = "100 dollars per passenger"


# --- scripted models --------------------------------------------------------


def _plans(named: list[str]):
    """A planner that routes the turn to exactly the sections it is handed."""

    def planner(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = {"goal": "Help them.", "policy_sections": named}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])

    return planner


def _refuses_to_plan(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Answers in prose rather than calling the output tool, until the run gives up."""
    return ModelResponse(parts=[TextPart("I would rather not.")])


def _records_instructions(seen: list[str]):
    """Capture what the actor was told, then end the turn without doing anything.

    Instructions arrive on the request itself rather than among its parts, which
    is the whole difference between them and a system prompt.
    """

    def actor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(m.instructions for m in messages if getattr(m, "instructions", None))
        return ModelResponse(parts=[TextPart("Hello.")])

    return actor


def _instructions_for(named: list[str] | None) -> str:
    """Everything the actor was told on one turn, with the turn routed as given.

    `None` scripts a planner that never answers, which is the fail-open path.
    """
    seen: list[str] = []
    kernel = Kernel(
        [LOOKUP],
        policy=POLICY,
        model=FunctionModel(_records_instructions(seen)),
        planner_model=FunctionModel(_refuses_to_plan if named is None else _plans(named)),
    )
    kernel.send(kernel.new_thread(), "I need to cancel.")
    return "\n".join(seen)


# --- splitting --------------------------------------------------------------


def test_a_policy_is_split_on_its_top_level_headings():
    assert [section.title for section in sections(POLICY)] == [
        "Domain Basic",
        "Book flight",
        "Cancel flight",
        "Refunds and Compensation",
    ]


def test_a_subsection_stays_inside_the_procedure_it_belongs_to():
    """`###` is how a procedure is organised, not a boundary anything is chosen across."""
    first, *_ = sections(POLICY)

    assert "### Reservation" in first.text
    assert DICTIONARY in first.text


def test_a_section_carries_its_own_heading():
    """So it can be quoted back looking like the document it came from."""
    assert sections(POLICY)[1].text.startswith("## Book flight")


def test_the_preamble_is_everything_before_the_first_heading():
    assert CONFIRMATION in preamble(POLICY)
    assert BAGGAGE not in preamble(POLICY)


# --- what always holds ------------------------------------------------------


def test_the_standing_policy_is_the_preamble_and_the_data_dictionary():
    """A rule with no procedure attached governs every turn, and the vocabulary
    the procedures are written in is needed to read any of them."""
    always = standing(POLICY)

    assert CONFIRMATION in always
    assert DICTIONARY in always
    assert BAGGAGE not in always
    assert CANCELLING not in always


def test_the_data_dictionary_is_not_offered_for_selection():
    """It is already standing. Offering it invites a plan that names it and
    nothing else, which would route a cancellation to no procedure at all."""
    assert titles(POLICY) == ["Book flight", "Cancel flight", "Refunds and Compensation"]
    assert "Domain Basic" not in contents(POLICY)


def test_a_policy_with_no_headings_is_all_standing():
    """Nothing to select between is not an error, and must not quietly empty the prompt."""
    flat = "Be helpful. Do not lie."

    assert standing(flat) == flat
    assert titles(flat) == []
    assert excerpt(flat, []) == ""


# --- what is shown for one turn ---------------------------------------------


def test_an_excerpt_holds_the_named_sections_and_no_others():
    shown = excerpt(POLICY, ["Cancel flight"])

    assert CANCELLING in shown
    assert BAGGAGE not in shown
    assert COMPENSATION not in shown


def test_sections_come_back_in_the_order_the_policy_states_them():
    """The document was written in an order. Reordering it to match the plan reads
    as an argument that the new order means something, which it does not."""
    shown = excerpt(POLICY, ["Refunds and Compensation", "Book flight"])

    assert shown.index(BAGGAGE) < shown.index(COMPENSATION)


def test_a_heading_is_matched_however_it_was_copied():
    """The planner is copying out of a list it was given, so what goes wrong is
    capitals and stray whitespace, not a different section."""
    assert CANCELLING in excerpt(POLICY, ["  cancel FLIGHT "])


def test_naming_nothing_shows_everything():
    """Selection is an economy. An economy that can quietly remove the rule an
    action needed is not one worth having, so it falls open."""
    shown = excerpt(POLICY, [])

    assert BAGGAGE in shown
    assert CANCELLING in shown
    assert COMPENSATION in shown


def test_naming_only_sections_that_do_not_exist_shows_everything():
    assert CANCELLING in excerpt(POLICY, ["Baggage Policy"])


def test_one_name_that_missed_does_not_cost_the_one_that_landed():
    """Falling open on a partial match would undo the whole saving on one typo."""
    shown = excerpt(POLICY, ["Cancel flight", "Baggage Policy"])

    assert CANCELLING in shown
    assert BAGGAGE not in shown


# --- through the Kernel -----------------------------------------------------


def test_the_actor_is_shown_the_sections_the_planner_named():
    assert CANCELLING in _instructions_for(["Cancel flight"])


def test_the_actor_is_not_shown_the_sections_the_planner_left_out():
    """The point of the exercise. A baggage table that belongs to booking is not a
    wasted token during a cancellation, it is a near miss to reach for."""
    told = _instructions_for(["Cancel flight"])

    assert BAGGAGE not in told
    assert COMPENSATION not in told


def test_the_standing_rules_reach_the_actor_whatever_the_turn_is_about():
    told = _instructions_for(["Cancel flight"])

    assert CONFIRMATION in told
    assert DICTIONARY in told


def test_the_actor_is_told_which_sections_exist_that_it_cannot_see():
    """So a turn that turns out to belong somewhere else ends in a question to the
    customer rather than in an invented rule or a transfer."""
    assert "Refunds and Compensation" in _instructions_for(["Cancel flight"])


def test_a_planner_that_never_answers_leaves_the_actor_the_whole_policy():
    """Fails open, in the strong sense: the prompt this node built before any of it
    was selective is the one a failure falls back to."""
    told = _instructions_for(None)

    assert BAGGAGE in told
    assert CANCELLING in told
    assert COMPENSATION in told
