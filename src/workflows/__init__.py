"""WORKFLOWS: the policy as branches, each one carrying the line it came from.

The planner already runs a workflow. It runs the wrong one. On task 7 it decided
a basic-economy reservation could neither be upgraded nor cancelled and handed
the customer to a human; the policy permits both, and the two rules it conflated
sit one line apart:

    Change flights:  Basic economy flights cannot be modified.
    Change cabin:    In other cases, all reservations, including basic economy,
                     can change cabin without changing the flights.

Twelve of the gate's forty-five write refusals cite basic economy. The belief is
not occasional and it is not the actor's -- it is held by both reviewing agents,
consistently, across the run. Writing the procedure down is what turns a wrong
belief nobody can see into an artifact somebody can read.

Which is also the danger. A hand-written workflow that misstates the policy stops
being an occasional mistake and becomes a guaranteed one, on every task that
touches it. So a rule here is not a claim -- it is a `quote` from the policy plus
a `statement` of what it means for us, and `unquoted` checks that the quote is
still in the document, verbatim. A rule that cannot show its source does not
survive `applicable`, and the check is on the live policy text at wire-up rather
than only in a test, because the policy travels with the benchmark data and we
pin that by revision.

Comparison is on flattened whitespace. The tau2 policies carry trailing spaces
after several headings ("Cabin: ", "Payment: ") and wrap their bullet lists, and
neither is a difference worth failing over.

Nothing in this module knows an airline from a bank. The domain sets live beside
it, one module each.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.policy import titles

__all__ = [
    "CUSTOMER",
    "Fact",
    "Rule",
    "Workflow",
    "applicable",
    "flat",
    "unquoted",
]


# Where a fact comes from when no tool returns it. Half of what a procedure needs
# is only knowable by asking -- the reason for a cancellation, whether insurance
# is wanted, which of three reservations was meant -- and a workflow that does not
# say so reads as though every fact were a lookup away.
CUSTOMER = "the customer"


@dataclass(frozen=True)
class Fact:
    """Something that has to be known, and the one place it can be got from.

    The source is the whole point. Task 39 refused three cancellations that the
    policy permits, having never called `get_reservation_details` -- the cabin,
    the purchase time and the insurance flag that decide the question are all on
    the reservation, and none of them were ever read. A branch that names its
    evidence cannot be evaluated before the evidence exists.
    """

    name: str
    source: str


@dataclass(frozen=True)
class Rule:
    """One line of the policy, and what it means for a call we are about to make.

    `statement` is ours and may be paraphrase. `quote` is the policy's, verbatim,
    and is what `unquoted` checks. Keeping both means the prompt can be written
    for a small model while the audit stays against the source document.
    """

    statement: str
    quote: str


@dataclass(frozen=True)
class Workflow:
    """One thing a customer can ask for, and what the policy says about it.

    Split finer than the policy's own headings where the headings group rules
    that do not share a subject. `## Modify flight` covers four procedures with
    four different answers for basic economy, and conflating them is the single
    most expensive mistake in the run -- so it becomes four workflows, all citing
    that one section.

    The three rule lists are separated because they are consulted at different
    moments. `blocks` is asked first and ends the matter. `permits` is asked next
    and **any one of them suffices** -- that shape appears twice in the airline
    policy and reading it as a conjunction is how a permitted cancellation gets
    refused. `rules` constrains the arguments of a call already established to be
    allowed.
    """

    name: str
    section: str
    facts: tuple[Fact, ...] = ()
    blocks: tuple[Rule, ...] = ()
    permits: tuple[Rule, ...] = ()
    rules: tuple[Rule, ...] = ()

    @property
    def cited(self) -> tuple[Rule, ...]:
        """Every rule, whatever it is consulted for."""
        return self.blocks + self.permits + self.rules


def flat(text: str) -> str:
    """One line, single-spaced. What both sides of a quote check are reduced to."""
    return " ".join(text.split())


def unquoted(workflow: Workflow, policy: str) -> list[str]:
    """Quotes this workflow claims that the policy does not contain.

    Empty is the only acceptable answer. A non-empty one means either the policy
    moved under us -- it is vendored data, pinned by revision, and bumping that
    revision is how it would -- or a rule was written from memory instead of
    copied, which is the failure this whole module exists to prevent.
    """
    body = flat(policy)
    return [rule.quote for rule in workflow.cited if flat(rule.quote) not in body]


def applicable(candidates: Iterable[Workflow], policy: str) -> list[Workflow]:
    """The workflows this policy actually backs, in the order given.

    Fails **closed**, unlike `policy.excerpt` beside it. An excerpt that selects
    nothing falls back to the whole policy, which is a cost; a workflow whose
    rules are not in the policy is a confident instruction to do the wrong thing,
    and there is no safe fallback for that. Dropping it leaves the agent reading
    the policy directly, which is where it was before any of this.

    Selection by grounding rather than by domain name is deliberate: the tau2
    agent is handed `domain_policy` as text and never told which domain it is.
    """
    kept = []
    for workflow in candidates:
        if unquoted(workflow, policy):
            continue
        if workflow.section not in titles(policy):
            continue
        kept.append(workflow)
    return kept
