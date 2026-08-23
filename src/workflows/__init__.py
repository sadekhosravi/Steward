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
    "PREAMBLE",
    "Fact",
    "Rule",
    "Workflow",
    "applicable",
    "flat",
    "for_policy",
    "render",
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


PREAMBLE = """
HOW EACH REQUEST IS HANDLED

Below is every request this policy covers, as the steps it actually takes. They
are copied from the policy, not a summary of it, and where one contradicts what
you would otherwise assume, these are right.

Read an entry like this:

- NEEDS is what has to be known before the request can be answered at all, and
  the one place each fact comes from. A fact you have not got is a lookup to make
  or a question to ask -- it is never something to assume, and it is never a
  reason to refuse.
- NEVER is what the policy forbids outright. Each one is decided by a fact from
  NEEDS. If you do not have that fact, you have not established the block.
- ALLOWED WHEN lists conditions of which **any single one is enough**. They are
  alternatives, not requirements. Needing all of them would refuse almost
  everything.
- ALSO is what constrains the call once it is allowed.

A rule appearing under one request says nothing about any other request, even
when both touch the same reservation.
""".strip()


def render(workflows: Iterable[Workflow], standing: Iterable[Rule] = ()) -> str:
    """The workflows as an agent is shown them, or "" when there are none.

    The `quote` is printed under `blocks` and `permits` and nowhere else. Those
    are the two places a decision is actually made, and where the run shows a
    prior belief overriding the document -- so the document gets the last word in
    its own characters. Everywhere else the statement is enough and the policy is
    already in the same prompt, a second copy of it being the thing this module
    is otherwise careful not to be.
    """
    workflows = list(workflows)
    if not workflows:
        return ""
    parts = [PREAMBLE]
    standing = list(standing)
    if standing:
        parts.append(
            "\n".join(["ON EVERY REQUEST, WHATEVER IT IS", *(_quoted(r) for r in standing)])
        )
    parts.extend(_one(workflow) for workflow in workflows)
    return "\n\n".join(parts)


def _one(workflow: Workflow) -> str:
    lines = [f"### {workflow.name}   (policy section: {workflow.section})"]
    if workflow.facts:
        lines.append("NEEDS")
        lines += [f"  - {fact.name}  <-  {fact.source}" for fact in workflow.facts]
    if workflow.blocks:
        lines.append("NEVER")
        lines += [_quoted(rule) for rule in workflow.blocks]
    if workflow.permits:
        lines.append("ALLOWED WHEN any single one of these holds")
        lines += [_quoted(rule) for rule in workflow.permits]
    if workflow.rules:
        lines.append("ALSO")
        lines += [f"  - {flat(rule.statement)}" for rule in workflow.rules]
    return "\n".join(lines)


def _quoted(rule: Rule) -> str:
    """A rule that decides something, with the policy's own words under it.

    The quote keeps its own line breaks. A policy that states a condition as a
    bulleted list means the list, and flattening it onto one line is how four
    alternatives come to read as one long conjunction.
    """
    lines = [flat(line) for line in rule.quote.splitlines() if line.strip()]
    if len(lines) == 1:
        return f"  - {flat(rule.statement)}\n      policy: {lines[0]}"
    body = "\n".join(f"        {line}" for line in lines)
    return f"  - {flat(rule.statement)}\n      policy:\n{body}"


def for_policy(policy: str) -> str:
    """The workflows this policy backs, rendered and ready to interpolate.

    One place knows which domains exist, and it is this function. Returns "" for
    a policy nothing is grounded in -- a domain we have not transcribed, or a
    vendored policy that moved -- and the prompts are written so that an empty
    string leaves them as they were before any of this existed.
    """
    from workflows import airline

    kept = applicable(airline.AIRLINE, policy)
    if not kept:
        return ""
    body = flat(policy)
    return render(kept, [rule for rule in airline.STANDING if flat(rule.quote) in body])
