"""The policy in sections, so the actor can be shown only the ones in play.

A tau2 domain policy is one document covering every procedure the domain has.
The airline one is 167 lines: sixteen of preamble, forty-six defining what a
reservation and a cabin class are, and four procedures -- book, modify, cancel,
refund -- of which any one turn needs one, occasionally two. The other three are
not merely wasted tokens. They are the near misses: a baggage allowance table
that belongs to booking, read during a modification; the four conditions that
permit a cancellation, applied to a change of cabin. Text that answers a question
nobody asked is where a small model goes to find an answer anyway.

Cost is the second reason and the sharper one. The actor's instructions are
rebuilt on *every request of a turn*, so a twenty-call turn pays for the whole
policy twenty times, while the planner beside it pays once. That asymmetry is
what makes the actor worth trimming and the planner worth leaving alone.

Splitting is on `##` headings. The preamble and the first section are standing:
across all three tau2 domains the first `##` section is the data dictionary --
`Domain Basic` in airline, `Domain basic` in retail, `Domain Basics` in telecom
-- which every procedure below it is written in terms of. It is taken by position
rather than by name precisely because those three spellings differ.

Nothing here knows an airline from a bank. The unit is a markdown heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Section", "contents", "excerpt", "preamble", "sections", "standing", "titles"]

_HEADING = re.compile(r"^## +(.+?) *$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    """One `##` section, heading included, so it can be quoted back verbatim."""

    title: str
    text: str


def sections(policy: str) -> list[Section]:
    """Every `##` section, in the order the policy states them.

    Subsections (`###`) are left inside their parent rather than split out. They
    are how a procedure is organised, not a boundary anything would be selected
    across -- nobody needs "Change cabin" without the rest of "Modify flight".
    """
    found = list(_HEADING.finditer(policy))
    starts = [match.start() for match in found]
    ends = [*starts[1:], len(policy)] if starts else []
    return [
        Section(title=match.group(1).strip(), text=policy[start:end].strip())
        for match, start, end in zip(found, starts, ends, strict=True)
    ]


def preamble(policy: str) -> str:
    """Everything before the first heading: the rules that govern every turn."""
    first = _HEADING.search(policy)
    return (policy[: first.start()] if first else policy).strip()


def standing(policy: str) -> str:
    """What the actor is shown whatever the turn is about.

    The preamble carries the rules with no procedure attached -- confirm before
    writing, one tool call at a time, do not invent policy -- and dropping any of
    them on a turn that did not seem to need them is how an agent stops asking
    for confirmation. The first section is the vocabulary the rest is written in.
    """
    body = sections(policy)
    return "\n\n".join(part for part in (preamble(policy), body[0].text if body else "") if part)


def selectable(policy: str) -> list[Section]:
    """The procedure sections: everything the planner may choose between."""
    return sections(policy)[1:]


def titles(policy: str) -> list[str]:
    """The names of the selectable sections, spelled as the policy spells them."""
    return [section.title for section in selectable(policy)]


def contents(policy: str) -> str:
    """The selectable sections as a list, for whoever has to name one."""
    return "\n".join(f"- {title}" for title in titles(policy))


def excerpt(policy: str, chosen: list[str]) -> str:
    """The named sections in the policy's own order, or all of them.

    Falls **open**: a turn nothing matched is shown the whole policy rather than
    none of it. Selection is an economy, and an economy that can silently remove
    the rule an action needed is not one worth having -- so the failure mode is
    the prompt this replaces, which is a cost and never a wrong answer.

    Matching is case-insensitive on the title alone. The planner is copying a
    heading out of a list it was given, and the way that goes wrong is capitals
    and stray whitespace, not a different section.
    """
    wanted = {title.strip().casefold() for title in chosen}
    picked = [section for section in selectable(policy) if section.title.casefold() in wanted]
    return "\n\n".join(section.text for section in picked or selectable(policy))
