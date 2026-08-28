"""Turning one airline proposal into the question the extractor is asked.

`agents.selector` is domain-free: it copies out how a customer described a record
and knows nothing about reservations. `core.kernel` is domain-free too, and holds
a `Describe` callable it never looks inside. This is the seam between them, and
it is the only place that knows both a cabin and a prompt.

WHAT THE EXTRACTOR IS GIVEN, AND WHAT IT IS NOT

The conversation, the call as one line, and what the call does. That is all.

It is not given the record. `means()` normally prints a diff against the
reservation and here it must not, because the extractor's answer is about to be
compared against that same record -- a model that has read the record will
describe the record, every comparison will match, and the check becomes an
expensive way to approve everything. So the meaning is taken from the catalogue
alone, with `observed` empty, and the diff is deliberately thrown away.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic_ai.models import Model

from core.verifiers import Describe, Evidence

from .context import exchange, means, spelled

__all__ = ["asking", "describing"]


def asking(call, evidence: Evidence) -> tuple[str, str, str]:
    """The three strings the extractor is asked about: dialogue, call, meaning.

    Pure, and separate from the agent that consumes it, so the one property this
    module exists to hold can be tested without a provider anywhere near it:
    `means` is given an empty ledger, so it returns the catalogue sentence and
    never the diff against the reservation.
    """
    name = getattr(call, "name", "")
    arguments = getattr(call, "arguments", {}) or {}
    return (
        exchange(evidence.dialogue),
        spelled(name, arguments),
        # Empty `observed` on purpose. See the module docstring -- showing the
        # record here would let the extractor read its answer off the very thing
        # its answer is about to be compared against.
        means(name, arguments, ()),
    )


def describing(model: str | Model | None = None, selector: object | None = None) -> Describe:
    """One extractor, bound once, asked per proposal.

    Built here rather than in the Kernel so that a domain with no such question
    simply passes nothing and the whole stage disappears -- which is also how the
    tests run every other verifier without a provider anywhere near them.
    """
    from agents.selector import build_selector, described

    agent = selector if selector is not None else build_selector(model)

    def describe(call, evidence: Evidence) -> Mapping[str, Any]:
        return described(agent, *asking(call, evidence))

    return describe
