"""Checks that are decided by arithmetic, and therefore not asked of a model.

Scored as a classifier over 247 write proposals that really happened, the
monolithic critic blocks 41% of the writes gold makes and 46% of the writes it
does not. Five points of separation, at one model call each. It is not choosing
badly between correct and incorrect actions so much as not distinguishing them:
a roughly uniform refusal rate applied to everything.

Read by the predicate each refusal turns on rather than by its wording, most are
facts this codebase already computes correctly somewhere else -- whether an
identifier was shown, whether a cabin permits a change, whether a segment has
flown -- and then hands to a 20B model as prose to re-derive. It gets them wrong.
Others cite rules that do not exist in the policy at all.

So the arithmetic stops being a question. A verifier is a pure function of the
proposed call and the evidence standing behind it, it returns the same answer
every time, and it can be measured against a known answer key before it is
allowed to stop anything.

WHAT A VERIFIER MAY DO

Block, and say why in words the assistant can act on. That is a real reversal:
every part shipped before this one computes a fact and *supplies* it, on the
principle that a wrong block and a wrong approval cost the same and supplying is
the safer half. The measurement that changed it is that the two are not
symmetric in frequency. Across 220 simulations the database component is exactly
`every gold write made AND no surplus write executed` -- 99 of 99 against 0 of
121 -- and 44 of those simulations made every write gold asks for and lost anyway
to a single surplus one. Supplying facts has been tried three times and surplus
writes stayed at 22 to 38 a run.

WHAT A VERIFIER MAY NOT DO

Guess. `Finding` is returned only when the evidence settles the matter against
the call. Anything a verifier cannot decide from what it was given is not its
business, and it returns None -- which is an approval only in the sense that this
particular check has nothing to say. The judges downstream may still block.

THE BAR FOR BLOCKING

A verifier ships only at zero false blocks against the 49-write answer key,
measured by `scripts/gate_bench.py`. Not "high precision" -- zero. There are 49
of them and they are all correct by definition, so any number above zero is a
defect that can be read and fixed rather than a rate to be traded off.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Describe", "Evidence", "Finding", "Panel", "Verifier", "first"]


@dataclass(frozen=True)
class Finding:
    """One check, fired, with the fix the assistant is supposed to carry out.

    `remediation` is not decoration. It is handed to the actor verbatim as its
    entire retry prompt, so a refusal that names no fix spends one of two
    revisions and leaves it exactly where it was -- which is what happened on 70
    of the 166 write refusals in the 50-task run. Written as an instruction, in
    the imperative, naming the tool or the question.

    `check` is the verifier's own name and exists so the harness can report which
    checks earn their place. A block nobody can attribute is a block nobody can
    improve.
    """

    check: str
    reason: str
    remediation: str

    recoverable: bool = False
    """Whether the assistant can carry the remediation out alone, right now.

    False by default, and the default is the common case: most of these findings
    end in "tell the customer this is not allowed", which is the turn ending
    correctly rather than something to send the actor back over. True belongs
    only to a fix that is entirely in the arguments -- send this many passengers,
    do not reduce the bag count -- because sending the actor back to ask the
    customer something loops it against a condition only the customer can clear.
    """


@dataclass(frozen=True)
class Evidence:
    """Everything a verifier is allowed to look at.

    Deliberately small, and deliberately the same three things the Kernel already
    carries, so a verifier can be run inside a live turn and replayed offline
    against a saved one without either side reshaping anything.

    `observed` is raw: tool results as the environment returned them and customer
    turns as they were typed, in the order they arrived. Parsing is the domain
    adapter's job, because knowing that a blob with `reservation_id` in it is a
    reservation is knowledge about an airline and `core` has none.

    `committed` is the tools whose writes have already gone through in this
    conversation. One policy rule turns on it -- a delay certificate is allowed
    only after the reservation has actually been changed or cancelled -- and
    there is no way to answer that from the text alone.

    `looked_up` is the same reads paired with the arguments that asked for them.
    Some of this domain's answers are unreadable without the question:
    `get_flight_status` returns the bare string `"delayed"` and nothing else, so
    which flight is delayed lives only in the call. A verifier given `observed`
    alone cannot use that result at all.
    """

    observed: tuple[str, ...] = ()
    dialogue: str = ""
    committed: tuple[str, ...] = ()
    looked_up: tuple[tuple[str, dict, str], ...] = ()

    stated: Mapping[str, Any] = field(default_factory=dict)
    """How the customer described the record, extracted from their own words.

    The one entry here that a model produced, and it is deliberately shaped as
    facts rather than a verdict: "one passenger", not "wrong reservation". The
    comparison against the record is still arithmetic, still in code, and still
    the only thing allowed to block.

    Empty is the common case and means the customer described nothing, never that
    nothing matches. A verifier reading this must treat a missing key exactly as
    it treats a record it has not been shown: silence.
    """

    @classmethod
    def of(
        cls,
        observed: list[str],
        dialogue: str = "",
        committed: list[str] | None = None,
        looked_up: list[tuple[str, dict, str]] | None = None,
        stated: Mapping[str, Any] | None = None,
    ) -> Evidence:
        return cls(
            tuple(observed),
            dialogue,
            tuple(committed or ()),
            tuple(looked_up or ()),
            dict(stated or {}),
        )


# A verifier sees one call and the evidence, and either has something to say or
# does not. Taking the call rather than a bag of arguments keeps the tool name
# available: most of these rules are about a particular tool and a verifier that
# had to be told which one separately would be one more thing to wire wrongly.
Verifier = Callable[[object, Evidence], Finding | None]

# The one thing in this file that a model produces, and it is kept behind a
# callable the domain supplies for the same reason the verifiers are: rendering a
# proposal into a question is knowledge about the domain, and `core` has none.
#
# It returns facts, never a verdict -- a mapping that becomes `Evidence.stated`
# and is then compared by an ordinary verifier. Nothing downstream can tell
# whether the mapping came from a model or from a file, which is the property that
# keeps the blocking half deterministic.
Describe = Callable[[object, Evidence], Mapping[str, Any]]


@dataclass
class Panel:
    """The verifiers that apply to a tool, in the order they should be asked.

    Order is cost, not precedence -- every verifier here is exact, so no two can
    disagree about the same call. Cheapest first only so the expensive ones are
    skipped once something has already fired.
    """

    verifiers: dict[str, list[Verifier]] = field(default_factory=dict)

    def for_tool(self, name: str) -> list[Verifier]:
        return self.verifiers.get(name, []) + self.verifiers.get("*", [])


def first(call: object, evidence: Evidence, panel: Panel) -> Finding | None:
    """The first check that fires on this call, or None if none of them does.

    Short-circuits. The assistant is given one instruction to carry out, not a
    list to triage, and a second finding on a call already refused is a fact
    about a call that is not going to happen.
    """
    for verifier in panel.for_tool(getattr(call, "name", "")):
        finding = verifier(call, evidence)
        if finding is not None:
            return finding
    return None
