"""Score a gate as a binary classifier, offline, against proposals that really happened.

The benchmark cannot referee work on the gate. Two runs of one configuration came
back 0.417 and 0.458 with eight of twenty-three tasks flipping between trials
*inside a single run*, so nothing we can build is large enough to see there. Every
gate design measured through the benchmark this far has been measured through
noise.

So the gate stops being scored by the benchmark and starts being scored the way a
classifier is: against a fixed corpus with known labels, offline, in minutes, with
no user simulator and no sampling error at all.

WHAT THE LABELS MEAN, AND WHY THEY ARE THE WHOLE SCORE

Across 220 saved simulations the database component is exactly determined by two
facts about the writes a run emitted:

    every gold write made, and no surplus write executed  ->  99 of 99 passed
    anything else                                         ->   0 of 121 passed

There are no other cases. A write that gold also makes must go through; a write
gold does not make must not. That is the entire job, and it makes `gold` and
`surplus` the only two labels this corpus needs.

    gold     the benchmark's own answer key makes this call. Blocking it is a defect.
    surplus  no gold action matches it. Letting it through loses the task.

Both errors cost exactly one task, so neither rate can be read alone -- a gate
that approves everything scores a perfect zero on false blocks, and one that
blocks everything catches all the surplus. Both numbers are always reported.

WHERE THE PROPOSALS COME FROM

Two sources, deliberately different in what they can show.

`--source runs` replays writes real runs emitted, with the evidence that stood
behind them at that moment. This is the only source that has surplus writes in it,
so it is the only one that can measure catching them.

`--source key` replays the 49 writes gold itself makes, through a live
environment, using the machinery in `gate_gold.py`. Every one is correct by
definition, so the false-block rate here is exact rather than sampled -- and the
context is the most favourable a gate could ever be given, which makes a refusal
here unambiguous.

    uv run python scripts/gate_bench.py --gate monolith
    uv run python scripts/gate_bench.py --gate none --source runs
    uv run python scripts/gate_bench.py --gate monolith --source key --limit 20

A gate is a callable `(Proposal) -> Decision`. Adding one means writing that
function and naming it in `GATES`; nothing else here changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

if "NVIDIA_API_KEY" in os.environ:
    os.environ.setdefault("NVIDIA_NIM_API_KEY", os.environ["NVIDIA_API_KEY"])

WRITES = {
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "cancel_reservation",
    "send_certificate",
}

# The runs this corpus is built from. All four ran the same shipped agent; three
# had the critic off and one had it on, which does not matter here -- what is
# being harvested is the proposals, not the verdicts anyone reached about them.
RUNS = ("gate_off_50", "part4a_50", "part4_final_50x3", "gate_on_50")

SIMULATIONS = "vendor/tau2-data/data/simulations/{name}.json/results.json"


@dataclass
class Proposal:
    """One write, and everything that stood behind it when it was proposed."""

    run: str
    task: str
    label: str  # "gold" or "surplus"
    name: str
    arguments: dict[str, Any]

    # The evidence ledger: every tool result and customer turn seen so far, as
    # raw text. This is what `state.invented`, `sources` and `mispriced` read, and
    # what the Kernel itself carries in `state.observed`.
    observed: list[str] = field(default_factory=list)

    # The same history rendered for a reader, used by checks that need to know
    # what was *said* rather than what was returned.
    dialogue: str = ""

    # pydantic-ai message history, for a gate that wants the conversation in the
    # shape a model is normally given it.
    messages: list[Any] = field(default_factory=list)

    # The reads paired with the arguments that asked for them, and the writes
    # already committed. Both are things `Evidence` needs and neither can be
    # recovered from `observed` alone -- see `core.verifiers.Evidence`.
    looked_up: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)

    # What the extractor said about this proposal, when a gate ran one. Empty for
    # every gate that does not, and carried only so `--out` can report it.
    stated: dict[str, Any] = field(default_factory=dict)
    committed: list[str] = field(default_factory=list)

    def evidence(self):
        from core.verifiers import Evidence

        return Evidence.of(self.observed, self.dialogue, self.committed, self.looked_up)

    @property
    def one_line(self) -> str:
        listed = ", ".join(f"{key}={value!r}" for key, value in self.arguments.items())
        return f"{self.name}({listed})"


@dataclass
class Decision:
    """What a gate did with one proposal.

    `check` names the thing that fired, so the report can say which check earns
    its place and which only adds cost. A gate that cannot attribute its refusal
    returns the empty string and is scored on the totals alone.
    """

    blocked: bool
    check: str = ""
    calls: int = 0


Gate = Callable[[Proposal], Decision]


# ----------------------------------------------------------------- the corpus


def gold_writes() -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Per task, the writes the benchmark's answer key makes.

    Matched by name and arguments exactly. tau2 supports a looser comparison
    through `compare_args`, but no airline action sets it, so exact match is what
    the benchmark itself is doing.
    """
    from tau2.domains.airline.environment import get_tasks

    answer = {}
    for task in get_tasks():
        criteria = task.evaluation_criteria
        actions = (criteria.actions if criteria else None) or []
        answer[task.id] = [(a.name, dict(a.arguments or {})) for a in actions if a.name in WRITES]
    return answer


def harvest(run: str, answer: dict[str, list[tuple[str, dict[str, Any]]]]) -> Iterator[Proposal]:
    """Every write a saved run executed, labelled, with the evidence of its moment.

    The evidence has to be rebuilt in message order rather than collected up
    front: a gate judging the first write of a task must not be shown what the
    third lookup returned. Walking the transcript once and yielding as we pass
    each write is the only way to get that right, and it is why this cannot be a
    comprehension over `tool_calls`.

    Errored writes are skipped. tau2's replay ignores a call the environment
    refused, so it never reaches the scored database and is neither a defect to
    catch nor a success to protect -- counting them would flatter or damn a gate
    for work that had no consequence.
    """
    path = SIMULATIONS.format(name=run)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        simulations = json.load(handle)["simulations"]

    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolReturnPart,
        UserPromptPart,
    )

    for simulation in simulations:
        wanted = list(answer.get(simulation["task_id"], []))
        observed: list[str] = []
        dialogue: list[str] = []
        history: list[Any] = []
        pending: dict[str, dict] = {}
        looked_up: list[tuple[str, dict[str, Any], str]] = []
        committed: list[str] = []

        for message in simulation.get("messages") or []:
            role, content = message.get("role"), message.get("content") or ""
            if role == "user":
                observed.append(content)
                dialogue.append(f"Customer: {content}")
                history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
            elif role == "assistant" and content:
                dialogue.append(f"Assistant: {content}")
                history.append(ModelResponse(parts=[TextPart(content)]))
            for call in message.get("tool_calls") or []:
                pending[call.get("id")] = call
                # A write is held back until after its proposal is yielded. It is
                # the thing being judged, and a transcript that already contains
                # it shows the gate an action it is about to be asked to permit
                # as one the assistant has taken -- which is the single easiest
                # way to build a harness that answers its own question.
                if call.get("name") not in WRITES:
                    history.append(_called(call))
                    dialogue.append(f"Assistant looks up: {call.get('name')}(...)")

            if role != "tool" or message.get("id") not in pending:
                continue
            call = pending[message["id"]]
            arguments = call.get("arguments") or {}
            if call.get("name") in WRITES and not message.get("error"):
                # Consumed on match, so a task whose answer key calls one tool
                # twice on one record -- task 32 does -- does not let a single
                # emitted call discharge both.
                hit = next(
                    (
                        index
                        for index, (name, args) in enumerate(wanted)
                        if name == call["name"] and args == arguments
                    ),
                    None,
                )
                if hit is not None:
                    wanted.pop(hit)
                yield Proposal(
                    run=run,
                    task=simulation["task_id"],
                    label="gold" if hit is not None else "surplus",
                    name=call["name"],
                    arguments=arguments,
                    observed=list(observed),
                    dialogue="\n".join(dialogue),
                    messages=list(history),
                    looked_up=list(looked_up),
                    committed=list(committed),
                )
                # Now it has been judged, it becomes part of the record the next
                # proposal on this task is judged against.
                history.append(_called(call))
                dialogue.append(f"Assistant calls: {call['name']}(...)")
                committed.append(call["name"])
            if not message.get("error"):
                observed.append(content)
                dialogue.append(f"Result: {content}")
                if call["name"] not in WRITES:
                    looked_up.append((call["name"], arguments, content))
                history.append(
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name=call["name"],
                                content=content,
                                tool_call_id=message["id"],
                            )
                        ]
                    )
                )


def _called(call: dict) -> Any:
    """One tool call as the model's own turn, so `gate.transcript` renders it."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    return ModelResponse(
        parts=[ToolCallPart(call["name"], call.get("arguments") or {}, tool_call_id=call.get("id"))]
    )


def from_answer_key(tasks: list[str] | None) -> Iterator[Proposal]:
    """The 49 writes gold makes, replayed through a live environment.

    Reuses `gate_gold` wholesale -- `look_up` for the reads gold does not bother
    to record, `history` for the transcript including the confirmation turn the
    policy requires. Those took five corrections to get right and re-deriving
    them here would only get them wrong again differently.
    """
    import gate_gold
    from tau2.domains.airline.environment import get_environment, get_tasks

    for task in get_tasks():
        if tasks and task.id not in tasks:
            continue
        criteria = task.evaluation_criteria
        actions = (criteria.actions if criteria else None) or []
        if not any(action.name in WRITES for action in actions):
            continue

        environment = get_environment()
        instructions = str(task.user_scenario.instructions)
        observed: list[str] = [instructions]
        seen: list[tuple[str, dict[str, Any], str]] = []
        committed: list[str] = []

        for action in actions:
            arguments = dict(action.arguments or {})
            if action.name in WRITES:
                gate_gold.look_up(environment, arguments, observed, seen)
                proposal = Proposal(
                    run="answer-key",
                    task=task.id,
                    label="gold",
                    name=action.name,
                    arguments=arguments,
                    observed=list(observed),
                    dialogue=_rendered(instructions, seen),
                    looked_up=list(seen),
                    committed=list(committed),
                )
                pending = type("Call", (), {"name": action.name, "arguments": arguments})()
                proposal.messages = gate_gold.history(instructions, seen, pending)
                yield proposal
                committed.append(action.name)
            try:
                result = environment.make_tool_call(
                    tool_name=action.name, requestor=action.requestor, **arguments
                )
            except Exception:  # gold this build cannot replay; the reads still stand
                continue
            text = gate_gold.text_of(result)
            # A write's result is evidence too, and leaving it out is not a small
            # omission. Gold's task 7 upgrades a basic economy reservation to
            # business and *then* cancels it, which is legal only because the
            # upgrade already happened; a ledger holding just the reads still
            # says basic economy, and every check that reasons from the record
            # refuses gold's own cancellation.
            observed.append(text)
            if action.name not in WRITES:
                seen.append((action.name, arguments, text))


def _rendered(instructions: str, seen: list[tuple[str, dict[str, Any], str]]) -> str:
    lines = [f"Customer: {instructions}"]
    for name, _arguments, result in seen:
        lines.append(f"Assistant looks up: {name}(...)")
        lines.append(f"Result: {result}")
    return "\n".join(lines)


# ------------------------------------------------------------------ the gates


def open_gate(_proposal: Proposal) -> Decision:
    """Approves everything. The control, and the number every gate has to beat.

    It scores 0% false blocks and 0% of surplus caught, which is exactly what the
    shipped configuration does today. A gate that cannot beat both halves of that
    at once is not worth its calls.
    """
    return Decision(blocked=False)


def monolith(proposal: Proposal) -> Decision:
    """Today's critic: one model, one enormous question, one structured verdict."""
    from agents.gate import decide, review
    from core.state import PendingCall

    gate = _critic()
    pending = [PendingCall(id="p", name=proposal.name, arguments=proposal.arguments)]
    verdict = decide(gate, review(proposal.messages, pending, proposal.observed))
    return Decision(
        blocked=not verdict.allowed, check="critic" if not verdict.allowed else "", calls=1
    )


_CRITIC: list[Any] = []


def _critic():
    """One agent for the whole run: building it per proposal re-reads the policy."""
    if not _CRITIC:
        from tau2.domains.airline.utils import AIRLINE_POLICY_PATH

        from agents.gate import build_gate

        policy = open(AIRLINE_POLICY_PATH, encoding="utf-8").read()
        _CRITIC.append(build_gate(policy, os.environ.get("STEWARD_LLM_MODEL")))
    return _CRITIC[0]


def tier1(proposal: Proposal) -> Decision:
    """The deterministic verifiers alone, with no model in the loop at all.

    Scored separately from the full sieve on purpose. Whatever these catch is
    caught for free, every time, and identically on a rerun -- so it is the floor
    the judges have to add to rather than a component of a number only reported
    together with them.
    """
    from adapters.tau2.verifiers import PANEL
    from core.state import PendingCall
    from core.verifiers import first

    call = PendingCall(id="p", name=proposal.name, arguments=proposal.arguments)
    finding = first(call, proposal.evidence(), PANEL)
    return Decision(blocked=finding is not None, check=finding.check if finding else "")


_JUDGE: list[Any] = []


def sieve(proposal: Proposal) -> Decision:
    """Tier 1, then the `requested` judge on whatever survives it.

    The order is the whole design. A deterministic check that fires costs nothing
    and cannot be talked out of its answer, so it goes first and short-circuits;
    the model is asked only about the cases arithmetic cannot reach. On this
    corpus that means roughly one proposal in five never reaches a model at all.
    """
    from adapters.tau2.context import already, exchange, facts, means, spelled
    from agents.requested import build_requested, requested

    settled = tier1(proposal)
    if settled.blocked:
        return settled

    if not _JUDGE:
        _JUDGE.append(build_requested(os.environ.get("STEWARD_LLM_MODEL")))
    verdict = requested(
        _JUDGE[0],
        exchange(proposal.dialogue),
        spelled(proposal.name, proposal.arguments),
        means(proposal.name, proposal.arguments, proposal.observed),
        facts(proposal.name, proposal.arguments, proposal.observed),
        already(proposal.committed),
    )
    # None means the two phrasings disagreed, which is a fact about the judge and
    # not about the action. It lets the write through.
    blocked = verdict is False
    return Decision(blocked=blocked, check="requested" if blocked else "", calls=2)


GATES: dict[str, Gate] = {
    "none": open_gate,
    "monolith": monolith,
    "tier1": tier1,
    "sieve": sieve,
}


# ----------------------------------------------------------------- the report


@dataclass
class Report:
    """Two rates that only mean anything together, and where they came from."""

    seen: dict[str, int] = field(default_factory=lambda: {"gold": 0, "surplus": 0})
    blocked: dict[str, int] = field(default_factory=lambda: {"gold": 0, "surplus": 0})
    calls: int = 0
    by_check: dict[str, dict[str, int]] = field(default_factory=dict)
    wrong: list[Proposal] = field(default_factory=list)
    missed: list[Proposal] = field(default_factory=list)

    def record(self, proposal: Proposal, decision: Decision) -> None:
        self.seen[proposal.label] += 1
        self.calls += decision.calls
        if decision.blocked:
            self.blocked[proposal.label] += 1
            tally = self.by_check.setdefault(decision.check or "(unattributed)", {})
            tally[proposal.label] = tally.get(proposal.label, 0) + 1
            if proposal.label == "gold":
                self.wrong.append(proposal)
        elif proposal.label == "surplus":
            self.missed.append(proposal)

    def render(self, gate: str, source: str) -> str:
        gold, surplus = self.seen["gold"], self.seen["surplus"]
        total = gold + surplus
        lines = [
            "",
            "=" * 74,
            f"  gate: {gate}    source: {source}    proposals: {total}",
            "=" * 74,
            f"  {'':<38}{'blocked':>12}{'rate':>10}",
            f"  {'GOLD  (blocking one is a defect)':<38}"
            f"{f'{self.blocked["gold"]}/{gold}':>12}{_rate(self.blocked['gold'], gold):>10}",
            f"  {'SURPLUS  (letting one through loses)':<38}"
            f"{f'{self.blocked["surplus"]}/{surplus}':>12}"
            f"{_rate(self.blocked['surplus'], surplus):>10}",
            "-" * 74,
            f"  model calls: {self.calls}"
            + (f"   ({self.calls / total:.2f} per proposal)" if total else ""),
        ]
        if self.by_check:
            lines.append("\n  what fired, and on what:")
            lines.append(f"    {'check':<30}{'surplus':>10}{'gold':>8}{'precision':>12}")
            for check, tally in sorted(
                self.by_check.items(), key=lambda pair: -sum(pair[1].values())
            ):
                good, bad = tally.get("surplus", 0), tally.get("gold", 0)
                lines.append(f"    {check:<30}{good:>10}{bad:>8}{_rate(good, good + bad):>12}")
        if self.wrong:
            lines.append("\n  gold writes it blocked:")
            for proposal in self.wrong[:20]:
                lines.append(f"    task {proposal.task:>3}  {proposal.name}")
        return "\n".join(lines)


def _rate(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="none", choices=sorted(GATES), help="which gate to score")
    parser.add_argument(
        "--source",
        default="both",
        choices=("runs", "key", "both"),
        help="runs: real proposals, the only ones with surplus in them. key: gold's own writes.",
    )
    parser.add_argument("--tasks", nargs="+", help="restrict the answer key to these task ids")
    parser.add_argument("--limit", type=int, help="stop after this many proposals")
    parser.add_argument("--out", help="write the per-proposal verdicts here as JSON")
    args = parser.parse_args()

    answer = gold_writes()
    proposals: list[Proposal] = []
    if args.source in ("runs", "both"):
        for run in RUNS:
            proposals.extend(harvest(run, answer))
    if args.source in ("key", "both"):
        proposals.extend(from_answer_key(args.tasks))
    if args.limit:
        proposals = proposals[: args.limit]

    gate = GATES[args.gate]
    report = Report()
    verdicts = []
    for proposal in proposals:
        decision = gate(proposal)
        report.record(proposal, decision)
        verdicts.append(
            {
                "run": proposal.run,
                "task": proposal.task,
                "label": proposal.label,
                "call": proposal.one_line[:200],
                "blocked": decision.blocked,
                "check": decision.check,
                "stated": proposal.stated,
            }
        )

    print(report.render(args.gate, args.source))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(verdicts, handle, indent=2)
        print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
