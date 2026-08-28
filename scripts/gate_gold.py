"""Put every write gold makes to the gate, and count how many it refuses.

The benchmark cannot answer questions about the gate. Two runs of one
configuration came back 0.440 and 0.420 with fifteen of fifty tasks flipping
between zero and non-zero, so any effect smaller than about 0.2 is invisible and
nothing we can build is that big. Reading a per-task flip as evidence -- which
was done repeatedly before the noise was measured -- is reading noise.

This measures the gate the way Parts 1-3 were measured instead: offline, against
the actions gold takes, where the right answer is known and there is no sampling
error to hide behind. Gold's writes are correct by definition, so every refusal
of one is a defect, and the count is a number that means something on its own
rather than only in comparison to another run.

    uv run python scripts/gate_gold.py              # every task with a gold write
    uv run python scripts/gate_gold.py --tasks 30 39 42
    uv run python scripts/gate_gold.py --deterministic   # skip the model entirely

What it does per task: take a fresh environment, walk gold's actions in order,
execute the reads and feed their results in as the provenance ledger, and put
each write to the gate as a proposal before executing it. The context the gate
gets is therefore the context it would have had if the run had gone perfectly --
which is the most favourable case, and still not one it passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), ".env"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

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


@dataclass
class Refusal:
    task: str
    tool: str
    reason: str
    remediation: str


@dataclass
class Report:
    """What the gate did to work that was known to be right."""

    proposals: int = 0
    refusals: list[Refusal] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)

    @property
    def tasks_hit(self) -> set[str]:
        return {refusal.task for refusal in self.refusals}

    def render(self) -> str:
        rate = len(self.refusals) / self.proposals if self.proposals else 0.0
        lines = [
            "",
            "=" * 72,
            f"  gold writes put to the gate : {self.proposals}",
            f"  refused                     : {len(self.refusals)}  ({rate:.0%})",
            f"  tasks affected              : {len(self.tasks_hit)}",
            "=" * 72,
        ]
        for refusal in self.refusals:
            lines.append(f"\n  task {refusal.task}  {refusal.tool}")
            lines.append(f"    reason      : {refusal.reason[:160]}")
            lines.append(f"    remediation : {refusal.remediation[:160]}")
        if self.unreachable:
            lines.append(f"\n  tasks skipped (gold could not be replayed): {self.unreachable}")
        return "\n".join(lines)


def history(instructions: str, seen: list[tuple[str, dict[str, Any], str]], pending=None):
    """A transcript of the reads gold made, as the gate would have been shown it.

    `pending` is the write about to be proposed. When it is given, the transcript
    ends the way the policy says it must -- the action listed to the customer and
    the customer answering yes -- because otherwise every refusal is the gate
    correctly noticing that this harness never asked. The first measurement had
    no such turn and the gate refused 49 of 49; most of those were that omission
    and not a defect. A harness that manufactures the failure it reports is worse
    than no harness.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    messages: list[Any] = [ModelRequest(parts=[UserPromptPart(content=instructions)])]
    for index, (name, arguments, result) in enumerate(seen):
        call_id = f"g{index}"
        messages.append(ModelResponse(parts=[ToolCallPart(name, arguments, tool_call_id=call_id)]))
        messages.append(
            ModelRequest(
                parts=[ToolReturnPart(tool_name=name, content=result, tool_call_id=call_id)]
            )
        )
    if pending is not None:
        listed = ", ".join(f"{key}={value!r}" for key, value in pending.arguments.items())
        messages.append(
            ModelResponse(
                parts=[
                    TextPart(
                        f"Here is what I am about to do: {pending.name} with {listed}. "
                        "Shall I go ahead?"
                    )
                ]
            )
        )
        messages.append(ModelRequest(parts=[UserPromptPart(content="Yes, please go ahead.")]))
    messages.append(ModelResponse(parts=[TextPart("")]))
    return messages


def text_of(result: Any) -> str:
    """A tool result as the assistant is really shown it.

    tau2 serialises a returned model with pydantic's own JSON, so a reservation
    reaches the agent as `{"reservation_id": "M61CQM", ...}`. `json.dumps` with
    `default=str` produces the model's *repr* instead --
    `reservation_id='M61CQM' user_id=...` -- which every substring check here
    happens to survive and every structured one would not. Replaying gold through
    a shape no run ever sees is the kind of harness error this file has already
    made five times.
    """
    if isinstance(result, str):
        return result
    try:
        # Handles a bare model, and equally a list of them -- which is what a
        # flight search returns, and where `model_dump_json` alone still leaves
        # every element as a repr.
        from pydantic_core import to_json

        return to_json(result).decode()
    except Exception:
        return json.dumps(result, default=str)


def run(
    task_ids: list[str] | None,
    deterministic: bool,
    model: str | None = None,
) -> Report:
    from tau2.domains.airline.environment import get_environment, get_tasks

    from agents.gate import build_gate
    from core.state import PendingCall

    report = Report()
    critic = model or os.environ.get("STEWARD_LLM_MODEL")
    gate = None if deterministic else build_gate(policy(), critic)

    for task in get_tasks():
        if task_ids and task.id not in task_ids:
            continue
        criteria = task.evaluation_criteria
        actions = (criteria.actions if criteria else None) or []
        if not any(action.name in WRITES for action in actions):
            continue

        environment = get_environment()
        instructions = str(task.user_scenario.instructions)
        # The customer's own words are evidence, and the Kernel treats them that
        # way --  puts the prompt into  beside the tool results.
        # Seeding this with lookups alone made every identifier the customer
        # simply stated look invented, and 29 of gold's writes unprovenanced.
        observed: list[str] = [instructions]
        seen: list[tuple[str, dict[str, Any], str]] = []

        for action in actions:
            arguments = dict(action.arguments or {})
            if action.name in WRITES:
                # Gold's action list records the writes a task must make and, for
                # some tasks, the reads. It is not a transcript: tasks 14, 15 and
                # 17 list writes alone. A run that made those writes had looked
                # the records up first -- it could not have named them otherwise --
                # so the harness does the lookups gold does not bother to record.
                # Without this every identifier looks invented and the provenance
                # check refuses 28 of gold's own writes.
                look_up(environment, arguments, observed, seen)
                report.proposals += 1
                proposal = [
                    PendingCall(id=f"p{report.proposals}", name=action.name, arguments=arguments)
                ]
                verdict = judge(
                    gate,
                    proposal,
                    history(instructions, seen, proposal[0]),
                    observed,
                )
                if verdict is not None and not verdict.allowed:
                    report.refusals.append(
                        Refusal(task.id, action.name, verdict.reason, verdict.remediation)
                    )
            try:
                result = environment.make_tool_call(
                    tool_name=action.name, requestor=action.requestor, **arguments
                )
            except Exception as failure:  # gold that this build cannot replay
                report.unreachable.append(f"{task.id}:{action.name}:{failure}")
                continue
            text = text_of(result)
            if action.name not in WRITES:
                observed.append(text)
                seen.append((action.name, arguments, text))

    return report


def look_up(environment, arguments: dict[str, Any], observed: list[str], seen: list) -> None:
    """Read the records this write names, as a run would have before writing.

    Only the two lookups whose arguments are already in hand. Flight numbers come
    from a search this cannot reconstruct -- it does not know which search the run
    would have made -- so provenance for those is not testable here and is not
    claimed to be.
    """
    wanted: list[tuple[str, dict[str, Any]]] = []
    if arguments.get("user_id"):
        wanted.append(("get_user_details", {"user_id": arguments["user_id"]}))
    if arguments.get("reservation_id"):
        wanted.append(("get_reservation_details", {"reservation_id": arguments["reservation_id"]}))
    for name, args in list(wanted):
        if name != "get_reservation_details":
            continue
        # A payment method lives on the user, and an update names only the
        # reservation -- so the owner has to be read through the record, exactly
        # as a run would.
        try:
            record = environment.make_tool_call(tool_name=name, requestor="assistant", **args)
        except Exception:
            continue
        owner = getattr(record, "user_id", None)
        if owner:
            wanted.append(("get_user_details", {"user_id": owner}))

    # Flight numbers come from a search. Which search a run would have made is a
    # guess, so the harness makes the two the domain offers between the endpoints
    # the proposal already names, on the dates it names.
    origin, destination = arguments.get("origin"), arguments.get("destination")
    for leg in arguments.get("flights") or []:
        if origin and destination and leg.get("date"):
            for search in ("search_direct_flight", "search_onestop_flight"):
                # Both directions: a round trip's return legs run the other way,
                # and searching only the outbound pair left them unprovenanced.
                for a, b in ((origin, destination), (destination, origin)):
                    wanted.append((search, {"origin": a, "destination": b, "date": leg["date"]}))

    for name, args in wanted:
        if any(name == done and args == with_ for done, with_, _ in seen):
            continue
        try:
            result = environment.make_tool_call(tool_name=name, requestor="assistant", **args)
        except Exception:
            continue
        text = text_of(result)
        observed.append(text)
        seen.append((name, args, text))


def judge(gate, proposal, messages, observed, preflight_only=False):
    """The gate's verdict on one proposal, or None when there is no gate to ask."""
    if gate is None:
        return None
    from agents.gate import decide, review

    return decide(gate, review(messages, proposal, observed))


def policy() -> str:
    from tau2.domains.airline.utils import AIRLINE_POLICY_PATH

    return open(AIRLINE_POLICY_PATH, encoding="utf-8").read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", help="task ids; default is every write task")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="skip the model and report only what gold replays cleanly",
    )
    parser.add_argument("--out", help="write the refusals to this file as JSON")
    parser.add_argument("--model", help="critic model; default is STEWARD_LLM_MODEL")
    args = parser.parse_args()
    report = run(args.tasks, args.deterministic, args.model)
    print(report.render())
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "proposals": report.proposals,
                    "refusals": [vars(refusal) for refusal in report.refusals],
                },
                handle,
                indent=2,
            )
        print(chr(10) + f"  written to {args.out}")
    return 1 if report.refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
