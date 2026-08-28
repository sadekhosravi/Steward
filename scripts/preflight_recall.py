"""What preflight would have refused, on writes a real run actually made.

Precision was measured against gold: it refuses effectively none of the 49 writes
gold makes. That is half a result. A check that approves everything scores the
same, so this is the other half -- of the writes a run emitted, which would it
have stopped, and were those the ones worth stopping?
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

WRITES = {
    "book_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "cancel_reservation",
    "send_certificate",
}


def rendered(lines: list[str]) -> str:
    return "\n".join(lines)


def walk(sim):
    """Every write the run emitted, with the evidence standing behind it."""
    from core.preflight import preflight

    from core.state import PendingCall

    checks = sim["reward_info"].get("action_checks") or []
    matched = [
        c["action"]
        for c in checks
        if c.get("action", {}).get("name") in WRITES and c.get("action_match")
    ]
    used: list[int] = []
    observed: list[str] = []
    dialogue: list[str] = []
    pending: dict[str, dict] = {}

    for message in sim.get("messages") or []:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "user":
            observed.append(content)
            dialogue.append(f"Customer: {content}")
        elif role == "assistant" and content:
            dialogue.append(f"Assistant: {content}")
        for call in message.get("tool_calls") or []:
            pending[call.get("id")] = call
            if call.get("name") not in WRITES:
                dialogue.append(f"Assistant looks up: {call.get('name')}(...)")
        if role == "tool" and message.get("id") in pending:
            call = pending[message["id"]]
            if call.get("name") in WRITES:
                arguments = call.get("arguments") or {}
                hit = None
                for index, gold in enumerate(matched):
                    if index in used:
                        continue
                    if gold["name"] == call["name"] and gold.get("arguments") == arguments:
                        hit = index
                        break
                if hit is not None:
                    used.append(hit)
                kind = (
                    "gold"
                    if hit is not None
                    else ("errored" if message.get("error") else "surplus")
                )
                verdict = preflight(
                    [PendingCall(id="p", name=call["name"], arguments=arguments)],
                    rendered(dialogue),
                    observed,
                )
                yield kind, verdict
            if not message.get("error"):
                observed.append(content)
                dialogue.append(f"Result: {content}")


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "part4a_50"
    path = f"vendor/tau2-data/data/simulations/{name}.json/results.json"
    sims = json.load(open(path, encoding="utf-8"))["simulations"]

    tally: dict[str, list[int]] = {"gold": [0, 0], "surplus": [0, 0], "errored": [0, 0]}
    grounds: dict[str, int] = {}
    for sim in sims:
        for kind, verdict in walk(sim):
            tally[kind][0] += 1
            if verdict is not None:
                tally[kind][1] += 1
                head = verdict.split(".")[0][:52]
                grounds[head] = grounds.get(head, 0) + 1

    print(f"\n  run: {name}")
    print(f"  {'writes emitted':<26}{'total':>7}{'preflight refuses':>20}")
    for kind, (total, caught) in tally.items():
        share = f"{caught}/{total}" + (f"  ({caught / total:.0%})" if total else "")
        print(f"  {kind:<26}{total:>7}{share:>20}")
    print("\n  what it refused for:")
    for ground, count in sorted(grounds.items(), key=lambda pair: -pair[1]):
        print(f"    {count:>3}  {ground}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
