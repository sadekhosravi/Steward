"""Probe a provider's catalog and check models against what the MAS needs.

The MAS design leans on two model capabilities that are uniform across frontier
APIs but *not* across self-hosted catalogs: native tool calling and structured
output. The Critic returning a typed verdict is the spine of the write gate, so
a model that only manages prose is unusable no matter how well it reasons.

    uv run python scripts/probe_models.py config
    uv run python scripts/probe_models.py list --provider nvidia --filter llama
    uv run python scripts/probe_models.py check --model meta/llama-3.3-70b-instruct

Output is deliberately ASCII: the Windows console codepage cannot encode the
symbols that would otherwise be nicer here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

import llm

TIMEOUT = 30.0


class Verdict(BaseModel):
    """Shaped like the Critic's real output, so the check measures the real ask."""

    decision: Literal["approve", "block", "need_evidence"]
    reason: str = Field(description="One sentence justifying the decision.")


@dataclass
class Check:
    name: str
    ok: bool
    seconds: float
    detail: str = ""


def _brief(exc: Exception, limit: int = 90) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _timed(name: str, fn) -> Check:
    start = time.perf_counter()
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a failed probe is a result, not a crash
        return Check(name, False, time.perf_counter() - start, _brief(exc))
    return Check(name, ok, time.perf_counter() - start, detail)


def check_chat(spec: llm.ModelSpec) -> tuple[bool, str]:
    agent = Agent(model=llm.build_model(spec))
    out = agent.run_sync("Reply with exactly the word: ready").output
    ok = "ready" in out.strip().lower()
    return ok, "" if ok else f"said {out.strip()[:40]!r}"


def check_structured(spec: llm.ModelSpec) -> tuple[bool, str]:
    agent = Agent(model=llm.build_model(spec), output_type=Verdict)
    out = agent.run_sync(
        "An agent proposes cancelling a reservation. Policy allows cancellation only "
        "within 24 hours of booking. The booking was made 9 days ago. "
        "Decide whether to approve, block, or ask for more evidence."
    ).output
    return isinstance(out, Verdict), f"decision={out.decision}"


def check_tools(spec: llm.ModelSpec) -> tuple[bool, str]:
    called: list[str] = []
    agent = Agent(model=llm.build_model(spec))

    @agent.tool_plain
    def get_reservation_status(reservation_id: str) -> str:
        """Look up the current status of a reservation by its ID."""
        called.append(reservation_id)
        return "status=confirmed; passengers=2; cabin=economy"

    agent.run_sync(
        "What is the status of reservation HJK4RT? You must use the tool to find out; do not guess."
    )
    detail = f"called with {called[0]!r}" if called else "tool never called"
    return bool(called), detail


def probe(spec: llm.ModelSpec) -> list[Check]:
    return [
        _timed("chat", lambda: check_chat(spec)),
        _timed("structured", lambda: check_structured(spec)),
        _timed("tools", lambda: check_tools(spec)),
    ]


def list_models(provider: str, needle: str | None) -> int:
    info = llm.get_provider(provider)
    headers = {}
    try:
        headers["Authorization"] = f"Bearer {llm.api_key_for(provider)}"
    except llm.LLMConfigError:
        pass  # some catalogs are public; let the request decide
    resp = httpx.get(f"{info.base_url}/models", headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    ids = sorted(m["id"] for m in resp.json().get("data", []))
    if needle:
        ids = [i for i in ids if needle.lower() in i.lower()]
    for model_id in ids:
        print(model_id)
    print(f"\n{len(ids)} model(s) on {provider}. Details: {info.catalog_url}")
    return 0


def show_config(defaults: llm.ModelSpec) -> int:
    print("Provider keys:")
    for name, info in sorted(llm.PROVIDERS.items()):
        present = "set" if os.environ.get(info.api_key_env, "").strip() else "MISSING"
        print(f"  {name:12} {info.api_key_env:20} {present}")

    print("\nDefaults (MAS_LLM_*), used when an agent omits an argument:")
    print(f"  provider     {defaults.provider}")
    print(f"  model        {defaults.model or '(unset)'}")
    print(f"  temperature  {defaults.temperature}")
    print(f"  timeout      {defaults.timeout}")
    print(f"  max_tokens   {defaults.max_tokens if defaults.max_tokens is not None else '(unset)'}")

    if defaults.model:
        print("\nFor tau2's user simulator (LiteLLM name, --llm-user):")
        print(f"  {llm.litellm_name(defaults)}")
    return 0


def check_models(specs: list[llm.ModelSpec]) -> int:
    usable: list[llm.ModelSpec] = []
    for spec in specs:
        print(f"\n{spec}")
        checks = probe(spec)
        for check in checks:
            mark = "ok  " if check.ok else "FAIL"
            print(f"  {mark} {check.name:11} {check.seconds:6.2f}s  {check.detail}")
        if all(c.ok for c in checks):
            usable.append(spec)

    print(f"\n{len(usable)}/{len(specs)} model(s) passed every check.")
    if usable:
        best = usable[0]
        print("\nTo use one, put this in .env:")
        print(f"  MAS_LLM_PROVIDER={best.provider}")
        print(f"  MAS_LLM_MODEL={best.model}")
    return 0 if len(usable) == len(specs) else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    defaults = llm.env_defaults()

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="show the environment defaults and key presence")

    p_list = sub.add_parser("list", help="enumerate a provider's model catalog")
    p_list.add_argument("--provider", default=defaults.provider)
    p_list.add_argument("--filter", dest="needle", help="substring to match against model ids")

    p_check = sub.add_parser("check", help="probe chat, structured output and tool calling")
    p_check.add_argument("--provider", default=defaults.provider)
    p_check.add_argument(
        "--model",
        dest="models",
        action="append",
        help="model id to probe; repeatable. Defaults to MAS_LLM_MODEL.",
    )

    args = parser.parse_args(argv)

    if args.command == "config":
        return show_config(defaults)
    if args.command == "list":
        return list_models(args.provider, args.needle)

    names = args.models or ([defaults.model] if defaults.model else [])
    if not names:
        print("No model to probe. Pass --model, or set MAS_LLM_MODEL in .env.")
        return 2
    return check_models([llm.resolve(name, provider=args.provider) for name in names])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except llm.LLMConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        sys.exit(2)
