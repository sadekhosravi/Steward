"""The Kernel: the LangGraph graph that runs the multi-agent system.

The Kernel is deliberately LLM-free. Every choice it makes is a routing decision
over state; the model calls happen inside nodes, in `agents`. Reward is binary
per task and pass^k only counts a task when every trial passes, so sampling
variance in *control flow* is a direct score loss -- worth spending determinism
on even where a model would be more flexible.

The graph pauses the way the benchmark does. The `act` node calls `interrupt()`
with the tool calls to make, control returns to whoever is driving the Kernel,
they run those calls against the real environment, and `resume()` feeds the
results back in. Sub-agents added later inherit this for free: an interrupt
resumes inside the node that raised it, however deep in the graph it sits.

    plan -> think -> gate  -> act -> think -> ... -> END
                  -> speak                       -> END

Nothing reaches `act` without passing `gate` first, and `act` emits `approved`
rather than `calls`. That is structural on purpose: an approval that does not
bind the thing executed is not an approval, so there is no path on which a call
is emitted that the gate never saw.

A turn has two exits and `gate` only ever guarded one of them. `speak` guards the
other: it asks, at the moment the actor tries to hand the turn back to the
customer, whether the writes the planner asked for have happened. It is cheap by
construction -- when nothing is outstanding it returns without a model call, which
is every turn of every task that needs no writes at all.

`plan` sits at the entry rather than in the loop, and that placement is the whole
of its cost control: a turn of twenty tool calls re-enters `think` twenty times
and `plan` never, because `resume()` returns into `act` and rejoins below it. One
planning call per user message, whatever the turn goes on to cost.

That asymmetry is also why `plan` is where the policy is routed. It reads the
whole policy once and hands the actor the sections this turn is about; the actor
carries those through twenty rebuilds of its instructions instead of the lot.

    kernel = Kernel(tools, policy)
    thread = kernel.new_thread()
    step = kernel.send(thread, "please cancel my flight")
    step = kernel.resume(thread, {call_id: result_text})
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from pydantic_core import to_jsonable_python

import tracing
from agents.assistant import Assistant, build_assistant
from agents.gate import Verdict, build_gate, decide, review
from agents.planner import Plan, brief, build_planner, render
from agents.speaker import HELD, build_speaker, hold, outstanding, permit
from core.policy import excerpt
from core.state import Deps, PendingCall, StewardState

__all__ = ["Act", "Kernel", "PendingCall", "Say", "Step", "build_graph"]

# One user turn can cost many tool round trips. LangGraph's default of 25 would
# cut a long-but-legitimate investigation short and score it as a failure.
RECURSION_LIMIT = 100

# Blocked write attempts allowed per user turn before the actor has to stop and
# talk to the customer. Counted per turn rather than per action, so no
# arrangement of reads and writes lets the loop run forever: the baseline
# produced four simulations that never terminated, and an unbounded correction
# loop is the obvious way to produce more.
REVISION_LIMIT = 2

# Replies held per user turn before the actor is allowed to speak regardless.
# Lower than the gate's budget on purpose: a held reply costs a whole assistant
# run, and an actor that comes back with the same message after being told once
# is telling us it has nothing else to give. One push is the intervention; two is
# nagging, and the customer is waiting through both.
DEFERRAL_LIMIT = 1

DENIAL = "This action was not performed. {reason} {remediation}"

ESCALATION = (
    "You have already tried to correct this and it is still not allowed. Stop here: "
    "tell the customer where things stand and what you need from them. Do not attempt "
    "another action."
)


@dataclass(frozen=True)
class Say:
    """The turn ended with something to tell the user."""

    text: str


@dataclass(frozen=True)
class Act:
    """The turn is paused until these calls come back."""

    calls: list[PendingCall]


Step = Say | Act


def _results(state: StewardState) -> DeferredToolResults | None:
    """Answers for the calls we last yielded: what came back, and what was refused.

    A refusal is a `ModelRetry`, pydantic-ai's own vehicle for "that call was
    wrong, here is why, try again". It lands in history as a retry prompt, so the
    actor sees a rejection rather than a fabricated result.
    """
    if not (state.tool_results or state.denied):
        return None
    return DeferredToolResults(
        calls={
            **dict(state.tool_results),
            **{key: ModelRetry(message) for key, message in state.denied.items()},
        }
    )


def _history(state: StewardState) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(state.messages)


def _plan(state: StewardState, planner: Agent[None, Plan], policy: str) -> dict[str, Any]:
    """Write down the route before the actor starts composing calls, and the rules it runs under.

    Fails *open*, which is the opposite of the gate beside it and for the same
    reason. A verdict that never arrived authorises nothing, so a missing one has
    to refuse; a plan that never arrived withholds nothing, so a missing one costs
    only the advice. The actor has solved tasks without it for three runs.

    The policy excerpt fails open in the stronger sense, because it *can* withhold:
    no plan means no sections named, and `excerpt` answers that with all of them.
    A turn the planner could not answer for is therefore exactly the prompt this
    node existed to build before any of it was selective.
    """
    try:
        plan = planner.run_sync(brief(_history(state), state.prompt)).output
    except UnexpectedModelBehavior:
        return {"plan": "", "policy": excerpt(policy, []), "changes": []}
    return {
        "plan": render(plan),
        "policy": excerpt(policy, plan.policy_sections),
        # Kept as the planner wrote them, for the speaker to count against what
        # the gate approves. A turn the planner could not answer for owes nothing,
        # which is the fail-open answer here too: no plan, no held messages.
        "changes": plan.changes,
    }


def _think(state: StewardState, assistant: Assistant) -> dict[str, Any]:
    """Ask the assistant what to do next, given everything that has happened.

    Also the one place new evidence enters, because it is the one node that sees
    both a user message and tool results before they are consumed. Denials are
    not evidence: the call never ran, so it showed us nothing.
    """
    seen = [t for t in [state.prompt, *state.tool_results.values()] if t]
    run = assistant.run_sync(
        state.prompt,
        message_history=_history(state),
        deferred_tool_results=_results(state),
        # The ledger as of now, not as of last turn. What just arrived is the most
        # likely source of an identifier the actor is about to use, and it does not
        # reach `observed` until this node returns -- so passing the stored ledger
        # alone would reject a reservation id the customer gave a moment ago.
        deps=Deps(
            observed=state.observed + seen,
            plan=state.plan,
            policy=state.policy,
            correction=state.correction,
        ),
    )
    output = run.output
    calls = (
        [
            PendingCall(
                id=c.tool_call_id, name=c.tool_name, arguments=c.args_as_dict()
            ).model_dump()
            for c in output.calls
        ]
        if isinstance(output, DeferredToolRequests)
        else []
    )
    return {
        "messages": to_jsonable_python(run.all_messages()),
        "observed": state.observed + seen,
        "prompt": None,
        "tool_results": {},
        "denied": {},
        # Consumed, like the prompt: a correction answers one attempt, and one
        # left standing would be re-read on every request of the rest of the turn.
        "correction": "",
        "calls": calls,
        "reply": "" if calls else output,
    }


def _gate(state: StewardState, gate: Agent[None, Verdict], gated: frozenset[str]) -> dict[str, Any]:
    """Approve or refuse the proposed step, as a whole.

    A step of pure lookups is approved without a model call: reads cannot damage
    the scored database and cannot end the conversation, and paying a critic to
    say so on every lookup would double the cost of the thing the critic is meant
    to make affordable.

    Refusal is all-or-nothing. The gate judges a plan, not a list, and a plan
    with one forbidden move in it is not half-executable -- letting the harmless
    calls through would leave the actor re-planning from a position it never
    chose.
    """
    proposal = [PendingCall(**call) for call in state.calls]
    writes = [call.name for call in proposal if call.name in gated]
    if not writes:
        return {"approved": state.calls, "calls": []}

    # `decide` retries and never raises: letting a failure propagate ends the
    # simulation and scores 0, where refusing costs one action and is
    # recoverable. An unanswered check still fails closed, but only after the
    # call has been given more than one chance to come back.
    verdict = decide(gate, review(_history(state), proposal, state.observed))
    if verdict.allowed:
        return {"approved": state.calls, "calls": [], "written": state.written + writes}

    message = DENIAL.format(reason=verdict.reason, remediation=verdict.remediation)
    return {
        "approved": [],
        "calls": [],
        "denied": {call.id: message for call in proposal},
        "revisions": state.revisions + 1,
    }


def _speak(state: StewardState, speaker: Agent[None, Verdict]) -> dict[str, Any]:
    """Let the reply go, or send the actor back to finish the turn.

    The deterministic half runs first and decides whether the model is asked at
    all. Nothing outstanding means nothing to argue about, and that is the common
    case by a wide margin -- every turn of every task that needs no writes, and
    every turn that has already made them. Those cost nothing here.

    Fails **open**, unlike the gate beside it. The asymmetry is the same one that
    governs `plan`, read the other way round: refusing to let the actor speak
    withholds the turn from the customer and cannot itself produce a write, so a
    check that did not answer must not be the reason a customer is left waiting.
    """
    owed = outstanding(state.changes, state.written)
    if not owed or state.deferrals >= DEFERRAL_LIMIT:
        return {}

    # `consulted` is returned on both branches, and that is the whole reason it
    # exists. An allowed message otherwise leaves no trace at all, which is exactly
    # the hole the gate had: a block and a proposal that was never made looked the
    # same from outside, and the block rate had to be recovered from Langfuse after
    # the fact. Two counters make "how often did it fire" and "how often did it
    # hold" separately readable while the run is still going.
    verdict = permit(speaker, hold(_history(state), state.reply, owed))
    if verdict.allowed:
        return {"consulted": state.consulted + 1}
    return {
        "correction": HELD.format(reason=verdict.reason, remediation=verdict.remediation),
        "reply": "",
        "deferrals": state.deferrals + 1,
        "consulted": state.consulted + 1,
        "holds": state.holds + 1,
    }


def _act(state: StewardState) -> dict[str, Any]:
    """The yield point, and nothing else.

    LangGraph re-runs a node from the top when it resumes, so anything with a
    side effect placed before `interrupt()` would happen twice. That is why this
    node does only this.
    """
    results: dict[str, str] = interrupt(state.approved)
    return {"tool_results": results, "approved": []}


def _escalate(state: StewardState, assistant: Assistant) -> dict[str, Any]:
    """Out of corrections: end the turn by talking to the customer.

    The reply is written by the actor rather than canned, so it fits the
    conversation. The run is stripped of its tools and forced to `str` output,
    which makes another attempt unrepresentable rather than merely discouraged --
    and that matters: with the tools still offered, a model that tried anyway
    would raise and take the whole simulation down with it.
    """
    denied = {
        key: ModelRetry(f"{message}\n\n{ESCALATION}") for key, message in state.denied.items()
    }
    run = assistant.run_sync(
        message_history=_history(state),
        deferred_tool_results=DeferredToolResults(calls=denied),
        output_type=str,
        toolsets=[],
        # No plan: this run exists to end the turn by talking to the customer, and
        # a list of changes still to make is the one thing it must not be urged
        # towards. The ledger stays -- what it says is still true.
        deps=Deps(observed=state.observed),
    )
    return {
        "messages": to_jsonable_python(run.all_messages()),
        "denied": {},
        "calls": [],
        "approved": [],
        "reply": run.output,
    }


def _route_think(state: StewardState) -> Literal["gate", "speak"]:
    """Tool calls go to the gate; a reply goes to the speaker. Both exits are guarded."""
    return "gate" if state.calls else "speak"


def _route_speak(state: StewardState) -> Literal["think", "__end__"]:
    """A held reply goes back for another attempt; anything else ends the turn."""
    return "think" if state.correction else END


def _route_gate(state: StewardState) -> Literal["act", "think", "escalate"]:
    """Approved work is emitted; a refusal goes back for a rewrite, until it cannot."""
    if state.approved:
        return "act"
    return "think" if state.revisions <= REVISION_LIMIT else "escalate"


def _traced(name: str, node: Callable[[StewardState], dict[str, Any]]):
    """Report what a node was given and what it decided.

    This is the layer that makes a model call legible: pydantic-ai records the
    call, the span around it records which agent made it and what the Kernel did
    with the answer. `act` is not wrapped, because it leaves by raising
    LangGraph's interrupt -- a span would file that as a failure, and what it
    yields is already the output of the `gate` span before it.
    """

    def traced(state: StewardState) -> dict[str, Any]:
        with tracing.span(name, **state.model_dump(exclude=tracing.BULK)) as span:
            delta = node(state)
            span.update(output=tracing.visible(delta))
            return delta

    return traced


def build_graph(
    assistant: Assistant,
    gate: Agent[None, Verdict],
    planner: Agent[None, Plan],
    speaker: Agent[None, Verdict],
    gated: frozenset[str],
    policy: str,
) -> Any:
    graph = StateGraph(StewardState)
    graph.add_node("plan", _traced("plan", partial(_plan, planner=planner, policy=policy)))
    graph.add_node("think", _traced("think", partial(_think, assistant=assistant)))
    graph.add_node("gate", _traced("gate", partial(_gate, gate=gate, gated=gated)))
    graph.add_node("speak", _traced("speak", partial(_speak, speaker=speaker)))
    graph.add_node("act", _act)
    graph.add_node("escalate", _traced("escalate", partial(_escalate, assistant=assistant)))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "think")
    graph.add_conditional_edges("think", _route_think, {"gate": "gate", "speak": "speak"})
    graph.add_conditional_edges(
        "gate", _route_gate, {"act": "act", "think": "think", "escalate": "escalate"}
    )
    graph.add_conditional_edges("speak", _route_speak, {"think": "think", END: END})
    graph.add_edge("act", "think")
    # Straight to END, bypassing `speak`. Escalation exists precisely to end a turn
    # the gate would not let continue, so a check whose only power is to send the
    # actor back to work has nothing to say about it -- and holding that reply
    # would put the turn in a loop between the two things that stop it.
    graph.add_edge("escalate", END)
    return graph.compile(checkpointer=InMemorySaver())


def _gated(tools: list[ToolDefinition]) -> frozenset[str]:
    """The tools whose effects cannot be taken back, per the adapter's labels.

    Which tools those are is a fact about the environment, so the adapter decides
    and this only reads the answer -- that is what keeps `core` free of tau2 and
    right in a domain nobody has looked at yet. A tool that arrives unlabelled is
    gated: the cost of reviewing a read by mistake is one model call, and the cost
    of missing a write is the task.
    """
    return frozenset(t.name for t in tools if (t.metadata or {}).get("gated", True))


class Kernel:
    """Drives one compiled graph over many conversations, one thread each."""

    def __init__(
        self,
        tools: list[ToolDefinition],
        policy: str,
        model: str | Model | None = None,
        gate_model: str | Model | None = None,
        planner_model: str | Model | None = None,
        speaker_model: str | Model | None = None,
        reference: str = "",
    ):
        """`gate_model`, `planner_model` and `speaker_model` let the two critics and
        the planner run on a different model from the actor. All default to the
        actor's, which keeps one knob until there is a reason for four; resolving
        *which* model is a caller's job, not the Kernel's.

        `reference` is domain knowledge the policy assumes and never states. The
        Kernel does not know what is in it and does not look: deciding that is the
        adapter's job, which is what keeps `core` free of any one environment."""
        self.graph = build_graph(
            build_assistant(tools, policy, model, reference),
            build_gate(policy, gate_model if gate_model is not None else model),
            build_planner(tools, policy, planner_model if planner_model is not None else model),
            build_speaker(policy, speaker_model if speaker_model is not None else model),
            _gated(tools),
            policy,
        )

    def new_thread(self) -> str:
        """A fresh conversation. State for it lives in the checkpointer, not here."""
        return uuid4().hex

    def send(self, thread: str, text: str) -> Step:
        """Deliver a user message and run until the Kernel needs something.

        Both budgets and both per-turn ledgers reset here, because a new message is
        a new plan: the changes about to be written down belong to this turn, and
        so does the record of which of them have been approved.
        """
        return self._run(
            thread,
            {
                "prompt": text,
                "revisions": 0,
                "deferrals": 0,
                "written": [],
                "correction": "",
            },
            "message",
            text=text,
        )

    def resume(self, thread: str, results: dict[str, str]) -> Step:
        """Hand back the results of the calls from the last `Act` and carry on."""
        return self._run(thread, Command(resume=results), "results", results=results)

    def _run(self, thread: str, payload: Any, name: str, **traced: Any) -> Step:
        """One trace per step, named for what arrived, all of them in one session.

        A step is the largest unit the Kernel controls end to end: emitting tool
        calls hands control back to the harness, so a span held across that would
        have to survive a return. The conversation is the session instead, which
        is how the pieces read as one story.
        """
        config = {
            "configurable": {"thread_id": thread},
            "recursion_limit": RECURSION_LIMIT,
        }
        with tracing.session(thread), tracing.span(name, **traced) as span:
            out = self.graph.invoke(payload, config)
            paused = out.get("__interrupt__")
            if paused:
                step: Step = Act(calls=[PendingCall(**call) for call in paused[0].value])
            else:
                step = Say(text=out["reply"])
            span.update(output=step)
            return step
