"""Internal tracing, through Langfuse.

The harness only records the external trajectory -- what the customer said, what
the assistant said back, what the database looks like at the end. Everything that
makes this a multi-agent system happens between those messages and is invisible
there: which sub-agent ran, what it was shown, what it decided. Run 002 made the
cost of that concrete. The gate's block rate had to be *inferred* from how many
writes came out the other side, because no verdict it ever produced was written
down anywhere a person could read it.

Two layers, and they nest:

- **Every model call**, from pydantic-ai's own OpenTelemetry instrumentation.
  Prompt, response, tool calls, token counts. Nothing to maintain: it comes from
  the library, and Langfuse renders the spans as generations because it knows the
  `pydantic-ai` instrumentation scope.
- **Every graph node**, from `span()` below, wrapped around the node in
  `core.kernel`. This is the layer that says *which agent* a model call belongs
  to and what the Kernel did with the answer.

One trace per Kernel step, tagged with the conversation as its session, so a
whole simulation reads top to bottom in the Langfuse session view.

Tracing is off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set,
and `setup()` is never called from library code -- only from an entry point
(`scripts/run_bench.py`). A test run traces nothing, and a benchmark run whose
trace server is unreachable still scores.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse, get_client, propagate_attributes

from . import journal

__all__ = [
    "BULK",
    "enabled",
    "flush",
    "journal",
    "label",
    "session",
    "setup",
    "shutdown",
    "span",
    "visible",
]

_client: Langfuse | None = None

# Facts about the whole run, repeated on every session so a trace can be told
# apart from the one beside it. Without them a scripted test and a 200-task
# benchmark look identical in the session list, which is a real way to lose an
# afternoon.
_labels: dict[str, str] = {}

# State the Kernel carries that is not worth shipping on every node span. The
# pydantic-ai message history is already on the generation spans in full, and
# both of these grow with the conversation, so including them would send the same
# text again for every node of every step of every one of 200 simulations.
BULK = frozenset({"messages", "observed", "policy"})


def setup(client: Langfuse | None = None) -> bool:
    """Turn tracing on. Idempotent, and never raises: returns whether it worked.

    Instrumenting pydantic-ai globally rather than per-agent means a sub-agent
    added later is traced without anyone remembering to opt it in.
    """
    global _client
    if _client is not None:
        return True
    if client is None:
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            return False
        client = get_client()
        # `auth_check` reaches the network, so it fails the way the network does
        # rather than by returning False -- a self-hosted Langfuse that is simply
        # not running raises `ConnectError` out of httpx. This function promises
        # never to raise and a run must not be lost to its own instrumentation, so
        # anything that comes back from that call means tracing is off.
        try:
            if not client.auth_check():
                return False
        except Exception:
            return False

    from pydantic_ai import Agent

    Agent.instrument_all()
    _client = client
    return True


def enabled() -> bool:
    return _client is not None


def label(**metadata: str | None) -> None:
    """Describe the run: domain, model, run name. Empty values are dropped."""
    _labels.update({key: value for key, value in metadata.items() if value})


class _Watched:
    """The object a call site attaches its output to.

    It stands in for a Langfuse observation when tracing is off, and it wraps one
    when tracing is on. Either way it accumulates the node's input and output and
    hands them to `journal` as it closes -- which is what lets the two sinks be
    switched on independently. Langfuse is for reading one conversation; the
    journal is for counting across two hundred. Neither is a substitute.
    """

    def __init__(self, name: str, input: dict[str, Any], inner: Any = None) -> None:
        self._line: dict[str, Any] = {"node": name, "input": input}
        self._inner = inner

    def update(self, **attributes: Any) -> None:
        if self._inner is not None:
            self._inner.update(**attributes)
        self._line.update(attributes)

    def close(self) -> None:
        journal.record(self._line)


@contextmanager
def span(name: str, **input: Any) -> Iterator[Any]:
    """One observation, with the model calls made inside it nested underneath.

    Yields the span so the caller can attach an output once it has one; when
    tracing is off it yields a stand-in, so the call sites read the same either
    way and carry no `if enabled()` branches.
    """
    payload = visible(input)
    if _client is None:
        watched = _Watched(name, payload)
        try:
            yield watched
        finally:
            watched.close()
        return
    with _client.start_as_current_observation(name=name, input=payload) as observation:
        watched = _Watched(name, payload, observation)
        try:
            yield watched
        finally:
            watched.close()


@contextmanager
def session(thread: str) -> Iterator[None]:
    """Group the traces of one conversation. Enter this *before* the root span:
    session is a property of the trace, and the trace is opened by that span.

    The journal is entered here too, and unconditionally: it needs the same
    grouping, and a line that cannot say which conversation it came from is not
    worth writing.
    """
    with journal.session(thread):
        if _client is None:
            yield
            return
        with propagate_attributes(
            session_id=thread,
            metadata=_labels or None,
            tags=sorted(_labels.values()) or None,
        ):
            yield


def visible(payload: dict[str, Any]) -> dict[str, Any]:
    """What is worth reading: the small fields, and only the ones with something
    in them. An empty list on every span is noise that hides the full one."""
    return {
        key: value
        for key, value in payload.items()
        if key not in BULK and value not in (None, "", [], {})
    }


def flush() -> None:
    """Send what is buffered. Worth calling at the end of a run: the exporter
    batches, and a benchmark that exits promptly can outrun it."""
    if _client is not None:
        _client.flush()


def shutdown() -> None:
    """Flush, stop the exporter, close the journal, and leave tracing off. Idempotent."""
    global _client
    journal.shut()
    if _client is None:
        return
    from pydantic_ai import Agent

    Agent.instrument_all(False)
    _client.shutdown()
    _client = None
    _labels.clear()
