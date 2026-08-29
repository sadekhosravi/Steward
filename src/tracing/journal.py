"""A local record of what each node was given and what it decided.

Langfuse answers "what happened in this one conversation" well and is the right
tool when a person is reading. It is the wrong tool for the other question, which
is the one this project keeps asking: *across 150 saved simulations, how often
did the gate refuse a handoff while a change was outstanding?* That needs the
decisions on disk, next to the trajectories, greppable, with no server running.

The gap is not hypothetical. Every offline measurement in `results/log.md` is
computed from tau2's saved trajectories, and a trajectory holds only what left
the system: the customer's turns, the assistant's, and the tool calls that were
actually emitted. A proposal the gate refused never becomes a tool call, so it
leaves no trace at all -- which means the whole internal half of this system is
unmeasurable after the fact, and every claim about why a turn went wrong has been
read off the assistant's prose rather than off the ruling. Task 24 fails in all
three trials by transferring while a booking is still owed, and whether the
planner still held that change at the moment of the transfer decides which of two
different fixes is the right one. There is currently no way to find out.

So: one JSON line per graph node, written as the node closes.

It is off unless `STEWARD_JOURNAL` names a path, and it is opened from an entry
point rather than from library code, exactly like `setup()` -- a test run writes
nothing. Failure to write is swallowed for the same reason tracing's is: a run
must not be lost to its own instrumentation.
"""

from __future__ import annotations

import json
import os
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python

__all__ = ["open_from_env", "record", "session", "shut", "writing"]

_file: Any = None

# Appends from several simulations interleave -- tau2 runs them on threads, and
# `--max-concurrency 8` is the normal setting. One lock around the write keeps
# lines whole; without it a partial line makes the rest of the file unreadable to
# a reader that parses per line, which is the only way anyone reads this.
_lock = threading.Lock()

# Which conversation a line belongs to. A ContextVar rather than an argument
# because the thread id is known in `Kernel._run` and needed in `_traced`, and
# threading it through every node signature would put a tracing concern in the
# type of every graph node.
_thread: ContextVar[str] = ContextVar("thread", default="")


def open_from_env() -> bool:
    """Start writing if `STEWARD_JOURNAL` names a path. Returns whether it did."""
    path = os.environ.get("STEWARD_JOURNAL")
    if not path:
        return False
    return open_at(path)


def open_at(path: str) -> bool:
    """Start writing to `path`, appending. Idempotent; never raises."""
    global _file
    if _file is not None:
        return True
    try:
        # Line buffered. A benchmark run is the thing most likely to be killed
        # part way through and the thing whose tail is most worth having: the
        # 15x2 of 2026-08-29 was stopped by hand and lost whatever was still in
        # an 8KB buffer. One line per node is a small enough write that flushing
        # each one costs nothing against a model call.
        _file = Path(path).open("a", encoding="utf-8", buffering=1)
    except OSError:
        return False
    return True


def writing() -> bool:
    return _file is not None


def session(thread: str) -> object:
    """Tag every line written under this context with the conversation."""
    return _Session(thread)


class _Session:
    def __init__(self, thread: str) -> None:
        self._thread = thread
        self._token: Any = None

    def __enter__(self) -> None:
        self._token = _thread.set(self._thread)

    def __exit__(self, *_: object) -> None:
        _thread.reset(self._token)


def record(line: dict[str, Any]) -> None:
    """Append one decision. Never raises: instrumentation may not end a run."""
    if _file is None:
        return
    try:
        text = json.dumps(
            {"thread": _thread.get(), **line}, default=lambda o: to_jsonable_python(o, fallback=str)
        )
    except (TypeError, ValueError):
        return
    try:
        with _lock:
            _file.write(text + "\n")
    except (OSError, ValueError):
        pass


def shut() -> None:
    """Close the file. Worth calling at the end of a run; safe to call twice."""
    global _file
    if _file is None:
        return
    try:
        _file.close()
    finally:
        _file = None
