"""What a run leaves behind in Langfuse.

No server: the client exports into memory. What is being checked is the shape of
what we send -- that every node is on the record, that a verdict can be read back
without inferring it from side effects, and that one conversation is one session.
"""

from __future__ import annotations

import json

import pytest
from langfuse import Langfuse, LangfuseOtelSpanAttributes
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import tracing
from tests.test_gate import approves, blocks_once, kernel, proposes_a_cancellation


@pytest.fixture(scope="module")
def exporter():
    """One client for the module: Langfuse installs a tracer provider, and a
    second one would quietly keep exporting to the first."""
    collected = InMemorySpanExporter()
    tracing.setup(
        Langfuse(public_key="pk-lf-test", secret_key="sk-lf-test", span_exporter=collected)
    )
    yield collected
    tracing.shutdown()


@pytest.fixture
def spans(exporter):
    exporter.clear()
    yield exporter
    exporter.clear()


def _finished(exporter):
    tracing.flush()
    return exporter.get_finished_spans()


def _attribute(span, key: str):
    value = span.attributes.get(key)
    return json.loads(value) if value else None


def _named(exporter, name: str):
    return [s for s in _finished(exporter) if s.name == name]


def test_every_decision_the_kernel_makes_is_on_the_record(spans):
    kernel(proposes_a_cancellation, approves).send("t", "cancel HKD3PS")

    names = {s.name for s in _finished(spans)}
    assert {"message", "think", "gate"} <= names


def test_the_gate_span_carries_the_verdict(spans):
    """Run 002 could only estimate the block rate from how many writes came out
    the other side. A refusal now says so, in the place it was made."""
    kernel(proposes_a_cancellation, blocks_once()).send("t", "cancel HKD3PS")

    refusal, approval = _named(spans, "gate")
    assert "The reservation was never looked up." in str(
        _attribute(refusal, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT)["denied"]
    )
    assert _attribute(approval, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT)["approved"]


def test_a_step_records_what_arrived_and_what_was_decided(spans):
    kernel(proposes_a_cancellation, approves).send("t", "cancel HKD3PS")

    (step,) = _named(spans, "message")
    assert _attribute(step, LangfuseOtelSpanAttributes.OBSERVATION_INPUT) == {
        "text": "cancel HKD3PS"
    }
    assert _attribute(step, LangfuseOtelSpanAttributes.OBSERVATION_OUTPUT)["calls"]


def test_one_conversation_is_one_session(spans):
    """Each step is its own trace -- emitting tool calls returns control to the
    harness -- so the session is what holds a simulation together."""
    k = kernel(proposes_a_cancellation, approves)
    k.send("thread-42", "cancel HKD3PS")

    sessions = {
        s.attributes.get(LangfuseOtelSpanAttributes.TRACE_SESSION_ID) for s in _finished(spans)
    }
    assert sessions == {"thread-42"}


def test_the_bulk_of_the_state_is_left_off():
    """The message history is on the generation spans in full; sending it again
    on every node of every step would be the same text many times over."""
    assert tracing.visible({"messages": [1], "observed": ["x"], "calls": [1], "denied": {}}) == {
        "calls": [1]
    }
