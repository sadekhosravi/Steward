"""The decision journal: the sink the offline measurements read. No network."""

from __future__ import annotations

import json

import pytest

import tracing
from tracing import journal


@pytest.fixture(autouse=True)
def _closed():
    """Never leave a file open across tests -- the module holds one globally."""
    journal.shut()
    yield
    journal.shut()


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_nothing_is_written_unless_a_path_is_named(monkeypatch, tmp_path):
    monkeypatch.delenv("STEWARD_JOURNAL", raising=False)
    assert journal.open_from_env() is False
    assert journal.writing() is False
    with tracing.session("sim-1"), tracing.span("gate", calls=[{"name": "book_reservation"}]):
        pass
    assert list(tmp_path.iterdir()) == []


def test_a_node_writes_one_line_carrying_its_input_and_its_ruling(tmp_path):
    path = tmp_path / "decisions.jsonl"
    assert journal.open_at(str(path)) is True
    with tracing.session("sim-24"):
        with tracing.span("gate", calls=[{"name": "transfer_to_human_agents"}]) as span:
            span.update(output={"blocked": 1})
    journal.shut()

    (line,) = _lines(path)
    assert line["thread"] == "sim-24"
    assert line["node"] == "gate"
    assert line["input"]["calls"] == [{"name": "transfer_to_human_agents"}]
    assert line["output"] == {"blocked": 1}


def test_a_refusal_is_written_even_though_it_never_becomes_a_tool_call(tmp_path):
    """The whole reason this file exists: the trajectory would show nothing here."""
    path = tmp_path / "decisions.jsonl"
    journal.open_at(str(path))
    with tracing.session("sim-24"):
        with tracing.span("gate", calls=[{"name": "transfer_to_human_agents"}]) as span:
            span.update(output={"denied": {"call-1": "A change is still owed."}, "approved": []})
    journal.shut()

    (line,) = _lines(path)
    assert line["output"]["approved"] == []
    assert "owed" in line["output"]["denied"]["call-1"]


def test_bulk_state_is_left_out_so_the_file_stays_readable(tmp_path):
    path = tmp_path / "decisions.jsonl"
    journal.open_at(str(path))
    with tracing.session("sim-1"), tracing.span("think", messages=["..."], policy="...", turns=2):
        pass
    journal.shut()

    (line,) = _lines(path)
    assert line["input"] == {"turns": 2}


def test_a_line_written_outside_a_session_still_says_which_thread_it_is_not(tmp_path):
    path = tmp_path / "decisions.jsonl"
    journal.open_at(str(path))
    with tracing.span("plan", turns=1):
        pass
    journal.shut()

    assert _lines(path)[0]["thread"] == ""


def test_something_with_no_json_form_is_written_as_text_rather_than_lost(tmp_path):
    """A node that hands back an object nobody anticipated still leaves a line.

    The alternative -- refusing to write it -- loses the one record of a decision
    at exactly the moment something unexpected happened, which is when it is worth
    the most.
    """
    path = tmp_path / "decisions.jsonl"
    journal.open_at(str(path))
    with tracing.session("sim-1"):
        with tracing.span("gate", turns=1) as span:
            span.update(output=object())
        with tracing.span("plan", turns=2):
            pass
    journal.shut()

    written = _lines(path)
    assert [line["node"] for line in written] == ["gate", "plan"]
    assert isinstance(written[0]["output"], str)


def test_a_path_that_cannot_be_opened_leaves_the_journal_off(tmp_path):
    assert journal.open_at(str(tmp_path)) is False
    assert journal.writing() is False


def test_shutting_twice_is_allowed(tmp_path):
    journal.open_at(str(tmp_path / "d.jsonl"))
    journal.shut()
    journal.shut()
    assert journal.writing() is False
