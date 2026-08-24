"""Cutting a leaked harmony control token out of a tool name.

The names here are the ones that actually killed simulations: `book_reservation`
on task 31 of the 50-task run, `transfer_to_human_agents` on task 37.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from llm.harmony import repaired


def response(*parts):
    return ModelResponse(parts=list(parts))


def names(result):
    return [p.tool_name for p in result.parts if isinstance(p, ToolCallPart)]


def test_a_leaked_channel_marker_is_cut_off_the_name():
    """Task 31, as it happened: two of these ended the run as an infrastructure
    error with no reward at all."""
    call = ToolCallPart(tool_name="book_reservation<|channel|>commentary", args={"x": 1})

    assert names(repaired(response(call))) == ["book_reservation"]


def test_the_arguments_and_the_call_id_survive_the_repair():
    """The id is the routing key back to the requesting sub-agent, so losing it
    would trade one failure for a worse one."""
    call = ToolCallPart(
        tool_name="transfer_to_human_agents<|channel|>commentary",
        args={"summary": "done"},
        tool_call_id="call_42",
    )

    fixed = repaired(response(call)).parts[0]

    assert (fixed.tool_name, fixed.args, fixed.tool_call_id) == (
        "transfer_to_human_agents",
        {"summary": "done"},
        "call_42",
    )


def test_a_clean_response_is_returned_exactly_as_it_came():
    """Every other provider pays a scan of the parts and nothing else."""
    clean = response(TextPart(content="hello"), ToolCallPart(tool_name="get_user_details", args={}))

    assert repaired(clean) is clean


def test_a_name_that_is_only_a_marker_is_left_alone():
    """There is no call to recover, and truncating it to the empty string turns a
    bad name into a confusing one."""
    call = ToolCallPart(tool_name="<|channel|>commentary", args={})

    assert names(repaired(response(call))) == ["<|channel|>commentary"]


def test_text_beside_a_broken_call_is_not_touched():
    parts = repaired(
        response(
            TextPart(content="one moment"), ToolCallPart(tool_name="calculate<|end|>", args={})
        )
    ).parts

    assert parts[0].content == "one moment"
    assert parts[1].tool_name == "calculate"


def test_every_broken_call_in_one_response_is_repaired():
    result = repaired(
        response(
            ToolCallPart(tool_name="calculate<|channel|>commentary", args={}),
            ToolCallPart(tool_name="get_reservation_details", args={}),
            ToolCallPart(tool_name="cancel_reservation<|end|>", args={}),
        )
    )

    assert names(result) == ["calculate", "get_reservation_details", "cancel_reservation"]


def test_the_repair_is_wired_into_every_model_the_system_builds():
    """Wiring, and worth its own test: nothing above `llm` knows this format
    exists, so a regression here is invisible until a run dies."""
    import os

    from llm.config import ModelSpec
    from llm.harmony import Harmonised
    from llm.providers import build_model

    os.environ.setdefault("NVIDIA_API_KEY", "test-key-not-used")
    model = build_model(ModelSpec(provider="nvidia", model="openai/gpt-oss-20b"))

    assert isinstance(model, Harmonised)
