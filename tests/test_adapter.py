"""What crosses the tau2 seam, and what the gate is told about it.

Only one fact travels with a tool definition: whether the Kernel has to have it
reviewed before it runs. tau2 answers a narrower question than the one we need --
`mutates_state` is about the database, and the action that ends the conversation
touches no database at all -- so the widening happens here and is pinned here.
"""

from __future__ import annotations

from pydantic import BaseModel
from tau2.environment.tool import Tool
from tau2.environment.toolkit import is_tool

from adapters.tau2.agent import HANDOFF, _tool_def
from core.kernel import _gated


@is_tool("read")
def get_reservation(reservation_id: str) -> str:
    """Look up a reservation.

    Args:
        reservation_id: The reservation to look up.
    """
    return "{}"


@is_tool("write")
def cancel_reservation(reservation_id: str) -> str:
    """Cancel a reservation.

    Args:
        reservation_id: The reservation to cancel.
    """
    return "{}"


@is_tool("read")
def transfer_to_human_agents(summary: str) -> str:
    """Transfer the user to a human agent.

    Args:
        summary: Why the transfer is needed.
    """
    return "Transfer successful"


def gated(*funcs) -> frozenset[str]:
    return _gated([_tool_def(Tool(f)) for f in funcs])


def test_a_read_is_not_gated():
    assert gated(get_reservation) == frozenset()


def test_a_write_is_gated():
    assert gated(cancel_reservation) == {"cancel_reservation"}


def test_the_handoff_is_gated_even_though_tau2_calls_it_a_read():
    """The regression this exists for.

    tau2 labels `transfer_to_human_agents` `mutates_state=False`, so a gate keyed
    on writes alone never reviewed it -- and the actor reached for it in 32 of 50
    simulations. Ending the conversation is as irreversible as any write.
    """
    assert transfer_to_human_agents.__mutates_state__ is False
    assert gated(transfer_to_human_agents) == {"transfer_to_human_agents"}


def test_every_handoff_name_is_one_tau2_actually_uses():
    """A name that no longer matches upstream would silently stop gating.

    The airline toolkit is the cheapest place to see the real spelling, and it is
    the same in every domain.
    """
    from tau2.domains.airline.tools import AirlineTools

    assert HANDOFF <= {name for name in dir(AirlineTools) if not name.startswith("_")}


class Record(BaseModel):
    """A return type with fields worth naming."""

    record_id: str
    balance: int


@is_tool("read")
def get_record(record_id: str) -> Record:
    """Look up a record.

    Args:
        record_id: The record to look up.
    """
    raise NotImplementedError


@is_tool("read")
def search_records(query: str) -> list[Record]:
    """Search records.

    Args:
        query: What to search for.
    """
    raise NotImplementedError


def description(func) -> str:
    return _tool_def(Tool(func)).description


def test_the_model_is_told_what_comes_back():
    """The regression this exists for.

    tau2's own line for `get_user_details` mentions reservations and stops there,
    so an agent asked for a gift card balance concluded no tool could give it one
    and said so -- with the balances sitting in the record that tool returns. A
    field it cannot see until after the call is a field it will not call for.
    """
    assert description(get_record) == (
        "Look up a record.\nReturns an object with: record_id, balance."
    )


def test_a_list_return_says_so():
    assert description(search_records).endswith(
        "Returns a list, each entry with: record_id, balance."
    )


def test_a_tool_with_nothing_to_say_says_nothing():
    """A `str` return has no fields, and an empty `Returns:` is worse than none."""
    assert description(get_reservation) == "Look up a reservation."


def test_tau2_no_longer_tells_the_model_to_transfer_when_it_is_stuck():
    """tau2's text licenses the bail-out the gate exists to prevent.

    A tool description argues with the system prompt from closer range: it sits
    beside the tool the model is reaching for. Ours has to state the same rule.
    """
    from tau2.domains.airline.tools import AirlineTools

    assert "cannot solve" in (AirlineTools.transfer_to_human_agents.__doc__ or "")
    assert "cannot solve" not in description(transfer_to_human_agents)
    assert "last resort" in description(transfer_to_human_agents)


def test_the_status_tool_says_what_it_does_not_return():
    """The one read whose return type carries no fields, and so the one the
    generated sentence cannot help. Task 44 called it five times looking for
    departure times, decided no tool had them, and transferred."""
    from tau2.domains.airline.environment import get_environment

    status = next(t for t in get_environment().get_tools() if t.name == "get_flight_status")
    said = description(status)
    assert "no times" in said
    assert "search_direct_flight" in said
