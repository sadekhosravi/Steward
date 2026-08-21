"""What crosses the tau2 seam, and what the gate is told about it.

Only one fact travels with a tool definition: whether the Kernel has to have it
reviewed before it runs. tau2 answers a narrower question than the one we need --
`mutates_state` is about the database, and the action that ends the conversation
touches no database at all -- so the widening happens here and is pinned here.
"""

from __future__ import annotations

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
