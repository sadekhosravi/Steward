"""tau2's own CLI, with our agent registered first.

    uv run python scripts/run_bench.py run --domain mock --agent steward \
        --agent-llm openai/gpt-oss-20b --user-llm nvidia_nim/openai/gpt-oss-20b

Every `tau2 ...` command works here; the only difference is that `--agent steward`
resolves. On Windows set PYTHONUTF8=1 first.

Two settings are worth passing on any long run:

  --user-llm-args '{"timeout": 300}'   tau2's user simulator goes through LiteLLM,
which this module does not configure and `tracing` does not instrument, so a call
that never returns leaves no span and stalls the whole run behind one task. One
50-task run lost an hour that way. LiteLLM's own `num_retries` defaults to 3, so
this bounds a turn at roughly four attempts rather than capping it outright; 300s
is above every attempt observed in a full run and well under the stall it is
there to catch. Our own sub-agents need nothing here -- `llm.config` already
defaults them to 120s, and 2528 calls in one run topped out at 103s.

  STEWARD_GATE=off                     approves every proposal without asking the
critic, leaving the plan, the ledger and the speaker exactly where they were. A
measurement arm for whether the critic pays for itself -- see `core.kernel`.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # before tau2 imports: it reads its data directory at import time

# tau2's user simulator goes through LiteLLM, which looks for its own variable
# name. One key, two names, so .env only has to carry the one.
if "NVIDIA_API_KEY" in os.environ:
    os.environ.setdefault("NVIDIA_NIM_API_KEY", os.environ["NVIDIA_API_KEY"])

from tau2.cli import main  # noqa: E402

import adapters.tau2  # noqa: E402
import tracing  # noqa: E402


def _flag(name: str) -> str | None:
    """Read a CLI flag back out of argv, for labelling only.

    tau2 owns the argument parser and we are not going to duplicate it. Nothing
    depends on the answer -- a label that comes back None costs a tag, not a run.
    """
    if name in sys.argv[:-1]:
        return sys.argv[sys.argv.index(name) + 1]
    return None


if __name__ == "__main__":
    adapters.tau2.register()
    # Here rather than in the Kernel: an entry point may decide to send traces
    # somewhere, a library may not. Off unless the Langfuse keys are in .env.
    on = tracing.setup()
    if on:
        # Every session carries these, so a scripted test and a real run are
        # distinguishable in Langfuse without opening either.
        tracing.label(
            domain=_flag("--domain"),
            model=_flag("--agent-llm"),
            run=(_flag("--save-to") or "").split("/")[-1].removesuffix(".json") or None,
        )
    print(f"Langfuse tracing: {'on' if on else 'off'}")
    # The other sink, and the one the offline measurements read. Langfuse answers
    # "what happened in this conversation"; the journal answers "how often, across
    # the whole run" -- and a refused proposal never becomes a tool call, so it is
    # absent from the saved trajectories entirely. Off unless STEWARD_JOURNAL
    # names a path.
    kept = tracing.journal.open_from_env()
    print(f"Decision journal: {os.environ['STEWARD_JOURNAL'] if kept else 'off'}")
    try:
        sys.exit(main())
    finally:
        # The exporter batches, and a run that exits promptly can outrun it.
        tracing.shutdown()
