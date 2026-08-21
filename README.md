# Steward

A customer service agent that can be trusted with the records. It serves
customers under a company policy it must actually follow, and no action
reaches the database that a second agent has not authorised. Evaluated
against [tau2-bench](https://github.com/sierra-research/tau2-bench).

tau2 is consumed as a **pinned upstream library**, never forked. It supplies the
domains, policies, tools, user simulator and scoring; everything about how the
agent is *built* lives in this repository.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (tau2 requires
`>=3.12,<3.14`).

```bash
uv sync                                # create .venv, install tau2 at its pinned rev
uv run python scripts/bootstrap_data.py  # fetch benchmark data into vendor/ (~145M)
cp .env.example .env                   # then fill in TAU2_DATA_DIR + provider keys
uv run tau2 check-data                 # should report "Data directory exists"
```

`TAU2_DATA_DIR` is required. tau2 normally resolves its data as
`<repo root>/data`, which does not exist when it is installed as a library, so
the variable points at the checkout `bootstrap_data.py` creates. tau2 calls
`load_dotenv()` itself at import time, so a `.env` at the repo root is picked up
without any explicit wiring.

Simulation results are written to `$TAU2_DATA_DIR/simulations/<run_name>`.

### Models

Each agent picks its own model, so a cheap fast one and a slower stronger one
can coexist in the same run:

```python
import llm
from pydantic_ai import Agent

investigator = Agent(model=llm.get_model("openai/gpt-oss-20b"))
critic = Agent(
    model=llm.get_model("anthropic/claude-sonnet-4.5", provider="openrouter"),
    output_type=Verdict,
)
```

`get_model` takes `provider`, `model`, `temperature`, `timeout` and
`max_tokens`; anything omitted falls back to the `STEWARD_LLM_*` defaults in
`.env`. Settings ride on the returned model, so `Agent` needs one argument.

Two independent paths pick models, and they name the same model differently:

| | Configured by | Provider layer |
|---|---|---|
| Steward agents | `llm.get_model(...)`, defaults from `STEWARD_LLM_*` | pydantic-ai |
| tau2's user simulator | `tau2 run --llm-user` | LiteLLM |

`nvidia` (NVIDIA Build) and `openrouter` are supported. Being listed in a
catalog does not mean a model is served, or that it does native tool calling
and structured output -- both of which the Critic and Proposer require. Check
before adopting one:

```bash
uv run python scripts/probe_models.py config               # defaults, key presence
uv run python scripts/probe_models.py list --filter gpt    # provider catalog
uv run python scripts/probe_models.py check --model <id>   # chat + structured + tools
```

### Windows note

The default console codepage cannot encode the emoji tau2 prints. Set
`PYTHONUTF8=1` (it must be set before the interpreter starts, so `.env` is too
late for it):

```powershell
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

## Layout

```
src/
  core/            the Kernel (LangGraph graph) and harness-agnostic primitives
  agents/          sub-agent implementations (planner, router, critic, ...)
  patterns/        multi-agent topologies, swappable for comparison
  tools/           private tools, invisible to the benchmark
  llm/             provider access, prompt assembly, cost accounting
  tracing/         internal tracing (the harness only records the outer trajectory)
  adapters/tau2/   HalfDuplexAgent subclass, factory, type translation
scripts/           bootstrap and evaluation entry points
configs/           run and topology configuration
tests/
vendor/            pinned benchmark data (gitignored)
```

Two rules hold the design together:

1. `core` never imports tau2, so sub-agents stay unit-testable without
   booting a benchmark environment and the patterns stay portable.
2. Every environment tool call goes out through `adapters.tau2` as
   `tool_calls` on the emitted message. Sub-agents never invoke a tau2 `Tool`
   object directly -- those are live callables bound to the environment, and
   calling one in-process mutates the scored database without appearing in the
   trajectory.

## Running the benchmark

`scripts/run_bench.py` is tau2's own CLI with our agent registered first --
tau2 has no plugin discovery, so registration has to happen in the same process
before the runner starts. Every `tau2 ...` command works through it.

```bash
uv run python scripts/run_bench.py run --domain mock --agent steward \
    --agent-llm openai/gpt-oss-20b --user-llm nvidia_nim/openai/gpt-oss-20b
```

`--agent-llm` takes a plain model id (resolved by `llm`); `--user-llm` takes a
LiteLLM name, because tau2's user simulator is on a separate provider path.

## How a turn runs

The Kernel is a LangGraph graph with two nodes. `think` asks a pydantic-ai
agent what to do; `act` is a bare `interrupt()`. If the agent wants tool calls,
the graph pauses, the adapter emits them as `tool_calls`, tau2 runs them
against the real environment, and `resume()` continues inside `act` with the
results. If the agent has something to say instead, the turn ends.

This works because tau2's yield semantics and LangGraph's interrupt are the
same shape, so no bookkeeping is needed to get back to where we paused --
which is what will let a tool call originate deep inside a sub-agent later.

The Kernel itself never calls a model. Reward is binary per task and pass^k
only counts a task when every trial passes, so variance in control flow is a
direct score loss.

## Tracing

The benchmark records only the outer trajectory, so everything that makes this a
multi-agent system is invisible in its output. Langfuse fills that in. Put the
keys in `.env` and it turns itself on:

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

`scripts/run_bench.py` prints `Langfuse tracing: on` at startup when it worked.
Leave the keys out, or set `LANGFUSE_TRACING_ENABLED=false`, and nothing is sent.

Two layers, and they nest:

| Layer | Comes from | Records |
|---|---|---|
| model calls | pydantic-ai's OpenTelemetry instrumentation | prompt, response, tool calls, tokens |
| graph nodes | `tracing.span()`, wrapped around each node in `core.kernel` | what the node was given, what it decided |

The second is what makes the first legible: a generation on its own does not say
which agent made it or what the Kernel did with the answer. It also puts the
gate's verdict on the record — Run 002 could only estimate the block rate from
how many writes came out the other end.

One trace per Kernel step, named for what arrived (`message` or `results`), and
the conversation as the session, so a whole simulation reads top to bottom in
Langfuse's session view. A step is the largest unit the Kernel controls end to
end: emitting tool calls hands control back to tau2, so a span cannot be held
across it.

The message history is deliberately left off the node spans. It is already on
the generation spans in full, and it grows with the conversation.

## Updating tau2

Bump `rev` under `[tool.uv.sources]` in `pyproject.toml`, then:

```bash
uv sync
uv run python scripts/bootstrap_data.py   # reads the same rev, keeps data in step
```
