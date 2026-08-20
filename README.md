# MAS

A multi-agent system, evaluated against [tau2-bench](https://github.com/sierra-research/tau2-bench).

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

### Windows note

The default console codepage cannot encode the emoji tau2 prints. Set
`PYTHONUTF8=1` (it must be set before the interpreter starts, so `.env` is too
late for it):

```powershell
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

## Layout

```
src/mas/
  core/            harness-agnostic primitives; must not import tau2
  agents/          sub-agent implementations (planner, router, critic, ...)
  patterns/        multi-agent topologies, swappable for comparison
  tools/           private tools, invisible to the benchmark
  llm/             provider access, prompt assembly, cost accounting
  trace/           internal tracing (the harness only records the outer trajectory)
  adapters/tau2/   HalfDuplexAgent subclass, factory, type translation
scripts/           bootstrap and evaluation entry points
configs/           run and topology configuration
tests/
vendor/            pinned benchmark data (gitignored)
```

Two rules hold the design together:

1. `mas.core` never imports tau2, so sub-agents stay unit-testable without
   booting a benchmark environment and the patterns stay portable.
2. Every environment tool call goes out through `mas.adapters.tau2` as
   `tool_calls` on the emitted message. Sub-agents never invoke a tau2 `Tool`
   object directly -- those are live callables bound to the environment, and
   calling one in-process mutates the scored database without appearing in the
   trajectory.

## Updating tau2

Bump `rev` under `[tool.uv.sources]` in `pyproject.toml`, then:

```bash
uv sync
uv run python scripts/bootstrap_data.py   # reads the same rev, keeps data in step
```
