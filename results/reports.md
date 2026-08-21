# Benchmark reports

One entry per benchmark run. Each entry records **where the architecture stood
at the time** and **what it scored**, so that every later number has something
to be compared against. A score on its own says nothing; a score next to the
run before it is a result.

Raw tau2 output lives beside this file in `results/` and is gitignored — it is
bulky and reproducible. This file is tracked.

## Conventions

- **Reward** is outcome-based: the product of the components in the task's
  `evaluation_criteria.reward_basis` (default DB + COMMUNICATE). Binary per
  simulation.
- **Pass^k** counts a task as passed only if *all* k trials passed. Pass^4 is
  the headline number, because it measures consistency, which is what an
  architecture is supposed to buy.
- The **user simulator model is part of the experiment**. Changing it
  invalidates comparison with earlier runs, so it is recorded every time.
- Every run pins agent model, user model, temperature and reasoning effort. A
  delta is only a result if nothing else moved.

---

## Run 001 — baseline: bare ReAct, no gates

**Date:** 2026-08-21
**Status:** in progress

### Architecture at this point

Two of the six designed runtime nodes exist:

| Node | Status |
|---|---|
| GUIDE (workflow-ordered prompting) | not built |
| ACTOR (ReAct loop) | **built** — `src/agents/assistant.py` |
| PRE-GATE (deterministic provenance/conduct check) | not built |
| GATE (dialogue-grounded write verifier) | not built |
| COMMIT (verbatim emission) | not built |
| CLOSE (deterministic obligation ledger) | not built |
| COMPILER (offline policy → checklists + workflow graph) | not built |

So this run measures a **single agent in a plain ReAct loop**: the whole domain
policy goes into the system prompt, the model calls tools until it decides to
reply, and nothing checks it. This is the floor every later node has to beat.

The LangGraph kernel (`src/core/kernel.py`) supplies the loop: `think` decides,
`act` yields tool calls through `interrupt()`, tau2 executes them, the results
resume the graph. `src/adapters/tau2/` is pure translation.

### Configuration

| | |
|---|---|
| Domain | `airline` (50 tasks) |
| Agent | `mas` |
| Agent model | `nvidia:openai/gpt-oss-20b`, temperature 0.0, reasoning effort `low` |
| User model | `nvidia_nim/openai/gpt-oss-20b`, temperature 0.0 |
| Trials | 4 |
| Max steps | 200 (tau2 default) |
| Concurrency | 8 |

```
uv run python scripts/run_bench.py run \
  --domain airline --agent mas \
  --agent-llm openai/gpt-oss-20b \
  --agent-llm-args '{"temperature":0.0,"reasoning_effort":"low"}' \
  --user-llm nvidia_nim/openai/gpt-oss-20b \
  --user-llm-args '{"temperature":0.0}' \
  --num-trials 4 --max-concurrency 8 \
  --save-to results/baseline_airline_gpt-oss-20b.json
```

### Result

_pending — filled in when the run completes._

### Notes

**`openai/gpt-oss-120b` was the intended model and was rejected on latency.**
It is in the NVIDIA Build catalog and answers correctly, but the endpoint is
too slow to benchmark against: measured 91.4 s for a 19-token reply, then two
consecutive 200 s read timeouts on the same prompt. At default reasoning effort
single calls took 124–362 s. `gpt-oss-20b` on the same endpoint answers the
same prompt in 0.7 s. The bottleneck is that model's capacity on NVIDIA Build,
not our settings — OpenRouter would be the alternative host, but no
`OPENROUTER_API_KEY` is configured.

**Reasoning effort is now a first-class setting** (`MAS_LLM_REASONING_EFFORT`,
`--agent-llm-args '{"reasoning_effort": ...}'`), added while diagnosing the
above. Open-weight reasoning models think for a very long time by default and
the difference is two orders of magnitude in wall-clock.
