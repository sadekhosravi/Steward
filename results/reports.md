# Benchmark reports

One entry per benchmark run. Each entry records **where the architecture stood
at the time** and **what it scored**, so that every later number has something
to be compared against. A score on its own says nothing; a score next to the
run before it is a result.

Every run also has a one-line entry in [`log.md`](log.md), for the trend at a
glance.

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
**Status:** complete

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
| Agent | `steward` |
| Agent model | `nvidia:openai/gpt-oss-20b`, temperature 0.0, reasoning effort `low` |
| User model | `nvidia_nim/openai/gpt-oss-20b`, temperature 0.0 |
| Trials | 4 |
| Max steps | 200 (tau2 default) |
| Concurrency | 8 |

```
uv run python scripts/run_bench.py run \
  --domain airline --agent steward \
  --agent-llm openai/gpt-oss-20b \
  --agent-llm-args '{"temperature":0.0,"reasoning_effort":"low"}' \
  --user-llm nvidia_nim/openai/gpt-oss-20b \
  --user-llm-args '{"temperature":0.0}' \
  --num-trials 4 --max-concurrency 8 \
  --save-to results/baseline_airline_gpt-oss-20b.json
```

### Result

| Metric | Value |
|---|---|
| Average reward | **0.365** |
| Pass^1 | 0.365 |
| Pass^2 | 0.260 |
| Pass^3 | 0.200 |
| **Pass^4** | **0.160** |
| Tasks solved on all 4 trials | 8 / 50 |
| Tasks never solved | 23 / 50 |

Reward components, over the 196 simulations that produced a verdict:

| Component | Pass rate |
|---|---|
| COMMUNICATE | 174/196 (88.8%) |
| DB | 77/196 (39.3%) |

Recall against the gold trajectory's actions, by tool type:

| Tool type | Gold actions performed |
|---|---|
| read | 156 / 361 (43.2%) |
| **write** | **11 / 195 (5.6%)** |

Wall clock: median simulation 40.9 s, p90 133 s, max 2050 s. Cost is not
reported: LiteLLM has no price map for `nvidia_nim`, so tau2 records
`agent_cost = None`. All 196 finished simulations ended in `user_stop`.

**Four simulations never terminated and are scored 0**: tasks `6` (trials 0
and 3), `35` (trial 3), `45` (trial 2). They ran for 35–95 minutes against a
median of 41 s. Two of the four are the same task, so this is a property of
the task, not sampling noise. An unfinished simulation is a failure — the
agent never handed control back — so counting them as 0 is the honest
treatment, not a penalty.

### Reading of this result

The headline is the read/write split. The agent performs **43% of the gold
read actions but 5.6% of the gold writes**, and DB is the component that
fails (39% vs COMMUNICATE's 89%). It talks to the user acceptably and then
does the wrong thing to the database — which is precisely the failure the
architecture is built to prevent, and it is concentrated exactly where the
design puts the gate.

Pass^4 at 0.160 against Pass^1 at 0.365 means **more than half of the tasks it
can solve, it cannot solve reliably**. Consistency, not capability, is the gap.

The four hung simulations are the second finding: an unbounded ReAct loop has
no deterministic stop. The design already calls for a revision cap that fails
toward asking the user; this run is the evidence that it is not optional.

### Notes

**`openai/gpt-oss-120b` was the intended model and was rejected on latency.**
It is in the NVIDIA Build catalog and answers correctly, but the endpoint is
too slow to benchmark against: measured 91.4 s for a 19-token reply, then two
consecutive 200 s read timeouts on the same prompt. At default reasoning effort
single calls took 124–362 s. `gpt-oss-20b` on the same endpoint answers the
same prompt in 0.7 s. The bottleneck is that model's capacity on NVIDIA Build,
not our settings — OpenRouter would be the alternative host, but no
`OPENROUTER_API_KEY` is configured.

**Reasoning effort is now a first-class setting** (`STEWARD_LLM_REASONING_EFFORT`,
`--agent-llm-args '{"reasoning_effort": ...}'`), added while diagnosing the
above. Open-weight reasoning models think for a very long time by default and
the difference is two orders of magnitude in wall-clock.

---

## Run 002 — GATE: an agentic critic on every write

**Date:** 2026-08-21
**Status:** complete, cut at 198/200 (see Notes)

### Architecture at this point

| Node | Status |
|---|---|
| GUIDE (workflow-ordered prompting) | not built |
| ACTOR (ReAct loop) | built — `src/agents/assistant.py` |
| PRE-GATE (deterministic provenance check) | **built** — `agents.gate.findings`, reported to GATE as evidence, not as a verdict |
| GATE (dialogue-grounded write verifier) | **built** — `src/agents/gate.py` |
| COMMIT (verbatim emission) | **built** — structural: `gate` writes `approved`, `act` emits only `approved` |
| CLOSE (deterministic obligation ledger) | not built |
| COMPILER (offline policy → checklists) | not built |

The graph is now `think -> gate -> act -> think -> ... -> END`, with `gate`
returning to `think` on a refusal and to `escalate` after `REVISION_LIMIT` (2)
refusals in one user turn. Writes are identified from tau2's own
`mutates_state` label carried across the seam in `ToolDefinition.metadata`, so
no hand-maintained list is involved. A step containing no write skips the gate
without a model call.

### Configuration

Identical to Run 001 in every respect except the gate. The gate runs on the
same model as the actor (`STEWARD_GATE_MODEL` unset).

| | |
|---|---|
| Domain | `airline` (50 tasks) |
| Agent model | `nvidia:openai/gpt-oss-20b`, temperature 0.0 (**ignored**, see Notes), reasoning effort `low` |
| Gate model | same as agent |
| User model | `nvidia_nim/openai/gpt-oss-20b`, temperature 0.0 |
| Trials | 4 |
| Concurrency | 8 |

### Result

| Metric | Run 001 | Run 002 | Δ |
|---|---|---|---|
| Average reward | 0.365 | **0.420** | +0.055 |
| Pass^1 | 0.360 | 0.340 | −0.020 |
| Pass^2 | 0.260 | 0.280 | +0.020 |
| Pass^3 | 0.200 | 0.260 | +0.060 |
| **Pass^4** | **0.160** | **0.220** | **+0.060** |
| Solved on all 4 trials | 8 / 50 | 11 / 50 | +3 |
| Never solved | 23 / 50 | 22 / 50 | −1 |

Reward components:

| Component | Run 001 | Run 002 |
|---|---|---|
| COMMUNICATE | 174/196 (88.8%) | 173/197 (87.8%) |
| **DB** | 77/196 (39.3%) | **92/197 (46.7%)** |

Actions, against the gold trajectory:

| | Run 001 | Run 002 |
|---|---|---|
| Gold read recall | 159/369 (43.1%) | 139/367 (37.9%) |
| Gold write recall | 11/195 (5.6%) | 8/194 (4.1%) |
| **Write calls emitted** | **240** | **125** |
| Read calls emitted | 618 | 614 |

Wall clock: median 35 s (was 41 s), p90 337 s (was 133 s), max 2797 s, 6.1 h
of simulation time.

### Reading of this result

**The gate blocked roughly half of every write the actor proposed** — 240
emitted in Run 001 against 125 here, on near-identical read volume. This is the
only direct evidence we have of its block rate, because gate verdicts are not
currently observable (see Notes).

**It did not make the agent write more correctly; it made it write less.** Gold
write recall fell, 11 correct writes to 8. DB rose 7.4 points anyway. With ~95%
of proposed writes wrong, abstention is worth more than attempts: the three
correct writes lost were bought back many times over by the incorrect ones
never made. Net +15 simulations passing DB.

**Pass^4 rose 0.160 → 0.220, +37% relative, with COMMUNICATE flat.**
Consistency was the gap Run 001 identified, the gate was the intervention aimed
at it, and it moved without damaging the conversation — the failure mode a
badly-tuned critic would have produced. Pass^1 fell 0.02 while Pass^3 and
Pass^4 rose 0.06: the gate trades a little peak capability for materially more
reliability, which is the trade the project exists to make.

**The ceiling is now visible.** Blocking is a blunt instrument — it converts a
wrong write into no write, which scores only when the task did not require that
write. Getting the write *right* needs the gate to check discrete conditions
rather than re-read a policy manual, which is COMPILER's job.

### Notes

**Cut at 198/200.** Task 6 trial 0 and task 45 trial 3 were still running after
75 minutes and were stopped; both are scored 0, the same treatment Run 001 gave
its four. The pathological tasks are the same across both runs — 6, 35, 45 —
which makes this a property of those tasks, not sampling. Task 45 takes 15–47
minutes and then *succeeds*; task 35 burns the same time and fails.

**`temperature: 0.0` is silently discarded on this model.** pydantic-ai emits
`UserWarning: Sampling parameters ['temperature'] are not supported when
reasoning is enabled`. Both runs sampled at the provider default. **No result in
this file is a controlled comparison in the strict sense**, and we have no
variance estimate to say how much of the +0.055 is real. Establishing that noise
floor is now a prerequisite for trusting any future delta.

**GATE can crash a simulation.** One `infrastructure_error` (task 32 trial 0):
`UnexpectedModelBehavior: Exceeded maximum output retries (1)`, raised inside
the `gate` node. A union output type gives a 20B model two output tools to pick
between and `output_retries=1` makes one fumble fatal. Replaying real writes
through the gate offline reproduced it in 3 of 8 cases, so the live rate of 1 in
198 is luck, not robustness. This is the worst failure mode available: the
simulation dies and scores 0.

**Gate verdicts are not observable.** A refusal becomes a `ModelRetry` in the
pydantic-ai history held in LangGraph state; tau2's `results.json` records only
tau2 messages, so no verdict ever reaches it. Block counts in this report are
inferred from emitted write volume. Instrumentation is required before the gate
can be tuned.
