<h1 align="center">Airlock</h1>

<p align="center">
  <strong>Multi-agent scaffolding and deterministic guardrails for small open-weight models on &tau;&sup2;-bench.</strong>
</p>

<p align="center">
  <em>How much of a customer-service task can a 20B open-weight model actually complete<br/>
  when you stop asking it to be smarter and start constraining what it is allowed to do?</em>
</p>

<p align="center">
  <code>Python 3.12</code> &middot;
  <code>uv</code> &middot;
  <code>pydantic-ai</code> &middot;
  <code>LangGraph</code> &middot;
  <code>&tau;&sup2;-bench (pinned)</code> &middot;
  <code>481 tests</code> &middot;
  <code>3,200+ benchmark simulations</code>
</p>

---

## TL;DR

Airlock wraps a **20B open-weight model** (`gpt-oss-20b`, reasoning effort `low`) in a
four-agent orchestration with a deterministic verifier tier, and evaluates it on the
**&tau;&sup2;-bench airline domain** against the *same model* running &tau;&sup2;'s stock single-agent
baseline.

| | scaffolded (build 017) | unscaffolded baseline | paired difference |
|---|---|---|---|
| **pass^1** | **0.530** | 0.460 | +0.070 &plusmn; 0.046 (1.5 SE) |
| **pass^4** | **0.380** | 0.280 | — |
| **gold-write recall** | **33.2%** | 11.9% | **+0.233 &plusmn; 0.049 (4.8 SE)** |
| **gold-read recall** | **86.5%** | 56.1% | **+0.212 &plusmn; 0.049 (4.3 SE)** |
| needless transfer to human | 42.2% | 49.2% | &minus;0.075 &plusmn; 0.042 (1.8 SE) |
| median latency / simulation | 433 s | 68 s | **6.4&times; slower** |

**The honest headline is not the reward.** The reward difference is real in direction but
**not statistically established** (1.5 SE). What *is* established, at 4&ndash;5 standard errors,
is behavioural: the scaffolded system **completes ~3&times; more of the database writes customers
actually asked for**, and reads ~1.5&times; more of the records it needs, while abstaining less
often.

That gap between "scores about the same" and "does three times the work" is the finding this
repository is built to demonstrate — and it is a caution about outcome-only benchmarks, not a
claim that Airlock is a better product.

> **Scope.** Every number here is `gpt-oss-20b` on the &tau;&sup2;-bench **airline** domain, 50 tasks
> &times; 4 trials. **The user simulator is also `gpt-oss-20b`**, which makes the absolute figures
> non-comparable to published results — see the
> [disclaimer](#%EF%B8%8F-disclaimer-the-user-simulator-is-also-gpt-oss-20b). Nothing here is
> claimed to generalise to other models or domains.

---

## Table of contents

- [Tech stack](#tech-stack)
- [The problem](#the-problem)
- [System architecture](#system-architecture)
- [Components and their roles](#components-and-their-roles)
- [Benchmark and experimental methodology](#benchmark-and-experimental-methodology)
  - [⚠️ Disclaimer: the user simulator is also `gpt-oss-20b`](#%EF%B8%8F-disclaimer-the-user-simulator-is-also-gpt-oss-20b)
- [Empirical results and statistical analysis](#empirical-results-and-statistical-analysis)
- [Decision verdict: why build 017 shipped](#decision-verdict-why-build-017-shipped)
- [Observability and failure taxonomy](#observability-and-failure-taxonomy)
- [System bottlenecks and ceiling analysis](#system-bottlenecks-and-ceiling-analysis)
- [Cost, latency and overhead](#cost-latency-and-overhead)
- [Getting started and reproduction](#getting-started-and-reproduction)
- [Repository layout](#repository-layout)
- [Limitations and what I would do next](#limitations-and-what-i-would-do-next)

---

## Tech stack

| layer | choice | why |
|---|---|---|
| **Sub-agent framework** | **[pydantic-ai](https://ai.pydantic.dev/)** | Every sub-agent (Planner, Actor, Execution Gate, Reply Gate) is a `pydantic_ai.Agent` with a **typed `output_type`**. Structured output is not a convenience here — it is the enforcement mechanism. A 20B model asked for prose will give you prose; asked to fill a Pydantic model, it either validates or is retried (`retries={"output": 3}`). Verdicts, plans and refusals are all schema-bound. |
| **Orchestration** | **[LangGraph](https://langchain-ai.github.io/langgraph/)** | The kernel is an explicit state machine (`plan → think → gate → act`), not a while-loop. Conditional edges make the control flow — and the re-plan budget — auditable and testable in isolation. |
| **Model routing** | pydantic-ai `OpenAIModel` over **NVIDIA Build** (primary) / **OpenRouter** (fallback) | Both speak the OpenAI protocol, so reasoning effort and timeouts travel identically. Each sub-agent can pick its own model, so a cheap fast one and a stronger slower one can coexist in one run. |
| **User simulator** | &tau;&sup2;'s own, via **LiteLLM** | Not our code and deliberately not configured by us — see the disclaimer below. |
| **Benchmark** | **&tau;&sup2;-bench**, pinned at rev `3571661` | Consumed as a library, never forked. |
| **Observability** | **Langfuse** + a purpose-built JSONL journal | Two tiers for two questions — see [Observability](#observability-and-failure-taxonomy). |
| **Packaging / tooling** | **uv** (0.9.x), **pytest** (481 tests), **ruff** (line length 100, `E,F,I,UP,B`) | Boring and standard-library-first by policy. |
| **Data modelling** | **Pydantic v2** | `Plan`, `Change`, `Verdict`, `Written`, `Consent` are all validated models; the blackboard is typed end to end. |

The one non-obvious choice is **pydantic-ai for the gate**. An LLM critic that answers "no,
because…" in free text is unusable as a control-flow signal. Binding it to a `Verdict` model
(`allowed`, `reason`, `remediation`) means a refusal is a *value* the kernel can route on, and a
malformed answer is a retry rather than a silently approved write.

---

## The problem

&tau;&sup2;-bench scores a customer-service agent on **outcomes, not effort**. Reward is the product
of two binary components:

```
reward  =  DB  ×  COMMUNICATE
```

`DB` compares the final database against a gold replay. It passes **only if every gold write was
made and no other write was**. One surplus write scores identically to doing nothing at all.
Read-only exploration is free; wrong *writes* are fatal.

This creates a perverse incentive that small models find immediately: **abstention is cheap**.
An agent that transfers the customer to a human touches nothing, so `DB` passes on every task
whose gold answer is "make no changes". On the airline domain that is roughly half the task set.

The stock single-agent baseline exploits exactly this — it transfers on **49.2%** of simulations
and completes **11.9%** of the writes customers asked for, and still scores 0.467 average reward
(pass^1 0.460). A benchmark number in that range can mean "competent" or it can mean "politely
useless", and the reward column cannot tell you which.

**Airlock's design question:** if the failure mode is *unauthorised or wrong writes*, can you
recover task completion by putting a second agent and a deterministic verifier tier between the
model and the database — rather than by making the model bigger?

---

## System architecture

The entire multi-agent system runs **inside a single `generate_next_message` call**. &tau;&sup2;'s
orchestrator has no concept of sub-agents, so the blackboard, the active sub-agent and the
pending tool-call table are all carried in the agent `State` object across orchestrator turns.

```
                          customer turn
                                |
                                v
   +==================================================================+
   |                            KERNEL                                |
   |                  (LangGraph state machine)                       |
   +==================================================================+
                                |
     +--------------+           v
     |  WORKFLOWS   |     +-----------+
     | policy rules |---->|  PLANNER  |  writes the ledger of intended changes
     | quote-checked|     +-----------+  (every rule quotes the live policy)
     +--------------+           |
            |                   v
            |  policy     +-----------+
            +------------>|   ACTOR   |  proposes tool calls  XOR  drafts a reply
                          +-----------+
                                |
              tool calls +------+------+ reply
                         |             |
                         v             v
              +------------------+   +------------------+
              |  EXECUTION GATE  |   |    REPLY GATE    |
              |  guards the DB   |   | guards the user  |
              |  fails CLOSED    |   |  fails OPEN      |
              +------------------+   +------------------+
                    |        |             |        |
      +-------------+        |             |        +--> allow --> customer
      |                      |             |
      v                      v             v
+---------------+        refuse      work still owed?
| VERIFIER PANEL|          |          `-> back to ACTOR
| 9 arithmetic  |          |
| policy checks |          v
| (no LLM call) |     back to ACTOR
+---------------+     (re-plan budget: 3)
      |
      +--> approve --+
                     |
                     v
            +-----------------+
            |   ENVIRONMENT   |  yield point, not a function call:
            |   (τ² tools)    |  control returns to the orchestrator,
            +-----------------+  results arrive as a ToolMessage
```

**Graph edges.** `plan -> think -> gate -> act -> plan`, or `think -> speak -> END`.
`escalate` bypasses `speak` when the re-plan budget (3 per turn) is spent.

Two properties are load-bearing:

1. **Nothing reaches the database that a second agent has not authorised.** The Execution Gate
   is a separate LLM call with its own policy context and its own refusal vocabulary. It fails
   **closed** — an unanswered check blocks the write.
2. **Nothing reaches the customer while the plan still owes work.** The Reply Gate fails
   **open** — an unanswered check lets the message through, because a refusal here cannot
   produce the missing write and only costs the customer a round-trip.

The asymmetry is deliberate: an irreversible bad write and an unanswered customer are not
symmetric harms.

---

## Components and their roles

### The agents (LLM-backed)

| agent | file | responsibility | fails |
|---|---|---|---|
| **Planner** | `src/agents/planner.py` | Reads the customer's request and the policy workflows; writes a **ledger of intended changes** (tool + record + description) and the lookups needed first. | — |
| **Actor** | `src/agents/assistant.py` | Given the plan and the policy excerpt, emits *either* tool calls *or* a reply — never both, never empty. | — |
| **Execution Gate** | `src/agents/gate.py` | Authorises every **write** before it reaches the environment. Sees the policy, the transcript, the customer's request, and the provenance of every identifier. | **closed** |
| **Reply Gate** | `src/agents/speaker.py` | Asked *only* when the plan has outstanding changes and the Actor is trying to speak. Decides whether stopping is legitimate (asking consent, missing a customer-only fact) or an abandonment. | **open** |

A fifth helper, `src/agents/requested.py`, extracts what the customer explicitly asked for so the
gate can judge **scope** — an action can be policy-legal and still not be what was requested.

### The deterministic tier

Nine hand-written verifiers in `src/adapters/tau2/` run **before** the Execution Gate's LLM call
and can veto a write on arithmetic alone. They encode policy invariants that need no judgement:

```
read_first              cancellable            not_yet_flown
flights_changeable      payment_for_change     payment_composition
passenger_count_fixed   baggage_only_grows     compensation
```

They are a **scalpel, not a net**: measured across 150 simulations they fired on 2.6% of write
proposals. They are cheap, they never hallucinate, and each is regression-tested against the gold
answer key (`tests/test_verifiers_against_gold.py`). Their honest limitation is that they check
**legality, not correctness** — `payment_for_change` verifies a payment method is *well-formed*,
not that it belongs to *this customer's profile*.

Two further deterministic functions shape the plan before the model ever sees it:

- **`performable(changes, gated)`** — drops planned changes whose tool the domain cannot write.
- **`misfiled(changes, gated, known)`** — re-files those as *lookups* instead of losing them.

These two functions alone cut surplus writes by **37%** (0.384 → 0.242 per simulation) at zero
prompt cost — established by an ablation that shipped the code *without* the prompt change it
originally travelled with, and reproduced the effect exactly.

### The memory / state layer

`src/core/state.py` — the blackboard that survives across orchestrator turns, because &tau;&sup2;
gives sub-agents nowhere else to live:

| field | what it holds | discipline |
|---|---|---|
| `changes` | the ledger of intended writes | **widens only** — a re-plan may add, never silently drop |
| `observed` | every record the system has actually read | the provenance source for gate decisions |
| `written` | writes the gate approved this turn | discharges ledger entries by *tool **and** record* |
| `ruled_out` | changes a verifier proved impossible | only arithmetic may write here |
| `consent` | confirmations the customer has given | prevents re-asking for agreement already granted |
| `calls` | pending tool-call table | `ToolCall.id` ↔ `ToolMessage.id` routes results back to the requesting sub-agent |

`outstanding()` reconciles `changes` against `written` + `ruled_out`, and is what lets the Reply
Gate be **free on most turns** — when nothing is owed, no model call happens at all.

### The policy layer

`src/workflows/` turns prose policy into structured, **quote-verified** rules. Each `Workflow`
carries `facts`, `blocks` (NEVER), `permits` (ALLOWED WHEN) and `rules` (ALSO); every `Rule`
pairs a statement with a **verbatim quote from the live policy document**. `unquoted()` checks
those quotes at load time and **drops any workflow whose quote no longer matches** — so the
system cannot silently drift from the policy it claims to follow. This caught a real bug: a
mistyped quote silently removed an entire booking workflow, shrinking the policy block by 2,313
characters before the check existed.

---

## Benchmark and experimental methodology

&tau;&sup2;-bench is consumed as a **pinned library dependency** (rev `3571661`) and never forked. It
supplies the domain, policy, tools, user simulator and scoring; everything about how the agent is
*built* lives in this repository. This matters for credibility: we cannot accidentally tune the
benchmark in our own favour.

### Experimental design

| control | choice | why |
|---|---|---|
| **Model** | `gpt-oss-20b`, both agent and user simulator | A weak open-weight model is the point. Scaffolding a frontier model proves nothing. |
| **Reasoning effort** | `low` on **both arms** | Verified a non-factor: default vs low differ by **+0.027 ± 0.032** (0.9 SE) over 148 paired simulations. |
| **Trials** | 4 per task, 50 tasks = 200 simulations/arm | 1-trial runs carry a standard error of **±0.071** — larger than any effect this project ever measured. |
| **Seed** | 300, identical across every run | Per-trial seeds derive deterministically, so runs pair task-for-task. |
| **Comparison** | **paired per (task, trial)** | Cancels task difficulty exactly. Unpaired comparison here is close to uninformative. |
| **Control** | &tau;&sup2;'s own `llm_agent`, run **the same day** | Guards against endpoint drift. Verified: the control scored 0.436 five days earlier vs 0.467 same-day. |

### ⚠️ Disclaimer: the user simulator is also `gpt-oss-20b`

**Both sides of every conversation in this repository are the same 20B open-weight model.**
&tau;&sup2;-bench simulates the customer with an LLM, and ours runs
`nvidia_nim/openai/gpt-oss-20b` — not a frontier model, which is what most published &tau;&sup2;
results use. This is a **non-standard evaluation configuration** and it has to be stated
plainly, because it cuts two ways.

**What it does affect: comparability of the absolute numbers.** A weaker customer asks less
precise questions, supplies information differently, and judges "am I satisfied?" differently
from a frontier user simulator. **The 0.530 pass^1 here is not comparable to any published
airline figure**, including ones using the same agent model, unless that run also used a 20B
user simulator. Treat the absolute number as internal-only.

**What it does not affect: the comparison this repo is built on.** Both arms — Airlock and the
`llm_agent` control — run the *identical* user simulator, identical seed, identical tasks, on
the same day. Every paired result is therefore unaffected by the choice, which is precisely why
the paired design was used.

**Did it degrade our results? Partly, and we measured which part.**

The user simulator holds the stopping decision almost absolutely:

| termination reason | Airlock 017 | control |
|---|---|---|
| `user_stop` (the simulated customer ends the conversation) | **99.5%** | 97.5% |
| `max_steps` | 0.5% | 2.0% |
| `too_many_errors` | 0% | 0.5% |

So in **199 of 200 simulations the customer decided when the conversation was over** — not the
step budget, not an error. That is a large amount of authority to hand a 20B model, and 56 gold
writes across earlier runs were lost to conversations that simply ended.

**But the obvious hypothesis — that it cuts conversations short before the agent can write — does
not survive measurement.** On tasks that demand a write:

| outcome | n | median customer turns |
|---|---|---|
| solved | 23 | **5.0** |
| failed, no write ever made | 26 | **5.0** |
| failed, wrote but wrote wrongly | 55 | 6.0 |

Failures got **the same number of turns as successes**. The agent was not cut off; it had the
conversation and still did not write. So the user simulator is a genuine confound for
*comparability*, and there is **no evidence** it is the cause of the write failures documented
below. The ceiling is ours, not the simulator's.

### Statistical discipline

Three rules, each adopted after it was violated, each of which changed a conclusion:

**1. Never read a partial run.** &tau;&sup2; runs tasks roughly in order and the airline tail is much
easier than the head, so partial slices are biased **even when paired** — the *pairs available*
are not a random sample. One build was called a regression at 32 simulations, a recovery at 51,
and was a regression at 99. Two of three readings were wrong, in opposite directions.

**2. A selected maximum is not an estimate.** Build 017 was chosen as the best of ~8 noisy runs
and recorded at 0.560. Re-measured without that selection it scored **0.533**. Build 019 was
recorded at 0.620 from a single 50-task run; re-measured at full size it scored **0.500**. Both
are the winner's curse, and both regressions were predicted before the runs finished.

**3. Report the interval, not the point.** The smallest reward difference this design can resolve
is roughly **±0.09**. Every architectural change in this project's history sits inside its own
noise, and the results below say so rather than hiding it.

### Verification layers

| layer | what it catches |
|---|---|
| **481 unit tests** (`uv run pytest`) | component contracts, state discipline, policy parsing |
| **gold-answer regression tests** | verifiers scored against &tau;&sup2;'s own answer key — a tripwire that must stay green |
| **offline gate corpus** (`scripts/gate_bench.py`) | replays every gate decision from saved runs **including refusals**, which never appear in a trajectory |
| **A/B arms** | `STEWARD_GATE=off` disables the critic in place, isolating its contribution with no code change |
| **structured journal** | one JSON line per graph node — 13,465 records for the shipped run |
| **Langfuse tracing** | per-conversation spans for human debugging (auto-enabled when keys are present) |

---

## Empirical results and statistical analysis

All runs: airline domain, 50 tasks, seed 300, `gpt-oss-20b` both sides, reasoning effort `low`.

### Headline table

| run | build | n | reward | DB | COMM | gold write | gold read | surplus/sim | transfer | solved |
|---|---|---|---|---|---|---|---|---|---|---|
| **026** | **Airlock 017 (shipped)** | 199 | **0.533** | 0.548 | 0.910 | 33.2% | **86.5%** | 0.613 | 42.2% | 106/199 |
| 025 | Airlock 019 | 200 | 0.500 | 0.530 | 0.915 | **45.9%** | 82.8% | 0.600 | **37.5%** | 100/200 |
| 024 | &tau;&sup2; `llm_agent` (control) | 197 | 0.467 | 0.503 | 0.863 | 11.9% | 56.1% | 0.655 | 49.2% | 92/197 |

### pass^k, computed &tau;&sup2;'s way

`comb(successes, k) / comb(trials, k)` averaged over tasks — **not** `scripts/score.py`'s
"first k trials all passed", which is a different and noisier quantity.

| system | pass^1 | pass^2 | pass^3 | pass^4 |
|---|---|---|---|---|
| **Airlock 017** | **0.530** | **0.433** | **0.395** | **0.380** |
| Airlock 019 | 0.500 | 0.383 | 0.320 | 0.280 |
| &tau;&sup2; `llm_agent` | 0.460 | 0.357 | 0.310 | 0.280 |

The scaffold's advantage **widens with k** (+0.070 at pass^1, +0.100 at pass^4). It is buying
*consistency*, not single-shot luck — the property that matters for anything deployed.

### Paired analysis: scaffold vs baseline

Per task, over 50 tasks. This is the comparison the project rests on.

| metric | Airlock 017 | baseline | difference | verdict |
|---|---|---|---|---|
| **gold-write recall** | 0.355 | 0.122 | **+0.233 ± 0.049** | **4.8 SE — established** |
| **gold-read recall** | 0.912 | 0.700 | **+0.212 ± 0.049** | **4.3 SE — established** |
| transfer to human | 0.425 | 0.500 | &minus;0.075 ± 0.042 | 1.8 SE — suggestive |
| COMMUNICATE | 0.910 | 0.865 | +0.045 ± 0.026 | 1.7 SE — suggestive |
| reward | 0.535 | 0.465 | +0.070 ± 0.046 | 1.5 SE — **not established** |
| DB | 0.550 | 0.500 | +0.050 ± 0.042 | 1.2 SE — not established |
| surplus writes | 0.610 | 0.650 | &minus;0.040 ± 0.120 | 0.3 SE — no difference |

**Read this table honestly.** The behavioural claims clear 4 SE. The reward claim does not clear
2. A system that completes three times the work and scores 0.07 higher is telling you something
about the *metric*, not only about the system.

### Paired analysis: 017 vs 019 — a null result worth publishing

| metric | 017 | 019 | difference | |
|---|---|---|---|---|
| reward | 0.535 | 0.500 | +0.035 ± 0.027 | 1.3 SE |
| DB | 0.550 | 0.530 | +0.020 ± 0.028 | 0.7 SE |
| COMMUNICATE | 0.910 | 0.915 | &minus;0.005 ± 0.017 | 0.3 SE |
| gold-write recall | 0.355 | 0.422 | &minus;0.067 ± 0.045 | 1.5 SE |
| gold-read recall | 0.912 | 0.875 | +0.036 ± 0.021 | 1.7 SE |
| surplus writes | 0.610 | 0.600 | +0.010 ± 0.087 | 0.1 SE |
| transfer | 0.425 | 0.375 | +0.050 ± 0.045 | 1.1 SE |

**Not one metric reaches 2 SE.** After 400 simulations the benchmark cannot distinguish two
builds that differ meaningfully in prompt and plan handling. Any claim that one is better than
the other would be reading point estimates as results.

### Replication

| build | first recorded | re-measured | paired |
|---|---|---|---|
| 017 | 0.560 (150 sims) | **0.533** (199 sims) | &minus;0.047 ± 0.037 |
| 019 | 0.620 (50 sims, 1 trial) | **0.500** (200 sims) | &minus;0.106 ± 0.063 |

Nine of 47 tasks (**19%**) flipped outcome between two runs of **byte-identical code** on the same
tasks and seed. `temperature: 0.0` is discarded by the endpoint whenever reasoning is enabled, so
nothing here is deterministic — two runs are two independent samples.

---

## Decision verdict: why build 017 shipped

**017 and 019 are statistically indistinguishable, so the decision was made on tiebreakers — and
it is reported as a tiebreak, not a verdict.**

| criterion | winner | margin |
|---|---|---|
| **Consistency** — tasks solved on *all four* trials | **017** (19/50) | vs 019 (14/50), control (14/50) |
| **pass^4** | **017** (0.380) | vs 019 (0.280) |
| Reward point estimate | **017** (0.533) | +0.035, not significant |
| Reward margin over control | **017** (+0.070) | vs 019 (+0.035) |
| gold-read recall | **017** (86.5%) | reading precedes writing |
| gold-write recall | *019* (45.9%) | 017 at 33.2%, not significant |
| needless transfers | *019* (37.5%) | only build with a significant reduction vs control (2.2 SE) |

**The decisive argument is consistency.** For a system whose purpose is to be trusted with a
production database, a task solved *every* time is worth more than a task solved *sometimes*.
017 solves 19 tasks on all four trials against 019's 14, and pass^4 separates them by 0.100 —
the largest gap between them on any measure.

**The honest counter-argument**, on the record: 019 attempts more of the real work (higher
gold-write recall) and is the only build with a statistically significant reduction in needless
transfers. If the thesis is that reward mismeasures work done, selecting on reward is in tension
with that thesis. The resolution is that gold-write recall *also* fails to separate them
(1.5 SE) — no metric picks 019 on evidence, so consistency breaks the tie.

---

## Observability and failure taxonomy

### Two tiers, for two different questions

**Langfuse** (`src/tracing/`) answers *"what happened in this one conversation"* and is the right
tool when a human is reading a trace. It auto-enables when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are present, and instruments every graph node.

**It is the wrong tool for the question this project kept asking**, which is population-level:
*across 200 simulations, how often did the gate refuse a write while a change was still owed?*
That needs decisions on disk, greppable, with no server running.

Worse, there is a **structural blind spot**: a &tau;&sup2; trajectory records only what *left* the
system — customer turns, assistant turns, and tool calls actually emitted. **A proposal the gate
refused never becomes a tool call, so it leaves no trace at all.** The entire internal half of a
multi-agent system is invisible to post-hoc trajectory analysis.

So `src/tracing/journal.py` writes **one JSON line per graph node** as the node closes — inputs,
decision, verdict. The shipped run produced **13,465 records**:

```
think 3698  ·  plan 3032  ·  gate 2062  ·  results 1729  ·  speak 1614  ·  message 1303  ·  escalate 27
```

That is what made the offline gate corpus possible: replaying all 381 write proposals from a
saved run **with their refusals**, labelled gold/surplus, so "did the gate wrongly refuse a
correct write?" became answerable without burning a benchmark run.

### Failure taxonomy — shipped build, 93 failed simulations of 199

**Which component failed:**

| | count | share |
|---|---|---|
| **DB failed, COMMUNICATE passed** | 75 | **81%** |
| both failed | 15 | 16% |
| COMMUNICATE failed only | 3 | 3% |

**What the assistant *says* is almost never the problem.** 84% of failures pass COMMUNICATE.
Every meaningful failure is a database failure.

**How the database was wrong:**

| shape | count | share of DB failures |
|---|---|---|
| **wrong write made instead** (bad arguments or wrong record) | 45 | **50%** |
| **write required, made none** | 26 | 29% |
| zero-write task, wrote anyway | 12 | 13% |
| some gold writes missing | 4 | 4% |
| all gold writes made, plus one surplus | 3 | 3% |

Half of all failures are the system **doing the right kind of thing to the wrong data**. The
wrong-argument fields cluster tightly — `payment_methods` (19), `flights` (17), `passengers` (8),
`payment_id` (7) — and critically, **every one of those values was already present in a record
the system had read**. This is a state-tracking and argument-construction defect, not a reasoning
defect, which is what makes it the highest-value remaining target.

**Abstention among failures:** 26 of 93 failures ended in a transfer to a human, and **25 of
those were on tasks demanding a write**. The system knew it was stuck and gave up rather than
writing wrongly — the safe failure, and still a failure.

### Near-miss analysis

The most instructive failures got everything right but one thing:

- **3 simulations** made *every* gold write correctly and were failed by **a single surplus
  write** — a write the policy permitted, on a record the customer never named.
- **12 simulations** were zero-write tasks where the system wrote anyway.

The deterministic panel was replayed over these and was **correctly silent on all of them**:
every surplus write was policy-*legal*. That finding redirected the design — the remaining errors
are **scope** errors, not legality errors, and no policy verifier can catch them. It is why the
Execution Gate was given the customer's original request and the provenance of every identifier:
an action can be legal and still not be what was asked for.

---

## System bottlenecks and ceiling analysis

### The ceiling is writes, and it is not subtle

| task type | simulations | solved | mean reward |
|---|---|---|---|
| **zero-write tasks** (gold makes no writes) | 95 | 83 | **0.874** |
| **write tasks** (gold makes ≥1 write) | 104 | 23 | **0.221** |

**The system is near-solved on tasks requiring no database changes and largely broken on tasks
requiring them.** All headroom lives in that second row. This single split explains the
benchmark's abstention incentive better than any other number here: an agent that never writes
scores 0.874 on half the task set for free.

### Structural limits of a 20B model, stated plainly

**1. Argument construction, not reasoning, is the binding constraint.** 50% of failures are
wrong-argument writes where the correct value was already in context. The model plans the right
action and then fills the call in wrong. Prompt engineering did not move this across four
different planner prompts (precision held at 24.2–24.7%, recall 61–69%).

**2. Plan quality has a hard floor.** Four substantially different planner prompts — adding
scope, record identity, tool naming, and carry-forward of prior goals — produced **no measurable
change in gold-write recall**. Two produced measurable *regressions*. The planner is not
prompt-limited; it is capability-limited.

**3. Nondeterminism swamps the effect sizes we care about.** 19% of tasks flip outcome between
identical runs. A 50-task run carries ±0.071 of pure noise — larger than every architectural
effect measured in this project's history. **No change to this architecture has ever been proven
at the 5% level on this benchmark**, which is a fact about the measurement design as much as
about the changes.

**4. Deterministic verification has a small reachable surface.** The panel fires on 2.6% of write
proposals and cannot grow much: the remaining errors are scope and argument errors that require
knowing what the customer meant, and 115 of 381 gated proposals are `transfer_to_human_agents`,
for which no verifier can exist.

**5. Some losses have no handle at all.** 56 gold writes were lost to conversations that simply
ended — the user simulator moved on, the turn budget ran out, or the assistant said goodbye.
There is no deterministic intervention for that.

### Known defects

- **&tau;&sup2;'s `--timeout` does not stop a wedged simulation.** Observed running to 11,133 s under
  `--timeout 1800`. Roughly 1.5% of simulations hang indefinitely; they are counted as failures.
- **`scripts/score.py`'s `pass^k`** uses "first k trials all passed" rather than &tau;&sup2;'s
  combinatorial estimator. Both are unbiased; they disagree below k = n. Every pass^k in this
  README uses &tau;&sup2;'s formula.
- **Cost accounting sums only emitted `AssistantMessage.cost`**, so internal sub-agent calls are
  invisible to &tau;&sup2;'s own cost column unless folded in deliberately.

---

## Cost, latency and overhead

The scaffold is **not free**, and reporting that is part of the point.

| | Airlock 017 | baseline | ratio |
|---|---|---|---|
| median simulation | **433 s** | 68 s | **6.4×** |
| mean simulation | 726 s | 332 s | 2.2× |
| p90 simulation | 1,931 s | 608 s | 3.2× |
| messages per simulation (median) | 24 | 18 | 1.3× |
| **sub-agent LLM calls per simulation** | **52.4** | ~1 per turn | — |

Roughly **52 internal model calls per simulation** buy +0.070 reward and +0.233 gold-write
recall. Whether that trade is worth making is a deployment question with a real answer, and it
depends entirely on what a wrong write costs relative to latency. For a system writing to a
billing database, it is probably worth it. For a low-stakes chat assistant, it is plainly not.

**Mitigations already in the design:** the Reply Gate's deterministic half (`outstanding()`)
skips the model call entirely when nothing is owed; the verifier panel runs before the gate's LLM
call and can veto without one; and read-only steps never reach the gate at all.

---

## Getting started and reproduction

### Requirements

- [uv](https://docs.astral.sh/uv/) (0.9.x) and Python **3.12** (&tau;&sup2; requires `>=3.12,<3.14`)
- An API key for an OpenAI-protocol provider serving `gpt-oss-20b`
  (NVIDIA Build primary, OpenRouter fallback)

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/airlock.git
cd airlock

uv sync                                    # creates .venv, installs τ² at its pinned rev
uv run python scripts/bootstrap_data.py    # fetches benchmark data into vendor/ (~145 MB)
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in `TAU2_DATA_DIR` (the checkout `bootstrap_data.py` created) and your provider key.
`.env` is gitignored; never commit real keys.

```bash
uv run tau2 check-data      # should report "Data directory exists"
```

### 3. Verify the build

```bash
uv run pytest                                        # 481 tests
uv run ruff check . && uv run ruff format --check .
```

### 4. Reproduce the shipped result (build 017, 50 × 4)

> On Windows, set `PYTHONUTF8=1` first — the default console codepage cannot encode the emoji
> &tau;&sup2; prints, and `.env` is loaded too late to fix stdout encoding.

```bash
export PYTHONUTF8=1
export STEWARD_GATE=on
export STEWARD_JOURNAL=runs/airlock017.jsonl        # optional: per-node decision journal

uv run python scripts/run_bench.py run \
  --domain airline \
  --agent steward   --agent-llm openai/gpt-oss-20b \
  --agent-llm-args '{"temperature": 0.0}' \
  --user-llm nvidia_nim/openai/gpt-oss-20b \
  --user-llm-args '{"temperature": 0.0, "timeout": 300}' \
  --num-trials 4 --max-concurrency 8 --seed 300 --auto-resume \
  --save-to airlock017.json
```

### 5. Reproduce the control arm

Identical in every argument except the agent — which is what makes the comparison paired:

```bash
uv run python scripts/run_bench.py run \
  --domain airline \
  --agent llm_agent --agent-llm nvidia_nim/openai/gpt-oss-20b \
  --agent-llm-args '{"temperature": 0.0, "reasoning_effort": "low"}' \
  --user-llm nvidia_nim/openai/gpt-oss-20b \
  --user-llm-args '{"temperature": 0.0, "timeout": 300}' \
  --num-trials 4 --max-concurrency 8 --seed 300 --auto-resume \
  --save-to control.json
```

### 6. Score and compare

```bash
uv run python scripts/score.py \
  "$TAU2_DATA_DIR/simulations/control.json/results.json" \
  "$TAU2_DATA_DIR/simulations/airlock017.json/results.json"
```

### Useful knobs

| variable | effect |
|---|---|
| `STEWARD_GATE=off` | Approves every proposal without asking the Execution Gate — the A/B arm for whether the critic pays for itself. |
| `STEWARD_JOURNAL=<path>` | Writes one JSON line per graph node. Off by default; a test run writes nothing. |
| `STEWARD_LLM_REASONING_EFFORT` | `low` for the results here. At default effort this model takes 120–360 s per call. |
| `LANGFUSE_*` | Tracing auto-enables when both keys are present. |

> **Note on naming:** the environment prefix and the `--agent steward` registration name are
> retained from the project's original codename so that saved runs and checkpoints stay loadable.

---

## Repository layout

```
src/
  core/          kernel (LangGraph state machine), state/blackboard, policy, refusal vocabulary
  agents/        planner · actor · execution gate · reply gate · request extractor · toolset
  adapters/tau2/ the τ² seam: agent registration, 9 deterministic verifiers, record parsing
  workflows/     policy → structured, quote-verified rules
  llm/           provider config, model routing, reasoning-effort plumbing
  tracing/       Langfuse spans + the JSONL decision journal
scripts/
  run_bench.py       τ²'s CLI with our agent registered first
  score.py           reward, pass^k, paired deltas with intervals
  gate_bench.py      offline replay of gate decisions, including refusals
  bootstrap_data.py  fetches benchmark data into vendor/
  probe_models.py    verifies a provider/model before adopting it
tests/             481 tests, incl. verifier regression against τ²'s gold answer key
results/log.md     the full run record — every run, every verdict, including the failures
```

**`results/log.md` is the primary research artefact.** It records every run this project ever
made, including the ones that regressed, the conclusions that were retracted, and the method
errors that produced them.

---

## Limitations and what I would do next

**Limitations, without hedging:**

- The reward improvement over the baseline is **not statistically established** (1.5 SE). Only
  the behavioural differences clear conventional significance.
- Results are **`gpt-oss-20b` on the airline domain only**. No claim is made about other models
  or domains; a stronger model may need none of this scaffolding.
- **The user simulator is `gpt-oss-20b` too**, which is a non-standard configuration and makes
  the absolute numbers non-comparable to published &tau;&sup2; results. It does not affect the paired
  comparisons, and it is measurably *not* the cause of the write failures — failed simulations
  got the same number of customer turns as successful ones.
- Published leaderboard figures for this benchmark are dominated by frontier models under
  different harnesses. **They are not comparable to these numbers.** The only apples-to-apples
  comparison in this repository is against the same model, unscaffolded, on the same grid, run
  the same day.
- A scaffolded agent is a **"custom" leaderboard submission**, not a standard one.
- 6.4× median latency is a real deployment cost, not a rounding error.

**The next three things I would build**, in order of expected value:

1. **Profile-aware payment verification.** `payment_for_change` checks that a payment method is
   well-formed but never that it belongs to *this customer*. 26 field mismatches are checkable
   against `get_user_details`, which the system has usually already called.
2. **Itinerary completeness verification.** `update_reservation_flights` must carry the *whole*
   itinerary; 17 failures passed a partial one. Checkable against the reservation already read.
3. **A pre-write argument diff.** Half of all failures are wrong arguments to the right tool
   where the correct value was already in context. A deterministic step — "here are the values
   you are about to send; here are the values in the record you read" — needs no model call and
   targets the largest failure class directly.

---

## Acknowledgements

Built on [&tau;&sup2;-bench](https://github.com/sierra-research/tau2-bench) by Sierra Research,
consumed as a pinned upstream dependency and never forked. The benchmark supplies the domains,
policies, tools, user simulator and scoring; the failures are ours.
