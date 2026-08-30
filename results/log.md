# Run log

Every benchmark run on one line. The reasoning lives in
[`reports.md`](reports.md); this file exists so the trend can be read at a
glance.

**Held fixed across all runs:** domain `airline` (50 tasks), agent model
`nvidia:openai/gpt-oss-20b`, user simulator `nvidia_nim/openai/gpt-oss-20b`,
max 200 steps, concurrency 8.

**Not fixed, and it matters:** the trial count fell from 4 to 1 as the loop got
tighter, and `temperature` was dropped to 0.0 at `5fbde41`. A 1-trial run buys
speed with the entire error bar — see *How to read this*.

## Scores

`Change` names the commit that was HEAD when the run started, not the commit the
file is named after. Avg reward is `scripts/score.py`, which counts an unscored
simulation as 0; `Scored` is how often that happened, and a row that lost many is
not comparable to one that lost none.

| Run | Name on disk | About | Trials | Avg reward | Pass^1 | Pass^2 | Pass^3 | Pass^4 | Scored |
|---|---|---|---|---|---|---|---|---|---|
| 001 | — | baseline: bare ReAct, no gates | 4 | 0.365 ±0.023 | 0.360 | 0.260 | 0.200 | **0.160 ±0.029** | 196/200 |
| 002 | — | + PRE-GATE, GATE, COMMIT | 4 | 0.420 ±0.021 | 0.340 | 0.280 | 0.260 | **0.220 ±0.031** | 197/200 |
| diag | — | rename to Steward, temperature 0.0, Langfuse | 1 | 0.380 | 0.380 | — | — | — | 50/50 |
| 003 | — | + enforced tool schemas, first real system prompt | 1 | 0.420 | 0.420 | — | — | — | 49/50 |
| 004b | — | **control: tau2's own `llm_agent`, no scaffold** | 1 | **0.500** | 0.500 | — | — | — | 47/50 |
| 005 | — | + planner agent, policy sections routed per turn | 3 | 0.333 ±0.031 | 0.360 | 0.220 | 0.120 | — | 130/150 |
| 006 | `speaker_49` | + airport codes the policy assumes (see note) | 2 | 0.270 ±0.041 | 0.420 | 0.100 | — | — | 63/100 |
| 007 | `fixes_50` | + SPEAKER, the check on the reply | 1 | 0.280 | 0.280 | — | — | — | 38/50 |
| 008 | `parts_50` | + workflows, planner in the loop, deterministic checks | 1 | 0.420 | 0.420 | — | — | — | 47/50 |
| 009 | `all_parts_50` | + Parts 1–6, harmony repair | 1 | **0.460** | 0.460 | — | — | — | **50/50** |
| 010 | `stable_ef6cffd_50x3` | *arm C.* `ef6cffd`: monolithic critic on, no sieve | 3 | **0.493 ±0.030** | 0.500 | 0.360 | 0.300 | — | 150/150 |
| 011 | `ab_control_50x3` | *arm B.* **control: tau2's own `llm_agent`, no scaffold** | 3 | 0.433 ±0.026 | 0.460 | 0.320 | 0.300 | — | 149/150 |
| 012 | `veto_consent_50x3` | + verifiers as veto over the critic, consent ledger | 3 | 0.507 ±0.021 | 0.480 | 0.460 | 0.400 | — | 150/150 |
| 013 | `ab_steward_50x3` | *arm A.* `73ad27f`: deterministic panel on, monolith off | 3 | 0.487 ±0.023 | 0.500 | 0.380 | 0.380 | — | 150/150 |
| 014 | `gateoff_ef6cffd_50x3` | *cell 1.* `ef6cffd` + `STEWARD_GATE=off`: nothing gating | 3 | 0.420 ±0.031 | 0.420 | 0.273 | 0.220 | — | 147/150 |
| 015 | `sievegate_73ad27f_50x3` | *cell 2.* `73ad27f` + `STEWARD_GATE=on`: **sieve and critic together** | 3 | **0.500 ±0.027** | 0.500 | 0.393 | 0.340 | — | 150/150 |
| 016 | `final50x3` | `bdf1f3e`: sieve and critic, consent ledger, records seam fixed | 3 | 0.540 ±0.023 | 0.540 | 0.460 | 0.420 | — | 150/150 |
| **017** | `steps1234_50x3` | `ea1e6ac`: gate given the request, the consent, the provenance. **Best run in the project.** | 3 | **0.560 ±0.023** | 0.560 | 0.460 | 0.400 | — | 150/150 |
| 018 | `planner17` | planner: REPLACE + scope + record, 17 targeted tasks. **Aborted at 20/51 — regression.** | 3 | 0.150 (partial) | — | — | — | — | 20/51 |
| 019 | `run019` | planner: REPLACE + quote rule + `performable`. Surplus writes 26 → 7; plan recall 72.9% → 63.3% | 1 | 0.620 | 0.620 | — | — | — | 50/50 |
| 020 | `run020` | planner: every write workflow names its tool; commit-after-lookup; one verdict per record. **Regression, −0.050 paired.** | 2 | 0.530 ±0.050 | 0.530 | 0.360 | — | — | 100/100 |

## How runs are named

`Run` is the number to use in conversation. `Name on disk` is the directory under
`vendor/tau2-data/data/simulations/`, and the journal of the same name under
`scratchpad/keep/`. Runs 001–005 predate the convention and keep the labels the
analysis below already refers to; everything from 006 is numbered in order.

**Run 017 is the current baseline** — the highest score the project has recorded,
and the one to compare against unless a row says otherwise. Its journal
(`keep/steps1234.jsonl`, 150 simulations, every planner/gate/actor node) is the
corpus for offline work.

The last two rows' Pass^k are computed with tau2's own formula,
`comb(successes, k) / comb(trials, k)` averaged over tasks
(`tau2/metrics/agent_metrics.py:113`). `scripts/score.py` uses "the first k
trials all passed", which is a different and wrong quantity -- on
`steps1234_50x3` it prints Pass^1 0.580 against the true 0.560. Every Pass^k
above these two rows is `scripts/score.py`'s number and is not comparable to
them.

`speaker_49` started at 01:57 on 08-23 and the SPEAKER commit landed at 02:54, so
either the name is aspirational or the work was in the tree uncommitted. Which one
is not recoverable from the artefacts, so the row is labelled by the last commit
certainly in it. Two of its four trial-slots never ran; treat 0.270 as a floor.

Deltas, paired on the shared tasks:

| Runs | Avg reward | Tasks better / worse / unchanged | Verdict |
|---|---|---|---|
| 002 − 001 | +0.055 ± 0.033 (1.7 SE, p ≈ 0.10) | 14 / 5 / 31 | unproven at 5% |
| 003 − diag | +0.040 ± 0.057 (0.7 SE, p ≈ 0.48) | 5 / 3 / 42 | unproven at 5% |
| parts_50 − 003 | +0.000 ± 0.064 (0.0 SE, p ≈ 1.00) | 5 / 5 / 40 | unproven at 5% |
| all_parts_50 − parts_50 | +0.040 ± 0.081 (0.5 SE, p ≈ 0.62) | 9 / 7 / 34 | unproven at 5% |
| arm C − arm B | +0.060 ± 0.039 (1.5 SE, p ≈ 0.12) | 14 / 6 / 30 | unproven at 5% |
| arm A − arm C | −0.007 ± 0.044 (0.2 SE, p ≈ 0.88) | 11 / 14 / 25 | unproven at 5% |
| arm A − arm B | +0.053 ± 0.042 (1.3 SE, p ≈ 0.20) | 11 / 6 / 33 | unproven at 5% |

**No change to the architecture has ever been proven at the 5% level on this
benchmark.** Every row above sits inside its own noise. That is a fact about the
measurement design as much as about the changes — see *How to read this*.

## Behaviour

Recomputed for every row with one script, so the columns are internally
consistent. The 001/002 figures differ slightly from the earlier printing of this
table because `transfer_to_human_agents` is now counted as a write, which moves
four gold actions per trial-set out of the read column.

| Run | DB | COMMUNICATE | Writes emitted | Off-record writes | Gold write recall | Gold read recall |
|---|---|---|---|---|---|---|
| 001 | 77/196 (39%) | 174/196 (89%) | 240 | 228 (95%) | 11/199 (5.5%) | 156/365 (43%) |
| 002 | 92/198 (46%) | 173/198 (87%) | **125** | 117 (94%) | 8/200 (4.0%) | 136/366 (37%) |
| diag | 20/50 (40%) | 44/50 (88%) | 43 | 43 (100%) | 0/50 (0%) | 36/92 (39%) |
| 003 | 21/49 (43%) | 44/49 (90%) | 34 | 29 (85%) | 5/49 (10%) | 52/92 (57%) |
| 004b (control) | 26/47 (55%) | 41/47 (87%) | 38 | 35 (92%) | 3/49 (6.1%) | 59/90 (66%) |
| 005 | 54/130 (42%) | 116/130 (89%) | 139 | 126 (91%) | 13/137 (9.5%) | 175/210 (83%) |
| speaker_49 | 28/63 (44%) | 56/63 (89%) | 46 | 41 (89%) | 5/64 (7.8%) | 85/90 (94%) |
| fixes_50 | 16/38 (42%) | 32/38 (84%) | 33 | 31 (94%) | 2/44 (4.5%) | 31/49 (63%) |
| parts_50 | 21/48 (44%) | 44/48 (92%) | 41 | 34 (83%) | 6/50 (12%) | 57/89 (64%) |
| all_parts_50 | **24/50 (48%)** | 46/50 (92%) | 57 | 37 (**65%**) | **20/50 (40%)** | **75/92 (82%)** |
| arm C `stable_ef6cffd_50x3` | **79/150 (53%)** | 134/150 (89%) | 218 | 155 (71%) | **63/150 (42%)** | **215/276 (78%)** |
| arm B `ab_control_50x3` (control) | 70/150 (47%) | 131/150 (87%) | 264 | 251 (95%) | 13/150 (9%) | 167/275 (61%) |
| arm A `ab_steward_50x3` | 75/150 (50%) | **136/150 (91%)** | 219 | 184 (84%) | 35/150 (23%) | 206/276 (75%) |
| cell 1 `gateoff_ef6cffd_50x3` | 63/150 (42%) | 132/150 (88%) | 246 | 209 (85%) | 37/150 (25%) | **231/278 (83%)** |
| cell 2 `sievegate_73ad27f_50x3` | 79/150 (53%) | **136/150 (91%)** | 196 | 159 (81%) | 37/150 (25%) | 218/279 (78%) |

## What the trend actually says

- **Gold read recall is solved; gold write recall was not, until the last run.**
  Reads went 43% → 82%, touching 94% on `speaker_49`. Writes sat between 0% and
  12% for nine consecutive runs and only moved at `all_parts_50`, to 40%.
- **Reward has not followed either of them.** `all_parts_50` reads and writes far
  better than `parts_50` and scores 0.460 against 0.420 — but on the 47 tasks both
  runs actually scored it is **0.426 against 0.447**. The entire headline gain is
  three tasks `parts_50` never scored (one harmony crash, two that never ran), all
  three of which passed this time. Net of that, six parts of work moved reward by
  approximately zero.
- **The reason is off-record writes.** DB is exact-match against a gold replay, so
  a correct write plus one extra scores the same as no write at all.
  `all_parts_50` emits 20 correct gold writes and 37 off-record ones. The rate
  improved sharply (95% → 65%) but the absolute count did not fall. This is the
  binding constraint now, and nothing in the current architecture addresses it.
- **The control still leads.** tau2's own `llm_agent`, with no scaffold at all,
  scored 0.500 against Steward's best of 0.460.
  **Superseded 2026-08-25:** at three trials each the control scores 0.433 and
  arm C scores 0.493. Both single-trial figures above were draws from a
  distribution too wide to rank — see *The three-arm experiment*. Everything built so far has bought
  intermediate accuracy that the scoring function does not pay for.
- **DB is where the losses are.** COMMUNICATE has sat between 84% and 92% since
  the first run and has never been the bottleneck.

## The three-arm experiment

Run 2026-08-25. The first comparison on this project with enough trials to say
anything, and the first with a control run under identical conditions rather than
quoted from an older row.

**Held fixed across all three:** domain `airline`, 50 tasks x 3 trials, `--seed
300`, `--auto-resume`, concurrency 8, tau2 rev `3571661`, `gpt-oss-20b` on both
sides of the conversation at `temperature` 0.0. The arms differ only in the agent.

| arm | commit | configuration |
|---|---|---|
| **C** | `ef6cffd` | monolithic critic **on** (its default at that commit), no sieve |
| **B** | — | tau2's own `llm_agent`, no scaffold |
| **A** | `73ad27f` | Part 4 deterministic sieve **on**, monolithic critic **off** |

`src/agents/gate.py` is byte-identical between `ef6cffd` and `73ad27f`; only
`kernel.py` changed. The critic is the same component in both arms — what moved
was its wiring and its default, which flipped from `on` to `off`.

### Reward says nothing. Behaviour says a great deal.

Paired per task, which cancels task difficulty exactly:

| | arm C − arm B | arm C − arm A |
|---|---|---|
| avg reward | +0.060 ± 0.039 (1.5 SE, p ≈ 0.12) | +0.007 ± 0.044 (0.2 SE, p ≈ 0.88) |
| **gold-write recall** | **+0.311 ± 0.053 (5.9 SE, p < 0.0001)** | **+0.149 ± 0.044 (3.4 SE, p = 0.0007)** |
| gold-read recall | +0.128 ± 0.032 (4.0 SE, p = 0.0001) | — |
| transfer to human | −0.193 ± 0.048 (4.1 SE, p = 0.0001) | −0.267 ± 0.049 (5.5 SE) |
| transfer, write-tasks only | −0.205 ± 0.067 (3.1 SE, p = 0.002) | −0.321 ± 0.068 (4.7 SE) |

Transfer rates: **arm C 34.0%, arm B 53.0%, arm A 60.7%.** `transfer_to_human_agents`
appears in the airline answer key exactly once across all 50 tasks, so a transfer
is almost always an abstention rather than a fix.

**Three systems this benchmark cannot separate on score — 0.493 / 0.487 / 0.433,
every interval overlapping every other — complete 42.9% / 23.8% / 8.8% of the
writes the customer actually asked for.** The control buys most of its reward by
handing the customer to a human, and reward does not notice. That is the finding
this project has to report; the architecture is the thing that made it visible,
not the result.

The smallest reward difference this design can resolve is **0.109**. Nothing in
the reward column reaches it and nothing here claims otherwise.

### Part 4 is a measured regression

> **VOID (2026-08-29).** Every "sieve on" cell below was measured while
> `records._loaded` could not parse a decorated tool result, so the deterministic
> tier was one check firing falsely on 100% of proposals and five checks that
> never executed once. Of 151 write proposals in a later journalled run, 11 were
> approved. The sieve scored positively here because refusing writes buys
> abstention and abstention carries reward on this benchmark -- that is a defect
> being rewarded, not a mechanism working. The numbers are kept as the record;
> none of them describes the system. See "The verifier tier had never once read a
> record". Not yet re-measured.


Arm A abstains more than arm C on every cut, and more than the *unscaffolded
control* — which arm C was comfortably below. Six pre-sieve runs transfer at
42–48%; arm A transfers at 60.7%. The Part 4 sieve bought write precision
(over-writes 15.4% against arm C's 24.4%) by declining to act, and paid for it in
work completed.

**The experiment cannot say which half of Part 4 caused it.** Arm A changed two
things at once — the sieve was added *and* the monolithic critic was switched
off — so "the sieve blocks too much" and "the critic was doing useful work" fit
the data equally well. The 2x2 has an empty cell:

| | critic off | critic on |
|---|---|---|
| **sieve off** | `gate_off_50` 0.420 (1 trial) | **arm C 0.493** |
| **sieve on** | **arm A 0.487** | *never run* |

At `73ad27f` the deterministic panel runs unconditionally — `kernel._gate` calls
it before consulting `REVIEWING` — so `STEWARD_GATE=on` at HEAD fills that cell
with no code change at all.

### Where arm C's remaining reward actually sits

The scoring law holds again at this size: DB passes iff every gold write was made
and no other write was.

| arm C, 150 sims | count | DB pass |
|---|---|---|
| made every gold write, and nothing else | 75 | 74 |
| made every gold write, **plus a surplus** | 21 | 5 |
| missed a gold write | 54 | **0** |

- **Under-writing is fatal without exception** — 0 of 54. Blocking cannot fix
  this; a gate cannot cause a write that was never proposed.
- **16 simulations did the whole job and lost to a surplus write**; 8 of those
  lost to exactly one. Blocking every surplus write in the run would take reward
  from 0.493 to **0.600** — a +0.107 ceiling, which is right at the resolution
  limit and is a ceiling, not a forecast.
- Surplus writes by tool: `cancel_reservation` 13, `book_reservation` 12,
  `update_reservation_flights` 4, `update_reservation_baggages` 2,
  `send_certificate` 1.

### What these runs cannot show

Gate decisions are **invisible in saved trajectories**. The whole system runs
inside one `generate_next_message` call, so a refusal, a revision and an
escalation never become emitted messages: searching all 300 simulations for the
`DENIAL` text finds zero in both Steward arms. Every question of the form "did
the gate cause this behaviour" is unanswerable from the artefacts on disk, which
is why the 2x2 above has to be filled by running it rather than by re-reading it.
Instrumenting this — a sidecar record of every proposal, verdict and reason —
is the prerequisite for the next round of gate work.

### Filling the 2x2: the sieve and the critic are not additive

> **VOID (2026-08-29).** Every "sieve on" cell below was measured while
> `records._loaded` could not parse a decorated tool result, so the deterministic
> tier was one check firing falsely on 100% of proposals and five checks that
> never executed once. Of 151 write proposals in a later journalled run, 11 were
> approved. The sieve scored positively here because refusing writes buys
> abstention and abstention carries reward on this benchmark -- that is a defect
> being rewarded, not a mechanism working. The numbers are kept as the record;
> none of them describes the system. See "The verifier tier had never once read a
> record". Not yet re-measured.


The three-arm experiment left one cell empty and could not attribute the Part 4
regression. Two more runs, same conditions, close it.

| | critic **off** | critic **on** |
|---|---|---|
| **sieve off** | **0.420** (cell 1) | **0.493** (arm C) |
| **sieve on** | **0.487** (arm A) | **0.500** (cell 2) |

Stock `llm_agent` scores 0.433 on the same grid. Cell 1's three simulations that
never returned bound it between 0.420 and 0.440; the ordering holds at either.

**Both mechanisms work, separately, and by about the same amount.** Against the
bare scaffold: the critic is +0.073 ± 0.043 (1.7 SE, p ~ 0.09), the sieve
+0.067 ± 0.037 (1.8 SE, p ~ 0.07). These are the two closest-to-significant
results this project has produced. They also correct an earlier claim in this
file: the `b653227` pair that put gate-on and gate-off both at 0.420 was a
single-trial coincidence, not evidence the critic was inert.

**Together they are not additive.** Additive would predict 0.560; cell 2 measures
0.500, so the interaction term is **-0.060** -- the second mechanism cancels
almost exactly one of the first. Cell 2 beats arm C by +0.007 (0.2 SE), which is
nothing, and beats the bare scaffold by +0.070 (1.9 SE, p ~ 0.058).

**They reach the same score by opposite routes, and the routes collide.**

| | gold write | under-wrote | full recall + surplus | lost to surplus | no-write tasks |
|---|---|---|---|---|---|
| cell 2 sieve+critic | 25.2% | 37 | **15** | **10** | **0.875** |
| arm C critic only | **42.9%** | 25 | 21 | 16 | 0.792 |
| arm A sieve only | 23.8% | 44 | 17 | 13 | 0.847 |
| cell 1 nothing on | 25.3% | 33 | 31 | 26 | 0.700 |

The critic converts by doing *more correct work* -- gold-write recall 25.3% ->
42.9%, transfer 48.3% -> 34.0%. The sieve converts by doing *less wrong work* --
recall flat, transfer up to 60.7%, surplus 31 -> 17. Run together, cell 2 keeps
the sieve's surplus control (best of any cell, only 0.033 recoverable) and loses
the critic's recall entirely (25.2%, the bare scaffold's number).

**The cause is ordering, and it is visible in the code.** `kernel._gate` runs the
deterministic panel first and returns `_refused` the moment a verifier fires, so
the critic only ever sees proposals the sieve has already cleared. The sieve
pre-empts the critic on exactly the proposals where the critic was earning its
+0.073.

**What this says to do:** ship the critic alone -- the `ef6cffd` configuration on
`main`. The sieve's verifiers are not wrong (cell 2 has the lowest surplus of any
configuration measured); they are wired to fire instead of inform. Making the
panel advisory, so what it finds becomes evidence the critic rules on rather than
a block that pre-empts it, is a `kernel._gate` change and the obvious next
experiment.

### Where the missed writes are, and three checks that cannot recover them

Arm C, decomposed by whether the task needs a write at all:

| arm | write tasks completed | zero-write tasks abstained | reward |
|---|---|---|---|
| gate off | 19 / 77 (25%) | 48 / 70 (69%) | 0.429 |
| critic on (arm C) | 23 / 78 (29%) | 59 / 72 (82%) | 0.493 |
| sieve + critic | 17 / 78 (22%) | 62 / 72 (86%) | 0.500 |

**The reward is carried by abstention.** On tasks that require writing, fewer
than one in three are finished, and the sieve arm bought its abstention gain by
doing less work -- the -0.060 interaction, in the units that matter.

Of the 55 simulations that miss a gold write, **39 make none of gold's writes at
all** and only 16 do part of the job. Of the 39, 23 made a different write
instead and 13 wrote nothing.

**It is not retrieval.** Every missed gold write whose record carries an
identifier was on a record the system had already read: 39 never-attempted plus
21 wrong-argument, and **zero** where the identifier never appeared in a tool
result. All 55 ended by `user_stop` -- the record sat in the transcript while the
conversation ran out.

The wrong arguments, field by field, over the 32 near-miss writes: `flights` 17,
`payment_methods` 10, `payment_id` 9, `reservation_id` 6, then `passengers`,
`destination`, `total_baggages`.

`flights` is the largest single field and it looked deterministic. It is not.
Three checks were built as replays over arm C and each is recorded here so it is
not built again:

| check | gold blocked | surplus caught |
|---|---|---|
| itinerary must be a connected chain, and keep the reservation's endpoints | 0 / 27 | **1 / 29** |
| every `(flight_number, date)` pair must have been returned by a search for that date, or already be on the record | 0 / 32 | **1 / 43** |
| a looser proximity version of the same grounding | 4 / 32 | 17 / 43 |

Gold itineraries are 20/20 connected and 20/20 endpoint-preserving, so the first
check is sound and nearly never fires: the wrong itineraries are *valid* ones.
The second says the same thing from the other side -- the flights the actor chose
were in the search results, for the right dates. It picked the wrong ones.

And it is not picking the cheapest one wrongly either. Where gold's itinerary
appears in the pool of options the run was shown, its price rank in the
reservation's cabin is 1/11, 3/11, 6/11, 9/10, 9/11, 17/18. **Gold is not the
cheapest**, so "choose the cheapest unless told otherwise" would be wrong more
often than right.

**Conclusion: `flights` is a choice, not a validity property.** What decides it
is the customer's stated preference in dialogue, and no rule over the records
reconstructs it. `ranking.py` -- which computes the comparison and refuses to
recommend -- is already the correct shape for this field, and the remaining error
is downstream of it.

### The payment family is already as covered as it gets

The other large near-miss field, and the same answer. Replaying the merged panel
over arm C's 135 executed writes: **0 of 62 gold blocked, 5 of 73 surplus caught**
-- `cancellable` 2, `compensation` 1, `not_yet_flown` 1, `payment_composition` 1.
That last is task 14, which pays a booking with two credit cards where the policy
allows one, and it is caught only because the sieve is now on the same branch as
the critic.

Two further payment checks were built and measured, and neither survives:

- **Amounts must not exceed the card's balance.** Fires on nothing: 0 of 19
  payment-bearing calls in the run overspends a gift card or certificate.
- **Amounts must sum to the reservation's total.** Already measured dead and
  documented in `adapters/tau2/payment.py` -- the fare lives in a search result
  the proposal does not carry, so any reconstruction is sometimes guessing.

What is left in `payment_id` and `payment_methods` is which of several *valid*
cards the customer wanted, and how the total was split across them. Like
`flights`, that is a dialogue fact, not a record fact.

**So both argument-side levers are closed.** The remaining reward is not in
making the writes more correct; it is in making them happen at all -- 39 of the
55 simulations that miss a gold write make none of gold's writes, and 13 of those
attempt no write whatsoever.

### The verifier reordering and the consent ledger: no movement, and a sharper trade

`veto_consent_50x3` is `d1384f6` -- the sieve on `dev`, the deterministic checks
moved to *after* the critic so they veto what it allowed rather than pre-empting
it, and consent recorded as a typed ledger entry instead of re-derived from the
transcript on every turn. Measured against arm C over the same 150 simulations:

    arm C     0.493
    this run  0.507      delta +0.013

There is no effect here. Seventeen simulations gained and fifteen lost, so a
paired sign test over the 32 discordant pairs gives **p = 0.86** -- a coin. Four
tasks (3, 10, 30, 34) flip in *both* directions across their own three trials at
temperature 0.0, and the per-trial means inside each arm span 0.44 to 0.54, which
is four times the difference being claimed. The +0.020 that replaying the
reordering over arm C's saved trajectories predicted is not visible at n=150.

What did move is the composition, and it moved the wrong way:

| | write tasks (n=78) | zero-write tasks (n=72) | mean |
|---|---|---|---|
| arm C | 0.218 | 0.792 | 0.493 |
| this run | **0.141** | **0.903** | 0.507 |

Six fewer tasks that need a write got one; eight more tasks that need no write
were correctly declined. The two cancel. This is the third consecutive arm to
show that trade, and the clearest: **every component added so far buys abstention
and pays for it in work.** The reward is flat because the two halves are close to
the same size, not because nothing happened.

It is worth being exact about what this does and does not condemn. The
reordering and the consent ledger were shipped for reasons that are still sound
-- the -0.060 interaction is real, and 70 of 166 refusals really were conditions
the customer had already answered. What the run says is that neither reaches
reward, because the binding constraint is not that the assistant is refused too
often. It is that the assistant does not try.

### The handoff verifier: built, measured live, reverted

> **SUPERSEDED (2026-08-29).** The nine-each-way result below was produced by a
> predicate reading an inflated ledger -- 38% of `Change.record` entries were
> prose placeholders that could never be discharged. With the ledger anchored
> (`49414eb`) the same check fires 27 times where gold does not transfer and 2
> where it does, and both of the 2 are task 13, which a record-level bar now
> excludes. Restored in `3738b25` at 27 and 0.


The one lever the abstention finding seemed to leave open. Splitting arm C's 51
transfers by whether the task needed a write at all is as clean a separation as
this corpus offers:

    transfers on tasks needing no write   41, of which 41 scored 1.00
    transfers on tasks needing a write    10, of which  9 scored 0.00

So a deterministic check was written to refuse a handoff while a planned change
was still outstanding -- the rule `agents.gate` already states in prose and let
ten through anyway. Whether the customer had asked for a person was tested first
and discarded: true in 12 of the 51, and it separates nothing.

It does not work, and the reason is worth keeping. **The predicate the check can
actually see is not the predicate that was scored.** Offline, "is a change owed"
was stood in for by ground truth -- does gold make a write on this task. Live,
it is `outstanding(state.changes, ...)`, and the planner records a change
whenever the *customer asks* for one, including on tasks where the answer is that
the policy forbids it. Task 13 is the proof: its gold action *is* a transfer, the
customer opens by asking for a cancellation, the planner writes
`cancel_reservation` into the ledger, and the check then reads a correct handoff
as abandonment.

Measured over a 15-task, 2-trial run with the decision journal on (28 of 30
simulations completed before it was stopped):

| | fired | outcome |
|---|---|---|
| simulations where transferring is correct | **9** | 1.000 -> 0.938 |
| simulations where it abandoned work | **9** | 0.083 -> 0.000 |

Nine each way is no signal, and it is exactly the failure this package was built
to escape -- `core.verifiers` opens by describing the critic as "a roughly
uniform refusal rate applied to everything", and this reproduced it in code with
fewer words. It cost one simulation, converted none, and added 35% to wall clock
(494s against 366s per simulation). Reverted in `1d0da84`.

Two things found on the way that outlive it:

- **`state.ruled_out`** stays. A change the policy forbids used to be owed for
  the rest of the conversation, so the planner, the critic and the speaker were
  all reasoning from a debt that could never be paid.
- **The decision journal is what settled this**, and it settled it in 25 minutes
  against a hypothesis that two separate offline replays had endorsed. Every
  measurement above this line was computed from tau2's saved trajectories, which
  contain only what left the system -- a refused proposal never becomes a tool
  call. That blind spot is what let a check with no discriminating power get
  built, tested, and shipped with a zero-false-block claim attached.

### The verifier tier had never once read a record

The journal was opened to settle a question about handoffs and answered a much
larger one. `records._loaded` called `json.loads` on a tool result that
`adapters/tau2/agent.py:_noted` had already appended English to -- the money
card, the baggage card, the eligibility card, the search comparison. It raised
`Extra data` and returned `None`, silently, on every record in every
conversation the sieve has ever run.

Measured over the 15x2 run of 2026-08-29 (28 scored simulations, 2,293 journal
records):

| | |
|---|---|
| reservations the agent read | 72 |
| reservations visible to `records.reservations()` | **0** |
| threads affected | 29 of 31 |
| write proposals reaching the gate | 151 |
| approved | **11 (7%)** |
| refused | 140 (93%) |
| refusals from `read_first`, on records already read | **56, all false** |
| simulations in which `read_first` fired | 5 -- **every one scored 0.0** |

The consequences ran both ways at once. `intended.read_first` fired on every
write, because no record was ever visible to it; and `cancellable`,
`not_yet_flown`, the three `modifications` checks and `compensation` returned
`None` on every proposal, because none of them could see a record either. So the
deterministic tier -- the centrepiece of Part 4 -- was in production exactly one
check, and that check was wrong 100% of the time. `context.facts` was handed to
the critic as "No record for this has been read" on all 63 of its refusals, which
is why it spent them re-deriving cancellation eligibility from prose.

`read_first`'s remediation is `recoverable`, so the Kernel appends `SELF_FIX` and
sends the actor straight back to fix something already correct. Task 19 trial 0,
verbatim:

    gate  get_reservation_details(Z7GOZK)     ALLOW
    gate  update_reservation_flights(Z7GOZK)  DENY: Nothing ... has read reservation Z7GOZK
    gate  get_reservation_details(Z7GOZK)     ALLOW
    gate  update_reservation_flights(Z7GOZK)  DENY: Nothing ... has read reservation Z7GOZK
    gate  update_reservation_flights(Z7GOZK)  DENY: ...

19 write proposals, 19 refusals, 16 turns, reward 0.0. Z7GOZK was refused 25
times, OBUT9V 20, 8C8K4E 11. **"The model does not act" was never true.** The
actor proposed 151 writes and the gate destroyed 140 of them.

It survived because both halves were tested apart. `test_verifiers_against_gold`
and `scripts/gate_bench.py` both replay the *raw* environment return, a shape
production never produces, so 49/49 gold and "8 surplus caught" were certified
against data that does not exist. Every number this log records about the sieve's
precision was measured on that shape and says nothing about the runs.

Replaying the run's own 94 gated proposals through the fixed panel: `read_first`
fires 0 times, 81 of 94 records are visible, and the five checks that had never
executed refuse 10 proposals -- 6 `cancellable`, 3 `flights_changeable`, 1
`not_yet_flown`. Tasks 19 and 21, the two worst livelocks, clear the sieve
entirely.

Fixed in `cf5262f`: the leading JSON value is parsed with `raw_decode` and the
tail is checked rather than ignored -- a blank line then an upper-case heading is
a note, anything else is refused, because bare `raw_decode` reads `2024-05-15` as
2024 and two records run together as the first. `tests/test_records.py` pins
every result shape a real run emits, and the gold replay now goes through
`_noted` (still 49/49).

**Nothing here is a score claim.** The fix removes 56 false refusals and switches
on five checks that have never been observed live. The write half of the
benchmark -- 2 of 14 here -- is where the headroom is, and it has been gated shut
for the whole life of the sieve. It has to be re-run before anything is claimed.

### The planner's ledger owed the same change eight times over

Found in the same journal. `Change.record` is meant to be an identifier, and 38%
of the 5,136 ledger entries carry prose instead: "the same reservation id",
"reservation_id_from_get_reservation_details", "the reservation ID that matches
the Houston-to-Denver return flight on 2024-05-27". Each phrasing is a different
`Change.key`, so `_widen` files every re-wording as a new commitment, and none of
them can ever be discharged -- `outstanding` matches an approved call's
identifiers against this text and a placeholder contains none. The ledger only
grew: task 21 ended owing ten changes that were four; across the run, 100 held
entries were 76 real targets.

That permanent debt is what kept `outstanding` non-empty on every write task,
which is what the speaker held 122 replies against (18 followed by an approved
write, ~15%), and what made the reverted handoff verifier fire on nine correct
transfers. The handoff verifier was not wrong about the ledger; the ledger was
wrong.

Fixed in `49414eb`. A bare identifier is kept whether or not anyone has read it,
because two bookings named in one breath have been read by nobody and dropping
both would merge two commitments into one -- `test_speaker` caught exactly that
when the first version of the rule was too eager.

### The fixed run, and what the write half fails at now

15x2, same 15 tasks, `STEWARD_GATE=on`, journalled both times. Both columns are
28 scored simulations counted by the same script.

| metric | before (`7b7d0d7`) | after (`49414eb`) |
|---|---|---|
| write proposals | 151 | 74 |
| writes approved | 11 | 43 |
| **approval rate** | **7.3%** | **58.1%** |
| gate refusals | 140 | 31 |
| -- `read_first`, all false | 56 | **0** |
| -- other deterministic | 21 | 3 |
| -- critic | 63 | 28 |
| speaker holds | 122 | 59 |
| act / talk turns | 336 / 367 | 232 / 218 |
| write-task reward | 0.143 | 0.231 |
| zero-write reward | 0.929 | 1.000 |
| overall | 0.536 | 0.643 |

Proposals per simulation 5.4 -> 2.6: the actor stopped re-proposing a write after
being falsely refused. **26 paired simulations, 1 gained, 0 lost.** One
discordant pair is p = 0.5; the reward column is not a result and is not claimed
as one. What is a result is that the risk did not materialise -- unblocking the
writes did not let surplus writes through, and zero-write went 13/14 to 15/15.

Both runs stopped short of 30 simulations: the first killed by hand, the second
stalled on tau2's user simulator, which goes through LiteLLM uninstrumented at a
300s timeout and 3 retries. The two missing simulations scored 0.0 in the
baseline.

**Where the write half fails now**, 13 simulations, 3 passing:

| bucket | sims | example |
|---|---|---|
| full gold recall, lost to **one surplus write** | 2 | task 44 |
| right tool, right record, **one wrong argument** | 2 | task 21 |
| wrong route or record entirely | 2 | task 24 |
| under-write: a gold write never made | 4 | 35, 39x2, 19/0 |

Task 44 makes all three gold `update_reservation_flights` calls with every field
exact -- reservation, cabin, flights, payment -- and loses to one extra write on
`S61CZX`. Task 21 gets reservation, cabin, payment and three of four segments
right and misses on one flight number.

That last one is a duration question, and it is arithmetic. Five one-stop options
were on the page; the run took the slowest, which was printed first; gold took the
quickest that had seats in the cabin. `ranking.py` computed cheapest-per-cabin
and earliest-departure and nothing else, and nothing else in the package computed
duration either. Closed in `64850c9`.

### A note on the Pass^k columns

The table above reports `scripts/score.py`'s estimator, "the first k trials all
passed", so it stays comparable with every earlier row. tau2's own definition
(`agent_metrics.py:126`) is `C(successes, k) / C(trials, k)`, the average over
every k-subset. Both are unbiased; tau2's has lower variance and the two coincide
at k = n. For these arms tau2's formula gives C 0.493 / 0.360 / 0.300, B 0.433 /
0.333 / 0.300, A 0.487 / 0.407 / 0.380. **Quote the tau2 figures in anything
outward-facing**, and note that `score.py`'s docstring claim that tau2 defines
pass^k as first-k is wrong.

## How to read this

- **At 50 tasks × 4 trials, a change smaller than ~0.09 average reward cannot be
  told from noise. At 50 × 1 it is ~0.23.** Most of the later rows use the cheap
  design and cannot resolve anything short of a very large effect. Each "unproven"
  verdict is a statement about the design, not evidence the change did nothing.
- **Pass^k is only defined where the trials allow it.** A 1-trial run has pass^1
  equal to its average reward and nothing else, which is why those columns are
  blank. Pass^4 was the headline while it existed, because it measured
  consistency; at one trial there is no substitute for it.
- **A simulation with no verdict scores 0**, whether it hung, crashed, or was cut.
  `Scored` makes that visible. `speaker_49` and `fixes_50` lost a quarter to a
  third of their simulations, so their rewards are floors and should not be read
  against complete runs.
- **Off-record writes is the number to watch now**, not gold write recall. Recall
  says the agent found the right action; off-record says what else it did on the
  way, and DB only pays when the second number is zero.

## Caveats that apply to every row

- **No run is deterministic, and none is meant to be.** `temperature` is discarded
  by this model whenever reasoning is enabled, and production would not run at 0.0
  anyway. Pass^k measured the thing determinism was standing in for — whether the
  system gets a task right *every* time — and at one trial nothing measures it.
- **Every ± above is a real measurement, not a rule of thumb.** Each task in a
  multi-trial run is run under one fixed config, so the spread across its own
  trials is the sampling noise; `scripts/score.py` computes it from the run file.
  Splitting a run into two disjoint 2-trial halves puts the same config 0.01–0.08
  apart on average reward, which is the floor seen directly. A 1-trial run has no
  such spread to draw on and gets no error bar at all.
- **Runs are labelled by the commit that was HEAD when they started**, recovered
  from simulation timestamps against `git log`. Where a run began within minutes
  of a commit, or where uncommitted work may have been in the tree, the label is
  the last commit certainly included.
- Pass^1 and the recall columns are recomputed from the raw results with one
  script, so they can differ by a few thousandths from the figures tau2's own
  summary printed for Run 001 (0.365 / 43.2%). The columns above are internally
  consistent.

## Per-task history

Every Steward simulation ever run at full scale, collapsed to one line per task:
**820 simulations across nine runs** (001, 002, diag, 003, 005, speaker_49,
fixes_50, parts_50, all_parts_50). The control is excluded from the mean --
it is a different agent -- and shown in its own column.

This table exists because a single trial cannot tell a change from weather.
Task 22 scored 1.0 in `all_parts_50` and 0.0 in the seventeen other
simulations of it; read against that one trial, a later run looks like a
regression when nothing regressed. `Mean` is the number to compare against.

`Band` is just the mean bucketed: **solid** >= 0.8, **volatile** 0.2-0.8,
**cold** 0-0.2, **never** exactly 0. `Writes` is how many write actions the
task's gold trajectory contains.

| Task | Sims | Mean | Band | Writes | Control |
|---|---|---|---|---|---|
| 0 | 18 | 0.89 | solid | 0 | 1 |
| 1 | 18 | 0.94 | solid | 0 | 1 |
| 2 | 16 | 0.62 | volatile | 0 | 1 |
| 3 | 16 | 0.38 | volatile | 0 | 0 |
| 4 | 18 | 0.94 | solid | 0 | 1 |
| 5 | 17 | 0.65 | volatile | 0 | 1 |
| 6 | 14 | 0.50 | volatile | 0 | - |
| 7 | 18 | 0.00 | never | 3 | 0 |
| 8 | 17 | 0.06 | cold | 1 | 1 |
| 9 | 18 | 0.67 | volatile | 0 | 1 |
| 10 | 18 | 0.56 | volatile | 0 | 0 |
| 11 | 18 | 0.11 | cold | 1 | 0 |
| 12 | 18 | 0.06 | cold | 1 | 1 |
| 13 | 18 | 0.83 | solid | 0 | 1 |
| 14 | 18 | 0.00 | never | 2 | 0 |
| 15 | 18 | 0.11 | cold | 1 | 0 |
| 16 | 18 | 0.11 | cold | 1 | 1 |
| 17 | 18 | 0.00 | never | 3 | 0 |
| 18 | 17 | 0.00 | never | 5 | 0 |
| 19 | 18 | 0.50 | volatile | 1 | 0 |
| 20 | 17 | 0.06 | cold | 1 | 0 |
| 21 | 17 | 0.00 | never | 2 | 0 |
| 22 | 18 | 0.06 | cold | 3 | 0 |
| 23 | 17 | 0.00 | never | 4 | 0 |
| 24 | 17 | 0.00 | never | 1 | 0 |
| 25 | 17 | 0.53 | volatile | 1 | 1 |
| 26 | 17 | 0.94 | solid | 0 | 1 |
| 27 | 16 | 0.81 | solid | 0 | 1 |
| 28 | 16 | 1.00 | solid | 0 | 1 |
| 29 | 17 | 0.00 | never | 2 | 0 |
| 30 | 17 | 0.18 | cold | 1 | 1 |
| 31 | 16 | 0.62 | volatile | 0 | 1 |
| 32 | 16 | 0.31 | volatile | 2 | 0 |
| 33 | 16 | 0.00 | never | 2 | 1 |
| 34 | 17 | 0.59 | volatile | 0 | 0 |
| 35 | 13 | 0.00 | never | 1 | - |
| 36 | 15 | 0.87 | solid | 0 | 1 |
| 37 | 16 | 0.00 | never | 1 | 0 |
| 38 | 16 | 0.75 | volatile | 0 | 1 |
| 39 | 16 | 0.00 | never | 3 | 0 |
| 40 | 16 | 0.50 | volatile | 1 | 0 |
| 41 | 15 | 0.67 | volatile | 0 | 1 |
| 42 | 15 | 0.00 | never | 2 | 0 |
| 43 | 15 | 0.93 | solid | 0 | 1 |
| 44 | 14 | 0.00 | never | 3 | 0 |
| 45 | 9 | 1.00 | solid | 0 | - |
| 46 | 15 | 1.00 | solid | 0 | 1 |
| 47 | 15 | 0.73 | volatile | 0 | 1 |
| 48 | 15 | 0.47 | volatile | 0 | 1 |
| 49 | 15 | 0.73 | volatile | 0 | 1 |

### What the table says

**Reward is a function of one variable, and it is not difficulty.**

| Gold writes | Tasks | Mean reward |
|---|---|---|
| 0 | 24 | 0.75 |
| 1 | 13 | 0.17 |
| 2 | 6 | 0.05 |
| 3 | 5 | 0.01 |
| 4 | 1 | 0.00 |
| 5 | 1 | 0.00 |

- The 24 tasks needing **no** write average **0.75**.
  The 26 that need one or more average **0.10**.
- **Not one of the 11 solid tasks requires a write. Not one of the
  14 never-passing tasks is write-free.** The separation is total.
- The collapse is monotone in the number of writes, which is what DB being
  all-or-nothing over the whole database predicts: each additional write is
  another chance to get an argument wrong, and one wrong argument scores the
  same as no write at all.

**Steward's mean of per-task means is 0.414**, over 820 simulations.
That is a better estimate of where the system actually sits than any single
run's headline -- `all_parts_50` printed 0.460 and `005` printed 0.333, and
neither is as informative as this.

**The control averages 0.532** on the 47 tasks it ran, still ahead. It
scores 1.0 on five tasks Steward averages under 0.2 on: 8, 12, 16, 30, 33.
Every one of those needs one or two writes. They are the cheapest evidence
available that those tasks are winnable by this model.

**Headroom, if the bands were converted:**

- the 14 never-passing tasks are worth **+0.28** reward
- the 8 cold tasks are worth **+0.15** on top of that
- the 17 volatile tasks are worth +0.14, but they are
  where run-to-run noise lives, so buying them is not the same kind of work.

### How to use it

Before claiming a change helped, look up the tasks it targeted here. A task
that flips from 0 to 1 having never passed in eighteen simulations is
evidence. A volatile task moving is not. And a run whose headline sits above
or below 0.414 has said nothing until the per-task rows are read: `all_parts_50`
scored 0 on tasks 2 and 6, which average 0.62 and 0.50, and 1.0 on task 22,
which averages 0.06.

## `steps1234_50x3` — what the gate was given

Commit `ea1e6ac`, `STEWARD_GATE=on`, seed 626729, 50 tasks × 3 trials, all 150
scored. Four changes, all to what the critic sees or is told, none to the
deterministic panel:

1. a "yes" to the assistant's *own* question is now filed as consent, so the
   first confirmation is no longer invisible to the gate;
2. the customer's request, at their scope, is shown above the proposed action;
3. every identifier in the proposal is labelled by whether the customer named it
   themselves or it appeared only in a lookup result;
4. the instructions name two authorities, the policy *and* the request, and the
   scope bullet is promoted out of the block list into the framing.

Paired against `final50x3` (`bdf1f3e`) on all 150 shared simulations:

| | reward | DB | COMMUNICATE | writes/sim | gold-write recall | surplus/sim |
|---|---|---|---|---|---|---|
| before `bdf1f3e` | 0.540 | 0.553 | 0.907 | 0.91 | 49/147 (33.3%) | 0.41 |
| after `ea1e6ac` | 0.560 | 0.580 | 0.900 | 0.88 | 52/147 (35.4%) | 0.37 |

**+0.020 ± 0.073** (0.5 SE), 17 better / 14 worse / 119 unchanged, sign test
p = 0.72. **Unproven**, and the design cannot prove anything smaller than about
+0.09.

What did move, and in the predicted direction: zero-write tasks that wrote
anyway fell **8 → 5**, surplus writes per simulation 0.41 → 0.37, DB +0.027.
COMMUNICATE fell 0.907 → 0.900 and ate part of it. Gold-write recall moved 2
points, which is inside its own noise.

The offline corpus predicted +3.6 clean simulations with a CI of [+1.3, +5.9];
the run delivered +3. That is the first time the offline proxy and the benchmark
have agreed, which makes the corpus worth more than this run's headline does.

**The gate is now close to spent.** 95 of 147 gold writes never reached the
database, and on the previous run only 23 of those were writes the gate refused
-- the rest were never proposed by anyone. That is planner and actor territory,
and there is no offline corpus for it yet.

## Run 019 — the deterministic seam works, the prompt rule does not

50 tasks, 1 trial, seed 626729, paired against run 017's trial 0 (which happens to
sit exactly on run 017's own average, 0.560, so it is a fair stand-in).

| | reward | DB | COMM | gold read | gold write |
|---|---|---|---|---|---|
| 017 t0 | 0.560 | 0.560 | 0.900 | 86.0% | 42.9% |
| 019 | 0.620 | 0.640 | 0.920 | **69.9%** | 40.8% |

Paired **+0.060 ± 0.103** (1.1 SE), 5 better / 2 worse / 43 unchanged. Unproven.

At the plan level, against criteria fixed before the run:

| | gold writes planned | unwanted | zero-write | reads/fake | surplus executed |
|---|---|---|---|---|---|
| 017 t0 | 72.9% | 101 (73%) | 37 | 8 | 26 |
| 019 | **63.3%** | 97 (**76%**) | 41 | **0** | **7** |

- **`performable`/`misfiled` did exactly what was specified.** Entries naming a read
  or a tool that does not exist: 8 → 0. This was the one row with a predicted value
  rather than a hoped-for one, and it is why the run was worth its cost.
- **Surplus writes reaching the database fell 26 → 7.** That, not recall, is where
  the reward gain comes from -- gold-write recall was flat.
- **The quote rule failed**, the same way run 018's scope instruction failed:
  unwanted changes 73% → 76%, the zero-write pool 37 → 41, and plan recall down
  9.6 points. Two measured attempts at telling this planner what *not* to plan,
  both negative. A third phrasing is not worth a run.
- **Gold read fell 86.0% → 69.9%** and is unexplained. Reads are free and do not
  enter reward, but they are how the actor grounds its arguments, and wrong
  arguments were run 017's largest failure class.

Method note: reward figures reported mid-run during this run were wrong twice --
at 37 of 50 tasks gold-write recall read 33.3% → 48.7% and finished flat, and the
absolute rewards were depressed because tau2 runs tasks roughly in order and the
unrun tail (38, 43, 45-49) averages 0.952. Partial slices of this benchmark are
biased, not merely noisy. Do not report them.

## Run 020 — the writes the workflows never named

The plans were read this time, not the aggregates, and the aggregates had been
pointing at the wrong thing. Three findings, in order of how much they cost.

**The gate refused zero gold writes.** Of the 24 gold writes run 019 missed, 13
were never planned and 11 were planned and never proposed by the actor. The
critic, which three runs of work went into, is not where writes are being lost.

**Three of the six writes this domain can make were named in no workflow at
all**: `update_reservation_passengers`, `update_reservation_baggages`,
`send_certificate`. On task 22 the actor told the customer *"the system can't
remove a passenger from an existing booking"* and offered to cancel and re-book
instead — against three gold writes, one of which was that exact call. The
workflows named every block and only one of the routes through them, and the
model generalised the blocks. Every write workflow now names its tool and says
what it can still do; `test_every_write_the_domain_can_make_is_named_by_the_
workflow_that_makes_it` fails if that stops being true.

**Under-writing is the failure mode, by 11 DB failures to 4.** Of run 019's 18
DB failures: 11 missed gold writes and wrote nothing wrong, 3 called the right
tool with wrong arguments, 2 did both, 2 wrote only surplus. The planner section
headed *"MANY REQUESTS ARE FINISHED BY ANSWERING THEM"* opened by calling a write
nobody asked for *"the single largest thing this plan gets wrong"* — and measured
against run 019 it changed the empty-`changes` rate not at all (37% before, 37%
after). It has been rebalanced to name both mistakes at their real weights.

Two plan-reading findings got a paragraph each in the planner instructions. Task
44: six consecutive plans reading "collect the data", "calculate the difference",
"determine which qualify", then a handoff, with `changes` never once filled in —
every fact needed had arrived by the third plan. Task 39: seven reservations
read, one sentence settling all seven as *already flown*, three gold
cancellations never proposed.

### What the ledger turned out not to be

Run 019's final-plan ledger holds 1.86 entries per simulation and 66% of them
name a record that is not gold, which the earlier analysis read as the planner
proposing three unwanted changes for every wanted one. Reading the plans, most of
that is *search*: the planner names a change on each candidate record while it is
still working out which record the customer means, and `_widen` cannot retract
(`kernel.py`, deliberately — "a re-plan may add and may not take away"). The
entries are also largely inert: **7 of 61 ever reached the gate.** The metric was
measuring an artefact of how the ledger accumulates, and the "should not do"
instruction written against it in run 019 was aimed at a target that was partly
manufactured. Left alone this round.

### The theme the next round is aimed at

Of the 15 run-019 DB failures with a write discrepancy, 6 involve reading three
or more reservations — and in **all six the agent had already read every gold
record before it went wrong.** Twelve of the fifteen had. The failure is almost
never that the record could not be found; it is that the request was matched to
the wrong one out of a set already in hand. Tasks 17, 21, 22, 39, 42 and 44 are
all this shape.

| Run | Reward | DB | COMM | Gold write | Gold read | Notes |
|---|---|---|---|---|---|---|
| 017 (baseline, 50×3) | 0.560 ±0.023 | — | — | 42.9% | 86.0% | best in project |
| 019 (50×1) | 0.620 | 0.640 | 0.920 | 40.8% | 69.9% | surplus writes 26 → 7 |
| 020 (50×2) | 0.530 ±0.050 | 0.566 | 0.909 | 43.9% | — | **paired −0.050 ±0.039 vs 017** |

### Run 020's verdict: one hallucination killed, one brake added by mistake

**Paired against run 017 over all 100 shared (task, trial): −0.050 ±0.039.**
Five simulations better, ten worse, 85 unchanged. Not significant at this design's
resolution, but the point estimate is negative and every supporting number agrees
with it: gold-write recall 51.7% → 43.9%, surplus writes per simulation 0.373 →
0.500, plan-level gold naming 68.7% → 61.1%.

**What worked.** Naming `update_reservation_flights` in the cabin workflow
eliminated the `update_reservation_cabin` hallucination completely — 0.069 entries
per plan in run 017, **0.000** in run 020, about 150 phantom commitments gone.
`book_reservation` recall went 40.0% → 70.0% and handoffs planned fell 0.023 →
0.018 per plan. The tool-naming half of this change is worth keeping.

**What broke, and it was one sentence.** The paragraph written against task 44's
determine-loop contained *"A record you have not read yet is a line in `lookups`,
not an entry."* That is a brake on naming changes, sitting inside a change whose
whole purpose was to get more of them named. Goals opening with "determine" went
**0.185 → 0.295 per plan (+59%)** and plans with empty `changes` went **37% →
42%** — the determine-loop got worse, not better, and it got worse by the amount
the new sentence discouraged. The three tools the workflows now name are planned
more often (baggages +41%, passengers +18%, book +90%) and executed *less*
successfully, because the plans that would have carried them out are the ones
that stayed empty a turn longer.

Recall by tool, run 017 → run 020, which is the clearest statement of the trade:

| tool | 017 | 020 |
|---|---|---|
| `book_reservation` | 40.0% | **70.0%** |
| `cancel_reservation` | 42.4% | 50.0% |
| `update_reservation_flights` | 56.7% | **32.5%** |
| `update_reservation_passengers` | 77.8% | **33.3%** |
| `update_reservation_baggages` | 60.0% | **30.0%** |

A correction to this file's earlier note: the "4× more" figure taken mid-run
compared run 020 against run 019's *partial* journal and did not survive the full
one. The real movements are the table above.

## Run 021 — in flight, on a degraded endpoint

Two changes on top of run 020, which was a regression.

**Taken back:** *"A record you have not read yet is a line in `lookups`, not an
entry."* Run 020's own numbers convicted it — see that section.

**Added, and structural rather than prose:** the planner is a fresh model call
every time it runs, so it cannot tell a first attempt at a question from a
fourth. That is why task 44 wrote six consecutive plans meaning *collect,
calculate, determine* while every fact it had asked for was already in hand.
`recap()` now records the last plan's goal and lookups on `StewardState.before`,
and `brief()` shows them back under a heading saying what a repeated goal means.
The changes are deliberately left out of the recap: they already reach the next
plan as what is still owed, and saying them twice would show one commitment as
two. Wired end to end and covered by a test that fails if the two halves come
apart.

**Kept from run 020:** every write workflow naming the tool it ends in. That half
worked and the numbers are in the run 020 section.

### A note on this run's wall clock

NVIDIA Build degraded badly part-way through. Bare `"Say OK."` calls against
`openai/gpt-oss-20b`, single request, no concurrency, timed at **14.7s, 46.5s,
71.4s, 47.7s** — against roughly 1s while run 020 was executing. Run 020 finished
100 simulations in about 65 minutes; run 021 took 80 minutes to finish 16.

The run was left going rather than killed or moved: `--auto-resume` keys finished
work on `(trial, task_id, seed)` so nothing already done is lost, and switching to
the OpenRouter fallback would have changed the provider under a comparison whose
baselines (017, 019, 020) are all on NVIDIA — a far larger confound than the
effect being measured. Concurrency was left at 8 rather than raised, because a
*single* uncontended call taking 71s is server-side saturation, and more parallel
requests would have bought timeouts rather than throughput.

Nothing about the slowdown was caused by the change: threads advanced normally
throughout, with varied, sensible goals and no repetition, and the log carries
zero timeout or rate-limit lines.
