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

| Run | Change | Trials | Avg reward | Pass^1 | Pass^2 | Pass^3 | Pass^4 | Scored |
|---|---|---|---|---|---|---|---|---|
| 001 | baseline: bare ReAct, no gates | 4 | 0.365 ±0.023 | 0.360 | 0.260 | 0.200 | **0.160 ±0.029** | 196/200 |
| 002 | + PRE-GATE, GATE, COMMIT | 4 | 0.420 ±0.021 | 0.340 | 0.280 | 0.260 | **0.220 ±0.031** | 197/200 |
| diag | rename to Steward, temperature 0.0, Langfuse | 1 | 0.380 | 0.380 | — | — | — | 50/50 |
| 003 | + enforced tool schemas, first real system prompt | 1 | 0.420 | 0.420 | — | — | — | 49/50 |
| 004b | **control: tau2's own `llm_agent`, no scaffold** | 1 | **0.500** | 0.500 | — | — | — | 47/50 |
| 005 | + planner agent, policy sections routed per turn | 3 | 0.333 ±0.031 | 0.360 | 0.220 | 0.120 | — | 130/150 |
| speaker_49 | + airport codes the policy assumes (see note) | 2 | 0.270 ±0.041 | 0.420 | 0.100 | — | — | 63/100 |
| fixes_50 | + SPEAKER, the check on the reply | 1 | 0.280 | 0.280 | — | — | — | 38/50 |
| parts_50 | + workflows, planner in the loop, deterministic checks | 1 | 0.420 | 0.420 | — | — | — | 47/50 |
| all_parts_50 | + Parts 1–6, harmony repair | 1 | **0.460** | 0.460 | — | — | — | **50/50** |

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
  scored 0.500 against Steward's best of 0.460. Everything built so far has bought
  intermediate accuracy that the scoring function does not pay for.
- **DB is where the losses are.** COMMUNICATE has sat between 84% and 92% since
  the first run and has never been the bottleneck.

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
