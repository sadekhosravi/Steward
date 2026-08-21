# Run log

Every benchmark run on one line. The reasoning lives in
[`reports.md`](reports.md); this file exists so the trend can be read at a
glance.

**Held fixed across all runs so far:** domain `airline` (50 tasks), agent model
`nvidia:openai/gpt-oss-20b` at reasoning effort `low`, user simulator
`nvidia_nim/openai/gpt-oss-20b`, 4 trials, max 200 steps, concurrency 8.

## Scores

| Run | Change | Avg reward | Pass^1 | Pass^2 | Pass^3 | **Pass^4** | Solved 4/4 | Never solved |
|---|---|---|---|---|---|---|---|---|
| 001 | baseline: bare ReAct, no gates | 0.365 ±0.023 | 0.360 | 0.260 | 0.200 | **0.160 ±0.029** | 8 / 50 | 23 / 50 |
| 002 | + PRE-GATE, GATE, COMMIT | **0.420 ±0.021** | 0.340 | 0.280 | 0.260 | **0.220 ±0.031** | 11 / 50 | 22 / 50 |

Deltas against the run before, paired on the same 50 tasks:

| Runs | Avg reward | Tasks better / worse / unchanged | Verdict |
|---|---|---|---|
| 002 − 001 | +0.055 ± 0.033 (1.7 SE, p ≈ 0.10) | 14 / 5 / 31 | unproven at 5% |

## Behaviour

| Run | DB | COMMUNICATE | Writes emitted | Gold write recall | Gold read recall | Scored sims | Median / max wall clock |
|---|---|---|---|---|---|---|---|
| 001 | 77/196 (39.3%) | 174/196 (88.8%) | 240 | 11/195 (5.6%) | 159/369 (43.1%) | 196 / 200 (4 hung) | 41 s / 2050 s |
| 002 | **92/197 (46.7%)** | 173/197 (87.8%) | **125** | 8/194 (4.1%) | 139/367 (37.9%) | 197 / 198 (1 crash, 2 cut) | 35 s / 2797 s |

## How to read this

- **Pass^4 is the headline; average reward is what decisions are made on.**
  Pass^4 counts a task only when all four trials passed, so it measures
  consistency — the thing an architecture is supposed to buy. It is also the
  noisiest number here, because it reduces four trials to one yes/no. Average
  reward, paired on the same tasks, is roughly three times tighter and is the
  number to judge a change by.
- **At 50 tasks × 4 trials, a change smaller than ~0.09 average reward cannot be
  told from noise.** Worth knowing before building something, not after. Halving
  that bar means four times the simulations; adding a second domain buys the same
  precision and more coverage.
- **DB is where the losses are.** COMMUNICATE has sat near 88% from the start;
  every point of headroom is in what the agent does to the database.
- **Writes emitted is the gate's block rate, indirectly.** Gate verdicts were not
  instrumented for these two runs, so the drop from 240 to 125 on near-identical
  read volume is the only measurement of how often it refuses. From Run 003 the
  verdicts themselves are on the record, in Langfuse.
- **A simulation with no verdict scores 0**, whether it hung, crashed, or was
  cut. Those are counted in the denominator of Pass^k but not in DB/COMMUNICATE.

## Caveats that apply to every row

- **No run is deterministic, and none is meant to be.** `temperature` is
  discarded by this model whenever reasoning is enabled, and production would not
  run at 0.0 anyway. Pass^k already measures the thing determinism was standing in
  for: whether the system gets a task right *every* time.
- **Every ± above is a real measurement, not a rule of thumb.** Each task is run
  four times under one fixed config, so the spread across its own trials is the
  sampling noise; `scripts/score.py` computes it from the run file. Splitting a
  run into two disjoint 2-trial halves puts the same config 0.01–0.08 apart on
  average reward, which is the floor seen directly.
- Pass^1 and recall are recomputed from the raw results with one script, so they
  can differ by a few thousandths from the figures tau2's own summary printed
  for Run 001 (0.365 / 43.2%). The columns above are internally consistent.
