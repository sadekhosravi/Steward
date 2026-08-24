"""Score a benchmark run and say how much of it is noise.

Every task is run several times under one fixed config, so the spread across a
task's own trials *is* the sampling noise. That makes the error bar estimable
from a single run -- no repeat needed, and no reliance on `temperature`, which
this endpoint discards whenever reasoning is enabled.

    uv run python scripts/score.py results/001_baseline.json
    uv run python scripts/score.py results/001_baseline.json results/002_gate.json

Given more than one run, each is also compared against the one before it, paired
on the tasks they share. Pairing is what makes the comparison sensitive: task
difficulty is by far the largest source of variation and it cancels exactly.

Read the two error bars differently. The one on average reward is a closed form
and is trustworthy. The one on pass^k comes from re-simulating the benchmark
from each task's observed rate, which is a fair estimate of the *spread* but a
biased estimate of the value -- pass^k is a product of estimated rates, so plugging
in a rate measured from four trials overstates it. Use it as a width, not a level.

Output is ASCII: the Windows console codepage cannot encode anything nicer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

# Draws for the pass^k error bar. Large enough that the interval is stable to
# the thousandth it is printed at, and it costs well under a second.
DRAWS = 20000

# Power to detect a difference at, for the "how big would it have to be" line.
# 2.8 = 1.96 (5% two-sided) + 0.84 (80% power), the textbook constant.
DETECTABLE = 2.8


@dataclass
class Run:
    """One benchmark run as a task -> per-trial rewards grid."""

    name: str
    grid: dict[str, list[float]]
    unscored: int

    @property
    def trials(self) -> int:
        return len(next(iter(self.grid.values())))

    @property
    def sims(self) -> int:
        return len(self.grid) * self.trials

    def avg(self) -> float:
        return sum(sum(v) for v in self.grid.values()) / self.sims

    def passk(self, k: int) -> float:
        """Share of tasks whose first k trials all passed.

        First k rather than best k: tau2 defines it that way, and any other
        choice would let the metric pick its own evidence.
        """
        return sum(1 for v in self.grid.values() if sum(v[:k]) == k) / len(self.grid)

    def se_avg(self) -> float:
        """Standard error of average reward under re-running the same tasks.

        A task's successes are Binomial(n, p) with its own p. The unbiased
        estimate of p(1-p) from a successes in n trials is a(n-a)/(n(n-1)) --
        which is why this needs no assumption about p being near a half, and why
        a task that went 4/4 or 0/4 contributes zero, as it should.
        """
        n = self.trials
        # One trial per task leaves nothing to estimate the spread *from*: the
        # within-task variance this is built on needs at least two draws of the
        # same task. Rather than divide by zero, say so with a nan and let the
        # report print the point estimate on its own -- a run with no error bar
        # is still worth reading, and a crash at the end of a 50-task run is not
        # a good way to learn that the design could not carry one.
        if n < 2:
            return float("nan")
        var = sum(a * (n - a) / (n * (n - 1)) / n for a in map(sum, self.grid.values()))
        return math.sqrt(var) / len(self.grid)

    def se_passk(self, k: int, seed: int = 0) -> float:
        """Spread of pass^k over re-runs, by re-simulating from each task's rate."""
        rng = random.Random(seed)
        rates = [sum(v) / self.trials for v in self.grid.values()]
        draws = [
            sum(all(rng.random() < p for _ in range(k)) for p in rates) / len(rates)
            for _ in range(DRAWS)
        ]
        return statistics.stdev(draws)


def load(path: Path) -> Run:
    """A run file as a complete grid, with missing simulations scored 0.

    A simulation that hung, crashed or was cut produced no verdict, and the
    honest reading is a failure rather than a gap: the agent never handed control
    back. Dropping them would quietly score a run on the subset it survived.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    trials = max(s["trial"] for s in data["simulations"]) + 1
    grid = {task["id"]: [0.0] * trials for task in data["tasks"]}
    scored = 0
    for sim in data["simulations"]:
        reward = (sim.get("reward_info") or {}).get("reward")
        if reward is None:
            continue
        grid[sim["task_id"]][sim["trial"]] = float(reward)
        scored += 1
    return Run(path.stem, grid, len(grid) * trials - scored)


def report(run: Run) -> None:
    se = run.se_avg()
    ci = 1.96 * se
    print(f"{run.name}")
    print(f"  simulations   {run.sims}  ({run.unscored} unscored, counted as 0)")
    if math.isnan(se):
        print(f"  avg reward    {run.avg():.3f}   (one trial per task: no error bar)")
    else:
        print(
            f"  avg reward    {run.avg():.3f} +/- {se:.3f}"
            f"      95% {run.avg() - ci:.3f} - {run.avg() + ci:.3f}"
        )
    for k in range(1, run.trials + 1):
        bar = f" +/- {run.se_passk(k):.3f}" if k > 1 else ""
        print(f"  pass^{k}        {run.passk(k):.3f}{bar}")
    print()


def compare(before: Run, after: Run) -> None:
    """Paired delta on the shared tasks, which is the only comparison worth making."""
    shared = sorted(set(before.grid) & set(after.grid))
    deltas = [
        sum(after.grid[t]) / after.trials - sum(before.grid[t]) / before.trials for t in shared
    ]
    delta = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas))
    z = delta / se if se else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)

    print(f"{after.name}  -  {before.name}      paired on {len(shared)} shared tasks")
    print(f"  avg reward    {delta:+.3f}   SE {se:.3f}   {z:+.1f} SE   p ~ {p:.2f}")
    same = len(shared) - better - worse
    print(f"  tasks better / worse / unchanged    {better} / {worse} / {same}")
    # The floor under every claim this run makes: an effect smaller than this
    # cannot be told from noise at this many tasks and trials, however real it is.
    print(f"  smallest detectable difference      {DETECTABLE * se:.3f}")
    print(f"  {'holds up' if p < 0.05 else 'unproven'} at the 5% level")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="tau2 results JSON, oldest first")
    args = parser.parse_args()

    runs = [load(path) for path in args.runs]
    print("=" * 78)
    for run in runs:
        report(run)
    for before, after in zip(runs, runs[1:], strict=False):
        print("=" * 78)
        compare(before, after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
