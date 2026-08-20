"""Harness-agnostic primitives: messages, tool specs, run state, blackboard.

Nothing in this package may import tau2. Everything else in `mas` is allowed to
depend on it, which is what keeps the agent design portable and unit-testable
without booting a benchmark environment."""
