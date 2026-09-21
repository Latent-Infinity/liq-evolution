"""Oracles, harnesses and contract tests that fact evidence leans on.

Nothing here is evidence in its own right, and that separation is the point: a
fact can be weakened without its evidence module changing at all — loosen a
tolerance in an oracle, swap the stream a harness replays — so these files are
declared inside the fact surfaces that depend on them and are watched with them.

Port contract tests (Tier 2, no ledger row) live in `tests/support/contract/`.
"""
