"""Fact evidence.

One module per declared fact surface. A module here is the evidence a fact is
true: it is cited by the evidence index, it is watched by the fact-surface gate
(`liq-experiments/src/liq/experiments/checks/fact_surface_diff.py`), and it
changes only under a Fact Change or an Evidence Maintenance task.

Oracles, harnesses, comparators and port contract tests are *not* evidence and
live in `tests/support/`. Everything else stays where it is.
"""
