"""Port contract tests — adapter-parametrised bases every adapter is run against.

These are Tier-2 boundaries: they carry no fact-ledger row, so they are not
evidence and do not belong under `tests/facts/`. The bases defined here are
imported by consumers so that one contract governs every implementation of a
port, in this repository and in the repositories that depend on it.
"""
