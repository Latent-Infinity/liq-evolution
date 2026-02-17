"""Genome v2: multi-program genome with optional exit logic and risk parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

from liq.gp.program.ast import Program


@dataclass(frozen=True)
class Genome:
    """Multi-program genome with optional exit logic and risk parameters.

    Attributes:
        entry_program: The primary GP program (entry signal).
        exit_program: Optional GP program for exit signals.
        risk_genes: Evolved risk management parameters.
    """

    entry_program: Program
    exit_program: Program | None = None
    risk_genes: dict[str, float] = field(default_factory=dict)
