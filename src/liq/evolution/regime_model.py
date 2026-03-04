"""Typed regime domain model for block-structured GP programs.

The model is intentionally lightweight and dependency-free at runtime:
it only validates structure and local arithmetic constraints and does not
perform evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import math

from liq.evolution.program import Program


class RegimeId(StrEnum):
    """Supported high-level regime labels.

    This list mirrors the structured regime state labels currently used in
    `liq.evolution.adapters.signal_output`.
    """

    trend = "trend"
    range = "range"
    neutral = "neutral"
    fallback = "fallback"
    no_trade = "no_trade"
    empty = "empty"


@dataclass(frozen=True)
class RegimeWeights:
    """Container for expert blend coefficients.

    Values are validated eagerly and immutable.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("RegimeWeights values cannot be empty")

        for value in self.values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError("RegimeWeights values must be int/float")
            if not math.isfinite(float(value)):
                raise ValueError("RegimeWeights values must be finite")
            if value < 0.0:
                raise ValueError("RegimeWeights values must be non-negative")

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)


@dataclass(frozen=True)
class _RegimeBlock:
    """Internal base class for role-specific blocks."""

    label: RegimeId
    program: Program


@dataclass(frozen=True)
class RegimeDetector(_RegimeBlock):
    """Regime detector block."""


@dataclass(frozen=True)
class RegimeGate(_RegimeBlock):
    """Regime gate block."""


@dataclass(frozen=True)
class RegimeExpert(_RegimeBlock):
    """Expert block for a regime branch."""


@dataclass(frozen=True)
class RegimeRisk(_RegimeBlock):
    """Optional risk block."""


@dataclass(frozen=True)
class RegimeModel:
    """Typed, validated regime model.

    Required roles:
    - one detector
    - one gate
    - at least one expert
    - optional risk
    """

    detector: RegimeDetector
    gate: RegimeGate
    experts: Sequence[RegimeExpert]
    risk: RegimeRisk | None = None
    weights: RegimeWeights | None = None

    def __post_init__(self) -> None:
        if not self.experts:
            raise ValueError("RegimeModel requires at least one expert")

        if self.weights is None:
            object.__setattr__(
                self,
                "weights",
                RegimeWeights(tuple(1.0 for _ in self.experts)),
            )
        elif len(self.weights) != len(self.experts):
            raise ValueError(
                "RegimeWeights length must match number of experts: "
                f"{len(self.weights)} != {len(self.experts)}"
            )

