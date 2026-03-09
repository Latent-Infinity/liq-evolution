"""Signal output dataclass for GP strategy predictions.

Duck-type compatible with ``liq.signals.output.SignalOutput`` so that
``GPStrategyAdapter`` can satisfy the ``liq-runner`` Strategy protocol
without depending on ``liq-signals``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

RegimeLabel = Literal["trend", "range", "neutral", "fallback", "no_trade", "empty"]


@dataclass(frozen=True)
class RegimeState:
    """Typed regime annotation for strategy outputs."""

    label: RegimeLabel
    confidence: float
    occupancy: float
    reason_code: str = "ok"
    turnover: float = 0.0

    def __post_init__(self) -> None:
        if self.label not in {
            "trend",
            "range",
            "neutral",
            "fallback",
            "no_trade",
            "empty",
        }:
            raise ValueError("label must be a supported regime state")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not np.isfinite(self.occupancy) or not 0.0 <= self.occupancy <= 1.0:
            raise ValueError("occupancy must be finite and in [0, 1]")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("reason_code must be a non-empty string")
        if not np.isfinite(self.turnover) or self.turnover < 0.0:
            raise ValueError("turnover must be finite and >= 0")


@dataclass(frozen=True)
class GPSignalOutput:
    """Prediction output from a GP-evolved trading strategy.

    Attributes:
        scores: Prediction scores as a polars Series.
        labels: Optional ground-truth label Series.
        metadata: Arbitrary metadata dict.
        regime_state: Optional typed regime annotation for downstream
            deterministic policy checks.
    """

    scores: pl.Series
    labels: pl.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    regime_state: RegimeState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scores, pl.Series):
            raise TypeError("scores must be a polars Series")
        if self.labels is not None and not isinstance(self.labels, pl.Series):
            raise TypeError("labels must be a polars Series when provided")
        if self.labels is not None and self.labels.len() != self.scores.len():
            raise ValueError(
                "scores and labels must have equal length when labels provided"
            )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")
        if self.regime_state is not None and not isinstance(self.regime_state, RegimeState):
            raise TypeError("regime_state must be a RegimeState when provided")
