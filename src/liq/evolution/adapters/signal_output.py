"""Signal output dataclass for GP strategy predictions.

Duck-type compatible with ``liq.signals.output.SignalOutput`` so that
``GPStrategyAdapter`` can satisfy the ``liq-runner`` Strategy protocol
without depending on ``liq-signals``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl


@dataclass(frozen=True)
class GPSignalOutput:
    """Prediction output from a GP-evolved trading strategy.

    Attributes:
        scores: Prediction scores as a polars Series.
        labels: Optional ground-truth label Series.
        metadata: Arbitrary metadata dict.
    """

    scores: pl.Series
    labels: pl.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
