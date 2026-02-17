"""GP strategy adapter for liq-runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl


class GPStrategyAdapter:
    """Adapts GP-evolved programs into liq-runner strategies.

    Implements the :class:`~liq.evolution.protocols.GPStrategy` protocol
    by wrapping a GP program tree as a trading strategy.
    """

    def fit(self, features: pl.DataFrame, labels: pl.Series | None) -> None:
        """Fit the strategy to training data.

        Args:
            features: Feature DataFrame.
            labels: Optional label Series.
        """
        raise NotImplementedError

    def predict(self, features: pl.DataFrame) -> Any:
        """Generate predictions from features.

        Args:
            features: Feature DataFrame.

        Returns:
            Predictions or signals.
        """
        raise NotImplementedError
