"""Signal provider adapter for GP strategies.

This adapter turns GP strategy outputs (score vectors) into LIQ ``Signal`` objects.
The implementation is intentionally conservative and produces one signal per symbol,
using the latest bar from each provided DataFrame.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from liq.core import PortfolioState
from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.protocols import GPStrategy
from liq.signals import Signal


@dataclass
class GPSignalProvider:
    """Wrap a GP strategy into a LIQ signal provider.

    Args:
        strategy: GP strategy compatible with :class:`GPStrategy`.
        symbols: Optional explicit symbol universe.
        long_threshold: Score >= threshold becomes ``long``.
        short_threshold: Score <= threshold becomes ``short``.
        name: Provider name.
    """

    strategy: GPStrategy
    symbols: Sequence[str] | None = None
    long_threshold: float = 0.0
    short_threshold: float = 0.0
    lookback: int = 0
    _name: str = "gp_signal_provider"

    def __post_init__(self) -> None:
        if self.lookback < 0:
            raise ValueError("lookback must be >= 0")
        if self.long_threshold < 0 and self.short_threshold < 0:
            raise ValueError("thresholds should be >= 0")

        self.symbols = list(self.symbols) if self.symbols else []

    @property
    def required_history(self) -> int:
        return self.lookback

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(
        self,
        data: Any = None,
        portfolio_state: PortfolioState | None = None,  # noqa: ARG002
    ) -> Iterable[Signal]:
        """Generate signals for provided bar data.

        Supported input shapes:
        - ``pl.DataFrame``: single-symbol batch
        - ``dict[str, pl.DataFrame]``: per-symbol batches
        """

        if data is None:
            raise ValueError("data is required")

        if isinstance(data, pl.DataFrame):
            symbol_map = {self._resolve_symbol_name(None): data}
        elif isinstance(data, dict):
            symbol_map = self._coerce_symbol_dict(data)
        else:
            raise TypeError("data must be a polars DataFrame or dict[str, DataFrame]")

        symbols = self.symbols or list(symbol_map.keys())
        signals: list[Signal] = []

        for symbol in symbols:
            df = symbol_map.get(symbol)
            if df is None or df.height == 0:
                continue

            output = self._predict_df(df)
            if not isinstance(output, GPSignalOutput):
                continue

            scores = output.scores.to_numpy()
            if len(scores) == 0:
                continue

            score = float(scores[-1])
            direction = self._direction_from_score(score)
            if direction == "flat":
                continue

            timestamp = self._extract_timestamp(df)
            signals.append(
                Signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    direction=direction,
                    strength=float(abs(score)),
                    target_weight=None,
                    horizon=None,
                    metadata={"score": score},
                )
            )

        return signals

    def _resolve_symbol_name(self, symbol: str | None) -> str:
        if symbol:
            return symbol
        if self.symbols:
            return self.symbols[0]
        return "UNKNOWN"

    @staticmethod
    def _coerce_symbol_dict(data: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
        out: dict[str, pl.DataFrame] = {}
        for key, value in data.items():
            if not isinstance(value, pl.DataFrame):
                raise TypeError("dictionary values must be polars DataFrame")
            out[str(key)] = value
        return out

    def _predict_df(self, df: pl.DataFrame) -> GPSignalOutput:
        output = self.strategy.predict(df)
        if not isinstance(output, GPSignalOutput):
            raise TypeError("strategy.predict must return GPSignalOutput")
        return output

    @staticmethod
    def _extract_timestamp(df: pl.DataFrame) -> datetime:
        if "timestamp" in df.columns:
            value = df[-1, "timestamp"]
            if isinstance(value, datetime):
                if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                    return value.replace(tzinfo=UTC)
                return value.astimezone(UTC)
        return datetime.now(UTC)

    def _direction_from_score(self, score: float) -> str:
        if score > self.long_threshold:
            return "long"
        short_cutoff = abs(self.short_threshold)
        if short_cutoff > 0 and score <= -short_cutoff:
            return "short"
        if short_cutoff == 0 and score < 0:
            return "short"
        return "flat"
