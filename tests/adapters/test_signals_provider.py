"""Tests for :class:`GPSignalProvider`."""

from __future__ import annotations

import pytest

pytest.importorskip("liq.core", reason="liq-core not installed")

from datetime import UTC, datetime, timedelta  # noqa: E402

import polars as pl  # noqa: E402

from liq.evolution.adapters.signal_output import GPSignalOutput  # noqa: E402
from liq.evolution.adapters.signals_provider import GPSignalProvider  # noqa: E402


class _SymbolScoreStrategy:
    def __init__(self, scores_by_symbol: dict[str, float]):
        self._scores = scores_by_symbol

    def fit(self, features: pl.DataFrame, labels: pl.Series | None = None) -> None:
        return None

    def predict(self, features: pl.DataFrame) -> GPSignalOutput:
        symbol = (
            str(features["symbol"][0]) if "symbol" in features.columns else "UNKNOWN"
        )
        score = self._scores.get(symbol, 0.0)
        values = [0.0] * (len(features) - 1) + [float(score)]
        return GPSignalOutput(scores=pl.Series("scores", values))


def _bar_df(score_symbol: str, score_ts: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [score_symbol, score_symbol],
            "timestamp": [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 1, 0, 0, tzinfo=UTC) + timedelta(hours=score_ts),
            ],
            "close": [100.0, 101.0],
        }
    )


class TestGPSignalProviderProtocol:
    def test_protocol_properties(self) -> None:
        provider = GPSignalProvider(
            strategy=_SymbolScoreStrategy({"AAPL": 0.2}),
            symbols=["AAPL", "MSFT"],
            lookback=10,
            long_threshold=0.5,
        )

        assert provider.required_history == 10
        assert provider.symbols == ["AAPL", "MSFT"]
        assert provider.name == "gp_signal_provider"


class TestGPSignalProviderSignals:
    def test_generate_signals_dict_input(self) -> None:
        strategy = _SymbolScoreStrategy({"AAPL": 0.7, "MSFT": -0.2, "GOOG": 0.0})
        provider = GPSignalProvider(
            strategy=strategy,
            symbols=["AAPL", "MSFT", "GOOG"],
            long_threshold=0.5,
            short_threshold=0.1,
        )

        output = provider.generate_signals(
            {
                "AAPL": _bar_df("AAPL", 1),
                "MSFT": _bar_df("MSFT", 2),
                "GOOG": _bar_df("GOOG", 3),
            }
        )
        signals = list(output)
        assert len(signals) == 2
        assert {s.symbol for s in signals} == {"AAPL", "MSFT"}

        long_signal = next(s for s in signals if s.symbol == "AAPL")
        short_signal = next(s for s in signals if s.symbol == "MSFT")
        assert long_signal.direction == "long"
        assert short_signal.direction == "short"

    def test_generate_signals_flat_on_zero_by_default(self) -> None:
        provider = GPSignalProvider(strategy=_SymbolScoreStrategy({"AAPL": 0.0}))
        signals = list(provider.generate_signals(_bar_df("AAPL", 10)))
        assert len(signals) == 0

    def test_timestamp_extracted_from_last_row(self) -> None:
        provider = GPSignalProvider(
            strategy=_SymbolScoreStrategy({"AAPL": 0.6}),
            symbols=["AAPL"],
            long_threshold=0.2,
        )

        ts_a = datetime(2026, 1, 2, tzinfo=UTC)
        ts_b = datetime(2026, 1, 2, 1, 0, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "timestamp": [ts_a, ts_b],
                "close": [10.0, 10.5],
            }
        )
        signal = list(provider.generate_signals(df))[0]
        assert signal.timestamp == ts_b
        assert signal.metadata["score"] == pytest.approx(0.6)

    def test_generate_signals_none_data_raises(self) -> None:
        provider = GPSignalProvider(strategy=_SymbolScoreStrategy({"AAPL": 0.1}))
        with pytest.raises(ValueError, match="data is required"):
            provider.generate_signals(None)
