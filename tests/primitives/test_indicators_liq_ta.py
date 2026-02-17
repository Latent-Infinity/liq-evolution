"""Tests for liq-ta indicator backend and registration."""

from __future__ import annotations

import numpy as np
import pytest

liq_ta = pytest.importorskip("liq_ta")

from liq.evolution.primitives.indicators_liq_ta import (  # noqa: E402
    LiqTAIndicatorBackend,
    register_liq_ta_indicators,
)
from liq.gp.primitives.registry import PrimitiveRegistry  # noqa: E402
from liq.gp.types import BoolSeries, Series  # noqa: E402


@pytest.fixture
def backend() -> LiqTAIndicatorBackend:
    return LiqTAIndicatorBackend()


@pytest.fixture
def sample_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 100
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    return {
        "data": close,
        "data0": close,
        "data1": close * 1.01,
        "open": close - rng.uniform(0, 1, n),
        "high": close + rng.uniform(0, 2, n),
        "low": close - rng.uniform(0, 2, n),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
        "periods": np.linspace(2, 30, n, dtype=np.float64),
    }


class TestBackendDiscovery:
    def test_discovers_indicators(self, backend: LiqTAIndicatorBackend) -> None:
        assert backend.indicator_count > 50

    def test_discovers_candlestick_patterns(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        assert backend.candlestick_count > 40

    def test_list_indicators_returns_all(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        all_names = backend.list_indicators()
        assert len(all_names) == backend.indicator_count

    def test_list_by_category(self, backend: LiqTAIndicatorBackend) -> None:
        momentum = backend.list_indicators(category="momentum")
        assert len(momentum) > 5
        assert "rsi" in momentum
        assert "macd" in momentum


class TestBackendCompute:
    def test_sma(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        result = backend.compute("sma", {"period": 20}, sample_data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 100

    def test_rsi(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        result = backend.compute("rsi", {"period": 14}, sample_data)
        assert isinstance(result, np.ndarray)
        # RSI values (where not NaN) should be between 0 and 100
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_multi_output_macd(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        params = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        # Get MACD line (index 0)
        macd_line = backend.compute("macd", params, sample_data, output_index=0)
        assert isinstance(macd_line, np.ndarray)
        # Get signal line (index 1)
        signal = backend.compute("macd", params, sample_data, output_index=1)
        assert isinstance(signal, np.ndarray)

    def test_candlestick(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        result = backend.compute("cdl_doji", {}, sample_data)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


class TestRegistration:
    def test_registers_indicators(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        indicators = reg.list_primitives(category="indicator")
        assert len(indicators) > 100  # many indicators + multi-output splits

    def test_sma_registered(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        sma = reg.get("ta_sma")
        assert sma.output_type == Series
        assert sma.arity == 1
        assert len(sma.param_specs) == 1
        assert sma.param_specs[0].name == "period"

    def test_rsi_registered(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        rsi = reg.get("ta_rsi")
        assert rsi.output_type == Series

    def test_macd_split_into_outputs(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        # MACD should produce 3 primitives
        macd_line = reg.get("ta_macd_macd_line")
        signal = reg.get("ta_macd_signal_line")
        histogram = reg.get("ta_macd_histogram")
        assert macd_line.output_type == Series
        assert signal.output_type == Series
        assert histogram.output_type == Series

    def test_bollinger_split(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        upper = reg.get("ta_bollinger_upper")
        middle = reg.get("ta_bollinger_middle")
        lower = reg.get("ta_bollinger_lower")
        assert upper.output_type == Series
        assert middle.output_type == Series
        assert lower.output_type == Series

    def test_candlestick_registered(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        doji = reg.get("ta_cdl_doji")
        assert doji.output_type == BoolSeries
        assert doji.arity == 4  # open, high, low, close

    def test_sma_golden_value(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        sma = reg.get("ta_sma")
        data = np.arange(1.0, 11.0)
        result = sma.callable(data, period=5)
        # SMA(5) at bar 4: mean(1,2,3,4,5) = 3.0
        assert result.dtype == np.float64
        np.testing.assert_allclose(result[4], 3.0)

    def test_candlestick_coercion(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        """Candlestick output should be float64 even though Rust returns i32."""
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        doji = reg.get("ta_cdl_doji")
        n = 50
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.standard_normal(n))
        open_ = close - rng.uniform(0, 1, n)
        high = close + rng.uniform(0, 2, n)
        low = close - rng.uniform(0, 2, n)
        result = doji.callable(open_, high, low, close)
        assert result.dtype == np.float64
