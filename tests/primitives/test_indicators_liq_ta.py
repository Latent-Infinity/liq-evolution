"""Tests for liq-ta indicator backend and registration."""

from __future__ import annotations

import numpy as np
import pytest

liq_ta = pytest.importorskip("liq_ta")

from liq.evolution.primitives.indicators_liq_ta import (  # noqa: E402
    LiqTAIndicatorBackend,
    _coerce_output,
    _make_multi_output_callable,
    _make_single_output_callable,
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


class TestCoerceOutput:
    """Direct tests for _coerce_output helper."""

    def test_none_result_no_buffer_raises(self) -> None:
        with pytest.raises(TypeError, match="returned None"):
            _coerce_output(None, out=None)

    def test_none_result_with_buffer_returns_buffer(self) -> None:
        buf = np.zeros(5, dtype=np.float64)
        result = _coerce_output(None, out=buf)
        assert result is buf

    def test_with_out_buffer_populates(self) -> None:
        buf = np.zeros(3, dtype=np.float64)
        data = np.array([1.0, 2.0, 3.0])
        result = _coerce_output(data, out=buf)
        assert result is buf
        np.testing.assert_array_equal(buf, [1.0, 2.0, 3.0])

    def test_no_buffer_returns_array(self) -> None:
        data = [1, 2, 3]
        result = _coerce_output(data, out=None)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class TestMakeSingleOutputCallableArity0:
    """Test _make_single_output_callable with n_inputs=0."""

    def test_no_out(self) -> None:
        def fake(**kwargs: object) -> np.ndarray:
            return np.array([1.0, 2.0])

        wrapper = _make_single_output_callable(fake, n_inputs=0, supports_out=False)
        result = wrapper()
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_supports_out_with_out(self) -> None:
        buf = np.zeros(3, dtype=np.float64)

        def fake(out: np.ndarray | None = None, **kwargs: object) -> None:
            out[:] = [7.0, 8.0, 9.0]  # type: ignore[index]

        wrapper = _make_single_output_callable(fake, n_inputs=0, supports_out=True)
        result = wrapper(out=buf)
        assert result is buf
        np.testing.assert_array_equal(buf, [7.0, 8.0, 9.0])

    def test_supports_out_without_out_falls_through(self) -> None:
        def fake(**kwargs: object) -> np.ndarray:
            return np.array([4.0, 5.0])

        wrapper = _make_single_output_callable(fake, n_inputs=0, supports_out=True)
        result = wrapper()
        np.testing.assert_array_equal(result, [4.0, 5.0])


class TestMakeSingleOutputCallableArity1:
    """Test _make_single_output_callable with n_inputs=1, supports_out path."""

    def test_supports_out_with_out(self) -> None:
        buf = np.zeros(3, dtype=np.float64)

        def fake(
            a: np.ndarray, out: np.ndarray | None = None, **kwargs: object
        ) -> None:
            out[:] = a * 2  # type: ignore[index]

        wrapper = _make_single_output_callable(fake, n_inputs=1, supports_out=True)
        result = wrapper(np.array([1.0, 2.0, 3.0]), out=buf)
        assert result is buf
        np.testing.assert_array_equal(buf, [2.0, 4.0, 6.0])


class TestMakeSingleOutputCallableArity2:
    """Test _make_single_output_callable with n_inputs=2."""

    def test_no_out(self) -> None:
        def fake(a: np.ndarray, b: np.ndarray, **kwargs: object) -> np.ndarray:
            return a + b

        wrapper = _make_single_output_callable(fake, n_inputs=2, supports_out=False)
        result = wrapper(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_supports_out_with_out(self) -> None:
        buf = np.zeros(2, dtype=np.float64)

        def fake(
            a: np.ndarray,
            b: np.ndarray,
            out: np.ndarray | None = None,
            **kwargs: object,
        ) -> None:
            out[:] = a - b  # type: ignore[index]

        wrapper = _make_single_output_callable(fake, n_inputs=2, supports_out=True)
        result = wrapper(np.array([5.0, 6.0]), np.array([1.0, 2.0]), out=buf)
        assert result is buf
        np.testing.assert_array_equal(buf, [4.0, 4.0])


class TestMakeSingleOutputCallableArity3:
    """Test _make_single_output_callable with n_inputs=3."""

    def test_no_out(self) -> None:
        def fake(
            a: np.ndarray, b: np.ndarray, c: np.ndarray, **kwargs: object
        ) -> np.ndarray:
            return a + b + c

        wrapper = _make_single_output_callable(fake, n_inputs=3, supports_out=False)
        result = wrapper(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
        )
        np.testing.assert_array_equal(result, [6.0])

    def test_supports_out_with_out(self) -> None:
        buf = np.zeros(1, dtype=np.float64)

        def fake(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            out: np.ndarray | None = None,
            **kwargs: object,
        ) -> None:
            out[:] = a + b + c  # type: ignore[index]

        wrapper = _make_single_output_callable(fake, n_inputs=3, supports_out=True)
        result = wrapper(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            out=buf,
        )
        assert result is buf
        np.testing.assert_array_equal(buf, [6.0])


class TestMakeSingleOutputCallableArity4:
    """Test _make_single_output_callable with n_inputs=4."""

    def test_no_out(self) -> None:
        def fake(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            d: np.ndarray,
            **kwargs: object,
        ) -> np.ndarray:
            return a + b + c + d

        wrapper = _make_single_output_callable(fake, n_inputs=4, supports_out=False)
        result = wrapper(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
        )
        np.testing.assert_array_equal(result, [10.0])

    def test_supports_out_with_out(self) -> None:
        buf = np.zeros(1, dtype=np.float64)

        def fake(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            d: np.ndarray,
            out: np.ndarray | None = None,
            **kwargs: object,
        ) -> None:
            out[:] = a + b + c + d  # type: ignore[index]

        wrapper = _make_single_output_callable(fake, n_inputs=4, supports_out=True)
        result = wrapper(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
            out=buf,
        )
        assert result is buf
        np.testing.assert_array_equal(buf, [10.0])


class TestMakeMultiOutputCallable:
    """Test _make_multi_output_callable for various arities."""

    def test_arity1(self) -> None:
        def fake(a: np.ndarray, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
            return (a * 2, a * 3)

        wrapper = _make_multi_output_callable(fake, n_inputs=1, output_index=1)
        result = wrapper(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, [3.0, 6.0])

    def test_arity1_with_out(self) -> None:
        buf = np.zeros(2, dtype=np.float64)

        def fake(a: np.ndarray, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
            return (a * 2, a * 3)

        wrapper = _make_multi_output_callable(fake, n_inputs=1, output_index=0)
        result = wrapper(np.array([1.0, 2.0]), out=buf)
        assert result is buf
        np.testing.assert_array_equal(buf, [2.0, 4.0])

    def test_arity2(self) -> None:
        def fake(
            a: np.ndarray, b: np.ndarray, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray]:
            return (a + b, a - b)

        wrapper = _make_multi_output_callable(fake, n_inputs=2, output_index=0)
        result = wrapper(np.array([5.0]), np.array([3.0]))
        np.testing.assert_array_equal(result, [8.0])

    def test_arity2_index1(self) -> None:
        def fake(
            a: np.ndarray, b: np.ndarray, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray]:
            return (a + b, a - b)

        wrapper = _make_multi_output_callable(fake, n_inputs=2, output_index=1)
        result = wrapper(np.array([5.0]), np.array([3.0]))
        np.testing.assert_array_equal(result, [2.0])

    def test_arity3(self) -> None:
        def fake(
            a: np.ndarray, b: np.ndarray, c: np.ndarray, **kwargs: object
        ) -> tuple[np.ndarray, np.ndarray]:
            return (a + b + c, a * b * c)

        wrapper = _make_multi_output_callable(fake, n_inputs=3, output_index=1)
        result = wrapper(np.array([2.0]), np.array([3.0]), np.array([4.0]))
        np.testing.assert_array_equal(result, [24.0])

    def test_arity4(self) -> None:
        def fake(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            d: np.ndarray,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (a + b + c + d, a * b * c * d)

        wrapper = _make_multi_output_callable(fake, n_inputs=4, output_index=0)
        result = wrapper(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
        )
        np.testing.assert_array_equal(result, [10.0])

    def test_arity4_index1(self) -> None:
        def fake(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            d: np.ndarray,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            return (a + b + c + d, a * b * c * d)

        wrapper = _make_multi_output_callable(fake, n_inputs=4, output_index=1)
        result = wrapper(
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
        )
        np.testing.assert_array_equal(result, [24.0])


class TestComputeEdgeCases:
    """Test compute() edge cases in LiqTAIndicatorBackend."""

    def test_unknown_indicator_raises(self, backend: LiqTAIndicatorBackend) -> None:
        # Use a name that exists as a liq_ta attribute but is not
        # a registered indicator nor a candlestick pattern.
        with pytest.raises(ValueError, match="Unknown indicator"):
            backend.compute("get_indicator_info", {}, {})

    def test_compute_with_supports_out(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        """SMA supports_out -- call compute with out buffer."""
        buf = np.zeros(100, dtype=np.float64)
        result = backend.compute("sma", {"period": 14}, sample_data, out=buf)
        # The buffer should be returned and populated
        assert result is buf
        # At least some values should be non-zero
        assert not np.all(buf == 0.0)
