"""Tests for liq-ta indicator backend and registration."""

from __future__ import annotations

import numpy as np
import pytest

liq_ta = pytest.importorskip("liq_ta")

from liq.evolution.primitives.indicators_liq_ta import (  # noqa: E402
    _build_primitive_name,
    _backend_candlestick_patterns,
    _backend_indicators,
    _canonical_output_suffixes,
    _canonical_param_name,
    _candidate_name_aliases,
    _candidate_output_aliases,
    _coerce_discrete_default,
    _coerce_output,
    _legacy_output_suffix,
    LiqTAIndicatorBackend,
    LiqFeaturesBackend,
    _make_param_specs_from_metadata,
    _make_multi_output_callable,
    _make_cached_indicator_callable,
    _make_single_output_callable,
    _make_candlestick_callable,
    _normalize_indicator_name,
    register_liq_ta_indicators,
)
from liq.gp.primitives.registry import PrimitiveRegistry  # noqa: E402
from liq.gp.types import BoolSeries, Series  # noqa: E402
from types import SimpleNamespace
from typing import Any


@pytest.fixture
def backend() -> LiqTAIndicatorBackend:
    return LiqTAIndicatorBackend()


@pytest.fixture
def features_backend() -> LiqFeaturesBackend:
    return LiqFeaturesBackend()


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
        sma = reg.get("sma")
        assert sma.output_type == Series
        assert sma.arity == 1
        assert len(sma.param_specs) == 1
        assert sma.param_specs[0].name == "period"

    def test_rsi_registered(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        rsi = reg.get("rsi")
        assert rsi.output_type == Series

    def test_macd_split_into_outputs(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        # MACD should produce 3 primitives
        macd_line = reg.get("macd_macd_line")
        signal = reg.get("macd_signal_line")
        histogram = reg.get("macd_histogram")
        assert macd_line.output_type == Series
        assert signal.output_type == Series
        assert histogram.output_type == Series

    def test_bollinger_split(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        upper = reg.get("bollinger_upper")
        middle = reg.get("bollinger_middle")
        lower = reg.get("bollinger_lower")
        assert upper.output_type == Series
        assert middle.output_type == Series
        assert lower.output_type == Series

    def test_candlestick_registered(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        doji = reg.get("cdl_doji")
        assert doji.output_type == BoolSeries
        assert doji.arity == 4  # open, high, low, close

    def test_sma_golden_value(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        sma = reg.get("sma")
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
        doji = reg.get("cdl_doji")
        n = 50
        rng = np.random.default_rng(42)
        close = 100.0 + np.cumsum(rng.standard_normal(n))
        open_ = close - rng.uniform(0, 1, n)
        high = close + rng.uniform(0, 2, n)
        low = close - rng.uniform(0, 2, n)
        result = doji.callable(open_, high, low, close)
        assert result.dtype == np.float64


class TestRegistrationCanonicalNames:
    """Canonical registration expectations."""

    def test_single_output_uses_canonical_name(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)

        rsi_canonical = reg.get("rsi")

        sample = np.arange(1.0, 6.0)
        canonical_out = rsi_canonical.callable(sample, period=14)

        assert canonical_out.shape == sample.shape

    def test_multi_output_uses_canonical_names(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)

        macd_signal = reg.get("macd_signal")
        macd_signal_line = reg.get("macd_signal_line")

        sample = np.arange(1.0, 12.0)
        signal = macd_signal.callable(sample, 12, 26, 9)
        signal_line = macd_signal_line.callable(sample, 12, 26, 9)

        np.testing.assert_array_equal(signal, signal_line)


class TestLiqFeaturesBackend:
    """Direct tests for LiqFeaturesBackend behavior."""

    def test_list_indicators_contains_expected(
        self,
        features_backend: LiqFeaturesBackend,
    ) -> None:
        indicators = features_backend.list_indicators()
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "sma" in indicators

    def test_compute_requires_supported_output_names(
        self,
        features_backend: LiqFeaturesBackend,
        sample_data: dict,
    ) -> None:
        rsi = features_backend.compute("rsi", {"period": 14}, sample_data)
        macd_signal = features_backend.compute(
            "macd",
            {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            sample_data,
            output_index=1,
        )
        assert rsi.shape == (100,)
        assert macd_signal.shape == (100,)
        assert rsi.dtype == np.float64

    def test_compute_with_unknown_indicator_raises(self) -> None:
        backend = LiqFeaturesBackend()
        with pytest.raises(ValueError, match="Unknown indicator: not_a_real_indicator"):
            backend.compute("not_a_real_indicator", {}, {"close": np.arange(10.0)})

    def test_compute_rejects_unsupported_kwargs(self) -> None:
        backend = LiqFeaturesBackend()
        with pytest.raises(TypeError, match="Unsupported kwargs"):
            backend.compute("rsi", {"period": 14}, {"close": np.arange(10.0)}, bad=True)

    def test_compute_with_out_buffer(
        self,
        sample_data: dict[str, np.ndarray],
    ) -> None:
        backend = LiqFeaturesBackend()
        output = np.empty(len(sample_data["close"]), dtype=np.float64)
        result = backend.compute(
            "rsi",
            {"period": 14},
            sample_data,
            out=output,
        )
        assert result is output

    def test_list_indicators_category_filter(self) -> None:
        backend = LiqFeaturesBackend()
        by_group = backend.list_indicators(category="trend")
        assert isinstance(by_group, list)

    def test_candlestick_registration_only_uses_available_liq_features_names(self) -> None:
        backend = LiqFeaturesBackend()
        patterns = _backend_candlestick_patterns(backend)
        available = set(backend.list_indicators())
        assert set(patterns).issubset(available)

    def test_indicator_backend_normalize_length_without_close(self) -> None:
        assert LiqFeaturesBackend._normalize_length({"data": np.arange(5.0)}) == 5

    def test_align_output_branches(self) -> None:
        target = 6
        np.testing.assert_array_equal(
            LiqFeaturesBackend._align_output(np.array([1.0, 2.0]), None, target),
            np.array([np.nan, np.nan, np.nan, np.nan, 1.0, 2.0]),
        )
        np.testing.assert_array_equal(
            LiqFeaturesBackend._align_output(
                np.array([1.0, 2.0, 3.0, 4.0]),
                None,
                3,
            ),
            np.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            LiqFeaturesBackend._align_output(
                np.array([9.0, 10.0]),
                np.array([], dtype=int),
                4,
            ),
            np.array([np.nan, np.nan, 9.0, 10.0]),
        )
        np.testing.assert_array_equal(
            LiqFeaturesBackend._align_output(
                np.array([3.0, 4.0]),
                np.array([4, 1], dtype=np.int64),
                6,
            ),
            np.array([np.nan, 4.0, np.nan, np.nan, 3.0, np.nan]),
        )
        np.testing.assert_array_equal(
            LiqFeaturesBackend._align_output(
                np.array([6.0, 7.0]),
                np.array([4.0, 1.0], dtype=np.float64),
                2,
            ),
            np.array([6.0, 7.0]),
        )
        np.testing.assert_array_equal(
            LiqFeaturesBackend._align_output(
                np.array([2.0, 5.0, 7.0]),
                np.array([10.0, 11.0, 12.0], dtype=np.float64),
                2,
            ),
            np.array([5.0, 7.0]),
        )


class TestParamSpecsFromMetadata:
    """ParamSpec generation from metadata."""

    def test_metadata_param_grid_lookup_by_indicator_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def _fake_get_param_grid(name: str) -> dict[str, list[int]]:
            calls.append(name)
            return {"period": [2, 3, 5, 8]}

        monkeypatch.setattr(
            "liq.features.indicators.param_grids.get_param_grid",
            _fake_get_param_grid,
        )

        specs = _make_param_specs_from_metadata(
            {"name": "rsi", "default_params": {"period": 14}}
        )

        assert calls == ["rsi"]
        assert len(specs) == 1
        assert specs[0].name == "period"
        assert specs[0].allowed_values == [2, 3, 5, 8]


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


class TestIndicatorHelperFunctions:
    """Unit tests for internal helper behavior."""

    def test_legacy_output_suffixes(self) -> None:
        assert _legacy_output_suffix("macd", "histogram") == "histogram"
        assert _legacy_output_suffix("bollinger", "upper") == "upperband"
        assert _legacy_output_suffix("bollinger", "middle") == "middleband"
        assert _legacy_output_suffix("aroon", "aroon_up") == "aroonup"
        assert _legacy_output_suffix("aroon", "aroon_down") == "aroondown"
        assert _legacy_output_suffix("rsi", "value") == "value"

    def test_canonical_output_suffixes(self) -> None:
        assert _canonical_output_suffixes("bollinger", "upper") == {
            "upper",
            "upperband",
        }
        assert _canonical_output_suffixes("macd", "macd") == {"macd", "macd_line"}
        assert _canonical_output_suffixes("bollinger", "middle") == {
            "middle",
            "middleband",
        }
        assert _canonical_output_suffixes("bollinger", "lower") == {
            "lower",
            "lowerband",
        }
        assert _canonical_output_suffixes("stochastic", "k") == {
            "k",
            "fastk",
            "slowk",
            "stoch_k",
            "slowk",
        }
        assert _canonical_output_suffixes("stochastic", "d") == {
            "d",
            "fastd",
            "slowd",
            "stoch_d",
            "slowd",
        }
        assert _canonical_output_suffixes("sma", "real") == {"real", "value"}
        assert _canonical_output_suffixes("aroon", "aroon_down") == {
            "aroon_down",
            "aroondown",
        }

    def test_alias_builders(self) -> None:
        assert _candidate_name_aliases("ta_sma") == ["sma"]
        assert _candidate_name_aliases("macd_signal") == ["macd_signal"]
        assert _candidate_name_aliases("foo_foo_bar") == [
            "foo_bar",
            "foo_foo_bar",
        ]
        assert _candidate_output_aliases("macd", "signal") == sorted(
            {"signal", "signal_line"}
        )
        assert _build_primitive_name("macd", "signal_line") == "macd_signal_line"
        assert _build_primitive_name("macd", None) == "macd"

    def test_backend_helper_fallbacks(self) -> None:
        assert isinstance(
            _backend_indicators(SimpleNamespace()),
            dict,
        )
        assert _backend_candlestick_patterns(SimpleNamespace(_candlestick_patterns=[])) == []
        nested = SimpleNamespace(_backend=SimpleNamespace())
        assert isinstance(_backend_candlestick_patterns(nested), list)
        assert len(_backend_candlestick_patterns(nested)) > 0

    def test_canonical_param_name(self) -> None:
        assert _canonical_param_name("timeperiod3") == "period3"
        assert _canonical_param_name("SLOWPERIOD") == "slow_period"
        assert _canonical_param_name("custom") == "custom"

    def test_coerce_discrete_default(self) -> None:
        assert _coerce_discrete_default(14, [14, 20, 30]) == 14
        assert _coerce_discrete_default("x", [2, 4, 6]) == 2
        assert _coerce_discrete_default(9, [2, 4, 6]) == 6

    def test_make_param_specs_from_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake_grid(name: str) -> dict[str, list[int]]:
            return {"period": [2, 3, 5, 8]}

        monkeypatch.setattr(
            "liq.features.indicators.param_grids.get_param_grid",
            _fake_grid,
            raising=False,
        )

        specs = _make_param_specs_from_metadata(
            {"name": "rsi", "default_params": {"timeperiod": 14}}
        )
        assert specs
        assert len(specs) == 1
        assert specs[0].name == "period"
        assert 2 in specs[0].allowed_values
        assert 8 in specs[0].allowed_values

    def test_make_param_specs_from_metadata_supports_discrete_grid(self) -> None:
        specs = _make_param_specs_from_metadata(
            {
                "name": "rsi",
                "default_params": {"timeperiod": 14},
                "parameters": {"timeperiod": 14},
            }
        )
        assert len(specs) == 1
        assert specs[0].name == "period"
        assert specs[0].allowed_values is not None
        assert specs[0].default in specs[0].allowed_values

    def test_make_param_specs_from_metadata_falls_back_to_range(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake_grid(name: str) -> dict[str, list[int]]:
            return {"period": []}

        monkeypatch.setattr(
            "liq.features.indicators.param_grids.get_param_grid",
            _fake_grid,
            raising=False,
        )

        specs = _make_param_specs_from_metadata(
            {"name": "sma", "default_params": {"period": 14}, "params": ["period"]}
        )
        assert specs
        assert specs[0].name == "period"
        assert specs[0].allowed_values is None

    def test_make_param_specs_from_metadata_skips_bool_defaults(self) -> None:
        specs = _make_param_specs_from_metadata(
            {"name": "sma", "default_params": {"period": True}}
        )
        assert specs == []


class TestCallableFactories:
    """Unit tests for internal callable factories."""

    def test_single_output_wrapper_branches(self) -> None:
        out = np.zeros(3, dtype=float)
        base = np.array([1.0, 2.0, 3.0])

        def _writer(a: np.ndarray, out: np.ndarray | None = None) -> None:
            if out is not None:
                out[:] = a * 2
                return None
            return a * 2

        direct = _make_single_output_callable(_writer, 1, supports_out=True)
        np.testing.assert_array_equal(direct(base, out=out), [2.0, 4.0, 6.0])

        def _fn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a + b

        wrapped_two = _make_single_output_callable(_fn, 2, supports_out=False)
        np.testing.assert_array_equal(
            wrapped_two(np.array([1.0, 1.0]), np.array([2.0, 3.0])),
            np.array([3.0, 4.0]),
        )

        def _fn_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
            return a + b + c

        wrapped_three = _make_single_output_callable(_fn_three, 3, supports_out=False)
        np.testing.assert_array_equal(
            wrapped_three(
                np.array([1.0, 1.0]),
                np.array([2.0, 2.0]),
                np.array([3.0, 3.0]),
            ),
            np.array([6.0, 6.0]),
        )

        def _fn_four(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
            return a + b + c + d

        wrapped_four = _make_single_output_callable(_fn_four, 4, supports_out=False)
        np.testing.assert_array_equal(
            wrapped_four(
                np.array([1.0, 1.0]),
                np.array([2.0, 2.0]),
                np.array([3.0, 3.0]),
                np.array([4.0, 4.0]),
            ),
            np.array([10.0, 10.0]),
        )

    def test_multi_output_wrapper_branches(self) -> None:
        def _fn(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return a, a * 2

        wrapped_single = _make_multi_output_callable(_fn, 1, 1)
        np.testing.assert_array_equal(
            wrapped_single(np.array([1.0, 2.0])), np.array([2.0, 4.0])
        )

        def _fn4(
            a: np.ndarray,
            b: np.ndarray,
            c: np.ndarray,
            d: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return a + b + c + d, a

        wrapped_else = _make_multi_output_callable(_fn4, 4, 0)
        np.testing.assert_array_equal(
            wrapped_else(
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
                np.array([1.0]),
            ),
            np.array([4.0]),
        )

    def test_candlestick_wrapper(self) -> None:
        def _fn(
            open_: np.ndarray,
            high: np.ndarray,
            low: np.ndarray,
            close: np.ndarray,
        ) -> np.ndarray:
            return np.asarray(open_) + (np.asarray(high) - np.asarray(low)) - close

        wrapped = _make_candlestick_callable(_fn)
        sample = np.array([1.0, 2.0, 3.0])
        result = wrapped(sample, sample, sample, sample)
        assert result.dtype == np.float64

    def test_cached_indicator_callable_positional_params_and_errors(
        self,
    ) -> None:
        class Backend:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, Any], dict[str, np.ndarray], Any]] = []

            def compute(
                self,
                name: str,
                params: dict[str, Any],
                data: dict[str, np.ndarray],
                **kwargs: Any,
            ) -> np.ndarray:
                if name == "insufficient":
                    raise ValueError("insufficient data for indicator")
                self.calls.append((name, params, data, kwargs))
                return np.asarray(data["a"]) * 2

        backend = Backend()
        wrapper = _make_cached_indicator_callable(
            backend,
            "ok",
            ["a"],
            0,
            param_names=["multiplier", "other"],
        )

        input_data = np.array([1.0, 2.0, 3.0])
        output = wrapper(input_data, 2, 7)
        assert len(backend.calls) == 1
        assert backend.calls[0][1] == {"multiplier": 2, "other": 7}
        np.testing.assert_array_equal(output, np.array([2.0, 4.0, 6.0]))

        with pytest.raises(TypeError, match="expects at least"):
            _make_cached_indicator_callable(backend, "ok", ["a", "b"], 0)(input_data)

        short = _make_cached_indicator_callable(backend, "insufficient", ["a"], 0)
        result = short(np.array([1.0, 2.0]))
        assert np.isnan(result).all()

        out = np.zeros(2, dtype=float)
        result_with_out = short(np.array([1.0, 2.0]), out=out)
        np.testing.assert_array_equal(result_with_out, out)
        assert np.isnan(out).all()

    def test_backend_helpers_and_normalization(self) -> None:
        source = {"a": {"inputs": ["a"], "outputs": ["value"]}}
        direct = SimpleNamespace(_indicators=source, _candlestick_patterns=[])
        nested = SimpleNamespace(_backend=direct)

        assert _backend_indicators(direct) == source
        assert _backend_indicators(nested) == source
        assert _normalize_indicator_name("TA_TEST") == "test"
        assert _backend_candlestick_patterns(direct) == []
        assert _backend_candlestick_patterns(nested) == []

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
        # Use a name that exists in liq_ta but is not a registered indicator.
        with pytest.raises(ValueError, match="Unknown indicator"):
            backend.compute("get_indicator_registry", {}, {})

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


class TestAllIndicatorsRegistered:
    """Verify every indicator in liq_ta.INDICATORS produces canonical primitives."""

    def test_all_liq_ta_indicators_registered(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        """Every indicator in liq_ta.INDICATORS must produce at least one primitive."""
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        registered_names = {
            name
            for name, primitive in reg._primitives.items()
            if primitive.category == "indicator"
        }

        missing = []
        for name, meta in liq_ta.INDICATORS.items():
            outputs = meta["outputs"]
            if len(outputs) == 1:
                expected = name
                if expected not in registered_names:
                    missing.append(expected)
            else:
                for out_name in outputs:
                    expected = f"{name}_{out_name}"
                    if expected not in registered_names:
                        missing.append(expected)

        assert missing == [], f"Missing primitives: {missing}"

    def test_align_output_returns_nan_for_empty_values(self) -> None:
        np.testing.assert_array_equal(
            LiqFeaturesBackend._align_output(np.array([]), None, 4),
            np.array([np.nan, np.nan, np.nan, np.nan]),
        )

    def test_rust_only_indicators_now_visible(self) -> None:
        """Indicators from the Rust registry should now be in liq_ta.INDICATORS."""
        rust_only = [
            "ao",
            "bears_power",
            "bulls_power",
            "connors_rsi",
            "demarker",
            "dpo",
            "dss_bressert",
            "laguerre_rsi",
            "osma",
            "rvi",
            "stc",
            "gaussian_filter",
            "hma",
            "supertrend",
            "vortex",
            "autocorr",
            "chop",
            "gaussian_channel",
            "hma_atr_bands",
            "hma_bollinger_bands",
            "hurst",
            "vwap_atr_bands",
            "vwap_bollinger_bands",
            "ulcer_index",
        ]
        for name in rust_only:
            assert name in liq_ta.INDICATORS, f"{name} not in INDICATORS"


class TestNewlyAddedIndicators:
    """Spot-check newly visible indicators for correct arity, output type, params."""

    def test_hma_registered(self, backend: LiqTAIndicatorBackend) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        hma = reg.get("hma")
        assert hma.output_type == Series
        assert hma.arity == 1  # single input: data
        assert len(hma.param_specs) == 1
        assert hma.param_specs[0].name == "period"

    def test_supertrend_multi_output(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        st = reg.get("supertrend_supertrend")
        assert st.output_type == Series
        assert st.arity == 3  # high, low, close
        # Check multiplier param is present
        param_names = [p.name for p in st.param_specs]
        assert "period" in param_names
        assert "multiplier" in param_names

    def test_vortex_multi_output(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        plus_vi = reg.get("vortex_plus_vi")
        minus_vi = reg.get("vortex_minus_vi")
        assert plus_vi.output_type == Series
        assert minus_vi.output_type == Series
        assert plus_vi.arity == 3

    def test_connors_rsi_params(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        crsi = reg.get("connors_rsi")
        param_names = [p.name for p in crsi.param_specs]
        assert "rsi_period" in param_names
        assert "streak_period" in param_names
        assert "rank_period" in param_names

    def test_laguerre_rsi_gamma_param(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        lrsi = reg.get("laguerre_rsi")
        param_names = [p.name for p in lrsi.param_specs]
        assert "gamma" in param_names
        gamma_spec = next(p for p in lrsi.param_specs if p.name == "gamma")
        assert gamma_spec.dtype is float
        assert gamma_spec.min_value == 0.01
        assert gamma_spec.max_value == 0.99

    def test_autocorr_lag_is_registered_as_int_and_accepts_float_input(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)

        autocorr = reg.get("autocorr")
        lag_spec = next((p for p in autocorr.param_specs if p.name == "lag"), None)
        assert lag_spec is not None
        assert lag_spec.dtype is int

        result = autocorr.callable(sample_data["data"], lag=5.0)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == len(sample_data["data"])


class TestParamCoverage:
    """Verify no indicator params are silently dropped."""

    def test_params_registered_for_each_indicator(self) -> None:
        """Each indicator metadata parameter is represented in registered primitives."""
        missing = []
        for name, meta in liq_ta.INDICATORS.items():
            metadata = {
                "name": name,
                "params": list(meta["params"]),
            }
            specs = _make_param_specs_from_metadata(metadata)
            spec_names = {spec.name for spec in specs}

            for param in meta["params"]:
                canonical = _canonical_param_name(param)
                if canonical and canonical in spec_names:
                    continue
                missing.append(f"{name}.{param}")
        assert missing == []

    def test_existing_indicator_params_not_dropped(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        """Ichimoku, keltner_channel, qqe should have all params registered."""
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)

        # Ichimoku has 4 params and 5 outputs
        tenkan = reg.get("ichimoku_tenkan")
        ichimoku_params = [p.name for p in tenkan.param_specs]
        assert "tenkan_period" in ichimoku_params
        assert "kijun_period" in ichimoku_params
        assert "senkou_b_period" in ichimoku_params
        assert "displacement" in ichimoku_params

        # Keltner channel has 2 params
        kc_upper = reg.get("keltner_channel_upper")
        kc_params = [p.name for p in kc_upper.param_specs]
        assert "period" in kc_params
        assert "atr_multiplier" in kc_params

        # QQE has 4 params
        qqe = reg.get("qqe_qqe")
        qqe_params = [p.name for p in qqe.param_specs]
        assert "rsi_period" in qqe_params
        assert "smoothing_period" in qqe_params
        assert "wilders_period" in qqe_params
        assert "factor" in qqe_params


class TestNewIndicatorGoldenValues:
    """Spot-check computation of new indicators through the cached backend."""

    def test_hma_golden_value(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        """HMA should produce finite values for sufficient data."""
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        hma = reg.get("hma")
        result = hma.callable(sample_data["data"], period=9)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert len(result) == 100
        # At least some non-NaN values in the tail
        valid = result[~np.isnan(result)]
        assert len(valid) > 50

    def test_connors_rsi_bounded(
        self,
        backend: LiqTAIndicatorBackend,
    ) -> None:
        """ConnorsRSI values should be bounded [0, 100]."""
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        crsi = reg.get("connors_rsi")
        # ConnorsRSI with rank_period=50 requires >= 51 data points
        rng = np.random.default_rng(42)
        data = 100.0 + np.cumsum(rng.standard_normal(200))
        result = crsi.callable(
            data,
            rsi_period=14,
            streak_period=2,
            rank_period=50,
        )
        assert isinstance(result, np.ndarray)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_supertrend_computes(
        self,
        backend: LiqTAIndicatorBackend,
        sample_data: dict,
    ) -> None:
        """Supertrend should produce finite values."""
        reg = PrimitiveRegistry()
        register_liq_ta_indicators(reg, backend)
        st = reg.get("supertrend_supertrend")
        result = st.callable(
            sample_data["high"],
            sample_data["low"],
            sample_data["close"],
            period=10,
            multiplier=3.0,
        )
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
