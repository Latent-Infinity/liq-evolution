"""Tests for FeatureContext in-memory indicator caching."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from liq.evolution.primitives.feature_context import FeatureContext
from liq.evolution.protocols import IndicatorBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend() -> MagicMock:
    """Return a mock that satisfies the IndicatorBackend protocol."""
    backend = MagicMock(spec=["compute", "list_indicators"])
    # Default: compute returns a fresh array each time it is called.
    backend.compute.side_effect = lambda name, params, data, **kw: np.arange(
        5, dtype=float
    )
    backend.list_indicators.return_value = ["sma", "ema", "rsi"]
    return backend


def _sample_data() -> dict[str, np.ndarray]:
    return {"close": np.arange(100, dtype=float)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureContextCaching:
    """Cache-hit / cache-miss semantics."""

    def test_cache_hit_returns_same_identity(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        first = ctx.compute("sma", {"period": 14}, data)
        second = ctx.compute("sma", {"period": 14}, data)

        assert first is second

    def test_cache_miss_calls_backend_once(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        ctx.compute("sma", {"period": 14}, data)

        backend.compute.assert_called_once()

    def test_second_call_does_not_call_backend(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        ctx.compute("sma", {"period": 14}, data)
        backend.compute.reset_mock()

        ctx.compute("sma", {"period": 14}, data)

        backend.compute.assert_not_called()

    def test_different_params_are_separate_entries(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        r1 = ctx.compute("sma", {"period": 14}, data)
        r2 = ctx.compute("sma", {"period": 20}, data)

        assert r1 is not r2
        assert backend.compute.call_count == 2

    def test_different_indicators_are_separate(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        r1 = ctx.compute("sma", {"period": 14}, data)
        r2 = ctx.compute("ema", {"period": 14}, data)

        assert r1 is not r2
        assert backend.compute.call_count == 2

    def test_dict_order_does_not_affect_cache_key(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        first = ctx.compute("sma", {"period": 14, "offset": 1}, data)
        second = ctx.compute("sma", {"offset": 1, "period": 14}, data)
        third = ctx.compute("sma", {"period": 20, "offset": 1}, data)

        assert first is second
        assert third is not first
        assert backend.compute.call_count == 2

    def test_list_order_does_not_affect_cache_key(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        first = ctx.compute("sma", {"levels": [1, 2, 3]}, data)
        second = ctx.compute("sma", {"levels": [3, 2, 1]}, data)
        third = ctx.compute("sma", {"levels": [1, 2, 4]}, data)

        assert first is second
        assert third is not first
        assert backend.compute.call_count == 2

    def test_different_input_data_do_not_share_cache_entry(self) -> None:
        backend = _make_backend()
        backend.compute.side_effect = (
            lambda name, params, data, **kw: np.asarray(data["close"], dtype=float)
        )
        ctx = FeatureContext(backend)

        data_a = {"close": np.array([1.0, 2.0, 3.0], dtype=float)}
        data_b = {"close": np.array([10.0, 20.0, 30.0], dtype=float)}

        first = ctx.compute("sma", {"period": 14}, data_a)
        second = ctx.compute("sma", {"period": 14}, data_b)

        assert backend.compute.call_count == 2
        assert not np.array_equal(first, second)


class TestFeatureContextPassthrough:
    """Kwargs and delegation."""

    def test_out_buffer_passthrough(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()
        out_buf = np.empty(5)

        ctx.compute("sma", {"period": 14}, data, out=out_buf)

        _, kwargs = backend.compute.call_args
        assert kwargs.get("out") is out_buf

    def test_list_indicators_delegates(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)

        result = ctx.list_indicators()

        backend.list_indicators.assert_called_once_with(None)
        assert result == ["sma", "ema", "rsi"]


class TestFeatureContextInvalidation:
    """Cache invalidation."""

    def test_invalidate_clears_all(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        first = ctx.compute("sma", {"period": 14}, data)
        ctx.invalidate()
        second = ctx.compute("sma", {"period": 14}, data)

        # After invalidation a fresh compute should have happened.
        assert first is not second
        assert backend.compute.call_count == 2

    def test_invalidate_by_name(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        ctx.compute("sma", {"period": 14}, data)
        ctx.compute("ema", {"period": 14}, data)
        backend.compute.reset_mock()

        ctx.invalidate(name="sma")

        # SMA should re-compute; EMA should still be cached.
        ctx.compute("sma", {"period": 14}, data)
        ctx.compute("ema", {"period": 14}, data)

        assert backend.compute.call_count == 1
        backend.compute.assert_called_once_with("sma", {"period": 14}, data)

    def test_invalidate_by_name_and_params(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        data = _sample_data()

        ctx.compute("sma", {"period": 14}, data)
        ctx.compute("sma", {"period": 20}, data)
        backend.compute.reset_mock()

        ctx.invalidate(name="sma", params={"period": 14})

        # Only period=14 should re-compute; period=20 should remain cached.
        ctx.compute("sma", {"period": 14}, data)
        ctx.compute("sma", {"period": 20}, data)

        assert backend.compute.call_count == 1
        backend.compute.assert_called_once_with("sma", {"period": 14}, data)


class TestFeatureContextProtocol:
    """Protocol conformance."""

    def test_implements_indicator_backend_protocol(self) -> None:
        backend = _make_backend()
        ctx = FeatureContext(backend)
        assert isinstance(ctx, IndicatorBackend)


class TestFeatureContextHashing:
    """Coverage for cache key normalization internals."""

    def test_hashable_handles_ndarray(self) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        hashed = FeatureContext._hashable(arr)

        assert hashed[0] == "ndarray"
        assert hashed[1] == arr.shape
        assert hashed[2] == arr.dtype.str

    def test_hashable_falls_back_to_repr_for_unhashable_unknown(self) -> None:
        class Unknown:
            __hash__ = None

            def __repr__(self) -> str:
                return "unknown-object"

        hashed = FeatureContext._hashable(Unknown())
        assert hashed == "unknown-object"

    def test_hashable_handles_dict_params_order_independent(self) -> None:
        hashable = FeatureContext._hashable({"b": [1, 2, 3], "a": {"x": 1, "y": 2}})
        hashable_reordered = FeatureContext._hashable({"a": {"y": 2, "x": 1}, "b": [3, 2, 1]})

        assert hashable == hashable_reordered

    def test_hashable_handles_tuple_and_list_with_fallback(self) -> None:
        class UnhashableTuple(tuple):
            __hash__: object | None = None

        tuple_fallback = FeatureContext._hashable(UnhashableTuple([1, {"value": 2}]))
        list_fallback = FeatureContext._hashable([1, {"value": 2}])
        tuple_sorted = FeatureContext._hashable(UnhashableTuple([2, 1]))

        assert tuple_fallback[0] == "tuple"
        assert len(tuple_fallback) == 2
        assert tuple_sorted[0] == "tuple"
        assert list_fallback[0] == "list"
