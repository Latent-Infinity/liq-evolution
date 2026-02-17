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
