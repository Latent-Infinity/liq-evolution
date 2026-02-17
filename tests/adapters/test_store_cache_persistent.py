"""Tests for persistent caching via StoreBackend integration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from liq.evolution.adapters.store_cache import EvolutionStoreCache
from liq.evolution.protocols import StoreBackend


class FakeStoreBackend:
    """Simple dict-backed StoreBackend for testing."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data


class FailingStoreBackend:
    """StoreBackend that raises on every operation."""

    def get(self, key: str) -> bytes | None:
        raise RuntimeError("store.get failed")

    def put(self, key: str, data: bytes) -> None:
        raise RuntimeError("store.put failed")


# ------------------------------------------------------------------ #
#  Protocol compliance
# ------------------------------------------------------------------ #


class TestStoreBackendProtocol:
    def test_fake_store_satisfies_protocol(self) -> None:
        backend = FakeStoreBackend()
        assert isinstance(backend, StoreBackend)

    def test_failing_store_satisfies_protocol(self) -> None:
        backend = FailingStoreBackend()
        assert isinstance(backend, StoreBackend)


# ------------------------------------------------------------------ #
#  Round-trip via store backend
# ------------------------------------------------------------------ #


class TestStoreBackendRoundTrip:
    def test_round_trip_via_fake_store(self, tmp_path: Path) -> None:
        """Data saved via store backend can be loaded back."""
        store = FakeStoreBackend()
        cache = EvolutionStoreCache(store=store)
        dest = tmp_path / "rt.pkl"

        cache.save({"hello": "world"}, dest)
        # Clear memory cache to force store lookup
        cache._memory_cache.clear()

        result = cache.load(dest)
        assert result == {"hello": "world"}

    def test_numpy_array_round_trip_via_store(self, tmp_path: Path) -> None:
        """Numpy arrays survive store-backed round-trip."""
        store = FakeStoreBackend()
        cache = EvolutionStoreCache(store=store)
        dest = tmp_path / "arr.pkl"

        arr = np.array([1.0, 2.0, 3.0])
        cache.save(arr, dest)
        cache._memory_cache.clear()

        loaded = cache.load(dest)
        np.testing.assert_array_equal(loaded, arr)


# ------------------------------------------------------------------ #
#  Fallback behaviour
# ------------------------------------------------------------------ #


class TestFallbackBehaviour:
    def test_store_none_falls_back_to_file(self, tmp_path: Path) -> None:
        """When store=None, save/load use file persistence only."""
        cache = EvolutionStoreCache(store=None)
        dest = tmp_path / "file_only.pkl"

        cache.save(42, dest)
        cache._memory_cache.clear()

        result = cache.load(dest)
        assert result == 42

    def test_store_get_returns_none_falls_back_to_file(self, tmp_path: Path) -> None:
        """When store.get() returns None, load falls back to file."""
        store = FakeStoreBackend()
        cache = EvolutionStoreCache(store=store)
        dest = tmp_path / "fallback.pkl"

        # Save data (writes to store AND file)
        cache.save({"key": "value"}, dest)
        cache._memory_cache.clear()
        # Remove from store so get() returns None
        store._data.clear()

        result = cache.load(dest)
        assert result == {"key": "value"}


# ------------------------------------------------------------------ #
#  Error resilience
# ------------------------------------------------------------------ #


class TestErrorResilience:
    def test_store_put_raises_logs_warning_and_continues(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """store.put() failure is logged; file write still succeeds."""
        store = FailingStoreBackend()
        cache = EvolutionStoreCache(store=store)
        dest = tmp_path / "put_fail.pkl"

        with caplog.at_level(
            logging.WARNING, logger="liq.evolution.adapters.store_cache"
        ):
            cache.save(99, dest)

        assert "store.put failed" in caplog.text
        # File should still be written
        cache._memory_cache.clear()
        cache_no_store = EvolutionStoreCache(store=None)
        assert cache_no_store.load(dest) == 99

    def test_store_get_raises_logs_warning_and_falls_back(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """store.get() failure is logged; load falls back to file."""
        # First, save with no store so file exists
        cache_setup = EvolutionStoreCache(store=None)
        dest = tmp_path / "get_fail.pkl"
        cache_setup.save(77, dest)

        # Now load with a failing store
        store = FailingStoreBackend()
        cache = EvolutionStoreCache(store=store)

        with caplog.at_level(
            logging.WARNING, logger="liq.evolution.adapters.store_cache"
        ):
            result = cache.load(dest)

        assert result == 77
        assert "store.get failed" in caplog.text


# ------------------------------------------------------------------ #
#  Memory cache priority
# ------------------------------------------------------------------ #


class TestMemoryCachePriority:
    def test_memory_cache_checked_before_store(self, tmp_path: Path) -> None:
        """Memory cache hit avoids store.get() call entirely."""
        store = FakeStoreBackend()
        cache = EvolutionStoreCache(store=store)
        dest = tmp_path / "mem_first.pkl"

        cache.save({"cached": True}, dest)
        # Memory cache has the value; mutate store to prove it's not read
        store._data.clear()

        result = cache.load(dest)
        assert result == {"cached": True}


# ------------------------------------------------------------------ #
#  indicator_key static method
# ------------------------------------------------------------------ #


class TestIndicatorKey:
    def test_indicator_key_format(self) -> None:
        key = EvolutionStoreCache.indicator_key("AAPL", "sma", "abc123")
        assert key == "AAPL/indicators/sma/abc123"

    def test_indicator_key_different_params(self) -> None:
        key = EvolutionStoreCache.indicator_key("BTC-USD", "ema", "xyz")
        assert key == "BTC-USD/indicators/ema/xyz"
