"""Edge-case tests to increase coverage for store_cache.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from liq.evolution.adapters.store_cache import EvolutionStoreCache
from liq.evolution.errors import AdapterError


class TestSavePathTypes:
    """Cover path coercion in save()."""

    def test_save_with_string_path(self, tmp_path: Path) -> None:
        cache = EvolutionStoreCache()
        dest = str(tmp_path / "str_save.pkl")
        cache.save({"data": 1}, dest)
        assert cache._memory_cache[dest] == {"data": 1}

    def test_save_with_path_object(self, tmp_path: Path) -> None:
        cache = EvolutionStoreCache()
        dest = tmp_path / "path_save.pkl"
        cache.save({"data": 2}, dest)
        assert cache._memory_cache[str(dest)] == {"data": 2}

    def test_save_with_invalid_path_type_raises(self) -> None:
        cache = EvolutionStoreCache()
        with pytest.raises(AdapterError, match="path must be a string or pathlib.Path"):
            cache.save({"data": 3}, 12345)


class TestLoadPathTypes:
    """Cover path coercion in load()."""

    def test_load_with_string_path(self, tmp_path: Path) -> None:
        cache = EvolutionStoreCache()
        dest = str(tmp_path / "str_load.pkl")
        cache.save(42, dest)
        cache._memory_cache.clear()
        result = cache.load(dest)
        assert result == 42

    def test_load_with_path_object(self, tmp_path: Path) -> None:
        cache = EvolutionStoreCache()
        dest = tmp_path / "path_load.pkl"
        cache.save(42, dest)
        cache._memory_cache.clear()
        result = cache.load(dest)
        assert result == 42

    def test_load_with_invalid_path_type_raises(self) -> None:
        cache = EvolutionStoreCache()
        with pytest.raises(AdapterError, match="path must be a string or pathlib.Path"):
            cache.load(12345)


class TestIndicatorKeyFallback:
    """Cover the indicator_key fallback when liq.store is not available."""

    def test_indicator_key_returns_canonical_format(self) -> None:
        key = EvolutionStoreCache.indicator_key("AAPL", "rsi", "params_v1")
        # Should return canonical format (either from liq.store or fallback)
        assert "AAPL" in key
        assert "rsi" in key
        assert "params_v1" in key
