"""Edge-case tests to increase coverage for eval_cache.py."""

from __future__ import annotations

import pytest

from liq.evolution.fitness.eval_cache import (
    FitnessEvaluationCache,
    compute_fingerprint,
    compute_program_hash,
)
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


class TestFitnessEvaluationCacheProperties:
    """Cover property accessors and utility methods."""

    def test_max_entries_property(self) -> None:
        cache = FitnessEvaluationCache(max_entries=42)
        assert cache.max_entries == 42

    def test_entries_property_empty(self) -> None:
        cache = FitnessEvaluationCache(max_entries=10)
        assert cache.entries == 0

    def test_hits_property_initial(self) -> None:
        cache = FitnessEvaluationCache(max_entries=10)
        assert cache.hits == 0

    def test_misses_property_initial(self) -> None:
        cache = FitnessEvaluationCache(max_entries=10)
        assert cache.misses == 0

    def test_clear_resets_all(self) -> None:
        cache = FitnessEvaluationCache(max_entries=10)
        cache.put("hash1", "slice1", "fp1", {"data": 1})
        cache.get("hash1", "slice1", "fp1")
        cache.get("hash2", "slice2", "fp2")

        assert cache.entries > 0
        assert cache.hits > 0
        assert cache.misses > 0

        cache.clear()

        assert cache.entries == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_stats_returns_dict(self) -> None:
        cache = FitnessEvaluationCache(max_entries=5)
        cache.put("h", "s", "f", "payload")
        cache.get("h", "s", "f")
        cache.get("miss", "s", "f")

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert stats["max_entries"] == 5


class TestFitnessEvaluationCacheConstruction:
    """Cover construction validation."""

    def test_invalid_max_entries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            FitnessEvaluationCache(max_entries=0)

    def test_negative_max_entries_raises(self) -> None:
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            FitnessEvaluationCache(max_entries=-1)


class TestFitnessEvaluationCacheNoneSliceId:
    """Cover None slice_id paths in get/put."""

    def test_get_with_none_slice_id_returns_none(self) -> None:
        cache = FitnessEvaluationCache(max_entries=10)
        result = cache.get("hash", None, "fp")
        assert result is None

    def test_put_with_none_slice_id_is_noop(self) -> None:
        cache = FitnessEvaluationCache(max_entries=10)
        cache.put("hash", None, "fp", {"data": 1})
        assert cache.entries == 0


class TestFitnessEvaluationCacheOverwrite:
    """Cover overwrite (existing key) path in put."""

    def test_put_overwrites_existing_key(self) -> None:
        cache = FitnessEvaluationCache(max_entries=10)
        cache.put("h", "s", "f", "first")
        cache.put("h", "s", "f", "second")

        result = cache.get("h", "s", "f")
        assert result == "second"
        assert cache.entries == 1


class TestHelperFunctions:
    """Cover compute_program_hash and compute_fingerprint."""

    def test_compute_program_hash_is_deterministic(self) -> None:
        program = TerminalNode("close", Series)
        h1 = compute_program_hash(program)
        h2 = compute_program_hash(program)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_compute_fingerprint_is_deterministic(self) -> None:
        fp1 = compute_fingerprint("1.0", {"a": 1})
        fp2 = compute_fingerprint("1.0", {"a": 1})
        assert fp1 == fp2
        assert len(fp1) == 64

    def test_compute_fingerprint_changes_with_version(self) -> None:
        fp1 = compute_fingerprint("1.0", {"a": 1})
        fp2 = compute_fingerprint("2.0", {"a": 1})
        assert fp1 != fp2

    def test_compute_fingerprint_changes_with_config(self) -> None:
        fp1 = compute_fingerprint("1.0", {"a": 1})
        fp2 = compute_fingerprint("1.0", {"a": 2})
        assert fp1 != fp2
