"""Caching primitives for expensive fitness evaluation calls."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from liq.gp.program.ast import Program
from liq.gp.program.serialize import serialize


def _canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def compute_program_hash(program: Program) -> str:
    """Compute a stable hash of a GP program structure and constants."""
    payload = serialize(program)
    digest = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(digest).hexdigest()


def compute_fingerprint(evaluator_version: str, config: Any) -> str:
    """Compute a deterministic fingerprint for evaluator version + config."""
    payload = {
        "version": evaluator_version,
        "config": config,
    }
    digest = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(digest).hexdigest()


@dataclass(frozen=True)
class _CacheStats:
    hits: int
    misses: int
    entries: int
    max_entries: int


CacheKey = tuple[str, str, str]


class FitnessEvaluationCache:
    """Simple bounded LRU cache for strategy evaluation payloads."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[CacheKey, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def entries(self) -> int:
        return len(self._entries)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def clear(self) -> None:
        """Reset cache contents and counters."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    def make_key(
        self,
        strategy_hash: str,
        slice_id: str | None,
        evaluator_fingerprint: str,
    ) -> CacheKey | None:
        """Build a cache key if slice_id is available."""
        if not slice_id:
            return None
        return (strategy_hash, slice_id, evaluator_fingerprint)

    def get(
        self,
        strategy_hash: str,
        slice_id: str | None,
        evaluator_fingerprint: str,
    ) -> Any | None:
        """Return cached payload or ``None`` when unavailable."""
        key = self.make_key(strategy_hash, slice_id, evaluator_fingerprint)
        if key is None:
            return None
        if key not in self._entries:
            self._misses += 1
            return None

        self._hits += 1
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(
        self,
        strategy_hash: str,
        slice_id: str | None,
        evaluator_fingerprint: str,
        payload: Any,
    ) -> None:
        """Store payload in LRU cache."""
        key = self.make_key(strategy_hash, slice_id, evaluator_fingerprint)
        if key is None:
            return

        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = payload

        if len(self._entries) <= self._max_entries:
            return

        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        """Return cache statistics snapshot."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": self.entries,
            "max_entries": self._max_entries,
        }
