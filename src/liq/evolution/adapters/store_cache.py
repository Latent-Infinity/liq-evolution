"""Persistent cache for evolution outputs and strategy specs.

The cache is intentionally lightweight:
- in-memory map for hot-path reuse
- optional file-backed persistence for stable warm-start workflows
- optional pluggable storage backend via :class:`StoreBackend` protocol
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liq.evolution.errors import AdapterError

if TYPE_CHECKING:
    from liq.evolution.protocols import StoreBackend

logger = logging.getLogger(__name__)


class EvolutionStoreCache:
    """Caches and persists evolution artifacts.

    The current implementation stores payloads as pickled bytes, which is
    portable enough for deterministic strategy specs and evaluation metadata used by
    this project. The API accepts an arbitrary ``path`` and returns the same object
    graph on load.

    When a :class:`~liq.evolution.protocols.StoreBackend` is provided, payloads
    are additionally written to / read from the backend.  Store failures are
    logged but never prevent the file-based fallback from working.
    """

    def __init__(self, store: StoreBackend | None = None) -> None:
        self._memory_cache: dict[str, Any] = {}
        self._store = store

    def save(self, data: Any, path: Any) -> None:
        """Persist ``data`` to ``path``.

        Args:
            data: Arbitrary serializable evolution result/object.
            path: Destination path (str or ``Path``).
        """

        if isinstance(path, Path):
            key = str(path)
        elif isinstance(path, str):
            key = path
        else:
            raise AdapterError("path must be a string or pathlib.Path")

        payload = pickle.dumps(data)

        # Memory cache
        self._memory_cache[key] = data

        # Store backend (best-effort)
        if self._store is not None:
            try:
                self._store.put(key, payload)
            except Exception:
                logger.warning("store.put failed for key=%s", key, exc_info=True)

        # File persistence
        Path(key).parent.mkdir(parents=True, exist_ok=True)
        with open(key, "wb") as handle:
            handle.write(payload)

    def load(self, path: Any) -> Any:
        """Load evolution payload previously saved with :meth:`save`.

        Args:
            path: Source path (str or ``Path``).

        Returns:
            The persisted object.
        """

        if isinstance(path, Path):
            key = str(path)
        elif isinstance(path, str):
            key = path
        else:
            raise AdapterError("path must be a string or pathlib.Path")

        # 1. Memory cache (fastest)
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached

        # 2. Store backend (best-effort)
        if self._store is not None:
            try:
                raw = self._store.get(key)
            except Exception:
                logger.warning("store.get failed for key=%s", key, exc_info=True)
                raw = None
            if raw is not None:
                obj = pickle.loads(raw)
                self._memory_cache[key] = obj
                return obj

        # 3. File fallback
        with open(key, "rb") as handle:
            payload = handle.read()
        obj = pickle.loads(payload)
        self._memory_cache[key] = obj
        return obj

    @staticmethod
    def indicator_key(symbol: str, indicator: str, params_id: str) -> str:
        """Build a canonical cache key for indicator data.

        Returns:
            Key matching ``liq-store`` convention for indicator artifacts.
        """
        try:
            from liq.store import key_builder

            return key_builder.indicators(symbol, indicator, params_id)
        except Exception:
            return f"{symbol}/indicators/{indicator}/{params_id}"
