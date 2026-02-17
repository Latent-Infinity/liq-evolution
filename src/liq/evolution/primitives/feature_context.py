"""In-memory indicator caching layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from liq.evolution.protocols import IndicatorBackend


class FeatureContext:
    """Caching layer over any IndicatorBackend.

    Cache key: (indicator_name, tuple(sorted(params.items())))
    Cache value: np.ndarray (identity preserved on hit)
    """

    def __init__(self, backend: IndicatorBackend) -> None:
        self._backend = backend
        self._cache: dict[tuple[str, tuple[tuple[str, Any], ...]], np.ndarray] = {}

    def compute(
        self,
        name: str,
        params: dict[str, Any],
        data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        key = (name, tuple(sorted(params.items())))
        if key in self._cache:
            return self._cache[key]
        result = self._backend.compute(name, params, data, **kwargs)
        self._cache[key] = result
        return result

    def list_indicators(self, category: str | None = None) -> list[str]:
        return self._backend.list_indicators(category)

    def invalidate(
        self,
        name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if name is None:
            self._cache.clear()
            return
        if params is not None:
            key = (name, tuple(sorted(params.items())))
            self._cache.pop(key, None)
        else:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != name}
