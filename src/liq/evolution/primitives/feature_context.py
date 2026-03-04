"""In-memory indicator caching layer."""

from __future__ import annotations

from collections.abc import Hashable
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
        self._cache: dict[
            tuple[
                str,
                tuple[tuple[str, Any], ...],
                tuple[tuple[str, Any], ...],
                tuple[tuple[str, Any], ...],
            ],
            np.ndarray,
        ] = {}

    @staticmethod
    def _hashable(value: Any) -> Any:
        """Convert cache values into deterministic, hashable forms."""
        if isinstance(value, Hashable):
            return value
        if isinstance(value, np.ndarray):
            return (
                "ndarray",
                value.shape,
                value.dtype.str,
                str(value.ravel()[:16].tolist()),
            )
        if isinstance(value, dict):
            return tuple(sorted((str(k), FeatureContext._hashable(v)) for k, v in value.items()))
        if isinstance(value, list):
            hashed = [FeatureContext._hashable(v) for v in value]
            try:
                return ("list", tuple(sorted(hashed)))
            except TypeError:
                return ("list", tuple(hashed))
        if isinstance(value, tuple):
            hashed = tuple(FeatureContext._hashable(v) for v in value)
            try:
                return ("tuple", tuple(sorted(hashed)))
            except TypeError:
                return ("tuple", hashed)
        return repr(value)

    def compute(
        self,
        name: str,
        params: dict[str, Any],
        data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        cache_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"out", "ts", "data", "cache"}
        }
        data_signature = tuple(
            sorted((k, self._hashable(v)) for k, v in data.items())
        )
        key = (
            name,
            tuple(sorted((k, self._hashable(v)) for k, v in params.items())),
            data_signature,
            tuple(sorted((k, self._hashable(v)) for k, v in cache_kwargs.items())),
        )
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
            params_key = tuple(sorted((k, self._hashable(v)) for k, v in params.items()))
            self._cache = {
                key: value
                for key, value in self._cache.items()
                if not (key[0] == name and key[1] == params_key)
            }
        else:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != name}
