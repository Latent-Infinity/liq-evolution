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
        self._cache: dict[
            tuple[str, tuple[tuple[str, Any], ...], tuple[tuple[str, int, tuple[int, ...]], ...]],
            np.ndarray,
        ] = {}

    @staticmethod
    def _data_signature(data: dict[str, np.ndarray]) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
        signature: list[tuple[str, int, tuple[int, ...]]] = []
        for name, value in sorted(data.items()):
            arr = np.asarray(value)
            ptr = int(arr.__array_interface__["data"][0])
            shape = tuple(int(x) for x in arr.shape)
            signature.append((name, ptr, shape))
        return tuple(signature)

    @staticmethod
    def _hashable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return ("ndarray", value.shape, value.dtype.str)
        if isinstance(value, dict):
            return tuple(
                (key, FeatureContext._hashable(val))
                for key, val in sorted(value.items())
            )
        if isinstance(value, tuple):
            return ("tuple", tuple(FeatureContext._hashable(item) for item in value))
        if isinstance(value, list):
            normalized = [FeatureContext._hashable(item) for item in value]
            return ("list", tuple(sorted(normalized, key=repr)))
        try:
            hash(value)
        except Exception:
            return repr(value)
        return value

    @staticmethod
    def _params_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (key, FeatureContext._hashable(value))
            for key, value in sorted(params.items())
        )

    def compute(
        self,
        name: str,
        params: dict[str, Any],
        data: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        key = (name, self._params_key(params), self._data_signature(data))
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
            params_key = self._params_key(params)
            self._cache = {
                k: v
                for k, v in self._cache.items()
                if not (k[0] == name and k[1] == params_key)
            }
        else:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != name}
