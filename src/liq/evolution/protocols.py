"""Protocols (extension points) for liq-evolution.

Protocols define structural interfaces that consumers implement.
They use ``typing.Protocol`` (not ABCs) so consumers never need to
inherit from liq-evolution classes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import polars as pl


@runtime_checkable
class IndicatorBackend(Protocol):
    """Backend for computing technical indicators.

    Implement this protocol to provide indicator values from any
    technical analysis library (e.g. liq-ta, ta-lib).

    Example::

        class MyBackend:
            def compute(self, name, params, data):
                return np.array(...)

            def list_indicators(self, category=None):
                return ["sma", "ema", "rsi"]
    """

    def compute(
        self,
        name: str,
        params: dict[str, Any],
        data: Any,
        **kwargs: Any,
    ) -> np.ndarray: ...

    def list_indicators(
        self,
        category: str | None = None,
    ) -> list[str]: ...


@runtime_checkable
class PrimitiveRegistry(Protocol):
    """Protocol for GP primitive registries.

    Adapters should provide a registry with ``register`` and
    ``lookup``/``get`` style access for primitive resolution.
    """

    def register(
        self,
        name: str,
        callable: Callable[..., Any],
        *,
        category: str = "default",
        arity: int | None = None,
        input_types: tuple[Any, ...] = (),
        output_type: Any | None = None,
        param_specs: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def get(
        self,
        name: str,
        *,
        category: str = "default",
    ) -> Any: ...

    def lookup(self, name: str) -> Any:
        """Compatibility shim for protocol consumers.

        ``liq.gp`` currently exposes ``get``; the phase-0 contract uses
        ``lookup``. Protocol default keeps both names available.
        """

        return self.get(name)

    def list_primitives(
        self,
        category: str | None = None,
    ) -> list[Any]: ...


@runtime_checkable
class GPStrategy(Protocol):
    """Interface for GP-evolved trading strategies.

    Bridges GP programs to the liq-runner strategy execution layer.

    Example::

        class MyStrategy:
            def fit(self, features, labels):
                ...

            def predict(self, features):
                return ...
    """

    def fit(
        self,
        features: pl.DataFrame,
        labels: pl.Series | None,
    ) -> None: ...

    def predict(
        self,
        features: pl.DataFrame,
    ) -> Any: ...


@runtime_checkable
class FitnessEvaluator(Protocol):
    """Evaluator for fitness scoring.

    The phase-0 contract uses ``evaluate`` while current package tests
    assert ``evaluate_fitness``. Both are provided for compatibility.
    """

    def evaluate(
        self,
        programs: list[Any],
        context: Any,
    ) -> list[Any]: ...


@runtime_checkable
class FitnessStageEvaluator(FitnessEvaluator, Protocol):
    """Backward-compatible stage-oriented evaluator protocol."""

    def evaluate_fitness(
        self,
        programs: list[Any],
        context: Any,
    ) -> list[Any]: ...

    def evaluate(
        self,
        programs: list[Any],
        context: Any,
    ) -> list[Any]:
        return self.evaluate_fitness(programs, context)
