"""Tests for protocol contracts (runtime_checkable)."""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.evolution.protocols import (
    FitnessEvaluator,
    FitnessStageEvaluator,
    GPStrategy,
    IndicatorBackend,
    PrimitiveRegistry,
)


class TestIndicatorBackendProtocol:
    """Verify IndicatorBackend is runtime-checkable."""

    def test_conforming_class_accepted(self) -> None:
        class MyBackend:
            def compute(
                self,
                name: str,
                params: dict[str, Any],
                data: Any,
                **kwargs: Any,
            ) -> np.ndarray:
                return np.array([])

            def list_indicators(self, category: str | None = None) -> list[str]:
                return []

        assert isinstance(MyBackend(), IndicatorBackend)

    def test_non_conforming_rejected(self) -> None:
        class NotABackend:
            pass

        assert not isinstance(NotABackend(), IndicatorBackend)


class TestGPStrategyProtocol:
    """Verify GPStrategy is runtime-checkable."""

    def test_conforming_class_accepted(self) -> None:
        class MyStrategy:
            def fit(self, features: Any, labels: Any) -> None:
                pass

            def predict(self, features: Any) -> Any:
                return None

        assert isinstance(MyStrategy(), GPStrategy)

    def test_non_conforming_rejected(self) -> None:
        class NotAStrategy:
            pass

        assert not isinstance(NotAStrategy(), GPStrategy)


class TestFitnessStageEvaluatorProtocol:
    """Verify FitnessStageEvaluator is runtime-checkable."""

    def test_conforming_class_accepted(self) -> None:
        class MyEvaluator:
            def evaluate_fitness(self, programs: list[Any], context: Any) -> list[Any]:
                return []

            def evaluate(self, programs: list[Any], context: Any) -> list[Any]:
                return self.evaluate_fitness(programs, context)

        assert isinstance(MyEvaluator(), FitnessStageEvaluator)

    def test_non_conforming_rejected(self) -> None:
        class NotAnEvaluator:
            pass

        assert not isinstance(NotAnEvaluator(), FitnessStageEvaluator)


class TestFitnessEvaluatorProtocol:
    """Verify the phase-0 FitnessEvaluator protocol is available."""

    def test_conforming_class_accepted(self) -> None:
        class MyEvaluator:
            def evaluate(self, programs: list[Any], context: Any) -> list[float]:
                return [0.0 for _ in programs]

        assert isinstance(MyEvaluator(), FitnessEvaluator)


class TestPrimitiveRegistryProtocol:
    """Verify PrimitiveRegistry protocol access names."""

    def test_conforming_registry_accepted(self) -> None:
        class MyRegistry:
            def register(
                self,
                name: str,
                callable: Any,
                *,
                category: str = "default",
                arity: int | None = None,
                input_types: tuple[Any, ...] = (),
                output_type: Any | None = None,
                param_specs: list[dict[str, Any]] | None = None,
            ) -> None:
                pass

            def get(self, name: str, *, category: str = "default") -> Any:
                return None

            def lookup(self, name: str) -> Any:
                return self.get(name)

            def list_primitives(self, category: str | None = None) -> list[Any]:
                return []

        assert isinstance(MyRegistry(), PrimitiveRegistry)

    def test_lookup_default_delegates_to_get(self) -> None:
        """Exercise PrimitiveRegistry.lookup default body (line 83)."""

        class MinimalRegistry:
            def register(self, name: str, callable: Any, **kw: Any) -> None:
                pass

            def get(self, name: str, **kw: Any) -> Any:
                return f"found:{name}"

            def list_primitives(self, category: str | None = None) -> list[Any]:
                return []

        reg = MinimalRegistry()
        # Call through the protocol default
        assert PrimitiveRegistry.lookup(reg, "test") == "found:test"


class TestFitnessStageEvaluatorDefault:
    """Verify FitnessStageEvaluator.evaluate default body (line 149)."""

    def test_evaluate_default_delegates_to_evaluate_fitness(self) -> None:
        class MinimalEvaluator:
            def evaluate_fitness(
                self,
                programs: list[Any],
                context: Any,
            ) -> list[Any]:
                return [1.0 for _ in programs]

        evaluator = MinimalEvaluator()
        result = FitnessStageEvaluator.evaluate(evaluator, ["p1", "p2"], {})
        assert result == [1.0, 1.0]
