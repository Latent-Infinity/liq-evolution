"""Tests for protocol contracts (runtime_checkable)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from liq.evolution.protocols import (
    FitnessEvaluator,
    FitnessStageEvaluator,
    GPStrategy,
    IndicatorBackend,
    CandidateArtifact,
    CandidateEvaluator,
    EvolutionArtifactStore,
    GP_PROTOCOL_VERSION,
    StrategyArtifact,
    EVOLUTION_PROTOCOL_VERSION,
    require_protocol_version,
    translate_protocol_exception,
    mask_sensitive_context,
    PrimitiveRegistry,
)
from liq.evolution.adapters.artifact_store import LiqStoreEvolutionArtifactStore
from liq.evolution.errors import (
    ProtocolVersionError,
    EvaluationContractError,
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
    """Verify the stage-0 FitnessEvaluator protocol is available."""

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


class TestCandidateProtocolContracts:
    """Verify candidate/strategy artifact contracts are concrete and stable."""

    def test_candidate_artifact_protocol(self) -> None:
        class Candidate:
            def __init__(self) -> None:
                self.candidate_id = "cand-1"
                self.payload = {"program": "x"}

        assert isinstance(Candidate(), CandidateArtifact)

    def test_strategy_artifact_protocol(self) -> None:
        class Strategy:
            protocol_version = "x"

            def __init__(self) -> None:
                self.strategy_id = "strat-1"
                self.candidate = type(
                    "Candidate",
                    (),
                    {"candidate_id": "cand-1", "payload": {}},
                )()

            def model_dump(self) -> dict[str, object]:
                return {}

        assert isinstance(Strategy(), StrategyArtifact)

    def test_candidate_evaluator_protocol(self) -> None:
        class Evaluator:
            protocol_version = "1.0"

            def evaluate_candidate(self, candidate: CandidateArtifact, context: dict[str, object]) -> bool:
                return True

        evaluator = Evaluator()
        assert isinstance(evaluator, CandidateEvaluator)
        require_protocol_version("candidate", evaluator.protocol_version, GP_PROTOCOL_VERSION)

    def test_evolution_artifact_store_protocol(self) -> None:
        class Store:
            protocol_version = "1.0"

            def save_artifact(self, artifact_id: str, artifact: object) -> None:
                self.last = artifact_id

            def load_artifact(self, artifact_id: str) -> object | None:
                return None

        assert isinstance(Store(), EvolutionArtifactStore)

    def test_liq_store_artifact_bridge_satisfies_protocol(self) -> None:
        class Backend:
            def __init__(self) -> None:
                self.payloads: dict[str, bytes] = {}

            def get(self, key: str) -> bytes | None:
                return self.payloads.get(key)

            def put(self, key: str, data: bytes) -> None:
                self.payloads[key] = data

        bridge = LiqStoreEvolutionArtifactStore(backend=Backend())
        assert isinstance(bridge, EvolutionArtifactStore)


class TestProtocolHelpers:
    """Helpers for protocol translation and validation."""

    def test_mask_sensitive_context_none(self) -> None:
        assert mask_sensitive_context(None) is None

    def test_mask_sensitive_context(self) -> None:
        source = {
            "api_key": "top-secret",
            "model": "xgboost",
        }
        redacted = mask_sensitive_context(source)
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["model"] == "xgboost"

    def test_require_protocol_version_mismatch(self) -> None:
        with pytest.raises(ProtocolVersionError, match="protocol version mismatch"):
            require_protocol_version("test", "0.9", "1.0")

    def test_require_protocol_version_matches(self) -> None:
        require_protocol_version("test", EVOLUTION_PROTOCOL_VERSION, "1.0")

    def test_translate_protocol_exception_masks_context(self) -> None:
        err = translate_protocol_exception(
            RuntimeError("boom"),
            boundary="candidate-evaluator",
            context={"api_key": "secret", "run_id": "abc"},
        )
        assert isinstance(err, EvaluationContractError)
        assert "api_key" in str(err)
        assert "secret" not in str(err)
        assert "run_id" in str(err)


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
