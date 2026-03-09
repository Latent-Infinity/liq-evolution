"""Protocols (extension points) for liq-evolution.

Protocols define structural interfaces that consumers implement.
They use ``typing.Protocol`` (not ABCs) so consumers never need to
inherit from liq-evolution classes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from liq.core.security import (
    SENSITIVE_CONTEXT_KEYS as _SENSITIVE_CONTEXT_KEYS,
)
from liq.core.security import (
    mask_sensitive_context as _mask_sensitive_context,
)
from liq.evolution.errors import (
    EvaluationContractError,
    ProtocolVersionError,
)

if TYPE_CHECKING:
    import numpy as np
    import polars as pl


EVOLUTION_PROTOCOL_VERSION = "1.0"
GP_PROTOCOL_VERSION = "1.0"

SENSITIVE_CONTEXT_KEYS = tuple(_SENSITIVE_CONTEXT_KEYS)


def mask_sensitive_context(context: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a redacted copy of context for deterministic diagnostics."""
    return _mask_sensitive_context(context)


def require_protocol_version(
    protocol: str,
    actual: str,
    expected: str,
) -> None:
    """Reject mismatched protocol versions with deterministic diagnostics."""
    if actual != expected:
        raise ProtocolVersionError(
            f"{protocol} protocol version mismatch: actual={actual!r}, expected={expected!r}"
        )


def translate_protocol_exception(
    exc: Exception,
    *,
    boundary: str,
    context: Mapping[str, Any] | None = None,
) -> EvaluationContractError:
    """Wrap a contract boundary exception with context for replayability."""
    redacted = mask_sensitive_context(context)
    payload = f" ({redacted!r})" if redacted else ""
    msg = f"{boundary} protocol contract failure: {exc}{payload}"
    return EvaluationContractError(msg)


@runtime_checkable
class CandidateArtifact(Protocol):
    """Cross-boundary representation of an evolvable candidate."""

    candidate_id: str
    payload: Mapping[str, Any]


@runtime_checkable
class StrategyArtifact(Protocol):
    """Cross-boundary representation of an evolved strategy artifact."""

    strategy_id: str
    candidate: CandidateArtifact
    model_dump: Callable[..., Mapping[str, Any]]


@runtime_checkable
class CandidateEvaluator(Protocol):
    """Evaluator protocol used on the GP-evaluator boundary."""

    protocol_version: str

    def evaluate_candidate(
        self,
        candidate: CandidateArtifact,
        context: Any,
    ) -> Any: ...


@runtime_checkable
class IndicatorBackend(Protocol):
    """Backend for computing technical indicators.

    Implement this protocol to provide indicator values from any
    technical analysis library (e.g. liq-ta).

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

        ``liq.gp`` currently exposes ``get``; the stage-0 contract uses
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

    The stage-0 contract uses ``evaluate`` while current package tests
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


@runtime_checkable
class EvolutionArtifactStore(Protocol):
    """Persistence abstraction for artifacts crossing stage boundaries."""

    protocol_version: str

    def save_artifact(
        self,
        artifact_id: str,
        artifact: CandidateArtifact | StrategyArtifact,
    ) -> None: ...

    def load_artifact(
        self,
        artifact_id: str,
    ) -> CandidateArtifact | StrategyArtifact | None: ...


@runtime_checkable
class StoreBackend(Protocol):
    """Protocol for pluggable storage backends (e.g. liq-store).

    Consumers implement this to provide key-value persistence without
    requiring liq-evolution to depend on any specific storage library.
    """

    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, data: bytes) -> None: ...
