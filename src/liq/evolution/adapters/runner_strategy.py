"""GP strategy adapter for liq-runner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from liq.evolution.adapters.parallel_eval import ParallelEvaluator
from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.config import ParallelConfig
from liq.evolution.errors import AdapterError, SerializationError
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.evolution.init import validate_seed_programs
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate as gp_evaluate
from liq.gp.program.serialize import deserialize, serialize

if TYPE_CHECKING:
    from liq.gp.config import GPConfig
    from liq.gp.primitives.registry import PrimitiveRegistry
    from liq.gp.types import EvolutionResult


def _dataframe_to_context(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Convert a polars DataFrame to a GP evaluation context dict."""
    return {col: df[col].to_numpy() for col in df.columns}


class GPStrategyAdapter:
    """Adapts GP-evolved programs into liq-runner strategies.

    Implements the :class:`~liq.evolution.protocols.GPStrategy` protocol
    by wrapping a GP program tree as a trading strategy.

    When ``evaluator`` is ``None`` the adapter operates in predict-only
    mode (e.g. after :meth:`from_spec`).  Calling :meth:`fit` on a
    predict-only adapter raises :class:`AdapterError`.
    """

    _SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        registry: PrimitiveRegistry,
        gp_config: GPConfig,
        evaluator: object | None = None,
        *,
        seed_programs: Sequence[Program] | None = None,
        warm_start: bool = False,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self._registry = registry
        self._gp_config = gp_config
        self._evaluator = evaluator
        self._seed_programs = seed_programs
        self._warm_start = warm_start
        self._parallel_config = parallel_config
        self._program: Program | None = None
        self._evolution_result: EvolutionResult | None = None

        # Validate seeds at construction (fail-fast)
        if seed_programs is not None:
            validate_seed_programs(
                seed_programs,
                gp_config,
                registry=registry,
            )

    # -------------------------------------------------------------- #
    #  Strategy protocol
    # -------------------------------------------------------------- #

    def fit(
        self,
        features: pl.DataFrame,
        labels: pl.Series | None = None,
    ) -> None:
        """Fit the strategy by running GP evolution.

        Args:
            features: Feature DataFrame (columns become context arrays).
            labels: Optional label Series added to context as ``"labels"``.

        Raises:
            AdapterError: If the adapter is predict-only (no evaluator).
        """
        if self._evaluator is None:
            raise AdapterError("Cannot fit a predict-only adapter (evaluator is None)")

        context = _dataframe_to_context(features)
        if labels is not None:
            context["labels"] = labels.to_numpy()

        evaluator = self._evaluator
        if self._parallel_config is not None:
            evaluator = ParallelEvaluator(
                evaluator=evaluator,
                backend=self._parallel_config.backend,
                max_workers=self._parallel_config.max_workers,
                max_in_flight=self._parallel_config.max_in_flight,
                max_tasks_per_worker=self._parallel_config.max_tasks_per_worker,
                memory_limit_mb=self._parallel_config.memory_limit_mb,
                memory_warn_threshold_mb=self._parallel_config.memory_warn_threshold_mb,
                auto_fallback=self._parallel_config.auto_fallback,
            )

        result = evolve(
            self._registry,
            self._gp_config,
            evaluator,
            context,
            seed_programs=self._seed_programs,
        )
        self._program = result.best_program
        self._evolution_result = result

        if self._warm_start:
            self._seed_programs = [result.best_program]

    def predict(self, features: pl.DataFrame) -> GPSignalOutput:
        """Generate predictions from features.

        Args:
            features: Feature DataFrame.

        Returns:
            :class:`GPSignalOutput` with prediction scores.

        Raises:
            AdapterError: If :meth:`fit` has not been called.
        """
        if self._program is None:
            raise AdapterError("predict() called before fit()")

        context = _dataframe_to_context(features)
        scores_array = gp_evaluate(self._program, context)
        return GPSignalOutput(scores=pl.Series("scores", scores_array))

    # -------------------------------------------------------------- #
    #  Properties
    # -------------------------------------------------------------- #

    @property
    def program(self) -> Program | None:
        """The best evolved program, or ``None`` if not yet fitted."""
        return self._program

    @property
    def evolution_result(self) -> EvolutionResult | None:
        """Full evolution result, or ``None`` if not yet fitted."""
        return self._evolution_result

    # -------------------------------------------------------------- #
    #  Spec export / import
    # -------------------------------------------------------------- #

    def export_spec(self) -> dict[str, Any]:
        """Export fitted program + config as a serializable spec dict.

        Returns:
            Dict with ``schema_version``, ``program``, and ``config``.

        Raises:
            AdapterError: If no program has been fitted yet.
        """
        if self._program is None:
            raise AdapterError("export_spec() called before fit()")

        return {
            "schema_version": self._SCHEMA_VERSION,
            "program": serialize(self._program),
            "config": self._gp_config.model_dump(),
        }

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        registry: PrimitiveRegistry,
    ) -> GPStrategyAdapter:
        """Create a predict-only adapter from a serialized spec.

        Args:
            spec: Dict from :meth:`export_spec`.
            registry: Registry for deserializing primitives.

        Returns:
            A predict-only :class:`GPStrategyAdapter` with the deserialized
            program loaded.

        Raises:
            AdapterError: If spec is malformed (missing required keys).
            SerializationError: If program deserialization fails.
        """
        if "schema_version" not in spec or "program" not in spec:
            raise AdapterError(
                "Malformed spec: missing 'schema_version' or 'program' key"
            )

        if spec["schema_version"] != cls._SCHEMA_VERSION:
            raise SerializationError(
                f"Unsupported schema_version={spec['schema_version']!r}; expected {cls._SCHEMA_VERSION!r}"
            )

        try:
            program = deserialize(spec["program"], registry)
        except Exception as exc:
            raise SerializationError(f"Failed to deserialize program: {exc}") from exc

        config_data = spec.get("config", {})
        try:
            gp_config = LiqGPConfig(**config_data) if config_data else LiqGPConfig()
        except Exception as exc:
            raise SerializationError(f"Failed to parse config: {exc}") from exc

        adapter = cls(registry=registry, gp_config=gp_config, evaluator=None)
        adapter._program = program
        return adapter
