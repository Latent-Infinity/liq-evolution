"""GP strategy adapter for liq-runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import polars as pl

from liq.evolution.adapters.parallel_eval import Evaluator, ParallelEvaluator
from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.config import EvolutionConfig, ParallelConfig
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
        warm_start_mode: Literal["replace", "augment"] = "replace",
        seed_programs_path: str | Path | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        self._registry = registry
        self._gp_config = gp_config
        self._evaluator = evaluator
        if seed_programs is not None:
            self._seed_programs = list(seed_programs)
        else:
            self._seed_programs = (
                _load_seed_programs(seed_programs_path, registry)
                if seed_programs_path is not None
                else None
            )
        self._warm_start = warm_start
        self._warm_start_mode = warm_start_mode
        self._parallel_config = parallel_config
        self._program: Program | None = None
        self._evolution_result: EvolutionResult | None = None

        # Validate seeds at construction (fail-fast)
        if self._seed_programs is not None:
            validate_seed_programs(
                self._seed_programs,
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
            if evaluator is None:
                raise AdapterError("Parallel evaluation requires an evaluator")

            evaluator = ParallelEvaluator(
                evaluator=cast(Evaluator, evaluator),
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
            if self._warm_start_mode == "augment" and self._seed_programs is not None:
                self._seed_programs = [*self._seed_programs, result.best_program]
            else:
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

    @classmethod
    def from_evolution_config(
        cls,
        registry: PrimitiveRegistry,
        gp_config: GPConfig,
        evolution_config: EvolutionConfig,
        evaluator: object | None = None,
        *,
        seed_programs: Sequence[Program] | None = None,
        warm_start: bool = False,
        parallel_config: ParallelConfig | None = None,
    ) -> GPStrategyAdapter:
        """Create an adapter from an EvolutionConfig using warm-start settings.

        Args:
            registry: GP primitive registry.
            gp_config: GP configuration for evolution.
            evolution_config: Evolution pipeline config providing warm-start
                defaults (seed path and mode).
            evaluator: Optional evaluator used during fitting.
            seed_programs: Explicit seed programs (takes precedence over path-based
                seeds from ``evolution_config``).
            warm_start: Whether to use latest evolved programs as warm seeds in
                subsequent fits.
            parallel_config: Optional parallel evaluation config.
        """
        warm_start_config = evolution_config.warm_start
        return cls(
            registry=registry,
            gp_config=gp_config,
            evaluator=evaluator,
            seed_programs=seed_programs,
            warm_start=warm_start,
            warm_start_mode=warm_start_config.mode,
            seed_programs_path=warm_start_config.seed_programs_path,
            parallel_config=parallel_config,
        )


def _load_seed_programs(
    seed_programs_path: str | Path | None,
    registry: PrimitiveRegistry,
) -> list[Program]:
    """Load seed programs from JSON.

    Supported payload formats:
    - ``[program, ...]``
    - ``{"seed_programs": [program, ...]}``
    - ``{"best_program": program}``
    - ``{"pareto_front": [program, ...]}``
    - ``{"entry_program": program}``
    """
    if seed_programs_path is None:
        return []

    path = Path(seed_programs_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AdapterError(f"Failed to read seed programs from {path!s}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AdapterError(f"Invalid JSON in seed programs path {path!s}: {exc}") from exc

    if isinstance(payload, dict):
        if "seed_programs" in payload:
            payload = payload["seed_programs"]
        elif "best_program" in payload:
            payload = [payload["best_program"]]
        elif "pareto_front" in payload:
            payload = payload["pareto_front"]
        elif "entry_program" in payload:
            payload = [payload["entry_program"]]
        else:
            raise AdapterError(
                f"Unsupported seed payload for {path!s}; expected "
                "list, seed_programs, best_program, pareto_front, or entry_program"
            )

    if not isinstance(payload, list):
        raise AdapterError(
            f"Unsupported seed payload for {path!s}; expected list of program payloads"
        )
    if not payload:
        raise AdapterError(f"No seed programs found in {path!s}")

    seed_programs: list[Program] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise AdapterError(
                f"Seed item at index {index} in {path!s} is not a program payload"
            )
        try:
            program = deserialize(item, registry)
        except Exception as exc:
            raise AdapterError(
                f"Failed to deserialize seed item at index {index} from {path!s}"
            ) from exc
        seed_programs.append(program)

    return seed_programs
