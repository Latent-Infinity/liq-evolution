"""Configuration models for liq-evolution (Pydantic v2).

Provides frozen Pydantic models that validate all parameters at construction
time (fail-fast).  Invalid values raise
:class:`~liq.evolution.errors.ConfigurationError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from liq.evolution.errors import ConfigurationError
from liq.gp.config import FitnessConfig as LiqGPFitnessConfig
from liq.gp.config import GPConfig as LiqGPConfig


class PrimitiveConfig(BaseModel, frozen=True):
    """Configuration for which trading primitive categories to register.

    Attributes:
        enable_numeric_ops: Register numeric operators (+, -, *, /).
        enable_comparison_ops: Register comparison operators (>, <, ==).
        enable_logic_ops: Register logical operators (and, or, not).
        enable_crossover_ops: Register crossover detection operators.
        enable_temporal_ops: Register temporal/lag operators.
        enable_series_sources: Register price/volume series terminals.
        enable_liq_ta: Register liq-ta indicator primitives (requires
            the ``[indicators]`` extra).
    """

    enable_numeric_ops: bool = True
    enable_comparison_ops: bool = True
    enable_logic_ops: bool = True
    enable_crossover_ops: bool = True
    enable_temporal_ops: bool = True
    enable_series_sources: bool = True
    enable_liq_ta: bool = False


class FitnessStageConfig(BaseModel, frozen=True):
    """Configuration for fitness evaluation stages.

    Attributes:
        label_metric: Metric for label-based fitness evaluation.
        label_top_k: Fraction of top predictions to consider.
        use_backtest: Whether to add a backtest stage after label fitness.
        backtest_top_n: Number of top programs to backtest.
        backtest_metric: Metric to optimize in backtesting.
    """

    label_metric: Literal["f1", "precision_at_k", "accuracy"] = "f1"
    label_top_k: float = 0.1
    use_backtest: bool = False
    backtest_top_n: int = 10
    backtest_metric: Literal["sharpe_ratio", "sortino_ratio", "total_return"] = (
        "sharpe_ratio"
    )

    @model_validator(mode="after")
    def _validate_fitness_stage(self) -> Self:
        if self.label_top_k <= 0.0 or self.label_top_k > 1.0:
            raise ConfigurationError("label_top_k must be in (0, 1]")
        if self.backtest_top_n < 1:
            raise ConfigurationError("backtest_top_n must be >= 1")
        return self


class ParallelConfig(BaseModel, frozen=True):
    """Configuration for parallel evaluation.

    Attributes:
        backend: Parallelisation backend to use.
        max_workers: Maximum number of parallel workers.
        max_in_flight: Maximum concurrent tasks submitted to ray.
        max_tasks_per_worker: Worker lifecycle recycling threshold.
        memory_limit_mb: Memory limit per worker in megabytes.
        memory_warn_threshold_mb: RSS threshold for warning log.
        auto_fallback: Fallback to sequential on memory pressure.
    """

    backend: Literal["sequential", "ray"] = "sequential"
    max_workers: int = 1
    max_in_flight: int = 4
    max_tasks_per_worker: int = 100
    memory_limit_mb: int = 2048
    memory_warn_threshold_mb: int = 1536
    auto_fallback: bool = True

    @model_validator(mode="after")
    def _validate_parallel(self) -> Self:
        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be >= 1")
        if self.max_in_flight < 1:
            raise ConfigurationError("max_in_flight must be >= 1")
        if self.max_tasks_per_worker < 1:
            raise ConfigurationError("max_tasks_per_worker must be >= 1")
        if self.memory_limit_mb < 128:
            raise ConfigurationError("memory_limit_mb must be >= 128")
        if self.memory_warn_threshold_mb < 0:
            raise ConfigurationError("memory_warn_threshold_mb must be >= 0")
        if self.memory_warn_threshold_mb >= self.memory_limit_mb:
            raise ConfigurationError(
                "memory_warn_threshold_mb must be < memory_limit_mb"
            )
        return self


class WarmStartConfig(BaseModel, frozen=True):
    """Configuration for warm-starting evolution with seed programs.

    Attributes:
        seed_programs_path: Path to serialized seed programs, or None.
        mode: How to incorporate seeds into the initial population.
    """

    seed_programs_path: Path | None = None
    mode: Literal["replace", "augment"] = "replace"


class GPConfig(BaseModel, frozen=True):
    """Canonical phase-0 configuration for core GP evolution controls."""

    population_size: int = 300
    max_depth: int = 8
    generations: int = 50
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism_count: int = 1
    seed: int = 42

    @model_validator(mode="after")
    def _validate_gp_config(self) -> Self:
        if self.population_size < 1:
            raise ConfigurationError("population_size must be >= 1")
        if self.max_depth < 1:
            raise ConfigurationError("max_depth must be >= 1")
        if self.generations < 1:
            raise ConfigurationError("generations must be >= 1")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ConfigurationError("mutation_rate must be in [0.0, 1.0]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ConfigurationError("crossover_rate must be in [0.0, 1.0]")
        if self.tournament_size < 1:
            raise ConfigurationError("tournament_size must be >= 1")
        if self.elitism_count < 0:
            raise ConfigurationError("elitism_count must be >= 0")

        total_rate = self.crossover_rate + self.mutation_rate
        if total_rate > 1.0:
            raise ConfigurationError("crossover_rate + mutation_rate must be <= 1.0")
        return self


class FitnessConfig(BaseModel, frozen=True):
    """Canonical phase-0 configuration for fitness-stage control."""

    metric: Literal["f1", "precision_at_k", "accuracy"] = "f1"
    stage_a_metric: Literal["f1", "precision_at_k", "accuracy"] = "f1"
    stage_b_enabled: bool = False
    top_k_for_backtest: int = 10

    @model_validator(mode="after")
    def _validate_fitness_config(self) -> Self:
        if self.top_k_for_backtest < 1:
            raise ConfigurationError("top_k_for_backtest must be >= 1")
        return self


class SerializationConfig(BaseModel, frozen=True):
    """Configuration for serialized strategy/program payloads."""

    schema_version: str = "1.0"
    enable_persistent_cache: bool = False


class EvolutionConfig(BaseModel, frozen=True):
    """Main configuration for the trading strategy evolution pipeline.

    All parameters have sensible defaults.  Invalid values raise
    :class:`~liq.evolution.errors.ConfigurationError` at construction time.

    Attributes:
        population_size: Number of programs in the population.
        max_depth: Maximum depth of GP program trees.
        generations: Number of evolution generations to run.
        seed: Random seed for reproducibility.
        primitives: Primitive category configuration.
        fitness_stages: Fitness evaluation stage configuration.
        parallel: Parallel evaluation configuration.
        warm_start: Warm-start configuration.
    """

    population_size: int = 300
    max_depth: int = 8
    generations: int = 50
    seed: int = 42
    batch_size: int | None = None
    full_eval_interval: int = 10
    primitives: PrimitiveConfig = PrimitiveConfig()
    fitness_stages: FitnessStageConfig = FitnessStageConfig()
    parallel: ParallelConfig = ParallelConfig()
    gp: GPConfig = Field(default_factory=GPConfig)
    warm_start: WarmStartConfig = WarmStartConfig()

    @model_validator(mode="after")
    def _validate_evolution(self) -> Self:
        if self.population_size < 10:
            raise ConfigurationError("population_size must be >= 10")
        if self.max_depth < 2:
            raise ConfigurationError("max_depth must be >= 2")
        if self.generations < 1:
            raise ConfigurationError("generations must be >= 1")
        if self.batch_size is not None and self.batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1 when set")
        if self.full_eval_interval < 1:
            raise ConfigurationError("full_eval_interval must be >= 1")
        return self


def build_gp_config(evo: EvolutionConfig) -> LiqGPConfig:
    """Convert an :class:`EvolutionConfig` to a :class:`liq.gp.config.GPConfig`.

    Direct fields (``population_size``, ``max_depth``, ``generations``,
    ``seed``) are mapped 1-to-1.  The embedded :class:`GPConfig` provides
    ``crossover_rate`` and ``mutation_rate`` and is used to derive the five
    operator rates required by :pyclass:`liq.gp.config.GPConfig`:

    * ``crossover_rate``              = *crossover_rate*
    * ``subtree_mutation_rate``       = 0.4 * *mutation_rate*
    * ``point_mutation_rate``         = 0.3 * *mutation_rate*
    * ``parameter_mutation_rate``     = 0.2 * *mutation_rate*
    * ``hoist_mutation_rate``         = 0.1 * *mutation_rate*

    ``tournament_size`` is taken from the embedded :class:`GPConfig`, clamped to
    a minimum of 2 (liq-gp requirement).
    ``elitism_count`` is taken from the embedded :class:`GPConfig`.
    """
    local = evo.gp

    cr = local.crossover_rate
    mr = local.mutation_rate

    return LiqGPConfig(
        population_size=evo.population_size,
        max_depth=evo.max_depth,
        generations=evo.generations,
        seed=evo.seed,
        crossover_rate=cr,
        subtree_mutation_rate=0.4 * mr,
        point_mutation_rate=0.3 * mr,
        parameter_mutation_rate=0.2 * mr,
        hoist_mutation_rate=0.1 * mr,
        tournament_size=max(local.tournament_size, 2),
        elitism_count=local.elitism_count,
        fitness=LiqGPFitnessConfig(
            batch_size=evo.batch_size,
            full_eval_interval=evo.full_eval_interval,
        ),
    )
