"""Tests for QD orchestration (`run_qd_evolution`)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from liq.evolution.config import PrimitiveConfig
from liq.evolution.fitness.evaluation_schema import (
    BEHAVIOR_DESCRIPTOR_TURNOVER,
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
)
from liq.evolution.primitives import prepare_evaluation_context
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.qd.orchestrator import (
    QDEvolutionResult,
    _extract_descriptors,
    _extract_objectives,
    _qd_parent_source,
    _safe_float,
    _seed_archive,
    run_qd_evolution,
)
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.evolution.qd_archive import QDArchive
from liq.gp.program.ast import Program
from liq.gp.types import FitnessResult, SelectionContext


class _ToyEvaluator:
    """Minimal evaluator that returns deterministic behavior descriptors."""

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],  # noqa: ARG002
    ) -> list[FitnessResult]:
        results: list[FitnessResult] = []
        for program in programs:
            turnover = (program.size % 7) / 6.0
            results.append(
                FitnessResult(
                    objectives=(float(program.size),),
                    metadata={
                        METADATA_KEY_PER_SPLIT_METRICS: {
                            "all": {"size": float(program.size)},
                        },
                        METADATA_KEY_RAW_OBJECTIVES: (float(program.size),),
                        METADATA_KEY_BEHAVIOR_DESCRIPTORS: {
                            BEHAVIOR_DESCRIPTOR_TURNOVER: turnover,
                        },
                        METADATA_KEY_CONSTRAINT_VIOLATIONS: {},
                        METADATA_KEY_SLICE_SCORES: {},
                    },
                )
            )
        return results


def _small_gp_config(*, seed: int = 17) -> LiqGPConfig:
    return LiqGPConfig(
        population_size=12,
        max_depth=3,
        generations=2,
        seed=seed,
        tournament_size=2,
        elitism_count=1,
        constant_opt_enabled=False,
        simplification_enabled=False,
    )


class TestRunQDEvolution:
    """Execute a tiny QD evolution loop and verify archive/portfolio output."""

    def test_run_qd_evolution_returns_portfolio_and_coverage(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        config = _small_gp_config()
        context = prepare_evaluation_context(
            {
                "open": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
                "high": np.array([1.1, 2.1, 3.1, 4.1], dtype=float),
                "low": np.array([0.9, 1.9, 2.9, 3.9], dtype=float),
                "close": np.array([1.05, 2.05, 3.05, 4.05], dtype=float),
                "volume": np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=float),
            },
        )
        result = run_qd_evolution(
            registry=registry,
            config=config,
            evaluator=_ToyEvaluator(),
            context=context,
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
            archive_bins_per_dim=4,
            coverage_weight=0.5,
            coverage_interval=1,
            portfolio_size=3,
        )

        assert isinstance(result, QDEvolutionResult)
        assert result.archive.n_dims == 1
        assert result.coverage_report["filled_bins"] > 0
        assert len(result.portfolio) == 3
        assert result.behavior_descriptor_names == (BEHAVIOR_DESCRIPTOR_TURNOVER,)

    def test_run_qd_evolution_invalid_coverage_weight(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        config = _small_gp_config(seed=42)
        context = prepare_evaluation_context(
            {
                "open": np.array([1.0, 2.0, 3.0], dtype=float),
                "high": np.array([1.1, 2.1, 3.1], dtype=float),
                "low": np.array([0.9, 1.9, 2.9], dtype=float),
                "close": np.array([1.05, 2.05, 3.05], dtype=float),
                "volume": np.array([1000.0, 1000.0, 1000.0], dtype=float),
            },
        )

        with pytest.raises(ValueError, match="coverage_weight"):
            run_qd_evolution(
                registry=registry,
                config=config,
                evaluator=_ToyEvaluator(),
                context=context,
                behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
                coverage_weight=1.5,
            )

    def test_run_qd_evolution_invalid_coverage_interval(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        config = _small_gp_config()
        context = prepare_evaluation_context(
            {
                "open": np.array([1.0], dtype=float),
                "high": np.array([1.1], dtype=float),
                "low": np.array([0.9], dtype=float),
                "close": np.array([1.05], dtype=float),
                "volume": np.array([1000.0], dtype=float),
            },
        )
        with pytest.raises(ValueError, match="coverage_interval must be >= 1"):
            run_qd_evolution(
                registry=registry,
                config=config,
                evaluator=_ToyEvaluator(),
                context=context,
                coverage_interval=0,
            )

    def test_run_qd_evolution_invalid_portfolio_size(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        config = _small_gp_config(seed=2)
        context = prepare_evaluation_context(
            {
                "open": np.array([1.0], dtype=float),
                "high": np.array([1.1], dtype=float),
                "low": np.array([0.9], dtype=float),
                "close": np.array([1.05], dtype=float),
                "volume": np.array([1000.0], dtype=float),
            },
        )
        with pytest.raises(ValueError, match="portfolio_size must be >= 1"):
            run_qd_evolution(
                registry=registry,
                config=config,
                evaluator=_ToyEvaluator(),
                context=context,
                portfolio_size=0,
            )

    def test_run_qd_evolution_requires_descriptor_names(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        config = _small_gp_config(seed=3)
        context = prepare_evaluation_context(
            {
                "open": np.array([1.0], dtype=float),
                "high": np.array([1.1], dtype=float),
                "low": np.array([0.9], dtype=float),
                "close": np.array([1.05], dtype=float),
                "volume": np.array([1000.0], dtype=float),
            },
        )
        with pytest.raises(
            ValueError, match="behavior_descriptor_names must be non-empty"
        ):
            run_qd_evolution(
                registry=registry,
                config=config,
                evaluator=_ToyEvaluator(),
                context=context,
                behavior_descriptor_names=(),
            )

    def test_run_qd_evolution_rejects_descriptor_count_mismatch(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        config = _small_gp_config(seed=4)
        context = prepare_evaluation_context(
            {
                "open": np.array([1.0], dtype=float),
                "high": np.array([1.1], dtype=float),
                "low": np.array([0.9], dtype=float),
                "close": np.array([1.05], dtype=float),
                "volume": np.array([1000.0], dtype=float),
            },
        )
        archive = QDArchive(
            n_dims=2,
            bins_per_dim=(4, 4),
            descriptor_bounds=((0.0, 1.0), (0.0, 1.0)),
            objective_directions=list(config.fitness.objective_directions),
            bin_capacity=1,
        )
        with pytest.raises(ValueError, match="must equal archive.n_dims"):
            run_qd_evolution(
                registry=registry,
                config=config,
                evaluator=_ToyEvaluator(),
                context=context,
                archive=archive,
                behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
            )


class _ArchiveStub:
    def __init__(self, *, n_dims: int, elites: list[Program] | None = None) -> None:
        self.n_dims = n_dims
        self._elites = elites or []

    def elites(self) -> list[Program]:
        return list(self._elites)

    def coverage_report(self) -> dict[str, object]:
        return {"filled_bins": 0}


def test_run_qd_evolution_falls_back_to_pareto_front_when_archive_empty() -> None:
    registry = build_trading_registry(PrimitiveConfig())
    config = _small_gp_config(seed=5)
    context = prepare_evaluation_context(
        {
            "open": np.array([1.0], dtype=float),
            "high": np.array([1.1], dtype=float),
            "low": np.array([0.9], dtype=float),
            "close": np.array([1.05], dtype=float),
            "volume": np.array([1000.0], dtype=float),
        },
    )
    portfolio_program = "best"

    from liq.evolution import qd

    original = qd.orchestrator.evolve

    try:
        qd.orchestrator.evolve = lambda **_: SimpleNamespace(
            best_program=portfolio_program,
            pareto_front=[portfolio_program],
            fitness_history=[],
            config=config,
        )
        result = run_qd_evolution(
            registry=registry,
            config=config,
            evaluator=_ToyEvaluator(),
            context=context,
            archive=_ArchiveStub(n_dims=1, elites=[]),
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
            portfolio_size=3,
        )
        assert result.portfolio == [portfolio_program]
    finally:
        qd.orchestrator.evolve = original


def test_safe_float_and_extraction_helpers() -> None:
    assert _safe_float("bad") is None
    assert _safe_float(float("inf")) is None
    assert _safe_float(np.float64(1.25)) == 1.25

    valid_fitness = FitnessResult(
        objectives=(1.0, 2.0),
        metadata={
            METADATA_KEY_RAW_OBJECTIVES: (1.0, 2.0),
            METADATA_KEY_BEHAVIOR_DESCRIPTORS: {"x": -0.25, "y": 1.5},
        },
    )
    invalid_fitness = FitnessResult(
        objectives=("bad",),
        metadata={},
    )

    assert _extract_objectives(valid_fitness) == (1.0, 2.0)
    assert _extract_objectives(invalid_fitness) is None
    assert _extract_descriptors(valid_fitness, ("x", "y")) == (0.0, 1.0)
    assert _extract_descriptors(invalid_fitness, ("x",)) is None


class _RecordingArchive:
    def __init__(self, *, raise_on_insert: bool = False) -> None:
        self.raise_on_insert = raise_on_insert
        self.inserted: list[tuple[Any, tuple[float, ...], tuple[float, ...]]] = []

    def insert(
        self,
        individual: Program,
        objectives: tuple[float, ...],
        descriptors: tuple[float, ...],
    ) -> bool:
        if self.raise_on_insert:
            raise RuntimeError("insert failed")
        self.inserted.append((individual, objectives, descriptors))
        return True


def test_seed_archive_skips_invalid_fitness_and_honors_insert_errors() -> None:
    archive = _RecordingArchive()
    population = ["a", "b"]
    valid = FitnessResult(
        objectives=(1.0,),
        metadata={
            METADATA_KEY_RAW_OBJECTIVES: (1.0,),
            METADATA_KEY_BEHAVIOR_DESCRIPTORS: {"turnover": 0.5},
        },
    )
    invalid = FitnessResult(objectives=("bad",), metadata={})
    _seed_archive(archive, population, [valid, invalid], ("turnover",))
    assert len(archive.inserted) == 1

    failing_archive = _RecordingArchive(raise_on_insert=True)
    _seed_archive(failing_archive, [population[0]], [valid], ("turnover",))
    assert len(failing_archive.inserted) == 0


def test_qd_parent_source_applies_coverage_interval_and_minimum_pick() -> None:
    from liq.evolution import qd

    original_select = qd.orchestrator.select
    qd.orchestrator.select = (
        lambda population, fitnesses, config, rng, *, fronts, ranks, crowding: [
            999,
            998,
        ]
    )
    archive = _Archive()
    try:
        source = _qd_parent_source(
            archive=archive,
            descriptor_names=("turnover",),
            coverage_weight=0.01,
            coverage_interval=2,
        )
        config = _small_gp_config(seed=9)
        fitness = FitnessResult(
            objectives=(1.0,),
            metadata={
                METADATA_KEY_RAW_OBJECTIVES: (1.0,),
                METADATA_KEY_BEHAVIOR_DESCRIPTORS: {"turnover": 0.5},
            },
        )
        rng = np.random.default_rng(0)
        ctx = SelectionContext()

        first = source([1], [fitness], config, rng, target_size=2, sel_ctx=ctx)
        second = source([2], [fitness], config, rng, target_size=2, sel_ctx=ctx)
        assert first[:1] == [111]
        assert len(second) == 2
    finally:
        qd.orchestrator.select = original_select


class _Archive:
    def __init__(self) -> None:
        self.filled_bins = 1

    def sample(self, n: int, rng, coverage_weight: float) -> list[int]:
        return [111]

    def insert(self, individual: Program, objectives: Any, descriptors: Any) -> bool:
        return True
