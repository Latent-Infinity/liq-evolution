"""Tests for QD orchestration (`run_qd_evolution`)."""

from __future__ import annotations

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
from liq.evolution.qd.orchestrator import QDEvolutionResult, run_qd_evolution
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.primitives import prepare_evaluation_context
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.program.ast import Program
from liq.gp.types import FitnessResult


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
        semantic_dedup_enabled=False,
        simplification_enabled=False,
    )


class TestRunQDEvolution:
    """Execute a tiny QD evolution loop and verify archive/portfolio output."""

    def test_run_qd_evolution_returns_portfolio_and_coverage(self) -> None:
        registry = build_trading_registry(PrimitiveConfig(enable_liq_ta=False))
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
        registry = build_trading_registry(PrimitiveConfig(enable_liq_ta=False))
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
