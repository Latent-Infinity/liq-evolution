"""End-to-end integration tests for Phase 3: adapter + fitness pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.config import FitnessStageConfig, PrimitiveConfig
from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator
from liq.evolution.fitness.objectives import wire_objectives
from liq.evolution.fitness.runner_backtest import BacktestFitnessEvaluator
from liq.evolution.fitness.two_stage import TwoStageFitnessEvaluator
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.primitives.series_sources import prepare_evaluation_context
from liq.evolution.protocols import GPStrategy
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.config import SeedInjectionConfig
from liq.gp.program.ast import FunctionNode, TerminalNode
from liq.gp.program.eval import evaluate as gp_evaluate
from liq.gp.types import EvolutionResult, Series


def _small_gp_config(*, seed: int = 42) -> LiqGPConfig:
    return LiqGPConfig(
        population_size=20,
        max_depth=4,
        generations=3,
        seed=seed,
        tournament_size=3,
        elitism_count=2,
        constant_opt_enabled=False,
        semantic_dedup_enabled=False,
        simplification_enabled=False,
    )


def _make_ohlcv(n: int = 50) -> dict[str, np.ndarray]:
    """Create deterministic OHLCV data."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    return {
        "open": close - rng.uniform(0, 1, n),
        "high": close + rng.uniform(0, 2, n),
        "low": close - rng.uniform(0, 2, n),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    }


def _make_features(n: int = 50) -> pl.DataFrame:
    """Create a full-context DataFrame including derived series."""
    ohlcv = _make_ohlcv(n)
    ctx = prepare_evaluation_context(ohlcv)
    return pl.DataFrame(ctx)


def _make_labels(features: pl.DataFrame) -> pl.Series:
    close = features["close"].to_numpy()
    mean_close = np.convolve(close, np.ones(5) / 5, mode="full")[: len(close)]
    mean_close[:4] = close[:4]
    return pl.Series("label", np.where(close > mean_close, 1.0, 0.0))


def _make_mock_backtest_runner(
    metric: str = "sharpe_ratio",
    value: float = 1.5,
    n_folds: int = 3,
):
    """Create a backtest runner that calls strategy.fit/predict and returns folds."""

    def runner(strategy: Any) -> Sequence[dict[str, Any]]:
        # Actually call fit/predict to verify the adapter works
        features = _make_features(30)
        labels = _make_labels(features)
        strategy.fit(features, labels)
        output = strategy.predict(features)
        assert isinstance(output, GPSignalOutput)
        assert len(output.scores) == 30
        return [{"metrics": {metric: value + i * 0.1}} for i in range(n_folds)]

    return runner


# ------------------------------------------------------------------ #
#  GPStrategyAdapter fit/predict round-trip
# ------------------------------------------------------------------ #


class TestAdapterFitPredictE2E:
    """End-to-end: real evolution → predict with polars DataFrames."""

    def test_fit_predict_roundtrip(self) -> None:
        """fit() runs real evolution, predict() returns GPSignalOutput."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        gp_config = _small_gp_config()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        adapter = GPStrategyAdapter(registry, gp_config, evaluator)

        features = _make_features(30)
        labels = _make_labels(features)
        adapter.fit(features, labels)

        assert adapter.program is not None
        assert adapter.evolution_result is not None
        assert isinstance(adapter.evolution_result, EvolutionResult)

        output = adapter.predict(features)
        assert isinstance(output, GPSignalOutput)
        assert len(output.scores) == 30
        assert output.scores.dtype == pl.Float64

    def test_adapter_satisfies_gp_strategy_protocol(self) -> None:
        """GPStrategyAdapter satisfies the GPStrategy protocol."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        adapter = GPStrategyAdapter(
            registry,
            _small_gp_config(),
            MagicMock(),
        )
        assert isinstance(adapter, GPStrategy)


# ------------------------------------------------------------------ #
#  Export/import spec round-trip
# ------------------------------------------------------------------ #


class TestSpecRoundTripE2E:
    """End-to-end: evolve → export_spec → from_spec → predict identity."""

    def test_export_from_spec_predict_identity(self) -> None:
        """Predictions from exported/restored adapter match original."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        gp_config = _small_gp_config()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        adapter = GPStrategyAdapter(registry, gp_config, evaluator)
        features = _make_features(30)
        labels = _make_labels(features)
        adapter.fit(features, labels)

        spec = adapter.export_spec()
        restored = GPStrategyAdapter.from_spec(spec, registry)

        original_out = adapter.predict(features)
        restored_out = restored.predict(features)
        np.testing.assert_array_equal(
            original_out.scores.to_numpy(),
            restored_out.scores.to_numpy(),
        )


# ------------------------------------------------------------------ #
#  Warm-start
# ------------------------------------------------------------------ #


class TestWarmStartE2E:
    """End-to-end: warm-start second fit uses best from first."""

    def test_warm_start_second_fit(self) -> None:
        """Second fit with warm_start=True uses previous best as seed."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        gp_config = _small_gp_config()
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        adapter = GPStrategyAdapter(registry, gp_config, evaluator, warm_start=True)

        features = _make_features(30)
        labels = _make_labels(features)

        # First fit
        adapter.fit(features, labels)
        first_program = adapter.program

        # Second fit (warm-start seeds from first)
        adapter.fit(features, labels)
        second_program = adapter.program

        assert first_program is not None
        assert second_program is not None
        # Both should produce valid outputs
        out1 = gp_evaluate(
            first_program, {c: features[c].to_numpy() for c in features.columns}
        )
        out2 = gp_evaluate(
            second_program, {c: features[c].to_numpy() for c in features.columns}
        )
        assert len(out1) == 30
        assert len(out2) == 30


# ------------------------------------------------------------------ #
#  Two-stage with mock backtest via wire_objectives
# ------------------------------------------------------------------ #


class TestTwoStageViaWireObjectivesE2E:
    """End-to-end: wire_objectives builds TwoStage with mock backtest."""

    def test_wire_objectives_two_stage(self) -> None:
        """wire_objectives returns TwoStageEvaluator with backtest_fn."""
        mock_fn = lambda strategy: [{"metrics": {"sharpe_ratio": 1.5}}]
        config = FitnessStageConfig(
            use_backtest=True,
            backtest_metric="sharpe_ratio",
            backtest_top_n=5,
        )
        evaluator = wire_objectives(config, backtest_fn=mock_fn)
        assert isinstance(evaluator, TwoStageFitnessEvaluator)

    def test_two_stage_evaluator_with_real_programs(self) -> None:
        """TwoStage evaluator works with real evolved programs."""
        prim_config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(prim_config)
        gp_config = _small_gp_config()
        label_evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        # Build context
        features = _make_features(30)
        labels = _make_labels(features)
        context = {col: features[col].to_numpy() for col in features.columns}
        context["labels"] = labels.to_numpy()

        # Evolve to get real programs
        from liq.gp.evolution.engine import evolve

        result = evolve(registry, gp_config, label_evaluator, context)
        programs = result.pareto_front[:3]  # Use a few evolved programs

        # Mock backtest that returns sharpe
        mock_fn = lambda strategy: [{"metrics": {"sharpe_ratio": 1.0}}]
        backtest_evaluator = BacktestFitnessEvaluator(
            backtest_runner=mock_fn, metric="sharpe_ratio"
        )

        two_stage = TwoStageFitnessEvaluator(
            stage_a=label_evaluator,
            stage_b=backtest_evaluator,
            top_k=2,
        )

        results = two_stage.evaluate(programs, context)
        assert len(results) == len(programs)
        for r in results:
            assert len(r.objectives) >= 1


# ------------------------------------------------------------------ #
#  BacktestFitnessEvaluator with real evolved programs
# ------------------------------------------------------------------ #


class TestBacktestWithEvolvedProgramE2E:
    """End-to-end: evolve → wrap in backtest evaluator."""

    def test_backtest_evaluator_with_evolved_program(self) -> None:
        """Evolved program evaluates through BacktestFitnessEvaluator."""
        prim_config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(prim_config)
        gp_config = _small_gp_config()
        label_evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        features = _make_features(30)
        labels = _make_labels(features)
        context = {col: features[col].to_numpy() for col in features.columns}
        context["labels"] = labels.to_numpy()

        from liq.gp.evolution.engine import evolve

        result = evolve(registry, gp_config, label_evaluator, context)

        # Use the evolved program with backtest evaluator
        mock_fn = _make_mock_backtest_runner(value=2.0, n_folds=2)
        evaluator = BacktestFitnessEvaluator(
            backtest_runner=mock_fn, metric="sharpe_ratio"
        )

        results = evaluator.evaluate([result.best_program], context)
        assert len(results) == 1
        assert results[0].objectives[0] > 0
        assert results[0].metadata["n_folds"] == 2


# ------------------------------------------------------------------ #
#  Public API re-exports
# ------------------------------------------------------------------ #


class TestPhase3PublicExports:
    """Verify Phase 3 types are exported from package root."""

    def test_adapter_exports_from_adapters_package(self) -> None:
        from liq.evolution.adapters import GPSignalOutput, GPStrategyAdapter

        assert GPStrategyAdapter is not None
        assert GPSignalOutput is not None

    def test_fitness_exports_from_fitness_package(self) -> None:
        from liq.evolution.fitness import (
            BacktestFitnessEvaluator,
            TwoStageFitnessEvaluator,
        )

        assert BacktestFitnessEvaluator is not None
        assert TwoStageFitnessEvaluator is not None

    def test_root_package_exports(self) -> None:
        from liq.evolution import (
            BacktestFitnessEvaluator,
            GPSignalOutput,
            GPStrategyAdapter,
            TwoStageFitnessEvaluator,
        )

        assert GPStrategyAdapter is not None
        assert GPSignalOutput is not None
        assert BacktestFitnessEvaluator is not None
        assert TwoStageFitnessEvaluator is not None

    def test_all_list_contains_phase3_exports(self) -> None:
        import liq.evolution

        expected = [
            "GPStrategyAdapter",
            "GPSignalOutput",
            "BacktestFitnessEvaluator",
            "TwoStageFitnessEvaluator",
        ]
        for name in expected:
            assert name in liq.evolution.__all__, f"{name} not in __all__"


def _make_bool_series_seeds(registry_obj) -> list:
    """Build simple BoolSeries seeds from comparison ops (no liq-ta needed).

    Creates ``close > open`` and ``high > low`` programs.
    """
    gt = registry_obj.get("gt")
    close_node = TerminalNode(name="close", output_type=Series)
    open_node = TerminalNode(name="open", output_type=Series)
    high_node = TerminalNode(name="high", output_type=Series)
    low_node = TerminalNode(name="low", output_type=Series)
    return [
        FunctionNode(gt, (close_node, open_node)),  # close > open
        FunctionNode(gt, (high_node, low_node)),  # high > low
    ]


class TestSeedInjectionThroughAdapterE2E:
    """Periodic seed injection works end-to-end through the adapter."""

    def test_fit_with_ramped_injection(self) -> None:
        """GPStrategyAdapter.fit() with ramped injection (no seeds needed)."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        gp_config = LiqGPConfig(
            population_size=20,
            max_depth=4,
            generations=4,
            seed=42,
            tournament_size=3,
            elitism_count=2,
            constant_opt_enabled=False,
            semantic_dedup_enabled=False,
            simplification_enabled=False,
            seed_injection=SeedInjectionConfig(
                method="ramped",
                count=2,
                interval=1,
            ),
        )
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        adapter = GPStrategyAdapter(registry, gp_config, evaluator)

        features = _make_features(30)
        labels = _make_labels(features)
        adapter.fit(features, labels)

        assert adapter.program is not None
        result = adapter.evolution_result
        assert result is not None
        total_injected = sum(s.injected_count for s in result.fitness_history)
        # Injection fires on generations 1, 2, 3 (not gen 0)
        assert total_injected > 0, (
            f"Expected injected_count > 0 with seed_injection enabled, "
            f"got {total_injected}"
        )

    def test_fit_with_direct_seed_injection(self) -> None:
        """Direct injection cycles BoolSeries seeds through evolution."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        seeds = _make_bool_series_seeds(registry)
        gp_config = LiqGPConfig(
            population_size=20,
            max_depth=4,
            generations=4,
            seed=42,
            tournament_size=3,
            elitism_count=2,
            constant_opt_enabled=False,
            semantic_dedup_enabled=False,
            simplification_enabled=False,
            seed_injection=SeedInjectionConfig(
                method="direct",
                count=2,
                interval=1,
            ),
        )
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        adapter = GPStrategyAdapter(registry, gp_config, evaluator, seed_programs=seeds)

        features = _make_features(30)
        labels = _make_labels(features)
        adapter.fit(features, labels)

        assert adapter.program is not None
        result = adapter.evolution_result
        assert result is not None
        total_injected = sum(s.injected_count for s in result.fitness_history)
        assert total_injected > 0, (
            f"Expected injected_count > 0 with direct injection, got {total_injected}"
        )

    def test_fit_with_variation_seed_injection(self) -> None:
        """Variation injection mutates BoolSeries seeds through evolution."""
        config = PrimitiveConfig(enable_liq_ta=False)
        registry = build_trading_registry(config)
        seeds = _make_bool_series_seeds(registry)
        gp_config = LiqGPConfig(
            population_size=20,
            max_depth=4,
            generations=4,
            seed=42,
            tournament_size=3,
            elitism_count=2,
            constant_opt_enabled=False,
            semantic_dedup_enabled=False,
            simplification_enabled=False,
            seed_injection=SeedInjectionConfig(
                method="variation",
                count=2,
                interval=1,
            ),
        )
        evaluator = LabelFitnessEvaluator(metric="f1", top_k=0.5)

        adapter = GPStrategyAdapter(registry, gp_config, evaluator, seed_programs=seeds)

        features = _make_features(30)
        labels = _make_labels(features)
        adapter.fit(features, labels)

        assert adapter.program is not None
        result = adapter.evolution_result
        assert result is not None
        total_injected = sum(s.injected_count for s in result.fitness_history)
        assert total_injected > 0, (
            f"Expected injected_count > 0 with variation injection, "
            f"got {total_injected}"
        )
