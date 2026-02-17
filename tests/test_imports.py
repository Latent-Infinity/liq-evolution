"""Tests for package imports and public API surface."""

from __future__ import annotations

import pytest


class TestSubpackageImports:
    """Verify all subpackages are importable."""

    def test_root_package(self) -> None:
        import liq.evolution

        assert hasattr(liq.evolution, "__version__")

    def test_config_module(self) -> None:
        from liq.evolution import config

        assert hasattr(config, "EvolutionConfig")
        assert hasattr(config, "GPConfig")
        assert hasattr(config, "FitnessConfig")
        assert hasattr(config, "SerializationConfig")

    def test_errors_module(self) -> None:
        from liq.evolution import errors

        assert hasattr(errors, "LiqEvolutionError")

    def test_protocols_module(self) -> None:
        from liq.evolution import protocols

        assert hasattr(protocols, "IndicatorBackend")
        assert hasattr(protocols, "PrimitiveRegistry")
        assert hasattr(protocols, "FitnessEvaluator")

    def test_primitives_registry(self) -> None:
        from liq.evolution.primitives.registry import build_trading_registry

        assert callable(build_trading_registry)

    def test_primitives_ops_numeric(self) -> None:
        from liq.evolution.primitives.ops_numeric import register_numeric_ops

        assert callable(register_numeric_ops)

    def test_primitives_ops_comparison(self) -> None:
        from liq.evolution.primitives.ops_comparison import register_comparison_ops

        assert callable(register_comparison_ops)

    def test_primitives_ops_logic(self) -> None:
        from liq.evolution.primitives.ops_logic import register_logic_ops

        assert callable(register_logic_ops)

    def test_primitives_ops_crossover(self) -> None:
        from liq.evolution.primitives.ops_crossover import register_crossover_ops

        assert callable(register_crossover_ops)

    def test_primitives_ops_temporal(self) -> None:
        from liq.evolution.primitives.ops_temporal import register_temporal_ops

        assert callable(register_temporal_ops)

    def test_primitives_series_sources(self) -> None:
        from liq.evolution.primitives.series_sources import register_series_sources

        assert callable(register_series_sources)

    def test_primitives_indicators_liq_ta(self) -> None:
        from liq.evolution.primitives.indicators_liq_ta import (
            register_liq_ta_indicators,
        )

        assert callable(register_liq_ta_indicators)

    def test_fitness_label_metrics(self) -> None:
        from liq.evolution.fitness.label_metrics import LabelFitnessEvaluator

        assert LabelFitnessEvaluator is not None

    def test_fitness_runner_backtest(self) -> None:
        from liq.evolution.fitness.runner_backtest import BacktestFitnessEvaluator

        assert BacktestFitnessEvaluator is not None

    def test_fitness_objectives(self) -> None:
        from liq.evolution.fitness.objectives import wire_objectives

        assert callable(wire_objectives)

    def test_adapters_runner_strategy(self) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter

        assert GPStrategyAdapter is not None

    def test_program_module(self) -> None:
        from liq.evolution.program import Program

        assert Program is not None

    def test_adapters_parallel_eval(self) -> None:
        from liq.evolution.adapters.parallel_eval import ParallelEvaluator

        assert ParallelEvaluator is not None

    def test_adapters_signals_provider(self) -> None:
        pytest.importorskip("liq.core", reason="liq-core not installed")
        from liq.evolution.adapters.signals_provider import GPSignalProvider

        assert GPSignalProvider is not None

    def test_adapters_store_cache(self) -> None:
        from liq.evolution.adapters.store_cache import EvolutionStoreCache

        assert EvolutionStoreCache is not None

    def test_cli_main(self) -> None:
        from liq.evolution.cli.main import main

        assert callable(main)


class TestPublicAPIReexports:
    """Verify the public API re-exports in __init__.py."""

    def test_all_exports_defined(self) -> None:
        import liq.evolution

        for name in liq.evolution.__all__:
            assert hasattr(liq.evolution, name), f"{name} not exported"

    def test_config_exports(self) -> None:
        from liq.evolution import (
            EvolutionConfig,
            FitnessConfig,
            FitnessStageConfig,
            GPConfig,
            ParallelConfig,
            PrimitiveConfig,
            SerializationConfig,
            WarmStartConfig,
        )

        assert EvolutionConfig is not None
        assert PrimitiveConfig is not None
        assert FitnessStageConfig is not None
        assert ParallelConfig is not None
        assert WarmStartConfig is not None
        assert GPConfig is not None
        assert FitnessConfig is not None
        assert SerializationConfig is not None

    def test_protocol_exports(self) -> None:
        from liq.evolution import (
            FitnessEvaluator,
            FitnessStageEvaluator,
            GPStrategy,
            IndicatorBackend,
            PrimitiveRegistry,
        )

        assert IndicatorBackend is not None
        assert GPStrategy is not None
        assert PrimitiveRegistry is not None
        assert FitnessEvaluator is not None
        assert FitnessStageEvaluator is not None

    def test_error_exports_and_version(self) -> None:
        from liq.evolution import (
            AdapterError,
            ConfigurationError,
            FitnessEvaluationError,
            LiqEvolutionError,
            ParallelExecutionError,
            PrimitiveSetupError,
            __version__,
        )

        assert LiqEvolutionError is not None
        assert PrimitiveSetupError is not None
        assert FitnessEvaluationError is not None
        assert AdapterError is not None
        assert ConfigurationError is not None
        assert ParallelExecutionError is not None
        assert isinstance(__version__, str)

    def test_phase4_genome_exports(self) -> None:
        from liq.evolution import (
            Genome,
            deserialize_genome,
            serialize_genome,
        )

        assert Genome is not None
        assert callable(serialize_genome)
        assert callable(deserialize_genome)

    def test_phase4_feature_context_export(self) -> None:
        from liq.evolution import FeatureContext

        assert FeatureContext is not None

    def test_phase4_store_backend_export(self) -> None:
        from liq.evolution import StoreBackend

        assert StoreBackend is not None
