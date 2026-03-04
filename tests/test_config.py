"""Tests for configuration models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from liq.evolution.config import (
    EvolutionConfig,
    EvolutionRunConfig,
    FitnessConfig,
    FitnessStageConfig,
    GPConfig,
    RegimeGateConfig,
    ParallelConfig,
    PrimitiveConfig,
    SerializationConfig,
    WarmStartConfig,
)
from liq.evolution.errors import ConfigurationError
from liq.gp.config import SeedInjectionConfig


class TestPrimitiveConfigDefaults:
    """Verify PrimitiveConfig default values and immutability."""

    def test_defaults(self) -> None:
        cfg = PrimitiveConfig()
        assert cfg.enable_numeric_ops is True
        assert cfg.enable_comparison_ops is True
        assert cfg.enable_logic_ops is True
        assert cfg.enable_crossover_ops is True
        assert cfg.enable_temporal_ops is True
        assert cfg.enable_series_sources is True

    def test_frozen(self) -> None:
        cfg = PrimitiveConfig()
        with pytest.raises(ValidationError):
            cfg.enable_numeric_ops = False  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = PrimitiveConfig(enable_numeric_ops=False)
        assert cfg.enable_numeric_ops is False


class TestFitnessStageConfigDefaults:
    """Verify FitnessStageConfig default values and validation."""

    def test_defaults(self) -> None:
        cfg = FitnessStageConfig()
        assert cfg.label_metric == "f1"
        assert cfg.label_top_k == 0.1
        assert cfg.use_backtest is False
        assert cfg.backtest_top_n == 10
        assert cfg.backtest_metric == "sharpe_ratio"

    def test_frozen(self) -> None:
        cfg = FitnessStageConfig()
        with pytest.raises(ValidationError):
            cfg.label_metric = "accuracy"  # type: ignore[misc]

    def test_label_top_k_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="label_top_k"):
            FitnessStageConfig(label_top_k=0.0)

    def test_label_top_k_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="label_top_k"):
            FitnessStageConfig(label_top_k=-0.5)

    def test_label_top_k_above_one_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="label_top_k"):
            FitnessStageConfig(label_top_k=1.5)

    def test_label_top_k_one_accepted(self) -> None:
        cfg = FitnessStageConfig(label_top_k=1.0)
        assert cfg.label_top_k == 1.0

    def test_backtest_top_n_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="backtest_top_n"):
            FitnessStageConfig(backtest_top_n=0)


class TestParallelConfigDefaults:
    """Verify ParallelConfig default values and validation."""

    def test_defaults(self) -> None:
        cfg = ParallelConfig()
        assert cfg.backend == "sequential"
        assert cfg.max_workers == 1
        assert cfg.memory_limit_mb == 2048

    def test_frozen(self) -> None:
        cfg = ParallelConfig()
        with pytest.raises(ValidationError):
            cfg.max_workers = 4  # type: ignore[misc]

    def test_max_workers_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_workers"):
            ParallelConfig(max_workers=0)

    def test_memory_limit_too_low_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="memory_limit_mb"):
            ParallelConfig(memory_limit_mb=64)

    def test_memory_limit_boundary_accepted(self) -> None:
        cfg = ParallelConfig(memory_limit_mb=128, memory_warn_threshold_mb=64)
        assert cfg.memory_limit_mb == 128


class TestWarmStartConfigDefaults:
    """Verify WarmStartConfig default values."""

    def test_defaults(self) -> None:
        cfg = WarmStartConfig()
        assert cfg.seed_programs_path is None
        assert cfg.mode == "replace"

    def test_frozen(self) -> None:
        cfg = WarmStartConfig()
        with pytest.raises(ValidationError):
            cfg.mode = "augment"  # type: ignore[misc]

    def test_custom_path(self) -> None:
        cfg = WarmStartConfig(seed_programs_path=Path("/tmp/seeds.json"))
        assert cfg.seed_programs_path == Path("/tmp/seeds.json")


class TestEvolutionConfigDefaults:
    """Verify EvolutionConfig default values and validation."""

    def test_defaults(self) -> None:
        cfg = EvolutionConfig()
        assert cfg.population_size == 300
        assert cfg.max_depth == 8
        assert cfg.generations == 50
        assert cfg.seed == 42
        assert isinstance(cfg.primitives, PrimitiveConfig)
        assert isinstance(cfg.fitness_stages, FitnessStageConfig)
        assert isinstance(cfg.parallel, ParallelConfig)
        assert isinstance(cfg.warm_start, WarmStartConfig)

    def test_frozen(self) -> None:
        cfg = EvolutionConfig()
        with pytest.raises(ValidationError):
            cfg.population_size = 100  # type: ignore[misc]

    def test_population_size_too_small(self) -> None:
        with pytest.raises(ConfigurationError, match="population_size"):
            EvolutionConfig(population_size=5)

    def test_population_size_boundary_accepted(self) -> None:
        cfg = EvolutionConfig(population_size=10)
        assert cfg.population_size == 10

    def test_max_depth_too_small(self) -> None:
        with pytest.raises(ConfigurationError, match="max_depth"):
            EvolutionConfig(max_depth=1)

    def test_max_depth_boundary_accepted(self) -> None:
        cfg = EvolutionConfig(max_depth=2)
        assert cfg.max_depth == 2

    def test_generations_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="generations"):
            EvolutionConfig(generations=0)

    def test_nested_config_override(self) -> None:
        cfg = EvolutionConfig(
            primitives=PrimitiveConfig(),
            parallel=ParallelConfig(max_workers=4),
            gp=GPConfig(mutation_rate=0.4, crossover_rate=0.5),
        )
        assert cfg.parallel.max_workers == 4
        assert cfg.gp.mutation_rate == 0.4
        assert cfg.gp.crossover_rate == 0.5

    def test_nested_gp_field_conflict_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="must match"):
            EvolutionConfig(
                population_size=300,
                gp=GPConfig(population_size=500),
            )

    def test_run_config_defaults_embedded(self) -> None:
        cfg = EvolutionConfig()
        assert cfg.run.protocol_version == "1.0"
        assert cfg.run.stage_a_candidate_budget > 0


class TestEvolutionRunConfig:
    """Verify stage-0 stage budget contract enforcement."""

    def test_defaults(self) -> None:
        cfg = EvolutionRunConfig()
        assert cfg.protocol_version == "1.0"
        assert cfg.stage_a_candidate_budget == 5000
        assert cfg.stage_b_candidate_budget == 500
        assert cfg.stage_a_threshold == 0.8

    def test_frozen(self) -> None:
        cfg = EvolutionRunConfig()
        with pytest.raises(ValidationError):
            cfg.stage_a_candidate_budget = 5  # type: ignore[misc]

    def test_invalid_threshold_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="stage_a_threshold"):
            EvolutionRunConfig(stage_a_threshold=0.0)

    def test_stage_a_candidate_min_validation(self) -> None:
        with pytest.raises(ConfigurationError, match="must be >= 1"):
            EvolutionRunConfig(stage_a_candidate_budget=0)

    def test_stage_a_min_candidates_rejects_nonpositive(self) -> None:
        with pytest.raises(ConfigurationError, match="stage_a_min_candidates must be >= 1"):
            EvolutionRunConfig(stage_a_min_candidates=0)

    def test_stage_b_min_candidates_rejects_nonpositive(self) -> None:
        with pytest.raises(ConfigurationError, match="stage_b_min_candidates must be >= 1"):
            EvolutionRunConfig(stage_b_min_candidates=0)

    def test_protocol_version_rejects_unknown_schema(self) -> None:
        with pytest.raises(ConfigurationError, match="protocol_version must currently"):
            EvolutionRunConfig(protocol_version="2.0")

    def test_min_candidates_cannot_exceed_budgets(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot exceed"):
            EvolutionRunConfig(stage_a_candidate_budget=10, stage_a_min_candidates=11)

    def test_stage_b_candidate_budget_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="stage_b_candidate_budget"):
            EvolutionRunConfig(stage_b_candidate_budget=0)

    def test_stage_b_min_candidates_exceeds_budget(self) -> None:
        with pytest.raises(ConfigurationError, match="cannot exceed"):
            EvolutionRunConfig(stage_b_candidate_budget=5, stage_b_min_candidates=6)


class TestGPConfig:
    """Verify GPConfig default values and validation."""

    def test_defaults(self) -> None:
        cfg = GPConfig()
        assert cfg.population_size == 300
        assert cfg.max_depth == 8
        assert cfg.generations == 50
        assert cfg.mutation_rate == 0.2
        assert cfg.crossover_rate == 0.8
        assert cfg.tournament_size == 3
        assert cfg.elitism_count == 1
        assert cfg.seed == 42
        assert cfg.seed_injection is None

    def test_seed_injection_roundtrip(self) -> None:
        injection = SeedInjectionConfig(interval=5, count=2, method="direct")
        cfg = GPConfig(seed_injection=injection)
        assert cfg.seed_injection == injection

    def test_frozen(self) -> None:
        cfg = GPConfig()
        with pytest.raises(ValidationError):
            cfg.population_size = 100  # type: ignore[misc]

    def test_population_size_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="population_size"):
            GPConfig(population_size=0)

    def test_population_size_boundary_accepted(self) -> None:
        cfg = GPConfig(population_size=10)
        assert cfg.population_size == 10

    def test_max_depth_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_depth"):
            GPConfig(max_depth=0)

    def test_max_depth_boundary_accepted(self) -> None:
        cfg = GPConfig(max_depth=2)
        assert cfg.max_depth == 2

    def test_generations_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="generations"):
            GPConfig(generations=0)

    def test_mutation_rate_out_of_range(self) -> None:
        with pytest.raises(ConfigurationError, match="mutation_rate"):
            GPConfig(mutation_rate=1.5)

    def test_mutation_rate_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="mutation_rate"):
            GPConfig(mutation_rate=-0.1)

    def test_crossover_rate_out_of_range(self) -> None:
        with pytest.raises(ConfigurationError, match="crossover_rate"):
            GPConfig(crossover_rate=1.5)

    def test_tournament_size_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="tournament_size"):
            GPConfig(tournament_size=0)

    def test_elitism_count_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="elitism_count"):
            GPConfig(elitism_count=-1)

    def test_mutation_and_crossover_total_greater_than_one_rejected(self) -> None:
        with pytest.raises(
            ConfigurationError,
            match=r"crossover_rate \+ mutation_rate must be <= 1.0",
        ):
            GPConfig(crossover_rate=0.8, mutation_rate=0.3)

    def test_constant_opt_defaults(self) -> None:
        cfg = GPConfig()
        assert cfg.constant_opt_enabled is True
        assert cfg.constant_opt_top_k == 0.1
        assert cfg.constant_opt_max_iter == 50
        assert cfg.constant_opt_max_time_seconds == 1.0

    def test_simplification_and_dedup_defaults(self) -> None:
        cfg = GPConfig()
        assert cfg.simplification_enabled is True

    def test_constant_opt_disabled(self) -> None:
        cfg = GPConfig(constant_opt_enabled=False)
        assert cfg.constant_opt_enabled is False

    def test_constant_opt_top_k_out_of_range_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_top_k"):
            GPConfig(constant_opt_top_k=0.0)

    def test_constant_opt_top_k_above_one_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_top_k"):
            GPConfig(constant_opt_top_k=1.5)

    def test_constant_opt_max_time_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_max_time_seconds"):
            GPConfig(constant_opt_max_time_seconds=0.0)

    def test_seed_injection_count_exceeds_replaceable_rejected(self) -> None:
        """seed_injection.count > population_size - elitism_count must fail."""
        with pytest.raises(ConfigurationError, match="seed_injection.count"):
            GPConfig(
                population_size=10,
                elitism_count=2,
                seed_injection=SeedInjectionConfig(count=9, interval=1),
            )

    def test_seed_injection_count_at_boundary_accepted(self) -> None:
        """seed_injection.count == population_size - elitism_count is valid."""
        cfg = GPConfig(
            population_size=10,
            elitism_count=2,
            seed_injection=SeedInjectionConfig(count=8, interval=1),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.count == 8

    def test_invalid_constant_opt_mode_rejected(self) -> None:
        with pytest.raises((ConfigurationError, ValidationError), match="constant_opt_mode"):
            GPConfig(constant_opt_mode="bad")

    def test_constant_opt_max_evals_rejects_nonpositive(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_max_evals"):
            GPConfig(constant_opt_max_evals=0)

    def test_constant_opt_mode_rejected_in_model_validator(self) -> None:
        cfg = GPConfig.model_construct(constant_opt_mode="bad")
        with pytest.raises(ConfigurationError, match="constant_opt_mode"):
            cfg._validate_gp_config()


class TestFitnessConfig:
    """Verify FitnessConfig default values and validation."""

    def test_defaults(self) -> None:
        cfg = FitnessConfig()
        assert cfg.metric == "f1"
        assert cfg.stage_a_metric == "f1"
        assert cfg.stage_b_enabled is False
        assert cfg.top_k_for_backtest == 10

    def test_frozen(self) -> None:
        cfg = FitnessConfig()
        with pytest.raises(ValidationError):
            cfg.metric = "accuracy"  # type: ignore[misc]

    def test_top_k_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="top_k_for_backtest"):
            FitnessConfig(top_k_for_backtest=0)

    def test_empty_objectives_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="objectives must be non-empty"):
            FitnessConfig(objectives=())

    def test_direction_count_mismatch_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="must equal"):
            FitnessConfig(
                objectives=("f1", "accuracy"),
                objective_directions=("maximize",),
            )

    def test_invalid_objective_direction_rejected(self) -> None:
        with pytest.raises(
            (ConfigurationError, ValidationError),
            match=r"'maximize' or 'minimize'",
        ):
            FitnessConfig(objectives=("f1",), objective_directions=("noop",))

    def test_objective_direction_rejected_in_model_validator(self) -> None:
        cfg = FitnessConfig.model_construct(objectives=("f1",), objective_directions=("noop",))
        with pytest.raises(ConfigurationError, match="objective_directions\\[0\\]"):
            cfg._validate_fitness_config()


class TestSerializationConfig:
    """Verify SerializationConfig default values."""

    def test_defaults(self) -> None:
        cfg = SerializationConfig()
        assert cfg.schema_version == "1.0"

    def test_frozen(self) -> None:
        cfg = SerializationConfig()
        with pytest.raises(ValidationError):
            cfg.schema_version = "2.0"  # type: ignore[misc]

    def test_custom_version(self) -> None:
        cfg = SerializationConfig(schema_version="2.0")
        assert cfg.schema_version == "2.0"


class TestRegimeGateConfig:
    """Verify RegimeGateConfig validation contract."""

    def test_regime_confidence_threshold_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="regime_confidence_threshold"):
            RegimeGateConfig(regime_confidence_threshold=1.25)

    def test_regime_occupancy_threshold_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="regime_occupancy_threshold"):
            RegimeGateConfig(regime_occupancy_threshold=-0.01)

    def test_regime_persistence_invalid(self) -> None:
        with pytest.raises(ConfigurationError, match="regime_min_persistence"):
            RegimeGateConfig(regime_min_persistence=0)

    def test_regime_hysteresis_margin_invalid(self) -> None:
        with pytest.raises(ConfigurationError, match="regime_hysteresis_margin"):
            RegimeGateConfig(regime_hysteresis_margin=-0.5)
