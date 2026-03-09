"""Tests for config bridge (EvolutionConfig -> liq.gp.GPConfig)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from liq.evolution.config import (
    EvolutionConfig,
    GPConfig,
    ParallelConfig,
    build_gp_config,
    map_evolution_pressure_to_scheduler_policy,
)
from liq.gp.config import GPConfig as LiqGPConfig
from liq.gp.config import SeedInjectionConfig


class TestBuildGPConfig:
    def test_default_mapping(self) -> None:
        evo = EvolutionConfig()
        gp = build_gp_config(evo)
        assert isinstance(gp, LiqGPConfig)
        assert gp.population_size == 300
        assert gp.max_depth == 8
        assert gp.generations == 50
        assert gp.seed == 42

    def test_custom_values(self) -> None:
        evo = EvolutionConfig(population_size=100, max_depth=5, generations=20, seed=99)
        gp = build_gp_config(evo)
        assert gp.population_size == 100
        assert gp.max_depth == 5
        assert gp.generations == 20
        assert gp.seed == 99

    def test_custom_gp_rates(self) -> None:
        evo = EvolutionConfig(
            population_size=10,
            gp=GPConfig(crossover_rate=0.5, mutation_rate=0.5),
        )
        gp = build_gp_config(evo)
        assert gp.crossover_rate == 0.5
        assert gp.subtree_mutation_rate == 0.2
        assert gp.point_mutation_rate == 0.15
        assert gp.parameter_mutation_rate == 0.1
        assert gp.hoist_mutation_rate == 0.05

    def test_operator_rates_sum_to_one(self) -> None:
        evo = EvolutionConfig()
        gp = build_gp_config(evo)
        import math

        rate_sum = (
            gp.crossover_rate
            + gp.subtree_mutation_rate
            + gp.point_mutation_rate
            + gp.parameter_mutation_rate
            + gp.hoist_mutation_rate
        )
        assert math.isclose(rate_sum, 1.0, abs_tol=1e-9)

    def test_returns_frozen_model(self) -> None:
        evo = EvolutionConfig()
        gp = build_gp_config(evo)
        with pytest.raises(ValidationError):
            cast(Any, gp).population_size = 999

    def test_fitness_objectives_default(self) -> None:
        evo = EvolutionConfig()
        gp = build_gp_config(evo)
        assert gp.fitness.objectives == ["fitness"]
        assert gp.fitness.objective_directions == ["maximize"]

    def test_constant_opt_fields_forwarded(self) -> None:
        evo = EvolutionConfig(
            gp=GPConfig(
                constant_opt_enabled=True,
                constant_opt_top_k=0.2,
                constant_opt_max_iter=100,
                constant_opt_max_time_seconds=2.0,
            ),
        )
        gp = build_gp_config(evo)
        assert gp.constant_opt_enabled is True
        assert gp.constant_opt_top_k == 0.2
        assert gp.constant_opt_max_iter == 100
        assert gp.constant_opt_max_time_seconds == 2.0

    def test_constant_opt_disabled_forwarded(self) -> None:
        evo = EvolutionConfig(
            gp=GPConfig(constant_opt_enabled=False),
        )
        gp = build_gp_config(evo)
        assert gp.constant_opt_enabled is False

    def test_simplification_forwarded(self) -> None:
        evo = EvolutionConfig(
            gp=GPConfig(simplification_enabled=False),
        )
        gp = build_gp_config(evo)
        assert gp.simplification_enabled is False

    def test_seed_injection_forwarded(self) -> None:
        injection = SeedInjectionConfig(interval=3, count=2, method="variation")
        evo = EvolutionConfig(gp=GPConfig(seed_injection=injection))
        gp = build_gp_config(evo)
        assert gp.seed_injection == injection

    def test_scheduler_policy_is_wired(self) -> None:
        evo = EvolutionConfig()
        gp = build_gp_config(evo)
        assert gp.scheduler.enabled is True
        assert gp.scheduler.max_in_flight == evo.parallel.max_in_flight
        assert gp.scheduler.memory_budget_mb == evo.parallel.memory_limit_mb

    def test_policy_mapping_preserves_safety_invariants(self) -> None:
        evo = EvolutionConfig()
        scheduler, meta = map_evolution_pressure_to_scheduler_policy(
            evo,
            constraint_violations={"max_leverage": 0.3, "negative_cash": 0.2},
        )
        assert scheduler.max_in_flight >= 1
        assert scheduler.queue_capacity >= scheduler.max_in_flight
        assert scheduler.eval_batch_size >= 1
        assert scheduler.memory_budget_mb >= 128
        assert scheduler.safe_fallback_mode in {"sequential", "fail"}
        assert meta["reason_code"] in {"ok", "constraint_saturation", "critical_constraint_saturation"}

    def test_policy_mapping_saturates_to_safe_fallback(self) -> None:
        evo = EvolutionConfig(
            parallel=ParallelConfig(
                backend="ray",
                max_workers=2,
                max_in_flight=4,
                memory_limit_mb=2048,
                memory_warn_threshold_mb=1536,
                auto_fallback=False,
            )
        )
        scheduler, meta = map_evolution_pressure_to_scheduler_policy(
            evo,
            constraint_violations={"future_reference": 1.0},
        )
        assert scheduler.max_in_flight == 1
        assert scheduler.queue_capacity == 1
        assert scheduler.safe_fallback_mode == "sequential"
        assert meta["reason_code"] == "critical_constraint_saturation"
