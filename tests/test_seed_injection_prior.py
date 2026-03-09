"""Stage-15 tests for deterministic seed-template priors and injection policy."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from liq.evolution.config import PrimitiveConfig
from liq.evolution.errors import ConfigurationError
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.seeds import (
    SeedInjectionCadence,
    SeedTemplateRole,
    StrategySeedTemplate,
    build_seed_champion_payload,
    build_strategy_seed,
    get_seed_template,
    inject_seed_programs_from_champion_pool,
    list_seed_templates_by_role,
    program_signature,
    select_seed_champion_pool,
    should_inject,
    validate_external_seed_payload,
)
from liq.gp.config import GPConfig
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program, TerminalNode
from liq.gp.program.serialize import serialize
from liq.gp.types import BoolSeries, FitnessResult


def _dummy_builder(_registry: PrimitiveRegistry):
    return TerminalNode(name="close", output_type=BoolSeries)


def _terminal_names(programs: list[Program]) -> list[str]:
    assert all(isinstance(program, TerminalNode) for program in programs)
    return [cast(TerminalNode, program).name for program in programs]


def _fitnesses(size: int) -> list[FitnessResult]:
    return [FitnessResult(objectives=(float(i),), metadata={}) for i in range(size)]


class TestSeedTemplateSchemaStage15:
    def test_template_roles_and_metadata_present(self) -> None:
        detector = get_seed_template("atr_volatility_spike")
        expert = get_seed_template("ema_crossover")
        risk = get_seed_template("carry_spread_expansion")

        assert detector.block_role is SeedTemplateRole.detector
        assert expert.block_role is SeedTemplateRole.expert
        assert risk.block_role is SeedTemplateRole.risk
        assert risk.arity == len(risk.expected_inputs)
        assert risk.regime_hints
        assert risk.failure_modes
        assert risk.turnover_expectation is not None

        assert "atr_volatility_spike" in list_seed_templates_by_role("detector")
        assert "ema_crossover" in list_seed_templates_by_role(SeedTemplateRole.expert)
        assert "carry_spread_expansion" in list_seed_templates_by_role("risk")

    def test_template_invalid_metadata_rejected(self) -> None:
        with pytest.raises(ValueError, match="template block_role must be one of"):
            StrategySeedTemplate(
                name="invalid-role",
                description="bad",
                builder=_dummy_builder,
                block_role=cast(Any, "unknown"),
            )

        with pytest.raises(
            ValueError,
            match="template arity must match expected_inputs length",
        ):
            StrategySeedTemplate(
                name="invalid-arity",
                description="bad",
                builder=_dummy_builder,
                arity=2,
                expected_inputs=("close",),
            )


class TestSeedInjectionCadenceStage15:
    def test_cadence_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="interval"):
            SeedInjectionCadence(interval=0)
        with pytest.raises(ValueError, match="champion_pool_size"):
            SeedInjectionCadence(champion_pool_size=0)
        with pytest.raises(ValueError, match="injection_count"):
            SeedInjectionCadence(injection_count=0)

    def test_should_inject_is_deterministic(self) -> None:
        cadence = SeedInjectionCadence(
            interval=3,
            champion_pool_size=4,
            injection_count=2,
            method="direct",
        )
        expected = [
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            True,
        ]
        actual = [should_inject(generation, cadence) for generation in range(10)]
        assert actual == expected

    def test_should_inject_rejects_negative_generation(self) -> None:
        with pytest.raises(ValueError, match="generation must be >= 0"):
            should_inject(-1, SeedInjectionCadence())

    def test_select_champion_pool_deduplicates_and_orders_by_objective(self) -> None:
        alpha = TerminalNode(name="alpha", output_type=BoolSeries)
        beta = TerminalNode(name="beta", output_type=BoolSeries)
        gamma = TerminalNode(name="gamma", output_type=BoolSeries)
        population = [alpha, beta, beta, gamma]
        fitnesses = [
            FitnessResult(objectives=(0.2,), metadata={}),
            FitnessResult(objectives=(0.8,), metadata={}),
            FitnessResult(objectives=(0.7,), metadata={}),
            FitnessResult(objectives=(0.9,), metadata={}),
        ]

        champions = select_seed_champion_pool(population, fitnesses, pool_size=3)
        assert _terminal_names(champions) == ["gamma", "beta", "alpha"]
        assert len({program_signature(p) for p in champions}) == len(champions)

    def test_select_champion_pool_validation_and_minimize_order(self) -> None:
        alpha = TerminalNode(name="alpha", output_type=BoolSeries)
        beta = TerminalNode(name="beta", output_type=BoolSeries)
        population = [alpha, beta]
        fitnesses = [
            FitnessResult(objectives=(0.9,), metadata={}),
            FitnessResult(objectives=(0.1,), metadata={}),
        ]
        champions = select_seed_champion_pool(
            population,
            fitnesses,
            pool_size=2,
            objective_direction="minimize",
        )
        assert _terminal_names(champions) == ["beta", "alpha"]

        assert select_seed_champion_pool([], [], pool_size=2) == []

        with pytest.raises(ValueError, match="pool_size must be >= 1"):
            select_seed_champion_pool(population, fitnesses, pool_size=0)
        with pytest.raises(ValueError, match="objective_index must be >= 0"):
            select_seed_champion_pool(population, fitnesses, pool_size=1, objective_index=-1)
        with pytest.raises(ValueError, match="matching lengths"):
            select_seed_champion_pool(population, fitnesses[:1], pool_size=1)
        with pytest.raises(ValueError, match="objective 1"):
            select_seed_champion_pool(population, fitnesses, pool_size=1, objective_index=1)
        with pytest.raises(ValueError, match="objective_direction"):
            select_seed_champion_pool(
                population,
                fitnesses,
                pool_size=1,
                objective_direction="sideways",  # type: ignore[arg-type]
            )


class TestExternalSeedPayloadStage15:
    def test_payload_validation_strict_and_non_strict(self) -> None:
        registry = PrimitiveRegistry()
        seed_a = TerminalNode(name="seed_a", output_type=BoolSeries)
        seed_b = TerminalNode(name="seed_b", output_type=BoolSeries)
        payload = {
            "seed_programs": [
                serialize(seed_a),
                {"program": serialize(seed_b)},
                {"program": "bad"},
            ]
        }

        programs = validate_external_seed_payload(payload, registry, strict=False)
        assert _terminal_names(programs) == ["seed_a", "seed_b"]

        with pytest.raises(
            ConfigurationError,
            match="invalid external seed entry at index 2",
        ):
            validate_external_seed_payload(payload, registry, strict=True)

        assert (
            validate_external_seed_payload({"unexpected": [1]}, registry, strict=False)
            == []
        )

    def test_payload_parsing_from_json_strings_and_paths(self, tmp_path) -> None:
        registry = PrimitiveRegistry()
        seed = TerminalNode(name="seed_file", output_type=BoolSeries)
        payload = build_seed_champion_payload([seed], source="external")

        path = tmp_path / "seed_payload.json"
        path.write_text(str(payload).replace("'", '"'), encoding="utf-8")

        from_path = validate_external_seed_payload(path, registry, strict=True)
        assert _terminal_names(from_path) == ["seed_file"]

        from_path_string = validate_external_seed_payload(
            str(path),
            registry,
            strict=True,
        )
        assert _terminal_names(from_path_string) == ["seed_file"]

        from_json_string = validate_external_seed_payload(
            str(payload).replace("'", '"'),
            registry,
            strict=True,
        )
        assert _terminal_names(from_json_string) == ["seed_file"]

    def test_payload_validation_error_branches(self, tmp_path) -> None:
        registry = PrimitiveRegistry()
        seed = TerminalNode(name="seed", output_type=BoolSeries)
        seed_payload = serialize(seed)

        with pytest.raises(ConfigurationError, match="Unsupported external seed payload"):
            validate_external_seed_payload({"unknown": 1}, registry, strict=True)

        with pytest.raises(ConfigurationError, match="'seed_programs' must be a list"):
            validate_external_seed_payload({"seed_programs": "bad"}, registry, strict=True)

        with pytest.raises(ConfigurationError, match="'pareto_front' must be a list"):
            validate_external_seed_payload({"pareto_front": "bad"}, registry, strict=True)

        with pytest.raises(
            ConfigurationError,
            match="invalid external seed entry at index 0",
        ):
            validate_external_seed_payload([1], registry, strict=True)

        with pytest.raises(ConfigurationError, match="invalid external seed entry at index 0"):
            validate_external_seed_payload([{"program": "bad"}], registry, strict=True)

        missing = tmp_path / "missing.json"
        with pytest.raises(ConfigurationError, match="Failed to read external seed payload"):
            validate_external_seed_payload(missing, registry, strict=True)
        assert validate_external_seed_payload(missing, registry, strict=False) == []

        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{not-json", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid JSON external seed payload"):
            validate_external_seed_payload(invalid_json, registry, strict=True)

        with pytest.raises(ConfigurationError, match="Invalid JSON external seed payload string"):
            validate_external_seed_payload("{bad-json", registry, strict=True)
        assert validate_external_seed_payload("{bad-json", registry, strict=False) == []

        assert _terminal_names(validate_external_seed_payload(
            {"best_program": seed_payload},
            registry,
            strict=True,
        )) == ["seed"]
        assert _terminal_names(validate_external_seed_payload(
            {"entry_program": seed_payload},
            registry,
            strict=True,
        )) == ["seed"]
        assert _terminal_names(validate_external_seed_payload(
            {"pareto_front": [seed_payload]},
            registry,
            strict=True,
        )) == ["seed"]
        assert _terminal_names(validate_external_seed_payload(
            {"program": seed_payload},
            registry,
            strict=True,
        )) == ["seed"]

        with pytest.raises(ConfigurationError, match="Unsupported external seed payload"):
            validate_external_seed_payload(123, registry, strict=True)


class TestChampionInjectionStage15:
    def test_direct_injection_increases_population_diversity(self) -> None:
        base = TerminalNode(name="base", output_type=BoolSeries)
        external_1 = TerminalNode(name="external_1", output_type=BoolSeries)
        external_2 = TerminalNode(name="external_2", output_type=BoolSeries)
        population = [base] * 10
        fitnesses = _fitnesses(len(population))
        cadence = SeedInjectionCadence(
            interval=2,
            champion_pool_size=4,
            injection_count=3,
            method="direct",
        )
        gp_config = GPConfig(
            population_size=10,
            max_depth=4,
            generations=2,
            elitism_count=0,
        )
        registry = PrimitiveRegistry()
        rng = np.random.default_rng(7)
        external_payload = build_seed_champion_payload(
            [external_1, external_2],
            source="external",
        )

        no_inject_population, no_inject_count = inject_seed_programs_from_champion_pool(
            population,
            fitnesses,
            generation=1,
            cadence=cadence,
            gp_config=gp_config,
            registry=registry,
            rng=rng,
            external_seed_payload=external_payload,
        )
        assert no_inject_count == 0
        assert len({program_signature(p) for p in no_inject_population}) == 1

        injected_population, injected_count = inject_seed_programs_from_champion_pool(
            population,
            fitnesses,
            generation=2,
            cadence=cadence,
            gp_config=gp_config,
            registry=registry,
            rng=rng,
            external_seed_payload=external_payload,
        )
        assert injected_count == 3
        assert len({program_signature(p) for p in injected_population}) > 1

    def test_variation_injection_produces_diversity_from_seed_pool(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        seed_a = build_strategy_seed("ema_crossover", registry)
        seed_b = build_strategy_seed("rsi_oversold", registry)
        population = [seed_a] * 10
        fitnesses = _fitnesses(len(population))
        cadence = SeedInjectionCadence(
            interval=1,
            champion_pool_size=4,
            injection_count=2,
            method="variation",
        )
        gp_config = GPConfig(
            population_size=10,
            max_depth=5,
            generations=2,
            elitism_count=0,
        )
        rng = np.random.default_rng(123)
        external_payload = build_seed_champion_payload([seed_b], source="external")

        injected_population, injected_count = inject_seed_programs_from_champion_pool(
            population,
            fitnesses,
            generation=1,
            cadence=cadence,
            gp_config=gp_config,
            registry=registry,
            rng=rng,
            external_seed_payload=external_payload,
        )
        assert injected_count == 2
        assert len({program_signature(p) for p in injected_population}) > 1
        assert all(program.output_type is BoolSeries for program in injected_population)

    def test_injection_validation_and_edge_paths(self) -> None:
        registry = build_trading_registry(PrimitiveConfig())
        base = build_strategy_seed("ema_crossover", registry)
        population = [base] * 10
        fitnesses = _fitnesses(10)
        gp_config = GPConfig(
            population_size=10,
            max_depth=4,
            generations=2,
            elitism_count=0,
        )
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="matching lengths"):
            inject_seed_programs_from_champion_pool(
                population,
                fitnesses[:1],
                generation=1,
                cadence=SeedInjectionCadence(),
                gp_config=gp_config,
                registry=registry,
                rng=rng,
            )

        blocked_population, blocked_count = inject_seed_programs_from_champion_pool(
            population,
            fitnesses,
            generation=1,
            cadence=SeedInjectionCadence(injection_count=2),
            gp_config=gp_config,
            registry=registry,
            rng=rng,
            elite_indices=set(range(len(population))),
        )
        assert blocked_count == 0
        assert blocked_population == population

        ramped_population, ramped_count = inject_seed_programs_from_champion_pool(
            population,
            fitnesses,
            generation=1,
            cadence=SeedInjectionCadence(method="ramped", injection_count=1),
            gp_config=gp_config,
            registry=registry,
            rng=rng,
            external_seed_payload={"unsupported": [1]},
            output_type=BoolSeries,
        )
        assert ramped_count == 1
        assert len(ramped_population) == len(population)

    def test_direct_injection_without_seed_sources_returns_unchanged(self) -> None:
        gp_config = GPConfig(
            population_size=10,
            max_depth=4,
            generations=2,
            elitism_count=0,
        )
        rng = np.random.default_rng(9)
        unchanged_population, unchanged_count = inject_seed_programs_from_champion_pool(
            [],
            [],
            generation=1,
            cadence=SeedInjectionCadence(method="direct", interval=1),
            gp_config=gp_config,
            registry=PrimitiveRegistry(),
            rng=rng,
        )
        assert unchanged_count == 0
        assert unchanged_population == []

    def test_duplicate_external_entries_are_deduplicated(self) -> None:
        base = TerminalNode(name="base", output_type=BoolSeries)
        population = [base] * 10
        fitnesses = _fitnesses(10)
        gp_config = GPConfig(
            population_size=10,
            max_depth=4,
            generations=2,
            elitism_count=0,
        )
        rng = np.random.default_rng(11)
        payload = build_seed_champion_payload([base, base], source="external")

        injected_population, injected_count = inject_seed_programs_from_champion_pool(
            population,
            fitnesses,
            generation=1,
            cadence=SeedInjectionCadence(interval=1, method="direct", injection_count=2),
            gp_config=gp_config,
            registry=PrimitiveRegistry(),
            rng=rng,
            external_seed_payload=payload,
        )
        assert injected_count == 2
        assert len({program_signature(p) for p in injected_population}) == 1
