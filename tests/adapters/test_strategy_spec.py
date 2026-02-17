"""Tests for GPStrategy export/import spec round-trip."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
from liq.evolution.errors import AdapterError, SerializationError
from liq.gp.config import GPConfig
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import TerminalNode
from liq.gp.types import EvolutionResult, GenerationStats, Series


def _make_terminal(name: str = "close") -> TerminalNode:
    return TerminalNode(name=name, output_type=Series)


def _make_evolution_result(program=None):
    if program is None:
        program = _make_terminal()
    return EvolutionResult(
        best_program=program,
        pareto_front=[program],
        fitness_history=[
            GenerationStats(
                generation=0,
                best_fitness=(0.5,),
                mean_fitness=(0.3,),
                best_program_size=1,
                mean_program_size=1.0,
                unique_semantics_ratio=1.0,
                pareto_front_size=1,
            )
        ],
        config=None,
    )


# ------------------------------------------------------------------ #
#  export_spec
# ------------------------------------------------------------------ #


class TestExportSpec:
    def test_export_before_fit_raises(self) -> None:
        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        with pytest.raises(AdapterError, match="export_spec.*before fit"):
            adapter.export_spec()

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_export_contains_schema_version(self, mock_evolve) -> None:
        mock_evolve.return_value = _make_evolution_result()
        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter.fit(pl.DataFrame({"close": np.zeros(20)}))
        spec = adapter.export_spec()
        assert spec["schema_version"] == "1.0"

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_export_contains_program(self, mock_evolve) -> None:
        mock_evolve.return_value = _make_evolution_result()
        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter.fit(pl.DataFrame({"close": np.zeros(20)}))
        spec = adapter.export_spec()
        assert "program" in spec
        assert isinstance(spec["program"], dict)

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_export_contains_config(self, mock_evolve) -> None:
        mock_evolve.return_value = _make_evolution_result()
        config = GPConfig(population_size=20, max_depth=4, generations=2)
        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            config,
            MagicMock(),
        )
        adapter.fit(pl.DataFrame({"close": np.zeros(20)}))
        spec = adapter.export_spec()
        assert "config" in spec
        assert spec["config"]["population_size"] == 20


# ------------------------------------------------------------------ #
#  from_spec
# ------------------------------------------------------------------ #


class TestFromSpec:
    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_roundtrip_predict_matches(self, mock_evolve) -> None:
        """export → from_spec → predict produces same scores."""
        program = _make_terminal("close")
        mock_evolve.return_value = _make_evolution_result(program)

        registry = PrimitiveRegistry()
        registry.register("close", lambda: None, output_type=Series)

        adapter = GPStrategyAdapter(
            registry,
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter.fit(pl.DataFrame({"close": [1.0, 2.0, 3.0]}))

        spec = adapter.export_spec()
        restored = GPStrategyAdapter.from_spec(spec, registry)

        df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        original = adapter.predict(df)
        restored_out = restored.predict(df)
        np.testing.assert_array_equal(
            original.scores.to_numpy(), restored_out.scores.to_numpy()
        )

    def test_from_spec_creates_predict_only(self) -> None:
        """from_spec adapter has program set."""
        registry = PrimitiveRegistry()
        registry.register("close", lambda: None, output_type=Series)

        from liq.gp.program.serialize import serialize

        program = _make_terminal("close")
        spec = {
            "schema_version": "1.0",
            "program": serialize(program),
            "config": GPConfig(
                population_size=20, max_depth=4, generations=2
            ).model_dump(),
        }
        restored = GPStrategyAdapter.from_spec(spec, registry)
        assert restored.program is not None

    def test_from_spec_malformed_raises(self) -> None:
        with pytest.raises(AdapterError, match="[Mm]alformed"):
            GPStrategyAdapter.from_spec({}, PrimitiveRegistry())

    def test_from_spec_missing_program_raises(self) -> None:
        with pytest.raises(AdapterError, match="[Mm]alformed"):
            GPStrategyAdapter.from_spec({"schema_version": "1.0"}, PrimitiveRegistry())

    def test_from_spec_unsupported_schema_version_raises(self) -> None:
        registry = PrimitiveRegistry()
        registry.register("close", lambda: None, output_type=Series)

        from liq.gp.program.serialize import serialize

        program = _make_terminal("close")
        spec = {
            "schema_version": "0.9",
            "program": serialize(program),
        }
        with pytest.raises(SerializationError, match="Unsupported schema_version"):
            GPStrategyAdapter.from_spec(spec, registry)

    def test_from_spec_fit_raises(self) -> None:
        """Predict-only adapter cannot fit."""
        registry = PrimitiveRegistry()
        registry.register("close", lambda: None, output_type=Series)

        from liq.gp.program.serialize import serialize

        program = _make_terminal("close")
        spec = {
            "schema_version": "1.0",
            "program": serialize(program),
            "config": GPConfig(
                population_size=20, max_depth=4, generations=2
            ).model_dump(),
        }
        restored = GPStrategyAdapter.from_spec(spec, registry)
        with pytest.raises(AdapterError, match="predict-only"):
            restored.fit(pl.DataFrame({"close": [1.0]}))

    def test_from_spec_without_config_key(self) -> None:
        """Works when config key is missing (uses defaults)."""
        registry = PrimitiveRegistry()
        registry.register("close", lambda: None, output_type=Series)

        from liq.gp.program.serialize import serialize

        program = _make_terminal("close")
        spec = {
            "schema_version": "1.0",
            "program": serialize(program),
        }
        restored = GPStrategyAdapter.from_spec(spec, registry)
        assert restored.program is not None

    def test_from_spec_invalid_config_raises(self) -> None:
        registry = PrimitiveRegistry()
        registry.register("close", lambda: None, output_type=Series)

        from liq.gp.program.serialize import serialize

        program = _make_terminal("close")
        spec = {
            "schema_version": "1.0",
            "program": serialize(program),
            "config": {"population_size": "invalid"},
        }
        with pytest.raises(SerializationError, match="Failed to parse config"):
            GPStrategyAdapter.from_spec(spec, registry)
