"""Tests for GPStrategyAdapter fit/predict and GPSignalOutput."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from liq.evolution.adapters.signal_output import GPSignalOutput
from liq.evolution.config import ParallelConfig
from liq.evolution.errors import AdapterError
from liq.evolution.protocols import GPStrategy
from liq.gp.program.ast import TerminalNode
from liq.gp.types import EvolutionResult, GenerationStats, Series

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _make_terminal(name: str = "close") -> TerminalNode:
    return TerminalNode(name=name, output_type=Series)


def _make_evolution_result(program=None, config=None):
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
        config=config,
    )


def _make_dataframe(n: int = 20) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "close": rng.standard_normal(n),
            "volume": rng.uniform(1000, 5000, n),
        }
    )


# ------------------------------------------------------------------ #
#  GPSignalOutput
# ------------------------------------------------------------------ #


class TestGPSignalOutput:
    def test_construction_with_scores(self) -> None:
        scores = pl.Series("scores", [1.0, 2.0, 3.0])
        out = GPSignalOutput(scores=scores)
        assert out.scores is scores
        assert out.labels is None
        assert out.metadata == {}

    def test_construction_with_labels(self) -> None:
        scores = pl.Series("scores", [1.0, 2.0])
        labels = pl.Series("labels", [0.0, 1.0])
        out = GPSignalOutput(scores=scores, labels=labels)
        assert out.labels is labels

    def test_construction_with_metadata(self) -> None:
        scores = pl.Series("scores", [1.0])
        out = GPSignalOutput(scores=scores, metadata={"key": "val"})
        assert out.metadata == {"key": "val"}

    def test_frozen(self) -> None:
        out = GPSignalOutput(scores=pl.Series("s", [1.0]))
        with pytest.raises(AttributeError):
            out.scores = pl.Series("s", [2.0])  # type: ignore[misc]


# ------------------------------------------------------------------ #
#  _dataframe_to_context
# ------------------------------------------------------------------ #


class TestDataframeToContext:
    def test_single_column(self) -> None:
        from liq.evolution.adapters.runner_strategy import _dataframe_to_context

        df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
        ctx = _dataframe_to_context(df)
        assert "close" in ctx
        np.testing.assert_array_equal(ctx["close"], [1.0, 2.0, 3.0])

    def test_multiple_columns(self) -> None:
        from liq.evolution.adapters.runner_strategy import _dataframe_to_context

        df = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        ctx = _dataframe_to_context(df)
        assert set(ctx.keys()) == {"a", "b", "c"}

    def test_numpy_arrays(self) -> None:
        from liq.evolution.adapters.runner_strategy import _dataframe_to_context

        df = pl.DataFrame({"x": [1.0, 2.0]})
        ctx = _dataframe_to_context(df)
        assert isinstance(ctx["x"], np.ndarray)


# ------------------------------------------------------------------ #
#  GPStrategyAdapter construction
# ------------------------------------------------------------------ #


class TestGPStrategyAdapterConstruction:
    def test_valid_construction(self) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        registry = PrimitiveRegistry()
        config = GPConfig(population_size=20, max_depth=4, generations=2)
        evaluator = MagicMock()
        adapter = GPStrategyAdapter(registry, config, evaluator)
        assert adapter.program is None
        assert adapter.evolution_result is None

    def test_program_initially_none(self) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
        )
        assert adapter.program is None

    def test_predict_only_construction(self) -> None:
        """Adapter with evaluator=None is valid (predict-only)."""
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            evaluator=None,
        )
        assert adapter.program is None


# ------------------------------------------------------------------ #
#  GPStrategyAdapter predict
# ------------------------------------------------------------------ #


class TestGPStrategyAdapterPredict:
    def test_predict_before_fit_raises(self) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        df = _make_dataframe()
        with pytest.raises(AdapterError, match="predict.*before fit"):
            adapter.predict(df)

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_predict_returns_gp_signal_output(self, mock_eval) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        mock_eval.return_value = np.array([0.1, 0.2, 0.3])
        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter._program = _make_terminal()
        result = adapter.predict(pl.DataFrame({"close": [1.0, 2.0, 3.0]}))
        assert isinstance(result, GPSignalOutput)
        assert isinstance(result.scores, pl.Series)
        assert result.scores.name == "scores"
        assert len(result.scores) == 3

    @patch("liq.evolution.adapters.runner_strategy.gp_evaluate")
    def test_predict_scores_dtype(self, mock_eval) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        mock_eval.return_value = np.array([1.0, 2.0], dtype=np.float64)
        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter._program = _make_terminal()
        result = adapter.predict(pl.DataFrame({"close": [1.0, 2.0]}))
        assert result.scores.dtype == pl.Float64


# ------------------------------------------------------------------ #
#  GPStrategyAdapter fit
# ------------------------------------------------------------------ #


class TestGPStrategyAdapterFit:
    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_fit_stores_program(self, mock_evolve) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        program = _make_terminal()
        mock_evolve.return_value = _make_evolution_result(program)

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter.fit(_make_dataframe())
        assert adapter.program is program

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_fit_with_parallel_config_wraps_evaluator(self, mock_evolve) -> None:
        from liq.evolution.adapters.parallel_eval import ParallelEvaluator
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        mock_evolve.return_value = _make_evolution_result()

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
            parallel_config=ParallelConfig(backend="ray", max_workers=4),
        )
        adapter.fit(_make_dataframe())

        evaluator = mock_evolve.call_args[0][2]
        assert isinstance(evaluator, ParallelEvaluator)
        assert evaluator.backend == "ray"

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_fit_stores_evolution_result(self, mock_evolve) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        result = _make_evolution_result()
        mock_evolve.return_value = result

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter.fit(_make_dataframe())
        assert adapter.evolution_result is result

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_fit_with_labels(self, mock_evolve) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        mock_evolve.return_value = _make_evolution_result()

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        labels = pl.Series("labels", [0.0, 1.0] * 10)
        adapter.fit(_make_dataframe(), labels=labels)

        # Verify context passed to evolve included labels
        call_ctx = mock_evolve.call_args[0][3]
        assert "labels" in call_ctx
        np.testing.assert_array_equal(call_ctx["labels"], labels.to_numpy())

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_fit_without_labels(self, mock_evolve) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        mock_evolve.return_value = _make_evolution_result()

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        adapter.fit(_make_dataframe())

        call_ctx = mock_evolve.call_args[0][3]
        assert "labels" not in call_ctx

    def test_fit_predict_only_adapter_raises(self) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            evaluator=None,
        )
        with pytest.raises(AdapterError, match="predict-only"):
            adapter.fit(_make_dataframe())


# ------------------------------------------------------------------ #
#  Warm-start
# ------------------------------------------------------------------ #


class TestGPStrategyAdapterWarmStart:
    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_warm_start_updates_seeds(self, mock_evolve) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        program = _make_terminal()
        mock_evolve.return_value = _make_evolution_result(program)

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
            warm_start=True,
        )
        adapter.fit(_make_dataframe())
        assert adapter._seed_programs == [program]

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_warm_start_disabled_no_update(self, mock_evolve) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        mock_evolve.return_value = _make_evolution_result()

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
            warm_start=False,
        )
        adapter.fit(_make_dataframe())
        assert adapter._seed_programs is None

    @patch("liq.evolution.adapters.runner_strategy.evolve")
    def test_second_fit_uses_prior_best_as_seed(self, mock_evolve) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        program1 = _make_terminal("close")
        program2 = _make_terminal("volume")
        mock_evolve.side_effect = [
            _make_evolution_result(program1),
            _make_evolution_result(program2),
        ]

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
            warm_start=True,
        )
        adapter.fit(_make_dataframe())
        adapter.fit(_make_dataframe())

        # Second evolve call should have seed_programs=[program1]
        second_call = mock_evolve.call_args_list[1]
        assert second_call.kwargs.get("seed_programs") == [program1]


# ------------------------------------------------------------------ #
#  Protocol compliance
# ------------------------------------------------------------------ #


class TestGPStrategyAdapterProtocol:
    def test_satisfies_gp_strategy_protocol(self) -> None:
        from liq.evolution.adapters.runner_strategy import GPStrategyAdapter
        from liq.gp.config import GPConfig
        from liq.gp.primitives.registry import PrimitiveRegistry

        adapter = GPStrategyAdapter(
            PrimitiveRegistry(),
            GPConfig(population_size=20, max_depth=4, generations=2),
            MagicMock(),
        )
        assert isinstance(adapter, GPStrategy)
