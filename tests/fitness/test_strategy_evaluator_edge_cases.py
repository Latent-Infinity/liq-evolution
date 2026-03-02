"""Edge-case tests to increase coverage for strategy_evaluator.py."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.errors import FitnessEvaluationError
from liq.evolution.fitness.evaluation_schema import (
    BEHAVIOR_DESCRIPTOR_TURNOVER,
)
from liq.evolution.fitness.strategy_evaluator import (
    StrategyEvaluator,
    _coerce_nonnegative,
    _normalize_slice_score_key,
    _safe_mapping_float_dict,
    _sanitize_objective,
    _to_float_sequence,
)
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


def _program() -> TerminalNode:
    return TerminalNode("close", Series)


def _single_split() -> list[WalkForwardSplit]:
    return [
        WalkForwardSplit(
            train=slice(0, 2),
            validate=slice(2, 4),
            test=slice(4, 6),
            slice_id="time_window:split_0",
        ),
    ]


def _simple_runner(
    strategy: object,
    context: object,
    split: WalkForwardSplit,
) -> dict[str, Any]:
    payload = {
        "metrics": {"cagr": 0.1, "max_drawdown": 0.2},
        "traces": {
            "position_trace": [1.0, 1.0],
            "equity_curve": [100.0, 102.0],
            "pnl_trace": [1.0, -1.0],
        },
    }
    return {"train": payload, "validate": payload, "test": payload}


# ------------------------------------------------------------------ #
#  Constructor validation edge cases
# ------------------------------------------------------------------ #


class TestConstructorValidation:
    """Cover constructor ValueError paths."""

    def test_empty_behavior_descriptors_raises(self) -> None:
        with pytest.raises(
            ValueError, match="behavior_descriptor_names must be non-empty"
        ):
            StrategyEvaluator(
                backtest_runner=_simple_runner,
                splits=_single_split(),
                behavior_descriptor_names=(),
            )

    def test_empty_splits_raises(self) -> None:
        with pytest.raises(ValueError, match="splits must be non-empty"):
            StrategyEvaluator(
                backtest_runner=_simple_runner,
                splits=[],
            )

    def test_empty_objectives_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one objective is required"):
            StrategyEvaluator(
                backtest_runner=_simple_runner,
                splits=_single_split(),
                objectives=(),
            )

    def test_too_many_custom_objectives_raises(self) -> None:
        """When objectives exceed DEFAULT_OBJECTIVE_DIRECTIONS length."""
        many_objectives = tuple(f"obj_{i}" for i in range(20))
        with pytest.raises(ValueError, match="missing default direction"):
            StrategyEvaluator(
                backtest_runner=_simple_runner,
                splits=_single_split(),
                objectives=many_objectives,
                objective_directions=None,
            )

    def test_directions_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must match objectives length"):
            StrategyEvaluator(
                backtest_runner=_simple_runner,
                splits=_single_split(),
                objectives=("cagr", "max_drawdown"),
                objective_directions=("maximize",),
            )

    def test_non_finite_split_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            StrategyEvaluator(
                backtest_runner=_simple_runner,
                splits=_single_split(),
                split_weights={"train": float("inf"), "validate": 1.0, "test": 0.0},
            )

    def test_nan_split_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            StrategyEvaluator(
                backtest_runner=_simple_runner,
                splits=_single_split(),
                split_weights={"train": float("nan"), "validate": 1.0, "test": 0.0},
            )


# ------------------------------------------------------------------ #
#  evaluate_fitness alias
# ------------------------------------------------------------------ #


class TestEvaluateFitnessAlias:
    def test_evaluate_fitness_delegates_to_evaluate(self) -> None:
        evaluator = StrategyEvaluator(
            backtest_runner=_simple_runner,
            splits=_single_split(),
            objectives=("cagr",),
            objective_directions=("maximize",),
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )
        results = evaluator.evaluate_fitness(
            [_program()],
            context={"labels": np.zeros(10)},
        )
        assert len(results) == 1
        assert len(results[0].objectives) == 1


# ------------------------------------------------------------------ #
#  Runner failure paths
# ------------------------------------------------------------------ #


class TestRunnerFailures:
    def test_runner_exception_wrapped(self) -> None:
        def failing_runner(
            strategy: object,
            context: object,
            split: WalkForwardSplit,
        ) -> dict[str, Any]:
            raise RuntimeError("boom")

        evaluator = StrategyEvaluator(
            backtest_runner=failing_runner,
            splits=_single_split(),
            objectives=("cagr",),
            objective_directions=("maximize",),
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )
        with pytest.raises(FitnessEvaluationError, match="Backtest runner failed"):
            evaluator.evaluate([_program()], context={"labels": np.zeros(10)})

    def test_runner_returns_none_raises(self) -> None:
        def none_runner(
            strategy: object,
            context: object,
            split: WalkForwardSplit,
        ) -> None:
            return None

        evaluator = StrategyEvaluator(
            backtest_runner=none_runner,
            splits=_single_split(),
            objectives=("cagr",),
            objective_directions=("maximize",),
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )
        with pytest.raises(FitnessEvaluationError, match="returned no payload"):
            evaluator.evaluate([_program()], context={"labels": np.zeros(10)})


# ------------------------------------------------------------------ #
#  Flat payload normalization path
# ------------------------------------------------------------------ #


class TestFlatPayload:
    def test_flat_payload_used_for_all_phases(self) -> None:
        """A payload without explicit train/validate/test keys is used for all."""

        def flat_runner(
            strategy: object,
            context: object,
            split: WalkForwardSplit,
        ) -> dict[str, Any]:
            return {
                "metrics": {"cagr": 0.3},
                "traces": {
                    "position_trace": [1.0],
                    "equity_curve": [100.0],
                    "pnl_trace": [0.5],
                },
            }

        evaluator = StrategyEvaluator(
            backtest_runner=flat_runner,
            splits=_single_split(),
            objectives=("cagr",),
            objective_directions=("maximize",),
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )
        results = evaluator.evaluate([_program()], context={"labels": np.zeros(10)})
        assert len(results) == 1
        # Train and validate should both have the same metric
        metadata = results[0].metadata
        assert metadata["per_split_metrics"]["time_window:split_0:train"]["cagr"] == 0.3
        assert (
            metadata["per_split_metrics"]["time_window:split_0:validate"]["cagr"] == 0.3
        )


# ------------------------------------------------------------------ #
#  Legacy traces (Iterable, not Mapping)
# ------------------------------------------------------------------ #


class TestLegacyTraces:
    def test_iterable_traces_treated_as_position_trace(self) -> None:
        def legacy_runner(
            strategy: object,
            context: object,
            split: WalkForwardSplit,
        ) -> dict[str, Any]:
            payload = {
                "metrics": {"cagr": 0.1},
                "traces": [1.0, 2.0, 0.0],
            }
            return {"train": payload, "validate": payload, "test": payload}

        evaluator = StrategyEvaluator(
            backtest_runner=legacy_runner,
            splits=_single_split(),
            objectives=("cagr",),
            objective_directions=("maximize",),
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )
        results = evaluator.evaluate([_program()], context={"labels": np.zeros(10)})
        assert len(results) == 1


# ------------------------------------------------------------------ #
#  Helper function tests
# ------------------------------------------------------------------ #


class TestSafeMapFloatDict:
    def test_non_mapping_returns_empty(self) -> None:
        assert _safe_mapping_float_dict("bad") == {}

    def test_none_returns_empty(self) -> None:
        assert _safe_mapping_float_dict(None) == {}

    def test_non_numeric_value_skipped(self) -> None:
        result = _safe_mapping_float_dict({"a": 1.0, "b": "bad", "c": 2.0})
        assert result == {"a": 1.0, "c": 2.0}

    def test_none_value_skipped(self) -> None:
        result = _safe_mapping_float_dict({"a": 1.0, "b": None})
        assert result == {"a": 1.0}


class TestToFloatSequence:
    def test_none_returns_empty(self) -> None:
        assert _to_float_sequence(None) == []

    def test_string_returns_empty(self) -> None:
        assert _to_float_sequence("not_a_sequence") == []

    def test_bytes_returns_empty(self) -> None:
        assert _to_float_sequence(b"bytes") == []

    def test_non_iterable_returns_empty(self) -> None:
        assert _to_float_sequence(42) == []

    def test_mapping_with_values_key(self) -> None:
        result = _to_float_sequence({"values": [1.0, 2.0, 3.0]})
        assert result == [1.0, 2.0, 3.0]

    def test_mapping_without_values_key_returns_empty(self) -> None:
        result = _to_float_sequence({"other": [1.0]})
        assert result == []

    def test_non_numeric_items_skipped(self) -> None:
        result = _to_float_sequence([1.0, "bad", 2.0, None])
        assert result == [1.0, 2.0]

    def test_list_of_floats(self) -> None:
        assert _to_float_sequence([1.0, 2.5, 3.0]) == [1.0, 2.5, 3.0]


class TestSanitizeObjective:
    def test_non_numeric_value_penalized_maximize(self) -> None:
        result = _sanitize_objective("bad", direction="maximize", nan_penalty=1e6)
        assert result == -1e6

    def test_non_numeric_value_penalized_minimize(self) -> None:
        result = _sanitize_objective("bad", direction="minimize", nan_penalty=1e6)
        assert result == 1e6

    def test_positive_inf_penalized(self) -> None:
        result = _sanitize_objective(
            float("inf"), direction="maximize", nan_penalty=1e6
        )
        assert result == -1e6

    def test_negative_inf_penalized(self) -> None:
        result = _sanitize_objective(
            float("-inf"), direction="minimize", nan_penalty=1e6
        )
        assert result == 1e6

    def test_finite_value_returned_as_is(self) -> None:
        assert _sanitize_objective(0.5, direction="maximize", nan_penalty=1e6) == 0.5


class TestCoerceNonnegative:
    def test_inf_returns_zero(self) -> None:
        assert _coerce_nonnegative(float("inf")) == 0.0

    def test_nan_returns_zero(self) -> None:
        assert _coerce_nonnegative(float("nan")) == 0.0

    def test_negative_returns_zero(self) -> None:
        assert _coerce_nonnegative(-1.0) == 0.0

    def test_positive_returns_value(self) -> None:
        assert _coerce_nonnegative(3.14) == 3.14


class TestNormalizeSliceScoreKey:
    def test_no_colon_adds_split_key(self) -> None:
        result = _normalize_slice_score_key("train", "my_metric")
        assert result == ("train:my_metric",)

    def test_time_window_prefix_adds_split_key(self) -> None:
        result = _normalize_slice_score_key("validate", "time_window:2023")
        assert result == ("time_window:2023:validate",)

    def test_time_window_prefix_with_split_key_present(self) -> None:
        result = _normalize_slice_score_key("validate", "time_window:validate:extra")
        assert result == ("time_window:validate:extra",)

    def test_event_prefix(self) -> None:
        result = _normalize_slice_score_key("train", "event:earnings")
        assert result == ("event:earnings:train",)

    def test_unknown_prefix_with_split_key_present(self) -> None:
        result = _normalize_slice_score_key("train", "custom:train:data")
        assert result == ("custom:train:data",)

    def test_unknown_prefix_without_split_key(self) -> None:
        result = _normalize_slice_score_key("train", "custom:other")
        assert result == ("train:custom:other",)
