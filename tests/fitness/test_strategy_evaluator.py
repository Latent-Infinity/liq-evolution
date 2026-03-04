"""Tests for StrategyEvaluator walk-forward aggregation and metadata contract."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.errors import FitnessEvaluationError
from liq.evolution.fitness.evaluation_schema import (
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
    BEHAVIOR_DESCRIPTOR_TURNOVER,
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
    to_loss_form,
)
from liq.evolution.fitness.strategy_evaluator import StrategyEvaluator
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


def _make_program() -> TerminalNode:
    return TerminalNode("close", Series)


def _make_split_set() -> list[WalkForwardSplit]:
    return [
        WalkForwardSplit(
            train=slice(0, 2),
            validate=slice(2, 4),
            test=slice(4, 6),
            slice_id="time_window:split_0",
        ),
        WalkForwardSplit(
            train=slice(10, 12),
            validate=slice(12, 14),
            test=slice(14, 16),
            slice_id="time_window:split_1",
        ),
    ]


class _SplitterRunner:
    """Simple backtest runner mock for StrategyEvaluator tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        strategy: Any,
        context: dict[str, np.ndarray],
        split: WalkForwardSplit,
    ) -> dict[str, Any]:  # noqa: ARG001, ANN401
        self.calls.append(split.slice_id)

        if split.slice_id == "time_window:split_0":
            return {
                "train": {
                    "metrics": {"cagr": 0.10, "max_drawdown": 0.20},
                    "traces": {
                        "position_trace": [1.0, 1.0],
                        "equity_curve": [100.0, 102.0],
                        "pnl_trace": [1.0, -1.0, 0.5],
                    },
                    "constraint_violations": {"liquidity": 0.01},
                },
                "validate": {
                    "metrics": {"cagr": 0.30, "max_drawdown": 0.40},
                    "traces": {
                        "position_trace": [2.0, 2.0, 0.0],
                        "equity_curve": [200.0, 199.0, 201.0],
                        "pnl_trace": [2.0, -0.5, 1.0],
                    },
                    "constraint_violations": {"liquidity": 0.20},
                },
                "test": {
                    "metrics": {"cagr": 0.20, "max_drawdown": 0.10},
                    "traces": {
                        "position_trace": [1.0, 0.0, 1.0],
                        "equity_curve": [300.0, 298.0, 301.0],
                        "pnl_trace": [1.0, 0.3, -0.4],
                    },
                },
            }

        return {
            "train": {
                "metrics": {"cagr": 0.20, "max_drawdown": 0.50},
                "traces": {
                    "position_trace": [2.0, 2.0, 2.0],
                    "equity_curve": [100.0, 101.0, 102.0],
                    "pnl_trace": [1.0, 1.0, 1.0],
                },
                "constraint_violations": {"leverage": 0.05},
            },
            "validate": {
                "metrics": {"cagr": 0.40, "max_drawdown": 0.60},
                "traces": {
                    "position_trace": [0.5, 1.0, 0.0],
                    "equity_curve": [100.0, 99.0, 100.0],
                    "pnl_trace": [0.2, -0.2, 0.2],
                },
                "constraint_violations": {"leverage": 0.10},
            },
            "test": {
                "metrics": {"cagr": 0.30, "max_drawdown": 0.20},
                "traces": {
                    "position_trace": [1.0, 1.0],
                    "equity_curve": [150.0, 149.0],
                    "pnl_trace": [0.1, -0.1],
                },
            },
        }


class _AdversarialRunner:
    """Runner that injects adversarial penalty cases for constraint testing."""

    def __call__(
        self,
        strategy: object,  # noqa: ARG002
        context: dict[str, object],  # noqa: ARG002
        split: WalkForwardSplit,  # noqa: ARG002
    ) -> dict[str, object]:
        payload = {
            "metrics": {"cagr": 0.1},
            "traces": {
                "position_trace": [1.0, 1.0],
                "equity_curve": [100.0, 101.0],
                "pnl_trace": [1.0, -0.5],
            },
            "adversarial_cases": {
                "high_spread": 0.42,
                "halted": 0.18,
                "gap": 0.27,
                "limit_move": 0.33,
            },
        }
        return {
            "train": payload,
            "validate": payload,
            "test": payload,
        }


class _RegimeSliceRunner:
    """Runner that emits regime-style slice keys in addition to legacy objectives."""

    def __call__(
        self,
        strategy: object,  # noqa: ARG002
        context: dict[str, object],  # noqa: ARG002
        split: WalkForwardSplit,  # noqa: ARG002
    ) -> dict[str, object]:
        payload = {
            "metrics": {"cagr": 0.25, "max_drawdown": 0.11},
            "traces": {
                "position_trace": [1.0, 1.0, 0.0],
                "equity_curve": [100.0, 101.0, 99.0],
                "pnl_trace": [1.0, -0.2, 0.1],
            },
            "slice_scores": {
                "event:earnings:pre_announcement": 0.41,
                "liquidity:tight_spread": 0.12,
                "instrument:SPY": 0.27,
                "time_window:split_legacy": 0.33,
            },
        }
        return {"train": payload, "validate": payload, "test": payload}


class _FoldSliceOverrideRunner:
    """Runner that supplies the effective split id inside payload."""

    def __call__(
        self,
        strategy: object,  # noqa: ARG002
        context: dict[str, object],  # noqa: ARG002
        split: WalkForwardSplit,  # noqa: ARG002
    ) -> dict[str, object]:
        payload = {
            "metrics": {"cagr": 0.2, "max_drawdown": 0.15},
            "traces": {
                "position_trace": [1.0, 0.0],
                "equity_curve": [100.0, 101.0],
                "pnl_trace": [1.0, -1.0],
            },
        }
        return {
            "slice_id": "time_window:fold_77",
            "train": payload,
            "validate": payload,
            "test": payload,
        }


class TestStrategyEvaluatorLoggingContracts:
    """Strategy evaluator emits structured logs for each candidate and split."""

    def test_evaluate_logs_run_id_and_candidate_split_progress(self, caplog) -> None:
        evaluator = StrategyEvaluator(
            backtest_runner=_SplitterRunner(),
            splits=_make_split_set(),
            objectives=("cagr",),
            objective_directions=("maximize",),
            split_weights={"train": 1.0, "validate": 1.0, "test": 0.0},
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )

        with caplog.at_level(logging.INFO, logger="liq.evolution.fitness.strategy_evaluator"):
            evaluator.evaluate(
                [_make_program()],
                context={"run_id": "run-strategy", "labels": np.zeros(100)},
            )

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "strategy_evaluator split_evaluated run_id=run-strategy" in message
            for message in messages
        )
        assert any(
            "candidate_hash=" in message and "strategy_evaluator candidate_complete" in message
            for message in messages
        )
        assert any(
            "strategy_evaluator batch_complete run_id=run-strategy" in message
            for message in messages
        )


class TestStrategyEvaluator:
    def test_evaluate_aggregates_split_objectives_and_payload_shape(self) -> None:
        runner = _SplitterRunner()
        evaluator = StrategyEvaluator(
            backtest_runner=runner,
            splits=_make_split_set(),
            objectives=("cagr", "max_drawdown"),
            objective_directions=("maximize", "minimize"),
            split_weights={"train": 1.0, "validate": 1.0, "test": 0.0},
            behavior_descriptor_names=(
                BEHAVIOR_DESCRIPTOR_TURNOVER,
                BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
            ),
        )

        results = evaluator.evaluate(
            [_make_program()],
            context={"labels": np.zeros(100)},
        )
        assert len(results) == 1

        result = results[0]
        assert result.objectives == pytest.approx((0.25, 0.425))

        metadata = result.metadata
        assert isinstance(metadata[METADATA_KEY_RAW_OBJECTIVES], tuple)
        assert metadata[METADATA_KEY_RAW_OBJECTIVES] == pytest.approx((0.25, 0.425))
        assert metadata[METADATA_KEY_PER_SPLIT_METRICS] == {
            "time_window:split_0:train": {"cagr": 0.1, "max_drawdown": 0.2},
            "time_window:split_0:validate": {"cagr": 0.3, "max_drawdown": 0.4},
            "time_window:split_1:train": {"cagr": 0.2, "max_drawdown": 0.5},
            "time_window:split_1:validate": {"cagr": 0.4, "max_drawdown": 0.6},
        }

        assert set(metadata[METADATA_KEY_BEHAVIOR_DESCRIPTORS]) == {
            BEHAVIOR_DESCRIPTOR_TURNOVER,
            BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
        }

        for key in (
            "time_window:split_0:train:cagr",
            "time_window:split_0:train:max_drawdown",
            "time_window:split_0:validate:cagr",
            "time_window:split_0:validate:max_drawdown",
            "time_window:split_1:train:cagr",
            "time_window:split_1:train:max_drawdown",
            "time_window:split_1:validate:cagr",
            "time_window:split_1:validate:max_drawdown",
        ):
            assert key in metadata[METADATA_KEY_SLICE_SCORES]

        train_cagr_score_key = "time_window:split_0:train:cagr"
        train_drawdown_score_key = "time_window:split_0:train:max_drawdown"
        assert metadata[METADATA_KEY_SLICE_SCORES][train_cagr_score_key] == (
            to_loss_form(0.10, "maximize")
        )
        assert metadata[METADATA_KEY_SLICE_SCORES][train_drawdown_score_key] == (
            to_loss_form(0.20, "minimize")
        )

        # Constraint failures should keep the maximum violation value per key.
        assert metadata[METADATA_KEY_CONSTRAINT_VIOLATIONS] == {
            "liquidity": 0.20,
            "leverage": 0.10,
        }

        assert runner.calls == ["time_window:split_0", "time_window:split_1"]

    def test_includes_test_scores_when_include_test_is_true(self) -> None:
        runner = _SplitterRunner()
        evaluator = StrategyEvaluator(
            backtest_runner=runner,
            splits=_make_split_set()[:1],
            objectives=("cagr",),
            objective_directions=("maximize",),
            split_weights={"train": 1.0, "validate": 1.0, "test": 1.0},
            include_test=True,
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )

        results = evaluator.evaluate(
            [_make_program()],
            context={"labels": np.zeros(100)},
        )
        metadata = results[0].metadata

        assert metadata[METADATA_KEY_RAW_OBJECTIVES] == pytest.approx((0.2,))
        assert metadata[METADATA_KEY_PER_SPLIT_METRICS] == {
            "time_window:split_0:train": {"cagr": 0.1},
            "time_window:split_0:validate": {"cagr": 0.3},
            "time_window:split_0:test": {"cagr": 0.2},
        }
        assert "time_window:split_0:test:cagr" in metadata[METADATA_KEY_SLICE_SCORES]
        assert metadata[METADATA_KEY_SLICE_SCORES]["time_window:split_0:test:cagr"] == (
            to_loss_form(0.2, "maximize")
        )

    def test_adversarial_cases_injected_as_penalty_cases(self) -> None:
        evaluator = StrategyEvaluator(
            backtest_runner=_AdversarialRunner(),
            splits=_make_split_set()[:1],
            objectives=("cagr",),
            objective_directions=("maximize",),
            include_test=True,
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )

        results = evaluator.evaluate([_make_program()], context={"labels": [0.0] * 10})
        metadata = results[0].metadata

        split_key = "time_window:split_0:train"
        expected = {
            "high_spread": 0.42,
            "halted": 0.18,
            "gap": 0.27,
            "limit_move": 0.33,
        }
        for adversarial_key, value in expected.items():
            constraint_key = f"{split_key}:adversarial:{adversarial_key}"
            slice_key = f"{split_key}:constraint:{constraint_key}"

            assert metadata[METADATA_KEY_CONSTRAINT_VIOLATIONS][constraint_key] == value
            assert metadata[METADATA_KEY_SLICE_SCORES][slice_key] == value

    def test_missing_labels_is_rejected(self) -> None:
        evaluator = StrategyEvaluator(
            backtest_runner=_SplitterRunner(),
            splits=_make_split_set()[:1],
        )
        with pytest.raises(
            FitnessEvaluationError,
            match="Context must include 'labels'",
        ):
            evaluator.evaluate([_make_program()], context={})

    def test_rejects_invalid_objective_directions(self) -> None:
        with pytest.raises(
            ValueError,
            match="objective_directions must be 'maximize' or 'minimize'",
        ):
            StrategyEvaluator(
                backtest_runner=_SplitterRunner(),
                splits=_make_split_set()[:1],
                objectives=("cagr", "max_drawdown"),
                objective_directions=("maximize", "up"),
            )

    def test_rejects_unknown_behavior_descriptors(self) -> None:
        with pytest.raises(ValueError, match="unsupported behavior descriptors"):
            StrategyEvaluator(
                backtest_runner=_SplitterRunner(),
                splits=_make_split_set()[:1],
                behavior_descriptor_names=("does_not_exist",),
            )

    def test_rejects_invalid_split_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            StrategyEvaluator(
                backtest_runner=_SplitterRunner(),
                splits=_make_split_set()[:1],
                split_weights={"train": -1.0, "validate": 1.0, "test": 0.0},
            )

        with pytest.raises(ValueError, match="unsupported keys"):
            StrategyEvaluator(
                backtest_runner=_SplitterRunner(),
                splits=_make_split_set()[:1],
                split_weights={
                    "train": 1.0,
                    "validate": 1.0,
                    "test": 0.0,
                    "unexpected": 0.0,
                },
            )

    def test_regime_slice_scores_are_preserved_with_regime_prefixes(self) -> None:
        evaluator = StrategyEvaluator(
            backtest_runner=_RegimeSliceRunner(),
            splits=_make_split_set()[:1],
            objectives=("cagr", "max_drawdown"),
            objective_directions=("maximize", "minimize"),
            split_weights={"train": 1.0, "validate": 1.0, "test": 0.0},
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )

        results = evaluator.evaluate(
            [_make_program()],
            context={"labels": np.zeros(100)},
        )
        metadata = results[0].metadata

        assert (
            metadata[METADATA_KEY_SLICE_SCORES][
                "event:earnings:pre_announcement:time_window:split_0:train"
            ]
            == 0.41
        )
        assert (
            metadata[METADATA_KEY_SLICE_SCORES][
                "liquidity:tight_spread:time_window:split_0:train"
            ]
            == 0.12
        )
        assert (
            metadata[METADATA_KEY_SLICE_SCORES][
                "instrument:SPY:time_window:split_0:train"
            ]
            == 0.27
        )
        assert (
            metadata[METADATA_KEY_SLICE_SCORES][
                "time_window:split_legacy:time_window:split_0:train"
            ]
            == 0.33
        )

        assert "time_window:split_0:train:cagr" in metadata[METADATA_KEY_SLICE_SCORES]

    def test_split_scores_are_keyed_by_runner_payload_slice_id(self) -> None:
        evaluator = StrategyEvaluator(
            backtest_runner=_FoldSliceOverrideRunner(),
            splits=_make_split_set()[:1],
            objectives=("cagr", "max_drawdown"),
            objective_directions=("maximize", "minimize"),
            behavior_descriptor_names=(BEHAVIOR_DESCRIPTOR_TURNOVER,),
        )

        result = evaluator.evaluate(
            [_make_program()],
            context={"labels": np.zeros(10)},
        )[0]

        metadata = result.metadata
        assert "time_window:fold_77:train:cagr" in metadata[METADATA_KEY_SLICE_SCORES]
        assert (
            "time_window:fold_77:validate:max_drawdown"
            in metadata[METADATA_KEY_SLICE_SCORES]
        )
        assert (
            "time_window:split_0:train:cagr" not in metadata[METADATA_KEY_SLICE_SCORES]
        )
