"""Tests for fitness-evaluation result caching."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.fitness.eval_cache import FitnessEvaluationCache, compute_program_hash
from liq.evolution.fitness import strategy_evaluator as strategy_evaluator_module
from liq.evolution.fitness.strategy_evaluator import StrategyEvaluator
from liq.gp.program.ast import TerminalNode
from liq.gp.types import Series


def _make_program() -> TerminalNode:
    return TerminalNode("close", Series)


def _make_alt_program() -> TerminalNode:
    return TerminalNode("volume", Series)


def _make_split(slice_id: str | None = "time_window:split_0") -> WalkForwardSplit:
    return WalkForwardSplit(
        train=slice(0, 2),
        validate=slice(2, 4),
        test=slice(4, 6),
        slice_id=slice_id,
    )


def _payload_for(program_value: float, *, override_slice_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "train": {
            "metrics": {"cagr": program_value},
            "traces": {
                "position_trace": [1.0, 1.0],
                "equity_curve": [100.0, 101.0],
                "pnl_trace": [1.0, 1.0],
            },
        },
        "validate": {
            "metrics": {"cagr": program_value},
            "traces": {
                "position_trace": [1.0, 0.0],
                "equity_curve": [100.0, 99.0],
                "pnl_trace": [0.5, -0.5],
            },
        },
        "test": {
            "metrics": {"cagr": program_value},
            "traces": {
                "position_trace": [0.0, 0.0],
                "equity_curve": [100.0, 100.0],
                "pnl_trace": [0.0, 0.0],
            },
        },
    }
    if override_slice_id is not None:
        payload["slice_id"] = override_slice_id
    return payload


class _CachingRunner:
    def __init__(self, *, override_slice_id: str | None = None, base: float = 0.1) -> None:
        self._override_slice_id = override_slice_id
        self._base = base
        self.calls: list[str | None] = []

    def __call__(
        self,
        _strategy: object,  # noqa: ARG002
        _context: dict[str, Any],
        split: WalkForwardSplit,
    ) -> dict[str, Any]:
        self.calls.append(split.slice_id)
        payload = _payload_for(self._base, override_slice_id=self._override_slice_id)
        return payload


class _MultiSplitRunner:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def __call__(
        self,
        _strategy: object,  # noqa: ARG002
        _context: dict[str, Any],
        split: WalkForwardSplit,
    ) -> dict[str, Any]:
        self.calls.append(split.slice_id)
        return _payload_for(0.25)


def test_evaluator_cache_hits_when_fingerprint_and_slice_match() -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split = _make_split("time_window:cached")
    runner = _CachingRunner()
    evaluator = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    first = evaluator.evaluate([_make_program()], context={"labels": np.zeros(6)})[0]
    second = evaluator.evaluate([_make_program()], context={"labels": np.zeros(6)})[0]

    assert runner.calls == ["time_window:cached"]
    assert first.metadata["cache"]["enabled"] is True
    assert first.metadata["cache"]["hits"] == 0
    assert first.metadata["cache"]["misses"] == 1
    assert second.metadata["cache"]["hits"] == 1
    assert second.metadata["cache"]["misses"] == 0
    assert second.objectives == first.objectives


def test_evaluator_cache_misses_when_split_id_changes() -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split_a = _make_split("time_window:split_a")
    split_b = _make_split("time_window:split_b")

    runner = _MultiSplitRunner()
    evaluator = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split_a, split_b],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    evaluator.evaluate([_make_program()], context={"labels": np.zeros(10)})
    assert runner.calls == ["time_window:split_a", "time_window:split_b"]

    evaluator.evaluate([_make_program()], context={"labels": np.zeros(10)})
    assert runner.calls == ["time_window:split_a", "time_window:split_b"]


def test_evaluator_cache_uses_runner_payload_slice_id_for_keying() -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split_a = _make_split("time_window:legacy_a")
    split_b = _make_split("time_window:legacy_b")
    runner = _CachingRunner(override_slice_id="time_window:fold_payload")
    program = _make_program()

    evaluator = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split_a],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )
    first = evaluator.evaluate([program], context={"labels": np.zeros(6)})[0]
    assert runner.calls == ["time_window:legacy_a"]
    assert "time_window:fold_payload:train:cagr" in first.metadata["slice_scores"]

    evaluator_with_different_split = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split_b],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )
    evaluator_with_different_split.evaluate([program], context={"labels": np.zeros(6)})

    assert len(runner.calls) == 2
    assert (
        cache.get(
            compute_program_hash(program),
            "time_window:fold_payload",
            first.metadata["cache"]["fingerprint"],
        )
        is not None
    )


def test_cache_disabled_without_slice_id() -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split = _make_split(None)
    runner = _CachingRunner()

    evaluator = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    evaluator.evaluate([_make_program()], context={"labels": np.zeros(6)})
    evaluator.evaluate([_make_program()], context={"labels": np.zeros(6)})

    assert runner.calls == [None, None]


def test_cache_miss_when_evaluator_fingerprint_changes() -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split = _make_split("time_window:fingerprint")
    runner = _CachingRunner(base=0.2)

    first = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )
    second = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr", "max_drawdown"),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    first.evaluate([_make_program()], context={"labels": np.zeros(6)})
    first.evaluate([_make_program()], context={"labels": np.zeros(6)})
    second.evaluate([_make_program()], context={"labels": np.zeros(6)})

    assert runner.calls == ["time_window:fingerprint", "time_window:fingerprint"]


def test_cache_miss_when_program_structure_changes() -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split = _make_split("time_window:program_structure")
    runner = _MultiSplitRunner()

    evaluator = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    evaluator.evaluate([_make_program()], context={"labels": np.zeros(6)})
    evaluator.evaluate([_make_alt_program()], context={"labels": np.zeros(6)})

    assert runner.calls == [
        "time_window:program_structure",
        "time_window:program_structure",
    ]


def test_cache_miss_when_nan_penalty_or_max_finite_change() -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split = _make_split("time_window:penalty_cache")
    runner = _MultiSplitRunner()

    first = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
        nan_penalty=1e6,
    )
    second = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
        nan_penalty=1e5,
    )

    first.evaluate([_make_program()], context={"labels": np.zeros(6)})
    second.evaluate([_make_program()], context={"labels": np.zeros(6)})

    assert runner.calls == [
        "time_window:penalty_cache",
        "time_window:penalty_cache",
    ]

    third = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
        max_finite=1e8,
    )
    fourth = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
        max_finite=1e9,
    )

    third.evaluate([_make_program()], context={"labels": np.zeros(6)})
    fourth.evaluate([_make_program()], context={"labels": np.zeros(6)})

    assert runner.calls == [
        "time_window:penalty_cache",
        "time_window:penalty_cache",
        "time_window:penalty_cache",
        "time_window:penalty_cache",
    ]


def test_cache_miss_when_descriptor_schema_version_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FitnessEvaluationCache(max_entries=10)
    split = _make_split("time_window:schema_version")
    runner = _MultiSplitRunner()

    first = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    monkeypatch.setattr(
        strategy_evaluator_module,
        "BEHAVIOR_DESCRIPTOR_SCHEMA_VERSION",
        "2.0",
    )

    second = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    first.evaluate([_make_program()], context={"labels": np.zeros(6)})
    second.evaluate([_make_program()], context={"labels": np.zeros(6)})

    assert runner.calls == [
        "time_window:schema_version",
        "time_window:schema_version",
    ]


def test_lru_cache_eviction() -> None:
    cache = FitnessEvaluationCache(max_entries=1)
    split_a = _make_split("time_window:a")
    split_b = _make_split("time_window:b")
    runner = _MultiSplitRunner()

    evaluator_a = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split_a],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )
    evaluator_b = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split_b],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )
    evaluator_a_again = StrategyEvaluator(
        backtest_runner=runner,
        splits=[split_a],
        objectives=("cagr",),
        behavior_descriptor_names=("turnover",),
        cache=cache,
    )

    evaluator_a.evaluate([_make_program()], context={"labels": np.zeros(6)})
    evaluator_b.evaluate([_make_program()], context={"labels": np.zeros(6)})
    evaluator_a_again.evaluate([_make_program()], context={"labels": np.zeros(6)})

    assert runner.calls == [
        "time_window:a",
        "time_window:b",
        "time_window:a",
    ]
    assert cache.entries == 1
