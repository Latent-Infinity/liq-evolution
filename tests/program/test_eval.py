"""Tests for evaluation engine integration with trading primitives."""

from __future__ import annotations

import numpy as np

from liq.evolution.config import PrimitiveConfig
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.primitives.series_sources import prepare_evaluation_context
from liq.evolution.program import (
    ConstantNode,
    EvaluationContext,
    FunctionNode,
    ParameterizedNode,
    TerminalNode,
    evaluate,
)
from liq.gp.types import Series


def _make_ohlcv(n: int = 10) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    return {
        "open": close - rng.uniform(0, 1, n),
        "high": close + rng.uniform(0, 2, n),
        "low": close - rng.uniform(0, 2, n),
        "close": close,
        "volume": rng.uniform(1000, 5000, n),
    }


class TestTerminalEvaluation:
    def test_terminal_reads_context(self) -> None:
        ctx: EvaluationContext = {
            "close": np.array([1.0, 2.0, 3.0]),
        }
        node = TerminalNode(name="close", output_type=Series)
        result = evaluate(node, ctx)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


class TestConstantEvaluation:
    def test_constant_broadcasts(self) -> None:
        ctx: EvaluationContext = {
            "close": np.array([1.0, 2.0, 3.0]),
        }
        node = ConstantNode(value=5.0)
        result = evaluate(node, ctx)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])
        assert result.dtype == np.float64


class TestFunctionEvaluation:
    def test_add_node(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        prim = reg.get("add")
        close_node = TerminalNode(name="close", output_type=Series)
        open_node = TerminalNode(name="open", output_type=Series)
        tree = FunctionNode(primitive=prim, children=(close_node, open_node))

        ctx: EvaluationContext = {
            "close": np.array([10.0, 20.0]),
            "open": np.array([9.0, 18.0]),
        }
        result = evaluate(tree, ctx)
        np.testing.assert_array_equal(result, [19.0, 38.0])

    def test_div_nan_safety(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        prim = reg.get("div")
        num = TerminalNode(name="a", output_type=Series)
        den = TerminalNode(name="b", output_type=Series)
        tree = FunctionNode(primitive=prim, children=(num, den))

        ctx: EvaluationContext = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 0.0]),
        }
        result = evaluate(tree, ctx)
        assert np.all(np.isnan(result))


class TestParameterizedEvaluation:
    def test_n_bars_ago(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        prim = reg.get("n_bars_ago")
        close_node = TerminalNode(name="close", output_type=Series)
        tree = ParameterizedNode(
            primitive=prim,
            children=(close_node,),
            params={"shift": 2},
        )

        ctx: EvaluationContext = {
            "close": np.array([10.0, 20.0, 30.0, 40.0]),
        }
        result = evaluate(tree, ctx)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_array_equal(result[2:], [10.0, 20.0])


class TestNestedTree:
    def test_add_of_close_and_constant(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        add_prim = reg.get("add")
        close_node = TerminalNode(name="close", output_type=Series)
        const_node = ConstantNode(value=1.0)
        tree = FunctionNode(
            primitive=add_prim,
            children=(close_node, const_node),
        )

        ctx: EvaluationContext = {
            "close": np.array([10.0, 20.0, 30.0]),
        }
        result = evaluate(tree, ctx)
        np.testing.assert_array_equal(result, [11.0, 21.0, 31.0])

    def test_nested_arithmetic(self) -> None:
        """(close + open) / constant(2.0) should equal midrange."""
        reg = build_trading_registry(PrimitiveConfig())
        add_prim = reg.get("add")
        div_prim = reg.get("div")

        close_node = TerminalNode(name="close", output_type=Series)
        open_node = TerminalNode(name="open", output_type=Series)
        const_two = ConstantNode(value=2.0)

        sum_node = FunctionNode(
            primitive=add_prim,
            children=(close_node, open_node),
        )
        tree = FunctionNode(
            primitive=div_prim,
            children=(sum_node, const_two),
        )

        ctx: EvaluationContext = {
            "close": np.array([10.0, 20.0]),
            "open": np.array([8.0, 18.0]),
        }
        result = evaluate(tree, ctx)
        np.testing.assert_array_equal(result, [9.0, 19.0])


class TestDeterminism:
    def test_same_tree_same_result(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        add_prim = reg.get("add")
        close_node = TerminalNode(name="close", output_type=Series)
        const_node = ConstantNode(value=1.0)
        tree = FunctionNode(
            primitive=add_prim,
            children=(close_node, const_node),
        )
        ctx: EvaluationContext = {
            "close": np.array([1.0, 2.0, 3.0]),
        }
        r1 = evaluate(tree, ctx)
        r2 = evaluate(tree, ctx)
        np.testing.assert_array_equal(r1, r2)


class TestOutputProperties:
    def test_output_length_matches_context(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        add_prim = reg.get("add")
        t1 = TerminalNode(name="a", output_type=Series)
        t2 = TerminalNode(name="b", output_type=Series)
        tree = FunctionNode(primitive=add_prim, children=(t1, t2))

        ctx: EvaluationContext = {
            "a": np.ones(50),
            "b": np.ones(50),
        }
        assert len(evaluate(tree, ctx)) == 50

    def test_output_is_float64(self) -> None:
        ctx: EvaluationContext = {"close": np.array([1.0, 2.0])}
        node = TerminalNode(name="close", output_type=Series)
        assert evaluate(node, ctx).dtype == np.float64


class TestWithOHLCVContext:
    def test_full_pipeline(self) -> None:
        """Build a tree that uses the evaluation context from OHLCV data."""
        ohlcv = _make_ohlcv(20)
        ctx = prepare_evaluation_context(ohlcv)

        reg = build_trading_registry(PrimitiveConfig())

        # Tree: add(close, n_bars_ago(close, shift=1))
        add_prim = reg.get("add")
        nba_prim = reg.get("n_bars_ago")

        close = TerminalNode(name="close", output_type=Series)
        lagged = ParameterizedNode(
            primitive=nba_prim,
            children=(close,),
            params={"shift": 1},
        )
        tree = FunctionNode(
            primitive=add_prim,
            children=(close, lagged),
        )

        result = evaluate(tree, ctx)
        assert len(result) == 20
        assert result.dtype == np.float64
        # Bar 0 should be NaN (from n_bars_ago)
        assert np.isnan(result[0])
        # Bar 1: close[1] + close[0]
        np.testing.assert_allclose(
            result[1],
            ctx["close"][1] + ctx["close"][0],
        )
