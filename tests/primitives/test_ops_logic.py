"""Tests for logic operator primitives."""

from __future__ import annotations

import numpy as np
import pytest

from liq.evolution.primitives.ops_logic import (
    register_logic_ops,
    safe_and,
    safe_if_then_else,
    safe_not,
    safe_or,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, Series


class TestRegistration:
    def test_registers_four_operators(self) -> None:
        reg = PrimitiveRegistry()
        register_logic_ops(reg)
        assert len(reg.list_primitives(category="logic")) == 4

    def test_and_or_not_output_bool_series(self) -> None:
        reg = PrimitiveRegistry()
        register_logic_ops(reg)
        prims = {p.name: p for p in reg.list_primitives(category="logic")}
        assert prims["and_op"].output_type == BoolSeries
        assert prims["or_op"].output_type == BoolSeries
        assert prims["not_op"].output_type == BoolSeries

    def test_if_then_else_output_series(self) -> None:
        reg = PrimitiveRegistry()
        register_logic_ops(reg)
        prims = {p.name: p for p in reg.list_primitives(category="logic")}
        assert prims["if_then_else"].output_type == Series


class TestAnd:
    def test_truth_table(self) -> None:
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(safe_and(a, b), [1.0, 0.0, 0.0, 0.0])

    def test_threshold(self) -> None:
        # Values > 0.5 are truthy
        a = np.array([0.6, 0.4])
        b = np.array([0.8, 0.9])
        np.testing.assert_array_equal(safe_and(a, b), [1.0, 0.0])


class TestOr:
    def test_truth_table(self) -> None:
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(safe_or(a, b), [1.0, 1.0, 1.0, 0.0])


class TestNot:
    def test_truth_table(self) -> None:
        a = np.array([1.0, 0.0])
        np.testing.assert_array_equal(safe_not(a), [0.0, 1.0])

    def test_threshold(self) -> None:
        a = np.array([0.6, 0.4])
        np.testing.assert_array_equal(safe_not(a), [0.0, 1.0])


class TestIfThenElse:
    def test_basic(self) -> None:
        cond = np.array([1.0, 0.0, 1.0])
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([40.0, 50.0, 60.0])
        np.testing.assert_array_equal(safe_if_then_else(cond, a, b), [10.0, 50.0, 30.0])

    def test_threshold_selection(self) -> None:
        cond = np.array([0.6, 0.4])
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        np.testing.assert_array_equal(safe_if_then_else(cond, a, b), [1.0, 4.0])


class TestOutputType:
    @pytest.mark.parametrize("op", [safe_and, safe_or])
    def test_binary_logic_float64(self, op) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert op(a, b).dtype == np.float64

    def test_not_float64(self) -> None:
        assert safe_not(np.array([1.0, 0.0])).dtype == np.float64

    def test_if_then_else_float64(self) -> None:
        result = safe_if_then_else(np.array([1.0]), np.array([2.0]), np.array([3.0]))
        assert result.dtype == np.float64
