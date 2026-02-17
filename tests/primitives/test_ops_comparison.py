"""Tests for comparison operator primitives."""

from __future__ import annotations

import numpy as np
import pytest

from liq.evolution.primitives.ops_comparison import (
    register_comparison_ops,
    safe_eq,
    safe_gt,
    safe_gte,
    safe_lt,
    safe_lte,
    safe_neq,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries


class TestRegistration:
    def test_registers_six_operators(self) -> None:
        reg = PrimitiveRegistry()
        register_comparison_ops(reg)
        assert len(reg.list_primitives(category="comparison")) == 6

    def test_all_output_bool_series(self) -> None:
        reg = PrimitiveRegistry()
        register_comparison_ops(reg)
        for p in reg.list_primitives(category="comparison"):
            assert p.output_type == BoolSeries


class TestGt:
    def test_basic(self) -> None:
        a = np.array([3.0, 1.0, 2.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_gt(a, b), [1.0, 0.0, 0.0])

    def test_nan_produces_zero(self) -> None:
        a = np.array([np.nan, 3.0])
        b = np.array([1.0, np.nan])
        result = safe_gt(a, b)
        np.testing.assert_array_equal(result, [0.0, 0.0])


class TestLt:
    def test_basic(self) -> None:
        a = np.array([1.0, 3.0, 2.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_lt(a, b), [1.0, 0.0, 0.0])


class TestGte:
    def test_basic(self) -> None:
        a = np.array([3.0, 2.0, 1.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_gte(a, b), [1.0, 1.0, 0.0])


class TestLte:
    def test_basic(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_lte(a, b), [1.0, 1.0, 0.0])


class TestEq:
    def test_basic(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_eq(a, b), [0.0, 1.0, 0.0])

    def test_nan_not_equal(self) -> None:
        a = np.array([np.nan])
        b = np.array([np.nan])
        np.testing.assert_array_equal(safe_eq(a, b), [0.0])


class TestNeq:
    def test_basic(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_neq(a, b), [1.0, 0.0, 1.0])

    def test_nan_not_not_equal(self) -> None:
        a = np.array([np.nan, 2.0, np.nan, 4.0])
        b = np.array([np.nan, 2.0, 3.0, np.nan])
        np.testing.assert_array_equal(safe_neq(a, b), [0.0, 0.0, 0.0, 0.0])


class TestOutputType:
    @pytest.mark.parametrize(
        "op", [safe_gt, safe_lt, safe_gte, safe_lte, safe_eq, safe_neq]
    )
    def test_output_is_float64(self, op) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 1.0])
        assert op(a, b).dtype == np.float64

    @pytest.mark.parametrize(
        "op", [safe_gt, safe_lt, safe_gte, safe_lte, safe_eq, safe_neq]
    )
    def test_output_length(self, op) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert len(op(a, b)) == 3
