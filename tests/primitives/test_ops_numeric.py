"""Tests for numeric operator primitives."""

from __future__ import annotations

import numpy as np
import pytest

from liq.evolution.primitives.ops_numeric import (
    register_numeric_ops,
    safe_abs,
    safe_add,
    safe_clip,
    safe_div,
    safe_log,
    safe_max_of,
    safe_min_of,
    safe_mul,
    safe_neg,
    safe_sqrt,
    safe_sub,
    safe_zscore,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import Series


class TestRegistration:
    def test_registers_twelve_operators(self) -> None:
        reg = PrimitiveRegistry()
        register_numeric_ops(reg)
        assert len(reg.list_primitives(category="numeric")) == 12

    def test_all_output_series(self) -> None:
        reg = PrimitiveRegistry()
        register_numeric_ops(reg)
        for p in reg.list_primitives(category="numeric"):
            assert p.output_type == Series


class TestAdd:
    def test_basic(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(safe_add(a, b), [5.0, 7.0, 9.0])

    def test_nan_propagation(self) -> None:
        a = np.array([np.nan, 1.0])
        b = np.array([1.0, 2.0])
        result = safe_add(a, b)
        assert np.isnan(result[0])
        assert result[1] == 3.0


class TestSub:
    def test_basic(self) -> None:
        np.testing.assert_array_equal(
            safe_sub(np.array([5.0, 3.0]), np.array([2.0, 1.0])), [3.0, 2.0]
        )


class TestMul:
    def test_basic(self) -> None:
        np.testing.assert_array_equal(
            safe_mul(np.array([2.0, 3.0]), np.array([4.0, 5.0])), [8.0, 15.0]
        )


class TestDiv:
    def test_basic(self) -> None:
        np.testing.assert_array_equal(
            safe_div(np.array([6.0, 8.0]), np.array([2.0, 4.0])), [3.0, 2.0]
        )

    def test_div_by_zero_nan(self) -> None:
        result = safe_div(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        assert np.all(np.isnan(result))

    def test_inf_replaced_with_nan(self) -> None:
        result = safe_div(np.array([1.0]), np.array([0.0]))
        assert np.isnan(result[0])


class TestAbs:
    def test_basic(self) -> None:
        np.testing.assert_array_equal(
            safe_abs(np.array([-1.0, 2.0, -3.0])), [1.0, 2.0, 3.0]
        )


class TestNeg:
    def test_basic(self) -> None:
        np.testing.assert_array_equal(
            safe_neg(np.array([1.0, -2.0, 0.0])), [-1.0, 2.0, 0.0]
        )


class TestClip:
    def test_basic(self) -> None:
        data = np.array([1.0, 5.0, 10.0])
        lo = np.array([2.0, 2.0, 2.0])
        hi = np.array([8.0, 8.0, 8.0])
        np.testing.assert_array_equal(safe_clip(data, lo, hi), [2.0, 5.0, 8.0])


class TestZscore:
    def test_basic(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_zscore(a)
        np.testing.assert_allclose(np.mean(result), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(result), 1.0, atol=1e-10)

    def test_zero_variance(self) -> None:
        a = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(safe_zscore(a), [0.0, 0.0, 0.0])


class TestLog:
    def test_positive(self) -> None:
        np.testing.assert_allclose(
            safe_log(np.array([1.0, np.e])), [0.0, 1.0], atol=1e-10
        )

    def test_nonpositive_nan(self) -> None:
        result = safe_log(np.array([0.0, -1.0]))
        assert np.all(np.isnan(result))


class TestSqrt:
    def test_positive(self) -> None:
        np.testing.assert_allclose(
            safe_sqrt(np.array([4.0, 9.0])), [2.0, 3.0], atol=1e-10
        )

    def test_negative_nan(self) -> None:
        result = safe_sqrt(np.array([-1.0]))
        assert np.isnan(result[0])


class TestMinMax:
    def test_min_of(self) -> None:
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(safe_min_of(a, b), [1.0, 4.0, 3.0])

    def test_max_of(self) -> None:
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_equal(safe_max_of(a, b), [2.0, 5.0, 6.0])


class TestNoInputMutation:
    def test_div_no_mutation(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 1.0])
        a_orig = a.copy()
        b_orig = b.copy()
        safe_div(a, b)
        np.testing.assert_array_equal(a, a_orig)
        np.testing.assert_array_equal(b, b_orig)

    def test_log_no_mutation(self) -> None:
        a = np.array([0.0, 1.0])
        a_orig = a.copy()
        safe_log(a)
        np.testing.assert_array_equal(a, a_orig)


class TestOutputLength:
    @pytest.mark.parametrize("op", [safe_add, safe_sub, safe_mul, safe_div])
    def test_binary_length(self, op) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert len(op(a, b)) == 3

    @pytest.mark.parametrize("op", [safe_abs, safe_neg, safe_log, safe_sqrt])
    def test_unary_length(self, op) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert len(op(a)) == 3
