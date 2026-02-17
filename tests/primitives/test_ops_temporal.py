"""Tests for temporal/lag operator primitives."""

from __future__ import annotations

import numpy as np
import pytest

from liq.evolution.primitives.ops_temporal import (
    register_temporal_ops,
    safe_change,
    safe_greater_count,
    safe_highest,
    safe_is_falling,
    safe_is_rising,
    safe_lower_count,
    safe_lowest,
    safe_n_bars_ago,
    safe_pct_change,
    safe_percentile_rank,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, Series


class TestRegistration:
    def test_registers_ten_operators(self) -> None:
        reg = PrimitiveRegistry()
        register_temporal_ops(reg)
        assert len(reg.list_primitives(category="temporal")) == 10

    def test_all_parameterized(self) -> None:
        reg = PrimitiveRegistry()
        register_temporal_ops(reg)
        for p in reg.list_primitives(category="temporal"):
            assert len(p.param_specs) > 0, f"{p.name} has no param_specs"

    def test_bool_output_types(self) -> None:
        reg = PrimitiveRegistry()
        register_temporal_ops(reg)
        prims = {p.name: p for p in reg.list_primitives(category="temporal")}
        assert prims["is_rising"].output_type == BoolSeries
        assert prims["is_falling"].output_type == BoolSeries

    def test_series_output_types(self) -> None:
        reg = PrimitiveRegistry()
        register_temporal_ops(reg)
        prims = {p.name: p for p in reg.list_primitives(category="temporal")}
        for name in [
            "n_bars_ago",
            "highest",
            "lowest",
            "percentile_rank",
            "greater_count",
            "lower_count",
            "change",
            "pct_change",
        ]:
            assert prims[name].output_type == Series, f"{name}"


class TestIsRising:
    def test_basic(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_is_rising(a, period=3)
        # bars 0,1 insufficient lookback -> 0.0
        # bar 2: a[2] > a[0] -> 3 > 1 -> True
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 1.0

    def test_flat_not_rising(self) -> None:
        a = np.array([5.0, 5.0, 5.0, 5.0])
        result = safe_is_rising(a, period=3)
        np.testing.assert_array_equal(result[2:], [0.0, 0.0])


class TestIsFalling:
    def test_basic(self) -> None:
        a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = safe_is_falling(a, period=3)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 1.0


class TestNBarsAgo:
    def test_shift_1(self) -> None:
        a = np.array([10.0, 20.0, 30.0, 40.0])
        result = safe_n_bars_ago(a, shift=1)
        assert np.isnan(result[0])
        np.testing.assert_array_equal(result[1:], [10.0, 20.0, 30.0])

    def test_shift_2(self) -> None:
        a = np.array([10.0, 20.0, 30.0, 40.0])
        result = safe_n_bars_ago(a, shift=2)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_array_equal(result[2:], [10.0, 20.0])


class TestHighest:
    def test_basic(self) -> None:
        a = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = safe_highest(a, period=3)
        # bars 0,1 insufficient -> NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # bar 2: max(1,5,3) = 5
        assert result[2] == 5.0
        # bar 3: max(5,3,2) = 5
        assert result[3] == 5.0
        # bar 4: max(3,2,4) = 4
        assert result[4] == 4.0


class TestLowest:
    def test_basic(self) -> None:
        a = np.array([5.0, 1.0, 3.0, 4.0, 2.0])
        result = safe_lowest(a, period=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # bar 2: min(5,1,3) = 1
        assert result[2] == 1.0
        # bar 3: min(1,3,4) = 1
        assert result[3] == 1.0
        # bar 4: min(3,4,2) = 2
        assert result[4] == 2.0


class TestPercentileRank:
    def test_basic(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_percentile_rank(a, period=5)
        # bars 0-3 insufficient -> NaN
        for i in range(4):
            assert np.isnan(result[i])
        # bar 4: 5 is the highest of [1,2,3,4,5] -> rank 100
        assert result[4] == 100.0

    def test_lowest_value(self) -> None:
        a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = safe_percentile_rank(a, period=5)
        # bar 4: 1 is lowest -> rank 0
        assert result[4] == 0.0


class TestGreaterCount:
    def test_basic(self) -> None:
        a = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        b = np.array([3.0, 6.0, 4.0, 7.0, 2.0])
        result = safe_greater_count(a, b, period=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # bar 2: count a>=b in [0,1,2] -> (5>=3, 5>=6, 5>=4) -> 2
        assert result[2] == 2.0
        # bar 3: count a>=b in [1,2,3] -> (5>=6, 5>=4, 5>=7) -> 1
        assert result[3] == 1.0

    def test_inclusive(self) -> None:
        a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        b = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
        result = safe_greater_count(a, b, period=3)
        assert result[2] == 2.0  # [1>=1,1>=2,1>=1]
        assert result[3] == 1.0  # [1>=2,1>=1,1>=3]


class TestLowerCount:
    def test_basic(self) -> None:
        a = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        b = np.array([3.0, 6.0, 4.0, 7.0, 2.0])
        result = safe_lower_count(a, b, period=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        # bar 2: count a<=b in [0,1,2] -> (5<=3=F, 5<=6=T, 5<=4=F) -> 1
        assert result[2] == 1.0

    def test_inclusive(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 6.0, 4.0])
        result = safe_lower_count(a, b, period=3)
        assert result[2] == 3.0  # [1<=1,2<=2,3<=3]
        assert result[4] == 2.0  # [3<=3,4<=6,5<=4]


class TestChange:
    def test_basic(self) -> None:
        a = np.array([10.0, 12.0, 15.0, 11.0])
        result = safe_change(a, period=1)
        assert np.isnan(result[0])
        assert result[1] == 2.0
        assert result[2] == 3.0
        assert result[3] == -4.0

    def test_period_2(self) -> None:
        a = np.array([10.0, 12.0, 15.0, 11.0])
        result = safe_change(a, period=2)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 5.0  # 15 - 10
        assert result[3] == -1.0  # 11 - 12


class TestPctChange:
    def test_basic(self) -> None:
        a = np.array([10.0, 12.0, 15.0])
        result = safe_pct_change(a, period=1)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], 0.2)  # (12-10)/10
        np.testing.assert_allclose(result[2], 0.25)  # (15-12)/12

    def test_zero_denominator(self) -> None:
        a = np.array([0.0, 5.0])
        result = safe_pct_change(a, period=1)
        assert np.isnan(result[1])  # division by zero


class TestOutputLength:
    @pytest.mark.parametrize(
        "op,kwargs",
        [
            (safe_is_rising, {"period": 3}),
            (safe_is_falling, {"period": 3}),
            (safe_n_bars_ago, {"shift": 1}),
            (safe_highest, {"period": 3}),
            (safe_lowest, {"period": 3}),
            (safe_change, {"period": 1}),
            (safe_pct_change, {"period": 1}),
        ],
    )
    def test_unary_length(self, op, kwargs) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(op(a, **kwargs)) == 5

    @pytest.mark.parametrize(
        "op,kwargs",
        [
            (safe_greater_count, {"period": 3}),
            (safe_lower_count, {"period": 3}),
        ],
    )
    def test_binary_length(self, op, kwargs) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert len(op(a, b, **kwargs)) == 5


class TestSafeChangeEdgeCases:
    """Edge cases for safe_change: period >= len(a)."""

    def test_period_equals_length_all_nan(self) -> None:
        a = np.array([10.0, 20.0, 30.0])
        result = safe_change(a, period=3)
        assert len(result) == 3
        assert np.all(np.isnan(result))

    def test_period_exceeds_length_all_nan(self) -> None:
        a = np.array([10.0, 20.0])
        result = safe_change(a, period=5)
        assert len(result) == 2
        assert np.all(np.isnan(result))

    def test_single_element_period_1(self) -> None:
        a = np.array([42.0])
        result = safe_change(a, period=1)
        assert len(result) == 1
        assert np.isnan(result[0])


class TestSafePctChangeEdgeCases:
    """Edge cases for safe_pct_change: period >= len(a)."""

    def test_period_equals_length_all_nan(self) -> None:
        a = np.array([10.0, 20.0, 30.0])
        result = safe_pct_change(a, period=3)
        assert len(result) == 3
        assert np.all(np.isnan(result))

    def test_period_exceeds_length_all_nan(self) -> None:
        a = np.array([10.0])
        result = safe_pct_change(a, period=5)
        assert len(result) == 1
        assert np.isnan(result[0])


class TestSafeNBarsAgoEdgeCases:
    """Edge cases for safe_n_bars_ago: shift >= len(a)."""

    def test_shift_equals_length_all_nan(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        result = safe_n_bars_ago(a, shift=3)
        assert len(result) == 3
        assert np.all(np.isnan(result))

    def test_shift_exceeds_length_all_nan(self) -> None:
        a = np.array([1.0, 2.0])
        result = safe_n_bars_ago(a, shift=10)
        assert len(result) == 2
        assert np.all(np.isnan(result))
