"""Tests for crossover detection operator primitives."""

from __future__ import annotations

import numpy as np
import pytest

from liq.evolution.primitives.ops_crossover import (
    register_crossover_ops,
    safe_closes_above,
    safe_closes_below,
    safe_crosses_above,
    safe_crosses_below,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries


class TestRegistration:
    def test_registers_four_operators(self) -> None:
        reg = PrimitiveRegistry()
        register_crossover_ops(reg)
        assert len(reg.list_primitives(category="crossover")) == 4

    def test_all_output_bool_series(self) -> None:
        reg = PrimitiveRegistry()
        register_crossover_ops(reg)
        for p in reg.list_primitives(category="crossover"):
            assert p.output_type == BoolSeries


class TestCrossesAbove:
    def test_basic_crossing(self) -> None:
        # A crosses above B at bar 2 (was below, now above)
        a = np.array([1.0, 2.0, 4.0, 5.0])
        b = np.array([3.0, 3.0, 3.0, 3.0])
        result = safe_crosses_above(a, b)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 0.0])

    def test_bar_zero_always_zero(self) -> None:
        a = np.array([5.0, 5.0])
        b = np.array([1.0, 1.0])
        result = safe_crosses_above(a, b)
        assert result[0] == 0.0

    def test_touch_without_cross(self) -> None:
        # A touches B but doesn't cross above (equal is not above)
        a = np.array([1.0, 3.0, 3.0])
        b = np.array([3.0, 3.0, 3.0])
        result = safe_crosses_above(a, b)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_already_above_no_cross(self) -> None:
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([3.0, 3.0, 3.0])
        result = safe_crosses_above(a, b)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])


class TestCrossesBelow:
    def test_basic_crossing(self) -> None:
        # A crosses below B at bar 2 (was above, now below)
        a = np.array([5.0, 4.0, 2.0, 1.0])
        b = np.array([3.0, 3.0, 3.0, 3.0])
        result = safe_crosses_below(a, b)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 0.0])

    def test_bar_zero_always_zero(self) -> None:
        a = np.array([1.0, 1.0])
        b = np.array([5.0, 5.0])
        result = safe_crosses_below(a, b)
        assert result[0] == 0.0


class TestClosesAbove:
    def test_basic(self) -> None:
        a = np.array([3.0, 1.0, 2.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_closes_above(a, b), [1.0, 0.0, 0.0])


class TestClosesBelow:
    def test_basic(self) -> None:
        a = np.array([1.0, 3.0, 2.0])
        b = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(safe_closes_below(a, b), [1.0, 0.0, 0.0])


class TestOutputType:
    @pytest.mark.parametrize(
        "op",
        [safe_crosses_above, safe_crosses_below, safe_closes_above, safe_closes_below],
    )
    def test_output_float64(self, op) -> None:
        a = np.array([1.0, 3.0, 5.0])
        b = np.array([2.0, 2.0, 2.0])
        assert op(a, b).dtype == np.float64

    @pytest.mark.parametrize(
        "op",
        [safe_crosses_above, safe_crosses_below, safe_closes_above, safe_closes_below],
    )
    def test_output_length(self, op) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert len(op(a, b)) == 3
