"""Tests for series source terminals and context preparation."""

from __future__ import annotations

import numpy as np

from liq.evolution.primitives.series_sources import (
    prepare_evaluation_context,
    register_series_sources,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import Series

EXPECTED_TERMINALS = {
    "close",
    "open",
    "high",
    "low",
    "volume",
    "log_returns",
    "midrange",
    "typical_price",
    "ha_open",
    "ha_high",
    "ha_low",
    "ha_close",
    "ephemeral_float",
}


class TestRegisterSeriesSources:
    """Verify terminal registration."""

    def test_registers_all_expected_terminals(self) -> None:
        reg = PrimitiveRegistry()
        register_series_sources(reg)
        names = {p.name for p in reg.list_primitives(category="terminal")}
        assert names >= EXPECTED_TERMINALS

    def test_all_have_arity_zero(self) -> None:
        reg = PrimitiveRegistry()
        register_series_sources(reg)
        for p in reg.list_primitives(category="terminal"):
            assert p.arity == 0, f"{p.name} has arity {p.arity}"

    def test_all_output_series(self) -> None:
        reg = PrimitiveRegistry()
        register_series_sources(reg)
        for p in reg.list_primitives(category="terminal"):
            assert p.output_type == Series, f"{p.name} outputs {p.output_type}"

    def test_count(self) -> None:
        reg = PrimitiveRegistry()
        register_series_sources(reg)
        assert len(reg.list_primitives(category="terminal")) == 13


class TestPrepareEvaluationContext:
    """Verify derived series computation."""

    def test_passthrough_ohlcv(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        for key in ("open", "high", "low", "close", "volume"):
            np.testing.assert_array_equal(ctx[key], sample_ohlcv[key])

    def test_does_not_mutate_input(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        orig_close = sample_ohlcv["close"].copy()
        prepare_evaluation_context(sample_ohlcv)
        np.testing.assert_array_equal(sample_ohlcv["close"], orig_close)

    def test_log_returns_bar0_nan(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        assert np.isnan(ctx["log_returns"][0])

    def test_log_returns_values(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        close = sample_ohlcv["close"]
        expected = np.log(close[1:] / close[:-1])
        np.testing.assert_allclose(ctx["log_returns"][1:], expected)

    def test_midrange(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        expected = (sample_ohlcv["high"] + sample_ohlcv["low"]) / 2.0
        np.testing.assert_allclose(ctx["midrange"], expected)

    def test_typical_price(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        expected = (
            sample_ohlcv["high"] + sample_ohlcv["low"] + sample_ohlcv["close"]
        ) / 3.0
        np.testing.assert_allclose(ctx["typical_price"], expected)

    def test_ha_close(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        expected = (
            sample_ohlcv["open"]
            + sample_ohlcv["high"]
            + sample_ohlcv["low"]
            + sample_ohlcv["close"]
        ) / 4.0
        np.testing.assert_allclose(ctx["ha_close"], expected)

    def test_ha_open_bar0(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        expected_0 = (sample_ohlcv["open"][0] + sample_ohlcv["close"][0]) / 2.0
        np.testing.assert_allclose(ctx["ha_open"][0], expected_0)

    def test_ha_open_recursive(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        # ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        for i in range(1, 5):
            expected = (ctx["ha_open"][i - 1] + ctx["ha_close"][i - 1]) / 2.0
            np.testing.assert_allclose(ctx["ha_open"][i], expected)

    def test_ha_high(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        expected = np.maximum(
            sample_ohlcv["high"],
            np.maximum(ctx["ha_open"], ctx["ha_close"]),
        )
        np.testing.assert_allclose(ctx["ha_high"], expected)

    def test_ha_low(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        expected = np.minimum(
            sample_ohlcv["low"],
            np.minimum(ctx["ha_open"], ctx["ha_close"]),
        )
        np.testing.assert_allclose(ctx["ha_low"], expected)

    def test_all_outputs_same_length(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        n = len(sample_ohlcv["close"])
        for key in EXPECTED_TERMINALS:
            assert len(ctx[key]) == n, f"{key} has length {len(ctx[key])}"

    def test_all_outputs_float64(self, sample_ohlcv: dict[str, np.ndarray]) -> None:
        ctx = prepare_evaluation_context(sample_ohlcv)
        for key in EXPECTED_TERMINALS:
            assert ctx[key].dtype == np.float64, f"{key} has dtype {ctx[key].dtype}"
