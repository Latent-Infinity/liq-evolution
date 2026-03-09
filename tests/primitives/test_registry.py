"""Tests for the trading primitive registry builder."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from liq.evolution.config import PrimitiveConfig
from liq.evolution.errors import PrimitiveSetupError
from liq.evolution.primitives.registry import build_trading_registry
from liq.gp.errors import PrimitiveError
from liq.gp.primitives.registry import PrimitiveRegistry


class TestBuildTradingRegistry:
    def test_returns_primitive_registry(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        assert isinstance(reg, PrimitiveRegistry)

    def test_default_config_registers_all_categories(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        assert len(reg.list_primitives(category="numeric")) == 12
        assert len(reg.list_primitives(category="comparison")) == 6
        assert len(reg.list_primitives(category="logic")) == 4
        assert len(reg.list_primitives(category="crossover")) == 4
        assert len(reg.list_primitives(category="temporal")) == 11
        assert len(reg.list_primitives(category="terminal")) == 14

    def test_total_primitive_count_default(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        total = len(reg.list_primitives())
        assert total > 100

    def test_disable_numeric(self) -> None:
        cfg = PrimitiveConfig(enable_numeric_ops=False)
        reg = build_trading_registry(cfg)
        assert len(reg.list_primitives(category="numeric")) == 0

    def test_disable_comparison(self) -> None:
        cfg = PrimitiveConfig(enable_comparison_ops=False)
        reg = build_trading_registry(cfg)
        assert len(reg.list_primitives(category="comparison")) == 0

    def test_disable_logic(self) -> None:
        cfg = PrimitiveConfig(enable_logic_ops=False)
        reg = build_trading_registry(cfg)
        assert len(reg.list_primitives(category="logic")) == 0

    def test_disable_crossover(self) -> None:
        cfg = PrimitiveConfig(enable_crossover_ops=False)
        reg = build_trading_registry(cfg)
        assert len(reg.list_primitives(category="crossover")) == 0

    def test_disable_temporal(self) -> None:
        cfg = PrimitiveConfig(enable_temporal_ops=False)
        reg = build_trading_registry(cfg)
        assert len(reg.list_primitives(category="temporal")) == 0

    def test_disable_series_sources(self) -> None:
        cfg = PrimitiveConfig(enable_series_sources=False)
        reg = build_trading_registry(cfg)
        assert len(reg.list_primitives(category="terminal")) == 0

    def test_disable_all(self) -> None:
        cfg = PrimitiveConfig(
            enable_numeric_ops=False,
            enable_comparison_ops=False,
            enable_logic_ops=False,
            enable_crossover_ops=False,
            enable_temporal_ops=False,
            enable_series_sources=False,
        )
        reg = build_trading_registry(cfg)
        assert len(reg.list_primitives()) > 100


class TestBuildTradingRegistryIndicators:
    """Test indicator-backed registry behavior."""

    class _CountingBackend:
        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            self.calls = 0

        def compute(
            self, name: str, params: dict[str, Any], data: dict[str, Any], **kwargs: Any
        ) -> np.ndarray:
            self.calls += 1
            return self.wrapped.compute(name, params, data, **kwargs)

        def list_indicators(
            self,
            category: str | None = None,
        ) -> list[str]:
            return self.wrapped.list_indicators(category)

    @pytest.fixture
    def liq_features_backend(self):
        from liq.evolution.primitives.indicators_liq_ta import LiqFeaturesBackend

        return LiqFeaturesBackend()

    def test_default_build_registers_indicators(self, liq_features_backend) -> None:
        reg = build_trading_registry(PrimitiveConfig(), backend=liq_features_backend)
        indicator_names = [
            name for name, primitive in reg._primitives.items() if primitive.category == "indicator"
        ]
        assert len(indicator_names) > 100
        assert reg.get("rsi") is not None
        with pytest.raises(PrimitiveError):
            reg.get("ta_rsi")
        assert reg.get("sma") is not None

    def test_default_build_registers_indicators_without_explicit_backend(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        indicator_names = [
            name for name, primitive in reg._primitives.items() if primitive.category == "indicator"
        ]
        assert len(indicator_names) > 100
        assert reg.get("rsi") is not None
        assert reg.get("sma") is not None

    def test_default_build_uses_canonical_indicator_names(
        self,
        liq_features_backend,
    ) -> None:
        reg = build_trading_registry(PrimitiveConfig(), backend=liq_features_backend)
        assert reg.get("rsi") is not None
        assert reg.get("macd_signal") is not None
        with pytest.raises(PrimitiveError):
            reg.get("ta_rsi")
        with pytest.raises(PrimitiveError):
            reg.get("ta_macd_signal")

    def test_default_build_preserves_other_categories(self, liq_features_backend) -> None:
        reg = build_trading_registry(PrimitiveConfig(), backend=liq_features_backend)
        # Standard categories still present when not explicitly disabled.
        assert len(reg.list_primitives(category="numeric")) == 12
        assert len(reg.list_primitives(category="temporal")) == 11

    def test_indicator_cache_uses_feature_context(self, liq_features_backend) -> None:
        counting = self._CountingBackend(liq_features_backend)

        reg = build_trading_registry(PrimitiveConfig(), backend=counting)
        sma = reg.get("sma")

        close = np.arange(1.0, 31.0)
        r1 = sma.callable(close, period=14)
        r2 = sma.callable(close, period=14)

        assert r1 is r2
        assert counting.calls == 1

    def test_indicator_cache_varies_by_params(self, liq_features_backend) -> None:
        counting = self._CountingBackend(liq_features_backend)

        reg = build_trading_registry(PrimitiveConfig(), backend=counting)
        sma = reg.get("sma")

        close = np.arange(1.0, 31.0)
        _ = sma.callable(close, period=14)
        _ = sma.callable(close, period=21)

        assert counting.calls == 2


class TestBuildTradingRegistryErrorHandling:
    """Test the PrimitiveSetupError wrapping on failure."""

    def test_registration_error_wrapped(self) -> None:
        with (
            patch(
                "liq.evolution.primitives.registry.register_series_sources",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(PrimitiveSetupError, match="boom"),
        ):
            build_trading_registry(PrimitiveConfig())


def test_public_imports_do_not_expose_legacy_liq_ta_symbols() -> None:
    module = importlib.import_module("liq.evolution")
    assert not hasattr(module, "LiqTAIndicatorBackend")
    assert not hasattr(module, "register_liq_ta_indicators")
