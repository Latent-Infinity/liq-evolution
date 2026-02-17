"""Tests for the trading primitive registry builder."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from liq.evolution.config import PrimitiveConfig
from liq.evolution.errors import PrimitiveSetupError
from liq.evolution.primitives.registry import build_trading_registry
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
        assert len(reg.list_primitives(category="temporal")) == 10
        assert len(reg.list_primitives(category="terminal")) == 13

    def test_total_primitive_count_default(self) -> None:
        reg = build_trading_registry(PrimitiveConfig())
        total = len(reg.list_primitives())
        assert total == 49  # 13+6+4+4+10+12

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

    def test_liq_ta_disabled_by_default(self) -> None:
        cfg = PrimitiveConfig()
        assert cfg.enable_liq_ta is False

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
        assert len(reg.list_primitives()) == 0


class TestBuildTradingRegistryLiqTA:
    """Test the liq-ta branch in build_trading_registry."""

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
    def liq_ta_backend(self):
        liq_ta = pytest.importorskip("liq_ta")  # noqa: F841
        from liq.evolution.primitives.indicators_liq_ta import LiqTAIndicatorBackend

        return LiqTAIndicatorBackend()

    def test_enable_liq_ta_registers_indicators(self, liq_ta_backend) -> None:
        cfg = PrimitiveConfig(enable_liq_ta=True)
        reg = build_trading_registry(cfg, backend=liq_ta_backend)
        indicators = reg.list_primitives(category="indicator")
        assert len(indicators) > 100

    def test_enable_liq_ta_without_backend_no_indicators(self) -> None:
        cfg = PrimitiveConfig(enable_liq_ta=True)
        reg = build_trading_registry(cfg, backend=None)
        indicators = reg.list_primitives(category="indicator")
        assert len(indicators) == 0

    def test_enable_liq_ta_preserves_other_categories(self, liq_ta_backend) -> None:
        cfg = PrimitiveConfig(enable_liq_ta=True)
        reg = build_trading_registry(cfg, backend=liq_ta_backend)
        # Standard categories still present
        assert len(reg.list_primitives(category="numeric")) == 12
        assert len(reg.list_primitives(category="temporal")) == 10

    def test_enable_liq_ta_uses_feature_context_caching(self, liq_ta_backend) -> None:
        cfg = PrimitiveConfig(enable_liq_ta=True)
        counting = self._CountingBackend(liq_ta_backend)

        reg = build_trading_registry(cfg, backend=counting)
        sma = reg.get("ta_sma")

        close = np.arange(1.0, 31.0)
        r1 = sma.callable(close, period=14)
        r2 = sma.callable(close, period=14)

        assert r1 is r2
        assert counting.calls == 1

    def test_enable_liq_ta_cache_varies_by_params(self, liq_ta_backend) -> None:
        cfg = PrimitiveConfig(enable_liq_ta=True)
        counting = self._CountingBackend(liq_ta_backend)

        reg = build_trading_registry(cfg, backend=counting)
        sma = reg.get("ta_sma")

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
