"""Tests for the trading primitive registry builder."""

from __future__ import annotations

from liq.evolution.config import PrimitiveConfig
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
