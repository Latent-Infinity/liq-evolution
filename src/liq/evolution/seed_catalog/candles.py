"""Seed strategy implementations and registrations for this domain."""

from __future__ import annotations

from liq.evolution.program import FunctionNode
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog._core import (
    Program,
    StrategySeedTemplate,
    _resolve_primitive,
    _terminal,
)


def build_cdl_engulfing_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source_open: str = "open",
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
) -> Program:
    """Candlestick bullish pattern: engulfing signal above zero."""
    return FunctionNode(
        _resolve_primitive(
            registry,
            "ta_cdl_engulfing",
            seed="cdl_engulfing_bullish",
        ),
        (
            _terminal(source_open),
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
    )


def build_cdl_evening_star_bearish_seed(
    registry: PrimitiveRegistry,
    *,
    source_open: str = "open",
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
) -> Program:
    """Candlestick bearish evening-star pattern."""
    return FunctionNode(
        _resolve_primitive(
            registry,
            "ta_cdl_evening_star",
            seed="cdl_evening_star_bearish",
        ),
        (
            _terminal(source_open),
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
    )


def build_cdl_hammer_seed(
    registry: PrimitiveRegistry,
    *,
    source_open: str = "open",
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
) -> Program:
    """Candlestick hammer-like setup: hammer signal above zero."""
    return FunctionNode(
        _resolve_primitive(registry, "ta_cdl_hammer", seed="cdl_hammer"),
        (
            _terminal(source_open),
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
    )


def build_cdl_morning_star_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source_open: str = "open",
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
) -> Program:
    """Candlestick bullish morning-star pattern."""
    return FunctionNode(
        _resolve_primitive(
            registry,
            "ta_cdl_morning_star",
            seed="cdl_morning_star_bullish",
        ),
        (
            _terminal(source_open),
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
    )


SEED_TEMPLATES = [
    StrategySeedTemplate(
        name="cdl_engulfing_bullish",
        description="Bullish engulfing candlestick pattern",
        builder=build_cdl_engulfing_bullish_seed,
    ),
    StrategySeedTemplate(
        name="cdl_hammer",
        description="Hammer candlestick pattern",
        builder=build_cdl_hammer_seed,
    ),
    StrategySeedTemplate(
        name="cdl_morning_star_bullish",
        description="Bullish morning-star candlestick pattern",
        builder=build_cdl_morning_star_bullish_seed,
    ),
    StrategySeedTemplate(
        name="cdl_evening_star_bearish",
        description="Bearish evening-star candlestick pattern",
        builder=build_cdl_evening_star_bearish_seed,
    ),
]

SEED_NAMES = [template.name for template in SEED_TEMPLATES]
