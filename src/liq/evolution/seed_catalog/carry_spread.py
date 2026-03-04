"""Seed strategy implementations and registrations for carry/spread motif family."""

from __future__ import annotations

from liq.evolution.program import FunctionNode, ParameterizedNode
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog._core import (
    Program,
    StrategySeedTemplate,
    _resolve_primitive,
    _terminal,
)


def _build_atr_nodes(
    registry: PrimitiveRegistry,
    spread_period: int,
    spread_smooth_period: int,
) -> tuple[ParameterizedNode, ParameterizedNode]:
    """Build ATR spread and smoothed spread nodes for spread-proxy motifs."""
    atr = _resolve_primitive(registry, "atr", seed="carry_spread")
    if atr.arity == 1:
        atr_inputs = (_terminal("close"),)
    elif atr.arity == 3:
        atr_inputs = (
            _terminal("high"),
            _terminal("low"),
            _terminal("close"),
        )
    else:
        raise ValueError(f"carry_spread requires ATR arity 1 or 3, got {atr.arity}")

    spread = ParameterizedNode(
        atr,
        atr_inputs,
        {"period": spread_period},
    )
    spread_smooth = ParameterizedNode(
        _resolve_primitive(registry, "ta_sma", seed="carry_spread"),
        (spread,),
        {"period": spread_smooth_period},
    )
    return spread, spread_smooth


def build_carry_spread_expansion_seed(
    registry: PrimitiveRegistry,
    *,
    spread_period: int = 14,
    spread_smooth_period: int = 40,
) -> Program:
    """Spread-proxy expansion: ATR spread above smoothed spread."""
    if spread_period <= 0:
        raise ValueError("spread period must be positive")
    if spread_smooth_period <= 0:
        raise ValueError("spread_smooth_period must be positive")
    if spread_smooth_period <= spread_period:
        raise ValueError("spread_smooth_period must be larger than spread_period")

    spread, spread_smooth = _build_atr_nodes(
        registry,
        spread_period=spread_period,
        spread_smooth_period=spread_smooth_period,
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="carry_spread_expansion"),
        (spread, spread_smooth),
    )


def build_carry_spread_contraction_seed(
    registry: PrimitiveRegistry,
    *,
    spread_period: int = 14,
    spread_smooth_period: int = 40,
) -> Program:
    """Spread-proxy contraction: ATR spread below smoothed spread."""
    if spread_period <= 0:
        raise ValueError("spread period must be positive")
    if spread_smooth_period <= 0:
        raise ValueError("spread_smooth_period must be positive")
    if spread_smooth_period <= spread_period:
        raise ValueError("spread_smooth_period must be larger than spread_period")

    spread, spread_smooth = _build_atr_nodes(
        registry,
        spread_period=spread_period,
        spread_smooth_period=spread_smooth_period,
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="carry_spread_contraction"),
        (spread, spread_smooth),
    )


SEED_TEMPLATES = [
    StrategySeedTemplate(
        name="carry_spread_expansion",
        description="ATR spread proxy expands above smoothed spread",
        builder=build_carry_spread_expansion_seed,
        regime_hints=("spread", "volatility"),
        turnover_expectation=0.2,
        failure_modes=("spread_warming", "flat_market"),
    ),
    StrategySeedTemplate(
        name="carry_spread_contraction",
        description="ATR spread proxy contracts below smoothed spread",
        builder=build_carry_spread_contraction_seed,
        regime_hints=("spread", "volatility"),
        turnover_expectation=0.2,
        failure_modes=("spread_warming", "flat_market"),
    ),
]

SEED_NAMES = [template.name for template in SEED_TEMPLATES]
