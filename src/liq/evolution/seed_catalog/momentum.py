"""Seed strategy implementations and registrations for this domain."""

from __future__ import annotations

from liq.evolution.program import ConstantNode, FunctionNode, ParameterizedNode
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog._core import (
    Program,
    StrategySeedTemplate,
    _resolve_primitive,
    _terminal,
)


def build_apo_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
) -> Program:
    """Absolute price oscillator confirmation: APO crosses above zero."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("apo periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")

    apo = ParameterizedNode(
        _resolve_primitive(registry, "ta_apo", seed="apo_bullish_cross"),
        (_terminal(source),),
        {"fast_period": int(fast_period), "slow_period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="apo_bullish_cross"),
        (apo, ConstantNode(0.0)),
    )


def build_cmo_overbought_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 14,
    threshold: float = 50.0,
) -> Program:
    """CMO overbought condition: CMO > threshold."""
    if period <= 0:
        raise ValueError("cmo period must be positive")
    if not -100 <= threshold <= 100:
        raise ValueError("threshold must be between -100 and 100")

    cmo = ParameterizedNode(
        _resolve_primitive(registry, "ta_cmo", seed="cmo_overbought"),
        (_terminal(source),),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="cmo_overbought"),
        (cmo, ConstantNode(float(threshold))),
    )


def build_cmo_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 14,
    threshold: float = -50.0,
) -> Program:
    """CMO oversold condition: CMO < threshold."""
    if period <= 0:
        raise ValueError("cmo period must be positive")
    if not -100 <= threshold <= 100:
        raise ValueError("threshold must be between -100 and 100")

    cmo = ParameterizedNode(
        _resolve_primitive(registry, "ta_cmo", seed="cmo_oversold"),
        (_terminal(source),),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="cmo_oversold"),
        (cmo, ConstantNode(float(threshold))),
    )


def build_linearreg_slope_up_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 20,
) -> Program:
    """Linear-regression slope confirmation: slope > 0."""
    if period <= 0:
        raise ValueError("linearreg period must be positive")

    slope = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_linearreg_slope",
            seed="linearreg_slope_up",
        ),
        (_terminal(source),),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "gt",
            seed="linearreg_slope_up",
        ),
        (slope, ConstantNode(0.0)),
    )


def build_mom_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 10,
) -> Program:
    """Momentum confirmation: momentum crosses above zero."""
    if period <= 0:
        raise ValueError("momentum period must be positive")

    momentum = ParameterizedNode(
        _resolve_primitive(registry, "ta_mom", seed="mom_bullish_cross"),
        (_terminal(source),),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="mom_bullish_cross"),
        (momentum, ConstantNode(0.0)),
    )


def build_ppo_bearish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
) -> Program:
    """PPO momentum correction: PPO crosses below zero."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("ppo periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")

    source_terminal = _terminal(source)
    ppo = ParameterizedNode(
        _resolve_primitive(registry, "ta_ppo", seed="ppo_bearish_cross"),
        (source_terminal,),
        {"fast_period": int(fast_period), "slow_period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_below", seed="ppo_bearish_cross"),
        (ppo, ConstantNode(0.0)),
    )


def build_ppo_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
) -> Program:
    """PPO momentum confirmation: PPO crosses above zero."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("ppo periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")

    source_terminal = _terminal(source)
    ppo = ParameterizedNode(
        _resolve_primitive(registry, "ta_ppo", seed="ppo_bullish_cross"),
        (source_terminal,),
        {"fast_period": int(fast_period), "slow_period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="ppo_bullish_cross"),
        (ppo, ConstantNode(0.0)),
    )


def build_roc_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 12,
) -> Program:
    """Rate of change momentum: ROC crosses above zero."""
    if period <= 0:
        raise ValueError("roc period must be positive")

    roc = ParameterizedNode(
        _resolve_primitive(registry, "ta_roc", seed="roc_bullish_cross"),
        (_terminal(source),),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="roc_bullish_cross"),
        (roc, ConstantNode(0.0)),
    )


def build_rsi_overbought_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 14,
    threshold: float = 70.0,
) -> Program:
    """RSI overbought signal: ``RSI(source, period) > threshold``."""
    if period <= 0:
        raise ValueError("rsi period must be positive")
    if not 0 <= threshold <= 100:
        raise ValueError("rsi threshold must be between 0 and 100")

    source_terminal = _terminal(source)
    rsi = ParameterizedNode(
        _resolve_primitive(registry, "ta_rsi", seed="rsi_overbought"),
        (source_terminal,),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="rsi_overbought"),
        (rsi, ConstantNode(float(threshold))),
    )


def build_rsi_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 14,
    threshold: float = 30.0,
) -> Program:
    """RSI oversold signal: ``RSI(source, period) < threshold``."""
    if period <= 0:
        raise ValueError("rsi period must be positive")
    if not 0 <= threshold <= 100:
        raise ValueError("rsi threshold must be between 0 and 100")

    source_terminal = _terminal(source)
    rsi = ParameterizedNode(
        _resolve_primitive(registry, "ta_rsi", seed="rsi_oversold"),
        (source_terminal,),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="rsi_oversold"),
        (rsi, ConstantNode(float(threshold))),
    )


def build_stochastic_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    k_period: int = 14,
    d_period: int = 3,
    k_slowing: int = 3,
) -> Program:
    """Stochastic bullish crossover: %K crosses above %D."""
    if k_period <= 0 or d_period <= 0 or k_slowing <= 0:
        raise ValueError("stochastic periods must be positive")

    params = {
        "k_period": int(k_period),
        "d_period": int(d_period),
        "k_slowing": int(k_slowing),
    }
    high = _terminal(source_high)
    low = _terminal(source_low)
    close = _terminal(source_close)
    stoch_k = ParameterizedNode(
        _resolve_primitive(
            registry, "ta_stochastic_k", seed="stochastic_bullish_cross"
        ),
        (high, low, close),
        params,
    )
    stoch_d = ParameterizedNode(
        _resolve_primitive(
            registry, "ta_stochastic_d", seed="stochastic_bullish_cross"
        ),
        (high, low, close),
        params,
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="stochastic_bullish_cross",
        ),
        (stoch_k, stoch_d),
    )


def build_stochastic_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    k_period: int = 14,
    d_period: int = 3,
    k_slowing: int = 3,
    threshold: float = 20.0,
) -> Program:
    """Stochastic oversold signal: %K < threshold."""
    if not 0 <= threshold <= 100:
        raise ValueError("stochastic threshold must be between 0 and 100")
    if k_period <= 0 or d_period <= 0 or k_slowing <= 0:
        raise ValueError("stochastic periods must be positive")

    params = {
        "k_period": int(k_period),
        "d_period": int(d_period),
        "k_slowing": int(k_slowing),
    }
    high = _terminal(source_high)
    low = _terminal(source_low)
    close = _terminal(source_close)
    stoch_k = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_stochastic_k",
            seed="stochastic_oversold",
        ),
        (high, low, close),
        params,
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="stochastic_oversold"),
        (stoch_k, ConstantNode(float(threshold))),
    )


def build_stochrsi_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> Program:
    """StochRSI bullish crossover: fast %K crosses above fast %D."""
    if rsi_period <= 0 or stoch_period <= 0 or k_period <= 0 or d_period <= 0:
        raise ValueError("stochrsi periods must be positive")

    params = {
        "rsi_period": int(rsi_period),
        "stoch_period": int(stoch_period),
        "k_period": int(k_period),
        "d_period": int(d_period),
    }
    source_terminal = _terminal(source)
    fast_k = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_stochrsi_fastk",
            seed="stochrsi_bullish_cross",
        ),
        (source_terminal,),
        params,
    )
    fast_d = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_stochrsi_fastd",
            seed="stochrsi_bullish_cross",
        ),
        (source_terminal,),
        params,
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="stochrsi_bullish_cross",
        ),
        (fast_k, fast_d),
    )


def build_stochrsi_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
    threshold: float = 20.0,
) -> Program:
    """StochRSI oversold signal: fast %K < threshold."""
    if not 0 <= threshold <= 100:
        raise ValueError("stochrsi threshold must be between 0 and 100")
    if rsi_period <= 0 or stoch_period <= 0 or k_period <= 0 or d_period <= 0:
        raise ValueError("stochrsi periods must be positive")

    params = {
        "rsi_period": int(rsi_period),
        "stoch_period": int(stoch_period),
        "k_period": int(k_period),
        "d_period": int(d_period),
    }
    source_terminal = _terminal(source)
    fast_k = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_stochrsi_fastk",
            seed="stochrsi_oversold",
        ),
        (source_terminal,),
        params,
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="stochrsi_oversold"),
        (fast_k, ConstantNode(float(threshold))),
    )


def build_trix_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 15,
) -> Program:
    """TRIX momentum confirmation: TRIX crosses above zero."""
    if period <= 0:
        raise ValueError("trix period must be positive")

    trix = ParameterizedNode(
        _resolve_primitive(registry, "ta_trix", seed="trix_bullish_cross"),
        (_terminal(source),),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="trix_bullish_cross"),
        (trix, ConstantNode(0.0)),
    )


SEED_TEMPLATES = [
    StrategySeedTemplate(
        name="rsi_oversold",
        description="RSI oversold (RSI < threshold)",
        builder=build_rsi_oversold_seed,
    ),
    StrategySeedTemplate(
        name="rsi_overbought",
        description="RSI overbought (RSI > threshold)",
        builder=build_rsi_overbought_seed,
    ),
    StrategySeedTemplate(
        name="stochastic_bullish_cross",
        description="Stochastic fast K crosses above D",
        builder=build_stochastic_bullish_seed,
    ),
    StrategySeedTemplate(
        name="stochrsi_bullish_cross",
        description="StochRSI fast K crosses above fast D",
        builder=build_stochrsi_bullish_seed,
    ),
    StrategySeedTemplate(
        name="stochastic_oversold",
        description="Stochastic fast K below threshold",
        builder=build_stochastic_oversold_seed,
    ),
    StrategySeedTemplate(
        name="stochrsi_oversold",
        description="StochRSI fast K below threshold",
        builder=build_stochrsi_oversold_seed,
    ),
    StrategySeedTemplate(
        name="roc_bullish_cross",
        description="Rate-of-change crosses above zero",
        builder=build_roc_bullish_seed,
    ),
    StrategySeedTemplate(
        name="trix_bullish_cross",
        description="TRIX crosses above zero",
        builder=build_trix_bullish_seed,
    ),
    StrategySeedTemplate(
        name="mom_bullish_cross",
        description="Momentum crosses above zero",
        builder=build_mom_bullish_seed,
    ),
    StrategySeedTemplate(
        name="apo_bullish_cross",
        description="APO crosses above zero",
        builder=build_apo_bullish_seed,
    ),
    StrategySeedTemplate(
        name="cmo_oversold",
        description="CMO below oversold threshold",
        builder=build_cmo_oversold_seed,
    ),
    StrategySeedTemplate(
        name="cmo_overbought",
        description="CMO above overbought threshold",
        builder=build_cmo_overbought_seed,
    ),
    StrategySeedTemplate(
        name="ppo_bullish_cross",
        description="PPO crosses above zero",
        builder=build_ppo_bullish_seed,
    ),
    StrategySeedTemplate(
        name="ppo_bearish_cross",
        description="PPO crosses below zero",
        builder=build_ppo_bearish_seed,
    ),
    StrategySeedTemplate(
        name="linearreg_slope_up",
        description="Linear-regression slope above zero",
        builder=build_linearreg_slope_up_seed,
    ),
]

SEED_NAMES = [template.name for template in SEED_TEMPLATES]
