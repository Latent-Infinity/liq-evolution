"""Seed strategy implementations and registrations for this domain."""

from __future__ import annotations

from liq.evolution.program import FunctionNode, ParameterizedNode
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog._core import (
    Program,
    StrategySeedTemplate,
    _resolve_primitive,
    _terminal,
    _validate_indicator_periods,
)


def build_bollinger_breakout_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 20,
    std_dev: float = 2.0,
) -> Program:
    """Bollinger breakout: ``source > BB_upper(source)``."""
    if period <= 0:
        raise ValueError("bollinger period must be positive")
    if std_dev <= 0:
        raise ValueError("std_dev must be greater than 0")

    source_terminal = _terminal(source)
    upper = ParameterizedNode(
        _resolve_primitive(registry, "ta_bollinger_upper", seed="bollinger_breakout"),
        (source_terminal,),
        {"period": int(period), "std_dev": float(std_dev)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="bollinger_breakout"),
        (source_terminal, upper),
    )


def build_dema_crossover_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 9,
    slow_period: int = 21,
) -> Program:
    """DEMA crossover: fast DEMA above slow DEMA."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("dema periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)

    source_terminal = _terminal(source)
    dema_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_dema", seed="dema_crossover"),
        (source_terminal,),
        {"period": int(fast_period)},
    )
    dema_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_dema", seed="dema_crossover"),
        (source_terminal,),
        {"period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="dema_crossover"),
        (dema_fast, dema_slow),
    )


def build_donchian_breakout_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 20,
    direction: str = "up",
) -> Program:
    """Donchian breakout: cross above upper or below lower channel."""
    if period <= 0:
        raise ValueError("donchian period must be positive")
    direction = direction.strip().lower()
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")

    source_terminal = _terminal(source)
    if direction == "up":
        boundary = ParameterizedNode(
            _resolve_primitive(registry, "ta_donchian_upper", seed="donchian_breakout"),
            (_terminal("high"), _terminal("low")),
            {"period": int(period)},
        )
        op = "gt"
    else:
        boundary = ParameterizedNode(
            _resolve_primitive(
                registry,
                "ta_donchian_lower",
                seed="donchian_breakout",
            ),
            (_terminal("high"), _terminal("low")),
            {"period": int(period)},
        )
        op = "lt"
    return FunctionNode(
        _resolve_primitive(
            registry,
            op,
            seed="donchian_breakout",
        ),
        (source_terminal, boundary),
    )


def build_ema_bearish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
) -> Program:
    """EMA bearish crossover: ``EMA(fast, source) < EMA(slow, source)``."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("ema periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)

    source_terminal = _terminal(source)
    ema_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_ema", seed="ema_bearish_cross"),
        (source_terminal,),
        {"period": int(fast_period)},
    )
    ema_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_ema", seed="ema_bearish_cross"),
        (source_terminal,),
        {"period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_below", seed="ema_bearish_cross"),
        (ema_fast, ema_slow),
    )


def build_ema_crossover_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
) -> Program:
    """EMA crossover: ``EMA(fast, source) > EMA(slow, source)``."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("ema periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)

    source_terminal = _terminal(source)
    ema_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_ema", seed="ema_crossover"),
        (source_terminal,),
        {"period": int(fast_period)},
    )
    ema_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_ema", seed="ema_crossover"),
        (source_terminal,),
        {"period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="ema_crossover"),
        (ema_fast, ema_slow),
    )


def build_ema_wilder_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 10,
    slow_period: int = 30,
) -> Program:
    """Wilder EMA crossover: fast EMA-Wilder above slow EMA-Wilder."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("ema_wilder periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)

    source_terminal = _terminal(source)
    ema_fast = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_ema_wilder",
            seed="ema_wilder_bullish_cross",
        ),
        (source_terminal,),
        {"period": int(fast_period)},
    )
    ema_slow = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_ema_wilder",
            seed="ema_wilder_bullish_cross",
        ),
        (source_terminal,),
        {"period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="ema_wilder_bullish_cross",
        ),
        (ema_fast, ema_slow),
    )


def build_kama_trend_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 10,
    slow_period: int = 30,
    adapt_fast_period: int = 2,
    adapt_slow_period: int = 30,
) -> Program:
    """KAMA trend confirmation: KAMA fast above KAMA slow."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("kama periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)
    if adapt_fast_period <= 0 or adapt_slow_period <= 0:
        raise ValueError("kama adaptive periods must be positive")
    if adapt_fast_period >= adapt_slow_period:
        raise ValueError("kama adaptive fast period must be smaller")

    source_terminal = _terminal(source)
    kama_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_kama", seed="kama_trend"),
        (source_terminal,),
        {
            "period": int(fast_period),
            "fast_period": int(adapt_fast_period),
            "slow_period": int(adapt_slow_period),
        },
    )
    kama_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_kama", seed="kama_trend"),
        (source_terminal,),
        {
            "period": int(slow_period),
            "fast_period": int(adapt_fast_period),
            "slow_period": int(adapt_slow_period),
        },
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="kama_trend"),
        (kama_fast, kama_slow),
    )


def build_macd_bearish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Program:
    """MACD bearish crossover: ``MACD < signal``."""
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("macd periods must be positive")
    if slow_period <= fast_period:
        raise ValueError("fast_period must be smaller than slow_period")

    source_terminal = _terminal(source)
    params = {
        "fast_period": int(fast_period),
        "slow_period": int(slow_period),
        "signal_period": int(signal_period),
    }
    macd_line = ParameterizedNode(
        _resolve_primitive(registry, "ta_macd_macd_line", seed="macd_bearish_cross"),
        (source_terminal,),
        params,
    )
    signal_line = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_macd_signal_line",
            seed="macd_bearish_cross",
        ),
        (source_terminal,),
        params,
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_below",
            seed="macd_bearish_cross",
        ),
        (macd_line, signal_line),
    )


def build_macd_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Program:
    """MACD bullish crossover: ``MACD > signal``."""
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("macd periods must be positive")
    if slow_period <= fast_period:
        raise ValueError("fast_period must be smaller than slow_period")

    source_terminal = _terminal(source)
    params = {
        "fast_period": int(fast_period),
        "slow_period": int(slow_period),
        "signal_period": int(signal_period),
    }
    macd_line = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_macd_macd_line",
            seed="macd_bullish_cross",
        ),
        (source_terminal,),
        params,
    )
    signal_line = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_macd_signal_line",
            seed="macd_bullish_cross",
        ),
        (source_terminal,),
        params,
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="macd_bullish_cross",
        ),
        (macd_line, signal_line),
    )


def build_sma_crossover_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 10,
    slow_period: int = 30,
) -> Program:
    """SMA crossover: ``SMA(fast, source) > SMA(slow, source)``."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("sma periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)

    source_terminal = _terminal(source)
    sma_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_sma", seed="sma_crossover"),
        (source_terminal,),
        {"period": int(fast_period)},
    )
    sma_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_sma", seed="sma_crossover"),
        (source_terminal,),
        {"period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="sma_crossover"),
        (sma_fast, sma_slow),
    )


def build_t3_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 10,
    slow_period: int = 30,
    vfactor: float = 0.7,
) -> Program:
    """T3 crossover: fast T3 above slow T3."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("t3 periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)
    if not 0.1 <= vfactor <= 10:
        raise ValueError("vfactor must be between 0.1 and 10")

    source_terminal = _terminal(source)
    t3_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_t3", seed="t3_bullish_cross"),
        (source_terminal,),
        {"period": int(fast_period), "vfactor": float(vfactor)},
    )
    t3_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_t3", seed="t3_bullish_cross"),
        (source_terminal,),
        {"period": int(slow_period), "vfactor": float(vfactor)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="t3_bullish_cross"),
        (t3_fast, t3_slow),
    )


def build_tema_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 10,
    slow_period: int = 30,
) -> Program:
    """TEMA crossover: fast TEMA above slow TEMA."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("tema periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)

    source_terminal = _terminal(source)
    tema_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_tema", seed="tema_bullish_cross"),
        (source_terminal,),
        {"period": int(fast_period)},
    )
    tema_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_tema", seed="tema_bullish_cross"),
        (source_terminal,),
        {"period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="tema_bullish_cross"),
        (tema_fast, tema_slow),
    )


def build_vwap_support_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    source_volume: str = "volume",
    op: str = "gt",
) -> Program:
    """VWAP trend filter: close above/below VWAP."""
    source = _terminal(source_close)
    vwap = ParameterizedNode(
        _resolve_primitive(registry, "ta_vwap", seed="vwap_support"),
        (
            _terminal(source_high),
            _terminal(source_low),
            source,
            _terminal(source_volume),
        ),
        {},
    )
    op = op.strip().lower()
    if op not in {"gt", "lt"}:
        raise ValueError("op must be 'gt' or 'lt'")
    return FunctionNode(
        _resolve_primitive(registry, op, seed="vwap_support"),
        (source, vwap),
    )


def build_wma_crossover_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_period: int = 9,
    slow_period: int = 21,
) -> Program:
    """WMA crossover: fast WMA above slow WMA."""
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("wma periods must be positive")
    _validate_indicator_periods(slow_period, fast_period)

    source_terminal = _terminal(source)
    wma_fast = ParameterizedNode(
        _resolve_primitive(registry, "ta_wma", seed="wma_crossover"),
        (source_terminal,),
        {"period": int(fast_period)},
    )
    wma_slow = ParameterizedNode(
        _resolve_primitive(registry, "ta_wma", seed="wma_crossover"),
        (source_terminal,),
        {"period": int(slow_period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="wma_crossover"),
        (wma_fast, wma_slow),
    )


SEED_TEMPLATES = [
    StrategySeedTemplate(
        name="ema_crossover",
        description="EMA crossover (fast EMA > slow EMA)",
        builder=build_ema_crossover_seed,
    ),
    StrategySeedTemplate(
        name="sma_crossover",
        description="SMA crossover (fast SMA > slow SMA)",
        builder=build_sma_crossover_seed,
    ),
    StrategySeedTemplate(
        name="macd_bullish_cross",
        description="MACD line crosses above signal line",
        builder=build_macd_bullish_seed,
    ),
    StrategySeedTemplate(
        name="bollinger_breakout",
        description="Close crosses above upper Bollinger band",
        builder=build_bollinger_breakout_seed,
    ),
    StrategySeedTemplate(
        name="macd_bearish_cross",
        description="MACD line crosses below signal line",
        builder=build_macd_bearish_seed,
    ),
    StrategySeedTemplate(
        name="ema_bearish_cross",
        description="EMA fast crosses below EMA slow",
        builder=build_ema_bearish_seed,
    ),
    StrategySeedTemplate(
        name="vwap_support",
        description="Close above VWAP",
        builder=build_vwap_support_seed,
    ),
    StrategySeedTemplate(
        name="wma_crossover",
        description="WMA crossover (fast WMA > slow WMA)",
        builder=build_wma_crossover_seed,
    ),
    StrategySeedTemplate(
        name="dema_crossover",
        description="DEMA crossover (fast DEMA > slow DEMA)",
        builder=build_dema_crossover_seed,
    ),
    StrategySeedTemplate(
        name="tema_bullish_cross",
        description="TEMA crossover (fast TEMA > slow TEMA)",
        builder=build_tema_bullish_seed,
    ),
    StrategySeedTemplate(
        name="kama_trend",
        description="KAMA fast above KAMA slow",
        builder=build_kama_trend_seed,
    ),
    StrategySeedTemplate(
        name="t3_bullish_cross",
        description="T3 crossover (fast T3 > slow T3)",
        builder=build_t3_bullish_seed,
    ),
    StrategySeedTemplate(
        name="ema_wilder_bullish_cross",
        description="Wilder EMA crossover (fast > slow)",
        builder=build_ema_wilder_bullish_seed,
    ),
    StrategySeedTemplate(
        name="donchian_breakout",
        description="Donchian breakout on close versus upper/lower channel",
        builder=build_donchian_breakout_seed,
    ),
]

SEED_NAMES = [template.name for template in SEED_TEMPLATES]
