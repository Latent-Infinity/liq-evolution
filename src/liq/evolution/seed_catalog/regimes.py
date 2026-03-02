"""Seed strategy implementations and registrations for this domain."""

from __future__ import annotations

from liq.evolution.program import ConstantNode, FunctionNode, ParameterizedNode
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog._core import (
    Program,
    StrategySeedTemplate,
    _resolve_primitive,
    _terminal,
    _validate_indicator_periods,
)


def build_adxr_trend_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    period: int = 14,
    threshold: float = 25.0,
) -> Program:
    """ADX-related trend strength: ADXR above threshold."""
    if period <= 0:
        raise ValueError("adxr period must be positive")
    if not 0 <= threshold <= 100:
        raise ValueError("adxr_threshold must be between 0 and 100")

    adxr = ParameterizedNode(
        _resolve_primitive(registry, "ta_adxr", seed="adxr_trend"),
        (
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "gt",
            seed="adxr_trend",
        ),
        (adxr, ConstantNode(float(threshold))),
    )


def build_aroon_momentum_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    period: int = 25,
    up_threshold: float = 70.0,
) -> Program:
    """Aroon momentum: Aroon-Up crosses above Aroon-Down while above threshold."""
    if period <= 0:
        raise ValueError("aroon period must be positive")
    if not 0 <= up_threshold <= 100:
        raise ValueError("up_threshold must be between 0 and 100")

    up = ParameterizedNode(
        _resolve_primitive(registry, "ta_aroon_aroon_up", seed="aroon_momentum"),
        (_terminal(source_high), _terminal(source_low)),
        {"period": int(period)},
    )
    down = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_aroon_aroon_down",
            seed="aroon_momentum",
        ),
        (_terminal(source_high), _terminal(source_low)),
        {"period": int(period)},
    )
    aroon_cross = FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="aroon_momentum",
        ),
        (up, down),
    )
    return FunctionNode(
        _resolve_primitive(registry, "and_op", seed="aroon_momentum"),
        (
            aroon_cross,
            FunctionNode(
                _resolve_primitive(registry, "gt", seed="aroon_momentum"),
                (up, ConstantNode(float(up_threshold))),
            ),
        ),
    )


def build_aroonosc_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    period: int = 25,
    threshold: float = 0.0,
) -> Program:
    """Aroon oscillator momentum: oscillator crosses above threshold."""
    if period <= 0:
        raise ValueError("aroonosc period must be positive")
    if not -100 <= threshold <= 100:
        raise ValueError("threshold must be between -100 and 100")

    aroonosc = ParameterizedNode(
        _resolve_primitive(registry, "ta_aroonosc", seed="aroonosc_bullish_cross"),
        (_terminal(source_high), _terminal(source_low)),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="aroonosc_bullish_cross",
        ),
        (aroonosc, ConstantNode(float(threshold))),
    )


def build_bop_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source_open: str = "open",
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
) -> Program:
    """Balance of Power confirmation: BOP above zero."""
    bop = ParameterizedNode(
        _resolve_primitive(registry, "ta_bop", seed="bop_bullish"),
        (
            _terminal(source_open),
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
        {},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="bop_bullish"),
        (bop, ConstantNode(0.0)),
    )


def build_mama_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
) -> Program:
    """MAMA crossover: MAMA line crosses above FAMA line."""
    if not 0 < fast_limit <= 1:
        raise ValueError("fast_limit must be between 0 and 1")
    if not 0 < slow_limit <= 1:
        raise ValueError("slow_limit must be between 0 and 1")

    source_terminal = _terminal(source)
    params = {
        "fast_limit": float(fast_limit),
        "slow_limit": float(slow_limit),
    }
    mama = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_mama_mama",
            seed="mama_bullish_cross",
        ),
        (source_terminal,),
        params,
    )
    fama = ParameterizedNode(
        _resolve_primitive(
            registry,
            "ta_mama_fama",
            seed="mama_bullish_cross",
        ),
        (source_terminal,),
        params,
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="mama_bullish_cross",
        ),
        (mama, fama),
    )


def build_regime_switching_momentum_volatility_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_std_period: int = 10,
    slow_std_period: int = 40,
    donchian_period: int = 20,
    roc_period: int = 12,
    roc_threshold: float = 0.0,
) -> Program:
    """Regime switch between momentum/trend and breakout based on volatility ratio."""
    if (
        fast_std_period <= 0
        or slow_std_period <= 0
        or donchian_period <= 0
        or roc_period <= 0
    ):
        raise ValueError("period must be positive")
    _validate_indicator_periods(slow_std_period, fast_std_period)

    source_terminal = _terminal(source)
    volatility_high = FunctionNode(
        _resolve_primitive(registry, "gt", seed="regime_switching_momentum_volatility"),
        (
            ParameterizedNode(
                _resolve_primitive(
                    registry,
                    "ta_rolling_stddev",
                    seed="regime_switching_momentum_volatility",
                ),
                (source_terminal,),
                {"period": int(fast_std_period)},
            ),
            ParameterizedNode(
                _resolve_primitive(
                    registry,
                    "ta_rolling_stddev",
                    seed="regime_switching_momentum_volatility",
                ),
                (source_terminal,),
                {"period": int(slow_std_period)},
            ),
        ),
    )
    breakout_mode = FunctionNode(
        _resolve_primitive(
            registry,
            "gt",
            seed="regime_switching_momentum_volatility",
        ),
        (
            source_terminal,
            ParameterizedNode(
                _resolve_primitive(
                    registry,
                    "ta_donchian_upper",
                    seed="regime_switching_momentum_volatility",
                ),
                (_terminal("high"), _terminal("low")),
                {"period": int(donchian_period)},
            ),
        ),
    )
    momentum_mode = FunctionNode(
        _resolve_primitive(
            registry,
            "gt",
            seed="regime_switching_momentum_volatility",
        ),
        (
            ParameterizedNode(
                _resolve_primitive(
                    registry,
                    "ta_roc",
                    seed="regime_switching_momentum_volatility",
                ),
                (source_terminal,),
                {"period": int(roc_period)},
            ),
            ConstantNode(float(roc_threshold)),
        ),
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "or_op",
            seed="regime_switching_momentum_volatility",
        ),
        (
            FunctionNode(
                _resolve_primitive(
                    registry,
                    "and_op",
                    seed="regime_switching_momentum_volatility",
                ),
                (volatility_high, breakout_mode),
            ),
            FunctionNode(
                _resolve_primitive(
                    registry,
                    "and_op",
                    seed="regime_switching_momentum_volatility",
                ),
                (
                    FunctionNode(
                        _resolve_primitive(
                            registry,
                            "not_op",
                            seed="regime_switching_momentum_volatility",
                        ),
                        (volatility_high,),
                    ),
                    momentum_mode,
                ),
            ),
        ),
    )


def build_regime_switching_trend_mean_reversion_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    fast_ma_period: int = 12,
    slow_ma_period: int = 26,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    range_threshold: float = 20.0,
) -> Program:
    """Regime switch: trend MA crossover when ADX strong, mean reversion otherwise."""
    if fast_ma_period <= 0 or slow_ma_period <= 0 or adx_period <= 0:
        raise ValueError("period must be positive")
    _validate_indicator_periods(slow_ma_period, fast_ma_period)
    if not 0 <= adx_threshold <= 100 or not 0 <= range_threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    if range_threshold > adx_threshold:
        raise ValueError("range_threshold must be <= adx_threshold")

    source_terminal = _terminal(source)
    ema_fast = ParameterizedNode(
        _resolve_primitive(
            registry, "ta_ema", seed="regime_switching_trend_mean_reversion"
        ),
        (source_terminal,),
        {"period": int(fast_ma_period)},
    )
    ema_slow = ParameterizedNode(
        _resolve_primitive(
            registry, "ta_ema", seed="regime_switching_trend_mean_reversion"
        ),
        (source_terminal,),
        {"period": int(slow_ma_period)},
    )
    trend_regime = FunctionNode(
        _resolve_primitive(
            registry, "gt", seed="regime_switching_trend_mean_reversion"
        ),
        (
            ParameterizedNode(
                _resolve_primitive(
                    registry, "ta_adx", seed="regime_switching_trend_mean_reversion"
                ),
                (
                    _terminal("high"),
                    _terminal("low"),
                    _terminal("close"),
                ),
                {"period": int(adx_period)},
            ),
            ConstantNode(float(adx_threshold)),
        ),
    )
    trend_entry = FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="regime_switching_trend_mean_reversion",
        ),
        (ema_fast, ema_slow),
    )
    range_regime = FunctionNode(
        _resolve_primitive(
            registry,
            "lt",
            seed="regime_switching_trend_mean_reversion",
        ),
        (
            ParameterizedNode(
                _resolve_primitive(
                    registry,
                    "ta_adx",
                    seed="regime_switching_trend_mean_reversion",
                ),
                (
                    _terminal("high"),
                    _terminal("low"),
                    _terminal("close"),
                ),
                {"period": int(adx_period)},
            ),
            ConstantNode(float(range_threshold)),
        ),
    )
    mean_reversion_entry = FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="regime_switching_trend_mean_reversion",
        ),
        (
            source_terminal,
            ParameterizedNode(
                _resolve_primitive(
                    registry,
                    "ta_bollinger_lower",
                    seed="regime_switching_trend_mean_reversion",
                ),
                (source_terminal,),
                {"period": 20, "std_dev": 2.0},
            ),
        ),
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "or_op",
            seed="regime_switching_trend_mean_reversion",
        ),
        (
            FunctionNode(
                _resolve_primitive(
                    registry,
                    "and_op",
                    seed="regime_switching_trend_mean_reversion",
                ),
                (trend_regime, trend_entry),
            ),
            FunctionNode(
                _resolve_primitive(
                    registry,
                    "and_op",
                    seed="regime_switching_trend_mean_reversion",
                ),
                (range_regime, mean_reversion_entry),
            ),
        ),
    )


def build_williams_r_overbought_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    period: int = 14,
    threshold: float = -20.0,
) -> Program:
    """Williams %R overbought: `%R > threshold`."""
    if period <= 0:
        raise ValueError("williams_r period must be positive")
    if not -100 <= threshold <= 0:
        raise ValueError("threshold must be between -100 and 0")

    williams_r = ParameterizedNode(
        _resolve_primitive(registry, "ta_williams_r", seed="williams_r_overbought"),
        (
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="williams_r_overbought"),
        (williams_r, ConstantNode(float(threshold))),
    )


def build_williams_r_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    period: int = 14,
    threshold: float = -80.0,
) -> Program:
    """Williams %R oversold: `%R < threshold`."""
    if period <= 0:
        raise ValueError("williams_r period must be positive")
    if not -100 <= threshold <= 0:
        raise ValueError("threshold must be between -100 and 0")

    williams_r = ParameterizedNode(
        _resolve_primitive(registry, "ta_williams_r", seed="williams_r_oversold"),
        (
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
        ),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="williams_r_oversold"),
        (williams_r, ConstantNode(float(threshold))),
    )


SEED_TEMPLATES = [
    StrategySeedTemplate(
        name="aroon_momentum",
        description="Aroon-Up crosses above Aroon-Down and above threshold",
        builder=build_aroon_momentum_seed,
    ),
    StrategySeedTemplate(
        name="mama_bullish_cross",
        description="MAMA crosses above FAMA",
        builder=build_mama_bullish_seed,
    ),
    StrategySeedTemplate(
        name="aroonosc_bullish_cross",
        description="Aroon oscillator crosses above threshold",
        builder=build_aroonosc_bullish_seed,
    ),
    StrategySeedTemplate(
        name="williams_r_oversold",
        description="Williams %R oversold condition",
        builder=build_williams_r_oversold_seed,
    ),
    StrategySeedTemplate(
        name="williams_r_overbought",
        description="Williams %R overbought condition",
        builder=build_williams_r_overbought_seed,
    ),
    StrategySeedTemplate(
        name="adxr_trend",
        description="ADXR above threshold",
        builder=build_adxr_trend_seed,
    ),
    StrategySeedTemplate(
        name="bop_bullish",
        description="Balance of Power above zero",
        builder=build_bop_bullish_seed,
    ),
    StrategySeedTemplate(
        name="regime_switching_trend_mean_reversion",
        description="Trend vs mean-reversion strategy based on ADX regime",
        builder=build_regime_switching_trend_mean_reversion_seed,
    ),
    StrategySeedTemplate(
        name="regime_switching_momentum_volatility",
        description="Momentum vs breakout strategy based on volatility regime",
        builder=build_regime_switching_momentum_volatility_seed,
    ),
]

SEED_NAMES = [template.name for template in SEED_TEMPLATES]
