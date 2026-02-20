"""Hand-crafted strategy seed programs for warm-start and periodic injection.

The module exposes a small registry of known trading patterns (EMA crossover,
RSI extremes, etc.) as reusable Program constructors. These are intended for
manual seed injection via ``seed_programs`` and for periodic seed injection via
the liq-gp ``seed_injection`` feature.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from liq.evolution.errors import ConfigurationError
from liq.evolution.program import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.evolution.protocols import PrimitiveRegistry
from liq.gp.types import Series


@dataclass(frozen=True)
class StrategySeedTemplate:
    """Metadata for a known seed strategy."""

    name: str
    description: str
    builder: Callable[[PrimitiveRegistry], Program]


def _terminal(name: str) -> TerminalNode:
    """Build a typed price-feature terminal."""
    return TerminalNode(name=name, output_type=Series)


def _resolve_primitive(
    registry: PrimitiveRegistry,
    name: str,
    *,
    seed: str,
):
    """Resolve a primitive from the registry with clear context on failure."""
    try:
        return registry.get(name)
    except Exception as exc:
        msg = (
            f"{seed!r} seed requires primitive {name!r} to be registered "
            "in the strategy registry"
        )
        raise ConfigurationError(msg) from exc


def _validate_indicator_periods(slow_period: int, fast_period: int) -> None:
    """Validate moving-average crossover period ordering."""
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")


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
        _resolve_primitive(registry, "ta_stochastic_k", seed="stochastic_bullish_cross"),
        (high, low, close),
        params,
    )
    stoch_d = ParameterizedNode(
        _resolve_primitive(registry, "ta_stochastic_d", seed="stochastic_bullish_cross"),
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
        (aroon_cross, FunctionNode(
            _resolve_primitive(registry, "gt", seed="aroon_momentum"),
            (up, ConstantNode(float(up_threshold))),
        )),
    )


def build_adx_trend_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    period: int = 14,
    adx_threshold: float = 25.0,
) -> Program:
    """ADX trend-strength confirmation: +DI crosses above -DI above threshold."""
    if period <= 0:
        raise ValueError("adx period must be positive")
    if adx_threshold < 0 or adx_threshold > 100:
        raise ValueError("adx_threshold must be between 0 and 100")

    high = _terminal(source_high)
    low = _terminal(source_low)
    close = _terminal(source_close)
    params = {"period": int(period)}

    adx = ParameterizedNode(
        _resolve_primitive(registry, "ta_adx", seed="adx_trend"),
        (high, low, close),
        params,
    )
    plus_di = ParameterizedNode(
        _resolve_primitive(registry, "ta_adx_plus_di", seed="adx_trend"),
        (high, low, close),
        params,
    )
    minus_di = ParameterizedNode(
        _resolve_primitive(registry, "ta_adx_minus_di", seed="adx_trend"),
        (high, low, close),
        params,
    )
    di_cross = FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="adx_trend"),
        (plus_di, minus_di),
    )
    return FunctionNode(
        _resolve_primitive(registry, "and_op", seed="adx_trend"),
        (
            di_cross,
            FunctionNode(
                _resolve_primitive(registry, "gt", seed="adx_trend"),
                (adx, ConstantNode(float(adx_threshold))),
            ),
        ),
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


def build_sar_bullish_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> Program:
    """Parabolic SAR trend filter: close above SAR."""
    if not 0 <= af_start <= 1:
        raise ValueError("af_start must be between 0 and 1")
    if not 0 <= af_step <= 1:
        raise ValueError("af_step must be between 0 and 1")
    if not 0 <= af_max <= 1:
        raise ValueError("af_max must be between 0 and 1")
    if af_step > af_max:
        raise ValueError("af_step must be smaller than or equal to af_max")

    sar = ParameterizedNode(
        _resolve_primitive(registry, "ta_sar", seed="sar_bullish"),
        (_terminal(source_high), _terminal(source_low)),
        {
            "af_start": float(af_start),
            "af_step": float(af_step),
            "af_max": float(af_max),
        },
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="sar_bullish"),
        (_terminal(source_close), sar),
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


def build_cci_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    period: int = 20,
    threshold: float = -100.0,
) -> Program:
    """CCI oversold signal: CCI < threshold."""
    if period <= 0:
        raise ValueError("cci period must be positive")
    if not -1000 <= threshold <= 1000:
        raise ValueError("cci threshold must be in a realistic range")

    cci = ParameterizedNode(
        _resolve_primitive(registry, "ta_cci", seed="cci_oversold"),
        (_terminal(source_high), _terminal(source_low), _terminal(source_close)),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="cci_oversold"),
        (cci, ConstantNode(float(threshold))),
    )


def build_cci_overbought_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    period: int = 20,
    threshold: float = 100.0,
) -> Program:
    """CCI overbought signal: CCI > threshold."""
    if period <= 0:
        raise ValueError("cci period must be positive")
    if not -1000 <= threshold <= 1000:
        raise ValueError("cci threshold must be in a realistic range")

    cci = ParameterizedNode(
        _resolve_primitive(registry, "ta_cci", seed="cci_overbought"),
        (_terminal(source_high), _terminal(source_low), _terminal(source_close)),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="cci_overbought"),
        (cci, ConstantNode(float(threshold))),
    )


def build_bollinger_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 20,
    std_dev: float = 2.0,
) -> Program:
    """Bollinger reversion entry: close crosses above lower band."""
    if period <= 0:
        raise ValueError("bollinger period must be positive")
    if std_dev <= 0:
        raise ValueError("std_dev must be greater than 0")

    source_terminal = _terminal(source)
    lower = ParameterizedNode(
        _resolve_primitive(registry, "ta_bollinger_lower", seed="bollinger_oversold"),
        (source_terminal,),
        {"period": int(period), "std_dev": float(std_dev)},
    )
    return FunctionNode(
        _resolve_primitive(
            registry,
            "crosses_above",
            seed="bollinger_oversold",
        ),
        (source_terminal, lower),
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


def build_atr_volatility_spike_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    period: int = 14,
    lookback: int = 20,
) -> Program:
    """ATR volatility spike: ATR currently higher than prior bar."""
    if period <= 0:
        raise ValueError("atr period must be positive")
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    atr = ParameterizedNode(
        _resolve_primitive(registry, "ta_atr", seed="atr_volatility_spike"),
        (_terminal(source_high), _terminal(source_low), _terminal(source_close)),
        {"period": int(period)},
    )
    prev_atr = ParameterizedNode(
        _resolve_primitive(
            registry,
            "n_bars_ago",
            seed="atr_volatility_spike",
        ),
        (atr,),
        {"shift": int(lookback)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="atr_volatility_spike"),
        (atr, prev_atr),
    )


def build_mfi_oversold_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    source_volume: str = "volume",
    period: int = 14,
    threshold: float = 20.0,
) -> Program:
    """MFI oversold signal: MFI < threshold."""
    if period <= 0:
        raise ValueError("mfi period must be positive")
    if not 0 <= threshold <= 100:
        raise ValueError("mfi threshold must be between 0 and 100")

    mfi = ParameterizedNode(
        _resolve_primitive(registry, "ta_mfi", seed="mfi_oversold"),
        (
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
            _terminal(source_volume),
        ),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "lt", seed="mfi_oversold"),
        (mfi, ConstantNode(float(threshold))),
    )


def build_mfi_overbought_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    source_volume: str = "volume",
    period: int = 14,
    threshold: float = 80.0,
) -> Program:
    """MFI overbought signal: MFI > threshold."""
    if period <= 0:
        raise ValueError("mfi period must be positive")
    if not 0 <= threshold <= 100:
        raise ValueError("mfi threshold must be between 0 and 100")

    mfi = ParameterizedNode(
        _resolve_primitive(registry, "ta_mfi", seed="mfi_overbought"),
        (
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
            _terminal(source_volume),
        ),
        {"period": int(period)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="mfi_overbought"),
        (mfi, ConstantNode(float(threshold))),
    )


def build_obv_uptrend_seed(
    registry: PrimitiveRegistry,
    *,
    source_close: str = "close",
    source_volume: str = "volume",
    lookback: int = 1,
) -> Program:
    """OBV momentum: OBV crosses above delayed OBV."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    obv = ParameterizedNode(
        _resolve_primitive(registry, "ta_obv", seed="obv_uptrend"),
        (_terminal(source_close), _terminal(source_volume)),
        {},
    )
    prev_obv = ParameterizedNode(
        _resolve_primitive(registry, "n_bars_ago", seed="obv_uptrend"),
        (obv,),
        {"shift": int(lookback)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="obv_uptrend"),
        (obv, prev_obv),
    )


def build_ad_ascending_seed(
    registry: PrimitiveRegistry,
    *,
    source_high: str = "high",
    source_low: str = "low",
    source_close: str = "close",
    source_volume: str = "volume",
    lookback: int = 1,
) -> Program:
    """A/D line increasing: AD crosses above delayed AD."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    ad = ParameterizedNode(
        _resolve_primitive(registry, "ta_ad", seed="ad_ascending"),
        (
            _terminal(source_high),
            _terminal(source_low),
            _terminal(source_close),
            _terminal(source_volume),
        ),
        {},
    )
    prev_ad = ParameterizedNode(
        _resolve_primitive(registry, "n_bars_ago", seed="ad_ascending"),
        (ad,),
        {"shift": int(lookback)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "crosses_above", seed="ad_ascending"),
        (ad, prev_ad),
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


def _normalize_seed_name(name: str) -> str:
    return name.strip().replace("-", "_").replace(" ", "_").lower()


_SEEDS: dict[str, StrategySeedTemplate] = {
    template.name: template
    for template in [
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
            name="sar_bullish",
            description="Close above Parabolic SAR",
            builder=build_sar_bullish_seed,
        ),
        StrategySeedTemplate(
            name="vwap_support",
            description="Close above VWAP",
            builder=build_vwap_support_seed,
        ),
        StrategySeedTemplate(
            name="adx_trend",
            description="+DI crosses above -DI while ADX above threshold",
            builder=build_adx_trend_seed,
        ),
        StrategySeedTemplate(
            name="cci_oversold",
            description="CCI below oversold threshold",
            builder=build_cci_oversold_seed,
        ),
        StrategySeedTemplate(
            name="cci_overbought",
            description="CCI above overbought threshold",
            builder=build_cci_overbought_seed,
        ),
        StrategySeedTemplate(
            name="bollinger_oversold",
            description="Close crosses above lower Bollinger band",
            builder=build_bollinger_oversold_seed,
        ),
        StrategySeedTemplate(
            name="donchian_breakout",
            description="Donchian breakout on close versus upper/lower channel",
            builder=build_donchian_breakout_seed,
        ),
        StrategySeedTemplate(
            name="atr_volatility_spike",
            description="ATR above value from lookback period ago",
            builder=build_atr_volatility_spike_seed,
        ),
        StrategySeedTemplate(
            name="mfi_oversold",
            description="MFI below oversold threshold",
            builder=build_mfi_oversold_seed,
        ),
        StrategySeedTemplate(
            name="mfi_overbought",
            description="MFI above overbought threshold",
            builder=build_mfi_overbought_seed,
        ),
        StrategySeedTemplate(
            name="obv_uptrend",
            description="OBV crosses above delayed OBV",
            builder=build_obv_uptrend_seed,
        ),
        StrategySeedTemplate(
            name="ad_ascending",
            description="A/D line crosses above delayed A/D",
            builder=build_ad_ascending_seed,
        ),
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
    ]
}


def list_known_strategy_seeds() -> list[str]:
    """Return all known seed names in registry order."""
    return list(_SEEDS)


def get_seed_template(name: str) -> StrategySeedTemplate:
    """Return the seed template metadata for a known seed name."""
    normalized = _normalize_seed_name(name)
    try:
        return _SEEDS[normalized]
    except KeyError as exc:
        available = ", ".join(list_known_strategy_seeds())
        msg = f"Unknown seed {name!r}. Available seeds: {available}"
        raise KeyError(msg) from exc


def build_strategy_seed(name: str, registry: PrimitiveRegistry) -> Program:
    """Build a single known strategy seed."""
    template = get_seed_template(name)
    return template.builder(registry)


def build_strategy_seeds(
    names: Iterable[str],
    registry: PrimitiveRegistry,
) -> list[Program]:
    """Build multiple known strategy seeds."""
    return [build_strategy_seed(name, registry) for name in names]


__all__ = [
    "StrategySeedTemplate",
    "build_bollinger_breakout_seed",
    "build_macd_bearish_seed",
    "build_ema_bearish_seed",
    "build_stochastic_bullish_seed",
    "build_stochrsi_bullish_seed",
    "build_stochastic_oversold_seed",
    "build_stochrsi_oversold_seed",
    "build_aroon_momentum_seed",
    "build_mama_bullish_seed",
    "build_aroonosc_bullish_seed",
    "build_williams_r_oversold_seed",
    "build_williams_r_overbought_seed",
    "build_mom_bullish_seed",
    "build_apo_bullish_seed",
    "build_cmo_oversold_seed",
    "build_cmo_overbought_seed",
    "build_sar_bullish_seed",
    "build_roc_bullish_seed",
    "build_trix_bullish_seed",
    "build_vwap_support_seed",
    "build_adx_trend_seed",
    "build_cci_oversold_seed",
    "build_cci_overbought_seed",
    "build_bollinger_oversold_seed",
    "build_donchian_breakout_seed",
    "build_atr_volatility_spike_seed",
    "build_mfi_oversold_seed",
    "build_mfi_overbought_seed",
    "build_obv_uptrend_seed",
    "build_ad_ascending_seed",
    "build_cdl_engulfing_bullish_seed",
    "build_cdl_hammer_seed",
    "build_ema_crossover_seed",
    "build_macd_bullish_seed",
    "build_rsi_oversold_seed",
    "build_rsi_overbought_seed",
    "build_sma_crossover_seed",
    "build_strategy_seed",
    "build_strategy_seeds",
    "get_seed_template",
    "list_known_strategy_seeds",
]
