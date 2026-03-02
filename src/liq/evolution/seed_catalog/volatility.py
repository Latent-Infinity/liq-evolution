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


def build_rolling_stddev_spike_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 20,
    lookback: int = 20,
) -> Program:
    """Volatility spike: rolling std-dev above prior value."""
    if period <= 0:
        raise ValueError("rolling_stddev period must be positive")
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    stddev = ParameterizedNode(
        _resolve_primitive(registry, "ta_rolling_stddev", seed="rolling_stddev_spike"),
        (_terminal(source),),
        {"period": int(period)},
    )
    prev_stddev = ParameterizedNode(
        _resolve_primitive(
            registry,
            "n_bars_ago",
            seed="rolling_stddev_spike",
        ),
        (stddev,),
        {"shift": int(lookback)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="rolling_stddev_spike"),
        (stddev, prev_stddev),
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


def build_var_spike_seed(
    registry: PrimitiveRegistry,
    *,
    source: str = "close",
    period: int = 20,
    lookback: int = 20,
) -> Program:
    """Volatility regime spike: variance above prior value."""
    if period <= 0:
        raise ValueError("var period must be positive")
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    variance = ParameterizedNode(
        _resolve_primitive(registry, "ta_var", seed="var_spike"),
        (_terminal(source),),
        {"period": int(period)},
    )
    prev_variance = ParameterizedNode(
        _resolve_primitive(registry, "n_bars_ago", seed="var_spike"),
        (variance,),
        {"shift": int(lookback)},
    )
    return FunctionNode(
        _resolve_primitive(registry, "gt", seed="var_spike"),
        (variance, prev_variance),
    )


SEED_TEMPLATES = [
    StrategySeedTemplate(
        name="adx_trend",
        description="+DI crosses above -DI while ADX above threshold",
        builder=build_adx_trend_seed,
    ),
    StrategySeedTemplate(
        name="sar_bullish",
        description="Close above Parabolic SAR",
        builder=build_sar_bullish_seed,
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
        name="rolling_stddev_spike",
        description="Rolling std-dev above prior value",
        builder=build_rolling_stddev_spike_seed,
    ),
    StrategySeedTemplate(
        name="var_spike",
        description="Variance above prior value",
        builder=build_var_spike_seed,
    ),
]

SEED_NAMES = [template.name for template in SEED_TEMPLATES]
