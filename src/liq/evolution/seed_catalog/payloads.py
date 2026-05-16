"""Serialized built-in seed payload catalogs for experiment warm starts."""

from __future__ import annotations

from typing import Any


def _terminal(name: str) -> dict[str, Any]:
    return {
        "type": "terminal",
        "name": name,
        "output_type": "Series",
    }


def _constant(value: float) -> dict[str, Any]:
    return {
        "type": "constant",
        "value": float(value),
        "output_type": "Series",
    }


def _function(name: str, *children: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "primitive": name,
        "children": list(children),
    }


def _parameterized(
    name: str,
    *children: dict[str, Any],
    **params: int | float,
) -> dict[str, Any]:
    return {
        "type": "parameterized",
        "primitive": name,
        "children": list(children),
        "params": dict(params),
    }


def _wrapped(program: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "program": program,
    }


def _regime_v1_seed_payloads() -> list[dict[str, Any]]:
    high = _terminal("high")
    low = _terminal("low")
    close = _terminal("close")

    adx14 = _parameterized("ta_adx_adx", high, low, close, period=14)
    strong_trend = _function("gt", adx14, _constant(25.0))
    ema12 = _parameterized("ta_ema", close, period=12)
    ema48 = _parameterized("ta_ema", close, period=48)
    trend_score = _function("div", _function("sub", ema12, ema48), close)
    rsi7 = _parameterized("ta_rsi", close, period=7)
    range_score = _function(
        "div",
        _function("sub", _constant(50.0), rsi7),
        _constant(50.0),
    )
    seed1 = _wrapped(_function("if_then_else", strong_trend, trend_score, range_score))

    atr14 = _parameterized("ta_atr", high, low, close, period=14)
    high_vol_regime = _function(
        "gt",
        _function("div", atr14, close),
        _constant(0.015),
    )
    rocr12 = _parameterized("ta_rocr", close, period=12)
    vol_trend_score = _function(
        "div",
        _function("sub", rocr12, _constant(1.0)),
        _constant(0.04),
    )
    rsi14 = _parameterized("ta_rsi", close, period=14)
    vol_range_score = _function(
        "div",
        _function("sub", _constant(50.0), rsi14),
        _constant(50.0),
    )
    seed2 = _wrapped(
        _function("if_then_else", high_vol_regime, vol_trend_score, vol_range_score)
    )

    ema20 = _parameterized("ta_ema", close, period=20)
    ema60 = _parameterized("ta_ema", close, period=60)
    spread = _function("div", _function("sub", ema20, ema60), close)
    trend_strength = _function("div", adx14, _constant(50.0))
    seed3 = _wrapped(_function("mul", spread, trend_strength))

    return [seed1, seed2, seed3]


def _regime_v2_seed_payloads() -> list[dict[str, Any]]:
    high = _terminal("high")
    low = _terminal("low")
    close = _terminal("close")

    adx14 = _parameterized("ta_adx_adx", high, low, close, period=14)
    atr14 = _parameterized("ta_atr", high, low, close, period=14)
    atr_pct = _function("div", atr14, close)
    rsi7 = _parameterized("ta_rsi", close, period=7)
    rsi14 = _parameterized("ta_rsi", close, period=14)

    strong_trend = _function("gt", adx14, _constant(22.0))
    ema20 = _parameterized("ta_ema", close, period=20)
    ema80 = _parameterized("ta_ema", close, period=80)
    trend_spread = _function("div", _function("sub", ema20, ema80), close)
    trend_bounded = _function("clip", trend_spread, _constant(-0.20), _constant(0.20))
    range_raw = _function(
        "div",
        _function("sub", _constant(50.0), rsi7),
        _constant(50.0),
    )
    range_bounded = _function("clip", range_raw, _constant(-0.35), _constant(0.35))
    seed1 = _wrapped(_function("if_then_else", strong_trend, trend_bounded, range_bounded))

    rocr8 = _parameterized("ta_rocr", close, period=8)
    momentum = _function("sub", rocr8, _constant(1.0))
    vol_norm = _function("div", momentum, _function("add", atr_pct, _constant(0.005)))
    vol_bounded = _function("clip", vol_norm, _constant(-0.40), _constant(0.40))
    low_vol = _function("lt", atr_pct, _constant(0.012))
    calm_raw = _function(
        "div",
        _function("sub", _constant(50.0), rsi14),
        _constant(70.0),
    )
    calm_bounded = _function("clip", calm_raw, _constant(-0.30), _constant(0.30))
    seed2 = _wrapped(_function("if_then_else", low_vol, calm_bounded, vol_bounded))

    ema12 = _parameterized("ta_ema", close, period=12)
    ema48 = _parameterized("ta_ema", close, period=48)
    persistent = _parameterized("is_rising", ema12, period=4)
    fast_spread = _function("div", _function("sub", ema12, ema48), close)
    fast_bounded = _function("clip", fast_spread, _constant(-0.25), _constant(0.25))
    pullback = _function("div", _function("sub", ema48, close), close)
    pullback_bounded = _function("clip", pullback, _constant(-0.20), _constant(0.20))
    seed3 = _wrapped(_function("if_then_else", persistent, fast_bounded, pullback_bounded))

    atr_rank = _parameterized("percentile_rank", atr_pct, period=40)
    high_vol = _function("gt", atr_rank, _constant(70.0))
    shock_raw = _function("div", _function("sub", rocr8, _constant(1.0)), _constant(0.03))
    shock_bounded = _function("clip", shock_raw, _constant(-0.50), _constant(0.50))
    quiet_raw = _function(
        "div",
        _function("sub", _constant(50.0), rsi14),
        _constant(80.0),
    )
    quiet_bounded = _function("clip", quiet_raw, _constant(-0.30), _constant(0.30))
    seed4 = _wrapped(_function("if_then_else", high_vol, shock_bounded, quiet_bounded))

    ema200 = _parameterized("ta_ema", close, period=200)
    above_long_term = _function("gt", close, ema200)
    bull_raw = _function("div", _function("sub", close, ema200), ema200)
    bull_bounded = _function("clip", bull_raw, _constant(0.0), _constant(0.30))
    bear_raw = _function("div", _function("sub", ema48, close), close)
    bear_bounded = _function("clip", bear_raw, _constant(-0.25), _constant(0.25))
    seed5 = _wrapped(_function("if_then_else", above_long_term, bull_bounded, bear_bounded))

    return [seed1, seed2, seed3, seed4, seed5]


def _regime_v3_seed_payloads() -> list[dict[str, Any]]:
    high = _terminal("high")
    low = _terminal("low")
    close = _terminal("close")

    adx14 = _parameterized("ta_adx_adx", high, low, close, period=14)
    atr14 = _parameterized("ta_atr", high, low, close, period=14)
    atr_pct = _function("div", atr14, close)
    atr_rank = _parameterized("percentile_rank", atr_pct, period=48)
    rsi7 = _parameterized("ta_rsi", close, period=7)
    rsi14 = _parameterized("ta_rsi", close, period=14)
    rocr8 = _parameterized("ta_rocr", close, period=8)

    ema12 = _parameterized("ta_ema", close, period=12)
    ema20 = _parameterized("ta_ema", close, period=20)
    ema48 = _parameterized("ta_ema", close, period=48)
    ema80 = _parameterized("ta_ema", close, period=80)
    ema180 = _parameterized("ta_ema", close, period=180)

    strong_trend = _function("gt", adx14, _constant(24.0))
    above_long_term = _function("gt", close, ema180)
    bull_trend = _function("and_op", strong_trend, above_long_term)
    bull_spread = _function("div", _function("sub", ema12, ema48), close)
    bull_bounded = _function("clip", bull_spread, _constant(-0.30), _constant(0.30))
    bear_spread = _function("div", _function("sub", ema48, close), close)
    bear_bounded = _function("clip", bear_spread, _constant(-0.28), _constant(0.28))
    seed1 = _wrapped(_function("if_then_else", bull_trend, bull_bounded, bear_bounded))

    high_vol = _function("gt", atr_rank, _constant(65.0))
    vol_mom = _function(
        "div",
        _function("sub", rocr8, _constant(1.0)),
        _function("add", atr_pct, _constant(0.004)),
    )
    vol_bounded = _function("clip", vol_mom, _constant(-0.45), _constant(0.45))
    calm_revert = _function(
        "div",
        _function("sub", _constant(50.0), rsi7),
        _constant(55.0),
    )
    calm_bounded = _function("clip", calm_revert, _constant(-0.30), _constant(0.30))
    seed2 = _wrapped(_function("if_then_else", high_vol, vol_bounded, calm_bounded))

    falling_fast = _parameterized("is_falling", ema20, period=4)
    bear_accel = _function("div", _function("sub", ema48, ema12), close)
    bear_accel_bounded = _function("clip", bear_accel, _constant(-0.30), _constant(0.30))
    trend_resume = _function("div", _function("sub", ema12, ema80), close)
    trend_resume_bounded = _function(
        "clip", trend_resume, _constant(-0.25), _constant(0.25)
    )
    seed3 = _wrapped(
        _function("if_then_else", falling_fast, bear_accel_bounded, trend_resume_bounded)
    )

    low_vol = _function("lt", atr_pct, _constant(0.010))
    low_vol_pullback = _function(
        "div",
        _function("sub", _constant(50.0), rsi14),
        _constant(60.0),
    )
    low_vol_bounded = _function(
        "clip", low_vol_pullback, _constant(-0.25), _constant(0.25)
    )
    high_vol_breakout = _function(
        "div",
        _function("sub", rocr8, _constant(1.0)),
        _constant(0.03),
    )
    high_vol_bounded = _function(
        "clip", high_vol_breakout, _constant(-0.40), _constant(0.40)
    )
    seed4 = _wrapped(_function("if_then_else", low_vol, low_vol_bounded, high_vol_bounded))

    drift = _function("div", _function("sub", close, ema180), ema180)
    trend_scale = _function("div", adx14, _constant(40.0))
    scaled_drift = _function("mul", drift, trend_scale)
    seed5 = _wrapped(_function("clip", scaled_drift, _constant(-0.30), _constant(0.30)))

    lag1 = _parameterized("n_bars_ago", close, shift=1)
    up_count = _parameterized("greater_count", close, lag1, period=6)
    down_count = _parameterized("lower_count", close, lag1, period=6)
    count_balance = _function("div", _function("sub", up_count, down_count), _constant(6.0))
    seed6 = _wrapped(_function("clip", count_balance, _constant(-0.35), _constant(0.35)))

    return [seed1, seed2, seed3, seed4, seed5, seed6]


def _regime_v4_seed_payloads() -> list[dict[str, Any]]:
    high = _terminal("high")
    low = _terminal("low")
    close = _terminal("close")
    volume = _terminal("volume")

    atr14 = _parameterized("ta_atr", high, low, close, period=14)
    atr_pct = _function("div", atr14, close)

    ema55 = _parameterized("ta_ema", close, period=55)
    trend_gate = _function("gt", close, ema55)
    donch_u21 = _parameterized("ta_donchian_upper", high, low, period=21)
    donch_m21 = _parameterized("ta_donchian_middle", high, low, period=21)
    breakout = _function("div", _function("sub", close, donch_u21), close)
    revert = _function("div", _function("sub", donch_m21, close), close)
    seed1 = _wrapped(
        _function(
            "if_then_else",
            trend_gate,
            _function("clip", breakout, _constant(-0.30), _constant(0.30)),
            _function("clip", revert, _constant(-0.25), _constant(0.25)),
        )
    )

    kama21 = _parameterized(
        "ta_kama",
        close,
        period=21,
        fast_period=5,
        slow_period=34,
    )
    kama55 = _parameterized(
        "ta_kama",
        close,
        period=55,
        fast_period=5,
        slow_period=55,
    )
    kama_spread = _function("div", _function("sub", kama21, kama55), close)
    atr_norm = _function("div", kama_spread, _function("add", atr_pct, _constant(0.004)))
    seed2 = _wrapped(_function("clip", atr_norm, _constant(-0.35), _constant(0.35)))

    vwap = _parameterized("ta_vwap", high, low, close, volume)
    vwap_revert = _function("div", _function("sub", vwap, close), close)
    ema21 = _parameterized("ta_ema", close, period=21)
    vol_trend = _function("div", _function("sub", close, ema21), close)
    high_vol = _function("gt", atr_pct, _constant(0.015))
    seed3 = _wrapped(
        _function(
            "if_then_else",
            high_vol,
            _function("clip", vol_trend, _constant(-0.30), _constant(0.30)),
            _function("clip", vwap_revert, _constant(-0.30), _constant(0.30)),
        )
    )

    lin5 = _parameterized("ta_linearreg", close, period=5)
    lin21 = _parameterized("ta_linearreg", close, period=21)
    slope = _function("div", _function("sub", lin5, lin21), close)
    donch_u55 = _parameterized("ta_donchian_upper", high, low, period=55)
    donch_l55 = _parameterized("ta_donchian_lower", high, low, period=55)
    width = _function("div", _function("sub", donch_u55, donch_l55), close)
    width_scaled = _function("div", slope, _function("add", width, _constant(0.010)))
    seed4 = _wrapped(_function("clip", width_scaled, _constant(-0.35), _constant(0.35)))

    bb_u = _parameterized("ta_bollinger_upper", close, period=21, std_dev=2.0)
    bb_m = _parameterized("ta_bollinger_middle", close, period=21, std_dev=2.0)
    bb_l = _parameterized("ta_bollinger_lower", close, period=21, std_dev=2.0)
    band_width = _function("div", _function("sub", bb_u, bb_l), close)
    squeeze = _function("lt", band_width, _constant(0.030))
    bb_revert = _function("div", _function("sub", bb_m, close), close)
    ema5 = _parameterized("ta_ema", close, period=5)
    ema_trend = _function("div", _function("sub", ema5, ema21), close)
    seed5 = _wrapped(
        _function(
            "if_then_else",
            squeeze,
            _function("clip", bb_revert, _constant(-0.30), _constant(0.30)),
            _function("clip", ema_trend, _constant(-0.30), _constant(0.30)),
        )
    )

    return [seed1, seed2, seed3, seed4, seed5]


def built_in_seed_payloads(seed_program_set: str) -> list[dict[str, Any]]:
    """Return serialized built-in seed payloads for the requested seed set."""

    if seed_program_set == "regime_v1":
        return _regime_v1_seed_payloads()
    if seed_program_set == "regime_v2":
        return _regime_v2_seed_payloads()
    if seed_program_set == "regime_v3":
        return _regime_v3_seed_payloads()
    if seed_program_set == "regime_v4":
        return _regime_v4_seed_payloads()
    raise ValueError(f"Unknown seed_program_set: {seed_program_set}")


__all__ = ["built_in_seed_payloads"]
