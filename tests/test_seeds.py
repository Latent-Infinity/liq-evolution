"""Tests for known trading strategy seed templates."""

from __future__ import annotations

import pytest

from liq.evolution.errors import ConfigurationError
from liq.evolution.program import FunctionNode
from liq.evolution.seeds import (
    StrategySeedTemplate,
    build_strategy_seed,
    build_strategy_seeds,
    get_seed_template,
    list_known_strategy_seeds,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, Series


def _seed_registry() -> PrimitiveRegistry:
    """Build a minimal registry with primitives required by trading seeds."""

    def _passthrough(input_value, /, **_params):
        return input_value

    registry = PrimitiveRegistry()
    registry.register(
        "ta_ema",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_sma",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_rsi",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_macd_macd_line",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_macd_signal_line",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_bollinger_upper",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_bollinger_lower",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_stochastic_k",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_stochastic_d",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_stochrsi_fastk",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_stochrsi_fastd",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_aroon_aroon_up",
        _passthrough,
        category="ta",
        input_types=(Series, Series),
        output_type=Series,
        arity=2,
    )
    registry.register(
        "ta_aroon_aroon_down",
        _passthrough,
        category="ta",
        input_types=(Series, Series),
        output_type=Series,
        arity=2,
    )
    registry.register(
        "ta_adx",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_adx_plus_di",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_adx_minus_di",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_aroonosc",
        _passthrough,
        category="ta",
        input_types=(Series, Series),
        output_type=Series,
        arity=2,
    )
    registry.register(
        "ta_mama_mama",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_mama_fama",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_williams_r",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_mom",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_apo",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_cmo",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_sar",
        _passthrough,
        category="ta",
        input_types=(Series, Series),
        output_type=Series,
        arity=2,
    )
    registry.register(
        "ta_roc",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_trix",
        _passthrough,
        category="ta",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    registry.register(
        "ta_vwap",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series, Series),
        output_type=Series,
        arity=4,
    )
    registry.register(
        "ta_cci",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_donchian_upper",
        _passthrough,
        category="ta",
        input_types=(Series, Series),
        output_type=Series,
        arity=2,
    )
    registry.register(
        "ta_donchian_lower",
        _passthrough,
        category="ta",
        input_types=(Series, Series),
        output_type=Series,
        arity=2,
    )
    registry.register(
        "ta_atr",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series),
        output_type=Series,
        arity=3,
    )
    registry.register(
        "ta_mfi",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series, Series),
        output_type=Series,
        arity=4,
    )
    registry.register(
        "ta_obv",
        _passthrough,
        category="ta",
        input_types=(Series, Series),
        output_type=Series,
        arity=2,
    )
    registry.register(
        "ta_ad",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series, Series),
        output_type=Series,
        arity=4,
    )
    registry.register(
        "ta_cdl_engulfing",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series, Series),
        output_type=BoolSeries,
        arity=4,
    )
    registry.register(
        "ta_cdl_hammer",
        _passthrough,
        category="ta",
        input_types=(Series, Series, Series, Series),
        output_type=BoolSeries,
        arity=4,
    )
    registry.register(
        "crosses_above",
        lambda a, b: a,
        category="crossover",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "crosses_below",
        lambda a, b: a,
        category="crossover",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "lt",
        lambda a, b: a,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "gt",
        lambda a, b: a,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "and_op",
        lambda a, b: a,
        category="logic",
        input_types=(BoolSeries, BoolSeries),
        output_type=BoolSeries,
    )
    registry.register(
        "n_bars_ago",
        lambda a, shift=1: a,
        category="temporal",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    return registry


class TestStrategySeedRegistry:
    """Verify known seed registry metadata and lookup behavior."""

    def test_known_seed_names(self) -> None:
        assert set(list_known_strategy_seeds()) == {
            "ema_crossover",
            "sma_crossover",
            "rsi_oversold",
            "rsi_overbought",
            "macd_bullish_cross",
            "bollinger_breakout",
            "macd_bearish_cross",
            "ema_bearish_cross",
            "stochastic_bullish_cross",
            "stochrsi_bullish_cross",
            "stochastic_oversold",
            "stochrsi_oversold",
            "aroon_momentum",
            "aroonosc_bullish_cross",
            "mom_bullish_cross",
            "apo_bullish_cross",
            "cmo_oversold",
            "cmo_overbought",
            "sar_bullish",
            "adx_trend",
            "mama_bullish_cross",
            "williams_r_oversold",
            "williams_r_overbought",
            "roc_bullish_cross",
            "trix_bullish_cross",
            "vwap_support",
            "cci_oversold",
            "cci_overbought",
            "bollinger_oversold",
            "donchian_breakout",
            "atr_volatility_spike",
            "mfi_oversold",
            "mfi_overbought",
            "obv_uptrend",
            "ad_ascending",
            "cdl_engulfing_bullish",
            "cdl_hammer",
        }

    def test_get_unknown_seed_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown seed"):
            get_seed_template("not_a_seed")

    @pytest.mark.parametrize(
        "name",
        ["EMA-Crossover", "ema crossover", "  rsi_oversold  ", "CCI-OVERSOLD"],
    )
    def test_seed_lookup_normalizes_name(self, name: str) -> None:
        template = get_seed_template(name)
        assert isinstance(template, StrategySeedTemplate)

    def test_get_template_metadata(self) -> None:
        template = get_seed_template("ema_crossover")
        assert isinstance(template, StrategySeedTemplate)
        assert template.name == "ema_crossover"


class TestStrategySeedBuilders:
    """Verify building stock signal templates into valid programs."""

    def test_build_strategy_seed_raises_for_missing_primitive(self) -> None:
        template = get_seed_template("ema_crossover")
        with pytest.raises(
            ConfigurationError,
            match=r"'ema_crossover' seed requires primitive 'ta_ema'",
        ):
            template.builder(PrimitiveRegistry())

    def test_build_all_known_seeds(self) -> None:
        registry = _seed_registry()
        seeds = list_known_strategy_seeds()
        programs = build_strategy_seeds(seeds, registry)
        assert len(programs) == len(seeds)

    @pytest.mark.parametrize(
        "seed_name",
        [
            "ema_crossover",
            "sma_crossover",
            "rsi_oversold",
            "rsi_overbought",
            "macd_bullish_cross",
            "bollinger_breakout",
            "macd_bearish_cross",
            "ema_bearish_cross",
            "stochastic_bullish_cross",
            "stochrsi_bullish_cross",
            "stochastic_oversold",
            "stochrsi_oversold",
            "aroon_momentum",
            "mama_bullish_cross",
            "williams_r_oversold",
            "williams_r_overbought",
            "roc_bullish_cross",
            "trix_bullish_cross",
            "mom_bullish_cross",
            "apo_bullish_cross",
            "cmo_oversold",
            "cmo_overbought",
            "sar_bullish",
            "vwap_support",
            "aroonosc_bullish_cross",
            "cci_oversold",
            "cci_overbought",
            "bollinger_oversold",
            "donchian_breakout",
            "atr_volatility_spike",
            "mfi_oversold",
            "mfi_overbought",
            "obv_uptrend",
            "ad_ascending",
            "cdl_engulfing_bullish",
            "cdl_hammer",
            "adx_trend",
        ],
    )
    def test_build_strategy_seed_dispatch(self, seed_name: str) -> None:
        registry = _seed_registry()
        program = build_strategy_seed(seed_name, registry)
        assert isinstance(program, FunctionNode)
        assert program.output_type == BoolSeries

    @pytest.mark.parametrize(
        ("seed_name", "invalid_kwargs", "match"),
        [
            ("ema_crossover", {"fast_period": 0}, "ema periods must be positive"),
            ("ema_crossover", {"fast_period": 20, "slow_period": 20}, "fast_period must be smaller"),
            ("sma_crossover", {"slow_period": 0}, "sma periods must be positive"),
            ("sma_crossover", {"fast_period": 20, "slow_period": 20}, "fast_period must be smaller"),
            ("rsi_oversold", {"threshold": -1}, "rsi threshold must be between 0 and 100"),
            ("rsi_oversold", {"period": 0}, "rsi period must be positive"),
            ("rsi_overbought", {"threshold": 101}, "rsi threshold must be between 0 and 100"),
            ("rsi_overbought", {"period": 0}, "rsi period must be positive"),
            (
                "macd_bullish_cross",
                {"fast_period": 20, "slow_period": 20},
                "fast_period must be smaller than slow_period",
            ),
            (
                "macd_bullish_cross",
                {"signal_period": 0},
                "macd periods must be positive",
            ),
            (
                "bollinger_breakout",
                {"std_dev": 0},
                "std_dev must be greater than 0",
            ),
            ("bollinger_breakout", {"period": 0}, "bollinger period must be positive"),
            (
                "macd_bearish_cross",
                {"fast_period": 20, "slow_period": 20},
                "fast_period must be smaller than slow_period",
            ),
            ("macd_bearish_cross", {"fast_period": 0, "slow_period": 26}, "macd periods must be positive"),
            (
                "ema_bearish_cross",
                {"fast_period": 20, "slow_period": 20},
                "fast_period must be smaller than slow_period",
            ),
            ("ema_bearish_cross", {"fast_period": 0}, "ema periods must be positive"),
            (
                "stochastic_bullish_cross",
                {"k_period": 0},
                "stochastic periods must be positive",
            ),
            (
                "stochrsi_bullish_cross",
                {"d_period": 0},
                "stochrsi periods must be positive",
            ),
            (
                "stochastic_oversold",
                {"threshold": 101},
                "stochastic threshold must be between 0 and 100",
            ),
            (
                "stochastic_oversold",
                {"k_period": 0},
                "stochastic periods must be positive",
            ),
            (
                "stochrsi_oversold",
                {"threshold": -1},
                "stochrsi threshold must be between 0 and 100",
            ),
            (
                "stochrsi_oversold",
                {"rsi_period": 0},
                "stochrsi periods must be positive",
            ),
            ("aroon_momentum", {"up_threshold": -1}, "up_threshold must be between 0 and 100"),
            ("aroon_momentum", {"period": 0}, "aroon period must be positive"),
            (
                "mama_bullish_cross",
                {"fast_limit": 0},
                "fast_limit must be between 0 and 1",
            ),
            ("mama_bullish_cross", {"slow_limit": 0}, "slow_limit must be between 0 and 1"),
            (
                "aroonosc_bullish_cross",
                {"threshold": 101},
                "threshold must be between -100 and 100",
            ),
            ("aroonosc_bullish_cross", {"period": 0}, "aroonosc period must be positive"),
            (
                "williams_r_oversold",
                {"threshold": 1},
                "threshold must be between -100 and 0",
            ),
            ("williams_r_oversold", {"period": 0}, "williams_r period must be positive"),
            (
                "williams_r_overbought",
                {"threshold": 1},
                "threshold must be between -100 and 0",
            ),
            ("williams_r_overbought", {"period": 0}, "williams_r period must be positive"),
            ("roc_bullish_cross", {"period": 0}, "roc period must be positive"),
            ("trix_bullish_cross", {"period": 0}, "trix period must be positive"),
            ("mom_bullish_cross", {"period": 0}, "momentum period must be positive"),
            (
                "apo_bullish_cross",
                {"fast_period": 20, "slow_period": 20},
                "fast_period must be smaller than slow_period",
            ),
            ("apo_bullish_cross", {"fast_period": 0}, "apo periods must be positive"),
            ("cmo_oversold", {"threshold": -200}, "threshold must be between -100 and 100"),
            ("cmo_oversold", {"period": 0}, "cmo period must be positive"),
            ("cmo_overbought", {"threshold": 200}, "threshold must be between -100 and 100"),
            ("cmo_overbought", {"period": 0}, "cmo period must be positive"),
            ("sar_bullish", {"af_start": 1.1}, "af_start must be between 0 and 1"),
            ("sar_bullish", {"af_step": 0.8, "af_max": 0.5}, "af_step must be smaller than or equal to af_max"),
            ("sar_bullish", {"af_step": -0.2}, "af_step must be between 0 and 1"),
            ("sar_bullish", {"af_max": 1.1}, "af_max must be between 0 and 1"),
            ("adx_trend", {"adx_threshold": 101.0}, "adx_threshold must be between 0 and 100"),
            ("adx_trend", {"period": 0}, "adx period must be positive"),
            ("vwap_support", {"op": "gte"}, "op must be 'gt' or 'lt'"),
            ("cci_oversold", {"threshold": 5001}, "cci threshold must be in a realistic range"),
            ("cci_oversold", {"period": 0}, "cci period must be positive"),
            ("cci_overbought", {"threshold": -5001}, "cci threshold must be in a realistic range"),
            ("cci_overbought", {"period": 0}, "cci period must be positive"),
            ("bollinger_oversold", {"std_dev": 0}, "std_dev must be greater than 0"),
            ("bollinger_oversold", {"period": 0}, "bollinger period must be positive"),
            ("donchian_breakout", {"direction": "sideways"}, "direction must be 'up' or 'down'"),
            ("donchian_breakout", {"period": 0}, "donchian period must be positive"),
            ("atr_volatility_spike", {"lookback": 0}, "lookback must be positive"),
            ("atr_volatility_spike", {"period": 0}, "atr period must be positive"),
            ("mfi_oversold", {"threshold": -10}, "mfi threshold must be between 0 and 100"),
            ("mfi_oversold", {"period": 0}, "mfi period must be positive"),
            ("mfi_overbought", {"threshold": 120}, "mfi threshold must be between 0 and 100"),
            ("mfi_overbought", {"period": 0}, "mfi period must be positive"),
            ("obv_uptrend", {"lookback": 0}, "lookback must be positive"),
            ("ad_ascending", {"lookback": 0}, "lookback must be positive"),
        ],
    )
    def test_builder_validation_branches(
        self,
        seed_name: str,
        invalid_kwargs: dict[str, object],
        match: str,
    ) -> None:
        template = get_seed_template(seed_name)
        with pytest.raises(ValueError, match=match):
            template.builder(_seed_registry(), **invalid_kwargs)

    def test_donchian_breakout_can_build_direction_down(self) -> None:
        template = get_seed_template("donchian_breakout")
        program = template.builder(_seed_registry(), direction="down")
        assert isinstance(program, FunctionNode)
