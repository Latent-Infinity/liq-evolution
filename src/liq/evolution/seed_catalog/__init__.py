"""Composable registry for built-in strategy seed templates."""

from __future__ import annotations

from collections.abc import Iterable

from liq.evolution.errors import ConfigurationError
from liq.evolution.program import Program
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog._core import (
    SeedTemplateRole,
    StrategySeedTemplate,
    _normalize_seed_name,
)

from .candles import SEED_TEMPLATES as _CANDLE_SEEDS
from .carry_spread import SEED_TEMPLATES as _CARRY_SPREAD_SEEDS
from .momentum import SEED_TEMPLATES as _MOMENTUM_SEEDS
from .regimes import SEED_TEMPLATES as _REGIME_SEEDS
from .trend import SEED_TEMPLATES as _TREND_SEEDS
from .volatility import SEED_TEMPLATES as _VOLATILITY_SEEDS


_DETECTOR_TEMPLATES = {
    "atr_volatility_spike",
    "bollinger_breakout",
    "bollinger_oversold",
    "cci_oversold",
    "cci_overbought",
    "donchian_breakout",
    "obv_uptrend",
    "ad_ascending",
    "rolling_stddev_spike",
    "regime_switching_momentum_volatility",
    "var_spike",
    "vwap_support",
    "mfi_oversold",
    "mfi_overbought",
}

_RISK_TEMPLATES = {
    "carry_spread_expansion",
    "carry_spread_contraction",
    "regime_switching_trend_mean_reversion",
    "var_spike",
}

_DEFAULT_REGIME_HINTS = {
    SeedTemplateRole.expert: ("momentum", "trend"),
    SeedTemplateRole.detector: ("volatility", "trend", "liquidity"),
    SeedTemplateRole.risk: ("risk", "volatility_cap"),
}

_TURNOVER_EXPECTATION = {
    SeedTemplateRole.detector: 0.18,
    SeedTemplateRole.risk: 0.25,
}


def _template_role(name: str) -> SeedTemplateRole:
    if name in _RISK_TEMPLATES:
        return SeedTemplateRole.risk
    if name in _DETECTOR_TEMPLATES:
        return SeedTemplateRole.detector
    return SeedTemplateRole.expert


def _template_turnover_expectation(name: str, role: SeedTemplateRole) -> float | None:
    if name in _CARRY_SPREAD_SEEDS:
        return _TEMPLATE_TURNOVER_BY_NAME.get(name)
    return _TURNOVER_EXPECTATION.get(role)


_TEMPLATE_TURNOVER_BY_NAME = {
    "carry_spread_expansion": 0.2,
    "carry_spread_contraction": 0.2,
    "atr_volatility_spike": 0.32,
    "rolling_stddev_spike": 0.28,
    "var_spike": 0.29,
}


def _template_regime_hints(name: str, role: SeedTemplateRole) -> tuple[str, ...]:
    if role == SeedTemplateRole.risk:
        if name.startswith("carry_spread"):
            return ("spread", "volatility", "kill_switch")
        if name.startswith("regime_switching"):
            return ("volatility", "trend_transition")
        return ("risk",)

    if name in _DETECTOR_TEMPLATES:
        if name.startswith("vwap"):
            return ("liquidity", "support")
        if name.startswith("obv") or name.startswith("ad_"):
            return ("flow", "pressure")
        if name.endswith("_momentum") or name.startswith("mfi_"):
            return ("momentum", "volatility")
        return ("trend", "breakout")

    if name in {
        "rsi_oversold",
        "rsi_overbought",
        "stochastic_oversold",
        "stochrsi_oversold",
        "stochrsi_bullish_cross",
        "williams_r_oversold",
        "williams_r_overbought",
    }:
        return ("momentum", "mean_reversion")

    return _DEFAULT_REGIME_HINTS[role]


def _template_failure_modes(name: str) -> tuple[str, ...]:
    if name in _DETECTOR_TEMPLATES:
        return ("stale_volatility", "weak_signal")
    if name in _RISK_TEMPLATES:
        return ("risk_breach", "excess_turnover")
    return ("flat_market",)


def _template_expected_inputs(name: str) -> tuple[str, ...]:
    if name in {"carry_spread_expansion", "carry_spread_contraction"}:
        return ("close", "high", "low")
    if name in {"vwap_support", "obv_uptrend", "ad_ascending"}:
        return ("open", "high", "low", "close", "volume")
    if name in {"mfi_oversold", "mfi_overbought"}:
        return ("high", "low", "close", "volume")
    return ("close",)


def _coerce_template(template: StrategySeedTemplate) -> StrategySeedTemplate:
    name = _normalize_seed_name(template.name)
    role = _template_role(name)

    turnover = _template_turnover_expectation(name, role)
    if turnover is None and template.turnover_expectation is not None:
        turnover = template.turnover_expectation

    expected_inputs = _template_expected_inputs(name)
    return StrategySeedTemplate(
        name=template.name,
        description=template.description,
        builder=template.builder,
        block_role=role,
        arity=len(expected_inputs),
        expected_inputs=expected_inputs,
        regime_hints=_template_regime_hints(name, role),
        turnover_expectation=turnover,
        failure_modes=_template_failure_modes(name),
    )

_STRATEGY_MODULES = (
    _TREND_SEEDS,
    _MOMENTUM_SEEDS,
    _VOLATILITY_SEEDS,
    _CANDLE_SEEDS,
    _CARRY_SPREAD_SEEDS,
    _REGIME_SEEDS,
)

_SEED_NAMES: list[str] = []
_SEED_REGISTRY: dict[str, StrategySeedTemplate] = {}

for _module_seeds in _STRATEGY_MODULES:
    for _template in _module_seeds:
        _normalized = _normalize_seed_name(_template.name)
        if _normalized in _SEED_REGISTRY:
            raise ConfigurationError(
                f"Duplicate seed strategy registration: {_normalized}"
            )

        _template = _coerce_template(_template)
        _SEED_NAMES.append(_normalized)
        _SEED_REGISTRY[_normalized] = _template


def list_known_strategy_seeds() -> list[str]:
    """Return all known seed names in registry order."""
    return list(_SEED_NAMES)


def get_seed_template(name: str) -> StrategySeedTemplate:
    """Return metadata for a known seed name."""
    normalized = _normalize_seed_name(name)
    try:
        return _SEED_REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(list_known_strategy_seeds())
        msg = f"Unknown seed {name!r}. Available seeds: {available}"
        raise KeyError(msg) from exc


def build_strategy_seed(name: str, registry: PrimitiveRegistry) -> Program:
    """Build a single known strategy seed."""
    template = get_seed_template(name)
    return template.builder(registry)


def build_strategy_seeds(
    names: Iterable[str], registry: PrimitiveRegistry
) -> list[Program]:
    """Build multiple known strategy seeds."""
    return [build_strategy_seed(name, registry) for name in names]


def list_seed_templates_by_role(role: SeedTemplateRole | str) -> list[str]:
    """Return canonical template names for the requested seed role."""
    if isinstance(role, str):
        try:
            role = SeedTemplateRole(role)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in SeedTemplateRole)
            raise ValueError(
                f"template role must be one of: {allowed}"
            ) from exc

    return [
        name
        for name in _SEED_NAMES
        for template in [_SEED_REGISTRY[name]]
        if template.block_role == role
    ]


__all__ = [
    "StrategySeedTemplate",
    "SeedTemplateRole",
    "build_strategy_seed",
    "build_strategy_seeds",
    "get_seed_template",
    "list_known_strategy_seeds",
    "list_seed_templates_by_role",
]
