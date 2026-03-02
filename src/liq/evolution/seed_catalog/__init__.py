"""Composable registry for built-in strategy seed templates."""

from __future__ import annotations

from collections.abc import Iterable

from liq.evolution.errors import ConfigurationError
from liq.evolution.program import Program
from liq.evolution.protocols import PrimitiveRegistry
from liq.evolution.seed_catalog._core import StrategySeedTemplate, _normalize_seed_name

from .candles import SEED_TEMPLATES as _CANDLE_SEEDS
from .momentum import SEED_TEMPLATES as _MOMENTUM_SEEDS
from .regimes import SEED_TEMPLATES as _REGIME_SEEDS
from .trend import SEED_TEMPLATES as _TREND_SEEDS
from .volatility import SEED_TEMPLATES as _VOLATILITY_SEEDS

_STRATEGY_MODULES = (
    _TREND_SEEDS,
    _MOMENTUM_SEEDS,
    _VOLATILITY_SEEDS,
    _CANDLE_SEEDS,
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


__all__ = [
    "StrategySeedTemplate",
    "build_strategy_seed",
    "build_strategy_seeds",
    "get_seed_template",
    "list_known_strategy_seeds",
]
