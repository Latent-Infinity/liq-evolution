"""Trading primitive registration for GP evolution."""

from liq.evolution.primitives.feature_context import FeatureContext
from liq.evolution.primitives.registry import build_trading_registry
from liq.evolution.primitives.series_sources import (
    prepare_evaluation_context,
    register_series_sources,
)

__all__ = [
    "FeatureContext",
    "build_trading_registry",
    "prepare_evaluation_context",
    "register_series_sources",
]
