"""Trading primitive registry builder.

Constructs a :class:`~liq.gp.PrimitiveRegistry` populated with
domain-specific trading primitives based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from liq.evolution.config import PrimitiveConfig
from liq.evolution.errors import PrimitiveSetupError
from liq.evolution.primitives.feature_context import FeatureContext
from liq.evolution.primitives.ops_comparison import register_comparison_ops
from liq.evolution.primitives.ops_crossover import register_crossover_ops
from liq.evolution.primitives.ops_logic import register_logic_ops
from liq.evolution.primitives.ops_numeric import register_numeric_ops
from liq.evolution.primitives.ops_temporal import register_temporal_ops
from liq.evolution.primitives.series_sources import register_series_sources
from liq.gp.primitives.registry import PrimitiveRegistry

if TYPE_CHECKING:
    from liq.evolution.protocols import IndicatorBackend


def build_trading_registry(
    config: PrimitiveConfig,
    backend: IndicatorBackend | None = None,
) -> PrimitiveRegistry:
    """Build a GP primitive registry with trading-specific primitives.

    Args:
        config: Primitive configuration controlling which categories
            to register.
        backend: Optional indicator backend for liq-ta primitives.

    Returns:
        A populated ``PrimitiveRegistry``.

    Raises:
        PrimitiveSetupError: If registration fails.
    """
    try:
        registry = PrimitiveRegistry()

        if config.enable_series_sources:
            register_series_sources(registry)
        if config.enable_numeric_ops:
            register_numeric_ops(registry)
        if config.enable_comparison_ops:
            register_comparison_ops(registry)
        if config.enable_logic_ops:
            register_logic_ops(registry)
        if config.enable_crossover_ops:
            register_crossover_ops(registry)
        if config.enable_temporal_ops:
            register_temporal_ops(registry)
        if config.enable_liq_ta and backend is not None:
            from liq.evolution.primitives.indicators_liq_ta import (
                register_liq_ta_indicators,
            )

            # Enable in-memory caching by default for indicator computations.
            # Avoid double-wrapping when callers already provide a cache wrapper.
            cached_backend = backend
            if not isinstance(backend, FeatureContext):
                cached_backend = FeatureContext(backend)

            register_liq_ta_indicators(registry, cached_backend)

        return registry
    except Exception as exc:
        raise PrimitiveSetupError(f"Failed to build trading registry: {exc}") from exc
