"""Series source terminal primitives (OHLCV, derived, Heiken Ashi)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from numpy.random import Generator, default_rng

from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import Series

_TERMINAL_NAMES = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "log_returns",
    "midrange",
    "typical_price",
    "ha_open",
    "ha_high",
    "ha_low",
    "ha_close",
    "ephemeral_float",
]


def _terminal_missing(name: str) -> Callable[[], np.ndarray]:
    """Create a context-only terminal callable placeholder."""

    def _callable() -> np.ndarray:
        msg = (
            f"Terminal '{name}' is resolved from the evaluation context."
            " Add it to evaluation context before calling evaluate()."
        )
        raise KeyError(msg)

    return _callable


def register_series_sources(registry: PrimitiveRegistry) -> None:
    """Register price/volume series source terminals into the registry.

    Each terminal is registered with arity=0 (resolved from context by name).

    Args:
        registry: The GP primitive registry to populate.
    """
    for name in _TERMINAL_NAMES:
        registry.register(
            name,
            _terminal_missing(name),
            category="terminal",
            input_types=(),
            output_type=Series,
        )


def prepare_evaluation_context(
    ohlcv: dict[str, np.ndarray],
    *,
    rng: Generator | None = None,
) -> dict[str, np.ndarray]:
    """Expand raw OHLCV data into a full evaluation context.

    Computes derived series (log returns, midrange, typical price)
    and Heiken Ashi candles from raw OHLCV arrays.

    Args:
        ohlcv: Dict with keys ``open``, ``high``, ``low``, ``close``, ``volume``.
        rng: Optional random generator for ephemeral constant terminal.

    Returns:
        Dict with all original keys plus derived series.
    """
    ctx: dict[str, np.ndarray] = {}

    close = ohlcv["close"].astype(np.float64, copy=True)
    high = ohlcv["high"].astype(np.float64, copy=True)
    low = ohlcv["low"].astype(np.float64, copy=True)
    open_ = ohlcv["open"].astype(np.float64, copy=True)
    volume = ohlcv["volume"].astype(np.float64, copy=True)

    ctx["close"] = close
    ctx["open"] = open_
    ctx["high"] = high
    ctx["low"] = low
    ctx["volume"] = volume

    # Log returns: bar 0 is NaN (no prior bar)
    log_returns = np.empty_like(close)
    log_returns[0] = np.nan
    log_returns[1:] = np.log(close[1:] / close[:-1])
    ctx["log_returns"] = log_returns

    # Midrange and typical price
    ctx["midrange"] = (high + low) / 2.0
    ctx["typical_price"] = (high + low + close) / 3.0

    # Heiken Ashi
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = np.empty_like(close)
    ha_open[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, len(close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

    ctx["ha_open"] = ha_open
    ctx["ha_high"] = ha_high
    ctx["ha_low"] = ha_low
    ctx["ha_close"] = ha_close

    local_rng = rng or default_rng()
    ctx["ephemeral_float"] = np.full(
        len(close),
        local_rng.uniform(0.0, 1.0),
        dtype=np.float64,
    )

    return ctx
