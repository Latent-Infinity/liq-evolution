"""Temporal/lag operator primitives (shift, rolling, etc.)."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, ParamSpec, Series


def safe_is_rising(a: np.ndarray, *, period: int = 3) -> np.ndarray:
    """1.0 where a[t] > a[t - period + 1]; leading bars are 0.0."""
    result = np.zeros(len(a), dtype=np.float64)
    offset = period - 1
    if offset < len(a):
        result[offset:] = np.where(a[offset:] > a[: len(a) - offset], 1.0, 0.0)
    return result


def safe_is_falling(a: np.ndarray, *, period: int = 3) -> np.ndarray:
    """1.0 where a[t] < a[t - period + 1]; leading bars are 0.0."""
    result = np.zeros(len(a), dtype=np.float64)
    offset = period - 1
    if offset < len(a):
        result[offset:] = np.where(a[offset:] < a[: len(a) - offset], 1.0, 0.0)
    return result


def safe_n_bars_ago(a: np.ndarray, *, shift: int = 1) -> np.ndarray:
    """Shift series by `shift` bars; leading bars are NaN."""
    result = np.empty(len(a), dtype=np.float64)
    result[:shift] = np.nan
    result[shift:] = a[: len(a) - shift]
    return result


def safe_highest(a: np.ndarray, *, period: int = 20) -> np.ndarray:
    """Rolling maximum over `period` bars; leading bars are NaN."""
    result = np.empty(len(a), dtype=np.float64)
    result[: period - 1] = np.nan
    if period <= len(a):
        windows = sliding_window_view(a, period)
        result[period - 1 :] = np.max(windows, axis=1)
    return result


def safe_lowest(a: np.ndarray, *, period: int = 20) -> np.ndarray:
    """Rolling minimum over `period` bars; leading bars are NaN."""
    result = np.empty(len(a), dtype=np.float64)
    result[: period - 1] = np.nan
    if period <= len(a):
        windows = sliding_window_view(a, period)
        result[period - 1 :] = np.min(windows, axis=1)
    return result


def safe_percentile_rank(a: np.ndarray, *, period: int = 20) -> np.ndarray:
    """Percentile rank of current value within rolling window; leading bars NaN.

    Returns 0-100 scale.
    """
    result = np.empty(len(a), dtype=np.float64)
    result[: period - 1] = np.nan
    if period <= len(a):
        windows = sliding_window_view(a, period)
        # For each window, count values <= current value, normalize to 0-100
        current = a[period - 1 :]
        count_le = np.sum(windows <= current[:, np.newaxis], axis=1)
        # Rank: (count_le - 1) / (period - 1) * 100
        result[period - 1 :] = (count_le - 1) / (period - 1) * 100.0
    return result


def safe_greater_count(
    a: np.ndarray,
    b: np.ndarray,
    *,
    period: int = 10,
) -> np.ndarray:
    """Count bars where a >= b within rolling window; leading bars NaN."""
    result = np.empty(len(a), dtype=np.float64)
    result[: period - 1] = np.nan
    if period <= len(a):
        gt = np.where(a >= b, 1.0, 0.0)
        windows = sliding_window_view(gt, period)
        result[period - 1 :] = np.sum(windows, axis=1)
    return result


def safe_lower_count(
    a: np.ndarray,
    b: np.ndarray,
    *,
    period: int = 10,
) -> np.ndarray:
    """Count bars where a <= b within rolling window; leading bars NaN."""
    result = np.empty(len(a), dtype=np.float64)
    result[: period - 1] = np.nan
    if period <= len(a):
        lt = np.where(a <= b, 1.0, 0.0)
        windows = sliding_window_view(lt, period)
        result[period - 1 :] = np.sum(windows, axis=1)
    return result


def safe_change(a: np.ndarray, *, period: int = 1) -> np.ndarray:
    """Absolute change: a[t] - a[t - period]; leading bars NaN."""
    result = np.empty(len(a), dtype=np.float64)
    result[:period] = np.nan
    result[period:] = a[period:] - a[: len(a) - period]
    return result


def safe_pct_change(a: np.ndarray, *, period: int = 1) -> np.ndarray:
    """Percent change: (a[t] - a[t-period]) / a[t-period]; leading bars NaN."""
    result = np.empty(len(a), dtype=np.float64)
    result[:period] = np.nan
    prev = a[: len(a) - period]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (a[period:] - prev) / prev
    pct = np.where(np.isfinite(pct), pct, np.nan)
    result[period:] = pct
    return result


def register_temporal_ops(registry: PrimitiveRegistry) -> None:
    """Register temporal operator primitives into the registry.

    Args:
        registry: The GP primitive registry to populate.
    """
    registry.register(
        "is_rising",
        safe_is_rising,
        category="temporal",
        input_types=(Series,),
        output_type=BoolSeries,
        param_specs=[ParamSpec("period", int, 3, 2, 50)],
    )
    registry.register(
        "is_falling",
        safe_is_falling,
        category="temporal",
        input_types=(Series,),
        output_type=BoolSeries,
        param_specs=[ParamSpec("period", int, 3, 2, 50)],
    )
    registry.register(
        "n_bars_ago",
        safe_n_bars_ago,
        category="temporal",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ParamSpec("shift", int, 1, 1, 50)],
    )
    registry.register(
        "highest",
        safe_highest,
        category="temporal",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ParamSpec("period", int, 20, 2, 50)],
    )
    registry.register(
        "lowest",
        safe_lowest,
        category="temporal",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ParamSpec("period", int, 20, 2, 50)],
    )
    registry.register(
        "percentile_rank",
        safe_percentile_rank,
        category="temporal",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ParamSpec("period", int, 20, 2, 50)],
    )
    registry.register(
        "greater_count",
        safe_greater_count,
        category="temporal",
        input_types=(Series, Series),
        output_type=Series,
        param_specs=[ParamSpec("period", int, 10, 2, 50)],
    )
    registry.register(
        "lower_count",
        safe_lower_count,
        category="temporal",
        input_types=(Series, Series),
        output_type=Series,
        param_specs=[ParamSpec("period", int, 10, 2, 50)],
    )
    registry.register(
        "change",
        safe_change,
        category="temporal",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ParamSpec("period", int, 1, 1, 50)],
    )
    registry.register(
        "pct_change",
        safe_pct_change,
        category="temporal",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ParamSpec("period", int, 1, 1, 50)],
    )
