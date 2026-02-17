"""Crossover detection operator primitives."""

from __future__ import annotations

import numpy as np

from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, Series


def safe_crosses_above(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Detect where A crosses above B; bar 0 is always 0.0.

    A[t-1] <= B[t-1] AND A[t] > B[t].
    """
    result = np.zeros(len(a), dtype=np.float64)
    result[1:] = np.where(
        (a[:-1] <= b[:-1]) & (a[1:] > b[1:]),
        1.0,
        0.0,
    )
    return result


def safe_crosses_below(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Detect where A crosses below B; bar 0 is always 0.0.

    A[t-1] >= B[t-1] AND A[t] < B[t].
    """
    result = np.zeros(len(a), dtype=np.float64)
    result[1:] = np.where(
        (a[:-1] >= b[:-1]) & (a[1:] < b[1:]),
        1.0,
        0.0,
    )
    return result


def safe_closes_above(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise A > B (simple)."""
    return np.where(a > b, 1.0, 0.0)


def safe_closes_below(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise A < B (simple)."""
    return np.where(a < b, 1.0, 0.0)


def register_crossover_ops(registry: PrimitiveRegistry) -> None:
    """Register crossover detection operator primitives into the registry.

    Args:
        registry: The GP primitive registry to populate.
    """
    registry.register(
        "crosses_above",
        safe_crosses_above,
        category="crossover",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "crosses_below",
        safe_crosses_below,
        category="crossover",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "closes_above",
        safe_closes_above,
        category="crossover",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "closes_below",
        safe_closes_below,
        category="crossover",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
