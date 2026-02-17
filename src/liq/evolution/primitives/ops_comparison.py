"""Comparison operator primitives (>, <, >=, <=, ==, !=)."""

from __future__ import annotations

import numpy as np

from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, Series


def safe_gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise greater than; NaN comparisons produce 0.0."""
    return np.where(a > b, 1.0, 0.0)


def safe_lt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise less than; NaN comparisons produce 0.0."""
    return np.where(a < b, 1.0, 0.0)


def safe_gte(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise greater than or equal; NaN comparisons produce 0.0."""
    return np.where(a >= b, 1.0, 0.0)


def safe_lte(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise less than or equal; NaN comparisons produce 0.0."""
    return np.where(a <= b, 1.0, 0.0)


def safe_eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise equality; NaN comparisons produce 0.0."""
    return np.where(a == b, 1.0, 0.0)


def safe_neq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise inequality; NaN comparisons produce 0.0 (NaN != NaN is True in IEEE 754)."""
    mask = (a != b) & ~np.isnan(a) & ~np.isnan(b)
    return np.where(mask, 1.0, 0.0)


def register_comparison_ops(registry: PrimitiveRegistry) -> None:
    """Register comparison operator primitives into the registry.

    Args:
        registry: The GP primitive registry to populate.
    """
    registry.register(
        "gt",
        safe_gt,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "lt",
        safe_lt,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "gte",
        safe_gte,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "lte",
        safe_lte,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "eq",
        safe_eq,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    registry.register(
        "neq",
        safe_neq,
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
