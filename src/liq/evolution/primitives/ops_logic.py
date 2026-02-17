"""Logical operator primitives (and, or, not, if_then_else)."""

from __future__ import annotations

import numpy as np

from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import BoolSeries, Series

_TRUTH_THRESHOLD = 0.5


def safe_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise logical AND; values > 0.5 are truthy."""
    return np.where((a > _TRUTH_THRESHOLD) & (b > _TRUTH_THRESHOLD), 1.0, 0.0)


def safe_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise logical OR; values > 0.5 are truthy."""
    return np.where((a > _TRUTH_THRESHOLD) | (b > _TRUTH_THRESHOLD), 1.0, 0.0)


def safe_not(a: np.ndarray) -> np.ndarray:
    """Element-wise logical NOT; values > 0.5 are truthy."""
    return np.where(a > _TRUTH_THRESHOLD, 0.0, 1.0)


def safe_if_then_else(
    cond: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Element-wise conditional selection; cond > 0.5 selects a, else b."""
    return np.where(cond > _TRUTH_THRESHOLD, a, b)


def register_logic_ops(registry: PrimitiveRegistry) -> None:
    """Register logical operator primitives into the registry.

    Args:
        registry: The GP primitive registry to populate.
    """
    registry.register(
        "and_op",
        safe_and,
        category="logic",
        input_types=(BoolSeries, BoolSeries),
        output_type=BoolSeries,
    )
    registry.register(
        "or_op",
        safe_or,
        category="logic",
        input_types=(BoolSeries, BoolSeries),
        output_type=BoolSeries,
    )
    registry.register(
        "not_op",
        safe_not,
        category="logic",
        input_types=(BoolSeries,),
        output_type=BoolSeries,
    )
    registry.register(
        "if_then_else",
        safe_if_then_else,
        category="logic",
        input_types=(BoolSeries, Series, Series),
        output_type=Series,
    )
