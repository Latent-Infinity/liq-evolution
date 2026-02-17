"""Numeric operator primitives (+, -, *, /, abs, neg, etc.)."""

from __future__ import annotations

import numpy as np

from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import Series


def safe_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition."""
    return a + b


def safe_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise subtraction."""
    return a - b


def safe_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication."""
    return a * b


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division; division by zero produces NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
    result = np.where(np.isfinite(result), result, np.nan)
    return result


def safe_abs(a: np.ndarray) -> np.ndarray:
    """Element-wise absolute value."""
    return np.abs(a)


def safe_neg(a: np.ndarray) -> np.ndarray:
    """Element-wise negation."""
    return np.negative(a)


def safe_clip(data: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Element-wise clip between lo and hi."""
    return np.clip(data, lo, hi)


def safe_zscore(a: np.ndarray) -> np.ndarray:
    """Z-score normalization; zero-variance series returns zeros."""
    std = np.std(a)
    if std == 0.0:
        return np.zeros_like(a)
    return (a - np.mean(a)) / std


def safe_log(a: np.ndarray) -> np.ndarray:
    """Natural log; non-positive values produce NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(a)
    result = np.where(np.isfinite(result), result, np.nan)
    return result


def safe_sqrt(a: np.ndarray) -> np.ndarray:
    """Square root; negative values produce NaN."""
    with np.errstate(invalid="ignore"):
        return np.sqrt(a)


def safe_min_of(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise minimum."""
    return np.minimum(a, b)


def safe_max_of(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise maximum."""
    return np.maximum(a, b)


def register_numeric_ops(registry: PrimitiveRegistry) -> None:
    """Register numeric operator primitives into the registry.

    Args:
        registry: The GP primitive registry to populate.
    """
    registry.register(
        "add",
        safe_add,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "sub",
        safe_sub,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "mul",
        safe_mul,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "div",
        safe_div,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "abs",
        safe_abs,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    registry.register(
        "neg",
        safe_neg,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    registry.register(
        "clip",
        safe_clip,
        category="numeric",
        input_types=(Series, Series, Series),
        output_type=Series,
    )
    registry.register(
        "zscore",
        safe_zscore,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    registry.register(
        "log",
        safe_log,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    registry.register(
        "sqrt",
        safe_sqrt,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    registry.register(
        "min_of",
        safe_min_of,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "max_of",
        safe_max_of,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
