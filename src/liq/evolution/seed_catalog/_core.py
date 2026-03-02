"""Shared types and helpers for seed catalog builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from liq.evolution.errors import ConfigurationError
from liq.evolution.program import (
    Program,
    TerminalNode,
)
from liq.evolution.protocols import PrimitiveRegistry
from liq.gp.types import Series


@dataclass(frozen=True)
class StrategySeedTemplate:
    """Metadata for a known seed strategy."""

    name: str
    description: str
    builder: Callable[[PrimitiveRegistry], Program]


def _terminal(name: str) -> TerminalNode:
    """Build a typed price-feature terminal."""
    return TerminalNode(name=name, output_type=Series)


def _resolve_primitive(
    registry: PrimitiveRegistry,
    name: str,
    *,
    seed: str,
):
    """Resolve a primitive from the registry with clear context on failure."""
    try:
        return registry.get(name)
    except Exception as exc:
        msg = (
            f"{seed!r} seed requires primitive {name!r} to be registered "
            "in the strategy registry"
        )
        raise ConfigurationError(msg) from exc


def _validate_indicator_periods(slow_period: int, fast_period: int) -> None:
    """Validate moving-average crossover period ordering."""
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")


def _normalize_seed_name(name: str) -> str:
    return name.strip().replace("-", "_").replace(" ", "_").lower()
