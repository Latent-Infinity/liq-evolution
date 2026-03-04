"""Shared types and helpers for seed catalog builders."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from liq.evolution.errors import ConfigurationError
from liq.evolution.program import (
    Program,
    TerminalNode,
)
from liq.evolution.protocols import PrimitiveRegistry
from liq.gp.types import Series


class SeedTemplateRole(StrEnum):
    """Typed block role for structured seed templates."""

    detector = "detector"
    expert = "expert"
    risk = "risk"


def _normalize_str_tuple(values: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("template metadata fields must be strings")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("template metadata fields must contain non-empty strings")
        normalized.append(cleaned)
    return tuple(normalized)


@dataclass(frozen=True)
class StrategySeedTemplate:
    """Metadata for a known seed strategy."""

    name: str
    description: str
    builder: Callable[[PrimitiveRegistry], Program]
    block_role: SeedTemplateRole = SeedTemplateRole.expert
    arity: int = 1
    expected_inputs: tuple[str, ...] = ("close",)
    regime_hints: tuple[str, ...] = ()
    turnover_expectation: float | None = None
    failure_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("template name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("template description must be a non-empty string")
        if not callable(self.builder):
            raise TypeError("template builder must be callable")

        if isinstance(self.block_role, str):
            try:
                role = SeedTemplateRole(self.block_role)
            except ValueError as exc:
                allowed = ", ".join(role.value for role in SeedTemplateRole)
                raise ValueError(
                    "template block_role must be one of: " + allowed
                ) from exc
            object.__setattr__(self, "block_role", role)
        elif not isinstance(self.block_role, SeedTemplateRole):
            raise TypeError("template block_role must be a SeedTemplateRole")

        if not isinstance(self.arity, int) or self.arity < 1:
            raise ValueError("template arity must be a positive integer")

        object.__setattr__(self, "expected_inputs", _normalize_str_tuple(self.expected_inputs))
        if len(self.expected_inputs) != self.arity:
            raise ValueError(
                "template arity must match expected_inputs length: "
                f"{self.arity} != {len(self.expected_inputs)}"
            )

        object.__setattr__(self, "regime_hints", _normalize_str_tuple(self.regime_hints))
        object.__setattr__(
            self,
            "failure_modes",
            _normalize_str_tuple(self.failure_modes),
        )

        if self.turnover_expectation is not None:
            if not isinstance(self.turnover_expectation, (int, float)):
                raise TypeError("template turnover_expectation must be numeric when set")
            turnover = float(self.turnover_expectation)
            if not math.isfinite(turnover):
                raise ValueError("template turnover_expectation must be finite")
            if not 0.0 <= turnover <= 1.0:
                raise ValueError("template turnover_expectation must be in [0, 1]")
            object.__setattr__(self, "turnover_expectation", turnover)


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
    legacy_aliases: dict[str, tuple[str, ...]] = {
        "ta_macd_macd_line": ("ta_macd_macd", "macd_macd"),
        "macd_macd_line": ("ta_macd_macd", "macd_macd"),
        "ta_macd_signal_line": ("ta_macd_signal", "macd_signal"),
        "macd_signal_line": ("ta_macd_signal", "macd_signal"),
        "ta_stochastic_k": (
            "ta_stochastic_slow_k",
            "stochastic_slow_k",
            "ta_stochastic_fast_k",
            "stochastic_fast_k",
            "stochastic_fastk",
        ),
        "stochastic_k": (
            "ta_stochastic_slow_k",
            "stochastic_slow_k",
            "ta_stochastic_fast_k",
            "stochastic_fast_k",
            "stochastic_fastk",
        ),
        "ta_stochastic_d": (
            "ta_stochastic_slow_d",
            "stochastic_slow_d",
            "ta_stochastic_fast_d",
            "stochastic_fast_d",
            "stochastic_d",
        ),
        "stochastic_d": (
            "ta_stochastic_slow_d",
            "stochastic_slow_d",
            "ta_stochastic_fast_d",
            "stochastic_fast_d",
            "stochastic_d",
        ),
    }

    def _candidate_primitive_names(requested: str) -> list[str]:
        lowered = requested.lower()
        candidates: list[str] = []
        pending: list[str] = [lowered]
        seen_candidates: set[str] = set()

        while pending:
            candidate = pending.pop()
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            candidates.append(candidate)

            if candidate.startswith("ta_"):
                base = candidate[3:]
                if base not in seen_candidates:
                    pending.append(base)
            else:
                prefixed = f"ta_{candidate}"
                if prefixed not in seen_candidates:
                    pending.append(prefixed)

            parts = candidate.split("_")
            if len(parts) > 1 and parts[0]:
                for idx in range(1, len(parts)):
                    if parts[idx] == parts[0]:
                        collapsed = "_".join(parts[:idx] + parts[idx + 1 :])
                        if collapsed and collapsed not in seen_candidates:
                            pending.append(collapsed)
                        ta_collapsed = f"ta_{collapsed}"
                        if ta_collapsed not in seen_candidates:
                            pending.append(ta_collapsed)

        expanded: list[str] = []
        for candidate in candidates:
            expanded.append(candidate)
            parts = candidate.split("_")
            if len(parts) > 1 and parts[0]:
                for idx in range(1, len(parts)):
                    if parts[idx] == parts[0]:
                        collapsed = "_".join(parts[:idx] + parts[idx + 1 :])
                        expanded.append(collapsed)
                        if idx == 1 and not collapsed.startswith(f"{parts[0]}_"):
                            expanded.append(f"{parts[0]}_{collapsed}")

        normalized: list[str] = []
        seen: set[str] = set()
        for item in expanded:
            if item not in seen:
                seen.add(item)
                normalized.append(item)

        if lowered not in seen:
            normalized.append(lowered)

        # Backward-compatible aliases for indicator primitive naming changes.
        for candidate in list(normalized):
            for alias in legacy_aliases.get(candidate, ()):
                if alias not in seen:
                    seen.add(alias)
                    normalized.append(alias)
        return normalized

    requested = _candidate_primitive_names(name)

    available: set[str] = {item.name for item in registry.list_primitives()}
    alias_lookup: dict[str, str] = {}
    for available_name in available:
        for alias in _candidate_primitive_names(available_name):
            alias_lookup.setdefault(alias, available_name)

    for candidate in requested:
        if candidate in available:
            try:
                return registry.get(candidate)
            except Exception:  # pragma: no cover - fallback
                continue
        canonical = alias_lookup.get(candidate)
        if canonical is not None:
            try:
                return registry.get(canonical)
            except Exception:  # pragma: no cover - fallback
                continue

    msg = (
        f"{seed!r} seed requires primitive {name!r} to be registered "
        f"in the strategy registry"
    )
    raise ConfigurationError(msg)


def _validate_indicator_periods(slow_period: int, fast_period: int) -> None:
    """Validate moving-average crossover period ordering."""
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")


def _normalize_seed_name(name: str) -> str:
    return name.strip().replace("-", "_").replace(" ", "_").lower()
