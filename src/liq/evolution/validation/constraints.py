"""Strategy-level constraint checks and policy composition."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from liq.gp.program.ast import Program

ConstraintCheck = Callable[[Program, Mapping[str, Any]], Mapping[str, float] | None]


def _coerce_finite_nonneg(value: Any) -> float | None:
    """Return a finite non-negative float if possible."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _coerce_finite_float(value: Any) -> float | None:
    """Return a finite float if possible, including negatives."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _extract_sequence_floats(values: Any, *, absolute: bool = False) -> list[float]:
    """Extract finite float values from an iterable payload."""
    if not isinstance(values, (list, tuple)):
        return []

    output: list[float] = []
    for value in values:
        normalized = _coerce_finite_float(value)
        if normalized is None:
            continue
        output.append(abs(normalized) if absolute else normalized)
    return output


def constraint_no_future_reference(
    program: Program,
    payload: Mapping[str, Any],
) -> Mapping[str, float] | None:
    """Detect explicit future-reference leakage markers.

    Strategy evaluators can emit this marker directly in the split payload.
    """
    del program
    direct_flag = payload.get("future_reference")
    explicit = payload.get("future_reference_detected")
    direct_value = payload.get("future_reference_penalty")

    penalty = 0.0
    bool_penalty = _coerce_finite_nonneg(direct_flag)
    if bool_penalty is not None:
        penalty = max(penalty, bool_penalty)

    marker_penalty = _coerce_finite_nonneg(explicit)
    if marker_penalty is not None:
        penalty = max(penalty, marker_penalty)

    direct_scalar = _coerce_finite_nonneg(direct_value)
    if direct_scalar is not None:
        penalty = max(penalty, direct_scalar)

    if penalty <= 0.0:
        return None

    return {"future_reference": penalty}


def constraint_no_negative_cash(
    program: Program,
    payload: Mapping[str, Any],
) -> Mapping[str, float] | None:
    """Detect negative-cash violations from emitted traces."""
    del program
    traces = payload.get("traces")
    if not isinstance(traces, Mapping):
        return None

    values = _extract_sequence_floats(traces.get("cash_trace"))
    if not values:
        return None

    negative_values = [value for value in values if value < 0.0]
    if not negative_values:
        return None
    return {"negative_cash": abs(min(negative_values))}


def constraint_max_leverage(
    program: Program,
    payload: Mapping[str, Any],
    *,
    limit: float = 1.0,
) -> Mapping[str, float] | None:
    """Detect max-leverage violations in payload or traces."""
    del program
    direct = payload.get("max_leverage")

    leverage = _coerce_finite_nonneg(direct)
    if leverage is None and isinstance(payload.get("traces"), Mapping):
        traces = payload["traces"]
        position_trace = _extract_sequence_floats(
            traces.get("position_trace"),
            absolute=True,
        )
        equity_trace = _extract_sequence_floats(
            traces.get("equity_curve"),
            absolute=True,
        )

        if position_trace and equity_trace:
            max_leverage = 0.0
            for position, equity in zip(position_trace, equity_trace, strict=False):
                if equity <= 0.0:
                    continue
                ratio = position / equity
                if ratio > max_leverage:
                    max_leverage = ratio
            leverage = max_leverage if max_leverage > 0.0 else None

    if leverage is None:
        return None

    if leverage <= limit:
        return None
    return {"max_leverage": leverage - limit}


@dataclass(frozen=True)
class ConstraintPolicy:
    """Collection of strategy constraints used during evaluation."""

    checks: tuple[ConstraintCheck, ...] = ()
    enable_default_checks: bool = False

    def __post_init__(self) -> None:
        if self.enable_default_checks and not self.checks:
            object.__setattr__(
                self,
                "checks",
                (
                    constraint_no_future_reference,
                    constraint_no_negative_cash,
                    constraint_max_leverage,
                ),
            )

    def evaluate(
        self,
        program: Program,
        payload: Mapping[str, Any],
    ) -> dict[str, float]:
        """Aggregate all check violations for one strategy execution slice."""
        violations: dict[str, float] = {}

        for check in self.checks:
            raw = check(program, payload)
            if raw is None:
                continue
            for key, value in raw.items():
                penalty = _coerce_finite_nonneg(value)
                if penalty is None:
                    continue
                violations[key] = max(violations.get(key, 0.0), penalty)

        return violations
