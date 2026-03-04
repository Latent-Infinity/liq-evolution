"""Strategy-level constraint checks and policy composition."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from liq.gp.program.ast import Program

ConstraintCheck = Callable[[Program, Mapping[str, Any]], Mapping[str, float] | None]

_ROBUSTNESS_ROLLOUT_SCALES: dict[str, tuple[float, float] | None] = {
    "disabled": None,
    "canary": (0.85, 1.15),
    "standard": (1.0, 1.0),
    "strict": (1.10, 0.90),
}


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


def _resolve_metric_value(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        for key in keys:
            value = _coerce_finite_float(metrics.get(key))
            if value is not None:
                return value

    for key in keys:
        value = _coerce_finite_float(payload.get(key))
        if value is not None:
            return value
    return None


def _robustness_gate_violations(
    payload: Mapping[str, Any],
    *,
    robustness_rollout: str,
    regime_coverage_floor: float | None,
    turnover_cap: float | None,
    drawdown_cap: float | None,
) -> dict[str, float]:
    profile = _ROBUSTNESS_ROLLOUT_SCALES[robustness_rollout]
    if profile is None:
        return {}

    coverage_scale, cap_scale = profile
    violations: dict[str, float] = {}

    if regime_coverage_floor is not None:
        effective_floor = max(0.0, min(1.0, regime_coverage_floor * coverage_scale))
        coverage = _resolve_metric_value(
            payload,
            ("regime_coverage", "regime_occupancy", "regime_confidence"),
        )
        if coverage is None:
            violations["robustness:regime_coverage_missing"] = 1.0
        elif coverage < effective_floor:
            violations["robustness:regime_coverage_below_floor"] = (
                effective_floor - coverage
            )

    if turnover_cap is not None:
        effective_cap = max(0.0, turnover_cap * cap_scale)
        turnover = _resolve_metric_value(
            payload,
            ("turnover", "avg_turnover"),
        )
        if turnover is None:
            violations["robustness:turnover_missing"] = 1.0
        elif turnover > effective_cap:
            violations["robustness:turnover_above_cap"] = turnover - effective_cap

    if drawdown_cap is not None:
        effective_cap = max(0.0, drawdown_cap * cap_scale)
        drawdown = _resolve_metric_value(
            payload,
            ("max_drawdown", "drawdown", "max_dd"),
        )
        if drawdown is None:
            violations["robustness:drawdown_missing"] = 1.0
        else:
            normalized_drawdown = abs(drawdown)
            if normalized_drawdown > effective_cap:
                violations["robustness:max_drawdown_above_cap"] = (
                    normalized_drawdown - effective_cap
                )

    return violations


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
    robustness_rollout: str = "disabled"
    regime_coverage_floor: float | None = None
    turnover_cap: float | None = None
    drawdown_cap: float | None = None

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
        if self.robustness_rollout not in _ROBUSTNESS_ROLLOUT_SCALES:
            raise ValueError(
                "robustness_rollout must be one of "
                + ", ".join(sorted(_ROBUSTNESS_ROLLOUT_SCALES))
            )
        for name, value in (
            ("regime_coverage_floor", self.regime_coverage_floor),
            ("turnover_cap", self.turnover_cap),
            ("drawdown_cap", self.drawdown_cap),
        ):
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric when provided")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name == "regime_coverage_floor" and value > 1.0:
                raise ValueError("regime_coverage_floor must be in [0, 1]")

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

        for key, value in _robustness_gate_violations(
            payload,
            robustness_rollout=self.robustness_rollout,
            regime_coverage_floor=self.regime_coverage_floor,
            turnover_cap=self.turnover_cap,
            drawdown_cap=self.drawdown_cap,
        ).items():
            penalty = _coerce_finite_nonneg(value)
            if penalty is None:
                continue
            violations[key] = max(violations.get(key, 0.0), penalty)

        return violations
