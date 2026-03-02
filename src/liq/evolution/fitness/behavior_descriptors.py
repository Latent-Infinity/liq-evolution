"""Behavior descriptor extraction helpers for strategy QD workflows."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from liq.evolution.fitness.evaluation_schema import (
    BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION,
    BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
    BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,
    BEHAVIOR_DESCRIPTOR_NET_EXPOSURE,
    BEHAVIOR_DESCRIPTOR_STABILITY,
    BEHAVIOR_DESCRIPTOR_TAIL_RISK,
    BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY,
    BEHAVIOR_DESCRIPTOR_TURNOVER,
)
from liq.sim.fx_eval import (
    capacity_proxy,
    cvar_from_pnl,
    tail_stability,
    turnover_from_positions,
)

logger = logging.getLogger(__name__)

BEHAVIOR_DESCRIPTOR_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class _DescriptorSpec:
    raw_min: float
    raw_max: float
    normalizer: str


_DESCRIPTOR_SPECS: dict[str, _DescriptorSpec] = {
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY: _DescriptorSpec(
        raw_min=0.0,
        raw_max=512.0,
        normalizer="ratio_to_max",
    ),
    BEHAVIOR_DESCRIPTOR_TURNOVER: _DescriptorSpec(
        raw_min=0.0,
        raw_max=5.0,
        normalizer="ratio_to_max",
    ),
    BEHAVIOR_DESCRIPTOR_NET_EXPOSURE: _DescriptorSpec(
        raw_min=-1.0,
        raw_max=1.0,
        normalizer="sign_aware_halfshift",
    ),
    BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE: _DescriptorSpec(
        raw_min=0.0,
        raw_max=10.0,
        normalizer="ratio_to_max",
    ),
    BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE: _DescriptorSpec(
        raw_min=0.0,
        raw_max=1.0,
        normalizer="identity",
    ),
    BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY: _DescriptorSpec(
        raw_min=0.0,
        raw_max=1.0,
        normalizer="ratio_to_max",
    ),
    BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION: _DescriptorSpec(
        raw_min=-1.0,
        raw_max=1.0,
        normalizer="sign_aware_halfshift",
    ),
    BEHAVIOR_DESCRIPTOR_TAIL_RISK: _DescriptorSpec(
        raw_min=0.0,
        raw_max=10.0,
        normalizer="ratio_to_max",
    ),
    BEHAVIOR_DESCRIPTOR_STABILITY: _DescriptorSpec(
        raw_min=0.0,
        raw_max=5.0,
        normalizer="ratio_to_max",
    ),
}


@dataclass(frozen=True)
class BehaviorDescriptorProfile:
    """Container for raw and normalized behavior descriptors."""

    raw: dict[str, float]
    normalized: dict[str, float]
    version: str = BEHAVIOR_DESCRIPTOR_SCHEMA_VERSION


def _to_float_list(values: object) -> list[float]:
    """Parse nested trace payloads into finite floats."""
    source: Any = values
    if isinstance(values, Mapping):
        mapped_values = cast(Mapping[str, Any], values)
        source = mapped_values.get("values")
        if source is None:
            source = values
    if source is None:
        return []

    if isinstance(source, (str, bytes)) or not isinstance(source, Iterable):
        return []

    values_list: list[float] = []
    for item in cast(Iterable[Any], source):
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed, float) and not isinstance(parsed, int):
            continue
        if parsed == float("inf") or parsed == float("-inf"):
            continue
        values_list.append(float(parsed))
    return values_list


def _clamp(value: float, min_value: float, max_value: float) -> float:
    clamped = max(min_value, min(max_value, value))
    if clamped != value:
        logger.warning("descriptor value clamped from %s to %s", value, clamped)
    return clamped


def _normalize(name: str, raw_value: float) -> float:
    spec = _DESCRIPTOR_SPECS[name]
    if spec.normalizer == "ratio_to_max":
        normalized = raw_value / spec.raw_max if spec.raw_max else 0.0
    elif spec.normalizer == "identity":
        normalized = raw_value
    elif spec.normalizer == "sign_aware_halfshift":
        normalized = (raw_value + 1.0) / 2.0
    else:
        normalized = raw_value
    return _clamp(normalized, 0.0, 1.0)


def _max_drawdown(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        if value > peak:
            peak = value
        if peak == 0.0:
            continue
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max(0.0, min(1.0, max_drawdown))


def _safe_returns(values: Sequence[float]) -> list[float]:
    if len(values) < 2:
        return []
    result: list[float] = []
    for prev, current in zip(values, values[1:], strict=False):
        if prev == 0.0:
            result.append(0.0)
        else:
            result.append((current / prev) - 1.0)
    return result


def _correlation(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n = min(len(a), len(b))
    a_values = list(a[:n])
    b_values = list(b[:n])
    a_mean = sum(a_values) / n
    b_mean = sum(b_values) / n
    a_var = sum((value - a_mean) ** 2 for value in a_values) / n
    b_var = sum((value - b_mean) ** 2 for value in b_values) / n
    if a_var == 0.0 or b_var == 0.0:
        return 0.0
    covariance = (
        sum((a_values[idx] - a_mean) * (b_values[idx] - b_mean) for idx in range(n)) / n
    )
    return covariance / (a_var**0.5 * b_var**0.5)


def _holding_period_proxy(position: Sequence[float]) -> float:
    if not position:
        return 0.0
    run_lengths: list[int] = []
    run_len = 0
    in_run = False
    for value in position:
        active = abs(float(value)) > 1e-12
        if active:
            run_len += 1
            in_run = True
        elif in_run:
            run_lengths.append(run_len)
            run_len = 0
            in_run = False
    if in_run:
        run_lengths.append(run_len)
    if not run_lengths:
        return 0.0
    return sum(run_lengths) / len(run_lengths)


def _trade_frequency(position: Sequence[float]) -> float:
    if len(position) < 2:
        return 0.0
    changes = 0
    for current, next_value in zip(position, position[1:], strict=False):
        if abs(next_value - current) > 1e-12:
            changes += 1
    return changes / max(len(position) - 1, 1)


def _net_exposure(position: Sequence[float]) -> float:
    if not position:
        return 0.0
    gross = sum(abs(value) for value in position)
    if gross == 0.0:
        return 0.0
    return sum(position) / gross


def _max_leverage(position: Sequence[float], equity: Sequence[float]) -> float:
    if not position:
        return 0.0
    if not equity:
        return sum(abs(value) for value in position) / len(position)
    return capacity_proxy(position, equity)


def _benchmark_correlation(
    position: Sequence[float],
    pnl: Sequence[float],
    benchmark_returns: Sequence[float],
) -> float:
    base = _safe_returns(position if position else pnl)
    if not base or not benchmark_returns:
        return 0.0
    return _correlation(base, benchmark_returns)


def _tail_risk(pnl: Sequence[float]) -> float:
    if not pnl:
        return 0.0
    return cvar_from_pnl(pnl)


def _stability(pnl: Sequence[float]) -> float:
    if not pnl:
        return 0.0
    return tail_stability(pnl)


def _drawdown_profile(equity: Sequence[float]) -> float:
    if not equity:
        return 0.0
    if len(equity) < 2:
        return 0.0
    return float(_max_drawdown(equity))


def extract_behavior_descriptors(
    trace_payload: Mapping[str, object] | None,
    *,
    benchmark_returns: Sequence[float] | None = None,
    descriptor_names: Sequence[str] | None = None,
) -> BehaviorDescriptorProfile:
    """Extract deterministic raw and normalized behavior descriptors."""
    requested = (
        descriptor_names if descriptor_names is not None else tuple(_DESCRIPTOR_SPECS)
    )
    payload = trace_payload or {}
    position = _to_float_list(payload.get("position_trace"))
    equity = _to_float_list(payload.get("equity_curve"))
    pnl = _to_float_list(payload.get("pnl_trace"))
    returns = _safe_returns(pnl) if pnl else []
    bench = (
        _to_float_list(benchmark_returns)
        if benchmark_returns is not None
        else _to_float_list(payload.get("benchmark_returns"))
    )

    raw: dict[str, float] = {}
    normalized: dict[str, float] = {}

    calculators = {
        BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY: lambda: _holding_period_proxy(
            position
        ),
        BEHAVIOR_DESCRIPTOR_TURNOVER: lambda: turnover_from_positions(position),
        BEHAVIOR_DESCRIPTOR_NET_EXPOSURE: lambda: _net_exposure(position),
        BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE: lambda: _max_leverage(position, equity),
        BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE: lambda: _drawdown_profile(equity),
        BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY: lambda: _trade_frequency(position),
        BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION: lambda: _benchmark_correlation(
            position,
            returns,
            bench,
        ),
        BEHAVIOR_DESCRIPTOR_TAIL_RISK: lambda: _tail_risk(returns),
        BEHAVIOR_DESCRIPTOR_STABILITY: lambda: _stability(returns),
    }

    for name in requested:
        if name not in _DESCRIPTOR_SPECS:
            raise ValueError(f"unsupported behavior descriptor {name!r}")
        raw_value = float(calculators[name]())
        spec = _DESCRIPTOR_SPECS[name]
        clamped = _clamp(raw_value, spec.raw_min, spec.raw_max)
        raw[name] = clamped
        normalized[name] = _normalize(name, clamped)

    return BehaviorDescriptorProfile(raw=raw, normalized=normalized)


def normalize_behavior_descriptor_value(name: str, raw_value: float) -> float:
    """Normalize a raw descriptor value into [0, 1]."""
    if name not in _DESCRIPTOR_SPECS:
        raise ValueError(f"unsupported behavior descriptor {name!r}")
    clamped = _clamp(float(raw_value), *_get_spec(name))
    return _normalize(name, clamped)


def _get_spec(name: str) -> tuple[float, float]:
    spec = _DESCRIPTOR_SPECS[name]
    return spec.raw_min, spec.raw_max
