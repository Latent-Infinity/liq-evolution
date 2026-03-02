"""Shared fitness metadata contract and helpers for strategy evaluation.

The trading GP stack relies on a small vocabulary for metadata keys so that
selection operators, quality-diversity archives, and verification tooling can
consume the same payload shape across evaluators.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

ObjectiveDirection = Literal["maximize", "minimize"]


# ---------------------------------------------------------------------------
# Metadata key constants
# ---------------------------------------------------------------------------

METADATA_KEY_PER_SPLIT_METRICS = "per_split_metrics"
METADATA_KEY_RAW_OBJECTIVES = "raw_objectives"
METADATA_KEY_BEHAVIOR_DESCRIPTORS = "behavior_descriptors"
METADATA_KEY_CONSTRAINT_VIOLATIONS = "constraint_violations"
METADATA_KEY_SLICE_SCORES = "slice_scores"


# ---------------------------------------------------------------------------
# Slice taxonomy constants
# ---------------------------------------------------------------------------

SLICE_TYPE_TIME_WINDOW = "time_window"
SLICE_TYPE_INSTRUMENT = "instrument"
SLICE_TYPE_EVENT = "event"
SLICE_TYPE_LIQUIDITY = "liquidity"


# ---------------------------------------------------------------------------
# Behavior descriptor key constants
# ---------------------------------------------------------------------------

BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY = "holding_period_proxy"
BEHAVIOR_DESCRIPTOR_TURNOVER = "turnover"
BEHAVIOR_DESCRIPTOR_NET_EXPOSURE = "net_exposure"
BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE = "max_leverage"
BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE = "drawdown_profile"
BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY = "trade_frequency"
BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION = "benchmark_correlation"
BEHAVIOR_DESCRIPTOR_TAIL_RISK = "tail_risk"
BEHAVIOR_DESCRIPTOR_STABILITY = "stability"

SUPPORTED_BEHAVIOR_DESCRIPTORS = frozenset(
    {
        BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
        BEHAVIOR_DESCRIPTOR_TURNOVER,
        BEHAVIOR_DESCRIPTOR_NET_EXPOSURE,
        BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,
        BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,
        BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY,
        BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION,
        BEHAVIOR_DESCRIPTOR_TAIL_RISK,
        BEHAVIOR_DESCRIPTOR_STABILITY,
    }
)


# ---------------------------------------------------------------------------
# Metric convention constants
# ---------------------------------------------------------------------------

TWEAKABLE_METRIC_KEY_TAIL_RISK = "tail_risk"
TWEAKABLE_METRIC_KEY_STABILITY = "stability"

_LEGACY_WINDOW_SLICE_ID_PATTERN = re.compile(r"^window_\d+:start=\d+:end=\d+$")


@dataclass(frozen=True)
class SchemaValidationError:
    """Structured validation error for one metadata field."""

    field: str
    message: str


def _add_error(
    target: list[SchemaValidationError],
    field: str,
    message: str,
) -> None:
    """Record a structured validation error."""
    target.append(SchemaValidationError(field=field, message=message))


def _validate_dict_str_to_float(
    source: object,
    *,
    field: str,
    issues: list[SchemaValidationError],
) -> dict[str, float] | None:
    """Validate a ``dict[str, float]`` payload and return a typed copy."""
    if not isinstance(source, Mapping):
        _add_error(issues, field, "must be a dict")
        return None

    result: dict[str, float] = {}
    for key, raw_value in source.items():
        if not isinstance(key, str):
            _add_error(issues, field, "keys must be strings")
            continue
        if not isinstance(raw_value, (int, float)):
            _add_error(issues, field, f"value for {key!r} must be a number")
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            _add_error(
                issues,
                field,
                f"value for {key!r} must be finite (got {value!r})",
            )
            continue
        result[key] = value

    return result


def _validate_nested_split_metrics(
    split_metrics: object,
    *,
    issues: list[SchemaValidationError],
) -> None:
    """Validate ``per_split_metrics`` shape: ``dict[str, dict[str, float]]``."""
    if not isinstance(split_metrics, Mapping):
        _add_error(
            issues,
            METADATA_KEY_PER_SPLIT_METRICS,
            "must be a dict[str, dict[str, float]]",
        )
        return

    for split_id, metrics in split_metrics.items():
        if not isinstance(split_id, str):
            _add_error(
                issues,
                METADATA_KEY_PER_SPLIT_METRICS,
                "split ids must be strings",
            )
            continue
        _validate_dict_str_to_float(
            metrics,
            field=f"{METADATA_KEY_PER_SPLIT_METRICS}[{split_id!r}]",
            issues=issues,
        )


def _validate_behavior_descriptors(
    behavior_descriptors: object,
    *,
    issues: list[SchemaValidationError],
) -> None:
    """Validate behavior descriptor payload and key set."""
    values = _validate_dict_str_to_float(
        behavior_descriptors,
        field=METADATA_KEY_BEHAVIOR_DESCRIPTORS,
        issues=issues,
    )
    if values is None:
        return

    for key in values:
        if key not in SUPPORTED_BEHAVIOR_DESCRIPTORS:
            _add_error(
                issues,
                METADATA_KEY_BEHAVIOR_DESCRIPTORS,
                f"unsupported behavior descriptor key {key!r}",
            )


def _validate_slice_scores(
    slice_scores: object,
    *,
    require: bool,
    issues: list[SchemaValidationError],
) -> None:
    """Validate case-level loss scores used by lexicase selection."""
    if slice_scores is None:
        if require:
            _add_error(
                issues,
                METADATA_KEY_SLICE_SCORES,
                "is required when selection uses lexicase",
            )
        return

    scores = _validate_dict_str_to_float(
        slice_scores,
        field=METADATA_KEY_SLICE_SCORES,
        issues=issues,
    )
    if scores is None:
        return

    for slice_id in scores:
        if not validate_slice_id(slice_id):
            _add_error(
                issues,
                METADATA_KEY_SLICE_SCORES,
                f"unsupported slice id {slice_id!r}",
            )


def _validate_constraint_violations(
    constraint_violations: object,
    *,
    issues: list[SchemaValidationError],
) -> None:
    """Validate constraint-violation payload as a mapping."""
    values = _validate_dict_str_to_float(
        constraint_violations,
        field=METADATA_KEY_CONSTRAINT_VIOLATIONS,
        issues=issues,
    )
    if values is None:
        return

    if any(value < 0.0 for value in values.values()):
        _add_error(
            issues,
            METADATA_KEY_CONSTRAINT_VIOLATIONS,
            "violation values should be non-negative",
        )


def validate_objective_vector(
    raw_objectives: object,
    expected_count: int,
    *,
    issues: list[SchemaValidationError],
) -> tuple[float, ...] | None:
    """Validate objective-vector shape and numeric finiteness."""
    if not isinstance(raw_objectives, (tuple, list)):
        _add_error(
            issues,
            METADATA_KEY_RAW_OBJECTIVES,
            "must be a sequence of floats",
        )
        return None

    if len(raw_objectives) != expected_count:
        _add_error(
            issues,
            METADATA_KEY_RAW_OBJECTIVES,
            f"length {len(raw_objectives)} must match expected objective count {expected_count}",
        )

    objectives = []
    for i, value in enumerate(raw_objectives):
        if not isinstance(value, (int, float)):
            _add_error(
                issues,
                f"{METADATA_KEY_RAW_OBJECTIVES}[{i}]",
                "must be numeric",
            )
            continue
        if not math.isfinite(float(value)):
            _add_error(
                issues,
                f"{METADATA_KEY_RAW_OBJECTIVES}[{i}]",
                f"must be finite (got {value!r})",
            )
            continue
        objectives.append(float(value))

    return tuple(objectives) if objectives else None


def _validate_objective_directions(
    objective_directions: Sequence[ObjectiveDirection] | None,
    expected_count: int,
    *,
    issues: list[SchemaValidationError],
) -> None:
    """Validate optional objective-direction payload."""
    if objective_directions is None:
        return

    if len(objective_directions) != expected_count:
        _add_error(
            issues,
            "objective_directions",
            (
                f"length {len(objective_directions)} must match expected objective count "
                f"{expected_count}"
            ),
        )

    for idx, direction in enumerate(objective_directions):
        if direction not in {"maximize", "minimize"}:
            _add_error(
                issues,
                f"objective_directions[{idx}]",
                "must be 'maximize' or 'minimize'",
            )


def validate_evaluation_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_objective_count: int,
    require_slice_scores: bool = False,
    objective_directions: Sequence[ObjectiveDirection] | None = None,
) -> list[SchemaValidationError]:
    """Validate a fitness metadata payload against phase-0 contract rules.

    Args:
        metadata: Metadata dict from ``FitnessResult``.
        expected_objective_count: Expected number of evaluator objectives.
        require_slice_scores: Enforce presence of ``slice_scores``.
        objective_directions: Optional direction contract to validate.

    Returns:
        A list of structured validation errors. Empty list means valid payload.
    """
    issues: list[SchemaValidationError] = []

    if not isinstance(metadata, Mapping):
        _add_error(issues, "metadata", "must be a dict")
        return issues

    raw_objectives = metadata.get(METADATA_KEY_RAW_OBJECTIVES)
    if raw_objectives is None:
        _add_error(
            issues,
            METADATA_KEY_RAW_OBJECTIVES,
            "missing required key",
        )
    else:
        validate_objective_vector(
            raw_objectives,
            expected_count=expected_objective_count,
            issues=issues,
        )

    if metadata.get(METADATA_KEY_PER_SPLIT_METRICS) is None:
        _add_error(
            issues,
            METADATA_KEY_PER_SPLIT_METRICS,
            "missing required key",
        )
    else:
        _validate_nested_split_metrics(
            metadata.get(METADATA_KEY_PER_SPLIT_METRICS),
            issues=issues,
        )

    if metadata.get(METADATA_KEY_BEHAVIOR_DESCRIPTORS) is None:
        _add_error(
            issues,
            METADATA_KEY_BEHAVIOR_DESCRIPTORS,
            "missing required key",
        )
    else:
        _validate_behavior_descriptors(
            metadata.get(METADATA_KEY_BEHAVIOR_DESCRIPTORS),
            issues=issues,
        )

    if metadata.get(METADATA_KEY_CONSTRAINT_VIOLATIONS) is None:
        _add_error(
            issues,
            METADATA_KEY_CONSTRAINT_VIOLATIONS,
            "missing required key",
        )
    else:
        _validate_constraint_violations(
            metadata.get(METADATA_KEY_CONSTRAINT_VIOLATIONS),
            issues=issues,
        )

    _validate_slice_scores(
        metadata.get(METADATA_KEY_SLICE_SCORES),
        require=require_slice_scores,
        issues=issues,
    )

    _validate_objective_directions(
        objective_directions,
        expected_count=expected_objective_count,
        issues=issues,
    )

    return issues


def to_loss_form(
    value: float,
    direction: ObjectiveDirection,
    *,
    nan_penalty: float = 1e6,
    max_finite: float = 1e12,
) -> float:
    """Return a finite loss-form scalar from a raw objective value.

    Args:
        value: Raw objective value.
        direction: Objective optimization direction.
        nan_penalty: Loss assigned to non-finite raw inputs.
        max_finite: Absolute clamp bound for the final loss.

    Returns:
        A finite float where lower values are better.

    Raises:
        ValueError: If *direction* is invalid.
    """
    if direction not in {"maximize", "minimize"}:
        msg = f"direction must be 'maximize' or 'minimize', got {direction!r}"
        raise ValueError(msg)

    raw = float(value)
    if not math.isfinite(raw):
        penalty = abs(float(nan_penalty))
        loss = penalty
    else:
        loss = -raw if direction == "maximize" else raw
    limit = abs(float(max_finite))
    if loss > limit:
        return limit
    if loss < -limit:
        return -limit
    return loss


def validate_slice_id(slice_id: str) -> bool:
    """Quick helper for slice-ID prefix validation.

    Supports current canonical prefixes and legacy `window_` IDs for
    back-compatibility.
    """
    if slice_id.startswith(SLICE_TYPE_TIME_WINDOW + ":"):
        return True
    if slice_id.startswith(SLICE_TYPE_INSTRUMENT + ":"):
        return True
    if slice_id.startswith(SLICE_TYPE_EVENT + ":"):
        return True
    if slice_id.startswith(SLICE_TYPE_LIQUIDITY + ":"):
        return True
    return _LEGACY_WINDOW_SLICE_ID_PATTERN.match(slice_id) is not None
