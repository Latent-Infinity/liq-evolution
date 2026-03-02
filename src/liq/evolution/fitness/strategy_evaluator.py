"""Trade-strategy evaluator with walk-forward split aggregation and contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from liq.datasets.walk_forward import WalkForwardSplit
from liq.evolution.errors import FitnessEvaluationError
from liq.evolution.fitness.behavior_descriptors import (
    BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,
    BEHAVIOR_DESCRIPTOR_SCHEMA_VERSION,
    BehaviorDescriptorProfile,
    extract_behavior_descriptors,
    normalize_behavior_descriptor_value,
)
from liq.evolution.fitness.eval_cache import (
    FitnessEvaluationCache,
    compute_fingerprint,
    compute_program_hash,
)
from liq.evolution.fitness.evaluation_schema import (
    BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION,
    BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
    BEHAVIOR_DESCRIPTOR_NET_EXPOSURE,
    BEHAVIOR_DESCRIPTOR_STABILITY,
    BEHAVIOR_DESCRIPTOR_TAIL_RISK,
    BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY,
    BEHAVIOR_DESCRIPTOR_TURNOVER,
    METADATA_KEY_BEHAVIOR_DESCRIPTORS,
    METADATA_KEY_CONSTRAINT_VIOLATIONS,
    METADATA_KEY_PER_SPLIT_METRICS,
    METADATA_KEY_RAW_OBJECTIVES,
    METADATA_KEY_SLICE_SCORES,
    SLICE_TYPE_EVENT,
    SLICE_TYPE_INSTRUMENT,
    SLICE_TYPE_LIQUIDITY,
    SLICE_TYPE_TIME_WINDOW,
    SUPPORTED_BEHAVIOR_DESCRIPTORS,
    ObjectiveDirection,
    to_loss_form,
    validate_evaluation_metadata,
)
from liq.evolution.validation import ConstraintPolicy
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate as gp_evaluate
from liq.gp.types import FitnessResult
from liq.sim.fx_eval import (
    capacity_proxy,
    cvar_from_pnl,
    max_exposure,
    summarize_fx_performance,
    tail_stability,
    turnover_from_positions,
)

DEFAULT_OBJECTIVES = (
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "tail_risk",
    "turnover",
    "net_exposure",
    "capacity_proxy",
    "stability",
)

DEFAULT_OBJECTIVE_DIRECTIONS: tuple[ObjectiveDirection, ...] = (
    "maximize",
    "maximize",
    "maximize",
    "minimize",
    "minimize",
    "minimize",
    "minimize",
    "minimize",
    "minimize",
)

DEFAULT_BEHAVIOR_DESCRIPTOR_NAMES = (
    BEHAVIOR_DESCRIPTOR_HOLDING_PERIOD_PROXY,
    BEHAVIOR_DESCRIPTOR_TURNOVER,
    BEHAVIOR_DESCRIPTOR_NET_EXPOSURE,
    BEHAVIOR_DESCRIPTOR_DRAWDOWN_PROFILE,
    BEHAVIOR_DESCRIPTOR_TRADE_FREQUENCY,
    BEHAVIOR_DESCRIPTOR_BENCHMARK_CORRELATION,
    BEHAVIOR_DESCRIPTOR_TAIL_RISK,
    BEHAVIOR_DESCRIPTOR_STABILITY,
    BEHAVIOR_DESCRIPTOR_MAX_LEVERAGE,
)

_SYMBOLIC_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "cagr": ("cagr", "cagr_ratio", "total_return"),
    "sharpe": ("sharpe", "sharpe_ratio"),
    "sortino": ("sortino", "sortino_ratio"),
    "max_drawdown": ("max_drawdown", "drawdown", "max_dd"),
    "turnover": ("turnover", "avg_turnover"),
    "net_exposure": ("net_exposure", "net_exposure_abs", "exposure"),
    "capacity_proxy": ("capacity_proxy", "capacity"),
    "tail_risk": ("tail_risk", "cvar", "tail_var"),
    "stability": ("stability",),
}


@dataclass(frozen=True)
class StrategyEvaluatorConfig:
    objective_directions: tuple[ObjectiveDirection, ...]
    split_weights: tuple[float, float, float]
    nan_penalty: float
    max_finite: float


class StrategyEvaluator:
    """Evaluate strategies with walk-forward train/validate/test split aggregation."""

    evaluator_version = "1.0"

    def __init__(
        self,
        backtest_runner: Callable[
            [Any, Mapping[str, Any], WalkForwardSplit], Mapping[str, Any]
        ],
        splits: Sequence[WalkForwardSplit],
        *,
        objectives: Sequence[str] = DEFAULT_OBJECTIVES,
        objective_directions: Sequence[ObjectiveDirection] | None = None,
        split_weights: Mapping[str, float] | None = None,
        include_test: bool = False,
        behavior_descriptor_names: Sequence[str] | None = None,
        constraint_policy: ConstraintPolicy | None = None,
        nan_penalty: float = 1e6,
        max_finite: float = 1e12,
        cache: FitnessEvaluationCache | None = None,
    ) -> None:
        self._backtest_runner = backtest_runner
        self._splits = list(splits)
        self._objectives = tuple(objectives)
        self._include_test = include_test
        self._behavior_descriptor_names = (
            tuple(behavior_descriptor_names)
            if behavior_descriptor_names is not None
            else DEFAULT_BEHAVIOR_DESCRIPTOR_NAMES
        )
        self._constraint_policy = constraint_policy or ConstraintPolicy()
        if not self._behavior_descriptor_names:
            raise ValueError("behavior_descriptor_names must be non-empty")
        unsupported = {
            name
            for name in self._behavior_descriptor_names
            if name not in SUPPORTED_BEHAVIOR_DESCRIPTORS
        }
        if unsupported:
            raise ValueError(
                "unsupported behavior descriptors: " + ", ".join(sorted(unsupported))
            )

        if not self._splits:
            raise ValueError("splits must be non-empty")
        if not self._objectives:
            raise ValueError("at least one objective is required")
        if objective_directions is None:
            objective_directions = tuple(
                DEFAULT_OBJECTIVE_DIRECTIONS[: len(self._objectives)]
            )
            if len(objective_directions) < len(self._objectives):
                raise ValueError("missing default direction for custom objective list")
        else:
            objective_directions = tuple(objective_directions)
            if len(objective_directions) != len(self._objectives):
                raise ValueError("objective_directions must match objectives length")
            if any(
                direction not in {"maximize", "minimize"}
                for direction in objective_directions
            ):
                raise ValueError(
                    "objective_directions must be 'maximize' or 'minimize'"
                )

        configured_split_weights = split_weights or {}
        unknown_split_weight_keys = set(configured_split_weights) - {
            "train",
            "validate",
            "test",
        }
        if unknown_split_weight_keys:
            raise ValueError(
                "split_weights contains unsupported keys: "
                + ", ".join(sorted(unknown_split_weight_keys))
            )

        split_weight_values = {
            phase: float(configured_split_weights.get(phase, default))
            for phase, default in {
                "train": 1.0,
                "validate": 1.0,
                "test": 0.0,
            }.items()
        }
        for phase, weight in split_weight_values.items():
            if not math.isfinite(weight):
                raise ValueError(
                    f"split_weights[{phase}] must be finite, got {weight!r}"
                )
            if weight < 0.0:
                raise ValueError(
                    f"split_weights[{phase}] must be non-negative, got {weight!r}"
                )

        self._config = StrategyEvaluatorConfig(
            objective_directions=objective_directions,
            split_weights=(
                split_weight_values["train"],
                split_weight_values["validate"],
                split_weight_values["test"],
            ),
            nan_penalty=float(nan_penalty),
            max_finite=float(max_finite),
        )
        self._cache = cache
        self._cache_fingerprint = compute_fingerprint(
            self.evaluator_version,
            {
                "objectives": list(self._objectives),
                "objective_directions": list(self._config.objective_directions),
                "split_weights": list(self._config.split_weights),
                "include_test": self._include_test,
                "behavior_descriptor_names": list(self._behavior_descriptor_names),
                "nan_penalty": self._config.nan_penalty,
                "max_finite": self._config.max_finite,
                "descriptor_schema_version": BEHAVIOR_DESCRIPTOR_SCHEMA_VERSION,
            },
        )

    def evaluate(
        self,
        programs: list[Program],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        """Evaluate all programs and return ordered FitnessResult payloads."""
        if "labels" not in context:
            raise FitnessEvaluationError("Context must include 'labels'")

        results: list[FitnessResult] = []
        for program in programs:
            results.append(self._evaluate_single(program, context))
        return results

    def evaluate_fitness(
        self,
        programs: list[Program],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        """Backward-compatible fitness-evaluation alias."""
        return self.evaluate(programs, context)

    def _evaluate_single(
        self, program: Program, context: Mapping[str, Any]
    ) -> FitnessResult:
        strategy = _ProgramStrategy(program)
        split_metrics: dict[str, dict[str, float]] = {}
        slice_scores: dict[str, float] = {}
        raw_objective_sums = dict.fromkeys(self._objectives, 0.0)
        raw_objective_weights = dict.fromkeys(self._objectives, 0.0)
        constraint_violations: dict[str, float] = {}
        descriptor_profiles: list[BehaviorDescriptorProfile] = []
        program_hash = compute_program_hash(program)
        cache_hits = 0
        cache_misses = 0

        for split_index, split in enumerate(self._splits):
            try:
                split_id_fallback = split.slice_id
                split_payload = None
                if self._cache is not None:
                    cache_key = self._cache.make_key(
                        program_hash,
                        split_id_fallback,
                        self._cache_fingerprint,
                    )
                    if cache_key is not None:
                        cached_payload = self._cache.get(
                            program_hash,
                            split_id_fallback,
                            self._cache_fingerprint,
                        )
                        if cached_payload is not None:
                            cache_hits += 1
                            split_payload = cached_payload
                        else:
                            cache_misses += 1

                if split_payload is None:
                    split_payload = self._backtest_runner(strategy, context, split)
            except Exception as exc:
                raise FitnessEvaluationError(
                    f"Backtest runner failed for split {split.slice_id}"
                ) from exc

            if split_payload is None:
                raise FitnessEvaluationError("backtest runner returned no payload")

            split_id = self._resolve_split_id(split_payload, split.slice_id)
            if not split_id:
                split_id = self._fallback_split_id(split, split_index)
            if self._cache is not None and split_id:
                self._cache.put(
                    program_hash,
                    split_id,
                    self._cache_fingerprint,
                    split_payload,
                )
                if split_id_fallback is not None and split_id_fallback != split_id:
                    self._cache.put(
                        program_hash,
                        split_id_fallback,
                        self._cache_fingerprint,
                        split_payload,
                    )

            split_phases = self._normalize_split_payload(split_payload, split_id)
            for phase, phase_payload in split_phases.items():
                if phase == "test" and not self._include_test:
                    continue
                metrics = self._extract_phase_metrics(phase_payload)
                split_key = f"{split_id}:{phase}"
                split_metrics[split_key] = metrics

                weight = self._split_weight(phase)
                for index, objective_name in enumerate(self._objectives):
                    value = metrics[objective_name]
                    raw_objective_sums[objective_name] += value * weight
                    raw_objective_weights[objective_name] += weight
                    if weight > 0.0:
                        case_key = f"{split_key}:{objective_name}"
                        slice_scores[case_key] = to_loss_form(
                            value=value,
                            direction=self._config.objective_directions[index],
                            nan_penalty=self._config.nan_penalty,
                            max_finite=self._config.max_finite,
                        )

                for key, raw_penalty in self._extract_constraint_violations(
                    phase_payload,
                    program,
                    split_key,
                ).items():
                    constraint_violations[key] = max(
                        constraint_violations.get(key, 0.0),
                        raw_penalty,
                    )
                    slice_scores[f"{split_key}:constraint:{key}"] = raw_penalty

                for key, raw_score in self._extract_phase_slice_scores(
                    phase_payload,
                ).items():
                    for case_key in _normalize_slice_score_key(split_key, key):
                        slice_scores[case_key] = _coerce_nonnegative(raw_score)

                traces = self._extract_traces(phase_payload)
                if traces is not None:
                    descriptor_profiles.append(
                        extract_behavior_descriptors(
                            traces,
                            descriptor_names=self._behavior_descriptor_names,
                        )
                    )

        raw_objectives = tuple(
            self._aggregate_objective(
                name=objective_name,
                total=raw_objective_sums[objective_name],
                weight=raw_objective_weights[objective_name],
            )
            for objective_name in self._objectives
        )
        behavior_descriptors = self._aggregate_descriptors(descriptor_profiles)
        _ = validate_evaluation_metadata(
            {
                METADATA_KEY_PER_SPLIT_METRICS: split_metrics,
                METADATA_KEY_RAW_OBJECTIVES: raw_objectives,
                METADATA_KEY_BEHAVIOR_DESCRIPTORS: behavior_descriptors,
                METADATA_KEY_CONSTRAINT_VIOLATIONS: constraint_violations,
                METADATA_KEY_SLICE_SCORES: slice_scores,
            },
            expected_objective_count=len(self._objectives),
            require_slice_scores=True,
            objective_directions=self._config.objective_directions,
        )

        return FitnessResult(
            objectives=raw_objectives,
            metadata={
                METADATA_KEY_PER_SPLIT_METRICS: split_metrics,
                METADATA_KEY_RAW_OBJECTIVES: raw_objectives,
                METADATA_KEY_BEHAVIOR_DESCRIPTORS: behavior_descriptors,
                METADATA_KEY_CONSTRAINT_VIOLATIONS: constraint_violations,
                METADATA_KEY_SLICE_SCORES: slice_scores,
                "objectives": self._objectives,
                "objective_directions": self._config.objective_directions,
                "split_count": len(self._splits),
                "descriptor_schema_version": BEHAVIOR_DESCRIPTOR_SCHEMA_VERSION,
                "cache": {
                    "enabled": self._cache is not None,
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "fingerprint": self._cache_fingerprint,
                    "program_hash": program_hash,
                },
            },
        )

    def _normalize_split_payload(
        self,
        split_payload: Mapping[str, Any],
        split_id: str | None,
    ) -> dict[str, Mapping[str, Any]]:
        if not isinstance(split_payload, Mapping):
            raise FitnessEvaluationError(
                f"backtest runner must return mapping for split {split_id}"
            )

        if all(phase in split_payload for phase in ("train", "validate", "test")):
            return {
                "train": split_payload["train"],
                "validate": split_payload["validate"],
                "test": split_payload["test"],
            }

        return {
            "train": split_payload,
            "validate": split_payload,
            "test": split_payload,
        }

    @staticmethod
    def _resolve_split_id(
        split_payload: Any,
        fallback: str | None,
    ) -> str:
        """Resolve a canonical split id for cache keys.

        Prefer an explicit id coming from runner payload (e.g., propagated from
        FoldResult.slice_id), then fall back to split metadata.
        """
        if isinstance(split_payload, Mapping):
            for key in ("slice_id", "fold_slice_id", "split_id"):
                explicit = split_payload.get(key)
                if isinstance(explicit, str) and explicit:
                    return explicit

        if isinstance(fallback, str) and fallback:
            return fallback

        # Defensive fallback should never be reached under validated split inputs.
        return ""

    @staticmethod
    def _fallback_split_id(
        split: WalkForwardSplit,
        split_index: int,
    ) -> str:
        """Build a stable synthetic id when slice IDs are missing."""

        def _bound(
            window: slice | tuple[object, object],
        ) -> tuple[object | None, object | None]:
            if isinstance(window, tuple):
                return (window[0], window[1])
            return (window.start, window.stop)

        train_bounds = _bound(split.train)
        validate_bounds = _bound(split.validate)
        test_bounds = _bound(split.test)
        return (
            f"time_window:split_{split_index}:"
            f"train={train_bounds[0]}:{train_bounds[1]}|"
            f"validate={validate_bounds[0]}:{validate_bounds[1]}|"
            f"test={test_bounds[0]}:{test_bounds[1]}"
        )

    def _extract_phase_metrics(
        self, phase_payload: Mapping[str, Any]
    ) -> dict[str, float]:
        metrics = _safe_mapping_float_dict(phase_payload.get("metrics"))
        traces = self._extract_traces(phase_payload)
        traces = traces if traces is not None else {}
        equity = _to_float_sequence(traces.get("equity_curve"))
        pnl = _to_float_sequence(traces.get("pnl_trace"))
        position = _to_float_sequence(traces.get("position_trace"))

        # Trace-derived baseline metrics (used when phase metrics are missing).
        summary = summarize_fx_performance(equity) if equity else {}
        derived = {
            "cagr": summary.get("total_return", 0.0),
            "sharpe": summary.get("sharpe", 0.0),
            "sortino": summary.get("sortino", 0.0),
            "max_drawdown": summary.get("max_drawdown", 0.0),
            "turnover": turnover_from_positions(position),
            "net_exposure": max_exposure(position, equity),
            "capacity_proxy": capacity_proxy(position, equity),
            "tail_risk": cvar_from_pnl(pnl) if pnl else 0.0,
            "stability": tail_stability(pnl) if pnl else 0.0,
        }

        result = {}
        for objective_name in self._objectives:
            result[objective_name] = _resolve_metric(
                objective_name=objective_name,
                metrics=metrics,
                derived=derived,
            )
        return result

    def _extract_constraint_violations(
        self,
        phase_payload: Mapping[str, Any],
        program: Program,
        split_key: str,
    ) -> dict[str, float]:
        violations = {}
        raw_violations = _safe_mapping_float_dict(
            phase_payload.get("constraint_violations")
        )
        for key, value in raw_violations.items():
            safe_value = _coerce_nonnegative(float(value))
            if safe_value < 0.0:
                safe_value = 0.0
            violations[key] = safe_value

        policy_violations = self._constraint_policy.evaluate(program, phase_payload)
        for key, value in policy_violations.items():
            safe_value = _coerce_nonnegative(value)
            if safe_value > 0.0:
                violations[f"{split_key}:{key}"] = safe_value

        for key, value in _safe_mapping_float_dict(
            phase_payload.get("adversarial_cases")
        ).items():
            safe_value = _coerce_nonnegative(value)
            if safe_value > 0.0:
                violations[f"{split_key}:adversarial:{key}"] = safe_value

        return violations

    def _extract_phase_slice_scores(
        self,
        phase_payload: Mapping[str, Any],
    ) -> dict[str, float]:
        return _safe_mapping_float_dict(phase_payload.get("slice_scores"))

    def _extract_traces(
        self, phase_payload: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        traces = phase_payload.get("traces")
        if isinstance(traces, Mapping):
            return dict(traces)
        if isinstance(traces, Iterable) and not isinstance(traces, (str, bytes)):
            # Legacy payloads pass full traces at split level.
            return {"position_trace": traces}
        return None

    def _split_weight(self, phase: str) -> float:
        if phase == "train":
            return self._config.split_weights[0]
        if phase == "validate":
            return self._config.split_weights[1]
        return self._config.split_weights[2]

    def _aggregate_objective(self, *, name: str, total: float, weight: float) -> float:
        if weight <= 0.0:
            return 0.0
        objective_value = total / weight
        index = self._objectives.index(name)
        return _sanitize_objective(
            objective_value,
            direction=self._config.objective_directions[index],
            nan_penalty=self._config.nan_penalty,
        )

    def _aggregate_descriptors(
        self,
        profiles: list[BehaviorDescriptorProfile],
    ) -> dict[str, float]:
        if not profiles:
            profile = extract_behavior_descriptors(
                None,
                descriptor_names=self._behavior_descriptor_names,
            )
            return dict(profile.normalized.items())

        keys = set().union(*(profile.raw.keys() for profile in profiles))
        averaged_raw: dict[str, float] = {}
        for key in keys:
            values = [
                float(profile.raw[key]) for profile in profiles if key in profile.raw
            ]
            if not values:
                continue
            averaged_raw[key] = sum(values) / len(values)

        return {
            key: normalize_behavior_descriptor_value(key, value)
            for key, value in averaged_raw.items()
        }


class _ProgramStrategy:
    """Minimal GP strategy wrapper used by the backtest runner."""

    def __init__(self, program: Program) -> None:
        self._program = program

    def fit(self, features: Any, labels: Any | None = None) -> None:  # noqa: ARG002
        return None

    def predict(self, features: Any) -> Any:
        return (
            gp_evaluate(self._program, {"close": features.to_numpy()})
            if hasattr(features, "to_numpy")
            else gp_evaluate(self._program, features)
        )


def _safe_mapping_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, float] = {}
    for key, raw_value in value.items():
        try:
            output[str(key)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return output


def _normalize_slice_score_key(split_key: str, case_key: str) -> tuple[str, ...]:
    """Normalize case labels produced by backtest runners.

    Legacy keys without explicit slice prefix are kept in the existing
    ``{split_key}:{metric}`` layout. Known regime prefixes preserve their prefix so
    lexicase still sees explicit `event`, `instrument`, and `liquidity` cases.
    """
    key = str(case_key)
    if ":" not in key:
        return (f"{split_key}:{key}",)

    for prefix in (
        SLICE_TYPE_TIME_WINDOW,
        SLICE_TYPE_EVENT,
        SLICE_TYPE_INSTRUMENT,
        SLICE_TYPE_LIQUIDITY,
    ):
        if key.startswith(prefix + ":"):
            if split_key in key:
                return (key,)
            return (f"{key}:{split_key}",)

    if split_key in key:
        return (key,)
    return (f"{split_key}:{key}",)


def _coerce_nonnegative(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, value)


def _to_float_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        maybe_values = value.get("values")
        if maybe_values is None:
            return []
        value = maybe_values
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return []
    output: list[float] = []
    for item in value:
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            continue
    return output


def _resolve_metric(
    *,
    objective_name: str,
    metrics: Mapping[str, float],
    derived: Mapping[str, float],
) -> float:
    aliases = _SYMBOLIC_METRIC_ALIASES.get(objective_name, (objective_name,))
    for alias in aliases:
        if alias in metrics:
            return metrics[alias]
        alt = f"{alias}_loss"
        if alt in metrics:
            return float(metrics[alt])
    return float(derived.get(objective_name, 0.0))


def _sanitize_objective(
    value: float,
    *,
    direction: ObjectiveDirection,
    nan_penalty: float,
) -> float:
    if not isinstance(value, (int, float)):
        return -nan_penalty if direction == "maximize" else nan_penalty
    value = float(value)
    if value == float("inf") or value == float("-inf"):
        return -nan_penalty if direction == "maximize" else nan_penalty
    return value
